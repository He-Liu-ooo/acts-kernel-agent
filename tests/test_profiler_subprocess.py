"""Tests for ``_run_ncu`` — the NCU subprocess driver.

These tests verify only the subprocess plumbing: argv construction,
environment setup, and failure handling. CSV parsing is already covered
by ``tests/test_profiler_csv.py`` — the driver must pass stdout through
to the parser without interpreting it.

Tier 1: GPU-free. A ``fake_ncu`` shell script on ``$PATH`` stands in for
the real ``ncu`` binary. Runs in ``~/.venvs/acts_test_venv``.
"""

from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path

import pytest

from _profiler_helpers import force_ncu_discovery, two_phase_fake_ncu_body
from src.eval.profiler import _run_ncu
from src.kernels.kernel import Kernel, KernelSpec, KernelType


# The 18 stall reasons the driver must enumerate on the ncu command line.
# Wildcards don't expand in --metrics — see the memory file NCU discovery #2.
_STALL_REASONS = [
    "barrier",
    "branch_resolving",
    "dispatch_stall",
    "drain",
    "imc_miss",
    "lg_throttle",
    "long_scoreboard",
    "math_pipe_throttle",
    "membar",
    "mio_throttle",
    "misc",
    "no_instruction",
    "not_selected",
    "selected",
    "short_scoreboard",
    "sleeping",
    "tex_throttle",
    "wait",
]


# ── fixtures ───────────────────────────────────────────────────────────────


def _write_fake_ncu(tmp_path: Path, body: str) -> Path:
    """Drop a shell script named ``ncu`` into ``tmp_path`` and make it
    executable. The caller prepends ``tmp_path`` to ``$PATH`` so
    ``shutil.which('ncu')`` / ``subprocess.run(['ncu', ...])`` pick it up."""
    script = tmp_path / "ncu"
    script.write_text("#!/usr/bin/env bash\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture
def fake_ncu_path(tmp_path, monkeypatch):
    """Returns a ``(install_fn, argv_log)`` pair.

    ``install_fn(body)`` writes a shell script with the given body as the
    fake ``ncu`` and prepends ``tmp_path`` to ``$PATH``. ``argv_log`` is
    the file the script echoes its argv to — tests assert on its contents.
    """
    from src.eval import profiler as profiler_mod

    monkeypatch.setattr(profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False)
    # The permission-cache flag is module-global and persists across
    # tests; reset it per-test so a permission-failure scenario in one
    # test doesn't poison the next test's clean run. Also clear the
    # opt-out env var in case a previous test set it.
    profiler_mod._reset_ncu_permission_cache()
    monkeypatch.delenv(profiler_mod._NCU_DISABLE_ENV, raising=False)
    argv_log = tmp_path / "argv.log"

    def install(body: str) -> None:
        _write_fake_ncu(tmp_path, body)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")

    return install, argv_log


@pytest.fixture
def sample_kernel() -> Kernel:
    return Kernel(
        spec=KernelSpec(
            name="my_elementwise",
            kernel_type=KernelType.ELEMENTWISE,
            entrypoint="elementwise_add_kernel",
        ),
        source_code="# placeholder — driver doesn't exec source in Tier 1\n",
    )


@pytest.fixture
def sample_workload() -> dict:
    return {"uuid": "workload-0", "axes": {"N": 1024}, "inputs": {}}


def _identity_input_generator(seed: int = 0) -> tuple:
    """Stand-in for the real input_generator. The driver serializes a spec
    for the subprocess but never calls the generator itself in Tier 1 —
    the fake ncu never execs the real driver."""
    return ()


# ── happy path ────────────────────────────────────────────────────────────


def test_happy_path_returns_stdout_and_zero_exit(fake_ncu_path, sample_kernel, sample_workload):
    install, _ = fake_ncu_path
    canned_csv = (
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","elementwise_add_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","%","50"\n'
    )
    # Heredoc-free shell: escape quotes, one line per row.
    body = 'cat <<"EOF"\n' + canned_csv + "EOF\n"
    install(body)

    stdout, rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    assert rc == 0
    assert degraded is False
    assert reason is None
    assert "elementwise_add_kernel" in stdout
    assert "sm__warps_active" in stdout


# ── failure paths ──────────────────────────────────────────────────────────


def test_nonzero_exit_marks_degraded(fake_ncu_path, sample_kernel, sample_workload):
    """Generic non-zero exit (no permission signature on stderr) keeps
    the legacy ``ncu_nonzero_exit:<rc>`` prefix. When stderr is
    non-empty, we additionally append a sanitized fingerprint (here:
    the word ``boom``) so the run log distinguishes "permission" /
    "OOM" / "section typo" without re-running the kernel — see the
    ``test_transient_nonzero_exit_does_not_set_permission_cache`` test
    below for the longer-form fingerprint case."""
    install, _ = fake_ncu_path
    install('echo "boom" 1>&2\nexit 3\n')

    stdout, rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    assert rc == 3
    assert degraded is True
    # New shape: legacy prefix preserved, fingerprint appended after a
    # second colon. Bare ``ncu_nonzero_exit:3`` is still emitted when
    # stderr is empty (covered by
    # ``test_nonzero_exit_with_empty_stderr_falls_back_to_bare_legacy_shape``).
    assert reason.startswith("ncu_nonzero_exit:3")
    assert "boom" in reason


def test_timeout_marks_degraded(fake_ncu_path, sample_kernel, sample_workload):
    install, _ = fake_ncu_path
    # Sleep longer than the test's timeout so subprocess.run raises
    # TimeoutExpired.
    install("sleep 5\n")

    stdout, rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=0.5,
        mode="curated",
    )
    assert degraded is True
    assert reason == "ncu_timeout"
    # Return code is irrelevant on timeout; stdout may be empty.


def test_garbage_stdout_passes_through(fake_ncu_path, sample_kernel, sample_workload):
    """Driver is plumbing — it does NOT interpret stdout. Garbage is the
    parser's problem; driver just returns (stdout, 0, False, None)."""
    install, _ = fake_ncu_path
    install('echo "this is not a csv"\n')

    stdout, rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    assert rc == 0
    assert degraded is False
    assert reason is None
    assert "this is not a csv" in stdout


def test_binary_missing_marks_degraded(tmp_path, monkeypatch, sample_kernel, sample_workload):
    """No ``ncu`` on ``$PATH`` → driver returns degraded without raising
    FileNotFoundError."""
    # Force discovery to miss both PATH and the cuda-12.8 fallback so the
    # driver hits the "neither PATH nor host install" degradation path.
    force_ncu_discovery(monkeypatch, fallback_present=False)

    # Point PATH at an empty dir so `ncu` cannot resolve.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    stdout, rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    assert degraded is True
    assert reason == "ncu_binary_not_found"


# ── ncu binary discovery ───────────────────────────────────────────────────


def test_discover_ncu_binary_falls_back_to_cuda_12_8_when_path_misses(
    tmp_path, monkeypatch
):
    """When ``shutil.which('ncu')`` returns None (PATH doesn't include
    cuda's bin), discovery falls back to the hardcoded
    ``/usr/local/cuda-12.8/bin/ncu`` if that file exists. Regression
    guard for the ``ncu_binary_not_found`` live-run failure after a
    clean rebuild of ``~/.venvs/acts_run_venv``."""
    from src.eval import profiler as profiler_mod

    force_ncu_discovery(monkeypatch, fallback_present=True)

    assert profiler_mod._discover_ncu_binary() == profiler_mod._NCU_FALLBACK_PATH


def test_discover_ncu_binary_returns_none_when_neither_path_nor_fallback_exists(
    tmp_path, monkeypatch
):
    """Both ``shutil.which`` miss AND fallback path missing → discovery
    returns None so callers degrade with ``ncu_binary_not_found``."""
    from src.eval import profiler as profiler_mod

    force_ncu_discovery(monkeypatch, fallback_present=False)

    assert profiler_mod._discover_ncu_binary() is None


# ── argv wiring ────────────────────────────────────────────────────────────


def _install_argv_echo(install, argv_log: Path) -> None:
    """Fake ncu that dumps its argv, one arg per line, then exits 0 with
    a minimal valid-looking header so callers that need to keep moving
    can (happy-path tests should use the dedicated fixture CSV instead)."""
    body = textwrap.dedent(
        f"""\
        for a in "$@"; do
          printf '%s\\n' "$a" >> {argv_log}
        done
        echo "argv-captured"
        """
    )
    install(body)


def test_argv_includes_csv_and_print_metric_name(fake_ncu_path, sample_kernel, sample_workload):
    install, argv_log = fake_ncu_path
    _install_argv_echo(install, argv_log)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    args = argv_log.read_text().splitlines()
    assert "--csv" in args
    assert "--print-metric-name=name" in args
    assert "--launch-count" in args
    assert args[args.index("--launch-count") + 1] == "1"


def test_argv_includes_all_curated_sections(fake_ncu_path, sample_kernel, sample_workload):
    install, argv_log = fake_ncu_path
    _install_argv_echo(install, argv_log)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    args = argv_log.read_text().splitlines()

    # The four curated sections must each appear as a value after a
    # --section flag.
    def section_values(argv: list[str]) -> list[str]:
        return [argv[i + 1] for i, a in enumerate(argv) if a == "--section" and i + 1 < len(argv)]

    sections = section_values(args)
    for expected in ("Occupancy", "WarpStateStats", "MemoryWorkloadAnalysis", "ComputeWorkloadAnalysis"):
        assert expected in sections, f"missing --section {expected} in {sections}"


def test_argv_enumerates_all_18_stall_metrics(fake_ncu_path, sample_kernel, sample_workload):
    install, argv_log = fake_ncu_path
    _install_argv_echo(install, argv_log)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    args = argv_log.read_text().splitlines()

    # Stalls are not in any --section. They must be listed explicitly via
    # --metrics (possibly comma-joined — check both forms).
    metric_values: list[str] = []
    for i, a in enumerate(args):
        if a == "--metrics" and i + 1 < len(args):
            metric_values.extend(args[i + 1].split(","))

    for reason in _STALL_REASONS:
        name = f"smsp__average_warp_latency_issue_stalled_{reason}.pct"
        assert name in metric_values, f"stall metric {name} not in --metrics"


def test_argv_explicitly_requests_tensor_core_metric(
    fake_ncu_path, sample_kernel, sample_workload
):
    install, argv_log = fake_ncu_path
    _install_argv_echo(install, argv_log)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    args = argv_log.read_text().splitlines()

    metric_values: list[str] = []
    for i, a in enumerate(args):
        if a == "--metrics" and i + 1 < len(args):
            metric_values.extend(args[i + 1].split(","))

    assert (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        in metric_values
    )


def test_argv_explicitly_requests_grouped_debug_metrics(
    fake_ncu_path, sample_kernel, sample_workload
):
    install, argv_log = fake_ncu_path
    _install_argv_echo(install, argv_log)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    args = argv_log.read_text().splitlines()

    metric_values: list[str] = []
    for i, a in enumerate(args):
        if a == "--metrics" and i + 1 < len(args):
            metric_values.extend(args[i + 1].split(","))

    for name in (
        "l1tex__t_sector_hit_rate.pct",
        "sm__instruction_throughput.avg.pct_of_peak_sustained_active",
        "launch__occupancy_limit_registers",
    ):
        assert name in metric_values


def test_argv_includes_kernel_regex_for_entrypoint(fake_ncu_path, sample_kernel, sample_workload):
    install, argv_log = fake_ncu_path
    _install_argv_echo(install, argv_log)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    args = argv_log.read_text().splitlines()

    # spec §4 uses --kernel-name regex:<entrypoint>.
    found = False
    for i, a in enumerate(args):
        if a == "--kernel-name" and i + 1 < len(args):
            if sample_kernel.spec.entrypoint in args[i + 1]:
                found = True
                break
    assert found, f"--kernel-name <regex with entrypoint> not in argv: {args}"


def test_argv_full_mode_uses_set_full(fake_ncu_path, sample_kernel, sample_workload):
    """``mode='full'`` swaps --section for --set full per spec §4."""
    install, argv_log = fake_ncu_path
    _install_argv_echo(install, argv_log)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="full",
    )
    args = argv_log.read_text().splitlines()

    # --set full must appear.
    set_idxs = [i for i, a in enumerate(args) if a == "--set"]
    assert any(args[i + 1] == "full" for i in set_idxs if i + 1 < len(args)), (
        f"--set full not in argv: {args}"
    )
    # --section flags must NOT appear in full mode.
    assert "--section" not in args


# ── TMPDIR workaround ──────────────────────────────────────────────────────


def test_tmpdir_env_set_for_subprocess(fake_ncu_path, tmp_path, sample_kernel, sample_workload):
    """Memory discovery #7: set TMPDIR to a user-owned dir to dodge the
    sticky ``/tmp/nsight-compute-lock`` owned by another user. The fake
    ncu writes $TMPDIR to a known file so we can assert."""
    install, _ = fake_ncu_path
    out = tmp_path / "tmpdir.out"
    install(f'printf "$TMPDIR" > {out}\n')

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    tmpdir_val = out.read_text()
    assert tmpdir_val, "driver did not set TMPDIR for the subprocess"
    # Must be a user-scoped directory (contains the username or 'ncu'
    # suffix), not the shared /tmp.
    assert tmpdir_val != "/tmp"
    assert "ncu" in tmpdir_val.lower() or os.environ.get("USER", "") in tmpdir_val


# ── problem_definition_path threading ──────────────────────────────────────


def _install_json_capture(install, capture_path: Path) -> None:
    """Fake ncu that copies any ``.json`` argv into ``capture_path`` so the
    test can json-decode the spec written by ``_run_ncu``. Wildcards don't
    expand across kwargs — the real driver's spec is the only ``.json``
    argv, so the last write wins and the capture is deterministic."""
    body = textwrap.dedent(
        f"""\
        for a in "$@"; do
          case "$a" in
            *.json) cp "$a" {capture_path} ;;
          esac
        done
        echo "ok"
        """
    )
    install(body)


def test_problem_definition_path_written_to_spec_json(
    fake_ncu_path, tmp_path, sample_kernel, sample_workload
):
    """When ``problem_definition_path`` is passed, the spec JSON the driver
    reads must carry ``problem_dir=<str(parent)>`` — the *directory*
    ``load_problem`` expects, not the definition.json file itself."""
    install, _ = fake_ncu_path
    capture = tmp_path / "spec_capture.json"
    _install_json_capture(install, capture)

    problem_dir = tmp_path / "fake_problem"
    problem_dir.mkdir()
    problem_path = problem_dir / "definition.json"
    problem_path.write_text('{"name": "fake"}')  # driver never reads it in Tier 1

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
        problem_definition_path=problem_path,
    )

    assert capture.exists(), "fake ncu did not capture any .json argv"
    spec = json.loads(capture.read_text())
    assert spec["problem_dir"] == str(problem_dir)
    assert "problem_json" not in spec


def test_problem_definition_path_absent_when_not_passed(
    fake_ncu_path, tmp_path, sample_kernel, sample_workload
):
    """Default ``problem_definition_path=None`` must leave ``problem_dir``
    out of the spec entirely — the driver's priority order falls through
    to ``make_inputs`` / ``args``."""
    install, _ = fake_ncu_path
    capture = tmp_path / "spec_capture.json"
    _install_json_capture(install, capture)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert capture.exists(), "fake ncu did not capture any .json argv"
    spec = json.loads(capture.read_text())
    assert "problem_dir" not in spec
    assert "problem_json" not in spec


# ── driver _build_inputs: regression pin for the problem-dir path ──────────

def _write_sol_problem_dir(root: Path) -> Path:
    """Write a minimal valid SOL problem (definition.json + workload.jsonl)
    under ``root`` and return the directory path. Used by the Tier 1
    driver test to confirm ``_build_inputs`` can load a real problem
    directory — the regression was that the driver was handed the file
    path instead and silently crashed on ``<definition.json>/definition.json``."""
    problem_dir = root / "sol_elementwise"
    problem_dir.mkdir()
    (problem_dir / "definition.json").write_text(json.dumps({
        "name": "elementwise_id",
        "op_type": "elementwise",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x): return x\n",
    }))
    (problem_dir / "workload.jsonl").write_text(
        json.dumps({
            "uuid": "wl0",
            "axes": {"N": 128},
            "inputs": {"x": {"type": "random"}},
        }) + "\n"
    )
    return problem_dir


def test_driver_build_inputs_loads_problem_from_directory(tmp_path, monkeypatch):
    """``_build_inputs`` must accept the problem *directory* and successfully
    call ``load_problem`` on it. Regression test for the bug where the
    profiler serialized ``definition.json`` as the path and the driver
    then tried to open ``<definition.json>/definition.json``.

    ``build_input_generator`` requires torch + sol_execbench (not in the
    Tier 1 venv) — patched to a stub so this test exercises the
    load_problem + Workload construction path only."""
    from src.eval import _profiler_driver

    problem_dir = _write_sol_problem_dir(tmp_path)

    captured: dict = {}

    def _fake_build_input_generator(problem, workload, **kwargs):
        captured["problem_name"] = problem.name
        captured["problem_op_type"] = problem.op_type
        captured["workload_uuid"] = workload.uuid
        captured["workload_axes"] = dict(workload.axes)
        return lambda seed: ("ok", seed)

    # The lazy import inside ``_build_inputs`` resolves
    # ``from src.eval.inputs import build_input_generator`` — patch at the
    # source module so the lazy import picks up the stub.
    monkeypatch.setattr(
        "src.eval.inputs.build_input_generator",
        _fake_build_input_generator,
    )

    definition, workload, inputs = _profiler_driver._build_inputs(
        problem_dir,
        {"uuid": "wl0", "axes": {"N": 128}},
        seed=7,
    )

    assert inputs == ("ok", 7)
    assert definition.name == "elementwise_id"
    assert workload.uuid == "wl0"
    assert captured["problem_name"] == "elementwise_id"
    assert captured["problem_op_type"] == "elementwise"
    assert captured["workload_uuid"] == "wl0"
    assert captured["workload_axes"] == {"N": 128}


def test_driver_build_inputs_does_not_attribute_error_on_sol_load(monkeypatch):
    """Regression: ``_build_inputs`` must call the SOL-package
    re-exported ``load`` function directly — not as ``sol_load.load(...)``
    against an alias of the function itself. The latter raises
    ``AttributeError: 'function' object has no attribute 'load'`` and
    kills the profiler subprocess on every SOL-backed input rebuild.

    This test exercises the exact import path the subprocess driver uses
    against the committed ``tests/fixtures/sol_simple/`` SOL problem and
    asserts that no ``AttributeError`` escapes the call. It also asserts
    the returned tuple is non-empty so a future "import succeeds but
    nothing flows through" silent regression is caught.

    ``build_input_generator`` defaults to ``device='cuda'`` and would
    require torch + a GPU to materialize tensors, so it's stubbed — the
    bug being pinned is in the ``sol_load(...)`` call that runs *before*
    that import is even reached.
    """
    from src.eval import _profiler_driver

    fixture_dir = Path(__file__).parent / "fixtures" / "sol_simple"
    assert (fixture_dir / "definition.json").is_file(), (
        f"missing committed fixture: {fixture_dir}"
    )
    assert (fixture_dir / "workload.jsonl").is_file(), (
        f"missing committed fixture: {fixture_dir}"
    )

    def _fake_build_input_generator(definition, workload, **kwargs):
        # Returns a generator that yields a 2-tuple so the assertion below
        # has something concrete to bind against.
        return lambda seed: ("a_tensor", "b_tensor")

    monkeypatch.setattr(
        "src.eval.inputs.build_input_generator",
        _fake_build_input_generator,
    )

    # The first matching workload in the committed fixture is uuid=w1, N=1024.
    workload_dict = {
        "uuid": "w1",
        "axes": {"N": 1024},
        "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
    }

    # The bug class: ``sol_load.load(...)`` raised AttributeError. We
    # explicitly re-raise any AttributeError as a test failure with a
    # pointer comment so a future regressor knows where to look.
    try:
        definition, workload, inputs = _profiler_driver._build_inputs(
            fixture_dir,
            workload_dict,
            seed=0,
        )
    except AttributeError as exc:  # pragma: no cover — regression marker
        pytest.fail(
            f"_build_inputs raised AttributeError: {exc}"
        )

    assert isinstance(inputs, tuple)
    assert len(inputs) > 0, "stubbed generator should return a non-empty tuple"
    assert definition is not None
    assert workload is not None


def test_driver_build_inputs_rejects_file_path_with_clear_error(tmp_path):
    """If a caller regresses and passes the ``definition.json`` *file* instead
    of the directory, the driver's ``load_problem`` call must raise —
    no more silent ``<definition.json>/definition.json`` masquerade.
    Uses the real ``load_problem`` (pure Python, no GPU deps)."""
    from src.eval import _profiler_driver

    problem_dir = _write_sol_problem_dir(tmp_path)
    bad_path = problem_dir / "definition.json"  # file, not dir

    with pytest.raises((FileNotFoundError, NotADirectoryError, OSError)):
        _profiler_driver._build_inputs(
            bad_path,
            {"uuid": "wl0", "axes": {"N": 128}},
            seed=0,
        )


# ── spec contract: blob_roots + dps fields (G2 + G4 driver fix) ───────────


def test_blob_roots_serialized_into_spec_json(
    fake_ncu_path, tmp_path, sample_kernel, sample_workload
):
    """When ``blob_roots`` is passed to ``_run_ncu``, the spec JSON the driver
    reads must carry it as a list of strings under ``blob_roots``. The driver
    rehydrates these to ``list[Path]`` and threads them into
    ``build_input_generator`` so ``SafetensorsInput`` workloads resolve to
    real on-disk weights inside the subprocess. ``_profiler_driver`` is one
    of the parallel call sites that needs the same blob_roots thread-through
    every other input-generator construction site already has.
    """
    install, _ = fake_ncu_path
    capture = tmp_path / "spec_capture.json"
    _install_json_capture(install, capture)

    blob_root_a = tmp_path / "blobs_a"
    blob_root_b = tmp_path / "blobs_b"
    blob_root_a.mkdir()
    blob_root_b.mkdir()

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
        blob_roots=[blob_root_a, blob_root_b],
    )

    assert capture.exists(), "fake ncu did not capture any .json argv"
    spec = json.loads(capture.read_text())
    assert spec["blob_roots"] == [str(blob_root_a), str(blob_root_b)]


def test_blob_roots_absent_when_not_passed(
    fake_ncu_path, tmp_path, sample_kernel, sample_workload
):
    """``blob_roots=None`` (the default) must omit the field from the spec
    JSON entirely so the driver's back-compat default kicks in. Older
    cached specs / hand-crafted subprocess specs from earlier sub-commits
    must continue to parse cleanly."""
    install, _ = fake_ncu_path
    capture = tmp_path / "spec_capture.json"
    _install_json_capture(install, capture)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert capture.exists(), "fake ncu did not capture any .json argv"
    spec = json.loads(capture.read_text())
    assert "blob_roots" not in spec


def test_dps_serialized_into_spec_json_when_kernel_is_dps(
    fake_ncu_path, tmp_path, sample_workload
):
    """When ``kernel.dps=True``, the spec JSON must carry ``dps: true`` so
    the driver pre-allocates output buffers via ``allocate_outputs`` and
    calls ``kernel_fn(*inputs, *outputs)`` instead of ``kernel_fn(*inputs)``.
    Regression pin for the G4 follow-up: NCU profiling silently TypeErrored
    on DPS kernels because the driver bypassed the DPS branch present in
    benchmark.py / correctness.py.
    """
    install, _ = fake_ncu_path
    capture = tmp_path / "spec_capture.json"
    _install_json_capture(install, capture)

    dps_kernel = Kernel(
        spec=KernelSpec(
            name="dps_elementwise",
            kernel_type=KernelType.ELEMENTWISE,
            entrypoint="elementwise_add_kernel",
        ),
        source_code="# placeholder\n",
        dps=True,
    )

    _run_ncu(
        dps_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert capture.exists(), "fake ncu did not capture any .json argv"
    spec = json.loads(capture.read_text())
    assert spec["dps"] is True


def test_dps_default_false_when_kernel_is_not_dps(
    fake_ncu_path, tmp_path, sample_kernel, sample_workload
):
    """Non-DPS kernel: spec must either carry ``dps: false`` or omit the
    field entirely. Either is fine for back-compat; the driver defaults to
    False on missing key. Both shapes preserve the kernel_fn(*inputs) call."""
    install, _ = fake_ncu_path
    capture = tmp_path / "spec_capture.json"
    _install_json_capture(install, capture)

    # ``sample_kernel`` defaults dps=False.
    assert sample_kernel.dps is False

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    spec = json.loads(capture.read_text())
    assert spec.get("dps", False) is False


# ── driver _build_inputs threads blob_roots (G2 follow-up) ────────────────


def test_driver_build_inputs_threads_blob_roots(tmp_path, monkeypatch):
    """``_build_inputs`` must rehydrate ``spec['blob_roots']`` (list[str]) to
    ``list[Path]`` and forward it as ``blob_roots=`` into
    ``build_input_generator``. Regression pin for the G2 follow-up: the
    driver previously called ``build_input_generator(definition, workload)``
    with no blob_roots, causing FileNotFoundError inside the subprocess
    whenever the workload contained a ``SafetensorsInput``.
    """
    from src.eval import _profiler_driver

    problem_dir = _write_sol_problem_dir(tmp_path)
    blob_root = tmp_path / "shared_blobs"
    blob_root.mkdir()

    captured: dict = {}

    def _fake_build_input_generator(definition, workload, **kwargs):
        captured["blob_roots"] = kwargs.get("blob_roots")
        return lambda seed: ("ok", seed)

    monkeypatch.setattr(
        "src.eval.inputs.build_input_generator",
        _fake_build_input_generator,
    )

    _profiler_driver._build_inputs(
        problem_dir,
        {"uuid": "wl0", "axes": {"N": 128}},
        seed=0,
        blob_roots=[blob_root],
    )

    assert captured["blob_roots"] == [blob_root]
    assert all(isinstance(p, Path) for p in captured["blob_roots"])


def test_driver_build_inputs_blob_roots_none_default(tmp_path, monkeypatch):
    """Back-compat: when ``blob_roots`` is absent from the spec, the driver
    forwards ``blob_roots=None`` so older serialized specs and self-contained
    kernels keep working unchanged."""
    from src.eval import _profiler_driver

    problem_dir = _write_sol_problem_dir(tmp_path)

    captured: dict = {}

    def _fake_build_input_generator(definition, workload, **kwargs):
        captured["blob_roots"] = kwargs.get("blob_roots", "<MISSING>")
        return lambda seed: ()

    monkeypatch.setattr(
        "src.eval.inputs.build_input_generator",
        _fake_build_input_generator,
    )

    _profiler_driver._build_inputs(
        problem_dir,
        {"uuid": "wl0", "axes": {"N": 128}},
        seed=0,
    )

    assert captured["blob_roots"] is None


# ── driver _call_kernel: DPS branch (G4 follow-up) ────────────────────────


def test_call_kernel_non_dps_passes_inputs_only():
    """Non-DPS path: ``_call_kernel`` invokes ``kernel_fn(*inputs)`` and
    returns the function's return value. No allocate_outputs, no extra
    positional args."""
    from src.eval import _profiler_driver

    received: dict = {}

    def fake_kernel(a, b):
        received["args"] = (a, b)
        return ("returned", a + b)

    result = _profiler_driver._call_kernel(
        fake_kernel,
        inputs=(1, 2),
        definition=None,
        workload=None,
        dps=False,
        device="cpu",
    )

    assert received["args"] == (1, 2)
    assert result == ("returned", 3)


def test_call_kernel_dps_pre_allocates_outputs_and_appends(monkeypatch):
    """DPS path: ``_call_kernel`` must invoke
    ``kernel_fn(*inputs, *outputs)`` where ``outputs`` came from
    ``src.eval.inputs.allocate_dps_outputs``. Returns the outputs tuple
    (the kernel populated them in place)."""
    from src.eval import _profiler_driver

    fake_outputs = ("buf0", "buf1")

    # Stub allocate_dps_outputs at the import site the driver uses (lazy
    # import inside ``_call_kernel`` resolves against the live module object).
    import src.eval.inputs as eval_inputs

    monkeypatch.setattr(
        eval_inputs,
        "allocate_dps_outputs",
        lambda definition, workload, *, device: list(fake_outputs),
    )

    class FakeDefinition:
        pass

    class FakeWorkload:
        axes = {"N": 4}

    received: dict = {}

    def fake_kernel(a, b, out0, out1):
        received["args"] = (a, b, out0, out1)
        return "ignored"

    result = _profiler_driver._call_kernel(
        fake_kernel,
        inputs=(10, 20),
        definition=FakeDefinition(),
        workload=FakeWorkload(),
        dps=True,
        device="cpu",
    )

    # kernel_fn was called with inputs followed by pre-allocated outputs.
    assert received["args"] == (10, 20, "buf0", "buf1")
    # _call_kernel returns the outputs tuple, not the kernel's return value
    # (DPS kernels populate buffers in place; the return is conventionally None).
    assert tuple(result) == fake_outputs


def test_call_kernel_dps_requires_definition_and_workload():
    """Calling ``_call_kernel(..., dps=True, definition=None, workload=None)``
    is a contract bug — surface it loudly rather than letting a None reach
    ``allocate_dps_outputs`` and TypeError obscurely."""
    from src.eval import _profiler_driver

    def fake_kernel(*args):
        return None

    with pytest.raises((ValueError, AttributeError, TypeError)):
        _profiler_driver._call_kernel(
            fake_kernel,
            inputs=(1, 2),
            definition=None,
            workload=None,
            dps=True,
            device="cpu",
        )


# ── permission-failure caching (host kernel NVreg_RestrictProfilingToAdminUsers=1) ─


# Counter-shim helpers: count how many times the fake ncu shell actually
# fires. Each invocation appends a line to ``count_path`` so the test can
# assert on subprocess call count after a sequence of ``_run_ncu`` calls.
def _install_counting_perm_failure(install, count_path: Path, signature: str) -> None:
    """Fake ncu that records each invocation in ``count_path`` and exits
    non-zero with the given permission-failure ``signature`` on stderr.
    Tests use this to verify the permission cache flips after the first
    call so subsequent calls do NOT re-invoke the subprocess."""
    body = textwrap.dedent(
        f"""\
        echo "called" >> {count_path}
        echo "{signature}" 1>&2
        exit 1
        """
    )
    install(body)


def test_permission_failure_nvgpuctrperm_marks_skipped_and_caches(
    fake_ncu_path, tmp_path, sample_kernel, sample_workload
):
    """First NCU call fails with the canonical ERR_NVGPUCTRPERM token;
    the reason must report ``ncu_skipped:permanently_unavailable:nvgpuctrperm``
    (NOT the generic ``ncu_nonzero_exit:1``) and the module-level cache
    flag must flip so the next call short-circuits."""
    from src.eval import profiler as profiler_mod

    install, _ = fake_ncu_path
    count_path = tmp_path / "ncu_calls.log"
    _install_counting_perm_failure(
        install, count_path, "==ERROR== ERR_NVGPUCTRPERM: counter access blocked"
    )

    stdout, rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert degraded is True
    assert reason == "ncu_skipped:permanently_unavailable:nvgpuctrperm"
    # The cache must hold the same slug so subsequent calls see it.
    assert profiler_mod._NCU_PERMANENTLY_UNAVAILABLE == reason


def test_permission_failure_counter_perm_phrase_marks_skipped(
    fake_ncu_path, tmp_path, sample_kernel, sample_workload
):
    """Older / alternate NCU phrasing (no ERR_NVGPUCTRPERM token but the
    free-text "does not have permission to access the GPU performance
    counter" phrase) must also trigger the permanent-unavailable slug."""
    install, _ = fake_ncu_path
    count_path = tmp_path / "ncu_calls.log"
    _install_counting_perm_failure(
        install,
        count_path,
        "ERROR: The user does not have permission to access the GPU performance counter on the target device.",
    )

    _stdout, _rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert degraded is True
    # First-match-wins: ``err_nvgpuctrperm`` is absent, so the
    # ``counter_perm`` slug fires next.
    assert reason == "ncu_skipped:permanently_unavailable:counter_perm"


def test_permission_failure_short_circuits_subsequent_calls(
    fake_ncu_path, tmp_path, sample_kernel, sample_workload
):
    """After the first permission failure flips the cache, subsequent
    ``_run_ncu`` calls must NOT fork a subprocess. Verified by counting
    the lines the fake ncu writes — only the first call should run."""
    install, _ = fake_ncu_path
    count_path = tmp_path / "ncu_calls.log"
    _install_counting_perm_failure(
        install, count_path, "==ERROR== ERR_NVGPUCTRPERM"
    )

    # Three back-to-back calls — only the first should actually exec ncu.
    for _ in range(3):
        _stdout, _rc, degraded, reason = _run_ncu(
            sample_kernel,
            sample_workload,
            _identity_input_generator,
            timeout_s=10.0,
            mode="curated",
        )
        assert degraded is True
        assert reason == "ncu_skipped:permanently_unavailable:nvgpuctrperm"

    # The fake ncu writes a line per real invocation. Only the first
    # call should have executed; the subsequent two short-circuited.
    invocation_count = (
        len(count_path.read_text().splitlines()) if count_path.exists() else 0
    )
    assert invocation_count == 1, (
        f"expected exactly 1 subprocess invocation after cache flip; "
        f"got {invocation_count}"
    )


def test_permission_cache_reset_hook_re_enables_subprocess(
    fake_ncu_path, tmp_path, sample_kernel, sample_workload
):
    """``_reset_ncu_permission_cache()`` must clear the flag so a fresh
    call hits the subprocess again. Lets test isolation work without
    re-importing the module per test."""
    from src.eval import profiler as profiler_mod

    install, _ = fake_ncu_path
    count_path = tmp_path / "ncu_calls.log"
    _install_counting_perm_failure(
        install, count_path, "==ERROR== ERR_NVGPUCTRPERM"
    )

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )
    assert profiler_mod._NCU_PERMANENTLY_UNAVAILABLE is not None

    profiler_mod._reset_ncu_permission_cache()
    assert profiler_mod._NCU_PERMANENTLY_UNAVAILABLE is None

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    invocation_count = len(count_path.read_text().splitlines())
    assert invocation_count == 2, (
        f"expected 2 subprocess invocations across the reset; "
        f"got {invocation_count}"
    )


# ── transient (non-permission) non-zero exit: keeps legacy behavior + adds fingerprint ─


def test_transient_nonzero_exit_does_not_set_permission_cache(
    fake_ncu_path, sample_kernel, sample_workload
):
    """A non-zero exit whose stderr lacks any permission signature must
    keep the legacy ``ncu_nonzero_exit:<rc>`` reason shape (optionally
    with a fingerprint suffix) and must NOT flip the permanent-skip
    cache — transient failures should be retried next iteration."""
    from src.eval import profiler as profiler_mod

    install, _ = fake_ncu_path
    install('echo "==ERROR== ProfilerReply error: Not enough memory" 1>&2\nexit 7\n')

    _stdout, rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert rc == 7
    assert degraded is True
    # Non-permission failure: cache must remain unset so we retry on the
    # next iteration.
    assert profiler_mod._NCU_PERMANENTLY_UNAVAILABLE is None
    # Legacy shape is preserved as the prefix; the fingerprint is
    # appended after a colon.
    assert reason.startswith("ncu_nonzero_exit:7")
    # Fingerprint must surface a recognizable token from the stderr
    # (sanitized for log-safety) so operators can distinguish "OOM" from
    # "section name typo" without grepping the raw run.log.
    assert "ProfilerReply" in reason or "Not_enough_memory" in reason


def test_nonzero_exit_with_empty_stderr_falls_back_to_bare_legacy_shape(
    fake_ncu_path, sample_kernel, sample_workload
):
    """When stderr is empty, the fingerprint helper returns ``""`` and the
    reason collapses to the bare ``ncu_nonzero_exit:<rc>`` shape (the same
    shape the original test_nonzero_exit_marks_degraded asserts on, so we
    don't drift from existing callers)."""
    install, _ = fake_ncu_path
    install("exit 4\n")

    _stdout, rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert rc == 4
    assert degraded is True
    # Exact match — no fingerprint suffix when stderr is empty.
    assert reason == "ncu_nonzero_exit:4"


# ── env-var opt-out ────────────────────────────────────────────────────────


def test_acts_disable_ncu_env_var_skips_subprocess(
    fake_ncu_path, tmp_path, sample_kernel, sample_workload, monkeypatch
):
    """``ACTS_DISABLE_NCU=1`` must short-circuit before the subprocess
    fork. The fake ncu would record an invocation if it ran — assert it
    didn't."""
    from src.eval import profiler as profiler_mod

    install, _ = fake_ncu_path
    count_path = tmp_path / "ncu_calls.log"
    _install_counting_perm_failure(
        install, count_path, "this should never appear in the log"
    )

    monkeypatch.setenv(profiler_mod._NCU_DISABLE_ENV, "1")

    _stdout, rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert degraded is True
    assert reason == "ncu_disabled_via_env"
    assert not count_path.exists() or count_path.read_text() == "", (
        "ACTS_DISABLE_NCU=1 must skip the subprocess fork entirely"
    )


def test_acts_disable_ncu_env_var_truthy_variants(
    fake_ncu_path, sample_kernel, sample_workload, monkeypatch
):
    """The env var accepts the standard truthy strings (1/true/yes/on,
    case-insensitive). Anything else is treated as 'not set'."""
    from src.eval import profiler as profiler_mod

    install, _ = fake_ncu_path
    install("exit 0\n")  # would succeed if reached

    for truthy in ("1", "true", "TRUE", "yes", "Yes", "on"):
        monkeypatch.setenv(profiler_mod._NCU_DISABLE_ENV, truthy)
        _stdout, _rc, degraded, reason = _run_ncu(
            sample_kernel,
            sample_workload,
            _identity_input_generator,
            timeout_s=10.0,
            mode="curated",
        )
        assert degraded is True, f"truthy='{truthy}' should disable NCU"
        assert reason == "ncu_disabled_via_env"


def test_acts_disable_ncu_env_var_falsy_does_not_skip(
    fake_ncu_path, sample_kernel, sample_workload, monkeypatch
):
    """Empty / falsy values leave the auto-detect path active. We use
    "0" — the env var is *opt-in*, not *opt-out-of-default*."""
    from src.eval import profiler as profiler_mod

    install, _ = fake_ncu_path
    install("echo 'ok'\nexit 0\n")

    monkeypatch.setenv(profiler_mod._NCU_DISABLE_ENV, "0")

    stdout, rc, degraded, reason = _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert rc == 0
    assert degraded is False
    assert reason is None
    assert "ok" in stdout


# ── permanent-failure classifier unit tests ────────────────────────────────


def test_classify_ncu_permanent_failure_recognizes_canonical_token():
    from src.eval.profiler import _classify_ncu_permanent_failure

    assert (
        _classify_ncu_permanent_failure(
            stderr="==ERROR== ERR_NVGPUCTRPERM: ...", stdout=""
        )
        == "nvgpuctrperm"
    )


def test_classify_ncu_permanent_failure_case_insensitive():
    from src.eval.profiler import _classify_ncu_permanent_failure

    assert (
        _classify_ncu_permanent_failure(
            stderr="err_nvgpuctrperm seen", stdout=""
        )
        == "nvgpuctrperm"
    )


def test_classify_ncu_permanent_failure_returns_none_for_transient():
    from src.eval.profiler import _classify_ncu_permanent_failure

    # Memory error and section-name typo are transient — operator can
    # change the run config and retry; we should NOT flip the cache.
    assert (
        _classify_ncu_permanent_failure(
            stderr="==ERROR== ProfilerReply error: Not enough memory", stdout=""
        )
        is None
    )
    assert (
        _classify_ncu_permanent_failure(
            stderr="==WARNING== Section 'Bogus' not found", stdout=""
        )
        is None
    )


def test_classify_ncu_permanent_failure_inspects_stdout_too():
    """NCU sometimes writes the actionable error to stdout when --csv is
    set (the CSV header is missing → the line operators see is the
    error). Classifier must check both streams."""
    from src.eval.profiler import _classify_ncu_permanent_failure

    assert (
        _classify_ncu_permanent_failure(
            stderr="", stdout="ERR_NVGPUCTRPERM"
        )
        == "nvgpuctrperm"
    )


# ── ncu-rep capture (search-tree-recording feature) ────────────────────────


def test_profiling_result_has_ncu_rep_path_field():
    """ProfilingResult exposes ncu_rep_path; default None."""
    from src.eval.profiler import AnalyticalMetrics, ProfilingResult
    a = AnalyticalMetrics(
        achieved_tflops=0.0, achieved_bandwidth_gb_s=0.0,
        pct_peak_compute=0.0, pct_peak_bandwidth=0.0,
    )
    pr = ProfilingResult(analytical=a)
    assert pr.ncu_rep_path is None


def test_ncu_argv_includes_output_flag(tmp_path):
    """_build_ncu_argv includes -o <path>.ncu-rep when out_path is set."""
    from src.eval.profiler import _build_ncu_argv
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    kernel = Kernel(spec=spec, source_code="")
    spec_json = tmp_path / "spec.json"
    spec_json.write_text("{}")
    out_path = tmp_path / "report.ncu-rep"
    argv = _build_ncu_argv(kernel, spec_json, mode="full",
                           kernel_name="kernel_fn", out_path=out_path)
    # ncu's flag is -o <basename> (no .ncu-rep suffix; ncu adds it)
    assert "-o" in argv
    out_idx = argv.index("-o")
    assert out_idx + 1 < len(argv)
    # Either basename or full path is acceptable; the key is the flag's there.
    assert str(out_path).rstrip(".ncu-rep") in argv[out_idx + 1] or \
           str(out_path) in argv[out_idx + 1]


def test_ncu_rep_path_set_when_cache_dir_is_none(tmp_path, monkeypatch):
    """The .ncu-rep file is written and exposed via ProfilingResult.ncu_rep_path
    even when no cache_dir is supplied — the orchestrator uses this code path.

    Mirrors the fake-ncu pattern in tests/test_profiler_cache.py: the script
    emits a canned CSV the parser can consume and ALSO touches a sentinel
    file at the ``-o`` argument so the post-run ``ncu_rep_out.exists()``
    check sees a real file. The shell script parses ``-o <basename>`` from
    its argv (NCU strips the ``.ncu-rep`` suffix before passing the path to
    ``-o``; the script appends it back).
    """
    from src.eval import profiler as profiler_mod
    from src.eval.profiler import (
        ProfilingResult,
        profile_kernel,
    )
    from conftest import rtx6000_ada_hardware

    # Reset NCU-discovery + permission caches so this test sees the fake
    # ncu on its monkeypatched PATH (the same dance test_profiler_cache.py
    # does in its autouse reset).
    monkeypatch.setattr(
        profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False
    )
    profiler_mod._reset_ncu_permission_cache()
    monkeypatch.delenv(profiler_mod._NCU_DISABLE_ENV, raising=False)

    # Canned CSV: parser-valid for the curated path. Mirrors the
    # _canned_csv() helper in test_profiler_cache.py — kept inline here to
    # avoid cross-file fixture import.
    header = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'

    def row(metric: str, value: str) -> str:
        return f'"0","elementwise_add_kernel","{metric}","%","{value}"\n'

    rows = [
        row("sm__warps_active.avg.pct_of_peak_sustained_active", "55.0"),
        row("lts__t_sector_hit_rate.pct", "72.5"),
        row(
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
            "0",
        ),
    ]
    stall_values = {
        "barrier": "0",
        "branch_resolving": "5",
        "dispatch_stall": "10",
        "drain": "15",
        "imc_miss": "20",
        "lg_throttle": "25",
        "long_scoreboard": "80",
        "math_pipe_throttle": "35",
        "membar": "40",
        "mio_throttle": "45",
        "misc": "50",
        "no_instruction": "55",
        "not_selected": "60",
        "selected": "65",
        "short_scoreboard": "70",
        "sleeping": "1",
        "tex_throttle": "2",
        "wait": "3",
    }
    for reason, val in stall_values.items():
        rows.append(
            row(f"smsp__average_warp_latency_issue_stalled_{reason}.pct", val)
        )
    banner = (
        "==PROF== Connected to process 1 (/usr/bin/python3.10)\n"
        "ok\n"
        "==PROF== Disconnected from process 1\n"
    )
    csv = banner + header + "".join(rows)

    body = two_phase_fake_ncu_body(csv)
    script = tmp_path / "ncu"
    script.write_text(body)
    script.chmod(
        script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")

    kernel = Kernel(
        spec=KernelSpec(
            name="my_elementwise",
            kernel_type=KernelType.ELEMENTWISE,
            entrypoint="elementwise_add_kernel",
        ),
        source_code=(
            "# kernel source — tier 1 stub; compile_kernel needs the "
            "entrypoint to resolve\n"
            "def elementwise_add_kernel(*args, **kwargs):\n"
            "    return None\n"
        ),
    )
    workload = {"uuid": "workload-0", "axes": {"N": 1024}, "inputs": {}}

    result = profile_kernel(
        kernel,
        workload,
        _identity_input_generator,
        hardware_spec=rtx6000_ada_hardware(),
        flops=1_000_000,
        nbytes=4_000_000,
        latency_s=1e-3,
        mode="curated",
        timeout_s=10.0,
        cache_dir=None,
    )

    assert isinstance(result, ProfilingResult)
    assert result.degraded is False, (
        f"profile_kernel went degraded with cache_dir=None: {result.degraded_reason}"
    )
    assert result.ncu_rep_path is not None, (
        "ncu_rep_path must be populated even when cache_dir is None — "
        "the orchestrator runs without a cache_dir and tree_dump needs the path"
    )
    assert result.ncu_rep_path.exists(), (
        f"ncu_rep_path points at {result.ncu_rep_path} but no file is on disk"
    )


def test_ncu_argv_includes_force_flag_when_out_path_set(tmp_path):
    """`_build_ncu_argv` includes -f (force-overwrite) along with -o so
    NCU doesn't fail on a stale .ncu-rep file in the persistent
    user-scoped tmpdir."""
    from src.eval.profiler import _build_ncu_argv
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    kernel = Kernel(spec=spec, source_code="")
    spec_json = tmp_path / "spec.json"
    spec_json.write_text("{}")
    out_path = tmp_path / "report.ncu-rep"
    argv = _build_ncu_argv(kernel, spec_json, mode="full",
                           kernel_name="kernel_fn", out_path=out_path)
    assert "-f" in argv
    # The order matters minimally (NCU accepts -f anywhere), but -f and
    # -o should both be present.
    assert "-o" in argv


def test_ncu_argv_omits_force_flag_when_out_path_none(tmp_path):
    """When no -o is requested, no -f either — keeps argv minimal."""
    from src.eval.profiler import _build_ncu_argv
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    kernel = Kernel(spec=spec, source_code="")
    spec_json = tmp_path / "spec.json"
    spec_json.write_text("{}")
    argv = _build_ncu_argv(kernel, spec_json, mode="full",
                           kernel_name="kernel_fn", out_path=None)
    assert "-f" not in argv
    assert "-o" not in argv


# ── --import CSV extraction (2026-05-08 NCU 2025.x two-subprocess fix) ─────


def test_ncu_argv_omits_log_file_when_out_path_set(tmp_path):
    """``_build_ncu_argv`` must NOT emit ``--log-file`` — that flag was
    explored (Option A) and rejected after empirical verification: when
    ``-o`` is set, NCU's stdout is banners-only and ``--log-file`` only
    redirects what would have been printed. The fix is the two-subprocess
    path (``_extract_ncu_csv`` post-profile), not a flag tweak."""
    from src.eval.profiler import _build_ncu_argv
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    kernel = Kernel(spec=spec, source_code="")
    spec_json = tmp_path / "spec.json"
    spec_json.write_text("{}")
    out_path = tmp_path / "report.ncu-rep"

    argv = _build_ncu_argv(kernel, spec_json, mode="full",
                           kernel_name="kernel_fn", out_path=out_path)

    assert "--log-file" not in argv
    # ``-o`` and ``-f`` still belong to the capture argv.
    assert "-o" in argv
    assert "-f" in argv


def test_extract_ncu_csv_argv_uses_import_csv_page_details(tmp_path, monkeypatch):
    """``_extract_ncu_csv`` invokes ncu with ``--import <rep> --csv --page
    details`` — the only flag combo that re-emits the parser-expected
    columns from a binary report. Mock ncu echoes its argv to a log file
    so the test can assert on the exact command line."""
    import os
    import stat
    from src.eval import profiler as profiler_mod
    from src.eval.profiler import _extract_ncu_csv

    monkeypatch.setattr(
        profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False
    )

    argv_log = tmp_path / "argv.log"
    body = (
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$@\" > {argv_log}\n"
        "echo 'ID,Kernel Name,Metric Name,Metric Unit,Metric Value'\n"
    )
    script = tmp_path / "ncu"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")

    rep_path = tmp_path / "report.ncu-rep"
    rep_path.write_text("ncu-rep-marker")

    stdout, rc, degraded, reason = _extract_ncu_csv(rep_path)
    assert rc == 0
    assert degraded is False
    assert reason is None
    assert "Kernel Name" in stdout

    args = argv_log.read_text().splitlines()
    assert "--import" in args
    assert str(rep_path) in args
    assert "--csv" in args
    assert "--page" in args
    page_idx = args.index("--page")
    assert args[page_idx + 1] == "details"


def test_extract_ncu_csv_nonzero_returns_ncu_import_failed(tmp_path, monkeypatch):
    """``_extract_ncu_csv`` returns the distinct ``ncu_import_failed:<rc>``
    slug on non-zero exit so the upstream classifier can tell "the binary
    is there but post-processing failed" from "no header in CSV"."""
    import os
    import stat
    from src.eval import profiler as profiler_mod
    from src.eval.profiler import _extract_ncu_csv

    monkeypatch.setattr(
        profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False
    )

    body = (
        "#!/usr/bin/env bash\n"
        "echo 'ncu: error: import failed' >&2\n"
        "exit 7\n"
    )
    script = tmp_path / "ncu"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")

    rep_path = tmp_path / "report.ncu-rep"
    rep_path.write_text("ncu-rep-marker")

    _stdout, rc, degraded, reason = _extract_ncu_csv(rep_path)
    assert rc == 7
    assert degraded is True
    assert reason == "ncu_import_failed:7"


def test_profile_kernel_extracts_csv_via_import_with_ncu_2025(
    tmp_path, monkeypatch
):
    """Regression guard for commit 166d697.

    Real NCU 2025.x suppresses CSV from stdout when ``-o`` is set —
    stdout is just ``==PROF==`` banners and the binary report is on
    disk. Before the two-subprocess fix, ``_parse_ncu_csv`` saw zero
    rows on stdout and degraded with ``csv_parse:no_header``. This test
    mocks NCU honoring the real contract: ``-o`` (capture mode) writes a
    marker rep + emits banners-only stdout; ``--import <rep>`` (extract
    mode) emits CSV. The fix must run the extract subprocess after the
    profile subprocess, hand its stdout to the parser, and produce a
    non-degraded ProfilingResult.
    """
    from src.eval import profiler as profiler_mod
    from src.eval.profiler import ProfilingResult, profile_kernel
    from conftest import rtx6000_ada_hardware

    monkeypatch.setattr(
        profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False
    )
    profiler_mod._reset_ncu_permission_cache()
    monkeypatch.delenv(profiler_mod._NCU_DISABLE_ENV, raising=False)

    header = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'

    def row(metric: str, value: str) -> str:
        return f'"0","elementwise_add_kernel","{metric}","%","{value}"\n'

    rows = [
        row("sm__warps_active.avg.pct_of_peak_sustained_active", "55.0"),
        row("lts__t_sector_hit_rate.pct", "72.5"),
        row("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "0"),
    ]
    stall_values = {
        "barrier": "0", "branch_resolving": "5", "dispatch_stall": "10",
        "drain": "15", "imc_miss": "20", "lg_throttle": "25",
        "long_scoreboard": "80", "math_pipe_throttle": "35", "membar": "40",
        "mio_throttle": "45", "misc": "50", "no_instruction": "55",
        "not_selected": "60", "selected": "65", "short_scoreboard": "70",
        "sleeping": "1", "tex_throttle": "2", "wait": "3",
    }
    for reason, val in stall_values.items():
        rows.append(
            row(f"smsp__average_warp_latency_issue_stalled_{reason}.pct", val)
        )
    csv_text = header + "".join(rows)

    body = two_phase_fake_ncu_body(csv_text)
    script = tmp_path / "ncu"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")

    kernel = Kernel(
        spec=KernelSpec(
            name="my_elementwise",
            kernel_type=KernelType.ELEMENTWISE,
            entrypoint="elementwise_add_kernel",
        ),
        source_code=(
            "def elementwise_add_kernel(*args, **kwargs):\n"
            "    return None\n"
        ),
    )
    workload = {"uuid": "workload-0", "axes": {"N": 1024}, "inputs": {}}

    result = profile_kernel(
        kernel,
        workload,
        _identity_input_generator,
        hardware_spec=rtx6000_ada_hardware(),
        flops=1_000_000,
        nbytes=4_000_000,
        latency_s=1e-3,
        mode="curated",
        timeout_s=10.0,
        cache_dir=None,
    )

    assert isinstance(result, ProfilingResult)
    assert result.degraded is False, (
        f"profile_kernel degraded with reason={result.degraded_reason!r} — "
        "expected the post-profile ``ncu --import`` extract to recover "
        "the parser. Without the two-subprocess fix, this would be "
        "'csv_parse:no_header' (banners-only stdout from the capture call)."
    )
    assert result.ncu is not None
    assert result.ncu.sm_occupancy_pct == pytest.approx(55.0)
    assert result.ncu.warp_stall_dominant == "long_scoreboard"


def test_parser_would_fail_without_import_extract_call():
    """Regression-guard for design point #8: explicit assertion that the
    parser fails with ``csv_parse:no_header`` if ``_run_ncu`` ever returns
    the profile subprocess's banner-only stdout (i.e. a future cleanup
    that drops the post-profile ``ncu --import`` extract call would
    re-trigger the bug). Pure parser test — no subprocess. Cheap defense
    against ``_extract_ncu_csv`` being pruned on a future "is this second
    call redundant?" pass."""
    from src.eval.profiler import _parse_ncu_csv

    # Exactly what NCU 2025.1.1.0 writes to stdout when ``-o`` is set:
    # four ``==PROF==`` banners and nothing else. The two-subprocess fix
    # exists because this stdout has no CSV anywhere — only ``ncu --import
    # <rep> --csv`` re-emits parseable rows.
    banners_only_stdout = (
        "==PROF== Connected to process 1\n"
        "==PROF== Profiling kernel\n"
        "==PROF== Disconnected from process 1\n"
        "==PROF== Report saved\n"
    )

    ncu, raw, degraded, reason = _parse_ncu_csv(
        banners_only_stdout, "elementwise_add_kernel"
    )

    assert ncu is None
    assert raw == {}
    assert degraded is True
    # This exact slug is what surfaced (newly loud, thanks to
    # _log_degradation) on the 2026-05-08 live run. If the post-profile
    # ``ncu --import`` extract call is ever pruned, _run_ncu would feed
    # this banners-only stdout to the parser and we'd be back here.
    assert reason == "csv_parse:no_header"


# ── fallback-binary threaded into the capture subprocess ──────────────────
#
# Regression guards for the Codex adversarial review (2026-05-08): the
# initial ``_discover_ncu_binary`` fallback was effectively dead code in
# the clean-venv scenario it was designed to fix because ``_build_ncu_argv``
# still hardcoded argv[0] to bare ``"ncu"``. ``_run_ncu`` must substitute
# the discovered absolute path before ``subprocess.run`` so a PATH-clean +
# fallback-present host actually reaches NCU. JOURNAL 2026-05-08 amendment.


def test_run_ncu_uses_discovered_binary_when_path_misses(
    monkeypatch, sample_kernel, sample_workload
):
    """PATH-clean + fallback-present: ``_run_ncu`` must invoke
    ``subprocess.run`` with the discovered absolute path (not bare
    ``"ncu"``). Without the fix, ``subprocess.run(["ncu", ...])`` raises
    FileNotFoundError on a clean venv and every profile degrades as
    ``ncu_binary_not_found`` even though ``_discover_ncu_binary`` returned
    a usable path."""
    import subprocess
    from src.eval import profiler as profiler_mod

    profiler_mod._reset_ncu_permission_cache()
    monkeypatch.delenv(profiler_mod._NCU_DISABLE_ENV, raising=False)
    # Force discovery to take the cuda-12.8 fallback path (simulates clean
    # ~/.venvs/acts_run_venv before the activate-script PATH patch).
    force_ncu_discovery(monkeypatch, fallback_present=True)

    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        # Return a healthy CompletedProcess so ``_run_ncu`` follows the
        # success path. ``ncu_rep_out`` is None in this test so the second
        # subprocess (``--import``) is not invoked.
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(profiler_mod.subprocess, "run", fake_run)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert "argv" in captured, "subprocess.run was never called"
    argv = captured["argv"]
    assert argv[0] == profiler_mod._NCU_FALLBACK_PATH, (
        f"_run_ncu invoked subprocess with argv[0]={argv[0]!r}; "
        f"expected the discovered absolute path "
        f"{profiler_mod._NCU_FALLBACK_PATH!r}. Bare 'ncu' would raise "
        "FileNotFoundError on a PATH-clean host and defeat the fallback."
    )


def test_extract_ncu_csv_uses_discovered_binary_when_path_misses(
    tmp_path, monkeypatch
):
    """Companion guard: ``_extract_ncu_csv`` already substitutes
    ``_discover_ncu_binary()`` for argv[0] (see profiler.py:506-516). Pin
    that behavior so a future refactor doesn't regress it back to bare
    ``"ncu"`` and re-create the same dead-fallback bug on the import call."""
    import subprocess
    from src.eval import profiler as profiler_mod
    from src.eval.profiler import _extract_ncu_csv

    force_ncu_discovery(monkeypatch, fallback_present=True)

    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(profiler_mod.subprocess, "run", fake_run)

    rep_path = tmp_path / "report.ncu-rep"
    rep_path.write_text("ncu-rep-marker")

    _extract_ncu_csv(rep_path)

    assert "argv" in captured
    argv = captured["argv"]
    assert argv[0] == profiler_mod._NCU_FALLBACK_PATH


def test_extract_ncu_csv_propagates_tmpdir_env(tmp_path, monkeypatch):
    """``_extract_ncu_csv`` must pass the same user-scoped ``TMPDIR``
    env that ``_run_ncu`` constructs, otherwise the import subprocess
    inherits the process default ``/tmp`` and can hit the
    ``nsight-compute-lock`` ownership/permission failure on shared hosts
    after capture succeeds."""
    import subprocess
    from src.eval import profiler as profiler_mod
    from src.eval.profiler import _extract_ncu_csv, _ncu_tmpdir

    monkeypatch.setattr(
        profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False
    )

    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(profiler_mod.subprocess, "run", fake_run)

    rep_path = tmp_path / "report.ncu-rep"
    rep_path.write_text("ncu-rep-marker")

    _extract_ncu_csv(rep_path)

    assert "env" in captured, "subprocess.run was never called"
    env = captured["env"]
    assert env is not None, (
        "_extract_ncu_csv invoked subprocess.run without env=; the import "
        "call inherits process /tmp and can hit nsight-compute-lock ownership "
        "failures on shared hosts. Mirror _run_ncu's env construction."
    )
    assert env.get("TMPDIR") == _ncu_tmpdir(), (
        f"TMPDIR mismatch: expected {_ncu_tmpdir()!r}, got {env.get('TMPDIR')!r}. "
        "Must match _run_ncu's TMPDIR exactly so capture and import see the "
        "same user-scoped tempdir."
    )


# ── NCU process-group isolation (start_new_session) ───────────────────────
#
# Both NCU subprocess invocations must launch in a new session so that a
# SIGKILL of the parent ACTS process does not propagate to NCU mid-profile.
# GPU 0 runs in persistence mode; an orphaned NCU killed mid-write leaves
# CUDA context + clock-lock state stranded for tens of seconds. Pin the
# kwarg here so a future refactor of either call site doesn't silently
# drop the isolation.


def test_run_ncu_uses_new_session(
    monkeypatch, sample_kernel, sample_workload
):
    """``_run_ncu`` must pass ``start_new_session=True`` to ``subprocess.run``
    so NCU runs in its own session — see module-docstring rationale."""
    import subprocess
    from src.eval import profiler as profiler_mod

    profiler_mod._reset_ncu_permission_cache()
    monkeypatch.delenv(profiler_mod._NCU_DISABLE_ENV, raising=False)
    force_ncu_discovery(monkeypatch, fallback_present=True)

    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["start_new_session"] = kwargs.get(
            "start_new_session", "<MISSING>"
        )
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(profiler_mod.subprocess, "run", fake_run)

    _run_ncu(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        timeout_s=10.0,
        mode="curated",
    )

    assert captured.get("start_new_session") is True, (
        "_run_ncu must pass start_new_session=True so NCU is isolated from "
        "the parent's signal group. A SIGKILL'd parent with persistence-mode "
        "GPU 0 leaves an orphan NCU stranding CUDA context + clock-lock state."
    )


def test_extract_ncu_csv_uses_new_session(tmp_path, monkeypatch):
    """``_extract_ncu_csv`` must pass ``start_new_session=True``. Same
    rationale as ``_run_ncu``: the import call also holds GPU state briefly
    via NCU's CUDA-aware decode path on some driver versions, and we want
    both legs of the profile to share isolation semantics."""
    import subprocess
    from src.eval import profiler as profiler_mod
    from src.eval.profiler import _extract_ncu_csv

    force_ncu_discovery(monkeypatch, fallback_present=True)

    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["start_new_session"] = kwargs.get(
            "start_new_session", "<MISSING>"
        )
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(profiler_mod.subprocess, "run", fake_run)

    rep_path = tmp_path / "report.ncu-rep"
    rep_path.write_text("ncu-rep-marker")

    _extract_ncu_csv(rep_path)

    assert captured.get("start_new_session") is True, (
        "_extract_ncu_csv must pass start_new_session=True so the import "
        "subprocess shares isolation semantics with the capture call."
    )
