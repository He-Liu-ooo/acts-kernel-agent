"""Tests for ``_run_ncu`` — the NCU subprocess driver.

These tests verify only the subprocess plumbing: argv construction,
environment setup, and failure handling. CSV parsing is already covered
by ``tests/test_profiler_csv.py`` — the driver must pass stdout through
to the parser without interpreting it.

Tier 1: GPU-free. A ``fake_ncu`` shell script on ``$PATH`` stands in for
the real ``ncu`` binary. Runs in ``/tmp/acts_test_venv``.
"""

from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path

import pytest

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
    assert reason == "ncu_nonzero_exit:3"


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
        json.dumps({"uuid": "wl0", "axes": {"N": 128}}) + "\n"
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

    result = _profiler_driver._build_inputs(
        problem_dir,
        {"uuid": "wl0", "axes": {"N": 128}},
        seed=7,
    )

    assert result == ("ok", 7)
    assert captured["problem_name"] == "elementwise_id"
    assert captured["problem_op_type"] == "elementwise"
    assert captured["workload_uuid"] == "wl0"
    assert captured["workload_axes"] == {"N": 128}


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
