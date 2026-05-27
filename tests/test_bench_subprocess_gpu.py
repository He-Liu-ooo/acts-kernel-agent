"""Tier 2 GPU smoke tests for the bench-subprocess isolation refactor.

Exercises ``src/eval/bench_subprocess.py`` + ``src/eval/bench_worker.py``
end-to-end against a real GPU and the live ``069_rms_norm`` SOL problem.
Verifies (1) successful subprocess spawn -> response.json round-trip,
(2) worker-crash ``WorkerCrashed`` signal path on a bad
``definition_path``, and (3) ``child_walltime_s`` sanity bound.

Each test spawns a real ``python -m src.eval.bench_worker`` child via
``run_bench_subprocess`` -- we deliberately operate at the helper level
rather than going through ``src/pipeline/optimize.py``, so the
subprocess plumbing is exercised in isolation from the LLM agent stack.

Requires the Tier 2 venv ``~/.venvs/acts_run_venv`` (Python 3.12 +
cu128 torch + sol_execbench + openai-agents + SOLAR). Tests are marked
``@pytest.mark.gpu`` so Tier 1 sweeps skip them.

See ``doc/specs/2026-05-24-bench-subprocess-isolation-design.md`` Section 10
for the test plan.

Skip discipline
---------------
The end-to-end + walltime tests skip cleanly if the SOL problem dir or
the smoke kernel fixture isn't on disk (so the suite stays portable to
a checkout that hasn't fetched SOL data yet). The crash test does not
depend on either fixture and always runs under the gpu marker.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RMSNORM_PROBLEM = Path(
    "~/workspace/projects/self-evolved-llm/repo/benchmark/"
    "SOL-ExecBench/data/benchmark/L1/069_rms_norm"
).expanduser()
SMOKE_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "operator_baselines" / "smoke_rmsnorm.py"
)


def _skip_if_problem_missing() -> None:
    """Skip the test when the SOL problem dir or the smoke fixture is absent."""
    if not RMSNORM_PROBLEM.exists():
        pytest.skip(f"SOL problem path missing: {RMSNORM_PROBLEM}")
    if not SMOKE_FIXTURE.exists():
        pytest.skip(f"smoke kernel fixture missing: {SMOKE_FIXTURE}")


def _smallest_workload(workloads):
    """Pick the workload with the lowest total element count.

    Element count proxy: product of all numeric axis values. The RMSNorm
    problem's ``hidden_size`` is a const (8192) and ``batch_size`` /
    ``seq_len`` are vars, so the smallest workload is whichever
    minimises ``batch_size * seq_len`` for the bench's bf16 tensors.
    """
    def _size(wl):
        prod = 1
        for v in wl.axes.values():
            if isinstance(v, (int, float)):
                prod *= int(v)
        return prod

    return min(workloads, key=_size)


def _build_smoke_request(tmp_path: Path) -> tuple[dict, Path]:
    """Construct a valid request targeting 069_rms_norm with K=1 smoke kernel.

    Returns ``(request, worker_dir)``. The caller invokes
    ``asyncio.run(run_bench_subprocess(request=..., worker_dir=...))``.

    Uses the production serialisation helpers
    (``_serialize_kernel_spec_for_request``, ``detect_hardware``) so the
    request shape mirrors what the orchestrator emits in a live run --
    if those helpers drift, this test fails the same way production
    would.
    """
    from src.benchmarks.sol_execbench import load as sol_load
    from src.config import detect_hardware
    from src.kernels.kernel import KernelSpec, KernelType
    from src.search.orchestrator import _serialize_kernel_spec_for_request

    definition, workloads = sol_load(RMSNORM_PROBLEM)
    picked = _smallest_workload(workloads)

    spec = KernelSpec(
        name=definition.name,
        kernel_type=KernelType.RMSNORM,
        input_shapes=[dict(definition.const_axes)] if definition.const_axes else [],
        definition_path=RMSNORM_PROBLEM,
        pytorch_reference=definition.reference,
        # smoke_rmsnorm.py exposes a host wrapper `kernel_fn(...)` which
        # is the default KernelSpec.entrypoint -- explicit here to make
        # the binding contract visible.
        entrypoint="kernel_fn",
    )

    source = SMOKE_FIXTURE.read_text()

    candidates = [
        {
            "candidate_idx": 0,
            "source_code": source,
            "triton_kernel_name": "rmsnorm_residual_kernel",
            "entrypoint": "kernel_fn",
            "dps": False,
        },
    ]

    hardware = detect_hardware()
    worker_dir = tmp_path / "iter_0" / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)

    from src.eval.bench_worker import build_request

    request = build_request(
        run_dir=tmp_path,
        iter_no=0,
        worker_dir=worker_dir,
        ncu_cache_dir=tmp_path / "ncu_cache",
        candidates=candidates,
        kernel_spec=_serialize_kernel_spec_for_request(spec),
        workloads=[picked.model_dump(mode="json")],
        definition_path=RMSNORM_PROBLEM,
        hardware_spec=hardware,
        # Minimal anti-cheat set -- the smoke kernel is honest, but the
        # worker's per-candidate ``per_iter_anti_cheat`` window still
        # needs to be exercised with a non-empty list so the patch-
        # snapshot path runs end-to-end.
        anti_cheat_critical_names=["elapsed_time", "synchronize"],
        # Small bench counts keep wallclock down; mirrors the shape used
        # in tests/test_bench_worker.py::_build_minimal_request.
        bench_config={"warmup_runs": 5, "timed_runs": 10, "burn_in_seed": -1},
        profile_config={
            "ncu_enabled": True,
            "analytical_enabled": True,
            # iter_flops / iter_nbytes = 0 is acceptable -- profile_kernel
            # routes around them; analytical metrics degrade gracefully
            # to None without affecting NCU collection.
            "iter_flops": 0,
            "iter_nbytes": 0,
            "repr_workload_idx": 0,
            # Wire t_sol_us / baseline_latency_us so the child's profile
            # gauntlet can rank survivors via sol_score without falling
            # back to the divide-by-zero defaults.
            "t_sol_us": 10.0,
            "baseline_latency_us": 100.0,
            "problem_definition_path": str(RMSNORM_PROBLEM),
            "blob_roots": [str(RMSNORM_PROBLEM)],
        },
    )
    return request, worker_dir


@pytest.mark.gpu
def test_real_subprocess_end_to_end_rmsnorm(tmp_path):
    """Spawn real bench_worker subprocess on a K=1 smoke rmsnorm candidate.

    Asserts: response.json schema_version=1; winner_idx=0; cand_0 status
    is success; channel-A flag false; child_walltime_s > 0; the worker
    directory carries response.json + events.jsonl artifacts; and a
    ``.ncu-rep`` lands for the winner whenever profile collection
    succeeded (degraded-NCU runs are allowed -- they leave
    ``winner_profile`` null but still mark the candidate successful).
    """
    _skip_if_problem_missing()

    from src.eval.bench_subprocess import run_bench_subprocess

    request, worker_dir = _build_smoke_request(tmp_path)

    response = asyncio.run(
        run_bench_subprocess(
            request=request,
            worker_dir=worker_dir,
            worker_crash_threshold=3,
            worker_timeout_s=600.0,
        )
    )

    # Response shape.
    assert response["schema_version"] == 1
    assert response["iter_no"] == 0
    assert response["aborted_by_channel_A"] is False
    assert response["child_walltime_s"] > 0.0, (
        f"child_walltime_s should be positive; got {response['child_walltime_s']}"
    )

    cands = response["candidates"]
    assert len(cands) == 1, f"expected 1 candidate result, got {len(cands)}"
    assert cands[0]["candidate_idx"] == 0
    assert cands[0]["status"] == "success", (
        f"cand 0 status={cands[0]['status']!r} reason={cands[0].get('reason')!r}"
    )
    assert response["winner_idx"] == 0, (
        f"winner_idx={response['winner_idx']!r}; expected 0"
    )

    # Filesystem artifacts always present after a clean exit.
    assert (worker_dir / "response.json").exists()
    assert (worker_dir / "request.json").exists()
    assert (worker_dir / "worker.log").exists()
    events_path = worker_dir / "events.jsonl"
    # The child only emits events from the profile gauntlet (per the
    # worker docstring -- bench/coder_failed events are parent-emitted),
    # so on the happy path the file may be empty or absent. We only
    # require its presence WHEN the profile gauntlet recorded a signal.
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            if line.strip():
                json.loads(line)  # well-formed JSON-per-line

    # NCU rep handling. The profile gauntlet always returns a
    # ProfilingResult on a winner -- even when NCU itself fails, the
    # result carries ``degraded_reason`` (e.g. permission denied on
    # GPU counters) and ``ncu_rep_path=None``. The right invariant for
    # the on-disk .ncu-rep file therefore keys on the response's
    # ``ncu_rep_path``, not on ``winner_profile is not None``.
    winner_profile = response["winner_profile"]
    assert winner_profile is not None, (
        "winner_profile should always be populated when winner_idx is "
        "set -- the gauntlet returns a degraded result rather than None "
        "on NCU failure"
    )
    rep_path = winner_profile.get("ncu_rep_path")
    if rep_path is not None:
        rep = worker_dir / "cand_0.ncu-rep"
        assert rep.exists(), (
            f"winner_profile.ncu_rep_path={rep_path!r} but cand_0.ncu-rep "
            f"was not written into the worker dir -- profile gauntlet "
            f"rename step may have regressed"
        )


@pytest.mark.gpu
def test_real_subprocess_worker_crashes_on_bad_definition_path(tmp_path):
    """Force a worker crash by pointing definition_path at an empty dir.

    The bench worker resolves ``definition.json`` from the request's
    ``definition_path``. Pointing it at a real directory that contains
    no ``definition.json`` makes the SOL loader raise FileNotFoundError
    inside ``_load_definition`` -- which propagates up through
    ``_build_input_generators`` because we also pass a non-empty
    workload list (forcing the production branch in ``run_iter``).
    ``_main_cli``'s top-level except prints the traceback to stderr and
    returns 1, and the helper translates that to ``WorkerCrashed``.

    We assert: returncode == 1; stderr_tail is non-empty and carries a
    recognisable failure marker; worker.log is preserved as a
    postmortem artifact; response.json was never written.
    """
    from src.eval.bench_subprocess import run_bench_subprocess, WorkerCrashed

    bad_problem_dir = tmp_path / "empty_problem"
    bad_problem_dir.mkdir()  # exists but contains no definition.json

    worker_dir = tmp_path / "iter_0" / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)

    # Minimal valid-shape request with a non-empty workload list so the
    # production branch ('workloads -> _build_input_generators ->
    # _load_definition') runs and raises. The candidate source never
    # gets executed -- the worker dies before the candidate loop.
    request = {
        "schema_version": 1,
        "run_dir": str(tmp_path),
        "iter_no": 0,
        "worker_dir": str(worker_dir),
        "ncu_cache_dir": str(tmp_path / "ncu_cache"),
        "candidates": [
            {
                "candidate_idx": 0,
                "source_code": "# placeholder -- never executed",
                "triton_kernel_name": "noop",
                "entrypoint": "kernel_fn",
                "dps": False,
            },
        ],
        "kernel_spec": {
            "name": "bad",
            "kernel_type": "custom",
            "flop_count": 0,
            "memory_bytes": 0,
            "input_shapes": [],
            "definition_path": str(bad_problem_dir),
            "pytorch_reference": "",
            "t_sol_us": None,
            "entrypoint": "kernel_fn",
        },
        # Non-empty workloads forces the worker into the production
        # 'load definition + build input generators' branch.
        "workloads": [
            {
                "uuid": "00000000-0000-0000-0000-000000000000",
                "axes": {"batch_size": 1, "seq_len": 1},
                "inputs": {},
                "tolerance": {"max_atol": 1e-05, "max_rtol": 0.05},
            },
        ],
        "definition_path": str(bad_problem_dir),
        "hardware_spec": {},
        "anti_cheat_critical_names": [],
        "bench_config": {"warmup_runs": 1, "timed_runs": 1, "burn_in_seed": -1},
        "profile_config": {
            "ncu_enabled": False,
            "analytical_enabled": False,
            "iter_flops": 0,
            "iter_nbytes": 0,
            "repr_workload_idx": 0,
            "t_sol_us": 1.0,
            "baseline_latency_us": 1.0,
            "problem_definition_path": str(bad_problem_dir),
            "blob_roots": [],
        },
    }

    with pytest.raises(WorkerCrashed) as exc_info:
        asyncio.run(
            run_bench_subprocess(
                request=request,
                worker_dir=worker_dir,
                worker_crash_threshold=3,
                worker_timeout_s=120.0,
            )
        )

    exc = exc_info.value
    assert exc.returncode == 1, (
        f"expected returncode=1 from the worker's top-level try/except "
        f"on FileNotFoundError; got {exc.returncode}. "
        f"stderr_tail:\n{exc.stderr_tail[-500:]}"
    )
    assert exc.stderr_tail, "stderr_tail should be non-empty on crash"
    tail_lower = exc.stderr_tail.lower()
    assert any(
        marker in tail_lower
        for marker in ("traceback", "filenotfound", "definition.json", "no such file")
    ), (
        "stderr_tail should carry a recognisable failure marker; got:\n"
        f"{exc.stderr_tail[-1000:]}"
    )

    # Postmortem artifacts: log persisted, response NOT written.
    assert (worker_dir / "worker.log").exists(), (
        "worker.log should be preserved for postmortem on crash"
    )
    assert not (worker_dir / "response.json").exists(), (
        "response.json should be absent when the worker crashed before write"
    )


@pytest.mark.gpu
def test_real_subprocess_walltime_sanity(tmp_path):
    """``child_walltime_s`` should land in a sane window for a smoke run.

    Generous upper bound (120s) -- NCU alone takes ~30s, plus Python +
    torch + SOL startup (~3-5s) and the autotune burn-in. Failing this
    bound on a healthy host indicates either a regression in the
    subprocess plumbing or a workload/NCU-cache miss that needs
    investigation rather than bumping the bound.
    """
    _skip_if_problem_missing()

    from src.eval.bench_subprocess import run_bench_subprocess

    request, worker_dir = _build_smoke_request(tmp_path)

    response = asyncio.run(
        run_bench_subprocess(
            request=request,
            worker_dir=worker_dir,
            worker_crash_threshold=3,
            worker_timeout_s=600.0,
        )
    )

    walltime = response["child_walltime_s"]
    assert isinstance(walltime, (int, float)), (
        f"child_walltime_s should be numeric; got {type(walltime).__name__}"
    )
    # `_encode` maps inf/nan to None, so a finite-numeric check is enough.
    assert walltime is not None and walltime == walltime  # noqa: PLR0124 -- NaN guard
    assert walltime > 0.0, f"walltime should be positive; got {walltime}"
    assert walltime < 120.0, (
        f"walltime={walltime:.2f}s exceeds 120s sanity bound -- investigate "
        f"NCU / autotune / subprocess startup overhead before bumping"
    )
