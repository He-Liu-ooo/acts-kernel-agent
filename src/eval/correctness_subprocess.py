"""Async parent-side helper around src/eval/correctness_worker.py.

Mirrors src/eval/bench_subprocess.py: write request.json, spawn the worker
as `python -m src.eval.correctness_worker`, await exit, parse response.json.
Any failure (non-zero exit, signal, missing/malformed response, timeout) is
reported as a FAIL-CLOSED CorrectnessResult — the parent is never poisoned by
the candidate launch, because the launch happened in the (now-dead) child.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from src.eval.worker_spawn import spawn_worker

# Overridable in tests to point at a stub worker script.
_WORKER_ARGV = [sys.executable, "-m", "src.eval.correctness_worker"]


class CorrectnessIsolationError(RuntimeError):
    """Raised when a correctness check cannot isolate an untrusted kernel launch
    (no ``problem_definition_path`` for the subprocess) and the caller did not
    explicitly opt into the in-parent fallback via ``allow_in_parent_fallback=True``.
    Distinct from a correctness FAILURE (``CorrectnessResult(passed=False)``) and a
    worker crash (``WorkerCrashed``): a misconfiguration tripwire that keeps the
    'no untrusted launch in the parent' invariant enforced by construction."""


@dataclass
class CorrectnessResult:
    passed: bool
    failed_stage: str | None = None
    error_message: str | None = None
    max_err: float = 0.0
    total_workloads: int = 0
    failed_workload_idx: int | None = None


async def run_correctness_subprocess(
    *,
    request: dict,
    worker_dir: Path,
    timeout_s: float,
) -> CorrectnessResult:
    # The spawn / wait / timeout-kill / malformed-guard sequence is shared
    # with the bench helper; ``spawn_worker`` owns it. ``_WORKER_ARGV`` is
    # forwarded as ``argv_prefix`` so tests can inject a stub worker script.
    # The correctness contract maps the outcome to a FAIL-CLOSED
    # ``CorrectnessResult`` (vs. bench's ``WorkerCrashed``): timeout and
    # crash both yield ``passed=False`` with the worker.log tail as
    # ``error_message``; the parent is never poisoned by the candidate
    # launch, because the launch happened in the (now-dead) child.
    outcome = await spawn_worker(
        module="src.eval.correctness_worker",
        request=request,
        worker_dir=worker_dir,
        timeout_s=timeout_s,
        argv_prefix=_WORKER_ARGV,
    )
    if outcome.status == "timeout":
        return CorrectnessResult(
            passed=False, failed_stage="timeout",
            error_message=outcome.log_tail,
        )
    if outcome.status == "crashed":
        return CorrectnessResult(
            passed=False, failed_stage="worker_crashed",
            error_message=outcome.log_tail,
        )
    data = outcome.response
    return CorrectnessResult(
        passed=bool(data.get("passed", False)),
        failed_stage=data.get("failed_stage"),
        error_message=data.get("error_message"),
        max_err=float(data.get("max_err", 0.0)),
        total_workloads=int(data.get("total_workloads", 0)),
        failed_workload_idx=data.get("failed_workload_idx"),
    )


def build_correctness_request(
    *,
    spec,
    source_code: str,
    dps: bool,
    definition_path,
    workloads,
    blob_roots,
    mode: str,
    input_seed: int,
    anti_cheat_critical_names,
    strict_atol: float | None = None,
    strict_rtol: float | None = None,
) -> dict:
    """Build the correctness-worker IPC request. Single source of truth for the
    request schema shared by the Coder gate tool, baseline post-verify, and the
    reward-hack strict re-eval (avoids three hand-maintained copies)."""
    # Lazy import dodges the correctness_subprocess <-> orchestrator import cycle.
    from src.search.orchestrator import _serialize_kernel_spec_for_request
    request = {
        "schema_version": 1,
        "mode": mode,
        "kernel_spec": _serialize_kernel_spec_for_request(spec),
        "source_code": source_code,
        "dps": dps,
        "definition_path": str(definition_path),
        "workloads": [w.model_dump(mode="json") for w in workloads],
        "blob_roots": [str(p) for p in (blob_roots or [])],
        "input_seed": input_seed,
        "anti_cheat_critical_names": list(anti_cheat_critical_names),
    }
    if mode == "strict_recheck":
        request["strict_atol"] = 1e-5 if strict_atol is None else strict_atol
        request["strict_rtol"] = 1e-4 if strict_rtol is None else strict_rtol
    return request
