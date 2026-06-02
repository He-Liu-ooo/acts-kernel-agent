# src/eval/correctness_worker.py
"""Crash-isolated correctness worker (child process).

Runs `python -m src.eval.correctness_worker --request r --response w`. Rebuilds
the candidate from the request, compiles, and verifies it against the PyTorch
oracle IN THIS PROCESS — so an out-of-bounds launch that poisons the CUDA
context dies here, never in the parent. Two modes:

  * "gate"          — verify_correctness 5-stage gate per workload (Coder tool).
  * "strict_recheck"— anti-cheat + compare_outputs(strict atol/rtol) per
                      workload (orchestrator reward-hack re-eval).

torch/SOL are imported lazily (via the seams below) so the module is importable
under the Tier-1 torchless venv; tests monkeypatch the seams.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Reuse the bench worker's rehydration helpers (same package; intentional).
from src.eval.bench_worker import (
    _load_definition,
    _rehydrate_kernel_spec,
    _rehydrate_workloads,
    _run_per_iter_anti_cheat,
)


def _build_input_generators(request: dict, workloads: list, definition=None) -> list:
    """Build per-workload input generators, reading ``blob_roots`` from the
    TOP-LEVEL request field (the correctness IPC schema). bench_worker's
    namesake reads ``request['profile_config']['blob_roots']``, which the
    correctness requests never send — reusing it silently dropped blob roots
    so SafetensorsInput workloads failed to load. (Codex adversarial review
    2026-05-30, finding 2.)"""
    from src.eval.inputs import build_input_generator
    if definition is None:
        definition = _load_definition(Path(request["definition_path"]))
    blob_roots_raw = request.get("blob_roots") or []
    blob_roots = [Path(p) for p in blob_roots_raw] if blob_roots_raw else None
    return [build_input_generator(definition, wl, blob_roots=blob_roots) for wl in workloads]


# --- lazy seams (monkeypatched in Tier-1 tests) ---------------------------
def _build_reference_fn(source: str):
    from src.eval.inputs import build_reference_fn
    return build_reference_fn(source)


def _make_kernel(spec, source_code: str, dps: bool):
    from src.kernels.kernel import Kernel
    return Kernel(spec=spec, source_code=source_code, dps=dps)


def _compile(kernel):
    from src.kernels.compiler import compile_kernel
    return compile_kernel(kernel)


def run_request(request: dict) -> dict:
    # SECURITY: capture trusted verifier references BEFORE the candidate's
    # module-scope code runs (it executes inside ``_compile`` via exec_module).
    # A candidate that rebinds ``src.eval.correctness.verify_correctness`` /
    # ``strict_compare_one_workload`` at import time cannot affect these
    # already-bound locals, so it cannot forge a pass by patching the oracle.
    # (Codex 2026-06-01 review. The broader vector — a candidate patching torch
    # primitives the oracle calls — remains under the deferred Tier-3 eval-driver
    # isolation; trigger: silent oracle corruption observed.)
    from src.eval.correctness import (
        verify_correctness as _trusted_verify,
        strict_compare_one_workload as _trusted_strict,
    )
    from sol_execbench.core.bench.reward_hack import RewardHackDetected

    mode = request.get("mode", "gate")
    spec = _rehydrate_kernel_spec(request["kernel_spec"])
    kernel = _make_kernel(spec, request["source_code"], bool(request.get("dps", False)))

    compiled = _compile(kernel)
    if not compiled.success or compiled.compiled_fn is None:
        return _resp(False, "compile", compiled.error_message, 0.0, 0, None)
    cand_fn = compiled.compiled_fn

    definition = _load_definition(Path(request["definition_path"]))
    workloads = _rehydrate_workloads(request["workloads"])
    generators = _build_input_generators(request, workloads, definition)
    reference_fn = _build_reference_fn(definition.reference)
    total = len(workloads)
    seed = int(request.get("input_seed", 0))
    anti_cheat = request.get("anti_cheat_critical_names", [])

    if mode == "strict_recheck":
        atol = float(request.get("strict_atol", 1e-5))
        rtol = float(request.get("strict_rtol", 1e-4))
        try:
            with _run_per_iter_anti_cheat(anti_cheat):
                for idx, (wl, gen) in enumerate(zip(workloads, generators)):
                    ok = _trusted_strict(
                        candidate_fn=cand_fn, reference_fn=reference_fn,
                        input_generator=gen, definition=definition,
                        kernel=kernel, workload=wl,
                        seed=seed, atol=atol, rtol=rtol,
                    )
                    if not ok:
                        return _resp(False, "strict_mismatch",
                                     f"strict re-eval mismatch on workload {idx + 1}/{total}",
                                     0.0, total, idx + 1)
        except RewardHackDetected:
            return _resp(False, "reward_hack", "RewardHackDetected during re-eval",
                         0.0, total, None)
        return _resp(True, None, None, 0.0, total, None)

    # mode == "gate"
    max_err = 0.0
    with _run_per_iter_anti_cheat(anti_cheat):
        for idx, (wl, gen) in enumerate(zip(workloads, generators)):
            result = _trusted_verify(candidate_fn=cand_fn, reference_fn=reference_fn,
                                     input_generator=gen, definition=definition,
                                     kernel=kernel, workload=wl, policy=None)
            if not result.passed:
                stage = result.failed_stage.value if result.failed_stage else "unknown"
                return _resp(False, stage, result.error_message, 0.0, total, idx + 1)
            max_err = max(max_err, result.max_abs_error)
    return _resp(True, None, None, max_err, total, None)


def _resp(passed, failed_stage, error_message, max_err, total, failed_idx) -> dict:
    return {
        "schema_version": 1, "passed": passed, "failed_stage": failed_stage,
        "error_message": error_message, "max_err": max_err,
        "total_workloads": total, "failed_workload_idx": failed_idx,
    }


def _main_cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="ACTS correctness worker subprocess")
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--response", required=True, type=Path)
    args = p.parse_args(argv)
    try:
        request = json.loads(args.request.read_text())
        response = run_request(request)
        args.response.write_text(json.dumps(response))
        return 0
    except Exception:  # noqa: BLE001 — uncaught → parent treats as worker_crashed
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(_main_cli(sys.argv[1:]))
