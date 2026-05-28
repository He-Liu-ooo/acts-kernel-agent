"""Per-iter K-way bench + NCU profile worker subprocess.

Spawned by ``src/eval/bench_subprocess.py`` once per iter after the
orchestrator's K-way Coder fan-out returns K candidate kernels. Runs
compile + autotune burn-in + benchmark + (winner) NCU profile inside a
fresh process, then exits. The parent reads ``response.json`` as the
source of truth for per-candidate verdicts.

Module-level lazy imports
-------------------------
torch / sol_execbench / SOL types are loaded LAZILY inside
``run_iter`` (and its helpers) so this module is importable on the
torchless Tier-1 venv for unit-testing the encoder + ``build_request``
in isolation. Tests that exercise ``run_iter`` monkeypatch the
lazy-bound symbols at ``src.eval.bench_worker.<name>``.

See ``doc/specs/2026-05-24-bench-subprocess-isolation-design.md`` (the
spec lives uncommitted; load-bearing decisions retire to
JOURNAL/PRD/PROCESS/doc as the feature lands).
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import logging
import math
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-bound symbols. Real bindings are installed by ``_lazy_imports()`` on
# the first ``run_iter`` call. Tests monkeypatch these directly via
# ``monkeypatch.setattr("src.eval.bench_worker.<name>", ...)``.
# ---------------------------------------------------------------------------
benchmark_kernel: Any = None
BenchmarkError: Any = None
profile_kernel: Any = None
ProfilerError: Any = None
check_lazy_outputs_after_bench: Any = None
RewardHackDetected: Any = None


def _lazy_imports() -> None:
    """Bind torch/sol_execbench-dependent symbols at first ``run_iter`` call.

    Tests pre-populate the call-site symbols (``benchmark_kernel`` /
    ``profile_kernel`` / ``check_lazy_outputs_after_bench``) via
    monkeypatch BEFORE calling ``run_iter``; when those are already
    non-None the corresponding production import is skipped so the
    monkeypatched stub isn't clobbered.

    The exception classes (``BenchmarkError`` / ``ProfilerError`` /
    ``RewardHackDetected``) are bound UNCONDITIONALLY — `except` clauses
    need a real class object regardless of whether the call site is
    mocked.
    """
    global benchmark_kernel, BenchmarkError, profile_kernel, ProfilerError
    global check_lazy_outputs_after_bench, RewardHackDetected

    if BenchmarkError is None:
        from src.eval.benchmark import BenchmarkError as _BenchErr
        BenchmarkError = _BenchErr
    if benchmark_kernel is None:
        from src.eval.benchmark import benchmark_kernel as _bench
        benchmark_kernel = _bench
    if ProfilerError is None:
        from src.eval.profiler import ProfilerError as _PrfErr
        ProfilerError = _PrfErr
    if profile_kernel is None:
        from src.eval.profiler import profile_kernel as _profile
        profile_kernel = _profile
    if check_lazy_outputs_after_bench is None:
        from src.eval.anti_cheat import check_lazy_outputs_after_bench as _check
        check_lazy_outputs_after_bench = _check
    if RewardHackDetected is None:
        from sol_execbench.core.bench.reward_hack import RewardHackDetected as _RHD
        RewardHackDetected = _RHD


# ---------------------------------------------------------------------------
# JSON encoders / decoders (Task 2)
# ---------------------------------------------------------------------------


def _encode(obj: Any) -> Any:
    """Recursive JSON-safe encoder mirroring ``tree_dump.py::_serialize_*`` style.

    Handles:
      * ``Path`` → ``str``
      * ``Enum`` → ``.value``
      * ``dataclass`` instance → recursive ``dataclasses.asdict``
      * ``float('inf')`` / ``float('nan')`` → ``None`` (RFC-8259 valid JSON)
      * ``dict`` / ``list`` / ``tuple`` → element-wise recurse
      * other primitives passed through unchanged
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _encode(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(data: Any, target_type: type | None = None) -> Any:
    """Reverse of ``_encode`` for known target dataclass types.

    Without ``target_type``, returns ``data`` unchanged — parents that
    want a raw dict back from response.json use the no-arg form. When a
    target dataclass is supplied, accepts only fields the dataclass
    declares (other keys silently dropped); used at the parent boundary
    for ``BenchmarkResult`` / ``ProfilingResult`` rehydration.
    """
    if target_type is None:
        return data
    if dataclasses.is_dataclass(target_type) and isinstance(data, dict):
        field_names = {f.name for f in dataclasses.fields(target_type)}
        kwargs = {k: v for k, v in data.items() if k in field_names}
        return target_type(**kwargs)
    return data


# ---------------------------------------------------------------------------
# request.json assembly (Task 2)
# ---------------------------------------------------------------------------


def build_request(
    *,
    run_dir: Path,
    iter_no: int,
    worker_dir: Path,
    ncu_cache_dir: Path,
    candidates: list[dict],
    kernel_spec: dict,
    workloads: list[dict],
    definition_path: Path,
    hardware_spec: Any,
    anti_cheat_critical_names: list[str],
    bench_config: dict,
    profile_config: dict,
) -> dict:
    """Assemble request.json payload for the bench-worker subprocess.

    The parent owns serialization; the child treats this dict as
    authoritative input. Per-candidate ``candidate_idx`` is preserved
    across the entrypoint-binding pre-filter — if the parent dropped
    cand_idx=1 from the original K, ``candidates`` here carries
    ``[{candidate_idx:0}, {candidate_idx:2}, ...]`` and the child uses
    each entry's carried ``candidate_idx`` when writing response.json.

    See spec §5.1 for the full schema.
    """
    return {
        # 2 (2026-05-28): profile_kernel forwards input_dtypes through to
        # AnalyticalMetrics; the response carries the dtype label and
        # calibration_warning. Schema-bump rationale + migration notes in
        # doc/specs/2026-05-28-pct-peak-dtype-and-warmup-traceback-design.md.
        "schema_version": 2,
        "run_dir": str(run_dir),
        "iter_no": iter_no,
        "worker_dir": str(worker_dir),
        "ncu_cache_dir": str(ncu_cache_dir),
        "candidates": candidates,
        "kernel_spec": kernel_spec,
        "workloads": workloads,
        "definition_path": str(definition_path),
        "hardware_spec": _encode(hardware_spec),
        "anti_cheat_critical_names": list(anti_cheat_critical_names),
        "bench_config": bench_config,
        "profile_config": profile_config,
    }


# ---------------------------------------------------------------------------
# Helpers copied verbatim from src/search/orchestrator.py to keep the
# worker independent of orchestrator imports. The spec calls out
# _safe_precompile (orchestrator.py:106-138) and _select_best_candidate
# (orchestrator.py:149-178); see plan Task 3 step 3.3 for the rationale.
# ---------------------------------------------------------------------------


def _safe_precompile(
    kernel: Any,
    *,
    role: str,
) -> tuple[Any | None, Any | None]:
    """Best-effort precompile that returns ``(compiled_fn, autotuner)``.

    Verbatim port of ``src/search/orchestrator.py::_safe_precompile`` so
    the worker does not import the orchestrator. Falls back to
    ``(None, None)`` with a WARNING on any failure; ``benchmark_kernel``
    then takes the lazy-compile path internally.
    """
    from src.kernels.compiler import compile_kernel

    try:
        result = compile_kernel(kernel)
    except Exception as exc:  # noqa: BLE001 — match orchestrator semantics
        logger.warning(
            "%s pre-compile raised (%s: %s) — falling back to lazy "
            "compile; autotune_winner will be None.",
            role, type(exc).__name__, exc,
        )
        return None, None
    if result.success and result.compiled_fn is not None:
        return result.compiled_fn, result.triton_autotuner
    return None, None


def _select_best_candidate_local(
    bench_results: list[tuple[int, Any, Any, Any, Any]],
    *,
    t_sol_us: float,
    baseline_latency_us: float,
) -> tuple[int, Any, Any, Any, Any]:
    """Pick the highest-SOL-Score entry from *bench_results*.

    Verbatim port of ``src/search/orchestrator.py::_select_best_candidate``
    so the worker does not import the orchestrator. Each entry is
    ``(candidate_idx, coder_output, child_kernel, bench, autotuner)``.
    Ranks by ``sol_execbench.sol_score.sol_score``; tie-break is lowest
    ``candidate_idx`` for deterministic first-survivor selection.

    Imported lazily because ``sol_execbench`` is absent on the Tier-1
    test venv — callers that mock the full survivor set out of the test
    path don't pay the import.
    """
    from sol_execbench.sol_score import sol_score as _sol_score

    def _sort_key(entry):
        cand_idx, _coder_out, _kernel, bench, _autotuner = entry
        score = _sol_score(
            t_k=bench.median_latency_us / 1000.0,
            t_p=baseline_latency_us / 1000.0,
            t_sol=t_sol_us / 1000.0,
        )
        return (-score, cand_idx)

    return min(bench_results, key=_sort_key)


def _representative_latency_s(bench: Any, workloads: list[Any], repr_idx: int) -> float | None:
    """Return the representative workload's latency in seconds.

    Verbatim port of ``src/search/orchestrator.py::_representative_latency_s``.
    ``None`` when the representative workload failed or is out of range;
    falls back to aggregate median on the placeholder path.
    """
    if not workloads:
        return bench.median_latency_us / 1e6
    if repr_idx >= len(workloads):
        return None
    uuid = workloads[repr_idx].uuid
    latency_us = bench.per_workload_latency_us.get(uuid)
    if latency_us is None or not math.isfinite(latency_us):
        return None
    return latency_us / 1e6


# ---------------------------------------------------------------------------
# Worker-internal helpers (rehydration, anti-cheat wrapper, event sink)
# ---------------------------------------------------------------------------


def _rehydrate_kernel_spec(data: dict) -> Any:
    """Reconstruct a KernelSpec from the request's dict shape.

    Mirrors ``KernelSpec.from_dict`` (src/kernels/kernel.py) but tolerates
    missing optional fields the request might omit.
    """
    from src.kernels.kernel import KernelSpec, KernelType

    def_path = data.get("definition_path")
    return KernelSpec(
        name=data.get("name", ""),
        kernel_type=KernelType(data.get("kernel_type", "custom")),
        flop_count=int(data.get("flop_count", 0)),
        memory_bytes=int(data.get("memory_bytes", 0)),
        input_shapes=data.get("input_shapes", []),
        definition_path=Path(def_path) if def_path else None,
        pytorch_reference=data.get("pytorch_reference", ""),
        t_sol_us=data.get("t_sol_us"),
        entrypoint=data.get("entrypoint", "kernel_fn"),
    )


def _rehydrate_workloads(raw_workloads: list[dict]) -> list[Any]:
    """SOL pydantic re-validation of the workload list."""
    from sol_execbench.core.data import Workload

    return [Workload.model_validate(d) for d in raw_workloads]


def _build_input_generators(request: dict, workloads: list[Any]) -> list[Callable[[int], tuple]]:
    """Build per-workload input generators from the request + rehydrated workloads."""
    from src.eval.inputs import build_input_generator

    definition = _load_definition(Path(request["definition_path"]))
    blob_roots_raw = request.get("profile_config", {}).get("blob_roots") or []
    blob_roots = [Path(p) for p in blob_roots_raw] if blob_roots_raw else None
    return [
        build_input_generator(definition, wl, blob_roots=blob_roots)
        for wl in workloads
    ]


def _load_definition(definition_path: Path) -> Any:
    """Load the SOL Definition from the problem directory.

    ``definition_path`` may point to ``definition.json`` itself or its
    parent directory; we route through the SOL loader which expects a
    directory and returns ``(definition, workloads)`` — we keep only
    the definition.

    Returns ``None`` when ``definition_path`` is empty / missing
    (Codex 2026-05-27 fix #6): the orchestrator's request-build
    coerces a None ``problem_definition_path`` to ``Path("")`` at the
    IPC boundary. The in-process bypass then calls this with
    ``Path("")`` → ``sol_load`` would have read ``./definition.json``
    and crashed with FileNotFoundError, propagating past the
    orchestrator's ``except WorkerCrashed`` to abort the run. Return
    None instead so the caller (``_build_input_generators`` or the
    profile gauntlet) sees a degraded-but-running state — tests that
    mock benchmark_kernel never read the definition anyway.
    """
    if not str(definition_path) or str(definition_path) in (".", ""):
        return None
    problem_dir = (
        definition_path.parent if definition_path.suffix else definition_path
    )
    if not problem_dir.is_dir():
        return None
    from src.benchmarks.sol_execbench import load as sol_load
    definition, _ = sol_load(problem_dir)
    return definition


@contextmanager
def _run_per_iter_anti_cheat(critical_names: list[str]) -> Iterator[Any]:
    """Thin wrapper around ``per_iter_anti_cheat`` so tests can stub the
    context manager without importing torch."""
    from src.eval.anti_cheat import per_iter_anti_cheat

    with per_iter_anti_cheat(critical_names) as ctx:
        yield ctx


def _emit(worker_dir: Path, kind: str, **fields: Any) -> None:
    """Child-side event emit; appends one JSON line to worker/events.jsonl.

    Mirrors the canonical ``src/runtime/events.py::emit`` shape:
    ``{"ts": iso_ts(), "kind": kind, ...fields}``. The matching shape is
    load-bearing for downstream consumers that key on ``ts`` for
    ordering/windowing — after the parent's ``merge_worker_artifacts``
    line-concatenates the worker chunk into canonical events.jsonl,
    both sources must have the same record shape (Codex 2026-05-27
    fix). Never raises (matches parent ``emit`` discipline).

    Most per-candidate events (``bench_done`` for the winner,
    ``profile_done``, ``coder_failed`` for bench/channel-A failures,
    ``reward_hack_detected``) are emitted PARENT-SIDE based on the
    response.json verdicts — the worker no longer emits those (would
    duplicate after merge). The remaining worker-side emits cover
    signals only the worker sees: profile-gauntlet ``coder_failed`` on
    a non-winner candidate that was bench-successful but profile-failed.
    """
    try:
        from src.runtime.timefmt import iso_ts
        payload = {"ts": iso_ts(), "kind": kind}
        for k, v in fields.items():
            payload[k] = _encode(v)
        line = json.dumps(payload, default=str)
        with (worker_dir / "events.jsonl").open("a") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001 — never-raise discipline
        pass


def _walltime_now() -> float:
    return time.monotonic()


# ---------------------------------------------------------------------------
# run_iter — the per-iter K-way eval (Tasks 3 + 4)
# ---------------------------------------------------------------------------


def run_iter(request: dict) -> dict:
    """K-way candidate eval loop. See spec §4 architecture box.

    Imports torch / sol_execbench lazily so the module remains importable
    under Tier-1 (torchless) for unit-testing encoders + request
    assembly. Tests pre-populate the lazy-imported names via
    ``monkeypatch.setattr("src.eval.bench_worker.<name>", ...)``.
    """
    _lazy_imports()

    from src.kernels.kernel import Kernel

    iter_no = int(request["iter_no"])
    worker_dir = Path(request["worker_dir"])
    worker_dir.mkdir(parents=True, exist_ok=True)
    candidates_in: list[dict] = request["candidates"]
    anti_cheat_critical = request.get("anti_cheat_critical_names", [])
    bench_config_raw = request.get("bench_config", {})
    profile_config = request.get("profile_config", {})

    start_walltime = _walltime_now()

    # Rehydrate kernel spec + workloads + input generators + definition.
    # Failures here are fatal to the iter (we can't run anything without
    # them); surface via traceback to stderr and let the CLI wrapper map
    # to non-zero exit.
    #
    # P2 fix (Codex 2026-05-26): the placeholder / no-workload run path
    # has ``workloads=[]`` and a meaningless ``definition_path`` ("" or
    # "."). ``benchmark_kernel`` handles empty workloads via a 100us
    # sentinel return; the worker must NOT try to load a SOL definition
    # or rebuild input generators when there's nothing to bench against
    # — otherwise the placeholder smoke path crashes at load before
    # ``benchmark_kernel`` can return the sentinel, three iters in a
    # row trip ``WorkerProcessUnstable``, and the whole run aborts.
    kernel_spec = _rehydrate_kernel_spec(request["kernel_spec"])
    workloads = _rehydrate_workloads(request["workloads"])
    if workloads:
        # Production: ``_build_input_generators`` resolves the real
        # SOL definition via ``_load_definition`` (which now guards
        # against empty / non-existent paths internally — Codex
        # 2026-05-27 fix #6 — and returns ``None`` rather than
        # crashing on the in-process-bypass test path that passes
        # ``definition_path=""``).
        input_generators = _build_input_generators(request, workloads)
        definition = _load_definition(Path(request["definition_path"] or ""))
    else:
        input_generators = []
        definition = None

    # bench_config is a dict in the request; benchmark_kernel takes an
    # ACTSConfig-like object. We construct a tiny shim so the only
    # fields benchmark_kernel reads (``warmup_runs`` / ``timed_runs``)
    # are addressable.
    bench_config = _bench_config_shim(bench_config_raw)

    response_candidates: list[dict] = []
    aborted_by_channel_A = False
    successful: list[tuple[int, dict, Any, Any, Any]] = []

    for idx, cand_in in enumerate(candidates_in):
        cand_idx = int(cand_in["candidate_idx"])
        cand_kernel = Kernel(
            spec=kernel_spec,
            source_code=cand_in["source_code"],
            triton_kernel_name=cand_in.get("triton_kernel_name", ""),
            dps=bool(cand_in.get("dps", False)),
        )
        try:
            with _run_per_iter_anti_cheat(anti_cheat_critical):
                cand_fn, cand_autotuner = _safe_precompile(cand_kernel, role="Child")
                bench_result = benchmark_kernel(
                    cand_kernel,
                    bench_config,
                    workloads=workloads,
                    input_generators=input_generators,
                    definition=definition,
                    kernel_fn=cand_fn,
                    autotuner=cand_autotuner,
                )
            check_lazy_outputs_after_bench(bench_result.last_outputs)
            bench_result.last_outputs.clear()
        except RewardHackDetected as exc:
            response_candidates.append({
                "candidate_idx": cand_idx,
                "status": "channel_a_tripped",
                "reason": str(exc)[:200],
                "bench_result": None,
                "autotune_winner": {},
                "channel_A_flag": True,
            })
            # NOTE (Codex 2026-05-27): worker does NOT emit
            # ``reward_hack_detected`` / ``coder_failed`` here — parent
            # is the sole emitter for per-candidate events. The worker
            # writing its own per-cand events into ``worker/events.jsonl``
            # + parent merging them + parent ALSO emitting from its
            # response-handling loop produced duplicate entries in
            # canonical events.jsonl (CI counts doubled, postmortem
            # aggregates inflated). The parent reads ``status`` +
            # ``aborted_by_channel_A`` from response.json and emits.
            aborted_by_channel_A = True
            # Mark every remaining (later-indexed) candidate not_run.
            for remaining in candidates_in[idx + 1:]:
                response_candidates.append({
                    "candidate_idx": int(remaining["candidate_idx"]),
                    "status": "not_run",
                    "reason": "channel_A trip on prior cand",
                    "bench_result": None,
                    "autotune_winner": {},
                    "channel_A_flag": False,
                })
            break
        except BenchmarkError as exc:
            response_candidates.append({
                "candidate_idx": cand_idx,
                "status": "bench_failed",
                "reason": str(exc)[:200],
                "bench_result": None,
                "autotune_winner": {},
                "channel_A_flag": False,
            })
            # See note above — parent is sole emitter for coder_failed.
            continue

        # P1 fix (Codex 2026-05-26): partial-bench failure must NOT
        # reach the success path. ``benchmark_kernel`` returns a result
        # with ``workload_errors`` populated when *some* workloads
        # failed but enough survived to skip ``BenchmarkError``. The
        # OLD orchestrator gated those with ``if not
        # is_fully_successful: _accumulate_iter_failure(...); continue``
        # — without that gate, a kernel that crashes on a
        # non-representative workload could become a winner with a
        # silently-wrong score in multi-workload runs.
        is_fully_successful = bool(
            getattr(bench_result, "is_fully_successful", False)
        )
        if not is_fully_successful:
            workload_errors = getattr(bench_result, "workload_errors", {}) or {}
            reason = f"partial bench failure: {workload_errors}"[:200]
            response_candidates.append({
                "candidate_idx": cand_idx,
                "status": "bench_failed",
                "reason": reason,
                "bench_result": None,
                "autotune_winner": {},
                "channel_A_flag": False,
            })
            # Parent emits coder_failed from its response-handling loop.
            continue

        # Success path. Parent emits bench_done for the WINNER (with
        # per_workload_us) via its existing emit at orchestrator.py
        # post-winner block; worker does NOT emit per-candidate
        # bench_done because the worker's emit + parent's winner emit
        # both used the same event kind and inflated the canonical
        # events.jsonl. (Per-candidate latency telemetry is a future
        # opt-in under a distinct event kind if needed.)
        response_candidates.append({
            "candidate_idx": cand_idx,
            "status": "success",
            "reason": "",
            "bench_result": _encode(bench_result),
            "autotune_winner": _encode(
                getattr(bench_result, "autotune_winner_per_workload", {})
            ),
            "channel_A_flag": False,
        })
        successful.append(
            (cand_idx, cand_in, cand_kernel, bench_result, cand_autotuner)
        )

    winner_idx: int | None = None
    winner_profile_dict: dict | None = None

    if successful and not aborted_by_channel_A:
        winner_idx, winner_profile_dict = _run_profile_gauntlet(
            successful=successful,
            worker_dir=worker_dir,
            iter_no=iter_no,
            workloads=workloads,
            input_generators=input_generators,
            profile_config=profile_config,
            request=request,
        )

    return {
        # 2 (2026-05-28): see _build_request_payload for rationale.
        "schema_version": 2,
        "iter_no": iter_no,
        "candidates": response_candidates,
        "winner_idx": winner_idx,
        "winner_profile": winner_profile_dict,
        "aborted_by_channel_A": aborted_by_channel_A,
        "child_walltime_s": _walltime_now() - start_walltime,
    }


def _run_profile_gauntlet(
    *,
    successful: list[tuple[int, dict, Any, Any, Any]],
    worker_dir: Path,
    iter_no: int,
    workloads: list[Any],
    input_generators: list[Callable[[int], tuple]],
    profile_config: dict,
    request: dict,
) -> tuple[int | None, dict | None]:
    """Rank-and-fallback profile pass over bench-successful candidates.

    Mirrors the orchestrator gauntlet (src/search/orchestrator.py:1230-1293):
    rank by SOL score, try profile on top-ranked, on ``ProfilerError``
    drop and try next-ranked, until a winner emerges or all exhaust.

    Returns ``(winner_idx, winner_profile_dict)``. Both ``None`` when no
    candidate clears the gauntlet.

    NCU rep handling: ``profile_kernel`` writes the binary report under
    ``cache_dir`` keyed by source hash. After success we rename / copy
    it to ``worker_dir / "cand_<winner_idx>.ncu-rep"`` so the parent
    knows where to look without scanning.
    """
    t_sol_us = float(profile_config.get("t_sol_us", 0.0)) or 1.0
    baseline_latency_us = float(profile_config.get("baseline_latency_us", 0.0)) or 1.0
    iter_flops = int(profile_config.get("iter_flops", 0))
    iter_nbytes = int(profile_config.get("iter_nbytes", 0))
    repr_idx = int(profile_config.get("repr_workload_idx", 0))
    problem_definition_path = profile_config.get("problem_definition_path")
    if problem_definition_path is not None:
        problem_definition_path = Path(problem_definition_path)
    blob_roots_raw = profile_config.get("blob_roots") or []
    blob_roots = [Path(p) for p in blob_roots_raw] if blob_roots_raw else None

    if workloads and repr_idx < len(workloads):
        repr_workload_axes = workloads[repr_idx].model_dump(mode="json")
        repr_input_generator = input_generators[repr_idx]
    else:
        repr_workload_axes = {}
        repr_input_generator = lambda seed: ()  # noqa: E731

    remaining = list(successful)
    while remaining:
        winner_entry = _select_best_candidate_local(
            remaining,
            t_sol_us=t_sol_us,
            baseline_latency_us=baseline_latency_us,
        )
        w_idx, _w_coder_out, w_kernel, w_bench, _w_autotuner = winner_entry
        remaining = [e for e in remaining if e[0] != w_idx]

        repr_lat_s = _representative_latency_s(w_bench, workloads, repr_idx)
        if repr_lat_s is None:
            _emit(
                worker_dir, "coder_failed",
                iter=iter_no, candidate_idx=w_idx,
                reason="representative workload latency unavailable",
            )
            continue

        # Best-effort capture of the materialized input dtypes so the
        # analytical pct_peak.compute denominator can pick the matching
        # tensor-core peak (see _pick_compute_peak). Failures collapse
        # to an empty list and the fp32_fallback path engages.
        from src.eval.profiler import _collect_input_dtypes
        try:
            _repr_inputs = repr_input_generator(0)
        except Exception:
            _repr_inputs = ()
        _repr_dtypes = _collect_input_dtypes(_repr_inputs)

        try:
            profile = profile_kernel(
                w_kernel,
                repr_workload_axes,
                repr_input_generator,
                hardware_spec=_rehydrate_hardware_spec(request["hardware_spec"]),
                flops=iter_flops,
                nbytes=iter_nbytes,
                latency_s=repr_lat_s,
                cache_dir=worker_dir,
                problem_definition_path=problem_definition_path,
                blob_roots=blob_roots,
                input_dtypes=_repr_dtypes,
            )
        except ProfilerError as exc:
            _emit(
                worker_dir, "coder_failed",
                iter=iter_no, candidate_idx=w_idx,
                reason=f"profile error: {str(exc)[:180]}",
            )
            continue

        # Winner — rename the .ncu-rep into ``cand_<idx>.ncu-rep`` so
        # the parent can find it without scanning. ``ncu_rep_path`` is
        # None on degraded profiles (NCU unavailable / parser failure).
        # On successful rename, REPLACE the profile's stale hashed path
        # with the new on-disk location before encoding — otherwise the
        # parent rehydrates the original (now-moved) path and
        # ``tree_dump.dump_node`` silently fails to copy the artifact
        # into ``tree/node_<id>/ncu.ncu-rep`` (Codex 2026-05-26 fix #3).
        src_rep = getattr(profile, "ncu_rep_path", None)
        if src_rep is not None and Path(src_rep).exists():
            dest = worker_dir / f"cand_{w_idx}.ncu-rep"
            renamed = False
            try:
                Path(src_rep).rename(dest)
                renamed = True
            except OSError:
                # Cross-device or permission issue → fall back to copy.
                try:
                    dest.write_bytes(Path(src_rep).read_bytes())
                    renamed = True
                except OSError:
                    pass
            if renamed:
                # ProfilingResult is frozen — use ``dataclasses.replace``.
                # The ``_FakeProfile`` test stand-in (a non-frozen
                # @dataclass) also satisfies ``replace``.
                try:
                    profile = dataclasses.replace(profile, ncu_rep_path=dest)
                except (TypeError, ValueError):
                    # Future ProfilingResult versions or stand-ins that
                    # don't expose ncu_rep_path: fall through with stale
                    # path rather than crash the iter.
                    pass

        # NOTE (Codex 2026-05-27): worker does NOT emit ``profile_done``
        # for the winner — parent emits ``profile_done`` from its
        # post-winner block (orchestrator.py), so worker emitting here
        # would duplicate the event in canonical events.jsonl after
        # merge. The two profile-gauntlet ``coder_failed`` emits ABOVE
        # (repr_lat_s unavailable, ProfilerError on non-winner) DO stay
        # — parent has no visibility into which non-winner candidates
        # were dropped inside the gauntlet, so those signals are
        # unique to the worker chunk.
        return w_idx, _encode(profile)

    return None, None


def _rehydrate_hardware_spec(data: dict) -> Any:
    """Reconstruct HardwareSpec from the request dict (frozen dataclass)."""
    from src.config import HardwareSpec

    field_names = {f.name for f in dataclasses.fields(HardwareSpec)}
    kwargs = {k: v for k, v in data.items() if k in field_names}
    return HardwareSpec(**kwargs)


def _bench_config_shim(bench_config_raw: dict) -> Any:
    """Build a lightweight attribute-bag exposing ``warmup_runs`` /
    ``timed_runs`` so ``benchmark_kernel`` (which expects ACTSConfig) can
    read them off the shim without us hauling the whole config.
    """
    class _Shim:
        pass

    shim = _Shim()
    shim.warmup_runs = int(
        bench_config_raw.get("warmup_runs")
        or bench_config_raw.get("warmup_iters")
        or 25
    )
    shim.timed_runs = int(
        bench_config_raw.get("timed_runs")
        or bench_config_raw.get("repeat_iters")
        or 100
    )
    return shim


# ---------------------------------------------------------------------------
# CLI wrapper (Task 5)
# ---------------------------------------------------------------------------


def _main_cli(argv: list[str]) -> int:
    """CLI entrypoint. Returns process exit code.

    Reads ``--request`` JSON, calls ``run_iter``, writes ``--response``
    JSON. Any uncaught exception prints a traceback to stderr and
    returns 1; the parent's helper translates that to a
    ``bench_worker_crashed`` event + counter bump.
    """
    parser = argparse.ArgumentParser(description="ACTS bench worker subprocess")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text())
        response = run_iter(request)
        args.response.write_text(json.dumps(response))
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(_main_cli(sys.argv[1:]))
