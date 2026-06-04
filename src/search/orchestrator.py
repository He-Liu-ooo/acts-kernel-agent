"""Top-level search loop orchestrator (deterministic, not LLM).

Coordinates 3 LLM agents + deterministic eval per iteration:
    Planner -> Coder (with tools) -> [eval] -> Reviewer
The Coder's compile/correctness tools handle self-correction internally.
"""

from __future__ import annotations

import asyncio
import logging
import math
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.agents.llm_backend import render_action_menu, render_technique_guidance
from src.agents.reviewer import BranchQuality, _render_review_for_planner
from src.eval.correctness_subprocess import run_correctness_subprocess
from src.memory.experience import ActionRecord
from src.runtime import tree_dump
from src.runtime.events import (
    DeadReason,
    ITER_ADVANCED,
    ITER_DEAD_END,
    ITER_SKIPPED,
    emit,
    finite_or_none,
)
from src.runtime.sdk_trace import trace_span
from src.runtime.usage import AgentLabel

logger = logging.getLogger(__name__)


# Sticky-state CUDA error patterns that one ``torch.cuda.synchronize()``
# can recover from. Anything else (e.g. "operation not implemented for
# CUDA") propagates so genuine bugs aren't silently treated as transient.
_CUDA_STICKY_PATTERNS = (
    "illegal memory access",
    "device-side assert",
    "unspecified launch failure",
    "misaligned address",
    "out of memory",
    "cublas",
    "cudnn",
)


class CUDAContextPoisoned(RuntimeError):
    """Raised when ``torch.cuda.synchronize()`` fails 3 times in a row.

    The orchestrator catches transient CUDA launch errors and recovers
    by syncing the device + marking the branch DEAD_END. After 3
    consecutive sync failures the context is presumed unrecoverable
    (sticky illegal-memory-access, stream poisoning) and the run is
    aborted to avoid burning iterations producing meaningless results.

    Bench-subprocess isolation (2026-05-24) retires this on the bench
    path — the parent no longer runs ``torch.cuda.synchronize()`` for
    bench (the child process owns CUDA touches; sticky errors die with
    the child). The class + ``_CUDA_STICKY_PATTERNS`` set stay in code
    as defensive guards for any remaining parent-side CUDA touches
    (Phase C re-profile, the Coder's SMEM compile-tool host-wrapper,
    and the opt-in in-parent correctness fallback). The Coder's
    correctness check itself moved to a crash-isolated subprocess
    2026-05-31 — see ``src/eval/correctness_subprocess.py``.
    """


class WorkerProcessUnstable(RuntimeError):
    """Raised when consecutive bench-worker subprocess crashes hit the
    configured threshold (``ACTSConfig.worker_crash_threshold``, default 3).

    The orchestrator catches transient worker crashes and recovers by
    treating the iter as failed and bumping a counter. After N
    consecutive crashes the worker is presumed unrecoverable
    (out-of-tree environment bug, broken Python module on PYTHONPATH,
    persistent OOM) and the run is aborted to avoid burning iterations
    producing meaningless results.

    Mirrors ``CUDAContextPoisoned`` (3-strike escalation) but for the
    out-of-process bench worker rather than parent-side CUDA sync.
    See doc/specs/2026-05-24-bench-subprocess-isolation-design.md §5.6.
    """


if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload

    from src.actions.registry import Action
    from src.agents.coder import CoderAgent
    from src.agents.reviewer import ReviewerAgent
    from src.agents.planner import PlannerAgent
    from src.config import ACTSConfig
    from src.eval.benchmark import BenchmarkResult
    from src.eval.roofline import RooflineResult
    from src.eval.types import BottleneckType
    from src.kernels.kernel import Kernel
    from src.memory.producer import Producer
    from src.memory.retriever import MemoryRetriever
    from src.search.tree import SearchTree, TreeNode

# Planner + Reviewer surface a prose profiling summary — when profiling
# is unavailable (pre-first-iter Planner call, or a profile that failed
# to produce any result), we still need a string to pass. The Planner's
# prompt format currently accepts any string; the Reviewer renders the
# real analytical+NCU blocks from the ProfilingResult dataclass when one
# is available (see ReviewerAgent.build_user_prompt). This stub only
# covers the Planner side and the "no profile yet" degenerate path.
_NO_PROFILE_SUMMARY = "[no profiling data available]"


def _safe_precompile(
    kernel: "Kernel",
    *,
    role: str,
) -> tuple[Any | None, Any | None]:
    """Best-effort precompile that returns ``(compiled_fn, autotuner)``,
    degrading to ``(None, None)`` with a WARNING on any failure.

    Used at the orchestrator's baseline (Phase A) and child (Phase B per-
    iter) call sites to retain the host entrypoint for ``kernel_fn=`` and
    the Triton Autotuner for cache introspection (A1 PR 1). Codex review
    2026-05-14 finding #1: ``compile_kernel`` can raise before returning
    a ``CompilationResult`` (read-only ``.acts_cache``, disk full); this
    helper routes those into the lazy-compile fallback in
    ``benchmark_kernel`` rather than aborting ``Orchestrator.run()``.

    *role* labels the warning ("Baseline" / "Child") for postmortem.
    """
    from src.kernels.compiler import compile_kernel

    try:
        result = compile_kernel(kernel)
    except Exception as exc:
        logger.warning(
            "%s pre-compile raised (%s: %s) — falling back to lazy "
            "compile; autotune_winner will be None.",
            role, type(exc).__name__, exc,
        )
        return None, None
    if result.success and result.compiled_fn is not None:
        return result.compiled_fn, result.triton_autotuner
    return None, None


def _record_autotune_winner(
    kernel: "Kernel",
    bench_result: "BenchmarkResult",
) -> None:
    """Copy per-workload autotune winners from BenchmarkResult onto kernel."""
    if bench_result.autotune_winner_per_workload:
        kernel.autotune_winner = bench_result.autotune_winner_per_workload


def _select_best_candidate(
    bench_results: list[tuple[int, Any, "Kernel", "BenchmarkResult", Any]],
    *,
    t_sol_us: float,
    baseline_latency_us: float,
) -> tuple[int, Any, "Kernel", "BenchmarkResult", Any]:
    """Pick the highest-SOL-Score entry from *bench_results*.

    Each entry is ``(candidate_idx, coder_output, child_kernel, bench,
    autotuner)``. Ranks by ``sol_execbench.sol_score.sol_score`` (the
    same formula ``compute_sol_score`` uses downstream — singularity
    and below-SOL > 1.0 behavior preserved); tie-break is lowest
    ``candidate_idx`` for deterministic first-survivor selection.
    """
    from sol_execbench.sol_score import sol_score as _sol_score

    def _sort_key(entry):
        cand_idx, _coder_out, _kernel, bench, _autotuner = entry
        # µs → ms; the unit factor cancels in both regimes so ranking
        # is unchanged relative to a µs-native formula.
        score = _sol_score(
            t_k=bench.median_latency_us / 1000.0,
            t_p=baseline_latency_us / 1000.0,
            t_sol=t_sol_us / 1000.0,
        )
        return (-score, cand_idx)

    return min(bench_results, key=_sort_key)


# ───────────────────── Bench-subprocess serialization helpers (2026-05-24) ─
# Used by the K-way bench dispatch path to (a) marshal `KernelSpec` /
# `HardwareSpec` into JSON for the worker's request.json, and (b) rehydrate
# `BenchmarkResult` / `ProfilingResult` from response.json so the post-winner
# code path stays unchanged. Defensive against extra/missing fields — uses
# `dataclasses.fields(...)` to filter unknown keys so the worker can add
# new fields in future versions without breaking parent-side rehydration.


def _serialize_kernel_spec_for_request(spec: "KernelSpec") -> dict:
    """Dump KernelSpec for cross-process JSON IPC.

    Walks the dataclass via ``dataclasses.asdict`` and coerces enum →
    ``.value`` + ``Path`` → ``str`` recursively so the result is
    JSON-safe (``dataclasses.asdict`` leaves Enum and Path as-is, which
    would crash ``json.dumps``). Mirrors ``src/eval/bench_worker.py::
    _encode`` style. The child rehydrates via ``KernelSpec.from_dict``
    when it needs the full object.
    """
    import enum
    from dataclasses import asdict

    def _coerce(v):
        if isinstance(v, enum.Enum):
            return v.value
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {k: _coerce(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_coerce(x) for x in v]
        return v

    return _coerce(asdict(spec))


def _rebuild_cand_kernel(
    request_candidates: list[dict], cand_idx: int, spec: "KernelSpec"
) -> "Kernel":
    """Reconstruct the `Kernel` for a single candidate index, sharing the
    iter's `KernelSpec` (entrypoint + tolerance + kernel_type all stable
    across the K-way fan-out). Used to attach kernels to failure-summary
    nodes after the worker subprocess returns."""
    from src.kernels.kernel import Kernel

    cand = next(c for c in request_candidates if c["candidate_idx"] == cand_idx)
    return Kernel(
        spec=spec,
        source_code=cand["source_code"],
        triton_kernel_name=cand["triton_kernel_name"],
        dps=cand.get("dps", False),
    )


def _rehydrate_bench_result(data: dict) -> "BenchmarkResult":
    """Reconstruct `BenchmarkResult` from response.json dict. `last_outputs`
    isn't transferred across IPC (GPU tensors) — child already cleared it
    and ran `check_lazy_outputs_after_bench` before returning; downstream
    code treats an empty list as "no outputs to re-check" which is the
    correct semantic post-worker."""
    from dataclasses import fields
    from src.eval.benchmark import BenchmarkResult

    known = {f.name for f in fields(BenchmarkResult)}
    kwargs = {k: v for k, v in data.items() if k in known}
    kwargs.setdefault("last_outputs", [])
    return BenchmarkResult(**kwargs)


def _rehydrate_profiling_result(data: dict | None) -> "ProfilingResult | None":
    """Reconstruct ``ProfilingResult`` (incl. nested ``AnalyticalMetrics``
    + ``NCUMetrics``) from response.json dict. Returns ``None`` when
    ``data`` is ``None`` (no winner profile — gauntlet exhausted in
    child or no winner).

    Defensive forward-compat (Codex 2026-05-27 fix #5): the ``_filtered``
    helper drops UNKNOWN keys (so future workers can add fields without
    breaking older parents), AND every dataclass construction is wrapped
    in try/except so MISSING required keys degrade to ``None`` for the
    nested block instead of crashing the entire orchestrator. Without
    the try/except, a worker that omits e.g. ``warp_stall_runner_up_pct``
    from NCUMetrics would raise TypeError past the orchestrator's
    ``except WorkerCrashed`` and abort the whole run on a survivable
    schema mismatch.
    """
    if data is None:
        return None
    from dataclasses import fields
    from src.eval.profiler import AnalyticalMetrics, NCUMetrics, ProfilingResult

    def _filtered(cls, src: dict) -> dict:
        known = {f.name for f in fields(cls)}
        return {k: v for k, v in src.items() if k in known}

    ad = data.get("analytical")
    if ad:
        try:
            analytical = AnalyticalMetrics(**_filtered(AnalyticalMetrics, ad))
        except TypeError as exc:
            logger.warning(
                "AnalyticalMetrics rehydration failed (missing required "
                "field?): %s — treating as analytical=None", exc,
            )
            analytical = None
    else:
        analytical = None
    nd = data.get("ncu")
    if nd:
        try:
            ncu = NCUMetrics(**_filtered(NCUMetrics, nd))
        except TypeError as exc:
            logger.warning(
                "NCUMetrics rehydration failed (missing required field?): "
                "%s — treating as ncu=None", exc,
            )
            ncu = None
    else:
        ncu = None
    rep_path = data.get("ncu_rep_path")
    pr_kwargs = _filtered(ProfilingResult, {
        "analytical": analytical,
        "ncu": ncu,
        "raw_metrics": data.get("raw_metrics", {}) or {},
        "metric_groups": data.get("metric_groups", {}) or {},
        "degraded_reason": data.get("degraded_reason"),
        "ncu_rep_path": Path(rep_path) if rep_path else None,
    })
    try:
        return ProfilingResult(**pr_kwargs)
    except TypeError as exc:
        logger.warning(
            "ProfilingResult rehydration failed: %s — treating winner "
            "profile as None", exc,
        )
        return None


class TerminationReason(str, Enum):
    """Why the search loop exited. str-subclass so legacy string comparisons
    still work during the transition (e.g. existing report/doc consumers)."""

    SOL_TARGET = "sol_target"
    PLATEAU = "plateau"
    BUDGET = "budget"
    ALL_DEAD_END = "all_dead_end"


@dataclass
class SearchResult:
    """Final result of the search process.

    ``tree`` is carried forward so Phase C (``pipeline/report.py``) can
    reconstruct the root-to-best path for ``technique_trace`` without the
    orchestrator having to denormalize every path-derived view upfront.

    ``run_bottleneck`` is the once-per-run classification from
    ``classify_run`` — invariant per ``(problem, representative_workload,
    hardware)`` so downstream consumers (Phase C report, memory store)
    share a single source of truth.
    """

    best_node: TreeNode
    total_iterations: int
    termination_reason: TerminationReason
    tree: SearchTree
    run_bottleneck: BottleneckType | None = None
    reference_baseline_latency_us: float | None = None
    baseline_root_latency_us: float | None = None


def _apply_baseline_feedback_to_root(root, feedback) -> None:
    """Attach baseline Reviewer feedback to root, with DEAD_END clamping.

    Always sets ``root.last_review = feedback`` so the iter-1 Planner
    receives the diagnosis / suggestions / conditional_assessment as
    prompt context. Propagates ``branch_quality`` to ``root.branch_quality``
    only when it's NOT ``DEAD_END`` — a baseline review has no parent
    delta to compare against, so an LLM hallucinating ``DEAD_END`` would
    otherwise empty the frontier and exit the run as ``ALL_DEAD_END``
    before iter 1 can plan. (Codex adversarial-review finding [high],
    2026-05-10.)
    """
    root.last_review = feedback
    if feedback.branch_quality != BranchQuality.DEAD_END:
        root.branch_quality = feedback.branch_quality
    else:
        logger.warning(
            "Baseline review returned DEAD_END (no parent delta to ground it) — "
            "ignoring for tree state; root stays expandable. last_review "
            "diagnosis still propagates to iter 1's Planner.",
        )


def _resolve_blob_roots(safetensors_blob_roots, problem_definition_path):
    """Resolve the ``blob_roots`` argument for ``profile_kernel``.

    Config override wins. When unset, fall back to the problem directory
    so safetensors-backed workloads resolve their blobs against the same
    root the in-process generator used. ``problem_definition_path`` is
    None on the placeholder path (no SOL problem dir to fall back to);
    pass None through so the profiler driver omits the field.
    """
    if problem_definition_path is None:
        return safetensors_blob_roots
    return safetensors_blob_roots or [problem_definition_path.parent]


def _representative_latency_s(bench, workloads, repr_idx: int) -> float | None:
    """Return the representative workload's latency in seconds, or ``None``
    when that workload failed on this run.

    ``bench.per_workload_latency_us`` is keyed by ``Workload.uuid``; when
    ``workloads`` is ``None`` we've hit the placeholder path (no SOL
    workloads) and fall back to the aggregate median.
    """
    if not workloads:
        # Placeholder path — benchmark returned a synthetic 100us sentinel;
        # use it so the analytical pct-of-peak metrics stay meaningful.
        return bench.median_latency_us / 1e6
    if repr_idx >= len(workloads):
        return None
    uuid = workloads[repr_idx].uuid
    latency_us = bench.per_workload_latency_us.get(uuid)
    if latency_us is None or not math.isfinite(latency_us):
        return None
    return latency_us / 1e6


def _render_profiling_for_planner(profiling, roofline=None) -> str:
    """Lightweight summary for the Planner's prompt. The Reviewer owns the
    full analytical + NCU rendering via ``reviewer.render_profiling_summary``;
    the Planner only needs a couple of numbers to reason about the *next*
    technique to try. Bottleneck classification is hoisted to a dedicated
    "Run context" section by the Planner prompt, so it's not repeated here.

    ``arithmetic_intensity`` is sourced from ``roofline`` (run-level
    invariant in MACs/byte). Omitted when ``roofline`` is None.

    Omits ``pct_peak_*`` when ``profiling.analytical`` is None (nbytes
    couldn't be derived) — NCU + roofline lines still ride through.
    """
    lines: list[str] = []
    if profiling.has_analytical:
        a = profiling.analytical
        lines.extend([
            f"pct_peak_compute={a.pct_peak_compute * 100:.1f}% [{a.compute_peak_dtype}]",
            f"pct_peak_bandwidth={a.pct_peak_bandwidth * 100:.1f}%",
        ])
    if roofline is not None:
        lines.append(f"arithmetic_intensity={roofline.arithmetic_intensity:.3f}")
    if profiling.has_ncu:
        n = profiling.ncu
        lines.append(f"sm_occupancy={n.sm_occupancy_pct:.1f}%")
        lines.append(f"l2_hit_rate={n.l2_hit_rate_pct:.1f}%")
        lines.append(f"dominant_stall={n.warp_stall_dominant}")
    elif profiling.degraded:
        lines.append(f"[DEGRADED: {profiling.degraded_reason or 'unknown'}]")
    return ", ".join(lines)


def _per_workload_us(bench) -> list[float | None]:
    """Sanitize a benchmark's per-workload latency dict for the event
    stream. Returns ``[]`` when ``bench`` is ``None``; maps the
    ``math.inf`` launch-failure sentinel to ``None`` so the payload is
    RFC-8259 JSON."""
    if bench is None:
        return []
    return [finite_or_none(v) for v in bench.per_workload_latency_us.values()]


def _emit_dead_end(iter_no: int, reason: DeadReason, *, detail: str | None = None) -> None:
    """Fire ``branch_dead_end`` + ``iter_end`` for a DEAD_END iteration.

    *reason* is a ``DeadReason`` enum member; its string value goes on
    the wire so telemetry consumers can pivot on the stable code rather
    than parse a free-form string. Any dynamic context (CUDA error
    message, exception text) goes into *detail* and is emitted as a
    separate payload field.
    """
    payload: dict[str, Any] = {"reason": reason.value}
    if detail is not None:
        payload["detail"] = detail
    emit("branch_dead_end", iter=iter_no, **payload)
    emit("iter_end", iter=iter_no, outcome=ITER_DEAD_END)


def _is_reviewer_rejected(node: Any) -> bool:
    """True iff *node* is a DEAD_END the Reviewer rejected (not beam-pruned).

    ``branch_quality == DEAD_END`` collapses several distinct causes (see
    ``DeadReason``). The opt-mem write gate must skip ONLY Reviewer-judged
    rejections — those are quality-shaped (reward-hack suspicion, over-
    optimization, style regression), not lessons worth distilling.

    A ``DeadReason.BEAM_PRUNED`` node is explicitly NOT reviewer-rejected:
    it ran fine and may well have improved its parent — it just lost the
    beam competition. Letting it through to ``Producer.consider`` is safe
    because the Producer's δ-improvement gate is the proper arbiter of
    whether the edge lands in the store.
    """
    return (
        node.branch_quality == BranchQuality.DEAD_END
        and node.dead_reason != DeadReason.BEAM_PRUNED
    )


def detect_plateau(
    score_history: list[float],
    window: int,
    delta: float,
) -> bool:
    """Return True if the best score hasn't improved beyond *delta*
    over the last *window* entries in *score_history*."""
    if len(score_history) < window:
        return False
    recent = score_history[-window:]
    return max(recent) - min(recent) <= delta + 1e-9


def _accumulate_iter_failure(
    iter_failures: "list",
    iter_failure_kernels: "list",
    *,
    iter_no: int,
    candidate_idx: int,
    kernel,
    reason: str,
) -> None:
    """Append one failed K-way candidate to the iter-level accumulators
    and emit the per-candidate ``coder_failed`` event.

    Replaces the legacy per-candidate persistence path (which mutated the
    tree immediately at each of the 5 failure sites). Under failure-node
    collapse, the tree-attach + per-node dump + ``failure_summary_added``
    event happen once per iter via ``_persist_iter_failure_summary``
    after the per-candidate bench loop has finished pushing every
    failure into the accumulators.
    """
    from src.search.tree import FailureDetail

    iter_failures.append(FailureDetail(
        candidate_idx=candidate_idx,
        reason=reason,
        has_kernel_source=(kernel is not None),
    ))
    iter_failure_kernels.append((candidate_idx, kernel))
    emit(
        "coder_failed",
        iter=iter_no,
        candidate_idx=candidate_idx,
        reason=reason,
    )


def _persist_iter_failure_summary(
    tree: "SearchTree",
    parent: "TreeNode",
    *,
    plan,
    iter_no: int,
    iter_failures: "list",
    iter_failure_kernels: "list",
) -> None:
    """End-of-iter: attach one failure-summary node + dump per-candidate
    artifacts + emit ``failure_summary_added``. No-op when
    ``iter_failures`` is empty (e.g. all-K-succeed iter; Planner-failed
    iter that never reached the gather).

    Profile-layer failures intentionally do not push into
    ``iter_failures`` — they emit ``coder_failed`` only and stay out of
    the summary, per the existing "downstream-of-truth, don't dilute the
    FAILED block" rationale.
    """
    if not iter_failures:
        return
    from src.runtime import tree_dump

    fn = tree.add_failure_summary(
        parent.id,
        action_applied=plan.technique,
        action_params=plan.params,
        iter_no=iter_no,
        failure_details=iter_failures,
    )
    tree_dump.dump_failure_summary_node(
        fn, iter_failure_kernels, iter_no=iter_no,
    )
    emit(
        "failure_summary_added",
        iter=iter_no,
        node_id=str(fn.id),
        parent_id=str(parent.id),
        action=plan.technique,
        params=plan.params,
        candidate_count=len(iter_failures),
    )


def _select_technique_guidance(applicable: list["Action"], technique: str) -> str:
    """Render the chosen technique's Coder guidance, or "" when the technique
    id is not in the applicable set (e.g. the no-model placeholder plan, whose
    technique id is not a registered action). Keeps the Coder render fail-soft."""
    selected = next((a for a in applicable if a.id == technique), None)
    return render_technique_guidance(selected) if selected is not None else ""


def _render_and_emit_sibling_context(
    tree: "SearchTree",
    parent: "TreeNode",
    *,
    iter_no: int,
    consumer: str,
    exclude_id: int | None = None,
    failure_cap: int = 8,
) -> tuple[str, list[tuple[str, int]]]:
    """Render the sibling section and emit ``sibling_context_rendered``.

    Returns ``(sibling_context, regressed)``. ``regressed`` is returned
    so the Reviewer-side caller can reuse it for the
    ``repeated_pathway_dead_end`` check without a second tree walk.

    ``consumer`` is forwarded to ``tree.render_siblings`` so the
    Planner sees FAILED lines for prior failed children (per
    doc/specs/2026-05-17-failure-node-retention-design.md) while the
    Reviewer keeps the success-only render. ``failure_cap`` (defaults
    to 8) limits the Planner-side FAILED-line count.
    """
    sibling_context = tree.render_siblings(
        parent.id, exclude_id=exclude_id,
        consumer=consumer, failure_cap=failure_cap,
    )
    regressed = tree.regressed_sibling_actions(parent.id, exclude_id=exclude_id)
    if sibling_context:
        sibling_count = len(parent.children_ids) - (1 if exclude_id is not None else 0)
        emit(
            "sibling_context_rendered",
            iter=iter_no,
            parent_node_id=str(parent.id),
            sibling_count=sibling_count,
            regressed_actions=[a for a, _ in regressed],
            consumer=consumer,
        )
    return sibling_context, regressed


class Orchestrator:
    """Deterministic orchestrator managing the tree search loop.

    Per iteration:
        1. Select node (epsilon-greedy)
        2. Retrieve past experiences from optimization memory
        3. Planner: profiling + memory + feedback -> plan
        4. Coder: plan + kernel -> optimized kernel (self-corrects via tools)
        5. Deterministic eval: benchmark + NCU + SOL score
        6. Reviewer: eval results -> structured feedback + branch_quality
        7. Tree update: add node, score, beam prune
        8. Memory update: store experience

    Termination: SOL target reached, plateau detected, budget exhausted,
    or all frontier nodes marked dead_end.
    """

    def __init__(
        self,
        config: ACTSConfig,
        planner: PlannerAgent,
        coder: CoderAgent,
        reviewer: ReviewerAgent,
        retriever: MemoryRetriever,
        producer: "Producer | None" = None,
    ) -> None:
        from src.actions.registry import build_default_registry

        self._config = config
        self._planner = planner
        self._coder = coder
        self._reviewer = reviewer
        self._retriever = retriever
        self._producer = producer
        self._action_registry = build_default_registry()
        self._tree: SearchTree | None = None
        # Cached SOL ``Environment`` for ``trace_emitted``. Built lazily
        # on first use so test paths that mock CUDA don't pay the
        # ``env_snapshot`` cost. Reset per ``run()`` invocation.
        self._environment = None
        # Bench-subprocess isolation (2026-05-24). Counts consecutive
        # ``WorkerCrashed`` outcomes from the per-iter bench worker;
        # reaches ``ACTSConfig.worker_crash_threshold`` (default 3) →
        # raise ``WorkerProcessUnstable`` and abort the run. Mirrors
        # ``consecutive_cuda_errors`` shape (counter + escalation).
        # Reset to 0 on every clean worker exit.
        self.consecutive_worker_crashes = 0

    async def _flush_opt_mem(self, root, tree, bottleneck=None) -> None:
        """Best-effort opt-mem finalize + flush. Called before every clean
        run() exit. Never raises — opt-mem is best-effort by design and a
        flush hiccup must not turn a successful search into a failure.
        See doc/specs/2026-05-24-optimization-memory-design.md §8.
        """
        if self._producer is None:
            return
        try:
            await self._producer.finalize(root, tree.best_node(), bottleneck=bottleneck)
            flushed = await self._producer.flush()
            if flushed:
                logger.info("opt-mem: flushed %d rows", flushed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("opt-mem flush failed (ignored): %s", exc)

    def _kill_branch(
        self,
        child: TreeNode,
        parent: TreeNode,
        iter_no: int,
        *,
        reason: DeadReason,
        detail: str | None = None,
        bumps_agent_failures: bool = False,
    ) -> None:
        """Mark a child DEAD_END, prune the beam, decay epsilon, and emit
        the standard ``branch_dead_end`` + ``iter_end`` pair.

        Records *reason* on ``child.dead_reason`` so downstream consumers
        (``best_node``, memory distillation, tree viz) can distinguish
        infrastructure-error kills from other DEAD_END causes —
        ``branch_quality`` alone collapses every cause into a single flag.

        *bumps_agent_failures* is True for agent-output failures
        (Coder/Planner produced a buggy/cheating kernel — accountable to
        the parent's quarantine counter); False for infra failures (CUDA
        error, profiler failure, partial bench failure) where the agent
        isn't accountable. Caller is responsible for the trailing
        ``epsilon = max(...)`` decay update because the local ``epsilon``
        and ``decay`` live in ``run()``'s frame; this helper handles the
        per-site DEAD_END side-effects only.
        """
        from src.search.beam import beam_prune

        child.mark_dead(reason)
        if bumps_agent_failures:
            parent.consecutive_agent_failures += 1
        beam_prune(
            self._tree,
            self._config.beam_width,
            enable_diversity=self._config.beam_diversity,
        )
        # Persist the dead-end node to <run_dir>/tree/node_<id>/ with the
        # kill reason in meta.json. Centralized here (rather than at each
        # of the six call sites) so the dump can't drift out of sync with
        # _kill_branch's other side-effects. No profiling on the dead-end
        # path → ncu_rep_src is None.
        # ``failure_reason`` is omitted — the categorical cause lives on
        # ``child.dead_reason`` (set above at line 318) and surfaces in
        # meta.json via ``_late_bound_fields``. Only the kill-site prose
        # (``detail``) needs to flow through ``dump_node``.
        tree_dump.dump_node(
            child,
            iter_no=iter_no,
            ncu_rep_src=None,
            failure_detail=detail,
        )
        _emit_dead_end(iter_no, reason, detail=detail)

    async def run(
        self,
        baseline: Kernel,
        workloads: list[Workload] | None = None,
        roofline: RooflineResult | None = None,
        *,
        reference_fn: Callable[..., Any] | None = None,
        input_generators: list[Callable[[int], tuple]] | None = None,
        problem_definition_path: Path | None = None,
        definition: Definition | None = None,
        run_dir: Path | None = None,
        ncu_cache_dir: Path | None = None,
        reference_baseline_latency_us: float | None = None,
    ) -> SearchResult:
        """Execute the full search loop from baseline to best kernel.

        *workloads*: representative subset for iterative benchmarking
        (SOL-ExecBench mode).  When ``None``, benchmarking uses
        ``kernel.spec.input_shapes`` (legacy mode).

        *roofline*: pre-computed SOLAR result (T_SOL + bottleneck).
        When ``None``, falls back to built-in roofline from
        ``KernelSpec.flop_count`` / ``KernelSpec.memory_bytes``.

        *reference_fn* / *input_generators*: the PyTorch oracle and one
        seed→args generator per selected workload. Threaded verbatim into
        the Coder's correctness tool so every iteration verifies against
        the full coverage set. Required when the Coder is LLM-driven; may
        be ``None`` / empty in the placeholder path where ``implement()``
        returns the source unchanged.

        *problem_definition_path*: SOL-ExecBench ``definition.json`` the
        profiler subprocess driver re-loads to rebuild the (unpicklable)
        input generator. When ``None`` the driver falls back to
        ``module.make_inputs`` or ``spec['args']`` — only safe for Tier 2
        self-contained kernels, not real Coder outputs.

        *definition*: the parsed SOL ``Definition`` — used each iteration
        to derive per-workload ``(flops, nbytes)`` for the analytical
        profiler (``KernelSpec.flop_count`` / ``memory_bytes`` are
        intentionally left at zero for SOL problems). ``None`` falls back
        to ``baseline.spec.flop_count`` / ``memory_bytes`` — correct for
        the placeholder starter kernels, which populate those fields
        directly.
        """
        from src.agents.coder import ImplementationError
        from src.agents.planner import PlanningError
        from src.agents.reviewer import BranchQuality
        from src.eval.anti_cheat import (
            check_lazy_outputs_after_bench,
            per_iter_anti_cheat,
        )
        from src.eval.benchmark import BenchmarkError, benchmark_kernel
        from src.eval.profiler import (
            ProfilerError,
            find_jit_name_in_entrypoint,
            profile_kernel,
        )
        from src.eval.roofline import classify_run, compute_roofline
        from src.eval.scorer import compute_sol_score
        from src.kernels.kernel import Kernel, KernelSpec
        from src.search.beam import beam_prune, select_next
        from src.search.tree import SearchTree
        from sol_execbench.core.bench.reward_hack import RewardHackDetected

        # CUDA sticky-state recovery counter. Each transient CUDA error
        # increments this; 3 consecutive failures raise CUDAContextPoisoned
        # to fail the run rather than burn iterations on a poisoned device.
        consecutive_cuda_errors = 0

        # Fail-fast: zeroed HardwareSpec (the ``detect_hardware()`` fallback)
        # would make every analytical profile raise ProfilerError and silently
        # DEAD_END every branch. This is a global config error, not a branch
        # event.
        if (
            self._config.hardware.peak_flops_fp32 <= 0
            or self._config.hardware.peak_memory_bandwidth_gb_s <= 0
        ):
            raise ValueError(
                "HardwareSpec has zero peaks "
                f"(peak_flops_fp32={self._config.hardware.peak_flops_fp32}, "
                f"peak_memory_bandwidth_gb_s={self._config.hardware.peak_memory_bandwidth_gb_s}) "
                "— load a populated spec via SOLAR arch YAML or implement detect_hardware"
            )

        tree = SearchTree()
        self._tree = tree

        # Phase A: baseline evaluation. Baseline is the SOL-score
        # denominator, so any partial-workload failure makes every
        # downstream child score meaningless — fail closed symmetric
        # with the majority-failure BenchmarkError path.
        root = tree.add_root(baseline)
        _baseline_fn, _baseline_autotuner = _safe_precompile(baseline, role="Baseline")

        baseline_bench = benchmark_kernel(
            baseline,
            self._config,
            workloads=workloads,
            input_generators=input_generators,
            definition=definition,
            kernel_fn=_baseline_fn,
            autotuner=_baseline_autotuner,
        )
        if not baseline_bench.is_fully_successful:
            raise BenchmarkError(
                f"baseline benchmark had partial-workload failures "
                f"(errors={baseline_bench.workload_errors}); "
                f"SOL scoring requires a complete baseline measurement"
            )

        # T_b for SOL scoring: the external reference's measured median when
        # provided (Option C), else the Triton root's own median (default).
        scoring_baseline_latency_us = (
            reference_baseline_latency_us
            if reference_baseline_latency_us is not None
            else baseline_bench.median_latency_us
        )

        # A1 PR 1/B: benchmark_kernel captures per-workload autotune
        # winners by diffing Triton's cache around each workload burn-in.
        # Skipped on the lazy-compile fallback path, on placeholder
        # baselines without an autotune decorator, and on legacy starters.
        if _baseline_autotuner is not None:
            _record_autotune_winner(baseline, baseline_bench)
            emit(
                "autotune_burn_in_done",
                iter=root.iter_no,
                workload_count=len(workloads or []),
                winner_count=len(baseline.autotune_winner),
            )

        emit(
            "baseline_ready",
            latency_us=baseline_bench.median_latency_us,
            per_workload_latency_us=_per_workload_us(baseline_bench),
        )

        # Children get per_workload_latency_us from their bench at L784;
        # root needs the same so Phase C re-profile can use it when no
        # child beats baseline (root becomes ``best_node``). Without this,
        # report.py degrades every winner workload as
        # ``per_workload_latency_missing``.
        root.per_workload_latency_us = baseline_bench.per_workload_latency_us
        # Opt-mem Producer reads node.runtime_ms to compute parent → child
        # speedup ratios. Stash the baseline median latency on root so the
        # G3 "baseline → best-of-run" finalize call works regardless of
        # whether iter 1 expanded root or some other parent.
        root.runtime_ms = baseline_bench.median_latency_us / 1000.0

        # Persist the baseline root to <run_dir>/tree/node_0/. Mirrors the
        # child-side dump_node call (advance path) and the dead-end dump
        # in _kill_branch — without this, finalize_tree would index the
        # root in tree/index.json while the per-node dir was missing on
        # disk (same half-truth motivating the dead-end dump fix). No
        # profiling for the baseline → ncu_rep_src=None.
        tree_dump.dump_node(root, iter_no=root.iter_no, ncu_rep_src=None)

        if roofline is None:
            roofline = compute_roofline(baseline.spec, self._config.hardware)

        # Bottleneck is invariant per (problem, representative workload,
        # hardware) so we classify once here and thread ``run_bottleneck``
        # through the retriever, planner, reviewer, and SearchResult —
        # no per-iteration re-classification.
        run_bottleneck = classify_run(
            hardware=self._config.hardware,
            roofline=roofline,
            baseline_spec=baseline.spec,
        )

        applicable_actions = self._action_registry.list_applicable(
            baseline.spec.kernel_type.value,
            hardware=self._config.hardware,
        )
        available_actions = [a.id for a in applicable_actions]
        action_menu = render_action_menu(applicable_actions)

        root.score = compute_sol_score(
            scoring_baseline_latency_us,
            baseline_bench.median_latency_us,
            roofline.t_sol_us,
        )

        # Phase B: search loop
        epsilon = self._config.epsilon_start
        decay = (self._config.epsilon_start - self._config.epsilon_end) / max(self._config.max_depth, 1)
        best_scores: list[float] = []

        # Run-invariant; stringified once so per-iter ``profile_done`` emits
        # don't re-format the enum each time.
        run_bottleneck_str = run_bottleneck.value if run_bottleneck is not None else ""
        # Tracks the best scored-node's sol_score so ``score_computed.is_new_best``
        # can be computed without an O(N) ``tree.best_node()`` walk every iter.
        running_best_score = root.score.sol_score

        # Representative workload index for per-iteration profiling (spec
        # §3.3). Middle of the selected-workload list so large/small-axis
        # outliers don't dominate the profile; falls back to 0 when
        # len(workloads) < 2 or workloads is None.
        repr_idx = (len(workloads) // 2) if workloads else 0

        # Workload shapes for the ``## Run context`` block — derived once
        # from ``Workload.axes.values()`` (SOL's canonical dict-of-int
        # axes). Threaded into every Planner / Reviewer / Coder call so
        # the agents reason against the iter-time shape envelope instead
        # of a single representative. None when the placeholder path runs
        # (no workloads) — render_run_context omits the line.
        workload_shapes_for_prompt: list[tuple[int, ...]] | None = (
            [tuple(w.axes.values()) for w in workloads] if workloads else None
        )

        # Per-iteration (flops, nbytes) are invariant across the run —
        # derived from (definition, representative workload) or from the
        # baseline spec in the placeholder path — so hoist them out of
        # the loop instead of recomputing every iteration.
        if definition is not None and workloads:
            from src.benchmark.roofline_shapes import compute_roofline_inputs
            # roofline= so SOLAR counts outrank shape formulas — shape
            # formulas bail on every op_type=None problem (every L1 case).
            iter_flops, iter_nbytes = compute_roofline_inputs(
                definition, workloads[repr_idx], roofline=roofline,
            )
            # The profiler driver receives the workload as a JSON-serializable
            # dict (mode="json" so SOL's pydantic input variants flatten to
            # plain dicts via the discriminated-union encoder).
            repr_workload_axes = workloads[repr_idx].model_dump(mode="json")
        else:
            iter_flops = baseline.spec.flop_count
            iter_nbytes = baseline.spec.memory_bytes
            repr_workload_axes = {}
        repr_input_generator = (
            input_generators[repr_idx] if input_generators else (lambda seed: ())
        )

        # Baseline review pass (spec 2026-05-10): profile + review root so
        # the Planner expanding root in iter 1 receives a meaningful
        # parent.last_review instead of None. Skipped on the placeholder
        # path (no input_generators / no workloads). All errors are
        # swallowed: a baseline-pass failure must not abort the run —
        # the worst case is iter 1 sees reviewer_feedback=None, which
        # is exactly the pre-feature behavior.
        baseline_repr_latency_s = _representative_latency_s(
            baseline_bench, workloads, repr_idx
        )
        # No (flops, nbytes) gate — profile_kernel handles nbytes=0
        # (analytical=None, NCU still runs). Baseline-pass is best-effort.
        if (
            input_generators and workloads
            and baseline_repr_latency_s is not None
            and math.isfinite(baseline_repr_latency_s)
        ):
            # Capture input dtypes for pct_peak.compute denominator
            # selection. Errors collapse to []; analytical falls back
            # to fp32_fallback with calibration_warning=True.
            from src.eval.profiler import _collect_input_dtypes
            try:
                _repr_inputs = repr_input_generator(0)
            except Exception as exc:
                logger.warning(
                    "baseline dtype-gather: input generator raised %s: %s — "
                    "baseline pct_peak.compute will use fp32_fallback",
                    type(exc).__name__, exc,
                )
                _repr_inputs = ()
            _repr_dtypes = _collect_input_dtypes(_repr_inputs)
            # Only the dtype strings are needed downstream; the
            # representative CUDA tensors would otherwise stay referenced
            # for the whole run(). Drop them so the device memory frees now.
            del _repr_inputs

            try:
                root.profiling = profile_kernel(
                    baseline,
                    repr_workload_axes,
                    repr_input_generator,
                    hardware_spec=self._config.hardware,
                    flops=iter_flops,
                    nbytes=iter_nbytes,
                    latency_s=baseline_repr_latency_s,
                    problem_definition_path=problem_definition_path,
                    blob_roots=_resolve_blob_roots(
                        self._config.safetensors_blob_roots,
                        problem_definition_path,
                    ),
                    input_dtypes=_repr_dtypes,
                )
            except ProfilerError as exc:
                logger.warning(
                    "Baseline profile failed (%s) — skipping baseline review; "
                    "iter 1 Planner will see reviewer_feedback=None.",
                    exc,
                )

        if root.profiling is not None:
            baseline_feedback = None
            try:
                with trace_span("acts_iter", iter_no=0, agent=AgentLabel.REVIEWER):
                    baseline_feedback = await self._reviewer.review(
                        kernel_source=baseline.render_condensed_source(
                            representative_workload_uuid=(
                                workloads[0].uuid if workloads else None
                            ),
                        ),
                        profiling_summary="",
                        sol_score=root.score.sol_score,
                        headroom_pct=(1.0 - root.score.sol_score) * 100,
                        bottleneck=run_bottleneck,
                        profiling=root.profiling,
                        roofline=roofline,
                        prev_sol_score=None,
                        iter_idx=0,
                        max_turns=self._config.reviewer_max_turns,
                        hardware=self._config.hardware,
                        workload_shapes=workload_shapes_for_prompt,
                    )
            except Exception as exc:  # noqa: BLE001 — baseline review must not abort
                logger.warning(
                    "Baseline review failed (%s) — root.last_review stays None; "
                    "iter 1 Planner will see reviewer_feedback=None.",
                    exc,
                )
            if baseline_feedback is not None:
                _apply_baseline_feedback_to_root(root, baseline_feedback)
                emit(
                    "reviewer_feedback",
                    iter=0,
                    verdict=baseline_feedback.branch_quality.value,
                    suggestion_short=baseline_feedback.outcome[:120],
                    degraded=baseline_feedback.degraded,
                )

        # ─── Bench-dispatch invariants — computed ONCE per run() ───
        # Codex 2026-05-27 fix #7: previously `_effective_run_dir`,
        # `_effective_ncu_cache`, and `_subprocess_enabled` were
        # computed inside the iter loop, so the `run_dir is None`
        # fallback branch allocated a fresh `tempfile.mkdtemp(...)` PER
        # iter and leaked them across long test runs (also broke
        # cross-iter NCU cache sharing). Compute once + best-effort
        # cleanup via the run()'s outer try/finally below.
        _subprocess_enabled = (
            self._config.bench_use_subprocess and run_dir is not None
        )
        # Codex 2026-05-27 fix #3: silent downgrade is an operator
        # safety regression — the cfg toggle for CUDA-isolation becomes
        # a no-op when RunContext fell back to run_dir=None (disk-full,
        # permission denied). Log loudly so operators see it.
        if self._config.bench_use_subprocess and run_dir is None:
            logger.warning(
                "ACTSConfig.bench_use_subprocess=True but run_dir is None — "
                "silently falling back to in-process bypass. CUDA-context "
                "isolation (the entire point of bench_use_subprocess) is "
                "DISABLED for this run. Verify RunContext.create() didn't "
                "OSError into its null-RunContext fallback."
            )
        if run_dir is not None:
            _effective_run_dir = run_dir
        else:
            _effective_run_dir = Path(
                tempfile.mkdtemp(prefix="acts_inproc_run_"),
            )
            # Best-effort cleanup at interpreter shutdown — keeps /tmp
            # from accumulating one acts_inproc_run_* per test session.
            # Fires even on uncaught exception inside the for-iter loop;
            # ignore_errors so a half-deleted dir doesn't mask the real
            # failure.
            import atexit
            import shutil as _shutil
            atexit.register(
                _shutil.rmtree, _effective_run_dir, ignore_errors=True,
            )
        _effective_ncu_cache = (
            ncu_cache_dir if ncu_cache_dir is not None
            else _effective_run_dir / "ncu_cache"
        )

        for iteration in range(self._config.max_depth):
            iter_no = iteration + 1
            frontier = tree.frontier()
            if not frontier:
                await self._flush_opt_mem(root, tree, run_bottleneck)
                return SearchResult(
                    tree.best_node(),
                    iteration,
                    TerminationReason.ALL_DEAD_END,
                    tree,
                    run_bottleneck=run_bottleneck,
                    reference_baseline_latency_us=reference_baseline_latency_us,
                    baseline_root_latency_us=baseline_bench.median_latency_us,
                )

            parent = select_next(tree, epsilon)

            parent_score = parent.score.sol_score if parent.score is not None else 0.0
            emit(
                "iter_start",
                iter=iter_no,
                parent_node_id=str(parent.id),
                parent_score=parent_score,
                selected_by="epsilon_greedy",
            )

            # Retriever surfaces distilled opt-mem lessons for the Planner.
            # Filtered by kernel_type, hardware-preferred, speedup-weighted
            # random sample. The run-level bottleneck is still threaded
            # through Planner / Reviewer for in-prompt reasoning, but the
            # retriever no longer keys on it (lessons are kernel-shaped,
            # not symptom-shaped). See
            # doc/specs/2026-05-24-optimization-memory-design.md §10.
            experiences = self._retriever.sample(
                baseline.spec.kernel_type.value,
                self._config.hardware.name,
            )

            # Root-to-parent trajectory — consumed by the Planner so it can
            # reason about which actions have already been tried on this branch.
            parent_profiling_summary = (
                _render_profiling_for_planner(parent.profiling, roofline)
                if parent.profiling is not None
                else _NO_PROFILE_SUMMARY
            )
            # Planner failure is branch-local: skip this iteration without
            # adding a tree node (no plan = no implementation to score) and
            # let the next select_next pick a different parent. Mirrors the
            # Coder skip-iter pattern below — a single agent hiccup cannot
            # kill a multi-iteration run.
            # Sibling context closes the prior contract drift where Planner
            # was blind to prior children of the same parent. See
            # doc/specs/2026-05-13-sibling-aware-agent-contracts-design.md.
            sibling_context, _ = _render_and_emit_sibling_context(
                tree, parent, iter_no=iter_no, consumer="planner",
                failure_cap=self._config.failure_sibling_cap,
            )
            try:
                with trace_span("acts_iter", iter_no=iter_no, agent=AgentLabel.PLANNER):
                    plan = await self._planner.plan(
                        kernel_source=parent.kernel.render_condensed_source(
                            representative_workload_uuid=(
                                workloads[0].uuid if workloads else None
                            ),
                        ),
                        profiling_summary=parent_profiling_summary,
                        past_experiences=experiences,
                        available_actions=available_actions,
                        action_menu=action_menu,
                        tree_context=tree.render_path(parent.id),
                        reviewer_feedback=_render_review_for_planner(parent.last_review),
                        bottleneck=run_bottleneck,
                        sibling_context=sibling_context,
                        max_turns=self._config.planner_max_turns,
                        hardware=self._config.hardware,
                        workload_shapes=workload_shapes_for_prompt,
                    )
            except PlanningError as exc:
                logger.warning(
                    "Iteration %d: Planner failed (%s) — skipping iteration",
                    iter_no, exc,
                )
                # Bump the parent's failure counter so a deterministically-
                # failing parent is quarantined from ``frontier()`` on the
                # next select_next instead of being re-picked forever and
                # silently consuming the entire ``max_depth`` budget.
                parent.consecutive_agent_failures += 1
                emit("planner_failed", iter=iter_no, reason=str(exc)[:200])
                emit("iter_end", iter=iter_no, outcome=ITER_SKIPPED)
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue
            emit(
                "planner_selected",
                iter=iter_no,
                technique=plan.technique,
                tier=plan.tier,
                rationale_short=plan.rationale[:120],
            )

            technique_guidance = _select_technique_guidance(
                applicable_actions, plan.technique
            )

            # K-way Coder fan-out: K parallel ``coder.implement`` calls
            # via ``asyncio.gather``. Decoder diversity at the forced
            # T=1.0 (configs/models/deepseek.json) produces variance —
            # no prompt perturbation, no per-call temperature schedule.
            # Per-candidate failures emit ``coder_failed`` with
            # ``candidate_idx`` and skip to the next candidate; all-K
            # failure marks the iter SKIPPED. The best-of-survivors
            # (highest SOL Score; tie-break lowest candidate_idx)
            # becomes the tree node — losers are not added to the tree.
            K = self._config.coder_n_candidates

            iter_failures: list[FailureDetail] = []
            iter_failure_kernels: list[tuple[int, "Kernel | None"]] = []

            async def _run_one_coder(_cand_idx: int):
                # Per-call trace_span so ``UsageAccumulator.invocations``
                # ticks K times per iter (a single outer span would close
                # once and underreport call count by K× in usage.json).
                with trace_span(
                    "acts_iter", iter_no=iter_no, agent=AgentLabel.CODER,
                    candidate_idx=_cand_idx,
                ):
                    return await self._coder.implement(
                        kernel_source=parent.kernel.source_code,
                        plan=plan,
                        kernel_spec=baseline.spec,
                        reference_fn=reference_fn,
                        input_generators=input_generators,
                        definition=definition,
                        workloads=workloads,
                        bottleneck=run_bottleneck,
                        iter_no=iter_no,
                        workload_shapes=workload_shapes_for_prompt,
                        technique_guidance=technique_guidance,
                        problem_definition_path=problem_definition_path,
                        blob_roots=_resolve_blob_roots(
                            self._config.safetensors_blob_roots,
                            problem_definition_path,
                        ),
                    )

            candidate_results = await asyncio.gather(
                *[_run_one_coder(i) for i in range(K)],
                return_exceptions=True,
            )

            # ``agent_failure_count`` tracks ImplementationError +
            # RewardHackDetected (agent-fault reasons that should bump
            # the parent's quarantine counter). Infra failures don't
            # count — they aren't the parent's fault.
            survivors: list[tuple[int, Any]] = []
            agent_failure_count = 0
            for cand_idx, result in enumerate(candidate_results):
                if isinstance(result, ImplementationError):
                    agent_failure_count += 1
                    # Turn-exhaust path: no kernel was submitted.
                    _accumulate_iter_failure(
                        iter_failures, iter_failure_kernels,
                        iter_no=iter_no, candidate_idx=cand_idx,
                        kernel=None, reason=str(result)[:200],
                    )
                elif isinstance(result, BaseException):
                    # Unexpected exception class — re-raise so the crash
                    # surfaces. Trade-off: later enumerate entries (both
                    # ImplementationErrors and successful outputs) are
                    # lost from this iter's event log; the crash is the
                    # priority signal.
                    raise result
                else:
                    survivors.append((cand_idx, result))

            if not survivors:
                logger.warning(
                    "Iteration %d: all %d Coder candidates failed — skipping",
                    iter_no, K,
                )
                if agent_failure_count > 0:
                    parent.consecutive_agent_failures += 1
                _persist_iter_failure_summary(
                    tree, parent, plan=plan, iter_no=iter_no,
                    iter_failures=iter_failures,
                    iter_failure_kernels=iter_failure_kernels,
                )
                emit("iter_end", iter=iter_no, outcome=ITER_SKIPPED)
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue

            # ───── Bench dispatch (subprocess or in-process bypass) ─────
            # Per-iter K-way bench + autotune burn-in + NCU profile runs
            # in a short-lived ``python -m src.eval.bench_worker`` child
            # for CUDA-context isolation (closes the trigger pattern
            # from `runs/sweep_l1_048/.../run_20260520T011532_929220Z` +
            # `runs/run_20260526T074709_405554Z` — sticky CUDA errors
            # like `cudaErrorIllegalAddress` die with the child, the
            # next iter spawns a fresh context). Falls back to inline
            # ``run_iter(...)`` when ``ACTSConfig.bench_use_subprocess
            # =False`` (debugging) OR ``run_dir is None`` (tests).
            # See doc/specs/2026-05-24-bench-subprocess-isolation-design.md.
            from src.eval.bench_subprocess import (
                WorkerCrashed,
                merge_worker_artifacts,
                run_bench_subprocess,
            )
            from src.eval.bench_worker import build_request
            from src.eval.bench_worker import run_iter as _run_iter_inproc

            # Entrypoint-binding pre-filter (stays in parent — mis-bound
            # candidates are dropped before they cross the IPC boundary
            # so the child only sees valid candidates).
            request_candidates: list[dict] = []
            for cand_idx, cand_output in survivors:
                cand_kernel = Kernel(
                    spec=baseline.spec,
                    source_code=cand_output.source_code,
                    triton_kernel_name=cand_output.triton_kernel_name,
                    dps=getattr(cand_output, "dps", False),
                )
                ok, reason = find_jit_name_in_entrypoint(
                    cand_output.source_code,
                    baseline.spec.entrypoint,
                    cand_output.triton_kernel_name,
                )
                if not ok:
                    _accumulate_iter_failure(
                        iter_failures, iter_failure_kernels,
                        iter_no=iter_no, candidate_idx=cand_idx,
                        kernel=cand_kernel,
                        reason=f"EntrypointBinding: {reason[:160]}",
                    )
                    continue
                request_candidates.append({
                    "candidate_idx": cand_idx,
                    "source_code": cand_output.source_code,
                    "triton_kernel_name": cand_output.triton_kernel_name,
                    "entrypoint": baseline.spec.entrypoint,
                    "dps": getattr(cand_output, "dps", False),
                })

            if not request_candidates:
                # All survivors dropped at entrypoint-binding pre-filter.
                # Same SKIPPED path as "all-K-fail-bench" today.
                parent.consecutive_agent_failures = 0
                _persist_iter_failure_summary(
                    tree, parent, plan=plan, iter_no=iter_no,
                    iter_failures=iter_failures,
                    iter_failure_kernels=iter_failure_kernels,
                )
                emit("iter_end", iter=iter_no, outcome=ITER_SKIPPED)
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue

            # Dispatch invariants (``_subprocess_enabled``,
            # ``_effective_run_dir``, ``_effective_ncu_cache``) are
            # hoisted out of the iter loop — see the once-per-run
            # block before the for-iteration loop (Codex 2026-05-27
            # fix #7). Per-iter only the worker_dir derives from them.
            worker_dir = _effective_run_dir / f"iter_{iter_no}" / "worker"
            worker_dir.mkdir(parents=True, exist_ok=True)

            request = build_request(
                run_dir=_effective_run_dir,
                iter_no=iter_no,
                worker_dir=worker_dir,
                ncu_cache_dir=_effective_ncu_cache,
                candidates=request_candidates,
                kernel_spec=_serialize_kernel_spec_for_request(baseline.spec),
                workloads=[
                    wl.model_dump(mode="json") for wl in (workloads or [])
                ],
                definition_path=problem_definition_path or Path(""),
                hardware_spec=self._config.hardware,
                anti_cheat_critical_names=list(
                    self._config.anti_cheat_critical_names
                ),
                bench_config={
                    "warmup_runs": self._config.warmup_runs,
                    "timed_runs": self._config.timed_runs,
                    "burn_in_seed": -1,
                },
                profile_config={
                    "ncu_enabled": True,
                    "analytical_enabled": True,
                    # Roofline-derived inputs computed once-per-iter in
                    # parent; threaded into child so the worker doesn't
                    # re-derive (cheap, but the dependency on SOLAR/
                    # roofline_shapes lives parent-side).
                    "iter_flops": iter_flops,
                    "iter_nbytes": iter_nbytes,
                    "repr_workload_idx": repr_idx,
                    # SOL ranking + profile context required by the
                    # child's profile gauntlet (Codex 2026-05-26 fix
                    # #1). Without ``t_sol_us`` / ``baseline_latency_us``
                    # the worker falls back to defaults that diverge
                    # from the parent's ranking contract. Without
                    # ``problem_definition_path`` / ``blob_roots``,
                    # safetensors-backed workloads crash the worker
                    # before any candidate is evaluated AND the NCU
                    # subprocess loses the inputs it needs to
                    # reconstruct workload state on profile.
                    "t_sol_us": roofline.t_sol_us,
                    "baseline_latency_us": scoring_baseline_latency_us,
                    "problem_definition_path": (
                        str(problem_definition_path)
                        if problem_definition_path is not None else None
                    ),
                    "blob_roots": [
                        str(p) for p in (_resolve_blob_roots(
                            self._config.safetensors_blob_roots,
                            problem_definition_path,
                        ) or [])
                    ],
                },
            )

            emit(
                "bench_worker_spawned",
                iter=iter_no,
                worker_dir=str(worker_dir),
                request_path=str(worker_dir / "request.json"),
            )
            walltime_start = time.time()

            try:
                if _subprocess_enabled:
                    response = await run_bench_subprocess(
                        request=request,
                        worker_dir=worker_dir,
                        worker_crash_threshold=self._config.worker_crash_threshold,
                        worker_timeout_s=self._config.worker_timeout_s,
                    )
                else:
                    response = _run_iter_inproc(request)
            except WorkerCrashed as exc:
                walltime_s = time.time() - walltime_start
                self.consecutive_worker_crashes += 1
                emit(
                    "bench_worker_crashed",
                    iter=iter_no,
                    returncode=exc.returncode,
                    stderr_tail=exc.stderr_tail,
                    consecutive=self.consecutive_worker_crashes,
                )
                emit(
                    "bench_worker_exited",
                    iter=iter_no,
                    returncode=exc.returncode,
                    walltime_s=walltime_s,
                )
                if (
                    self.consecutive_worker_crashes
                    >= self._config.worker_crash_threshold
                ):
                    raise WorkerProcessUnstable(
                        f"{self.consecutive_worker_crashes} consecutive "
                        f"worker crashes; last: rc={exc.returncode}"
                    ) from exc
                # Treat all K as failed → single failure-summary node.
                for cand in request_candidates:
                    _accumulate_iter_failure(
                        iter_failures, iter_failure_kernels,
                        iter_no=iter_no,
                        candidate_idx=cand["candidate_idx"],
                        kernel=_rebuild_cand_kernel(
                            request_candidates,
                            cand["candidate_idx"],
                            baseline.spec,
                        ),
                        reason=f"WorkerCrashed: rc={exc.returncode}",
                    )
                # NOTE (Codex 2026-05-26 review P2 fix #1): do NOT bump
                # ``parent.consecutive_agent_failures`` here. Worker
                # crashes are infrastructure failures already tracked
                # by ``self.consecutive_worker_crashes`` (3-strike
                # escalation to ``WorkerProcessUnstable``). Double-
                # counting them on the agent-quarantine axis would
                # remove the only frontier node from ``frontier()``
                # after just two transient crashes (``QUARANTINE_
                # THRESHOLD = 2``), ending the search as
                # ``ALL_DEAD_END`` *before* the third crash could
                # raise the correct ``WorkerProcessUnstable``.
                _persist_iter_failure_summary(
                    tree, parent, plan=plan, iter_no=iter_no,
                    iter_failures=iter_failures,
                    iter_failure_kernels=iter_failure_kernels,
                )
                emit("iter_end", iter=iter_no, outcome=ITER_SKIPPED)
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue

            walltime_s = time.time() - walltime_start
            emit(
                "bench_worker_exited",
                iter=iter_no,
                returncode=0,
                walltime_s=walltime_s,
            )
            self.consecutive_worker_crashes = 0  # reset on clean exit

            # Always merge per-iter worker chunk into canonical run
            # artifacts (events.jsonl concat + .ncu-rep copy into
            # shared cache + tree/node_<id>/), regardless of dispatch
            # mode. Codex 2026-05-27 fix #4: previously gated on
            # ``_subprocess_enabled`` — in-process bypass paths
            # (debugging, tests, RunContext OSError fallback) silently
            # lost any worker-emitted events AND any .ncu-rep files the
            # in-process worker call produced via
            # ``profile_kernel(cache_dir=worker_dir)``. The merge is a
            # no-op when worker wrote nothing, so unconditional merge
            # is safe + ensures consistent telemetry across modes.
            counts = merge_worker_artifacts(
                run_dir=_effective_run_dir,
                worker_dir=worker_dir,
                iter_no=iter_no,
                response=response,
                ncu_cache_dir=_effective_ncu_cache,
            )
            emit("worker_chunk_merged", iter=iter_no, **counts)

            # Per-candidate verdict accumulation — replaces the OLD
            # per-candidate try/except branches. ``not_run`` candidates
            # (post-channel-A-trip siblings) are silent; failed
            # candidates accumulate into iter_failures + emit
            # ``coder_failed``.
            bench_results: list[tuple[int, Any, Kernel, Any, Any]] = []
            for cand_resp in response["candidates"]:
                cand_idx = cand_resp["candidate_idx"]
                status = cand_resp["status"]
                if status == "success":
                    cand_kernel = _rebuild_cand_kernel(
                        request_candidates, cand_idx, baseline.spec,
                    )
                    cand_bench = _rehydrate_bench_result(
                        cand_resp["bench_result"]
                    )
                    # The live Triton ``Autotuner`` object can't cross
                    # IPC; ``_child_autotuner`` becomes a sentinel:
                    # non-None iff bench captured autotune winners.
                    # Downstream gate ``_child_autotuner is not None``
                    # at the autotune_burn_in_done emit behaves
                    # identically (records winners when present).
                    cand_autotuner = (
                        cand_bench.autotune_winner_per_workload or None
                    )
                    cand_output_obj = next(
                        s_out for s_idx, s_out in survivors
                        if s_idx == cand_idx
                    )
                    bench_results.append((
                        cand_idx, cand_output_obj, cand_kernel,
                        cand_bench, cand_autotuner,
                    ))
                elif status == "channel_a_tripped":
                    agent_failure_count += 1
                    emit(
                        "reward_hack_detected",
                        iter=iter_no,
                        candidate_idx=cand_idx,
                        reason=cand_resp.get("reason", "")[:200],
                    )
                    emit(
                        "coder_failed",
                        iter=iter_no,
                        candidate_idx=cand_idx,
                        reason=(
                            f"reward-hack: "
                            f"{cand_resp.get('reason', '')[:180]}"
                        ),
                    )
                elif status in ("bench_failed", "entrypoint_failed"):
                    cand_kernel = _rebuild_cand_kernel(
                        request_candidates, cand_idx, baseline.spec,
                    )
                    _accumulate_iter_failure(
                        iter_failures, iter_failure_kernels,
                        iter_no=iter_no, candidate_idx=cand_idx,
                        kernel=cand_kernel,
                        reason=cand_resp.get("reason", "")[:200],
                    )
                # ``not_run`` is silent — sibling of a channel-A trip;
                # never evaluated, no event, no failure node.

            # Channel-A iter-level abort (mirrors OLD iter_tainted path).
            # Child broke on first detection; remaining cands marked
            # not_run in response. Process taint dies with the child →
            # iter N+1 spawns a fresh context (spec §3 decision #9
            # cross-iter slice).
            if response.get("aborted_by_channel_A"):
                logger.warning(
                    "Iteration %d: channel-A reward-hack detected — "
                    "aborting iter (worker process terminated).",
                    iter_no,
                )
                parent.consecutive_agent_failures += 1
                _persist_iter_failure_summary(
                    tree, parent, plan=plan, iter_no=iter_no,
                    iter_failures=iter_failures,
                    iter_failure_kernels=iter_failure_kernels,
                )
                emit("iter_end", iter=iter_no, outcome=ITER_SKIPPED)
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue

            # No-winner case (combines OLD all-K-fail-bench branch +
            # all-profile-gauntlet-fail branch; child has run both
            # gauntlets and reports a single winner or None).
            winner_idx = response.get("winner_idx")
            if winner_idx is None:
                logger.warning(
                    "Iteration %d: no winner from bench/profile-gauntlet "
                    "— skipping",
                    iter_no,
                )
                # At least one Coder produced output → parent productive
                # → clear quarantine even if K-1 siblings failed
                # (stochastic decoder variance at T=1.0, not parent's
                # fault).
                parent.consecutive_agent_failures = 0
                _persist_iter_failure_summary(
                    tree, parent, plan=plan, iter_no=iter_no,
                    iter_failures=iter_failures,
                    iter_failure_kernels=iter_failure_kernels,
                )
                emit("iter_end", iter=iter_no, outcome=ITER_SKIPPED)
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue

            # Winner rehydration — child already ranked + profiled;
            # parent unpacks winner_idx, winner's bench tuple, and the
            # winner profile dataclass.
            winner_entry = next(
                e for e in bench_results if e[0] == winner_idx
            )
            (
                _w_idx, coder_output, child_kernel, bench, _child_autotuner,
            ) = winner_entry
            profiling = _rehydrate_profiling_result(
                response.get("winner_profile")
            )
            repr_workload_latency_s = _representative_latency_s(
                bench, workloads, repr_idx,
            )
            # ``profile_attempts`` was the OLD gauntlet's 1-indexed
            # rank of the profile-success winner. Child runs the
            # gauntlet now and doesn't currently report the rank; the
            # parent records 1 as a conservative placeholder. Surfaced
            # in ``coder_submitted.n_profile_attempts``.
            profile_attempts = 1

            # Winner committed to the tree (only after profile success).
            child = tree.add_child(
                parent.id,
                child_kernel,
                plan.technique,
                iter_no=iter_no,
                action_params=dict(plan.params) if plan.params else None,
            )
            # Successful child generation clears the parent's counter so
            # one transient blip earlier in the run doesn't permanently
            # quarantine an otherwise-productive node.
            parent.consecutive_agent_failures = 0
            emit(
                "coder_submitted",
                iter=iter_no,
                winner_candidate_idx=winner_idx,
                n_candidates=K,
                n_survivors=len(bench_results),
                n_profile_attempts=profile_attempts,
            )

            # A1 PR 1/B: copy benchmark-captured autotune winners onto the
            # winning kernel. Event fires only when the Autotuner reference
            # is real, so post-run analysis can distinguish "autotune ran"
            # from "pre-compile fallback path taken / no autotune."
            if bench.is_fully_successful and _child_autotuner is not None:
                _record_autotune_winner(child_kernel, bench)
                emit(
                    "autotune_burn_in_done",
                    iter=iter_no,
                    workload_count=len(workloads or []),
                    winner_count=len(child_kernel.autotune_winner),
                )

            emit(
                "bench_done",
                iter=iter_no,
                median_us=bench.median_latency_us,
                per_workload_us=_per_workload_us(bench),
                is_fully_successful=bench.is_fully_successful,
            )

            child.profiling = profiling
            if profiling is not None:
                ncu = profiling.ncu
                top_stalls: list[str] = []
                tc_util: float | None = None
                if ncu is not None:
                    if ncu.warp_stall_dominant:
                        top_stalls.append(ncu.warp_stall_dominant)
                    if ncu.warp_stall_runner_up:
                        top_stalls.append(ncu.warp_stall_runner_up)
                    tc_util = ncu.tensor_core_util_pct
                emit(
                    "profile_done",
                    iter=iter_no,
                    bottleneck=run_bottleneck_str,
                    top_stalls=top_stalls,
                    tensor_core_util_pct=tc_util,
                )
            child.score = compute_sol_score(
                scoring_baseline_latency_us,
                bench.median_latency_us,
                roofline.t_sol_us,
            )
            child.per_workload_latency_us = bench.per_workload_latency_us
            # Stash aggregate runtime (ms) for the opt-mem Producer's
            # parent → child speedup ratio. See TreeNode.runtime_ms.
            child.runtime_ms = bench.median_latency_us / 1000.0

            # Channel B reward-hack flow: ``reward_hack_suspect`` is the
            # SOL scorer's "T_k < ~T_SOL margin" signal. Re-eval with
            # strict tolerance + a fresh anti_cheat snapshot. If cleared,
            # accept the original score. If still flagged, mark the
            # branch DEAD_END so a candidate that beats the hardware
            # bound by gaming the bench doesn't propagate. We re-eval
            # BEFORE updating ``running_best_score`` and emitting
            # ``score_computed``: a confirmed hack must not be reported
            # as the run's new best, since downstream consumers track
            # that flag to update their own running-best mirror.
            if child.score.reward_hack_suspect:
                cleared = await self._reward_hack_re_eval(
                    child, child_kernel, workloads, input_generators,
                    reference_fn=reference_fn, definition=definition,
                    problem_definition_path=problem_definition_path,
                    blob_roots=_resolve_blob_roots(
                        self._config.safetensors_blob_roots,
                        problem_definition_path,
                    ),
                    run_dir=_effective_run_dir,
                    iter_no=iter_no,
                )
                if not cleared:
                    emit(
                        "reward_hack_confirmed",
                        iter=iter_no,
                        child_id=str(child.id),
                    )
                    self._kill_branch(
                        child, parent, iter_no,
                        reason=DeadReason.REWARD_HACK_CONFIRMED,
                        bumps_agent_failures=True,
                    )
                    # Channel-B confirmed-kill `continue`s past the
                    # end-of-iter persist below; flush K-1 sibling
                    # failures here so they aren't dropped.
                    _persist_iter_failure_summary(
                        tree, parent, plan=plan, iter_no=iter_no,
                        iter_failures=iter_failures,
                        iter_failure_kernels=iter_failure_kernels,
                    )
                    epsilon = max(self._config.epsilon_end, epsilon - decay)
                    continue
                emit("reward_hack_cleared", iter=iter_no, child_id=str(child.id))

            is_new_best = child.score.sol_score > running_best_score
            if is_new_best:
                running_best_score = child.score.sol_score
            emit(
                "score_computed",
                iter=iter_no,
                score=child.score.sol_score,
                is_new_best=is_new_best,
                reward_hack_suspect=child.score.reward_hack_suspect,
                calibration_warning=child.score.calibration_warning,
                t_k_us=bench.median_latency_us,
                t_b_us=scoring_baseline_latency_us,
                t_sol_us=roofline.t_sol_us,
                t_sol_source=roofline.source,
            )

            if child.score.calibration_warning:
                emit(
                    "calibration_warning",
                    iter=iter_no,
                    child_id=str(child.id),
                    t_k_us=bench.median_latency_us,
                    t_sol_us=roofline.t_sol_us,
                )

            # Tier 1 trace emission. Build a lightweight SOL ``Trace``
            # carrying the eval status + performance numbers + the
            # snapshotted environment and fire ``trace_emitted``.
            # Best-effort — never let a trace serialization hiccup
            # interrupt the search loop.
            self._emit_trace(
                iter_no=iter_no,
                child=child,
                bench=bench,
                roofline=roofline,
                definition=definition,
                workloads=workloads,
                repr_idx=repr_idx,
            )

            # Reviewer sees the same trajectory as the Planner, extended
            # through the just-scored child so `prev_sol_score` + the path's
            # last step let it ground its branch_quality in the real delta.
            # The reviewer also receives the live ProfilingResult — it renders
            # the analytical + NCU blocks from the dataclass directly.
            # When profiling was skipped, there's no meaningful analytical
            # block to hand the reviewer — default the branch to PROMISING
            # so it stays in the frontier and beam_prune treats it normally.
            if profiling is None:
                child.branch_quality = BranchQuality.PROMISING
            else:
                prev_sol = parent.score.sol_score if parent.score is not None else None
                reviewer_sibling_context, regressed = _render_and_emit_sibling_context(
                    tree, parent, iter_no=iter_no,
                    consumer="reviewer", exclude_id=child.id,
                    failure_cap=self._config.failure_sibling_cap,
                )
                with trace_span("acts_iter", iter_no=iter_no, agent=AgentLabel.REVIEWER):
                    feedback = await self._reviewer.review(
                        kernel_source=child.kernel.render_condensed_source(
                            representative_workload_uuid=(
                                workloads[0].uuid if workloads else None
                            ),
                        ),
                        profiling_summary=_NO_PROFILE_SUMMARY,  # superseded by profiling=
                        sol_score=child.score.sol_score,
                        headroom_pct=(1.0 - child.score.sol_score) * 100,
                        bottleneck=run_bottleneck,
                        tree_context=tree.render_path(child.id),
                        prev_sol_score=prev_sol,
                        profiling=profiling,
                        roofline=roofline,
                        reviewer_metric_queries=self._config.reviewer_metric_queries,
                        iter_idx=iter_no,
                        sibling_context=reviewer_sibling_context,
                        max_turns=self._config.reviewer_max_turns,
                        hardware=self._config.hardware,
                        workload_shapes=workload_shapes_for_prompt,
                    )
                if feedback.degraded:
                    logger.warning(
                        "Reviewer degraded at iteration %d (reason=%s) — branch_quality is rule-based.",
                        iter_no,
                        feedback.error_reason or "unknown",
                    )
                if feedback.branch_quality == BranchQuality.DEAD_END:
                    child.mark_dead(DeadReason.REVIEWER_JUDGED)
                    # Leading-indicator event: the existing system.md rule
                    # "regression + same pathway on sibling = dead_end"
                    # actually fired with sibling evidence. Lets postmortems
                    # count successful sibling-driven prunes without
                    # re-walking the tree.
                    for action, sibling_iter in regressed:
                        if action == child.action_applied:
                            emit(
                                "repeated_pathway_dead_end",
                                iter=iter_no,
                                action=action,
                                sibling_iter=sibling_iter,
                            )
                            break
                else:
                    child.branch_quality = feedback.branch_quality
                child.last_review = feedback
                emit(
                    "reviewer_feedback",
                    iter=iter_no,
                    verdict=feedback.branch_quality.value,
                    suggestion_short=feedback.outcome[:120],
                    degraded=feedback.degraded,
                )

            # Beam prune. Runs BEFORE ``tree_dump.dump_node`` below so that
            # the streamed ``meta.json`` reflects the post-prune
            # ``branch_quality`` (an evicted child gets DEAD_END here).
            # Pre-fix: dump_node ran first → meta.json said ``promising``
            # while the final ``index.json`` (built post-prune) said
            # ``dead_end`` for the same node.
            beam_prune(tree, self._config.beam_width, enable_diversity=self._config.beam_diversity)

            # Persist the committed node to <run_dir>/tree/node_<id>/.
            # Only fires on the ITER_ADVANCED path (post-score, post-reviewer,
            # post-prune); dead-end + skipped iterations dump from inside
            # ``_kill_branch`` instead. ``ncu_rep_src`` is the binary NCU
            # report path when the profiler captured one, else None.
            ncu_rep_src = (
                profiling.ncu_rep_path
                if profiling is not None and profiling.ncu is not None
                else None
            )
            tree_dump.dump_node(child, iter_no=iter_no, ncu_rep_src=ncu_rep_src)

            # Opt-mem write side: distill the improving (parent → child)
            # edge into a stored lesson. All gating happens inside the
            # Producer (write_enabled flag, δ improvement threshold,
            # cap eviction, summarizer-call failure handling) — best-
            # effort, never raises. See doc/specs/2026-05-24-
            # optimization-memory-design.md §8 + §9.
            #
            # Skip Reviewer-rejected DEAD_END children: a kernel that
            # benches faster but the Reviewer marked DEAD_END (reward-
            # hack suspicion, over-optimization, style regression) is
            # not a lesson worth distilling — writing it would pollute
            # the global store with patterns the Reviewer would reject
            # again on the next run. ``compiled``/``correct``-style gates
            # in the Producer don't catch this case because the kernel
            # IS compiled and correct; the rejection is quality-shaped.
            #
            # Note: ``beam_prune`` above also marks valid-but-non-frontier
            # children DEAD_END (``DeadReason.BEAM_PRUNED``). Those are
            # INTENTIONALLY allowed through — they ran fine and may have
            # improved their parent. The Producer's δ-improvement gate is
            # the arbiter for those edges, so gating on ``branch_quality``
            # alone (the pre-fix behavior) would have silently dropped an
            # improving edge once the beam filled. ``_is_reviewer_rejected``
            # distinguishes the two DEAD_END causes via ``dead_reason``.
            if self._producer is not None and not _is_reviewer_rejected(child):
                action_for_opt_mem = ActionRecord(
                    action_id=plan.technique,
                    tier=plan.tier,
                    name=plan.technique,
                    parameters={str(k): str(v) for k, v in plan.params.items()},
                )
                try:
                    await self._producer.consider(
                        parent, child, action_for_opt_mem, iter_no=iter_no,
                        bottleneck=run_bottleneck,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "opt-mem producer.consider failed (ignored): %s", exc
                    )

            # Attach summary sibling alongside the winner on mixed
            # outcomes; no-op when all K candidates succeeded.
            _persist_iter_failure_summary(
                tree, parent, plan=plan, iter_no=iter_no,
                iter_failures=iter_failures,
                iter_failure_kernels=iter_failure_kernels,
            )

            # Single end-of-iter best scan — reused for target / plateau checks.
            best = tree.best_node()
            emit("iter_end", iter=iter_no, outcome=ITER_ADVANCED)
            # Gate on the eligible winner: a REVIEWER_JUDGED child can
            # clear ``sol_target`` but be excluded by ``best_node()``;
            # using ``child.score`` here would ship a sub-target winner.
            if best.score.sol_score >= self._config.sol_target:
                await self._flush_opt_mem(root, tree, run_bottleneck)
                return SearchResult(
                    best,
                    iter_no,
                    TerminationReason.SOL_TARGET,
                    tree,
                    run_bottleneck=run_bottleneck,
                    reference_baseline_latency_us=reference_baseline_latency_us,
                    baseline_root_latency_us=baseline_bench.median_latency_us,
                )

            best_scores.append(best.score.sol_score)
            if detect_plateau(best_scores, self._config.sol_plateau_window, self._config.sol_plateau_delta):
                await self._flush_opt_mem(root, tree, run_bottleneck)
                return SearchResult(
                    best,
                    iter_no,
                    TerminationReason.PLATEAU,
                    tree,
                    run_bottleneck=run_bottleneck,
                    reference_baseline_latency_us=reference_baseline_latency_us,
                    baseline_root_latency_us=baseline_bench.median_latency_us,
                )

            epsilon = max(self._config.epsilon_end, epsilon - decay)

        await self._flush_opt_mem(root, tree, run_bottleneck)
        return SearchResult(
            tree.best_node(),
            self._config.max_depth,
            TerminationReason.BUDGET,
            tree,
            run_bottleneck=run_bottleneck,
            reference_baseline_latency_us=reference_baseline_latency_us,
            baseline_root_latency_us=baseline_bench.median_latency_us,
        )

    def _reward_hack_worker_dir(self, run_dir, iter_no, child) -> Path:
        """Worker scratch dir for a reward-hack re-eval child process.

        ``<run_dir>/iter_<n>/rh_<id>/`` when a real run_dir is set, else a
        fresh ``tempfile.mkdtemp()`` (placeholder / test paths). Mirrors how
        the per-iter bench path derives its ``worker_dir``.
        """
        if run_dir is not None:
            return Path(run_dir) / f"iter_{iter_no}" / f"rh_{child.id}"
        return Path(tempfile.mkdtemp(prefix="acts_rh_"))

    async def _reward_hack_re_eval(
        self,
        child,
        kernel,
        workloads,
        input_generators,
        *,
        reference_fn,
        definition,
        problem_definition_path=None,
        blob_roots=None,
        run_dir=None,
        iter_no=0,
        allow_in_parent_fallback: bool = False,
    ) -> bool:
        """Re-eval a suspect candidate with strict tolerance + fresh anti_cheat.

        Returns ``True`` when the re-eval cleared the suspicion (accept
        the original score), ``False`` when the candidate is still
        flagged (mark branch DEAD_END). Errors during the re-eval are
        treated as "not cleared" — fail-closed so a re-eval that crashes
        doesn't accidentally promote the original suspect score.

        Skip path: when no reference_fn / generators are available
        (placeholder runs), return True so the suspect is implicitly
        cleared — there's no oracle to compare against anyway, and the
        scorer's reward_hack_suspect bit is the only signal we'd have.

        Isolation: when a real ``problem_definition_path`` is supplied, the
        candidate is recompiled + re-launched inside a crash-isolated
        subprocess (``run_correctness_subprocess``, mode ``strict_recheck``)
        so an out-of-bounds launch poisons the (now-dead) child's CUDA
        context, never the parent's. When no definition path is available
        the legacy in-parent compare path below runs ONLY if the caller
        explicitly opts in via ``allow_in_parent_fallback=True`` (trusted /
        mocked contexts that carry no untrusted GPU candidates); absent both
        a path and the opt-in, ``CorrectnessIsolationError`` is raised so a
        forgotten ``problem_definition_path`` fails loud instead of silently
        launching an untrusted kernel in the parent CUDA context.
        """
        if reference_fn is None or not input_generators or not workloads:
            return True

        if problem_definition_path:
            from src.eval.correctness_subprocess import build_correctness_request

            request = build_correctness_request(
                spec=kernel.spec,
                source_code=kernel.source_code,
                dps=getattr(kernel, "dps", False),
                definition_path=problem_definition_path,
                workloads=workloads,
                blob_roots=blob_roots,
                mode="strict_recheck",
                input_seed=42,
                anti_cheat_critical_names=self._config.anti_cheat_critical_names,
                strict_atol=1e-5,
                strict_rtol=1e-4,
            )
            worker_dir = self._reward_hack_worker_dir(run_dir, iter_no, child)
            result = await run_correctness_subprocess(
                request=request,
                worker_dir=worker_dir,
                timeout_s=self._config.correctness_worker_timeout_s,
            )
            return result.passed

        if not allow_in_parent_fallback:
            from src.eval.correctness_subprocess import CorrectnessIsolationError

            raise CorrectnessIsolationError(
                "reward-hack re-eval needs a problem_definition_path to isolate "
                "the candidate launch; pass allow_in_parent_fallback=True only "
                "for trusted/mocked contexts."
            )

        # ── Legacy in-parent compare (opt-in, no definition path) ─────────
        from src.eval.anti_cheat import per_iter_anti_cheat
        from src.eval.correctness import strict_compare_one_workload
        from sol_execbench.core.bench.reward_hack import RewardHackDetected

        # Resolve the candidate entrypoint once. ``compile_kernel`` may
        # be expensive and we want all workloads to share one fn handle.
        try:
            from src.kernels.compiler import compile_kernel

            compiled = compile_kernel(kernel)
            if not compiled.success or compiled.compiled_fn is None:
                return False
            cand_fn = compiled.compiled_fn
        except Exception:
            return False

        # ``strict_compare_one_workload`` uses the normalized comparator so
        # multi-output (tuple/dict) returns are compared name-by-name via
        # SOL's ``normalize_outputs``. The prior tensor-only branch fell
        # through on tuple/dict and returned True, fail-OPEN — a suspect
        # multi-output kernel was auto-cleared without any output comparison.
        # When ``definition`` is absent (placeholder runs) the normalize
        # context is None and ``compare_outputs`` delegates to
        # ``policy.compare`` (single tensors), falling back to the catch-all
        # ``except Exception`` below for any other shape — fail-closed by
        # construction.
        try:
            with per_iter_anti_cheat(self._config.anti_cheat_critical_names):
                # Workloads + input_generators are 1:1 paired (same invariant
                # the benchmark loop relies on at src/eval/benchmark.py:156).
                # We zip here because the per-workload axes are needed to
                # resolve DPS output shapes for ``allocate_outputs`` (handled
                # inside ``strict_compare_one_workload``'s
                # ``maybe_wrap_dps_candidate``). An unwrapped
                # ``cand_fn(*inputs)`` raised TypeError on DPS kernels (host
                # wrapper expects ``(*inputs, *outputs)``); the catch-all
                # ``except Exception`` returned False, auto-confirming any DPS
                # branch that hit ``reward_hack_suspect`` as a hack regardless
                # of correctness.
                for wl, gen in zip(workloads, input_generators):
                    if not strict_compare_one_workload(
                        candidate_fn=cand_fn,
                        reference_fn=reference_fn,
                        input_generator=gen,
                        definition=definition,
                        kernel=kernel,
                        workload=wl,
                        seed=42,
                        atol=1e-5,
                        rtol=1e-4,
                    ):
                        return False
        except RewardHackDetected:
            return False
        except Exception:
            # Any other error during re-eval → fail-closed (treat as
            # not cleared). Surfacing the exception would crash the
            # whole run for what is supposed to be a per-branch check.
            return False
        return True

    def _emit_trace(
        self,
        *,
        iter_no: int,
        child,
        bench,
        roofline,
        definition,
        workloads,
        repr_idx: int,
    ) -> None:
        """Build a SOL ``Trace`` for this evaluation and emit ``trace_emitted``.
        All exceptions are swallowed — trace emission is best-effort
        observability.
        """
        try:
            from sol_execbench.core.data import (
                Correctness,
                Evaluation,
                EvaluationStatus,
                Performance,
                Trace,
            )
            from sol_execbench.core.utils import env_snapshot

            if self._environment is None:
                try:
                    self._environment = env_snapshot(device="cuda:0")
                except Exception:
                    # Tests / CPU-only paths — fabricate a stub
                    # environment so Trace's NonEmptyString validator
                    # doesn't reject it.
                    from sol_execbench.core.data import Environment

                    self._environment = Environment(hardware="unknown", libs={})

            if not workloads or definition is None:
                # Placeholder path — we don't have a real Workload to
                # bind the Trace to, so emit the event with a minimal
                # payload and skip the JSON dump.
                emit(
                    "trace_emitted",
                    iter=iter_no,
                    child_id=str(child.id),
                    latency_us=bench.median_latency_us,
                )
                return

            wl = workloads[repr_idx] if repr_idx < len(workloads) else workloads[0]
            from src.runtime.timefmt import iso_ts

            evaluation = Evaluation(
                status=EvaluationStatus.PASSED,
                environment=self._environment,
                timestamp=iso_ts(),
                correctness=Correctness(),
                performance=Performance(
                    latency_ms=bench.median_latency_us / 1000.0,
                    # SOL schema = PyTorch-reference latency; ACTS uses absolute SOL Score, not PyTorch-relative comparison, so unused.
                    reference_latency_ms=0.0,
                    speedup_factor=child.score.sol_score if child.score else 0.0,
                ),
            )
            trace = Trace(
                definition=definition.name,
                workload=wl,
                solution=str(child.id),
                evaluation=evaluation,
            )
            emit(
                "trace_emitted",
                iter=iter_no,
                child_id=str(child.id),
                trace=trace.model_dump(mode="json"),
            )
        except Exception as exc:
            logger.debug("trace_emitted: skipped (%s)", exc)
