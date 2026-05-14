"""Top-level search loop orchestrator (deterministic, not LLM).

Coordinates 3 LLM agents + deterministic eval per iteration:
    Planner -> Coder (with tools) -> [eval] -> Reviewer
The Coder's compile/correctness tools handle self-correction internally.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.agents.reviewer import BranchQuality, _render_review_for_planner
from src.runtime import tree_dump
from src.runtime.events import (
    DeadReason,
    ITER_ADVANCED,
    ITER_DEAD_END,
    ITER_SKIPPED,
    emit,
    finite_or_none,
)

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
    """

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload

    from src.agents.coder import CoderAgent
    from src.agents.reviewer import ReviewerAgent
    from src.agents.planner import PlannerAgent
    from src.config import ACTSConfig
    from src.eval.roofline import RooflineResult
    from src.eval.types import BottleneckType
    from src.kernels.kernel import Kernel
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
    autotuner: Any,
    workloads: list["Workload"],
    definition: "Definition | None" = None,
) -> None:
    """Populate ``kernel.autotune_winner`` from Triton's autotune cache (A1 PR 1).

    *autotuner* is the Triton ``Autotuner`` instance — i.e.
    ``module.<kernel.triton_kernel_name>`` — surfaced by
    ``compile_kernel(..).triton_autotuner``. The host wrapper (the
    entrypoint) doesn't carry the cache; the autotune decorator binds
    it to the wrapped ``@triton.jit`` function.

    Cache-key matching: Triton stores entries keyed on tuples that
    PREFIX with the user-declared ``key=`` values then append the
    pointer-arg dtypes (e.g. ``(M, N, K, "torch.float32", "torch.float32", "torch.float32")``).
    We compute the expected prefix from ``workload.axes`` plus any
    ``AxisConst`` values on *definition.axes* (Codex review finding #2,
    2026-05-14 — SOL problems with const definition axes were silently
    failing key lookup before the Definition was threaded through), then
    search the cache for the entry whose tuple starts with it.

    Best-effort: failures degrade to ``autotune_winner = None`` with a
    WARNING and never raise. Skipped when ``kernel.autotune_keys`` is
    empty (no per-workload identity to attribute) or when *autotuner*
    is ``None`` (no autotune wrapper present, e.g. legacy starters).
    """
    from src.eval.benchmark import _key_tuple_for

    if autotuner is None or not kernel.autotune_keys:
        return
    try:
        cache = getattr(autotuner, "cache", None)
        if not cache:
            return
        winners: dict[str, dict] = {}
        for wl in workloads:
            expected = _key_tuple_for(wl, kernel.autotune_keys, definition=definition)
            if expected is None:
                continue
            n = len(expected)
            matched_cfg = None
            for cache_key, cfg in cache.items():
                if isinstance(cache_key, tuple) and cache_key[:n] == expected:
                    matched_cfg = cfg
                    break
            if matched_cfg is None:
                continue
            winners[wl.uuid] = {
                "kwargs": dict(getattr(matched_cfg, "kwargs", {})),
                "num_warps": int(getattr(matched_cfg, "num_warps", 0)),
                "num_stages": int(getattr(matched_cfg, "num_stages", 0)),
            }
        if winners:
            kernel.autotune_winner = winners
    except Exception as exc:
        logger.warning(
            "autotune_winner unavailable: %s: %s",
            type(exc).__name__, exc,
        )


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
            f"pct_peak_compute={a.pct_peak_compute * 100:.1f}%",
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


try:
    from agents import trace as _agents_trace
except ModuleNotFoundError:
    _agents_trace = None  # SDK not installed (Tier-1 test venv); _iter_trace degrades.


def _iter_trace(iter_no: int, agent_name: str):
    """Return a context manager that wraps an agent invocation in an
    SDK ``trace`` tagged with the iteration index + agent name. Falls
    back to ``contextlib.nullcontext`` when the SDK isn't installed
    (Tier-1 test venv) so orchestration code stays harness-agnostic.
    """
    if _agents_trace is None:
        return contextlib.nullcontext()
    return _agents_trace(
        workflow_name="acts_iter",
        metadata={"iter": iter_no, "agent": agent_name},
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


def _render_and_emit_sibling_context(
    tree: "SearchTree",
    parent: "TreeNode",
    *,
    iter_no: int,
    consumer: str,
    exclude_id: int | None = None,
) -> tuple[str, list[tuple[str, int]]]:
    """Render the sibling section and emit ``sibling_context_rendered``.

    Returns ``(sibling_context, regressed)``. ``regressed`` is returned
    so the Reviewer-side caller can reuse it for the
    ``repeated_pathway_dead_end`` check without a second tree walk.
    """
    sibling_context = tree.render_siblings(parent.id, exclude_id=exclude_id)
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
    ) -> None:
        self._config = config
        self._planner = planner
        self._coder = coder
        self._reviewer = reviewer
        self._retriever = retriever
        self._tree: SearchTree | None = None
        # Cached SOL ``Environment`` for ``trace_emitted``. Built lazily
        # on first use so test paths that mock CUDA don't pay the
        # ``env_snapshot`` cost. Reset per ``run()`` invocation.
        self._environment = None

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
        from src.eval.profiler import ProfilerError, profile_kernel
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
        )
        if not baseline_bench.is_fully_successful:
            raise BenchmarkError(
                f"baseline benchmark had partial-workload failures "
                f"(errors={baseline_bench.workload_errors}); "
                f"SOL scoring requires a complete baseline measurement"
            )

        # A1 PR 1: record per-workload autotune winners for the baseline
        # when we have an Autotuner reference. Skipped on the lazy-compile
        # fallback path, on placeholder baselines without an autotune
        # decorator, and on legacy starters — autotune_winner stays None
        # in those cases (no observability, run continues).
        if _baseline_autotuner is not None:
            _record_autotune_winner(
                baseline, _baseline_autotuner, workloads, definition=definition,
            )
            emit(
                "autotune_burn_in_done",
                iter=root.iter_no,
                workload_count=len(workloads),
                winner_recorded=bool(baseline.autotune_winner),
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

        root.score = compute_sol_score(
            baseline_bench.median_latency_us,
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
                with _iter_trace(0, "reviewer"):
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

        for iteration in range(self._config.max_depth):
            iter_no = iteration + 1
            frontier = tree.frontier()
            if not frontier:
                return SearchResult(
                    tree.best_node(),
                    iteration,
                    TerminationReason.ALL_DEAD_END,
                    tree,
                    run_bottleneck=run_bottleneck,
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

            # Retriever + Planner + Reviewer all share the run-level
            # bottleneck — classification is invariant per
            # (problem, representative workload, hardware) so we do not
            # derive it per-iteration from profiling results.
            experiences = self._retriever.retrieve(
                baseline.spec.kernel_type.value,
                run_bottleneck,
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
            )
            try:
                with _iter_trace(iter_no, "planner"):
                    plan = await self._planner.plan(
                        kernel_source=parent.kernel.render_condensed_source(
                            representative_workload_uuid=(
                                workloads[0].uuid if workloads else None
                            ),
                        ),
                        profiling_summary=parent_profiling_summary,
                        past_experiences=experiences,
                        available_actions=[],
                        tree_context=tree.render_path(parent.id),
                        reviewer_feedback=_render_review_for_planner(parent.last_review),
                        bottleneck=run_bottleneck,
                        sibling_context=sibling_context,
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

            # Coder (with tools for self-correction). `kernel_spec` /
            # `reference_fn` / `input_generators` are threaded into the
            # compile + correctness tools the Coder binds per call — the
            # full generator list so cross-workload bugs surface in-turn.
            # Coder failure (turn-budget exhaustion, transient retry
            # exhaustion, missing submit_kernel call) is branch-local:
            # skip this iteration without adding a tree node and let the
            # next select_next pick a different parent. The full search
            # run survives one Coder hiccup the way it survives one
            # branch's benchmark crash.
            try:
                with _iter_trace(iter_no, "coder"):
                    coder_output = await self._coder.implement(
                        kernel_source=parent.kernel.source_code,
                        plan=plan,
                        kernel_spec=baseline.spec,
                        reference_fn=reference_fn,
                        input_generators=input_generators,
                    )
            except ImplementationError as exc:
                logger.warning(
                    "Iteration %d: Coder failed (%s) — skipping iteration",
                    iter_no, exc,
                )
                # See planner_failed branch above for the quarantine rationale.
                parent.consecutive_agent_failures += 1
                emit("coder_failed", iter=iter_no, reason=str(exc)[:200])
                emit("iter_end", iter=iter_no, outcome=ITER_SKIPPED)
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue
            emit("coder_submitted", iter=iter_no)
            new_source = coder_output.source_code

            # Build child kernel — carry the Coder's declared
            # ``triton_kernel_name`` so the profiler skips the regex
            # fallback and filters NCU on the symbol the Coder named.
            # ``dps`` is threaded so DPS kernels see the pre-allocated
            # output buffers in the benchmark + correctness paths.
            child_kernel = Kernel(
                spec=baseline.spec,
                source_code=new_source,
                triton_kernel_name=coder_output.triton_kernel_name,
                dps=getattr(coder_output, "dps", False),
            )
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

            # Child-benchmark failure is branch-local: BenchmarkError
            # (majority-failure) and non-empty workload_errors (partial
            # failure) both mark the child DEAD_END so a kernel that
            # crashes on a slice of the workload set cannot be scored or
            # promoted. Baseline failure is not caught — no baseline
            # means no signal, and the caller is expected to surface it.
            #
            # The eval block (benchmark) is wrapped in
            # ``per_iter_anti_cheat`` so a candidate that monkey-patches
            # torch.cuda.Event, spawns a background thread, or returns
            # lazy/proxy outputs trips ``RewardHackDetected`` and lands
            # on the DEAD_END path. CUDA sticky-state errors get one
            # ``synchronize()`` retry before counting against the 3-strike
            # ``CUDAContextPoisoned`` budget.
            dead_detail: str | None = None
            bench = None
            _compiled_fn: Any | None = None
            _child_autotuner: Any | None = None
            try:
                # A1 PR 1 + Codex finding #1: precompile MUST run inside
                # ``per_iter_anti_cheat`` so import-time side effects from
                # the candidate source (exec_module runs top-level Python)
                # register as drift against the snapshot rather than
                # silently becoming the new baseline.
                with per_iter_anti_cheat(self._config.anti_cheat_critical_names):
                    _compiled_fn, _child_autotuner = _safe_precompile(
                        child_kernel, role="Child",
                    )
                    bench = benchmark_kernel(
                        child_kernel,
                        self._config,
                        workloads=workloads,
                        input_generators=input_generators,
                        definition=definition,
                        kernel_fn=_compiled_fn,
                    )
                check_lazy_outputs_after_bench(bench.last_outputs)
                # Drop GPU tensor refs as soon as the lazy-output check
                # accepts them — for large workloads the captured outputs
                # are hundreds of MB pinned through the LLM round-trip
                # ahead (profile + score + reviewer).
                bench.last_outputs.clear()
                # A1 PR 1: query Triton's autotune cache for the per-workload
                # winning config. Skipped on the lazy-compile fallback (no
                # autotuner captured) and on kernels without an autotune
                # decorator (autotuner is None). Event fires only when the
                # Autotuner reference is real, so post-run analysis can
                # distinguish "autotune ran" from "pre-compile fallback
                # path taken / no autotune."
                if bench.is_fully_successful and _child_autotuner is not None:
                    _record_autotune_winner(
                        child_kernel, _child_autotuner, workloads,
                        definition=definition,
                    )
                    emit(
                        "autotune_burn_in_done",
                        iter=iter_no,
                        workload_count=len(workloads),
                        winner_recorded=bool(child_kernel.autotune_winner),
                    )
                # Bench succeeded without any reward-hack signal — clear
                # the CUDA error counter so a single transient blip earlier
                # in the run doesn't accumulate forever.
                consecutive_cuda_errors = 0
            except RewardHackDetected as e:
                emit(
                    "reward_hack_detected",
                    iter=iter_no,
                    reason=str(e)[:200],
                    child_id=str(child.id),
                )
                self._kill_branch(
                    child, parent, iter_no,
                    reason=DeadReason.REWARD_HACK,
                    detail=str(e)[:120],
                    bumps_agent_failures=True,
                )
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue
            except BenchmarkError as e:
                dead_detail = f"child benchmark failed ({e})"
            except RuntimeError as e:
                # CUDA sticky-state recovery. Match against known transient
                # launch-failure patterns only; a generic RuntimeError that
                # merely *mentions* "CUDA" (e.g. "operation not implemented
                # for CUDA") is a real bug and must propagate.
                msg = str(e)
                msg_lower = msg.lower()
                if not any(p in msg_lower for p in _CUDA_STICKY_PATTERNS):
                    raise
                try:
                    import torch
                    torch.cuda.synchronize()
                    # sync succeeded — branch-local failure, run continues.
                    consecutive_cuda_errors = 0
                except Exception:
                    consecutive_cuda_errors += 1
                    if consecutive_cuda_errors >= 3:
                        raise CUDAContextPoisoned(
                            f"3+ consecutive cuda.synchronize() failures: {e}"
                        ) from e
                logger.warning(
                    "Iteration %d: transient CUDA error (%s) — branch DEAD_END",
                    iter_no, msg[:200],
                )
                self._kill_branch(
                    child, parent, iter_no,
                    reason=DeadReason.CUDA_ERROR,
                    detail=msg[:120],
                )
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue
            else:
                if not bench.is_fully_successful:
                    dead_detail = (
                        f"child benchmark had partial-workload failures "
                        f"(errors={bench.workload_errors})"
                    )

            if dead_detail is not None:
                logger.warning("Iteration %d: %s — marking branch dead_end", iter_no, dead_detail)
                # bench_done fires even on partial-workload failure so the
                # event log has a row for every benchmark attempt.
                emit(
                    "bench_done",
                    iter=iter_no,
                    median_us=bench.median_latency_us if bench is not None else 0.0,
                    per_workload_us=_per_workload_us(bench),
                    is_fully_successful=False,
                )
                self._kill_branch(
                    child, parent, iter_no,
                    reason=DeadReason.BENCH_FAILURE,
                    detail=dead_detail,
                )
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue

            emit(
                "bench_done",
                iter=iter_no,
                median_us=bench.median_latency_us,
                per_workload_us=_per_workload_us(bench),
                is_fully_successful=bench.is_fully_successful,
            )

            # Profile the child on the representative workload. Analytical
            # failure (ProfilerError: zero latency, missing peaks) is
            # branch-killing — the latency measurement is meaningless
            # without a basis for roofline comparison. NCU subprocess
            # failure degrades the profile (ncu=None) but the analytical
            # block still drives per-iter metrics and the branch survives.
            # Score and per-workload latencies are deferred past this
            # gauntlet so ``best_node()`` (which filters on ``score is not
            # None``) cannot promote a profile-killed branch.
            profiling = None
            repr_workload_latency_s = _representative_latency_s(
                bench, workloads, repr_idx
            )
            if repr_workload_latency_s is None:
                # Representative workload's measurement is inf (partial
                # failure on this slice) — we've already caught fully-
                # dead children above, so this is the "majority survived
                # but the middle workload didn't" edge. Skip profiling;
                # the bench_failure path would have kicked in for ≥50%
                # failure.
                logger.warning(
                    "Iteration %d: representative workload latency unavailable "
                    "— skipping profile (child benchmark: %s)",
                    iter_no,
                    bench.per_workload_latency_us,
                )
                self._kill_branch(
                    child, parent, iter_no,
                    reason=DeadReason.REPR_LATENCY_UNAVAILABLE,
                )
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue

            # No gate — profile_kernel accepts flops=0 (pct_peak_compute=0)
            # and nbytes=0 (analytical=None, NCU still runs).
            profile_blob_roots = _resolve_blob_roots(
                self._config.safetensors_blob_roots,
                problem_definition_path,
            )
            try:
                profiling = profile_kernel(
                    child_kernel,
                    repr_workload_axes,
                    repr_input_generator,
                    hardware_spec=self._config.hardware,
                    flops=iter_flops,
                    nbytes=iter_nbytes,
                    latency_s=repr_workload_latency_s,
                    problem_definition_path=problem_definition_path,
                    blob_roots=profile_blob_roots,
                )
            except ProfilerError as e:
                logger.warning(
                    "Iteration %d: profile_kernel failed (%s) — marking branch dead_end",
                    iter_no,
                    e,
                )
                self._kill_branch(
                    child, parent, iter_no,
                    reason=DeadReason.PROFILER_ERROR,
                    detail=str(e)[:120],
                )
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue
            child.profiling = profiling
            if profiling is not None:
                ncu = profiling.ncu
                top_stalls: list[str] = []
                tc_util = 0.0
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
                baseline_bench.median_latency_us,
                bench.median_latency_us,
                roofline.t_sol_us,
            )
            child.per_workload_latency_us = bench.per_workload_latency_us

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
                t_b_us=baseline_bench.median_latency_us,
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
                )
                with _iter_trace(iter_no, "reviewer"):
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

            # Single end-of-iter best scan — reused for target / plateau checks.
            best = tree.best_node()
            emit("iter_end", iter=iter_no, outcome=ITER_ADVANCED)
            # Gate on the eligible winner: a REVIEWER_JUDGED child can
            # clear ``sol_target`` but be excluded by ``best_node()``;
            # using ``child.score`` here would ship a sub-target winner.
            if best.score.sol_score >= self._config.sol_target:
                return SearchResult(
                    best,
                    iter_no,
                    TerminationReason.SOL_TARGET,
                    tree,
                    run_bottleneck=run_bottleneck,
                )

            best_scores.append(best.score.sol_score)
            if detect_plateau(best_scores, self._config.sol_plateau_window, self._config.sol_plateau_delta):
                return SearchResult(
                    best,
                    iter_no,
                    TerminationReason.PLATEAU,
                    tree,
                    run_bottleneck=run_bottleneck,
                )

            epsilon = max(self._config.epsilon_end, epsilon - decay)

        return SearchResult(
            tree.best_node(),
            self._config.max_depth,
            TerminationReason.BUDGET,
            tree,
            run_bottleneck=run_bottleneck,
        )

    async def _reward_hack_re_eval(
        self,
        child,
        kernel,
        workloads,
        input_generators,
        *,
        reference_fn,
        definition,
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
        """
        from src.eval.anti_cheat import (
            generate_randomized_inputs,
            per_iter_anti_cheat,
        )
        from src.eval.correctness import (
            TorchComparisonPolicy,
            build_normalize_context,
            compare_outputs,
            maybe_wrap_dps_candidate,
        )
        from sol_execbench.core.bench.reward_hack import RewardHackDetected

        if reference_fn is None or not input_generators or not workloads:
            return True

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

        # Use the normalized comparator so multi-output (tuple/dict)
        # returns are compared name-by-name via SOL's ``normalize_outputs``.
        # The prior tensor-only branch fell through on tuple/dict and
        # returned True, fail-OPEN — a suspect multi-output kernel was
        # auto-cleared without any output comparison. ``norm`` is None
        # when ``definition`` is absent (placeholder runs); in that case
        # ``compare_outputs`` delegates to ``policy.compare`` which handles
        # single tensors and falls back to the catch-all ``except Exception``
        # below for any other shape — fail-closed by construction.
        policy = TorchComparisonPolicy()
        norm = build_normalize_context(definition)

        try:
            with per_iter_anti_cheat(self._config.anti_cheat_critical_names):
                # Workloads + input_generators are 1:1 paired (same invariant
                # the benchmark loop relies on at src/eval/benchmark.py:156).
                # We zip here because ``maybe_wrap_dps_candidate`` needs the
                # per-workload axes to resolve output shapes for
                # ``allocate_outputs``. The unwrapped ``cand_fn(*inputs)``
                # raised TypeError on DPS kernels (host wrapper expects
                # ``(*inputs, *outputs)``); the catch-all ``except Exception``
                # returned False, auto-confirming any DPS branch that hit
                # ``reward_hack_suspect`` as a hack regardless of correctness.
                for wl, gen in zip(workloads, input_generators):
                    wrapped_cand = maybe_wrap_dps_candidate(
                        cand_fn,
                        kernel=kernel,
                        workload=wl,
                        definition=definition,
                    )
                    inputs = generate_randomized_inputs(gen, seed=42)
                    cand_out = wrapped_cand(*inputs)
                    ref_out = reference_fn(*inputs)
                    outcome = compare_outputs(
                        cand_out,
                        ref_out,
                        policy=policy,
                        atol=1e-5,
                        rtol=1e-4,
                        norm=norm,
                    )
                    if not outcome.match:
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
