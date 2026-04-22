"""Top-level search loop orchestrator (deterministic, not LLM).

Coordinates 3 LLM agents + deterministic eval per iteration:
    Planner -> Coder (with tools) -> [eval] -> Reviewer
The Coder's compile/correctness tools handle self-correction internally.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.agents.coder import CoderAgent
    from src.agents.reviewer import ReviewerAgent
    from src.agents.planner import PlannerAgent
    from src.benchmark.problem import Problem, Workload
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


def _render_profiling_for_planner(profiling) -> str:
    """Lightweight summary for the Planner's prompt. The Reviewer owns the
    full analytical + NCU rendering via ``reviewer.render_profiling_summary``;
    the Planner only needs a couple of numbers to reason about the *next*
    technique to try. Bottleneck classification is hoisted to a dedicated
    "Run context" section by the Planner prompt, so it's not repeated here.
    """
    a = profiling.analytical
    lines = [
        f"pct_peak_compute={a.pct_peak_compute * 100:.1f}%",
        f"pct_peak_bandwidth={a.pct_peak_bandwidth * 100:.1f}%",
        f"arithmetic_intensity={a.arithmetic_intensity:.3f}",
    ]
    if profiling.ncu is not None:
        n = profiling.ncu
        lines.append(f"sm_occupancy={n.sm_occupancy_pct:.1f}%")
        lines.append(f"l2_hit_rate={n.l2_hit_rate_pct:.1f}%")
        lines.append(f"dominant_stall={n.warp_stall_dominant}")
    elif profiling.degraded:
        lines.append(f"[DEGRADED: {profiling.degraded_reason or 'unknown'}]")
    return ", ".join(lines)


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

    async def run(
        self,
        baseline: Kernel,
        workloads: list[Workload] | None = None,
        roofline: RooflineResult | None = None,
        *,
        reference_fn: Callable[..., Any] | None = None,
        input_generators: list[Callable[[int], tuple]] | None = None,
        problem_definition_path: Path | None = None,
        problem: Problem | None = None,
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

        *problem*: the parsed ``Problem`` — used each iteration to derive
        per-workload ``(flops, nbytes)`` for the analytical profiler
        (``problem_to_kernel_spec`` deliberately leaves the spec's flop /
        byte counts at zero for SOL problems). ``None`` falls back to
        ``baseline.spec.flop_count`` / ``memory_bytes`` — correct for the
        placeholder starter kernels, which populate those fields directly.
        """
        from src.agents.reviewer import BranchQuality
        from src.eval.benchmark import BenchmarkError, benchmark_kernel
        from src.eval.profiler import ProfilerError, profile_kernel
        from src.eval.roofline import classify_run, compute_roofline
        from src.eval.scorer import compute_sol_score
        from src.kernels.kernel import Kernel, KernelSpec
        from src.search.beam import beam_prune, select_next
        from src.search.tree import SearchTree

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
        baseline_bench = benchmark_kernel(
            baseline,
            self._config,
            workloads=workloads,
            input_generators=input_generators,
        )
        if not baseline_bench.is_fully_successful:
            raise BenchmarkError(
                f"baseline benchmark had partial-workload failures "
                f"(errors={baseline_bench.workload_errors}); "
                f"SOL scoring requires a complete baseline measurement"
            )

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

        # Representative workload index for per-iteration profiling (spec
        # §3.3). Middle of the selected-workload list so large/small-axis
        # outliers don't dominate the profile; falls back to 0 when
        # len(workloads) < 2 or workloads is None.
        repr_idx = (len(workloads) // 2) if workloads else 0

        # Per-iteration (flops, nbytes) are invariant across the run —
        # derived from (problem, representative workload) or from the
        # baseline spec in the placeholder path — so hoist them out of
        # the loop instead of recomputing every iteration.
        if problem is not None and workloads:
            from src.benchmark.roofline_shapes import compute_roofline_inputs
            iter_flops, iter_nbytes = compute_roofline_inputs(
                problem, workloads[repr_idx]
            )
            repr_workload_axes = dict(workloads[repr_idx].__dict__)
        else:
            iter_flops = baseline.spec.flop_count
            iter_nbytes = baseline.spec.memory_bytes
            repr_workload_axes = {}
        repr_input_generator = (
            input_generators[repr_idx] if input_generators else (lambda seed: ())
        )

        for iteration in range(self._config.max_depth):
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
                _render_profiling_for_planner(parent.profiling)
                if parent.profiling is not None
                else _NO_PROFILE_SUMMARY
            )
            plan = await self._planner.plan(
                kernel_source=parent.kernel.source_code,
                profiling_summary=parent_profiling_summary,
                past_experiences=experiences,
                available_actions=[],
                tree_context=tree.render_path(parent.id),
                reviewer_feedback=None,
                bottleneck=run_bottleneck,
            )

            # Coder (with tools for self-correction). `kernel_spec` /
            # `reference_fn` / `input_generators` are threaded into the
            # compile + correctness tools the Coder binds per call — the
            # full generator list so cross-workload bugs surface in-turn.
            coder_output = await self._coder.implement(
                kernel_source=parent.kernel.source_code,
                plan=plan,
                kernel_spec=baseline.spec,
                reference_fn=reference_fn,
                input_generators=input_generators,
            )
            new_source = coder_output.source_code

            # Build child kernel — carry the Coder's declared
            # ``triton_kernel_name`` so the profiler skips the regex
            # fallback and filters NCU on the symbol the Coder named.
            child_kernel = Kernel(
                spec=baseline.spec,
                source_code=new_source,
                triton_kernel_name=coder_output.triton_kernel_name,
            )
            child = tree.add_child(parent.id, child_kernel, plan.technique)

            # Child-benchmark failure is branch-local: BenchmarkError
            # (majority-failure) and non-empty workload_errors (partial
            # failure) both mark the child DEAD_END so a kernel that
            # crashes on a slice of the workload set cannot be scored or
            # promoted. Baseline failure is not caught — no baseline
            # means no signal, and the caller is expected to surface it.
            dead_reason: str | None = None
            bench = None
            try:
                bench = benchmark_kernel(
                    child_kernel,
                    self._config,
                    workloads=workloads,
                    input_generators=input_generators,
                )
            except BenchmarkError as e:
                dead_reason = f"child benchmark failed ({e})"
            else:
                if not bench.is_fully_successful:
                    dead_reason = (
                        f"child benchmark had partial-workload failures "
                        f"(errors={bench.workload_errors})"
                    )

            if dead_reason is not None:
                logger.warning("Iteration %d: %s — marking branch dead_end", iteration + 1, dead_reason)
                child.branch_quality = BranchQuality.DEAD_END
                beam_prune(tree, self._config.beam_width, enable_diversity=self._config.beam_diversity)
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue

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
                # dead_reason would have kicked in for ≥50% failure.
                logger.warning(
                    "Iteration %d: representative workload latency unavailable "
                    "— skipping profile (child benchmark: %s)",
                    iteration + 1,
                    bench.per_workload_latency_us,
                )
                child.branch_quality = BranchQuality.DEAD_END
                beam_prune(tree, self._config.beam_width, enable_diversity=self._config.beam_diversity)
                epsilon = max(self._config.epsilon_end, epsilon - decay)
                continue

            if iter_flops > 0 and iter_nbytes > 0:
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
                    )
                except ProfilerError as e:
                    logger.warning(
                        "Iteration %d: profile_kernel failed (%s) — marking branch dead_end",
                        iteration + 1,
                        e,
                    )
                    child.branch_quality = BranchQuality.DEAD_END
                    beam_prune(tree, self._config.beam_width, enable_diversity=self._config.beam_diversity)
                    epsilon = max(self._config.epsilon_end, epsilon - decay)
                    continue
            else:
                # No formula for this op_type. The profile is an enrichment,
                # not a gate — keep the branch alive and let the retriever
                # fall back to roofline.bottleneck.
                logger.warning(
                    "Iteration %d: skipping profile — no (flops, nbytes) for "
                    "op_type=%r (branch stays alive)",
                    iteration + 1,
                    problem.op_type if problem is not None else "<no-problem>",
                )
            child.profiling = profiling
            child.score = compute_sol_score(
                baseline_bench.median_latency_us,
                bench.median_latency_us,
                roofline.t_sol_us,
            )
            child.per_workload_latency_us = bench.per_workload_latency_us

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
                feedback = await self._reviewer.review(
                    kernel_source=new_source,
                    profiling_summary=_NO_PROFILE_SUMMARY,  # superseded by profiling=
                    sol_score=child.score.sol_score,
                    headroom_pct=(1.0 - child.score.sol_score) * 100,
                    bottleneck=run_bottleneck,
                    tree_context=tree.render_path(child.id),
                    prev_sol_score=prev_sol,
                    profiling=profiling,
                )
                if feedback.degraded:
                    logger.warning(
                        "Reviewer degraded at iteration %d (reason=%s) — branch_quality is rule-based.",
                        iteration + 1,
                        feedback.error_reason or "unknown",
                    )
                child.branch_quality = feedback.branch_quality

            # Beam prune
            beam_prune(tree, self._config.beam_width, enable_diversity=self._config.beam_diversity)

            # Single end-of-iter best scan — reused for target / plateau checks.
            best = tree.best_node()
            if child.score.sol_score >= self._config.sol_target:
                return SearchResult(
                    best,
                    iteration + 1,
                    TerminationReason.SOL_TARGET,
                    tree,
                    run_bottleneck=run_bottleneck,
                )

            best_scores.append(best.score.sol_score)
            if detect_plateau(best_scores, self._config.sol_plateau_window, self._config.sol_plateau_delta):
                return SearchResult(
                    best,
                    iteration + 1,
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
