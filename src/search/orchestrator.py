"""Top-level search loop orchestrator (deterministic, not LLM).

Coordinates 3 LLM agents + deterministic eval per iteration:
    Planner -> Coder (with tools) -> [eval] -> Reviewer
The Coder's compile/correctness tools handle self-correction internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.coder import CoderAgent
    from src.agents.evaluator import EvaluatorAgent
    from src.agents.planner import PlannerAgent
    from src.benchmark.problem import Workload
    from src.config import ACTSConfig
    from src.eval.roofline import RooflineResult
    from src.kernels.kernel import Kernel
    from src.memory.retriever import MemoryRetriever
    from src.search.tree import SearchTree, TreeNode


@dataclass
class SearchResult:
    """Final result of the search process."""

    best_node: TreeNode
    total_iterations: int
    termination_reason: str  # "sol_target", "plateau", "budget", "all_dead_end"


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
        evaluator: EvaluatorAgent,
        retriever: MemoryRetriever,
    ) -> None:
        self._config = config
        self._planner = planner
        self._coder = coder
        self._evaluator = evaluator
        self._retriever = retriever
        self._tree: SearchTree | None = None

    async def run(
        self,
        baseline: Kernel,
        workloads: list[Workload] | None = None,
        roofline: RooflineResult | None = None,
    ) -> SearchResult:
        """Execute the full search loop from baseline to best kernel.

        *workloads*: representative subset for iterative benchmarking
        (SOL-ExecBench mode).  When ``None``, benchmarking uses
        ``kernel.spec.input_shapes`` (legacy mode).

        *roofline*: pre-computed SOLAR result (T_SOL + bottleneck).
        When ``None``, falls back to built-in roofline from
        ``KernelSpec.flop_count`` / ``KernelSpec.memory_bytes``.
        """
        from src.eval.benchmark import benchmark_kernel
        from src.eval.profiler import profile_kernel
        from src.eval.roofline import compute_roofline
        from src.eval.scorer import compute_sol_score
        from src.kernels.kernel import Kernel, KernelSpec
        from src.search.beam import beam_prune, select_next
        from src.search.tree import SearchTree

        tree = SearchTree()
        self._tree = tree

        # Phase A: baseline evaluation
        root = tree.add_root(baseline)
        baseline_bench = benchmark_kernel(baseline, self._config, workloads=workloads)

        # T_SOL + bottleneck: use SOLAR result if provided, else built-in fallback
        if roofline is None:
            roofline = compute_roofline(baseline.spec, self._config.hardware)

        root.score = compute_sol_score(
            baseline_bench.median_latency_us,
            baseline_bench.median_latency_us,
            roofline.t_sol_us,
        )

        # Phase B: search loop
        epsilon = self._config.epsilon_start
        decay = (self._config.epsilon_start - self._config.epsilon_end) / max(self._config.max_depth, 1)
        best_scores: list[float] = []

        for iteration in range(self._config.max_depth):
            frontier = tree.frontier()
            if not frontier:
                return SearchResult(tree.best_node(), iteration, "all_dead_end")

            parent = select_next(tree, epsilon)

            # Retrieve experiences
            experiences = self._retriever.retrieve(
                baseline.spec.kernel_type.value,
                roofline.bottleneck.value,
            )

            # Planner
            plan = await self._planner.plan(
                kernel_source=parent.kernel.to_source(),
                profiling_summary="placeholder profiling summary",
                past_experiences=experiences,
                available_actions=[],
                reviewer_feedback=None,
            )

            # Coder (with tools for self-correction)
            new_source = await self._coder.implement(
                kernel_source=parent.kernel.to_source(),
                plan=plan,
            )

            # Build child kernel
            child_kernel = Kernel(spec=baseline.spec, source_code=new_source)
            child = tree.add_child(parent.id, child_kernel, plan.technique)

            # Orchestrator-side eval
            bench = benchmark_kernel(child_kernel, self._config, workloads=workloads)
            child.score = compute_sol_score(
                baseline_bench.median_latency_us,
                bench.median_latency_us,
                roofline.t_sol_us,
            )

            # Reviewer
            feedback = await self._evaluator.review(
                kernel_source=new_source,
                profiling_summary="placeholder profiling summary",
                sol_score=child.score.sol_score,
                headroom_pct=(1.0 - child.score.sol_score) * 100,
                bottleneck=roofline.bottleneck.value,
            )
            child.branch_quality = feedback.branch_quality

            # Beam prune
            beam_prune(tree, self._config.beam_width, enable_diversity=self._config.beam_diversity)

            # Check move-on criteria
            if child.score.sol_score >= self._config.sol_target:
                return SearchResult(tree.best_node(), iteration + 1, "sol_target")

            best_scores.append(tree.best_node().score.sol_score)
            if detect_plateau(best_scores, self._config.sol_plateau_window, self._config.sol_plateau_delta):
                return SearchResult(tree.best_node(), iteration + 1, "plateau")

            epsilon = max(self._config.epsilon_end, epsilon - decay)

        return SearchResult(tree.best_node(), self._config.max_depth, "budget")
