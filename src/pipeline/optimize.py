"""Main search loop entry point — Phase A + B."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import ACTSConfig
    from src.search.orchestrator import SearchResult


async def optimize(
    problem_path: str,
    config: ACTSConfig | None = None,
) -> SearchResult:
    """Run the full ACTS optimization pipeline.

    Phase A: Load problem, generate Triton baseline, derive T_SOL,
             benchmark baseline, select workloads.
    Phase B: Execute tree search loop via Orchestrator.

    *problem_path* is either:
      - A directory containing ``definition.json`` + ``workload.jsonl``
        (SOL-ExecBench mode).
      - The literal string ``"placeholder"`` for the built-in demo
        (matmul starter, no SOL-ExecBench dependency).

    This is the main entry point:  ``python -m src.pipeline.optimize``
    """
    from src.agents.coder import CoderAgent
    from src.agents.evaluator import EvaluatorAgent
    from src.agents.planner import PlannerAgent
    from src.config import ACTSConfig, detect_hardware
    from src.memory.retriever import MemoryRetriever
    from src.memory.store import MemoryStore
    from src.search.orchestrator import Orchestrator

    if config is None:
        config = ACTSConfig(hardware=detect_hardware())

    # Phase A: load problem
    problem_dir = Path(problem_path)
    if problem_dir.is_dir() and (problem_dir / "definition.json").exists():
        baseline, problem, workloads, roofline = _load_sol_execbench(problem_dir, config)
    else:
        baseline, problem, workloads, roofline = _load_placeholder(config)

    # Set up memory
    store_path = Path("memory_store.json")
    store = MemoryStore(store_path)
    if store_path.exists():
        store.load()
    retriever = MemoryRetriever(store, top_k=config.optimization_memory_top_k)

    # Set up agents (placeholder mode — no real model configured)
    planner = PlannerAgent()
    coder = CoderAgent()
    evaluator = EvaluatorAgent()

    # Run search
    orchestrator = Orchestrator(
        config=config,
        planner=planner,
        coder=coder,
        evaluator=evaluator,
        retriever=retriever,
    )
    return await orchestrator.run(baseline, workloads=workloads, roofline=roofline)


def _load_sol_execbench(
    problem_dir: Path,
    config: ACTSConfig,
) -> tuple:
    """Phase A for SOL-ExecBench problems."""
    from src.benchmark.baseline_generator import generate_triton_baseline
    from src.benchmark.problem_loader import load_problem, problem_to_kernel_spec
    from src.benchmark.workload_selector import select_workloads
    from src.eval.roofline import derive_t_sol_from_solar
    from src.kernels.kernel import Kernel

    problem = load_problem(problem_dir)
    spec = problem_to_kernel_spec(problem)

    # Derive T_SOL + bottleneck via SOLAR (returns None if SOLAR not installed)
    roofline = derive_t_sol_from_solar(problem)
    if roofline is not None:
        spec.t_sol_us = roofline.t_sol_us

    # Triton baseline (placeholder — real impl needs async Coder call)
    baseline = Kernel(spec=spec, source_code="# placeholder baseline")

    # Select representative workloads
    workloads = select_workloads(problem.workloads, count=config.benchmark_workload_count)

    return baseline, problem, workloads, roofline


def _load_placeholder(config: ACTSConfig) -> tuple:
    """Phase A fallback — matmul starter, no SOL-ExecBench dependency."""
    from src.kernels.starters.matmul import make_matmul_kernel

    baseline = make_matmul_kernel(1024, 1024, 1024)
    return baseline, None, None, None


def main() -> None:
    """CLI entry point."""
    result = asyncio.run(optimize("placeholder"))
    print(f"Search completed: {result.termination_reason}")
    print(f"  Iterations: {result.total_iterations}")
    if result.best_node.score:
        print(f"  Best SOL score: {result.best_node.score.sol_score:.4f}")
        print(f"  Speedup: {result.best_node.score.speedup:.2f}x")


if __name__ == "__main__":
    main()
