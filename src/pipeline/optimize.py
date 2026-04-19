"""Main search loop entry point — Phase A + B."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.coder import CoderAgent
    from src.config import ACTSConfig
    from src.search.orchestrator import SearchResult

DEFAULT_MODEL_CONFIG_PATH = Path("configs/models/deepseek.json")


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

    This is the main entry point: ``python -m src.pipeline.optimize``.
    An LLM is used when ``configs/models/<provider>.json`` exists (default
    path ``configs/models/deepseek.json``, overridable via
    ``ACTS_MODEL_CONFIG``); otherwise every agent runs in no-op mode and
    only the placeholder demo is exercised end-to-end.
    """
    from src.agents.coder import CoderAgent
    from src.agents.planner import PlannerAgent
    from src.agents.reviewer import ReviewerAgent
    from src.config import ACTSConfig, detect_hardware
    from src.memory.retriever import MemoryRetriever
    from src.memory.store import MemoryStore
    from src.search.orchestrator import Orchestrator

    if config is None:
        config = ACTSConfig(hardware=detect_hardware())

    # Gating the model load on SOL mode keeps the placeholder CLI runnable —
    # the placeholder baseline has no oracle, so a model-backed Coder would
    # raise ImplementationError on the first iteration.
    problem_dir = Path(problem_path)
    is_sol = problem_dir.is_dir() and (problem_dir / "definition.json").exists()

    model = _load_model_if_configured() if is_sol else None
    planner = PlannerAgent(model=model)
    coder = CoderAgent(model=model, config=config)
    reviewer = ReviewerAgent(model=model)

    if is_sol:
        (
            baseline, problem, workloads, roofline,
            reference_fn, input_generators,
        ) = await _load_sol_execbench(problem_dir, config, coder)
    else:
        baseline, problem, workloads, roofline = _load_placeholder(config)
        reference_fn = None
        input_generators = []

    store_path = Path("memory_store.json")
    store = MemoryStore(store_path)
    if store_path.exists():
        store.load()
    retriever = MemoryRetriever(store, top_k=config.optimization_memory_top_k)

    orchestrator = Orchestrator(
        config=config,
        planner=planner,
        coder=coder,
        reviewer=reviewer,
        retriever=retriever,
    )
    return await orchestrator.run(
        baseline,
        workloads=workloads,
        roofline=roofline,
        reference_fn=reference_fn,
        input_generators=input_generators,
    )


async def _load_sol_execbench(
    problem_dir: Path,
    config: ACTSConfig,
    coder: CoderAgent,
) -> tuple:
    """Phase A for SOL-ExecBench problems.

    Returns ``(baseline, problem, workloads, roofline, reference_fn,
    input_generators)``. The reference + generator list are forwarded to
    ``Orchestrator.run`` so Phase B's correctness tool binds to every
    selected workload.
    """
    from src.benchmark.baseline_generator import generate_triton_baseline
    from src.benchmark.problem_loader import load_problem, problem_to_kernel_spec
    from src.benchmark.workload_selector import select_workloads
    from src.eval.inputs import build_input_generator, build_reference_fn
    from src.eval.roofline import derive_t_sol_from_solar

    problem = load_problem(problem_dir)
    spec = problem_to_kernel_spec(problem)

    roofline = derive_t_sol_from_solar(problem)
    if roofline is not None:
        spec.t_sol_us = roofline.t_sol_us

    workloads = select_workloads(problem.workloads, count=config.benchmark_workload_count)

    baseline = await generate_triton_baseline(
        problem, spec,
        coder=coder,
        workloads=workloads,
        max_retries=config.max_baseline_retries,
    )

    reference_fn = build_reference_fn(problem.reference_source)
    input_generators = [build_input_generator(problem, w) for w in workloads]

    return baseline, problem, workloads, roofline, reference_fn, input_generators


def _load_placeholder(config: ACTSConfig) -> tuple:
    """Phase A fallback — matmul starter, no SOL-ExecBench dependency."""
    from src.kernels.starters.matmul import make_matmul_kernel

    baseline = make_matmul_kernel(1024, 1024, 1024)
    return baseline, None, None, None


def _load_model_if_configured():
    """Load the LLM model from ``$ACTS_MODEL_CONFIG`` or the default path.

    Returns ``None`` when the file is absent or the Agents SDK is not
    installed, so every agent stays in no-op mode.
    """
    from src.agents.llm_backend import _SDK_AVAILABLE, create_model, load_model_config

    if not _SDK_AVAILABLE:
        return None
    path = Path(os.environ.get("ACTS_MODEL_CONFIG", str(DEFAULT_MODEL_CONFIG_PATH)))
    try:
        config = load_model_config(path)
    except FileNotFoundError:
        return None
    return create_model(config)


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
