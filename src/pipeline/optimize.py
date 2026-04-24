"""Main search loop entry point — Phase A + B."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import HardwareSpec

if TYPE_CHECKING:
    from src.agents.coder import CoderAgent
    from src.config import ACTSConfig
    from src.search.orchestrator import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CONFIG_PATH = Path("configs/models/deepseek.json")

# Populated stand-in used when ``detect_hardware()`` still returns a zeroed
# spec (the placeholder path has no arch YAML on a dev machine). The
# orchestrator's profiler guard rejects zero peaks, so without this the
# default ``python -m src.pipeline.optimize`` smoke run would die on the
# first iteration. Values mirror ``_rtx6000_ada()`` in the Tier 1/2 test
# fixtures so the placeholder run produces representative roofline math.
_PLACEHOLDER_HARDWARE_SPEC = HardwareSpec(
    name="placeholder-RTX6000Ada",
    freq_GHz=2.5,
    SRAM_capacity=98_304 * 1024,
    SRAM_byte_per_cycle=4000.0,
    DRAM_capacity=48 * 1024**3,
    DRAM_byte_per_cycle=384.0,
    MAC_per_cycle_fp32_sm=12_800.0,
    MAC_per_cycle_fp16_tc=512_000.0,
    MAC_per_cycle_bf16_tc=512_000.0,
)


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
    if (
        config.hardware.peak_flops_fp32 <= 0
        or config.hardware.peak_memory_bandwidth_gb_s <= 0
    ):
        logger.warning(
            "HardwareSpec has zero peaks (name=%r) — substituting a populated "
            "placeholder (%s) so the orchestrator's profiler guard passes. "
            "Load a SOLAR arch YAML for real runs.",
            config.hardware.name,
            _PLACEHOLDER_HARDWARE_SPEC.name,
        )
        config = replace(config, hardware=_PLACEHOLDER_HARDWARE_SPEC)

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
        problem_definition_path=(problem.definition_path if problem is not None else None),
        problem=problem,
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


def _is_model_configured() -> bool:
    """True when the LLM model config file is present on disk and the SDK
    is importable. Used by ``main()`` to populate ``run_start.model_configured``
    before ``optimize()`` loads the model itself.
    """
    from src.agents.llm_backend import _SDK_AVAILABLE

    if not _SDK_AVAILABLE:
        return False
    path = Path(os.environ.get("ACTS_MODEL_CONFIG", str(DEFAULT_MODEL_CONFIG_PATH)))
    return path.exists()


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


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    ``argv`` is exposed so unit tests can drive ``main()`` without
    monkey-patching ``sys.argv`` — production callers (the ``__main__``
    block) pass ``None`` and argparse reads ``sys.argv[1:]``.
    """
    import argparse
    import atexit
    from datetime import datetime, timezone

    from src.pipeline.report import generate_report, render_report
    from src.runtime.events import emit
    from src.runtime.run_context import RunContext

    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.optimize",
        description=(
            "Run the ACTS optimization pipeline against a SOL-ExecBench problem "
            "directory (containing ``definition.json`` + ``workload.jsonl``), "
            "or the literal string ``placeholder`` for the matmul demo."
        ),
    )
    parser.add_argument(
        "problem_path",
        nargs="?",
        default="placeholder",
        help=(
            "Path to a SOL-ExecBench problem directory, or ``placeholder`` "
            "(default) to exercise the no-LLM matmul smoke path."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("./runs"),
        help=(
            "Root directory for per-invocation run_<UTC>/ subdirectories "
            "(each containing run.log, events.jsonl, traces/). Defaults to ./runs."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=str,  # str so the empty-string kill-switch is preserved verbatim
        default=None,
        help=(
            "Override directory for SDK trace JSONL files. When omitted, "
            "traces land under <run-dir>/<run_UTC>/traces/. Pass an empty "
            "value (``--trace-dir=``) to disable capture entirely."
        ),
    )
    args = parser.parse_args(argv)

    # Trace-dir tri-state mapped onto RunContext: ``--trace-dir=`` is
    # the kill switch; any other value (or its absence) is handed to
    # RunContext which decides default-under-run-dir vs explicit override.
    ctx = RunContext.create(
        root=args.run_dir,
        trace_dir=args.trace_dir if args.trace_dir else None,
        capture_traces=args.trace_dir != "",
    )
    atexit.register(ctx.close)

    model_configured = _is_model_configured()
    emit(
        "run_start",
        problem_path=str(args.problem_path),
        model_configured=model_configured,
    )
    result = None
    try:
        try:
            result = asyncio.run(optimize(args.problem_path))
        except Exception:
            emit(
                "run_end",
                termination_reason="ERROR",
                best_score=0.0,
                total_iterations=0,
                wallclock_s=round(
                    (datetime.now(timezone.utc) - ctx.started_at).total_seconds(), 3
                ),
            )
            raise
        best_score_val = (
            result.best_node.score.sol_score
            if result.best_node is not None and result.best_node.score is not None
            else 0.0
        )
        emit(
            "run_end",
            termination_reason=result.termination_reason.value,
            best_score=best_score_val,
            total_iterations=result.total_iterations,
            wallclock_s=round(
                (datetime.now(timezone.utc) - ctx.started_at).total_seconds(), 3
            ),
        )
        print(render_report(generate_report(result)))
        if ctx.trace_processor is not None and hasattr(ctx.trace_processor, "path"):
            print(f"\nLLM trace: {ctx.trace_processor.path}")
        if ctx.run_dir is not None:
            print(f"Run dir: {ctx.run_dir}")
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
