"""Main pipeline entry point — Phase A (load + baseline), Phase B
(orchestrator search loop), and Phase C (report generation).

Import-order contract: ``import sol_execbench`` MUST be the first
non-stdlib import. SOL's ``core.bench.reward_hack`` snapshots
``torch.cuda.Event.elapsed_time`` at module-load time; if any candidate
kernel touches torch first, the snapshot records a possibly-tampered
address and ``check_monkey_patch`` would silently agree. By landing this
import here — at the entry point of the pipeline module — we guarantee
the snapshot is taken before any user-supplied source is loaded.
"""

from __future__ import annotations

# IMPORT-ORDER CONTRACT — DO NOT REORDER
# This must be the first non-stdlib import (the dataclass/typing/asyncio
# imports below are all stdlib). See module docstring.
import sol_execbench  # noqa: F401 — load-bearing side effects (_ELAPSED_TIME_ADDR snapshot)

import asyncio
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from sol_execbench.core.bench.clock_lock import (
    lock_clocks,
    probe_clock_lock_available,
    unlock_clocks,
    verify_clocks,
)
from sol_execbench.core.bench.config.device_config import get_clock_preset

from src.config import HardwareSpec

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition

    from src.agents.coder import CoderAgent
    from src.config import ACTSConfig
    from src.kernels.kernel import KernelSpec
    from src.search.orchestrator import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CONFIG_PATH = Path("configs/models/deepseek.json")

# Clock-lock lifecycle state. Module-global because ``main()`` can be
# called at most once per process and the ``atexit`` cleanup needs to see
# the same state the lock helpers wrote. ``locked`` is True only when
# both GPU and DRAM clocks were verified after locking; the cleanup is
# idempotent so a second call (atexit + finally) is harmless.
_clock_lock_state: dict[str, object] = {"locked": False, "device_name": ""}


def _unlock_clocks_safe() -> None:
    """Idempotent unlock for atexit + finally — swallows all exceptions.

    Called twice on a normal exit (once by the explicit ``finally`` in
    ``main()``, once by ``atexit``). The flag flip in the ``finally``
    block makes the second call a no-op.
    """
    if not _clock_lock_state["locked"]:
        return
    try:
        unlock_clocks()
    except Exception as exc:
        logger.error(
            "Failed to unlock GPU clocks for %s: %s. Manual recovery: nvidia-smi -rgc",
            _clock_lock_state["device_name"], exc,
        )
    finally:
        _clock_lock_state["locked"] = False


class UnknownBenchmarkFormat(RuntimeError):
    """Raised when ``_load_problem`` cannot determine the benchmark format
    of the supplied problem directory.

    Two paths reach this:
    1. ``ACTSConfig.benchmark_adapter`` was set to an unrecognized value;
    2. the directory has neither ``definition.json`` (SOL-ExecBench) nor
       ``model.py`` (KernelBench, future).
    """


# Stand-in HardwareSpec for the smoke path when detect_hardware() returns
# zeroed peaks. Mirrors the RTX 6000 Ada test fixture so the placeholder
# run produces representative roofline math.
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
) -> tuple[SearchResult, "OptimizationReport"]:
    """Run the full ACTS optimization pipeline.

    Phase A: Load problem, generate Triton baseline, derive T_SOL,
             benchmark baseline, select workloads.
    Phase B: Execute tree search loop via Orchestrator.

    *problem_path* is either:
      - A directory containing ``definition.json`` + ``workload.jsonl``
        (SOL-ExecBench mode).
      - The literal string ``"placeholder"`` for the built-in demo
        (matmul starter, no SOL problem data).

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
        from src.config import validate_hardware_spec

        for msg in validate_hardware_spec(_PLACEHOLDER_HARDWARE_SPEC, config.hardware):
            logger.warning("placeholder substitution: %s", msg)
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
            baseline, definition, workloads, roofline,
            reference_fn, input_generators, definition_path,
        ) = await _load_problem(problem_dir, config, coder)
    else:
        baseline, definition, workloads, roofline = _load_placeholder(config)
        reference_fn = None
        input_generators = []
        definition_path = None

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
    result = await orchestrator.run(
        baseline,
        workloads=workloads,
        roofline=roofline,
        reference_fn=reference_fn,
        input_generators=input_generators,
        problem_definition_path=definition_path,
        definition=definition,
    )

    from src.pipeline.report import generate_report

    arch_yaml_path = (
        Path(config.arch_config_path) if config.arch_config_path else None
    )
    blob_roots = config.safetensors_blob_roots or (
        [definition_path.parent] if definition_path is not None else None
    )
    report = generate_report(
        result,
        workloads=workloads,
        input_generators=input_generators,
        hardware_spec=config.hardware,
        definition=definition,
        definition_path=definition_path,
        blob_roots=blob_roots,
        arch_yaml_path=arch_yaml_path,
    )
    return result, report


async def _load_problem(
    problem_dir: Path,
    config: ACTSConfig,
    coder: CoderAgent,
) -> tuple:
    """Auto-detect benchmark format and dispatch to the matching adapter.

    Precedence:
      1. ``config.benchmark_adapter`` — explicit override. Values:
         ``"sol_execbench"``, ``"kernelbench"`` (NotImplementedError until
         that adapter ships).
      2. ``definition.json`` present → SOL-ExecBench adapter.
      3. ``model.py`` present → KernelBench (NotImplementedError).
      4. otherwise raise ``UnknownBenchmarkFormat``.

    Returns whatever the dispatched adapter returns — currently the SOL
    adapter's 7-tuple ``(baseline, definition, workloads, roofline,
    reference_fn, input_generators, definition_path)``.
    """
    if config.benchmark_adapter is not None:
        if config.benchmark_adapter == "sol_execbench":
            return await _load_sol_problem(problem_dir, config, coder)
        if config.benchmark_adapter == "kernelbench":
            raise NotImplementedError(
                "KernelBench adapter is a future phase — set "
                "ACTSConfig.benchmark_adapter to 'sol_execbench' or leave None."
            )
        raise UnknownBenchmarkFormat(
            f"Unknown benchmark_adapter: {config.benchmark_adapter!r} "
            f"(expected 'sol_execbench' or 'kernelbench')"
        )

    if (problem_dir / "definition.json").exists():
        return await _load_sol_problem(problem_dir, config, coder)

    if (problem_dir / "model.py").exists():
        raise NotImplementedError(
            "KernelBench adapter is a future phase — model.py was detected "
            "but no adapter is wired yet."
        )

    raise UnknownBenchmarkFormat(
        f"Cannot determine benchmark format for {problem_dir!r}: "
        f"no definition.json (SOL-ExecBench) and no model.py (KernelBench)"
    )


async def _load_sol_problem(
    problem_dir: Path,
    config: ACTSConfig,
    coder: CoderAgent,
) -> tuple:
    """Phase A for SOL-ExecBench problems.

    Returns ``(baseline, definition, workloads, roofline, reference_fn,
    input_generators, definition_path)``. The reference + generator list
    are forwarded to ``Orchestrator.run`` so Phase B's correctness tool
    binds to every selected workload. ``definition_path`` is the source
    ``definition.json`` the profiler subprocess driver reloads to
    reconstruct the (unpicklable) input generator.
    """
    from src.benchmark.baseline_generator import generate_triton_baseline
    from src.benchmark.workload_selector import select_workloads
    # ``load`` is re-exported as a function from
    # ``src/benchmarks/sol_execbench/__init__.py``; bind it directly so
    # we can call ``sol_load(problem_dir)`` below.
    from src.benchmarks.sol_execbench import load as sol_load
    from src.eval.inputs import build_input_generator, build_reference_fn
    from src.eval.roofline import derive_t_sol_from_solar

    definition, all_workloads = sol_load(problem_dir)
    definition_path = problem_dir / "definition.json"
    spec = _definition_to_kernel_spec(definition, definition_path)

    workloads = select_workloads(all_workloads, count=config.benchmark_workload_count)
    # Pick the median-size workload as the representative for SOLAR's
    # static roofline analysis — a startup-cost tradeoff. Re-deriving
    # per workload would re-run SOLAR's 4-stage pipeline N times during
    # Phase A; per-workload reporting is handled later in Phase C
    # (``report.generate_report`` calls ``derive_t_sol_from_solar`` on
    # each selected workload when populating
    # ``winner_per_workload_bottlenecks``).
    representative = workloads[len(workloads) // 2] if workloads else None
    arch_yaml_path = Path(config.arch_config_path) if config.arch_config_path else None
    roofline = (
        derive_t_sol_from_solar(
            definition, representative, config.hardware, arch_yaml_path=arch_yaml_path,
        )
        if representative is not None
        else None
    )
    if roofline is not None:
        spec.t_sol_us = roofline.t_sol_us

    blob_roots = config.safetensors_blob_roots or [problem_dir]
    baseline = await generate_triton_baseline(
        definition, spec,
        coder=coder,
        workloads=workloads,
        max_retries=config.max_baseline_retries,
        blob_roots=blob_roots,
    )

    reference_fn = build_reference_fn(definition.reference)
    input_generators = [
        build_input_generator(definition, w, blob_roots=blob_roots) for w in workloads
    ]

    return (
        baseline, definition, workloads, roofline,
        reference_fn, input_generators, definition_path,
    )


_OP_TYPE_TO_KERNEL_TYPE: dict[str, str] = {
    "gemm": "GEMM",
    "matmul": "MATMUL",
    "rmsnorm": "RMSNORM",
    "layernorm": "LAYERNORM",
    "softmax": "SOFTMAX",
    "gqa": "GQA",
    "gqa_ragged": "GQA",
    "gqa_paged": "GQA",
    "attention": "ATTENTION",
    "moe": "MOE",
    "moe_dispatch": "MOE",
    "embedding": "EMBEDDING",
    "rope": "EMBEDDING",
    "linear": "LINEAR",
    "mlp": "MLP",
    "swiglu": "MLP",
    "conv": "CONV",
    "ssm": "SSM",
    "mamba": "SSM",
}


def _definition_to_kernel_spec(definition: Definition, definition_path: Path) -> KernelSpec:
    """Build a ``KernelSpec`` from a SOL ``Definition`` + source path.

    FLOPs and memory_bytes are left at 0 — they are derived by SOLAR
    or by ``compute_roofline_inputs`` from the workload axes, not from
    the static definition. The spec carries the PyTorch reference and
    the definition path so downstream code (compile, profile-driver
    rehydration) can access them.
    """
    from src.kernels.kernel import KernelSpec, KernelType

    op_type = (definition.op_type or "").lower()
    kernel_type_name = _OP_TYPE_TO_KERNEL_TYPE.get(op_type, "CUSTOM")
    kernel_type = KernelType[kernel_type_name]

    return KernelSpec(
        name=definition.name,
        kernel_type=kernel_type,
        input_shapes=[dict(definition.const_axes)] if definition.const_axes else [],
        definition_path=definition_path,
        pytorch_reference=definition.reference,
    )


def _load_placeholder(config: ACTSConfig) -> tuple:
    """Phase A fallback — matmul starter, no SOL problem data."""
    _ = config  # accepted for caller-site symmetry with _load_problem; currently unused
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

    from src.pipeline.report import render_report
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

    # Clock-lock lifecycle. We register the atexit cleanup before the
    # explicit ``finally`` so abnormal exits (interpreter shutdown via
    # Ctrl-C, segfault recovery) still hit the unlock path. The
    # ``finally`` block calls the same idempotent helper for the normal
    # case so we don't depend on atexit ordering. The actual
    # ``probe + lock`` happens after ``run_start`` so the event log opens
    # with the canonical run boundary.
    atexit.register(_unlock_clocks_safe)

    model_configured = _is_model_configured()
    emit(
        "run_start",
        problem_path=str(args.problem_path),
        model_configured=model_configured,
    )
    _try_acquire_clock_lock()
    result = None
    report = None
    try:
        try:
            result, report = asyncio.run(optimize(args.problem_path))
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
        print(render_report(report))
        if ctx.trace_processor is not None and hasattr(ctx.trace_processor, "path"):
            print(f"\nLLM trace: {ctx.trace_processor.path}")
        if ctx.run_dir is not None:
            print(f"Run dir: {ctx.run_dir}")
    finally:
        _unlock_clocks_safe()
        ctx.close()


def _try_acquire_clock_lock() -> None:
    """Probe + acquire GPU clock lock for stable benchmark timing.

    Best-effort: if no GPU is available, sudo isn't configured, or the
    device isn't in SOL's preset table, log a warning + emit a
    ``clock_lock_unavailable`` event and continue. Never raises — clock
    locking is a stability optimization, not a correctness gate.
    """
    from src.runtime.events import emit

    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return
    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        return
    _clock_lock_state["device_name"] = device_name

    if not probe_clock_lock_available():
        logger.warning(
            "Clock-lock unavailable (no sudo or unsupported GPU); "
            "expect timing variance"
        )
        emit("clock_lock_unavailable", device=device_name)
        return

    try:
        locked = lock_clocks(device_name)
    except Exception as exc:  # never let a clock-lock hiccup kill the run
        logger.warning("lock_clocks failed for %s: %s", device_name, exc)
        emit("clock_lock_unavailable", device=device_name, reason=str(exc)[:120])
        return
    if not locked:
        logger.warning(
            "lock_clocks returned False for %s — preset missing or partial unlock",
            device_name,
        )
        emit("clock_lock_unavailable", device=device_name, reason="lock_failed")
        return

    # Verify the actual clocks landed where we asked. ``lock_clocks``
    # already calls ``verify_clocks`` internally and returns False on
    # mismatch, so this is a belt-and-braces second pass for visibility.
    preset = get_clock_preset(device_name)
    if preset is not None:
        try:
            verified = verify_clocks(preset.gpu_clk_mhz, preset.dram_clk_mhz)
            if not verified:
                logger.warning(
                    "verify_clocks reported drift on %s after lock", device_name,
                )
        except Exception as exc:
            logger.warning("verify_clocks raised: %s", exc)

    _clock_lock_state["locked"] = True
    logger.info("GPU clocks locked for %s", device_name)


if __name__ == "__main__":
    main()
