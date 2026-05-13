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
# The cfg preparse + ``CUDA_VISIBLE_DEVICES`` override below must run
# before any non-stdlib import that touches CUDA. SOL's import chain
# transitively imports torch, which snapshots the visible-device set on
# first CUDA call; setting the env after that is a no-op. ``io``, ``os``,
# and ``sys`` are stdlib so importing them here is safe. ``libconf`` is
# pure-Python (no CUDA / torch / SOL imports) so importing it before
# ``import sol_execbench`` is also safe.
import io
import os
import sys

import libconf


def _preparse_config_path(argv: list[str]) -> str | None:
    """Minimal argv scan for ``--config <path>``.

    Returns the path string when ``--config <path>`` or ``--config=<path>``
    is present, ``None`` otherwise. Does not check that the file exists —
    argparse in ``main()`` handles bad-path errors with a clean message.
    A dangling ``--config`` (no following arg) is treated as absent.
    """
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _preparse_gpu_index(argv: list[str]) -> str:
    """Resolve the effective ``gpu_index`` before any CUDA-aware import.

    Reads ``hardware.gpu_index`` from the libconfig-format cfg path
    scanned out of argv. Returns the value as a string (the form
    ``CUDA_VISIBLE_DEVICES`` and ``nvidia-smi -i <idx>`` consume).
    Defaults to ``"0"`` when ``--config`` is absent, the cfg path doesn't
    exist, the section/key is missing, or the cfg is malformed. Never
    raises — argparse in ``main()`` surfaces user-facing errors later.
    """
    cfg_path = _preparse_config_path(argv)
    if cfg_path is None:
        return "0"
    if not os.path.exists(cfg_path):
        return "0"
    try:
        with io.open(cfg_path, encoding="utf-8") as f:
            cfg = libconf.load(f)
        return str(int((cfg.get("hardware") or {}).get("gpu_index", 0)))
    except (libconf.ConfigParseError, ValueError, OSError):
        return "0"


def _validate_gpu_visible(gpu_index: int, *, reset_only: bool) -> None:
    """Tier 2 existence check after ``CUDA_VISIBLE_DEVICES`` has been set.

    On failure: prints a single explanatory line to stderr and calls
    ``sys.exit(1)``. Logger isn't set up yet at this call point — errors
    go directly to stderr.

    ``reset_only=True`` (operator recovery path via ``--reset-clocks``)
    skips the torch.cuda checks and uses ``nvidia-smi --list-gpus -i N``
    instead — recovery shouldn't require torch to be importable.
    """
    if reset_only:
        rc = subprocess.run(
            ["nvidia-smi", "--list-gpus", "-i", str(gpu_index)],
            capture_output=True,
        ).returncode
        if rc != 0:
            print(
                f"GPU {gpu_index} not found by nvidia-smi "
                f"(--list-gpus -i {gpu_index} returned {rc})",
                file=sys.stderr,
            )
            sys.exit(1)
        return

    import torch
    if not torch.cuda.is_available():
        print(
            "ACTS requires a CUDA-capable PyTorch and a visible GPU; "
            "none detected. Check that you're in the Tier 2 venv "
            "(acts_run_venv) and that nvidia-smi works.",
            file=sys.stderr,
        )
        sys.exit(1)
    n = torch.cuda.device_count()
    if n == 0:
        print(
            f"GPU {gpu_index} not found. CUDA_VISIBLE_DEVICES={gpu_index} "
            f"filtered to zero visible devices — index is out of range "
            f"for this host.",
            file=sys.stderr,
        )
        sys.exit(1)
    if n > 1:
        print(
            f"unexpected: {n} visible CUDA devices after env override "
            f"(should be 1). Env-handling bug — please file.",
            file=sys.stderr,
        )
        sys.exit(1)


_GPU_INDEX = _preparse_gpu_index(sys.argv)
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU_INDEX

import sol_execbench  # noqa: F401 — load-bearing side effects (_ELAPSED_TIME_ADDR snapshot)

import asyncio
import logging
import os
import signal
import subprocess
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from sol_execbench.core.bench.clock_lock import probe_clock_lock_available
from sol_execbench.core.bench.config.device_config import ClockPreset, get_clock_preset

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
# idempotent so a second call (atexit + finally) is harmless. ``ctx``
# holds the live ``RunContext`` so the SIGTERM/SIGHUP handler can flush
# events.jsonl + SDK traces before the process dies — set by ``main()``
# after ``RunContext.create``; cleared by ``_close_ctx_safe`` to make
# the helper idempotent across signal-then-finally paths.
_clock_lock_state: dict[str, object] = {
    "locked": False, "device_name": "", "ctx": None,
}

# Every ``nvidia-smi`` invocation is scoped to ``-i $_GPU_INDEX``. Without
# ``-i <idx>`` clock locks apply to *all* GPUs on the host. The value of
# ``_GPU_INDEX`` is set at module top from the ``--gpu-index`` CLI flag
# (default ``"0"``); see the import-order block.


def _nvidia_smi(
    *args: str, capture_output: bool = True,
) -> subprocess.CompletedProcess[str] | None:
    """Run ``sudo -n nvidia-smi <args> -i _GPU_INDEX``, capturing output.

    Returns the ``CompletedProcess`` on success (return code 0) or
    ``None`` on any failure (``CalledProcessError``, ``FileNotFoundError``,
    subprocess raised). Centralizes the GPU-0 scoping suffix and the
    failure-mode swallowing so individual call sites stay short.
    """
    cmd = ["sudo", "-n", "nvidia-smi", *args, "-i", _GPU_INDEX]
    try:
        return subprocess.run(
            cmd, check=True, capture_output=capture_output, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class ClockLockReason(StrEnum):
    """Closed-vocabulary reasons emitted by ``clock_lock_unavailable``.

    Exception-derived reasons stay free-form (e.g. ``f"verify_raised:{exc}"``);
    ``_emit_clock_unavailable`` accepts either form.
    """

    OK = "ok"
    PROBE_RETURNED_FALSE = "probe_returned_false"
    NO_PRESET = "no_preset"
    LOCK_FAILED = "lock_failed"
    VERIFY_FAILED = "verify_failed"
    UNKNOWN = "unknown"


def _emit_clock_unavailable(device: str, reason: ClockLockReason | str) -> None:
    """Centralizes the ``clock_lock_unavailable`` event emit shape.

    Accepts either a closed-vocabulary ``ClockLockReason`` or a free-form
    string (for exception-derived reasons like ``"verify_raised:..."`` or
    ``"lock_raised:<exc>"``).
    """
    from src.runtime.events import emit

    emit("clock_lock_unavailable", device=device, reason=str(reason))


def _rollback_partial_lock(device_name: str, reason: ClockLockReason | str) -> None:
    """Roll back a partially-acquired GPU clock lock and emit the
    unavailability event.

    Used on the verify-failure paths (verify returned False, or verify
    raised) so the rollback shape doesn't diverge.
    """
    try:
        _unlock_gpu0_clocks()
    except Exception as exc:
        logger.warning(
            "Rollback _unlock_gpu0_clocks failed for %s: %s. Manual "
            "recovery: python -m src.pipeline.optimize --reset-clocks",
            device_name, exc,
        )
    _emit_clock_unavailable(device_name, reason)

# ACTS-side clock preset table — consulted before SOL's. Lets ACTS lock
# clocks on cards SOL doesn't ship presets for (workstation + Pro cards
# like Ada). Substring match against device_name; first hit wins.
#
# Value choice for RTX 6000 Ada: 2505 MHz GPU (the design-boost target
# from the datasheet) and 10001 MHz memory (the only supported memory
# clock per nvidia-smi -q -d SUPPORTED_CLOCKS on this card). 2505 also
# matches configs/arch/RTX6000Ada.yaml's freq_GHz=2.505 so SOLAR T_SOL
# stays calibrated against the locked clock.
#
# This is *more aggressive* than SOL's philosophy (SOL pins A100 at
# 1065 / boost 1410 ≈ 75% of boost, prioritizing thermal stability over
# perf). For workstation cards on a single-tenant dev box without
# sustained 100%-utilization workloads, design-boost is reasonable.
# Future entries that target multi-tenant or sustained-load envs may
# want SOL-style conservative values.
_ACTS_CLOCK_PRESETS: dict[str, ClockPreset] = {
    "RTX 6000 Ada": ClockPreset(gpu_clk_mhz=2505, dram_clk_mhz=10001),
}


def _resolve_clock_preset(device_name: str) -> ClockPreset | None:
    """ACTS-first preset lookup with SOL fallback.

    Checks ACTS's table first (covers workstation + Pro cards SOL omits).
    If no match, defers to SOL's ``get_clock_preset`` (datacenter-focused:
    B200 / H100 / A100). Returns None if neither has a preset, in which
    case ``_try_acquire_clock_lock`` emits clock_lock_unavailable with
    reason="no_preset" and the run continues with unlocked clocks.

    Substring match against device_name to mirror SOL's contract.
    """
    for key, preset in _ACTS_CLOCK_PRESETS.items():
        if key in device_name:
            return preset
    return get_clock_preset(device_name)


def _lock_gpu0_clocks(gpu_mhz: int, dram_mhz: int) -> bool:
    """Lock GPU + DRAM clocks on GPU 0 only.

    Returns True on success, False on any failure (preserving the
    return contract of SOL's ``lock_clocks``). Logs warnings via the
    module logger. If the DRAM lock fails after the GPU lock landed,
    rolls back the GPU lock so we don't leave a partial-lock state.
    """
    if _nvidia_smi("-lgc", f"{gpu_mhz},{gpu_mhz}") is None:
        logger.warning("Failed to lock GPU clocks on GPU %s", _GPU_INDEX)
        return False
    if _nvidia_smi("-lmc", f"{dram_mhz},{dram_mhz}") is None:
        logger.warning("Failed to lock DRAM clocks on GPU %s", _GPU_INDEX)
        # Roll back the GPU-clock lock so we don't leave a partial-lock state.
        _nvidia_smi("-rgc")
        return False
    return True


def _verify_gpu0_locked(
    expected_gpu_mhz: int, expected_dram_mhz: int, tolerance_mhz: int = 50,
) -> bool:
    """Verify GPU 0 is locked at the requested clock targets.

    Reads ``clocks.applications.{graphics,memory}`` (the field
    ``-lgc``/``-lmc`` write — reflects the lock target whether the GPU
    is busy or idle), scoped to GPU 0. Returns False on any subprocess
    or parse failure so the caller rolls back the partial lock.
    """
    result = _nvidia_smi(
        "--query-gpu=clocks.applications.graphics,clocks.applications.memory",
        "--format=csv,noheader,nounits",
    )
    if result is None:
        logger.warning("nvidia-smi clock query failed for GPU %s", _GPU_INDEX)
        return False

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = line.split(",")
    if len(parts) < 2:
        logger.warning("Unexpected nvidia-smi output for GPU %s: %r", _GPU_INDEX, line)
        return False
    try:
        actual_gpu = int(parts[0].strip())
        actual_dram = int(parts[1].strip())
    except ValueError:
        logger.warning("Could not parse nvidia-smi output for GPU %s: %r", _GPU_INDEX, line)
        return False

    if abs(actual_gpu - expected_gpu_mhz) > tolerance_mhz:
        logger.warning(
            "GPU %s graphics-clock mismatch — expected %d MHz, got %d MHz",
            _GPU_INDEX, expected_gpu_mhz, actual_gpu,
        )
        return False
    if abs(actual_dram - expected_dram_mhz) > tolerance_mhz:
        logger.warning(
            "GPU %s memory-clock mismatch — expected %d MHz, got %d MHz",
            _GPU_INDEX, expected_dram_mhz, actual_dram,
        )
        return False
    return True


def _unlock_gpu0_clocks() -> None:
    """Unlock GPU + DRAM clocks on GPU 0 only.

    Idempotent at the nvidia-smi level — calling -rgc / -rmc on an
    already-unlocked GPU is harmless. Used both by the normal cleanup
    path and by the ``--reset-clocks`` operator escape hatch.
    """
    _nvidia_smi("-rgc")
    _nvidia_smi("-rmc")


def _unlock_clocks_safe() -> None:
    """Idempotent unlock for atexit + finally — swallows all exceptions.

    Called twice on a normal exit (once by the explicit ``finally`` in
    ``main()``, once by ``atexit``). The flag flip in the ``finally``
    block makes the second call a no-op.
    """
    if not _clock_lock_state["locked"]:
        return
    try:
        _unlock_gpu0_clocks()
    except Exception as exc:
        logger.error(
            "Failed to unlock GPU clocks for %s: %s. Manual recovery: "
            "python -m src.pipeline.optimize --reset-clocks "
            "(or: sudo nvidia-smi -rgc -i 0 && sudo nvidia-smi -rmc -i 0)",
            _clock_lock_state["device_name"], exc,
        )
    finally:
        _clock_lock_state["locked"] = False


def _close_ctx_safe() -> None:
    """Idempotent RunContext close for signal + finally — swallows all exceptions.

    On SIGTERM/SIGHUP neither ``atexit`` nor ``main()``'s ``finally``
    runs, so without this the final batches of ``events.jsonl`` and SDK
    traces never flush. ``RunContext.close()`` is itself idempotent (it
    checks ``self._closed``); we also clear the slot so the mock-count
    assertion in tests stays at 1 across repeated calls.
    """
    ctx = _clock_lock_state.get("ctx")
    if ctx is None:
        return
    try:
        ctx.close()
    except Exception as exc:
        logger.error("RunContext.close() failed during cleanup: %s", exc)
    finally:
        _clock_lock_state["ctx"] = None


def _signal_unlock_handler(signum: int, frame) -> None:  # noqa: ARG001 — frame required by signal protocol
    """Best-effort unlock on SIGTERM / SIGHUP, then propagate the signal.

    ``atexit`` does NOT fire on uncaught signals (SIGTERM from
    ``kill <pid>`` / systemd, SIGHUP from SSH session drop), so we'd
    otherwise leak a clock lock and skip the RunContext flush. Unlock
    clocks first (faster + more critical for shared-GPU hosts), then
    close the ctx so events.jsonl + SDK traces flush. Finally restore
    the default disposition for *signum* and re-raise it via ``os.kill``
    rather than calling ``sys.exit`` — that preserves the conventional
    signal-class exit code (128 + signum) for shells / process
    supervisors.

    Not registered for SIGINT: KeyboardInterrupt → ``atexit`` already
    cleans up Ctrl-C cleanly, and a custom handler would interfere.
    """
    _unlock_clocks_safe()
    _close_ctx_safe()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _finalize_run_safe(ctx) -> None:
    """End-of-run cleanup: unlock clocks, close ctx. Idempotent / never raises.

    Called from ``main()``'s ``finally`` block. Unlock first (faster +
    more critical for shared-GPU hosts with persistence mode), then
    flush the RunContext (events.jsonl, SDK traces). ``ctx.close()`` is
    wrapped because the finally is bottom-of-stack and must not mask
    the original run-exit cause.
    """
    _unlock_clocks_safe()
    try:
        ctx.close()
    except Exception as exc:
        logger.error("RunContext.close() failed during finalize: %s", exc)


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
    freq_GHz=2.505,
    SRAM_capacity=100_663_296,        # 96 MiB L2
    SRAM_byte_per_cycle=2200.0,       # ~5.5 TB/s @ 2.505 GHz
    DRAM_capacity=51_539_607_552,     # 48 GiB GDDR6 ECC
    DRAM_byte_per_cycle=383.0,        # 960 GB/s / 2.505 GHz
    MAC_per_cycle_fp32_sm=18_185.0,   # 91.1 TFLOPS
    MAC_per_cycle_fp16_tc=72_695.0,   # 364.2 TFLOPS dense
    MAC_per_cycle_bf16_tc=72_695.0,   # 364.2 TFLOPS dense
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
    only the placeholder demo is exercised end-to-end. The API key may
    live in the JSON or in ``$OPENAI_API_KEY`` (see
    ``llm_backend.load_model_config``).
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
    from src.benchmark.solar_adapter import is_solar_available
    from src.benchmark.workload_selector import select_workloads
    # ``load`` is re-exported as a function from
    # ``src/benchmarks/sol_execbench/__init__.py``; bind it directly so
    # we can call ``sol_load(problem_dir)`` below.
    from src.benchmarks.sol_execbench import load as sol_load
    from src.eval.inputs import build_input_generator, build_reference_fn
    from src.eval.roofline import derive_t_sol_from_solar

    # Fail-fast: SOL-ExecBench problems leave ``KernelSpec.flop_count`` /
    # ``memory_bytes`` at zero — the per-workload roofline math is
    # populated later via ``compute_roofline_inputs(definition, workload)``.
    # That means the orchestrator's built-in ``compute_roofline(spec, hw)``
    # fallback (which fires when ``derive_t_sol_from_solar`` returns
    # ``None``) silently produces ``t_sol_us=0.0``, corrupting every SOL
    # score with no visible diagnostic. Rather than papering over a
    # broken installation at score time, refuse to load the problem so
    # the operator sees the actionable install hint immediately.
    if not is_solar_available():
        raise RuntimeError(
            "SOLAR is required for SOL-ExecBench problems but is not "
            "importable. T_SOL would silently fall back to 0.0 (zero "
            "flop_count / memory_bytes on SOL kernel specs), corrupting "
            "every score. Install SOLAR + torchview into the run venv:\n"
            "    VIRTUAL_ENV=~/.venvs/acts_run_venv uv pip install torchview\n"
            "    VIRTUAL_ENV=~/.venvs/acts_run_venv uv pip install \\\n"
            "        -e /path/to/SOLAR --no-deps\n"
            "See configs/venvs/3.12.md for the canonical recipe."
        )

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
    import json
    from dataclasses import asdict
    from datetime import datetime, timezone

    from src.config import ACTSConfig, detect_hardware, load_config
    from src.pipeline.report import render_report
    from src.runtime import tree_dump
    from src.runtime.events import emit
    from src.runtime.run_context import RunContext

    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.optimize",
        description=(
            "Run the ACTS optimization pipeline. All algorithmic + runtime "
            "knobs (problem path, gpu_index, reset_clocks, beam_width, …) "
            "live in a ``.cfg`` file passed via ``--config``; the CLI keeps "
            "only invocation-scoped flags. Without ``--config``, ACTSConfig "
            "defaults run the placeholder matmul smoke path."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to an ACTS ``.cfg`` file (see configs/example.cfg). "
            "When omitted, ACTSConfig defaults apply (problem=placeholder, "
            "gpu_index=0, reset_clocks=False). The module-top preparse "
            "reads ``[hardware] gpu_index`` from this file before any CUDA-"
            "aware import so ``CUDA_VISIBLE_DEVICES`` lands in time."
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

    # Build ACTSConfig: load from --config if given, else dataclass defaults
    # with detected hardware. Argparse already rejected --gpu-index /
    # --reset-clocks / positional problem_path — those live in cfg now.
    if args.config is not None:
        if not args.config.exists():
            parser.error(f"--config: file not found: {args.config}")
        acts_config = load_config(args.config)
    else:
        acts_config = ACTSConfig(hardware=detect_hardware())

    # Tripwire: the module-top preparse read [hardware] gpu_index from the
    # same cfg, so the two must agree. A mismatch means the cfg changed
    # between import time and main() — a deployment bug, not a user error.
    assert acts_config.gpu_index == int(_GPU_INDEX), (
        f"preparse/config desync: preparse={_GPU_INDEX!r} "
        f"config={acts_config.gpu_index}"
    )
    _validate_gpu_visible(acts_config.gpu_index, reset_only=acts_config.reset_clocks)

    # Reset-clocks short-circuit: skip the entire pipeline (no RunContext,
    # no model load, no optimize call) and just clear any sticky lock that
    # a prior crashed run left behind. Toggle via cfg ``[runtime] reset_clocks``.
    if acts_config.reset_clocks:
        _unlock_gpu0_clocks()
        print(f"GPU {_GPU_INDEX} clocks reset.")
        return

    # Trace-dir tri-state mapped onto RunContext: ``--trace-dir=`` is
    # the kill switch; any other value (or its absence) is handed to
    # RunContext which decides default-under-run-dir vs explicit override.
    ctx = RunContext.create(
        root=args.run_dir,
        trace_dir=args.trace_dir if args.trace_dir else None,
        capture_traces=args.trace_dir != "",
    )
    # Slot the ctx for ``_close_ctx_safe`` before signal handlers install
    # so SIGTERM/SIGHUP between this line and the first ``emit`` still
    # flushes (the handler is harmless if no events have been written).
    _clock_lock_state["ctx"] = ctx
    atexit.register(ctx.close)

    # Clock-lock lifecycle. We register the atexit cleanup before the
    # explicit ``finally`` so abnormal exits (interpreter shutdown via
    # Ctrl-C, segfault recovery) still hit the unlock path. The
    # ``finally`` block calls the same idempotent helper for the normal
    # case so we don't depend on atexit ordering. The actual
    # ``probe + lock`` happens after ``run_start`` so the event log opens
    # with the canonical run boundary.
    atexit.register(_unlock_clocks_safe)
    # ``atexit`` does NOT fire on SIGTERM (kill <pid>, systemd) or SIGHUP
    # (SSH session drop) — install best-effort signal handlers so those
    # paths also unlock + flush the ctx before the process dies. SIGINT
    # is intentionally not handled: KeyboardInterrupt → atexit already
    # covers Ctrl-C.
    signal.signal(signal.SIGTERM, _signal_unlock_handler)
    signal.signal(signal.SIGHUP, _signal_unlock_handler)

    model_configured = _is_model_configured()
    emit(
        "run_start",
        problem_path=acts_config.problem_path,
        model_configured=model_configured,
    )
    _try_acquire_clock_lock()
    # ``acts_config`` is the cfg-resolved instance built before clock
    # lock; reuse it as both the optimize() input and the report.txt
    # dump payload below.
    result = None
    report = None
    try:
        try:
            result, report = asyncio.run(
                optimize(acts_config.problem_path, config=acts_config)
            )
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
        # End-of-run tree dump: writes <run_dir>/tree/{index.json,
        # tree.txt, tree.dot, tree.mmd}. Must run before ``ctx.close()``
        # so the ``tree_dump.bind(...)`` is still live. Never raises
        # (tree_dump.finalize_tree swallows OSError); no-op when unbound.
        tree_dump.finalize_tree(result.tree)
        rendered_report = render_report(report)
        # Config dump appended to the persisted report only — keeps the
        # terminal print focused on results. ``default=str`` coerces
        # Path / enum values that aren't JSON-native.
        config_dump = (
            "\n\n=== ACTSConfig (resolved at run start) ===\n"
            + json.dumps(asdict(acts_config), default=str, indent=2)
            + "\n"
        )
        if ctx.run_dir is not None:
            try:
                (ctx.run_dir / "report.txt").write_text(
                    rendered_report + config_dump
                )
            except OSError as exc:
                logger.warning("report.txt write failed: %s", exc)
        print(rendered_report)
        if ctx.trace_processor is not None and hasattr(ctx.trace_processor, "path"):
            print(f"\nLLM trace: {ctx.trace_processor.path}")
        if ctx.run_dir is not None:
            print(f"Run dir: {ctx.run_dir}")
    finally:
        _finalize_run_safe(ctx)


def _try_acquire_clock_lock() -> None:
    """Probe + acquire GPU clock lock for stable benchmark timing.

    Best-effort: if no GPU is available, sudo isn't configured, or the
    device isn't in SOL's preset table, log a warning + emit a
    ``clock_lock_unavailable`` event and continue. Never raises — clock
    locking is a stability optimization, not a correctness gate.
    """
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

    # SOL's ``probe_clock_lock_available()`` currently returns a bare ``bool``
    # (see SOL-ExecBench's ``core/bench/clock_lock.py``), but its docstring
    # hints a ``(bool, str)`` tuple variant may land upstream. Handle both
    # shapes defensively so a future SOL bump doesn't silently break here,
    # and synthesize a ``reason`` string when the bare-bool form gives us
    # nothing — the event log needs *why* the probe failed.
    probe_result = probe_clock_lock_available()
    if isinstance(probe_result, tuple):
        ok = bool(probe_result[0])
        reason: ClockLockReason | str = (
            str(probe_result[1]) if len(probe_result) > 1 else ClockLockReason.UNKNOWN
        )
    else:
        ok = bool(probe_result)
        reason = ClockLockReason.OK if ok else ClockLockReason.PROBE_RETURNED_FALSE
    if not ok:
        logger.warning(
            "Clock-lock unavailable (%s); expect timing variance", reason,
        )
        _emit_clock_unavailable(device_name, reason)
        return

    preset = _resolve_clock_preset(device_name)
    if preset is None:
        logger.warning(
            "No clock preset for %s — skipping clock lock", device_name,
        )
        _emit_clock_unavailable(device_name, ClockLockReason.NO_PRESET)
        return

    try:
        locked = _lock_gpu0_clocks(preset.gpu_clk_mhz, preset.dram_clk_mhz)
    except Exception as exc:  # never let a clock-lock hiccup kill the run
        logger.warning("_lock_gpu0_clocks failed for %s: %s", device_name, exc)
        _emit_clock_unavailable(device_name, str(exc)[:120])
        return
    if not locked:
        logger.warning(
            "_lock_gpu0_clocks returned False for %s — partial lock rolled back",
            device_name,
        )
        _emit_clock_unavailable(device_name, ClockLockReason.LOCK_FAILED)
        return

    # Verification failure (drift detected or exception) is treated as
    # lock-acquisition failure: roll back the partial pin so we don't
    # leave the GPU clamped to a wrong/drifting frequency, emit
    # ``clock_lock_unavailable`` with a meaningful ``reason``, and return
    # *before* flipping ``_clock_lock_state`` — downstream plateau/scoring
    # logic must see unlocked clocks (the correct degraded-mode signal),
    # not a phantom locked-state.
    try:
        verified = _verify_gpu0_locked(preset.gpu_clk_mhz, preset.dram_clk_mhz)
    except Exception as exc:
        logger.warning(
            "verify_clocks raised on %s — rolling back partial lock: %s",
            device_name, exc,
        )
        _rollback_partial_lock(device_name, f"verify_raised:{str(exc)[:120]}")
        return
    if not verified:
        logger.warning(
            "verify_clocks reported drift on %s — rolling back partial lock",
            device_name,
        )
        _rollback_partial_lock(device_name, ClockLockReason.VERIFY_FAILED)
        return

    _clock_lock_state["locked"] = True
    logger.info("GPU clocks locked for %s (GPU %s)", device_name, _GPU_INDEX)


if __name__ == "__main__":
    main()
