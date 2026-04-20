"""Latency measurement via CUDA events.

Called by the orchestrator after the Coder returns a compiled, correct
kernel. Not part of the Coder's tool loop.

Protocol
--------
Each timed iteration runs the sequence: ``prepare → flush_l2 →
record_start → kernel_fn(*args) → record_end → finalize_ms``. L2 is
flushed **before** ``record_start`` so the kernel sees cold cache and
flush time is excluded from the measurement (KernelBench convention;
see ``repo/benchmark/KernelBench/src/kernelbench/timing.py``).

Inputs are regenerated per iter outside the timing window so in-place
kernels don't see degenerate inputs on iter N+1.

Aggregation
-----------
Per workload: median of the timed samples (first ``discard_first`` dropped).
Across workloads: median-of-medians as the scalar headline, with the
full per-workload dict preserved on the result.

Fail-closed
-----------
Per-workload launch failures mark that workload's latency as ``inf``
and record the reason. If strictly fewer than half the workloads
survive, ``BenchmarkError`` is raised so the orchestrator can mark the
branch dead.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from src.benchmark.problem import Workload
    from src.config import ACTSConfig
    from src.kernels.kernel import Kernel


class BenchmarkTimer(Protocol):
    """Per-iteration timing primitive.

    Production implementation uses ``torch.cuda.Event`` + an L2-thrashing
    dummy tensor. Tests inject a recorder that returns a scripted elapsed
    sequence so dispatch / aggregation / call-order can be verified
    without torch.
    """

    def prepare(self) -> None:
        """Synchronize device before the iteration starts."""

    def flush_l2(self) -> None:
        """Thrash L2 cache so the kernel sees cold inputs."""

    def record_start(self) -> None:
        """Record the start event on the current stream."""

    def record_end(self) -> None:
        """Record the end event on the current stream."""

    def finalize_ms(self) -> float:
        """Synchronize and return elapsed ms between start and end."""


@dataclass
class BenchmarkResult:
    """Latency benchmark result for a single kernel."""

    median_latency_us: float = 0.0
    min_latency_us: float = 0.0
    max_latency_us: float = 0.0
    warmup_runs: int = 0
    timed_runs: int = 0
    per_workload_latency_us: dict[str, float] = field(default_factory=dict)
    workload_errors: dict[str, str] = field(default_factory=dict)

    @property
    def is_fully_successful(self) -> bool:
        return not self.workload_errors


class BenchmarkError(RuntimeError):
    """Raised when too few workloads survived to produce a trustworthy result."""


def benchmark_kernel(
    kernel: Kernel,
    config: ACTSConfig,
    *,
    workloads: list[Workload] | None = None,
    input_generators: list[Callable[[int], tuple]] | None = None,
    timer_factory: Callable[[], BenchmarkTimer] | None = None,
    kernel_fn: Callable | None = None,
    discard_first: int = 1,
) -> BenchmarkResult:
    """Benchmark kernel latency via the injected timer.

    When ``workloads`` / ``input_generators`` are both empty, returns a
    100us sentinel so the pre-SOL pipeline stays runnable; 0.0 would
    collapse ``compute_sol_score`` to 1.0 and silently fabricate an
    optimum.
    """
    workloads = workloads or []
    input_generators = input_generators or []

    if not workloads and not input_generators:
        return BenchmarkResult(
            median_latency_us=100.0,
            min_latency_us=100.0,
            max_latency_us=100.0,
            warmup_runs=config.warmup_runs,
            timed_runs=config.timed_runs,
        )

    if len(workloads) != len(input_generators):
        raise ValueError(
            f"workloads ({len(workloads)}) and input_generators "
            f"({len(input_generators)}) must be the same length"
        )

    fn = kernel_fn if kernel_fn is not None else _compile_entrypoint(kernel)
    factory = timer_factory or _default_timer_factory

    per_wl: dict[str, float] = {}
    errors: dict[str, str] = {}

    for wl, gen in zip(workloads, input_generators):
        # Fresh timer per workload: a CUDA launch/event fault can leave
        # the stream in a sticky error state, so reusing the same timer
        # would turn a workload-local failure into order-dependent false
        # failures on every later workload on the same bench pass.
        timer = factory()
        median_ms, error = _time_workload(
            fn=fn,
            input_generator=gen,
            timer=timer,
            warmup=config.warmup_runs,
            timed=config.timed_runs,
            discard_first=discard_first,
        )
        if error is not None:
            per_wl[wl.uuid] = math.inf
            errors[wl.uuid] = error
        else:
            per_wl[wl.uuid] = median_ms * 1000.0

    finite_us = [v for v in per_wl.values() if math.isfinite(v)]
    survivors = len(finite_us)
    if 2 * survivors < len(workloads):
        raise BenchmarkError(
            f"only {survivors}/{len(workloads)} workloads survived benchmarking; "
            f"first error: {next(iter(errors.values()), 'unknown')}"
        )

    return BenchmarkResult(
        median_latency_us=statistics.median(finite_us),
        min_latency_us=min(finite_us),
        max_latency_us=max(finite_us),
        warmup_runs=config.warmup_runs,
        timed_runs=config.timed_runs,
        per_workload_latency_us=per_wl,
        workload_errors=errors,
    )


def _time_workload(
    *,
    fn: Callable,
    input_generator: Callable[[int], tuple],
    timer: BenchmarkTimer,
    warmup: int,
    timed: int,
    discard_first: int,
) -> tuple[float, str | None]:
    try:
        for i in range(warmup):
            args = input_generator(i)
            fn(*args)
    except Exception as e:
        return 0.0, f"warmup failed: {type(e).__name__}: {e}"

    samples: list[float] = []
    total_iters = timed + discard_first
    try:
        for i in range(total_iters):
            args = input_generator(warmup + i)
            elapsed_ms = _time_iter(timer, fn, args)
            if i >= discard_first:
                samples.append(elapsed_ms)
    except Exception as e:
        return 0.0, f"{type(e).__name__}: {e}"

    if not samples:
        return 0.0, "no timed samples collected"
    return statistics.median(samples), None


def _time_iter(timer: BenchmarkTimer, fn: Callable, args: tuple) -> float:
    timer.prepare()
    timer.flush_l2()
    timer.record_start()
    fn(*args)
    timer.record_end()
    return timer.finalize_ms()


def _compile_entrypoint(kernel: Kernel) -> Callable:
    from src.kernels.compiler import compile_kernel

    result = compile_kernel(kernel)
    if not result.success or result.compiled_fn is None:
        raise BenchmarkError(f"compile failed: {result.error_message}")
    return result.compiled_fn


def _default_timer_factory() -> BenchmarkTimer:
    """Torch-backed timer — CUDA events + L2 thrasher. Imported lazily."""
    return _TorchCudaTimer()


class _TorchCudaTimer:
    """CUDA-event timer with a reusable 256MB int64 tensor for L2 flush.

    Matches KernelBench's ``clear_l2_cache`` (256MB overwrite thrashes L2
    on every current NVIDIA arch through Blackwell). Tensor is allocated
    once per timer instance and filled on each iteration.
    """

    _L2_THRASH_ELEMS = 32 * 1024 * 1024  # 32M × int64 = 256MB

    def __init__(self) -> None:
        import torch  # lazy import — production path only

        self._torch = torch
        self._device = torch.cuda.current_device()
        self._thrash = torch.empty(
            self._L2_THRASH_ELEMS, dtype=torch.int64, device=self._device
        )
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def prepare(self) -> None:
        self._torch.cuda.synchronize(self._device)

    def flush_l2(self) -> None:
        self._thrash.fill_(42)

    def record_start(self) -> None:
        self._start.record()

    def record_end(self) -> None:
        self._end.record()

    def finalize_ms(self) -> float:
        self._torch.cuda.synchronize(self._device)
        return float(self._start.elapsed_time(self._end))
