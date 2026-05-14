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

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload

    from src.config import ACTSConfig
    from src.kernels.kernel import Kernel


_log = logging.getLogger(__name__)

# Reserved seed for the per-workload autotune burn-in call. Negative so it
# can't collide with warmup or timed iter seeds (which range over 0..N).
_BURN_IN_SEED = -1


def _key_tuple_for(
    workload: "Workload",
    autotune_keys: list[str],
    definition: "Definition | None" = None,
) -> tuple | None:
    """Resolve autotune key arg-names against a workload's resolved axes.

    Looks up each key string with two-stage resolution:
      1. ``workload.axes`` first (SOL's per-workload axis-name → integer
         map; e.g. ``{"B": 2}`` for a batch axis that varies per
         workload). Workload bindings always win — they're the immediate
         runtime context.
      2. If *definition* is provided AND the axis is declared as
         ``AxisConst`` on ``definition.axes``, use the const value. This
         covers the common SOL shape where ``key=["B", "M", "N"]`` spans
         a var axis (``B`` on workload) and const axes (``M``/``N`` on
         the Definition). Codex review 2026-05-14 finding #2.

    If any key fails to resolve via either stage, returns ``None`` — the
    orchestrator treats that as "autotune_winner unavailable for this
    workload" and continues. Empty ``autotune_keys`` returns ``()``
    (valid Triton cache lookup for kernels with ``key=[]``, though the
    Coder validator forbids that for emitted kernels).

    Non-const axis types (``AxisVar`` without a workload binding,
    ``AxisExpr``) degrade to None — they can't be statically resolved.
    """
    axes = getattr(workload, "axes", None) or {}
    def_axes = getattr(definition, "axes", None) or {} if definition is not None else {}
    values: list = []
    for k in autotune_keys:
        if k in axes:
            values.append(axes[k])
            continue
        def_axis = def_axes.get(k) if def_axes else None
        # Class-name check is the safest cross-Tier path: importing
        # ``AxisConst`` at module top would pull in ``sol_execbench``,
        # and the alternative ``getattr(def_axis, "type") == "const"``
        # is silently typo-prone.
        if def_axis is not None and type(def_axis).__name__ == "AxisConst":
            const_value = getattr(def_axis, "value", None)
            if const_value is not None:
                values.append(const_value)
                continue
        _log.warning(
            "autotune key %r not resolvable from workload %s axes or "
            "definition const axes; autotune_winner will be None for "
            "this workload.",
            k, getattr(workload, "uuid", "?"),
        )
        return None
    return tuple(values)


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
    """Latency benchmark result for a single kernel.

    ``last_outputs`` carries the *last* workload's last-iter output
    tensors (or whatever the kernel returned, in non-DPS mode) so the
    orchestrator can run ``check_lazy_outputs_after_bench`` on real
    output references. We retain only the last batch — full retention
    across all workloads × all iters would balloon GPU memory and we
    only need *some* concrete output to validate against.
    """

    median_latency_us: float = 0.0
    min_latency_us: float = 0.0
    max_latency_us: float = 0.0
    warmup_runs: int = 0
    timed_runs: int = 0
    per_workload_latency_us: dict[str, float] = field(default_factory=dict)
    workload_errors: dict[str, str] = field(default_factory=dict)
    last_outputs: list = field(default_factory=list)

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
    definition: Definition | None = None,
    discard_first: int = 1,
) -> BenchmarkResult:
    """Benchmark kernel latency via the injected timer.

    When ``workloads`` / ``input_generators`` are both empty, returns a
    100us sentinel so the placeholder/no-workload smoke path stays
    runnable; 0.0 would collapse ``compute_sol_score`` to 1.0 and
    silently fabricate an optimum.

    When ``kernel.dps`` is True, *definition* must be supplied — outputs
    are pre-allocated per iter via ``sol_execbench.core.bench.io.allocate_outputs``
    and threaded into ``kernel_fn(*inputs, *outputs)``. When ``kernel.dps``
    is False (the default), ``definition`` is unused and the kernel's
    return value is treated as the output (legacy path).
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

    if kernel.dps and definition is None:
        raise ValueError(
            "kernel.dps=True requires a Definition so allocate_outputs can "
            "pre-allocate output buffers per iter — pass definition=... to "
            "benchmark_kernel.",
        )

    fn = kernel_fn if kernel_fn is not None else _compile_entrypoint(kernel)
    factory = timer_factory or _default_timer_factory

    per_wl: dict[str, float] = {}
    errors: dict[str, str] = {}
    last_outputs: list = []

    for wl, gen in zip(workloads, input_generators):
        # Fresh timer per workload: a CUDA launch/event fault can leave
        # the stream in a sticky error state, so reusing the same timer
        # would turn a workload-local failure into order-dependent false
        # failures on every later workload on the same bench pass.
        timer = factory()
        # Wrap the input generator so DPS kernels see the inputs followed
        # by freshly-allocated outputs on every iteration. The wrapper is
        # outside the timed window — same shape, same dtypes, same device
        # as the workload's outputs — so allocator pauses don't leak into
        # the latency measurement.
        wl_gen = _wrap_dps_generator(gen, kernel=kernel, workload=wl, definition=definition)
        median_ms, error, captured = _time_workload(
            fn=fn,
            input_generator=wl_gen,
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
        # Overwrite each workload so we end up with the *last* workload's
        # last-iter outputs — matches the BenchmarkResult.last_outputs
        # contract (used for check_lazy_outputs_after_bench).
        if captured is not None:
            last_outputs = captured

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
        last_outputs=last_outputs,
    )


def _wrap_dps_generator(
    gen: Callable[[int], tuple],
    *,
    kernel: Kernel,
    workload: Workload,
    definition: Definition | None,
) -> Callable[[int], tuple]:
    """Return an input generator that appends pre-allocated output buffers
    when ``kernel.dps`` is True.

    ``allocate_outputs`` allocates once per iter — fresh buffers per call
    so write-after-write hazards across iterations don't depend on the
    kernel zeroing its outputs. The wrapper is invoked outside the timed
    window (in ``_time_workload``), so allocator latency doesn't leak
    into the measurement. For non-DPS kernels the original generator
    flows through unchanged.
    """
    if not kernel.dps:
        return gen

    # Lazy import — keep the torch-less unit tests importing benchmark.py
    # without paying for sol_execbench at module load.
    from src.eval.inputs import allocate_dps_outputs

    assert definition is not None  # ruled out at the benchmark_kernel call site

    # Resolve a concrete device for ``allocate_dps_outputs``. We peek at the
    # first input tensor's device on each call — keeps the wrapper agnostic
    # to whether the input generator returns pinned-CPU or already-on-CUDA
    # tensors. Cheap (a single attribute read).

    def _wrapped(seed: int) -> tuple:
        import torch

        inputs = gen(seed)
        device = "cuda"
        for arg in inputs:
            if isinstance(arg, torch.Tensor):
                device = str(arg.device)
                break
        outputs = allocate_dps_outputs(definition, workload, device=device)
        return tuple(inputs) + tuple(outputs)

    return _wrapped


def _time_workload(
    *,
    fn: Callable,
    input_generator: Callable[[int], tuple],
    timer: BenchmarkTimer,
    warmup: int,
    timed: int,
    discard_first: int,
) -> tuple[float, str | None, list | None]:
    """Returns ``(median_ms, error, last_call_outputs)``.

    ``last_call_outputs`` captures the kernel's outputs from the last
    successful timed iteration, so the caller can hand them to
    ``check_lazy_outputs_after_bench``. For DPS kernels the wrapper
    appends the pre-allocated output buffers to ``args``; for non-DPS
    kernels we capture the return value of ``fn(*args)``. The list is
    ``None`` when the workload errored out.

    A1 PR 1: one **burn-in** call fires before the warmup loop so
    Triton's ``@triton.autotune`` runs its config sweep + compile +
    micro-bench OUTSIDE the timed window. Seed ``-1`` is reserved for
    burn-in so it can't collide with warmup or timed seeds.
    """
    try:
        burn_args = input_generator(_BURN_IN_SEED)
        fn(*burn_args)
        timer.prepare()
    except Exception as e:
        return 0.0, f"autotune burn-in failed: {type(e).__name__}: {e}", None

    try:
        for i in range(warmup):
            args = input_generator(i)
            fn(*args)
    except Exception as e:
        return 0.0, f"warmup failed: {type(e).__name__}: {e}", None

    samples: list[float] = []
    last_outputs: list | None = None
    total_iters = timed + discard_first
    try:
        for i in range(total_iters):
            args = input_generator(warmup + i)
            elapsed_ms, ret = _time_iter(timer, fn, args)
            if i >= discard_first:
                samples.append(elapsed_ms)
            # Last successful iter wins. For non-DPS kernels we take the
            # return value (flattened into a list of tensors); for DPS
            # kernels we have no per-call return so we just hold a
            # reference to the outputs portion of args. We can't
            # disambiguate inputs vs outputs without reaching into the
            # wrapper, so the simpler rule wins: capture ``ret`` when
            # not None, else the last arg tensor (DPS kernels write into
            # output buffers passed last).
            if ret is not None:
                last_outputs = _flatten_to_output_list(ret)
            elif args:
                # Best-effort: pick up tensor-typed args. If nothing
                # tensor-shaped exists, just pass the last positional
                # arg through — ``check_lazy_outputs`` rejects non-Tensor
                # types so a real DPS write-through-output-buffer gets
                # validated.
                last_outputs = [args[-1]]
    except Exception as e:
        return 0.0, f"{type(e).__name__}: {e}", None

    if not samples:
        return 0.0, "no timed samples collected", None
    return statistics.median(samples), None, last_outputs


def _flatten_to_output_list(ret: object) -> list:
    """Normalize a kernel's return value to a flat list of output tensors.

    Used by ``_time_workload`` so ``check_lazy_outputs_after_bench`` sees
    actual ``torch.Tensor`` objects rather than container shells. Without
    this, a host wrapper that returns named outputs as a ``dict`` (e.g.
    LayerNorm returning ``{"y", "mean", "rstd"}``) would have the dict
    itself packed into ``[ret]`` and fail the post-bench lazy-output check
    — silently pruning every dict-return branch (fail-closed).

    Shape rules:
      * ``None`` → ``[]`` (no outputs to validate)
      * ``tuple`` / ``list`` → ``list(ret)`` (already a sequence)
      * ``dict`` → ``list(ret.values())`` (drop names, keep tensors)
      * anything else (e.g. a single ``torch.Tensor``) → ``[ret]``

    The unknown-shape fallback preserves prior behaviour and lets
    ``check_lazy_outputs_after_bench`` surface the issue if the value
    isn't a tensor.
    """
    if ret is None:
        return []
    if isinstance(ret, (list, tuple)):
        return list(ret)
    if isinstance(ret, dict):
        return list(ret.values())
    return [ret]


def _time_iter(timer: BenchmarkTimer, fn: Callable, args: tuple) -> tuple[float, object]:
    """Time a single call and return ``(elapsed_ms, return_value)``.

    The return value is forwarded to ``_time_workload`` so non-DPS
    kernels' output tensors flow through to ``BenchmarkResult.last_outputs``
    for the post-bench lazy-output check.
    """
    timer.prepare()
    timer.flush_l2()
    timer.record_start()
    ret = fn(*args)
    timer.record_end()
    return timer.finalize_ms(), ret


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
