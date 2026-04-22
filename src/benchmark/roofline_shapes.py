"""Derive per-workload ``(flops, nbytes)`` for the analytical profiler.

``problem_to_kernel_spec`` deliberately leaves ``KernelSpec.flop_count``
and ``memory_bytes`` at 0 because SOLAR supplies ``T_SOL`` directly. But
the hybrid profiler's arithmetic-intensity + achieved-peak math needs
nonzero counts, and zeros make ``_compute_analytical`` raise
``ProfilerError`` — which the orchestrator then treats as a dead branch.

This module rebuilds those counts from a ``Problem`` + the representative
``Workload`` the profiler will measure on. Flop formulas are intentionally
conservative and coarse: matmul / GEMM gets the canonical ``2·M·N·K``;
elementwise and small-compute ops get ``C·numel(output)`` with a low
constant per op type; anything we don't model returns ``(0, 0)`` so the
caller can skip analytical profiling for this iteration without killing
the branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.benchmark.problem import Problem, TensorDef, Workload


# Bytes per element for the dtype strings SOL-ExecBench emits. Keys are
# lower-cased before lookup so ``torch.float16`` and ``FLOAT16`` both hit.
_DTYPE_BYTES: dict[str, int] = {
    "float64": 8, "fp64": 8, "double": 8, "torch.float64": 8, "f64": 8,
    "float32": 4, "fp32": 4, "float": 4, "torch.float32": 4, "f32": 4,
    "float16": 2, "fp16": 2, "half": 2, "torch.float16": 2, "f16": 2,
    "bfloat16": 2, "bf16": 2, "torch.bfloat16": 2,
    "float8_e4m3": 1, "float8_e5m2": 1, "fp8": 1, "e4m3": 1, "e5m2": 1,
    "int64": 8, "long": 8, "torch.int64": 8, "i64": 8,
    "int32": 4, "int": 4, "torch.int32": 4, "i32": 4,
    "int16": 2, "short": 2, "torch.int16": 2, "i16": 2,
    "int8": 1, "uint8": 1, "torch.int8": 1, "i8": 1, "u8": 1,
    "bool": 1, "torch.bool": 1,
}

# Flop weight per output element for coarse-compute ops. Matmul / GEMM
# has its own ``2·M·N·K`` path and isn't in this table.
_PER_ELEM_FLOPS: dict[str, int] = {
    "elementwise": 1,
    "rope": 4,          # sin/cos rotation on each element
    "embedding": 1,     # gather copy
    "softmax": 5,       # max-reduce, exp, sum-reduce, divide, write
    "rmsnorm": 5,
    "layernorm": 5,
}


def compute_roofline_inputs(problem: Problem, workload: Workload) -> tuple[int, int]:
    """Return ``(flops, nbytes)`` for ``problem`` running at ``workload``.

    ``(0, 0)`` means "we don't have a formula here" — callers must treat
    that as a signal to skip analytical profiling rather than feed zeros
    into ``_compute_analytical`` (which would raise).
    """
    nbytes = _io_bytes(problem, workload)
    flops = _flops(problem, workload)
    if flops <= 0 or nbytes <= 0:
        return 0, 0
    return flops, nbytes


# ── internals ────────────────────────────────────────────────────────────


def _flops(problem: Problem, workload: Workload) -> int:
    op = (problem.op_type or "").lower()
    if op in ("matmul", "gemm", "linear"):
        return _matmul_flops(problem, workload)

    weight = _PER_ELEM_FLOPS.get(op)
    if weight is None or not problem.outputs:
        return 0
    out0 = next(iter(problem.outputs.values()))
    n = _numel(out0, problem, workload)
    return weight * n if n > 0 else 0


def _matmul_flops(problem: Problem, workload: Workload) -> int:
    """GEMM ``C[M, N] = A[M, K] @ B[K, N]`` → ``2·M·N·K``.

    K is resolved from (a) the first input's last-axis name or (b) common
    contraction-axis names (``K``, ``k``). The M·N product comes from the
    first output's numel so this also handles batched GEMMs where the
    output shape is ``[..., M, N]``.
    """
    if not problem.outputs or not problem.inputs:
        return 0
    out0 = next(iter(problem.outputs.values()))
    mn = _numel(out0, problem, workload)
    if mn <= 0:
        return 0

    k = _resolve_contraction_axis(problem, workload)
    return 2 * mn * k if k > 0 else 0


def _resolve_contraction_axis(problem: Problem, workload: Workload) -> int:
    """Find ``K`` — the inner / contraction dimension of a GEMM. Tries,
    in order: the first input's last-axis name, then common aliases.
    Returns 0 when unresolvable so callers bail cleanly."""
    inputs_iter = iter(problem.inputs.values())
    first_input = next(inputs_iter, None)
    if first_input is not None and first_input.shape:
        axis_name = first_input.shape[-1]
        value = _resolve_axis(axis_name, problem, workload)
        if value is not None and value > 0:
            return value
    for alias in ("K", "k", "inner", "contract"):
        value = _resolve_axis(alias, problem, workload)
        if value is not None and value > 0:
            return value
    return 0


def _io_bytes(problem: Problem, workload: Workload) -> int:
    """Total I/O traffic: sum of ``numel(t) · dtype_bytes(t.dtype)`` across
    every input + output tensor. Matches the coarse DRAM-traffic model the
    analytical profiler's bandwidth axis is built on."""
    total = 0
    for tensor in list(problem.inputs.values()) + list(problem.outputs.values()):
        n = _numel(tensor, problem, workload)
        if n <= 0:
            return 0
        total += n * _dtype_bytes(tensor.dtype)
    return total


def _numel(tensor: TensorDef, problem: Problem, workload: Workload) -> int:
    """Elements in a single tensor. ``shape=None`` (Python scalar) and
    ``shape=[]`` (0-D tensor) both collapse to 1. Unresolvable axis
    names return 0 so callers bail."""
    if tensor.shape is None or tensor.shape == []:
        return 1
    product = 1
    for axis_name in tensor.shape:
        value = _resolve_axis(axis_name, problem, workload)
        if value is None or value <= 0:
            return 0
        product *= value
    return product


def _resolve_axis(name: str, problem: Problem, workload: Workload) -> int | None:
    """Resolve an axis name to its concrete int value. Workload overrides
    win over const axes from the problem. ``expr`` axes aren't evaluated
    — callers bail through the ``None`` return path."""
    if name in workload.axes:
        return workload.axes[name]
    axis = problem.axes.get(name)
    if axis is None:
        return None
    if axis.type == "const" and axis.value is not None:
        return axis.value
    return None


def _dtype_bytes(dtype: str) -> int:
    """Bytes per element for ``dtype``. Unrecognised strings default to 4
    (fp32) — a pragmatic choice for roofline math where the penalty for
    being off by 2× is far less than the cost of refusing to profile."""
    return _DTYPE_BYTES.get(dtype.lower(), 4)
