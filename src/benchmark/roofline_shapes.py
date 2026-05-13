"""Derive per-workload ``(flops, nbytes)`` for the analytical profiler.

``_definition_to_kernel_spec`` deliberately leaves ``KernelSpec.flop_count``
and ``memory_bytes`` at 0 because SOLAR supplies ``T_SOL`` directly. But
the hybrid profiler's arithmetic-intensity + achieved-peak math needs
nonzero counts, and zeros make ``_compute_analytical`` raise
``ProfilerError`` — which the orchestrator then treats as a dead branch.

This module rebuilds those counts from a SOL ``Definition`` + the
representative ``Workload`` the profiler will measure on. Flop formulas
are intentionally conservative and coarse: matmul / GEMM gets the
canonical ``2·M·N·K``; elementwise and small-compute ops get
``C·numel(output)`` with a low constant per op type; anything we don't
model returns ``(0, 0)`` so the caller can skip analytical profiling
for this iteration without killing the branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload

    from src.eval.roofline import RooflineResult


# Bytes per element for the dtype strings SOL-ExecBench emits. Keys are
# lower-cased before lookup so ``torch.float16`` and ``FLOAT16`` both hit.
_DTYPE_BYTES: dict[str, int] = {
    "float64": 8, "fp64": 8, "double": 8, "torch.float64": 8, "f64": 8,
    "float32": 4, "fp32": 4, "float": 4, "torch.float32": 4, "f32": 4,
    "float16": 2, "fp16": 2, "half": 2, "torch.float16": 2, "f16": 2,
    "bfloat16": 2, "bf16": 2, "torch.bfloat16": 2,
    "float8_e4m3": 1, "float8_e5m2": 1, "float8_e4m3fn": 1, "fp8": 1,
    "e4m3": 1, "e5m2": 1,
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


def compute_roofline_inputs(
    definition: Definition, workload: Workload,
    *,
    roofline: "RooflineResult | None" = None,
) -> tuple[int, int]:
    """Return ``(flops, nbytes)`` for ``definition`` running at ``workload``.

    Preference: SOLAR counts on *roofline* when both positive → shape-formula
    fallback (dispatched on ``definition.op_type``) → nbytes-only when only
    flops can't be derived (fused / op_type=None kernels with resolvable
    shapes; ``_compute_analytical`` accepts flops=0). ``(0, 0)`` only when
    nbytes is unresolvable too — callers then skip analytical and run NCU
    alone.
    """
    if roofline is not None and roofline.total_flops > 0 and roofline.total_fused_bytes > 0:
        return int(roofline.total_flops), int(roofline.total_fused_bytes)

    nbytes = _io_bytes(definition, workload)
    flops = _flops(definition, workload)
    if nbytes <= 0:
        return 0, 0
    if flops <= 0:
        return 0, nbytes
    return flops, nbytes


# ── internals ────────────────────────────────────────────────────────────


def _flops(definition: Definition, workload: Workload) -> int:
    op = (definition.op_type or "").lower()
    if op in ("matmul", "gemm", "linear"):
        return _matmul_flops(definition, workload)

    weight = _PER_ELEM_FLOPS.get(op)
    if weight is None or not definition.outputs:
        return 0
    output_shapes = _safe_output_shapes(definition, workload)
    if not output_shapes:
        return 0
    first_shape = next(iter(output_shapes.values()))
    n = _shape_numel(first_shape)
    return weight * n if n > 0 else 0


def _matmul_flops(definition: Definition, workload: Workload) -> int:
    """GEMM ``C[M, N] = A[M, K] @ B[K, N]`` → ``2·M·N·K``.

    K is resolved from (a) the first input's last-axis name or (b) common
    contraction-axis names (``K``, ``k``). The M·N product comes from the
    first output's numel so this also handles batched GEMMs where the
    output shape is ``[..., M, N]``.
    """
    if not definition.outputs or not definition.inputs:
        return 0

    output_shapes = _safe_output_shapes(definition, workload)
    if not output_shapes:
        return 0
    mn = _shape_numel(next(iter(output_shapes.values())))
    if mn <= 0:
        return 0

    k = _resolve_contraction_axis(definition, workload)
    return 2 * mn * k if k > 0 else 0


def _resolve_contraction_axis(definition: Definition, workload: Workload) -> int:
    """Find ``K`` — the inner / contraction dimension of a GEMM. Tries,
    in order: the first input's last-axis name, then common aliases.
    Returns 0 when unresolvable so callers bail cleanly."""
    inputs_iter = iter(definition.inputs.values())
    first_input = next(inputs_iter, None)
    if first_input is not None and first_input.shape:
        axis_name = first_input.shape[-1]
        value = _resolve_axis(axis_name, definition, workload)
        if value is not None and value > 0:
            return value
    for alias in ("K", "k", "inner", "contract"):
        value = _resolve_axis(alias, definition, workload)
        if value is not None and value > 0:
            return value
    return 0


def _io_bytes(definition: Definition, workload: Workload) -> int:
    """Total I/O traffic: sum of ``numel(t) · dtype_bytes(t.dtype)`` across
    every input + output tensor. Matches the coarse DRAM-traffic model the
    analytical profiler's bandwidth axis is built on."""
    input_shapes = _safe_input_shapes(definition, workload)
    output_shapes = _safe_output_shapes(definition, workload)
    if input_shapes is None or output_shapes is None:
        return 0
    total = 0
    for name, spec in definition.inputs.items():
        n = _shape_numel(input_shapes.get(name))
        if n <= 0:
            return 0
        total += n * _dtype_bytes(_dtype_str(spec.dtype))
    for name, spec in definition.outputs.items():
        n = _shape_numel(output_shapes.get(name))
        if n <= 0:
            return 0
        total += n * _dtype_bytes(_dtype_str(spec.dtype))
    return total


def _shape_numel(shape) -> int:
    """Elements in a single tensor shape. ``None`` (Python scalar) and
    ``()`` (0-D tensor) both collapse to 1. Negative or zero-size dims
    return 0 so callers bail."""
    if shape is None or shape == ():
        return 1
    product = 1
    for value in shape:
        if value is None or value <= 0:
            return 0
        product *= value
    return product


def _safe_input_shapes(definition: Definition, workload: Workload):
    """Resolve input shapes via ``definition.get_input_shapes`` or return
    ``None`` if any axis can't be resolved (caller bails)."""
    try:
        return definition.get_input_shapes(workload.axes)
    except ValueError:
        # ValueError = workload-driven (missing var-axis values) → expected,
        # caller bails to (0, 0). KeyError = schema-level corruption (workload
        # references an undeclared axis) → propagate so the orchestrator sees
        # it loud rather than silently skipping profiling.
        return None


def _safe_output_shapes(definition: Definition, workload: Workload):
    """Resolve output shapes via ``definition.get_output_shapes`` or return
    ``None`` if any axis can't be resolved (caller bails)."""
    try:
        return definition.get_output_shapes(workload.axes)
    except ValueError:
        # ValueError = workload-driven (missing var-axis values) → expected,
        # caller bails to (0, 0). KeyError = schema-level corruption (workload
        # references an undeclared axis) → propagate so the orchestrator sees
        # it loud rather than silently skipping profiling.
        return None


def _resolve_axis(name: str, definition: Definition, workload: Workload) -> int | None:
    """Resolve an axis name to its concrete int value. Workload overrides
    win over const axes from the definition. ``expr`` axes aren't evaluated
    here — callers fall through the ``None`` return path."""
    if name in workload.axes:
        return workload.axes[name]
    axis = definition.axes.get(name)
    if axis is None:
        return None
    if axis.type == "const":
        return axis.value
    return None


def _dtype_str(dtype) -> str:
    """Coerce a SOL ``DType`` enum (or any string-like) to a lower-case
    string suitable for ``_DTYPE_BYTES`` lookup."""
    value = getattr(dtype, "value", None)
    return value if isinstance(value, str) else str(dtype)


def _dtype_bytes(dtype: str) -> int:
    """Bytes per element for ``dtype``. Unrecognised strings default to 4
    (fp32) — a pragmatic choice for roofline math where the penalty for
    being off by 2× is far less than the cost of refusing to profile."""
    return _DTYPE_BYTES.get(dtype.lower(), 4)
