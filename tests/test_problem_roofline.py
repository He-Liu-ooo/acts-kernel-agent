"""Tests for ``src.benchmark.roofline_shapes.compute_roofline_inputs``.

SOL problems arrive at the orchestrator with ``KernelSpec.flop_count`` and
``memory_bytes`` at zero — ``problem_to_kernel_spec`` deliberately leaves
them at zero because SOLAR supplies ``T_SOL`` directly. The profiler,
however, needs per-workload flops and bytes for its arithmetic-intensity
+ achieved-peak math. This helper computes them from the Problem +
representative Workload's axis bindings; Tier 1 verifies the common op
types and the (0, 0) fallback that tells the orchestrator to skip
profiling rather than fail-close.
"""

from __future__ import annotations

from src.benchmark.problem import AxisDef, Problem, TensorDef, Workload
from src.benchmark.roofline_shapes import compute_roofline_inputs


def _problem(
    op_type: str,
    axes: dict[str, AxisDef],
    inputs: dict[str, TensorDef],
    outputs: dict[str, TensorDef],
) -> Problem:
    return Problem(
        name="p",
        axes=axes,
        inputs=inputs,
        outputs=outputs,
        reference_source="def run(): pass\n",
        op_type=op_type,
    )


# ── matmul family ────────────────────────────────────────────────────────


def test_matmul_computes_2_m_n_k_flops_and_io_bytes():
    """Standard GEMM: ``C = A @ B`` with A=[M,K], B=[K,N], C=[M,N]. fp32."""
    problem = _problem(
        op_type="matmul",
        axes={
            "M": AxisDef(type="var"),
            "N": AxisDef(type="var"),
            "K": AxisDef(type="var"),
        },
        inputs={
            "a": TensorDef(shape=["M", "K"], dtype="float32"),
            "b": TensorDef(shape=["K", "N"], dtype="float32"),
        },
        outputs={"c": TensorDef(shape=["M", "N"], dtype="float32")},
    )
    wl = Workload(uuid="wl0", axes={"M": 256, "N": 128, "K": 64})

    flops, nbytes = compute_roofline_inputs(problem, wl)

    assert flops == 2 * 256 * 128 * 64
    assert nbytes == (256 * 64 + 64 * 128 + 256 * 128) * 4


def test_gemm_aliases_to_matmul_formula():
    """``op_type='gemm'`` must take the same 2MNK path as 'matmul'."""
    problem = _problem(
        op_type="gemm",
        axes={
            "M": AxisDef(type="var"),
            "N": AxisDef(type="var"),
            "K": AxisDef(type="var"),
        },
        inputs={
            "a": TensorDef(shape=["M", "K"], dtype="float16"),
            "b": TensorDef(shape=["K", "N"], dtype="float16"),
        },
        outputs={"c": TensorDef(shape=["M", "N"], dtype="float16")},
    )
    wl = Workload(uuid="wl0", axes={"M": 32, "N": 32, "K": 16})

    flops, nbytes = compute_roofline_inputs(problem, wl)

    assert flops == 2 * 32 * 32 * 16
    # fp16 = 2 bytes/element
    assert nbytes == (32 * 16 + 16 * 32 + 32 * 32) * 2


# ── elementwise / reduction family ───────────────────────────────────────


def test_softmax_flops_scale_with_numel():
    """Softmax: ~5 flops per output element (max-reduce, exp, sum, divide,
    write). I/O bytes sum input + output at fp32."""
    problem = _problem(
        op_type="softmax",
        axes={"N": AxisDef(type="var")},
        inputs={"x": TensorDef(shape=["N"], dtype="float32")},
        outputs={"y": TensorDef(shape=["N"], dtype="float32")},
    )
    wl = Workload(uuid="wl0", axes={"N": 1024})

    flops, nbytes = compute_roofline_inputs(problem, wl)

    assert flops == 5 * 1024
    assert nbytes == 2 * 1024 * 4  # input + output


def test_rmsnorm_flops_scale_with_numel():
    problem = _problem(
        op_type="rmsnorm",
        axes={"S": AxisDef(type="var"), "D": AxisDef(type="var")},
        inputs={
            "x": TensorDef(shape=["S", "D"], dtype="bfloat16"),
            "w": TensorDef(shape=["D"], dtype="bfloat16"),
        },
        outputs={"y": TensorDef(shape=["S", "D"], dtype="bfloat16")},
    )
    wl = Workload(uuid="wl0", axes={"S": 8, "D": 128})

    flops, nbytes = compute_roofline_inputs(problem, wl)

    # 5 flops per output element
    assert flops == 5 * 8 * 128
    # bf16 = 2 bytes; bytes sum input x + weight + output y
    assert nbytes == (8 * 128 + 128 + 8 * 128) * 2


def test_elementwise_flops_one_per_numel():
    problem = _problem(
        op_type="elementwise",
        axes={"N": AxisDef(type="var")},
        inputs={"x": TensorDef(shape=["N"], dtype="float32")},
        outputs={"y": TensorDef(shape=["N"], dtype="float32")},
    )
    wl = Workload(uuid="wl0", axes={"N": 512})

    flops, nbytes = compute_roofline_inputs(problem, wl)
    assert flops == 512
    assert nbytes == 2 * 512 * 4


# ── const axes (resolved from problem.axes) ──────────────────────────────


def test_const_axes_resolved_from_problem_even_when_missing_from_workload():
    """Const axes live on ``problem.axes``, not in ``workload.axes`` — the
    helper must resolve them by falling back to the problem definition."""
    problem = _problem(
        op_type="matmul",
        axes={
            "M": AxisDef(type="var"),
            "N": AxisDef(type="var"),
            "K": AxisDef(type="const", value=64),  # const axis
        },
        inputs={
            "a": TensorDef(shape=["M", "K"], dtype="float32"),
            "b": TensorDef(shape=["K", "N"], dtype="float32"),
        },
        outputs={"c": TensorDef(shape=["M", "N"], dtype="float32")},
    )
    wl = Workload(uuid="wl0", axes={"M": 128, "N": 128})

    flops, nbytes = compute_roofline_inputs(problem, wl)
    assert flops == 2 * 128 * 128 * 64


# ── fallback on unknown / unresolvable inputs ────────────────────────────


def test_unknown_op_type_returns_zero_zero():
    """Callers must treat (0, 0) as 'skip profiling for this iteration'
    rather than bubbling zeros into the analytical profiler (which would
    raise ProfilerError and kill the branch)."""
    problem = _problem(
        op_type="some_new_op_we_havent_modelled",
        axes={"N": AxisDef(type="var")},
        inputs={"x": TensorDef(shape=["N"], dtype="float32")},
        outputs={"y": TensorDef(shape=["N"], dtype="float32")},
    )
    wl = Workload(uuid="wl0", axes={"N": 256})

    flops, nbytes = compute_roofline_inputs(problem, wl)
    assert (flops, nbytes) == (0, 0)


def test_unresolvable_axis_returns_zero_zero():
    """Axes that don't appear on workload OR as const on the problem can't
    be resolved — the helper must bail rather than compute a wrong value."""
    problem = _problem(
        op_type="matmul",
        axes={
            "M": AxisDef(type="var"),
            "N": AxisDef(type="var"),
            "K": AxisDef(type="expr", expression="N // 2"),  # expr axis not evaluated
        },
        inputs={
            "a": TensorDef(shape=["M", "K"], dtype="float32"),
            "b": TensorDef(shape=["K", "N"], dtype="float32"),
        },
        outputs={"c": TensorDef(shape=["M", "N"], dtype="float32")},
    )
    wl = Workload(uuid="wl0", axes={"M": 128, "N": 128})  # K unresolvable

    flops, nbytes = compute_roofline_inputs(problem, wl)
    assert (flops, nbytes) == (0, 0)


def test_empty_outputs_returns_zero_zero():
    """A problem with no outputs can't be sized — bail out gracefully."""
    problem = _problem(
        op_type="elementwise",
        axes={"N": AxisDef(type="var")},
        inputs={"x": TensorDef(shape=["N"], dtype="float32")},
        outputs={},
    )
    wl = Workload(uuid="wl0", axes={"N": 512})
    assert compute_roofline_inputs(problem, wl) == (0, 0)


# ── dtype handling ──────────────────────────────────────────────────────


def test_mixed_dtypes_sum_bytes_per_tensor():
    """Each tensor's bytes are ``numel * dtype_bytes(tensor.dtype)``; the
    total is the per-tensor sum, not a single global dtype."""
    problem = _problem(
        op_type="elementwise",
        axes={"N": AxisDef(type="var")},
        inputs={"x": TensorDef(shape=["N"], dtype="float16")},  # 2 bytes
        outputs={"y": TensorDef(shape=["N"], dtype="float32")},  # 4 bytes
    )
    wl = Workload(uuid="wl0", axes={"N": 1000})

    flops, nbytes = compute_roofline_inputs(problem, wl)
    assert flops == 1000
    assert nbytes == 1000 * 2 + 1000 * 4
