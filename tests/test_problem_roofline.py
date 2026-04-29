"""Tests for ``src.benchmark.roofline_shapes.compute_roofline_inputs``.

SOL problems arrive at the orchestrator with ``KernelSpec.flop_count`` and
``memory_bytes`` at zero — ``_definition_to_kernel_spec`` deliberately leaves
them at zero because SOLAR supplies ``T_SOL`` directly. The profiler,
however, needs per-workload flops and bytes for its arithmetic-intensity
+ achieved-peak math. This helper computes them from the Definition +
representative Workload's axis bindings; Tier 1 verifies the common op
types and the (0, 0) fallback that tells the orchestrator to skip
profiling rather than fail-close.
"""

from __future__ import annotations

from sol_execbench.core.data import Definition, Workload

from src.benchmark.roofline_shapes import compute_roofline_inputs


def _definition(
    op_type: str,
    axes: dict,
    inputs: dict,
    outputs: dict,
    *,
    reference: str | None = None,
) -> Definition:
    """Build a Definition from dict-form axes/inputs/outputs.

    SOL's pydantic validators require a syntactically-valid ``run`` whose
    parameter list matches ``inputs`` in order. When ``reference`` isn't
    supplied, synthesize one from the input keys so each test stays
    focused on roofline math rather than reference-source bookkeeping.
    """
    if reference is None:
        params = ", ".join(inputs.keys())
        reference = f"def run({params}): pass\n"
    return Definition.model_validate({
        "name": "p",
        "axes": axes,
        "inputs": inputs,
        "outputs": outputs,
        "reference": reference,
        "op_type": op_type,
    })


def _workload(uuid: str, axes: dict[str, int]) -> Workload:
    return Workload.model_validate({"uuid": uuid, "axes": axes, "inputs": {}})


# ── matmul family ────────────────────────────────────────────────────────


def test_matmul_computes_2_m_n_k_flops_and_io_bytes():
    """Standard GEMM: ``C = A @ B`` with A=[M,K], B=[K,N], C=[M,N]. fp32."""
    definition = _definition(
        op_type="matmul",
        axes={
            "M": {"type": "var"},
            "N": {"type": "var"},
            "K": {"type": "var"},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"c": {"shape": ["M", "N"], "dtype": "float32"}},
    )
    wl = _workload("wl0", {"M": 256, "N": 128, "K": 64})

    flops, nbytes = compute_roofline_inputs(definition, wl)

    assert flops == 2 * 256 * 128 * 64
    assert nbytes == (256 * 64 + 64 * 128 + 256 * 128) * 4


def test_gemm_aliases_to_matmul_formula():
    """``op_type='gemm'`` must take the same 2MNK path as 'matmul'."""
    definition = _definition(
        op_type="gemm",
        axes={
            "M": {"type": "var"},
            "N": {"type": "var"},
            "K": {"type": "var"},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float16"},
            "b": {"shape": ["K", "N"], "dtype": "float16"},
        },
        outputs={"c": {"shape": ["M", "N"], "dtype": "float16"}},
    )
    wl = _workload("wl0", {"M": 32, "N": 32, "K": 16})

    flops, nbytes = compute_roofline_inputs(definition, wl)

    assert flops == 2 * 32 * 32 * 16
    # fp16 = 2 bytes/element
    assert nbytes == (32 * 16 + 16 * 32 + 32 * 32) * 2


# ── elementwise / reduction family ───────────────────────────────────────


def test_softmax_flops_scale_with_numel():
    """Softmax: ~5 flops per output element (max-reduce, exp, sum, divide,
    write). I/O bytes sum input + output at fp32."""
    definition = _definition(
        op_type="softmax",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"y": {"shape": ["N"], "dtype": "float32"}},
    )
    wl = _workload("wl0", {"N": 1024})

    flops, nbytes = compute_roofline_inputs(definition, wl)

    assert flops == 5 * 1024
    assert nbytes == 2 * 1024 * 4  # input + output


def test_rmsnorm_flops_scale_with_numel():
    definition = _definition(
        op_type="rmsnorm",
        axes={"S": {"type": "var"}, "D": {"type": "var"}},
        inputs={
            "x": {"shape": ["S", "D"], "dtype": "bfloat16"},
            "w": {"shape": ["D"], "dtype": "bfloat16"},
        },
        outputs={"y": {"shape": ["S", "D"], "dtype": "bfloat16"}},
    )
    wl = _workload("wl0", {"S": 8, "D": 128})

    flops, nbytes = compute_roofline_inputs(definition, wl)

    # 5 flops per output element
    assert flops == 5 * 8 * 128
    # bf16 = 2 bytes; bytes sum input x + weight + output y
    assert nbytes == (8 * 128 + 128 + 8 * 128) * 2


def test_elementwise_flops_one_per_numel():
    definition = _definition(
        op_type="elementwise",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"y": {"shape": ["N"], "dtype": "float32"}},
    )
    wl = _workload("wl0", {"N": 512})

    flops, nbytes = compute_roofline_inputs(definition, wl)
    assert flops == 512
    assert nbytes == 2 * 512 * 4


# ── const axes (resolved from definition.axes) ───────────────────────────


def test_const_axes_resolved_from_definition_even_when_missing_from_workload():
    """Const axes live on ``definition.axes``, not in ``workload.axes`` — the
    helper must resolve them by falling back to the definition."""
    definition = _definition(
        op_type="matmul",
        axes={
            "M": {"type": "var"},
            "N": {"type": "var"},
            "K": {"type": "const", "value": 64},  # const axis
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"c": {"shape": ["M", "N"], "dtype": "float32"}},
    )
    wl = _workload("wl0", {"M": 128, "N": 128})

    flops, nbytes = compute_roofline_inputs(definition, wl)
    assert flops == 2 * 128 * 128 * 64


# ── fallback on unknown / unresolvable inputs ────────────────────────────


def test_unknown_op_type_returns_zero_zero():
    """Callers must treat (0, 0) as 'skip profiling for this iteration'
    rather than bubbling zeros into the analytical profiler (which would
    raise ProfilerError and kill the branch)."""
    definition = _definition(
        op_type="some_new_op_we_havent_modelled",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"y": {"shape": ["N"], "dtype": "float32"}},
    )
    wl = _workload("wl0", {"N": 256})

    flops, nbytes = compute_roofline_inputs(definition, wl)
    assert (flops, nbytes) == (0, 0)


def test_unresolvable_axis_returns_zero_zero():
    """Axes that don't appear on workload OR as const on the definition can't
    be resolved — the helper must bail rather than compute a wrong value."""
    definition = _definition(
        op_type="matmul",
        axes={
            "M": {"type": "var"},
            "N": {"type": "var"},
            "K": {"type": "expr", "expression": "N // 2"},  # expr axis not evaluated
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"c": {"shape": ["M", "N"], "dtype": "float32"}},
    )
    wl = _workload("wl0", {"M": 128, "N": 128})  # K unresolvable

    flops, nbytes = compute_roofline_inputs(definition, wl)
    assert (flops, nbytes) == (0, 0)


def test_empty_outputs_returns_zero_zero():
    """A definition with no outputs can't be sized — bail out gracefully."""
    definition = _definition(
        op_type="elementwise",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={},
    )
    wl = _workload("wl0", {"N": 512})
    assert compute_roofline_inputs(definition, wl) == (0, 0)


# ── dtype handling ──────────────────────────────────────────────────────


def test_mixed_dtypes_sum_bytes_per_tensor():
    """Each tensor's bytes are ``numel * dtype_bytes(tensor.dtype)``; the
    total is the per-tensor sum, not a single global dtype."""
    definition = _definition(
        op_type="elementwise",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float16"}},  # 2 bytes
        outputs={"y": {"shape": ["N"], "dtype": "float32"}},  # 4 bytes
    )
    wl = _workload("wl0", {"N": 1000})

    flops, nbytes = compute_roofline_inputs(definition, wl)
    assert flops == 1000
    assert nbytes == 1000 * 2 + 1000 * 4
