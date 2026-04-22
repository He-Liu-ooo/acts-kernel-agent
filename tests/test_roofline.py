"""Tests for ``classify_run`` and ``classify_workload`` in src/eval/roofline.py.

Pure arithmetic, no GPU, no subprocess. Torch-free: runs in the default
/tmp/acts_test_venv (pytest + pyyaml). Fixtures are built inline to keep
conftest.py unchanged.
"""

from __future__ import annotations

import pytest

from conftest import rtx6000_ada_hardware as _rtx6000_ada
from src.benchmark.problem import AxisDef, Problem, TensorDef, Workload
from src.benchmark.roofline_shapes import compute_roofline_inputs
from src.config import HardwareSpec
from src.eval.roofline import (
    BottleneckType,
    RooflineResult,
    classify_bottleneck,
    classify_run,
    classify_workload,
)
from src.kernels.kernel import KernelSpec, KernelType


# ── inline fixtures ─────────────────────────────────────────────────────────


def _mk_matmul_problem() -> Problem:
    return Problem(
        name="matmul_test",
        axes={
            "M": AxisDef(type="var"),
            "N": AxisDef(type="var"),
            "K": AxisDef(type="var"),
        },
        inputs={
            "A": TensorDef(shape=["M", "K"], dtype="float32"),
            "B": TensorDef(shape=["K", "N"], dtype="float32"),
        },
        outputs={"C": TensorDef(shape=["M", "N"], dtype="float32")},
        reference_source="",
        op_type="matmul",
    )


def _mk_matmul_workload(m: int, n: int, k: int, uuid: str = "w0") -> Workload:
    return Workload(uuid=uuid, axes={"M": m, "N": n, "K": k})


def _mk_elementwise_problem() -> Problem:
    # Multi-input + fp64 tensors so total I/O traffic per elem is high enough
    # that AI = 1 flop / 24 bytes ≈ 0.042 sits well below the test-fixture
    # ridge of ~0.067. The default fixture's ridge is small because the
    # HardwareSpec derivation convention gives peak_flops_fp32 ≈ 0.064 TFLOPS
    # here — see conftest.rtx6000_ada_hardware.
    return Problem(
        name="elt_test",
        axes={"N": AxisDef(type="var")},
        inputs={
            "a": TensorDef(shape=["N"], dtype="float64"),
            "b": TensorDef(shape=["N"], dtype="float64"),
        },
        outputs={"y": TensorDef(shape=["N"], dtype="float64")},
        reference_source="",
        op_type="elementwise",
    )


# ── classify_run ────────────────────────────────────────────────────────────


def test_classify_run_uses_solar_when_provided():
    hw = _rtx6000_ada()
    rr = RooflineResult(
        t_sol_us=1.0,
        arithmetic_intensity=1000.0,
        bottleneck=BottleneckType.COMPUTE_BOUND,
        source="solar",
    )
    # baseline_spec is intentionally None — must be ignored when roofline given.
    assert (
        classify_run(hardware=hw, roofline=rr, baseline_spec=None)
        == BottleneckType.COMPUTE_BOUND
    )


def test_classify_run_falls_back_to_compute_roofline():
    hw = _rtx6000_ada()
    # AI = 100 / 10_000_000 = 1e-5, well below ridge (~66.7) → memory-bound.
    spec = KernelSpec(
        name="t",
        kernel_type=KernelType.ELEMENTWISE,
        flop_count=100,
        memory_bytes=10_000_000,
    )
    assert (
        classify_run(hardware=hw, roofline=None, baseline_spec=spec)
        == BottleneckType.MEMORY_BOUND
    )


def test_classify_run_falls_back_compute_bound():
    hw = _rtx6000_ada()
    # AI = 1e10 / 1000 = 1e7, well above ridge → compute-bound.
    spec = KernelSpec(
        name="t",
        kernel_type=KernelType.MATMUL,
        flop_count=10_000_000_000,
        memory_bytes=1_000,
    )
    assert (
        classify_run(hardware=hw, roofline=None, baseline_spec=spec)
        == BottleneckType.COMPUTE_BOUND
    )


def test_classify_run_raises_when_both_none():
    hw = _rtx6000_ada()
    with pytest.raises(ValueError):
        classify_run(hardware=hw, roofline=None, baseline_spec=None)


# ── classify_workload ──────────────────────────────────────────────────────


def test_classify_workload_matmul_compute_bound():
    hw = _rtx6000_ada()
    problem = _mk_matmul_problem()
    workload = _mk_matmul_workload(4096, 4096, 4096)
    # AI = 2·M·N·K / (3·M·N·4) = 2K/12 = K/6 ≈ 682 >> ridge ≈ 66.7
    assert classify_workload(problem, workload, hw) == BottleneckType.COMPUTE_BOUND


def test_classify_workload_elementwise_memory_bound():
    hw = _rtx6000_ada()
    problem = _mk_elementwise_problem()
    workload = Workload(uuid="w0", axes={"N": 1_000_000})
    # AI = 1 flop/elem / 24 bytes/elem (3 × fp64 tensors) ≈ 0.042, well below
    # the fixture ridge of ~0.067 → memory_bound.
    assert classify_workload(problem, workload, hw) == BottleneckType.MEMORY_BOUND


def test_classify_workload_raises_on_unknown_op_type():
    hw = _rtx6000_ada()
    problem = _mk_elementwise_problem()
    problem.op_type = "quantum_woo"
    workload = Workload(uuid="w0", axes={"N": 1_000_000})
    with pytest.raises(ValueError, match="no roofline formula"):
        classify_workload(problem, workload, hw)


def test_classify_workload_raises_on_zero_hardware_peaks():
    hw = HardwareSpec()  # all zeros
    problem = _mk_matmul_problem()
    workload = _mk_matmul_workload(128, 128, 128)
    with pytest.raises(ValueError, match="hardware peaks"):
        classify_workload(problem, workload, hw)


# ── shape-awareness + matches-low-level ────────────────────────────────────


def test_classify_workload_shape_dependent():
    """Shape is threaded correctly into compute_roofline_inputs — the helper
    must accept both tiny and large matmul workloads without crashing and
    return a well-defined BottleneckType for each."""
    hw = _rtx6000_ada()
    problem = _mk_matmul_problem()
    small = _mk_matmul_workload(16, 16, 16, uuid="small")
    large = _mk_matmul_workload(4096, 4096, 4096, uuid="large")
    small_cls = classify_workload(problem, small, hw)
    large_cls = classify_workload(problem, large, hw)
    assert isinstance(small_cls, BottleneckType)
    assert isinstance(large_cls, BottleneckType)


def test_classify_workload_matches_classify_bottleneck():
    """Helper must produce the same label as driving classify_bottleneck
    directly from compute_roofline_inputs + the ridge formula."""
    hw = _rtx6000_ada()
    problem = _mk_matmul_problem()
    workload = _mk_matmul_workload(256, 256, 256)

    flops, nbytes = compute_roofline_inputs(problem, workload)
    ridge = (hw.peak_flops_fp32 * 1e12) / (hw.peak_memory_bandwidth_gb_s * 1e9)
    expected = classify_bottleneck(flops / nbytes, ridge)

    assert classify_workload(problem, workload, hw) == expected
