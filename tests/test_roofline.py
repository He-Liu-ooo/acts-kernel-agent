"""Tests for ``classify_run`` in src/eval/roofline.py.

Pure arithmetic, no GPU, no subprocess. Torch-free: runs in the default
~/.venvs/acts_test_venv (pytest + pyyaml). Fixtures are built inline to keep
conftest.py unchanged.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from conftest import rtx6000_ada_hardware as _rtx6000_ada
from src.benchmark.solar_adapter import SolarResult
from src.eval.roofline import (
    BottleneckType,
    RooflineResult,
    classify_bottleneck,
    classify_run,
    compute_roofline,
    derive_t_sol_from_solar,
)
from src.kernels.kernel import KernelSpec, KernelType


# ── classify_run ────────────────────────────────────────────────────────────


def test_classify_run_uses_solar_when_provided():
    hw = _rtx6000_ada()
    rr = RooflineResult(
        t_sol_us=1.0,
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


# ── classify_bottleneck edge cases ──────────────────────────────────────


def test_classify_bottleneck_zero_ai_is_memory_bound():
    """``classify_bottleneck(0.0, ridge_point)`` must classify as
    memory-bound. Mirrors the SOLAR no-MAC case (rmsnorm, softmax,
    reductions) where SOLAR's einsum analyzer reports AI=0.0; the band
    classifier must still produce a sensible bottleneck for that input."""
    assert (
        classify_bottleneck(arithmetic_intensity=0.0, ridge_point=189.8)
        == BottleneckType.MEMORY_BOUND
    )


# ── compute_roofline populates AI / ridge_point in MACs/byte ───────────


def test_compute_roofline_arithmetic_intensity_is_macs_per_byte():
    """``compute_roofline`` (built-in fallback) must produce AI in
    MACs/byte — converting from KernelSpec.flop_count via FLOPs/2 ≈ MACs.
    """
    hw = _rtx6000_ada()
    spec = KernelSpec(
        name="t",
        kernel_type=KernelType.MATMUL,
        flop_count=1_000_000,
        memory_bytes=4_000_000,
    )
    result = compute_roofline(spec, hw)
    # FLOPs/byte would be 1e6 / 4e6 = 0.25; MACs/byte is half that.
    assert result.arithmetic_intensity == pytest.approx(0.125)


def test_compute_roofline_ridge_point_is_macs_per_byte():
    """``compute_roofline``'s ridge_point must be in MACs/byte, i.e.
    half of the FLOPs/byte ridge."""
    hw = _rtx6000_ada()
    spec = KernelSpec(
        name="t",
        kernel_type=KernelType.MATMUL,
        flop_count=1,
        memory_bytes=1,
    )
    result = compute_roofline(spec, hw)
    expected_ridge = (hw.peak_flops_fp32 / 2.0) * 1e12 / (hw.peak_memory_bandwidth_gb_s * 1e9)
    assert result.ridge_point == pytest.approx(expected_ridge)


def test_compute_roofline_zero_flops_yields_zero_ai():
    """A pure-memory kernel (flop_count=0) gets AI=0 — no MACs at all.
    Documents the FLOPs→MACs heuristic boundary case."""
    hw = _rtx6000_ada()
    spec = KernelSpec(
        name="t",
        kernel_type=KernelType.ELEMENTWISE,
        flop_count=0,
        memory_bytes=1_000_000,
    )
    result = compute_roofline(spec, hw)
    assert result.arithmetic_intensity == 0.0


def test_roofline_result_default_ai_and_ridge_are_zero():
    """Defaults exist so callers constructing ``RooflineResult`` without
    caring about AI / ridge_point (legacy fixtures) don't break."""
    rr = RooflineResult(t_sol_us=1.0, bottleneck=BottleneckType.MEMORY_BOUND)
    assert rr.arithmetic_intensity == 0.0
    assert rr.ridge_point == 0.0


# ── derive_t_sol_from_solar threads SOLAR's precision-aware ridge ──────


def test_derive_t_sol_from_solar_uses_solar_ridge_point():
    """``RooflineResult.ridge_point`` for SOLAR-sourced runs must come
    from ``SolarResult.ridge_point`` (SOLAR's precision-aware
    ``MAC_per_cycle / DRAM_byte_per_cycle``), NOT a locally-computed
    FP32-derived ridge. For tensor-core workloads the FP32-derived value
    is up to 4× too low and silently mis-classifies them as
    compute-bound. Use a value distinct from the FP32-derived ridge so a
    regression to local computation would fail this assertion."""
    hw = _rtx6000_ada()
    # bf16 tensor-core ridge for RTX 6000 Ada is ~190 MACs/byte —
    # nowhere near the ~47.5 the old FP32-derived path produced.
    solar_ridge = 189.8
    fake_solar_result = SolarResult(
        t_sol_us=12.3,
        bottleneck=BottleneckType.MEMORY_BOUND,
        arithmetic_intensity=12.5,
        ridge_point=solar_ridge,
    )

    # Patch the SOLAR adapter call site inside roofline.py — the function
    # imports it lazily so we patch the source module.
    with patch(
        "src.benchmark.solar_adapter.derive_t_sol",
        return_value=fake_solar_result,
    ):
        result = derive_t_sol_from_solar(
            definition=None,  # adapter is mocked, so unused
            workload=None,
            hardware_spec=hw,
        )

    assert result is not None
    assert result.source == "solar"
    assert result.ridge_point == pytest.approx(solar_ridge)
    # And it should NOT match the FP32-derived ridge (regression guard).
    fp32_ridge = (hw.peak_flops_fp32 / 2.0) * 1e12 / (hw.peak_memory_bandwidth_gb_s * 1e9)
    assert abs(result.ridge_point - fp32_ridge) > 1.0


def test_derive_t_sol_from_solar_returns_none_when_adapter_returns_none():
    """When SOLAR is unavailable / pipeline fails, the adapter returns
    ``None`` and so must ``derive_t_sol_from_solar`` — caller falls back
    to ``compute_roofline``."""
    hw = _rtx6000_ada()
    with patch(
        "src.benchmark.solar_adapter.derive_t_sol",
        return_value=None,
    ):
        result = derive_t_sol_from_solar(
            definition=None, workload=None, hardware_spec=hw,
        )
    assert result is None


