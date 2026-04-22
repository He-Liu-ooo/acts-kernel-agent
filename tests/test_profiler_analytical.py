"""Tests for ``_compute_analytical`` — the zero-overhead roofline metrics
path inside ``src/eval/profiler.py``.

Pure arithmetic, no GPU, no subprocess. Torch-free: runs in the default
``/tmp/acts_test_venv`` (pytest + pyyaml). Classification thresholds live
in ``tests/test_roofline.py`` now — this file only exercises the
per-iteration runtime metrics.
"""

from __future__ import annotations

import pytest

from conftest import rtx6000_ada_hardware as _rtx6000_ada
from src.config import HardwareSpec
from src.eval.profiler import (
    AnalyticalMetrics,
    ProfilerError,
    _compute_analytical,
)


def _hw_peaks(hw: HardwareSpec) -> tuple[float, float, float]:
    """Return (peak_tflops, peak_bw_gb_s, ridge_point) for a spec."""
    peak_tflops = hw.peak_flops_fp32
    peak_bw = hw.peak_memory_bandwidth_gb_s
    ridge = (peak_tflops * 1e12) / (peak_bw * 1e9)
    return peak_tflops, peak_bw, ridge


# ── structural ─────────────────────────────────────────────────────────────


def test_returns_analytical_metrics_dataclass():
    hw = _rtx6000_ada()
    result = _compute_analytical(
        flops=1_000_000,
        nbytes=4_000_000,
        latency_s=1e-3,
        hardware_spec=hw,
    )
    assert isinstance(result, AnalyticalMetrics)
    assert not hasattr(result, "classification")


def test_all_fields_populated_and_nonnegative():
    hw = _rtx6000_ada()
    r = _compute_analytical(
        flops=2_000_000,
        nbytes=8_000_000,
        latency_s=1e-3,
        hardware_spec=hw,
    )
    assert r.arithmetic_intensity >= 0
    assert r.ridge_point > 0
    assert r.achieved_tflops >= 0
    assert r.achieved_bandwidth_gb_s >= 0
    assert r.pct_peak_compute >= 0
    assert r.pct_peak_bandwidth >= 0


# ── math ───────────────────────────────────────────────────────────────────


def test_arithmetic_intensity_is_flops_over_bytes():
    hw = _rtx6000_ada()
    r = _compute_analytical(
        flops=1_000_000,
        nbytes=4_000_000,
        latency_s=1e-3,
        hardware_spec=hw,
    )
    assert r.arithmetic_intensity == pytest.approx(0.25)


def test_ridge_point_matches_hardware():
    hw = _rtx6000_ada()
    _, _, expected_ridge = _hw_peaks(hw)
    r = _compute_analytical(
        flops=1,
        nbytes=1,
        latency_s=1e-3,
        hardware_spec=hw,
    )
    assert r.ridge_point == pytest.approx(expected_ridge)


def test_achieved_tflops_and_bandwidth():
    hw = _rtx6000_ada()
    # latency 1ms, flops 1e9 → 1 TFLOPS; nbytes 1e9 → 1 TB/s = 1000 GB/s
    r = _compute_analytical(
        flops=1_000_000_000,
        nbytes=1_000_000_000,
        latency_s=1e-3,
        hardware_spec=hw,
    )
    assert r.achieved_tflops == pytest.approx(1.0)
    assert r.achieved_bandwidth_gb_s == pytest.approx(1000.0)


def test_pct_peak_fractions_in_zero_to_one_plus():
    hw = _rtx6000_ada()
    peak_tflops, peak_bw, _ = _hw_peaks(hw)
    # Sized so achieved ≈ 50% of peak bandwidth.
    nbytes = int(peak_bw * 1e9 * 0.5 * 1e-3)
    r = _compute_analytical(
        flops=1,
        nbytes=nbytes,
        latency_s=1e-3,
        hardware_spec=hw,
    )
    assert r.pct_peak_bandwidth == pytest.approx(0.5, rel=1e-3)
    assert 0.0 <= r.pct_peak_compute <= 0.001


# ── error paths ────────────────────────────────────────────────────────────


def test_zero_latency_raises():
    hw = _rtx6000_ada()
    with pytest.raises(ProfilerError, match="latency"):
        _compute_analytical(flops=1, nbytes=1, latency_s=0.0, hardware_spec=hw)


def test_negative_latency_raises():
    hw = _rtx6000_ada()
    with pytest.raises(ProfilerError, match="latency"):
        _compute_analytical(flops=1, nbytes=1, latency_s=-1e-6, hardware_spec=hw)


def test_zero_nbytes_raises():
    hw = _rtx6000_ada()
    with pytest.raises(ProfilerError, match="nbytes"):
        _compute_analytical(flops=1, nbytes=0, latency_s=1e-3, hardware_spec=hw)


def test_negative_nbytes_raises():
    hw = _rtx6000_ada()
    with pytest.raises(ProfilerError, match="nbytes"):
        _compute_analytical(flops=1, nbytes=-1, latency_s=1e-3, hardware_spec=hw)


def test_zeroed_hardware_peaks_raise():
    """Ridge point and pct-of-peak are undefined without hardware peaks.

    HardwareSpec() with all-zero fields (detect_hardware fallback) cannot
    produce meaningful metrics — treat as config bug, fail fast.
    """
    hw = HardwareSpec()  # all zeros
    with pytest.raises(ProfilerError, match="hardware"):
        _compute_analytical(flops=1, nbytes=1, latency_s=1e-3, hardware_spec=hw)


def test_zero_flops_is_ok():
    """Pure memory ops (no arithmetic) are valid input. AI = 0 is fine."""
    hw = _rtx6000_ada()
    r = _compute_analytical(flops=0, nbytes=1_000_000, latency_s=1e-3, hardware_spec=hw)
    assert r.arithmetic_intensity == 0.0
    assert r.achieved_tflops == 0.0
