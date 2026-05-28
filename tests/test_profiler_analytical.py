"""Tests for ``_compute_analytical`` — the zero-overhead roofline metrics
path inside ``src/eval/profiler.py``.

Pure arithmetic, no GPU, no subprocess. Torch-free: runs in the default
``~/.venvs/acts_test_venv`` (pytest + pyyaml). Classification thresholds live
in ``tests/test_roofline.py`` now — this file only exercises the
per-iteration runtime metrics.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import rtx6000_ada_hardware as _rtx6000_ada
from src.config import HardwareSpec
from src.eval.profiler import (
    AnalyticalMetrics,
    ProfilerError,
    _collect_input_dtypes,
    _compute_analytical,
    _pick_compute_peak,
)


def _hw_peaks(hw: HardwareSpec) -> tuple[float, float]:
    """Return (peak_tflops, peak_bw_gb_s) for a spec.

    Note: ridge_point is no longer a per-iter ``AnalyticalMetrics`` field;
    it lives on ``RooflineResult`` (run-level invariant). Tests for
    ridge_point math live in ``tests/test_roofline.py``.
    """
    return hw.peak_flops_fp32, hw.peak_memory_bandwidth_gb_s


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
    # AI / ridge_point are run-level invariants on RooflineResult — they
    # must not leak back onto AnalyticalMetrics.
    assert not hasattr(result, "arithmetic_intensity")
    assert not hasattr(result, "ridge_point")


def test_all_fields_populated_and_nonnegative():
    hw = _rtx6000_ada()
    r = _compute_analytical(
        flops=2_000_000,
        nbytes=8_000_000,
        latency_s=1e-3,
        hardware_spec=hw,
    )
    assert r.achieved_tflops >= 0
    assert r.achieved_bandwidth_gb_s >= 0
    assert r.pct_peak_compute >= 0
    assert r.pct_peak_bandwidth >= 0


# ── math ───────────────────────────────────────────────────────────────────


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
    peak_tflops, peak_bw = _hw_peaks(hw)
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
    """Pure memory ops (no arithmetic) are valid input. achieved_tflops=0
    is fine (and AI=0 lives on RooflineResult, not here)."""
    hw = _rtx6000_ada()
    r = _compute_analytical(flops=0, nbytes=1_000_000, latency_s=1e-3, hardware_spec=hw)
    assert r.achieved_tflops == 0.0


# ── _pick_compute_peak (dtype-aware denominator) ───────────────────────────


def test_pick_compute_peak_bfloat16_input():
    hw = _rtx6000_ada()
    peak, label, warn = _pick_compute_peak(["bfloat16"], hw)
    assert peak == pytest.approx(hw.peak_flops_bf16)
    assert label == "bf16"
    assert warn is False


def test_pick_compute_peak_lowest_precision_wins():
    hw = _rtx6000_ada()
    # bf16 should beat fp32 — lowest precision wins
    peak, label, warn = _pick_compute_peak(["bfloat16", "float32"], hw)
    assert peak == pytest.approx(hw.peak_flops_bf16)
    assert label == "bf16"
    assert warn is False


def test_pick_compute_peak_empty_list_falls_back_to_fp32():
    hw = _rtx6000_ada()
    peak, label, warn = _pick_compute_peak([], hw)
    assert peak == pytest.approx(hw.peak_flops_fp32)
    assert label == "fp32_fallback"
    assert warn is True


def test_pick_compute_peak_none_falls_back_to_fp32():
    hw = _rtx6000_ada()
    peak, label, warn = _pick_compute_peak(None, hw)
    assert peak == pytest.approx(hw.peak_flops_fp32)
    assert label == "fp32_fallback"
    assert warn is True


def test_pick_compute_peak_unknown_dtype_falls_back_to_fp32():
    hw = _rtx6000_ada()
    peak, label, warn = _pick_compute_peak(["mysterious_dtype"], hw)
    assert peak == pytest.approx(hw.peak_flops_fp32)
    assert label == "fp32_fallback"
    assert warn is True


def test_pick_compute_peak_bf16_zero_cascades_to_next_nonzero_peak():
    # Placeholder-style hardware: bf16/fp16/tf32 zeroed but fp32 populated.
    # The bf16 input dtype should cascade up the precision ladder.
    hw = replace(
        _rtx6000_ada(),
        MAC_per_cycle_bf16_tc=0.0,
        MAC_per_cycle_fp16_tc=0.0,
        MAC_per_cycle_tf32_tc=0.0,
    )
    peak, label, warn = _pick_compute_peak(["bfloat16"], hw)
    assert peak == pytest.approx(hw.peak_flops_fp32)
    assert label == "fp32_fallback"
    assert warn is True


def test_pick_compute_peak_all_zero_returns_zero_with_warning():
    # Fully-zero spec: helper returns (0.0, "fp32_fallback", True). Caller
    # (_compute_analytical) is responsible for raising ProfilerError on the
    # zero peak; this helper does not raise.
    hw = HardwareSpec()
    peak, label, warn = _pick_compute_peak(["bfloat16"], hw)
    assert peak == 0.0
    assert label == "fp32_fallback"
    assert warn is True


def test_pick_compute_peak_fp32_input_label_is_plain_fp32():
    hw = _rtx6000_ada()
    peak, label, warn = _pick_compute_peak(["float32"], hw)
    assert peak == pytest.approx(hw.peak_flops_fp32)
    assert label == "fp32"   # legitimate fp32 choice — NOT the fallback label
    assert warn is False


def test_pick_compute_peak_bf16_alias_accepted():
    hw = _rtx6000_ada()
    peak1, label1, _ = _pick_compute_peak(["bfloat16"], hw)
    peak2, label2, _ = _pick_compute_peak(["bf16"], hw)
    assert peak1 == peak2
    assert label1 == label2 == "bf16"


# ── _compute_analytical(input_dtypes=…) integration ────────────────────────


def test_compute_analytical_uses_bf16_peak_when_dtype_bfloat16():
    hw = _rtx6000_ada()
    # Size achieved TFLOPS to ~10% of bf16 peak so the ratio is checkable.
    target = hw.peak_flops_bf16 * 0.10
    latency_s = 1e-3
    flops = int(target * latency_s * 1e12)
    out = _compute_analytical(
        flops=flops, nbytes=1024, latency_s=latency_s,
        hardware_spec=hw, input_dtypes=["bfloat16"],
    )
    assert out.compute_peak_dtype == "bf16"
    assert out.compute_peak_calibration_warning is False
    assert 0.099 < out.pct_peak_compute < 0.101


def test_compute_analytical_falls_back_to_fp32_when_no_dtypes():
    hw = _rtx6000_ada()
    target = hw.peak_flops_fp32 * 0.10
    latency_s = 1e-3
    flops = int(target * latency_s * 1e12)
    out = _compute_analytical(
        flops=flops, nbytes=1024, latency_s=latency_s,
        hardware_spec=hw, input_dtypes=None,
    )
    assert out.compute_peak_dtype == "fp32_fallback"
    assert out.compute_peak_calibration_warning is True
    assert 0.099 < out.pct_peak_compute < 0.101


def test_compute_analytical_legitimate_fp32_input_label():
    hw = _rtx6000_ada()
    target = hw.peak_flops_fp32 * 0.10
    flops = int(target * 1e-3 * 1e12)
    out = _compute_analytical(
        flops=flops, nbytes=1024, latency_s=1e-3,
        hardware_spec=hw, input_dtypes=["float32"],
    )
    assert out.compute_peak_dtype == "fp32"
    assert out.compute_peak_calibration_warning is False


def test_compute_analytical_raises_on_all_zero_hardware_even_with_dtypes():
    hw = HardwareSpec()  # fully zero
    with pytest.raises(ProfilerError, match="hardware"):
        _compute_analytical(
            flops=1, nbytes=1, latency_s=1e-3,
            hardware_spec=hw, input_dtypes=["bfloat16"],
        )


def test_compute_analytical_input_dtypes_is_optional_kwarg():
    """Back-compat: pre-existing call sites that omit input_dtypes still
    work; they get the fp32_fallback path."""
    hw = _rtx6000_ada()
    out = _compute_analytical(
        flops=1, nbytes=1, latency_s=1e-3, hardware_spec=hw,
    )
    assert out.compute_peak_dtype == "fp32_fallback"
    assert out.compute_peak_calibration_warning is True


# ── profile_kernel(input_dtypes=…) integration ─────────────────────────────


def test_profile_kernel_forwards_input_dtypes_to_analytical(monkeypatch):
    """``profile_kernel`` threads ``input_dtypes`` into ``_compute_analytical``.

    ``ACTS_DISABLE_NCU=1`` short-circuits the NCU subprocess so the test
    runs without an NCU binary; the analytical path still executes and is
    what we assert on.
    """
    from src.eval.profiler import profile_kernel
    from src.kernels.kernel import Kernel, KernelSpec, KernelType

    monkeypatch.setenv("ACTS_DISABLE_NCU", "1")

    kernel = Kernel(
        spec=KernelSpec(
            name="fake",
            kernel_type=KernelType.ELEMENTWISE,
            entrypoint="fake_kernel",
        ),
        source_code="def fake_kernel(): pass\n",
    )
    hw = _rtx6000_ada()
    achieved_tflops = hw.peak_flops_bf16 * 0.10
    latency_s = 1e-3
    flops = int(achieved_tflops * latency_s * 1e12)

    result = profile_kernel(
        kernel,
        {"uuid": "wl-0", "axes": {}, "inputs": {}},
        lambda seed=0: (),  # input_generator
        hardware_spec=hw,
        flops=flops,
        nbytes=1024,
        latency_s=latency_s,
        input_dtypes=["bfloat16"],
    )
    assert result.analytical is not None
    assert result.analytical.compute_peak_dtype == "bf16"
    assert result.analytical.compute_peak_calibration_warning is False
    assert 0.099 < result.analytical.pct_peak_compute < 0.101


# ── _collect_input_dtypes (call-site adapter) ──────────────────────────────


class _FakeTensor:
    """Stand-in for torch.Tensor — only needs a ``.dtype`` attribute. The
    helper stringifies it via ``str(t.dtype).removeprefix('torch.')``."""

    def __init__(self, dtype_name: str) -> None:
        # Mimic torch's repr: ``torch.bfloat16``.
        self.dtype = f"torch.{dtype_name}"


def test_collect_input_dtypes_tuple_of_tensors():
    args = (_FakeTensor("bfloat16"), _FakeTensor("float32"))
    assert _collect_input_dtypes(args) == ["bfloat16", "float32"]


def test_collect_input_dtypes_args_kwargs_shape():
    args = (_FakeTensor("bfloat16"),)
    kwargs = {"weight": _FakeTensor("bfloat16"), "bias": _FakeTensor("float32")}
    out = _collect_input_dtypes((args, kwargs))
    assert sorted(out) == ["bfloat16", "bfloat16", "float32"]


def test_collect_input_dtypes_dict_of_tensors():
    inputs = {"x": _FakeTensor("bfloat16"), "w": _FakeTensor("bfloat16")}
    assert sorted(_collect_input_dtypes(inputs)) == ["bfloat16", "bfloat16"]


def test_collect_input_dtypes_skips_non_tensor_items():
    args = (_FakeTensor("bfloat16"), 1024, "foo", None)
    assert _collect_input_dtypes(args) == ["bfloat16"]


def test_collect_input_dtypes_empty_returns_empty():
    assert _collect_input_dtypes(()) == []
    assert _collect_input_dtypes(None) == []
    assert _collect_input_dtypes((1, 2, "foo")) == []
