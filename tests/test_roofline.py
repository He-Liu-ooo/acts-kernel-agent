"""Tests for ``classify_run`` in src/eval/roofline.py.

Pure arithmetic, no GPU, no subprocess. Torch-free: runs in the default
/tmp/acts_test_venv (pytest + pyyaml). Fixtures are built inline to keep
conftest.py unchanged.
"""

from __future__ import annotations

import pytest

from conftest import rtx6000_ada_hardware as _rtx6000_ada
from src.eval.roofline import (
    BottleneckType,
    RooflineResult,
    classify_run,
)
from src.kernels.kernel import KernelSpec, KernelType


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


