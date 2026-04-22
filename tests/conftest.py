"""Shared test helpers — torch-free stand-ins for the correctness/verify pipeline."""

from __future__ import annotations

import math

import pytest

from src.config import HardwareSpec
from src.eval.correctness import ComparisonResult
from src.kernels.kernel import KernelSpec, KernelType


class ScalarPolicy:
    """ComparisonPolicy over Python floats — drives correctness tests without torch."""

    def compare(self, output, expected, *, atol: float, rtol: float) -> ComparisonResult:
        err = abs(output - expected)
        threshold = atol + rtol * abs(expected)
        match = err <= threshold
        return ComparisonResult(
            match=match,
            max_abs_error=err,
            reason="" if match else f"abs_err={err} > atol+rtol*|exp|={threshold}",
        )

    def contains_non_finite(self, output) -> bool:
        return math.isnan(output) or math.isinf(output)

    def bitwise_equal(self, a, b) -> bool:
        return a == b


@pytest.fixture
def scalar_policy() -> ScalarPolicy:
    return ScalarPolicy()


def scalar_ref(x: float) -> float:
    return x * 2.0


def scalar_gen(seed: int) -> tuple[float]:
    return (float(seed + 1),)


def make_kernel_spec(
    name: str = "test_kernel",
    entrypoint: str = "kernel_fn",
) -> KernelSpec:
    return KernelSpec(
        name=name,
        kernel_type=KernelType.ELEMENTWISE,
        entrypoint=entrypoint,
    )


def rtx6000_ada_hardware() -> HardwareSpec:
    """Dev-machine GPU spec — ~64 TFLOPS fp32, ~960 GB/s DRAM, ridge
    ~66.7 ops/byte. Shared across profiler/orchestrator tests so the
    roofline math stays consistent across tiers."""
    return HardwareSpec(
        name="RTX6000Ada",
        freq_GHz=2.5,
        SRAM_capacity=98_304 * 1024,
        SRAM_byte_per_cycle=4000.0,
        DRAM_capacity=48 * 1024**3,
        DRAM_byte_per_cycle=384.0,
        MAC_per_cycle_fp32_sm=12_800.0,
        MAC_per_cycle_fp16_tc=512_000.0,
        MAC_per_cycle_bf16_tc=512_000.0,
    )


@pytest.fixture(name="rtx6000_ada_hardware")
def rtx6000_ada_hardware_fixture() -> HardwareSpec:
    """Fixture form of ``rtx6000_ada_hardware()`` for tests that prefer
    pytest-style injection."""
    return rtx6000_ada_hardware()
