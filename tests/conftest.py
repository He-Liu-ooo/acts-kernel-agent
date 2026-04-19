"""Shared test helpers — torch-free stand-ins for the correctness/verify pipeline."""

from __future__ import annotations

import math

import pytest

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
