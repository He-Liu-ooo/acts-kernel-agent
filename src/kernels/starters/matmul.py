"""Baseline Triton matrix multiplication kernel."""

from __future__ import annotations

from src.kernels.kernel import Kernel, KernelSpec, KernelType


def make_matmul_kernel(M: int, N: int, K: int) -> Kernel:
    """Create a baseline matmul kernel for the given dimensions."""
    spec = KernelSpec(
        name=f"matmul_{M}x{N}x{K}",
        kernel_type=KernelType.MATMUL,
        flop_count=2 * M * N * K,
        memory_bytes=(M * K + K * N + M * N) * 4,  # FP32
        input_shapes=[{"M": M, "N": N, "K": K}],
    )
    return Kernel(spec=spec, source_code="# placeholder matmul kernel")
