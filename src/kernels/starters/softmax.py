"""Baseline Triton softmax kernel."""

from __future__ import annotations

from src.kernels.kernel import Kernel, KernelSpec, KernelType


def make_softmax_kernel(rows: int, cols: int) -> Kernel:
    """Create a baseline softmax kernel for the given dimensions."""
    spec = KernelSpec(
        name=f"softmax_{rows}x{cols}",
        kernel_type=KernelType.SOFTMAX,
        flop_count=5 * rows * cols,  # exp + sum + div per element
        memory_bytes=2 * rows * cols * 4,  # read + write FP32
        input_shapes=[{"rows": rows, "cols": cols}],
    )
    return Kernel(spec=spec, source_code="# placeholder softmax kernel")
