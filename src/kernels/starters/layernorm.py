"""Baseline Triton layer normalization kernel."""

from __future__ import annotations

from src.kernels.kernel import Kernel, KernelSpec, KernelType


def make_layernorm_kernel(batch: int, hidden: int) -> Kernel:
    """Create a baseline layernorm kernel for the given dimensions."""
    spec = KernelSpec(
        name=f"layernorm_{batch}x{hidden}",
        kernel_type=KernelType.LAYERNORM,
        flop_count=5 * batch * hidden,  # mean + var + normalize
        memory_bytes=2 * batch * hidden * 4,  # read + write FP32
        input_shapes=[{"batch": batch, "hidden": hidden}],
    )
    return Kernel(spec=spec, source_code="# placeholder layernorm kernel")
