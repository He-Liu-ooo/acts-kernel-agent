"""Baseline Triton attention kernel."""

from __future__ import annotations

from src.kernels.kernel import Kernel, KernelSpec, KernelType


def make_attention_kernel(
    batch: int, heads: int, seq_len: int, head_dim: int
) -> Kernel:
    """Create a baseline attention kernel for the given dimensions."""
    spec = KernelSpec(
        name=f"attention_{batch}x{heads}x{seq_len}x{head_dim}",
        kernel_type=KernelType.ATTENTION,
        flop_count=4 * batch * heads * seq_len * seq_len * head_dim,
        memory_bytes=3 * batch * heads * seq_len * head_dim * 4,  # Q, K, V FP32
        input_shapes=[{
            "batch": batch, "heads": heads,
            "seq_len": seq_len, "head_dim": head_dim,
        }],
    )
    return Kernel(spec=spec, source_code="# placeholder attention kernel")
