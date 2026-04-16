"""Triton baseline generation from PyTorch references.

At problem-load time the Coder agent produces a one-shot PyTorch-to-Triton
translation.  The result becomes the root of the search tree and the
anchor for T_b in the SOL score formula (S = 0.5 at baseline).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.coder import CoderAgent
    from src.benchmark.problem import Problem
    from src.kernels.kernel import Kernel, KernelSpec


async def generate_triton_baseline(
    problem: Problem,
    spec: KernelSpec,
    coder: CoderAgent,
    max_retries: int = 3,
) -> Kernel | None:
    """Generate a functionally-correct Triton baseline from a PyTorch reference.

    1. Coder receives the PyTorch reference and problem definition.
    2. Coder produces a Triton kernel source string.
    3. Correctness is verified against the PyTorch reference (5-stage gate).
    4. On failure, retry up to *max_retries* times.
    5. Return ``None`` if all retries fail (problem should be skipped).

    Placeholder: returns a stub Kernel whose source is the PyTorch reference
    comment-wrapped.  Real implementation requires the Coder agent + SDK.
    """
    from src.kernels.kernel import Kernel

    # Placeholder: return a stub Triton kernel.
    stub_source = (
        "# Placeholder Triton baseline — real implementation uses Coder agent\n"
        "# to translate the PyTorch reference into Triton.\n"
        f"# Problem: {problem.name}\n"
        "import triton\nimport triton.language as tl\n"
    )
    return Kernel(spec=spec, source_code=stub_source)
