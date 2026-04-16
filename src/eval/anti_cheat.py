"""Anti-cheat mechanisms for correctness verification.

Called as part of the correctness gate (stage 5) inside the Coder's
check_correctness_tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.kernels.kernel import Kernel

import torch


def generate_randomized_inputs(
    kernel: Kernel,
    seed: int | None = None,
) -> list[torch.Tensor]:
    """Generate randomized inputs for anti-cheat correctness testing.

    Prevents kernels from caching outputs or hardcoding expected values.
    """
    # Placeholder: return a single random tensor.
    if seed is not None:
        torch.manual_seed(seed)
    return [torch.randn(64, 64)]


def strict_tolerance_check(
    candidate_output: torch.Tensor,
    baseline_output: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Check outputs match within strict tolerances."""
    return bool(torch.allclose(candidate_output, baseline_output, rtol=rtol, atol=atol))
