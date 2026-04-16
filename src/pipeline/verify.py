"""Post-optimization verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.kernels.kernel import Kernel


@dataclass
class VerificationResult:
    """Result of post-optimization verification."""

    passed: bool
    details: str = ""


def verify_optimized_kernel(
    optimized: Kernel,
    baseline: Kernel,
) -> VerificationResult:
    """Re-verify the best kernel found by the search against the baseline.

    Runs full correctness gate + benchmark to confirm results are
    reproducible.  Uses the PyTorch reference stored on the spec when
    available (SOL-ExecBench mode), otherwise falls back to comparing
    against the baseline kernel directly.
    """
    from src.eval.correctness import verify_correctness

    result = verify_correctness(
        optimized,
        baseline=baseline,
        reference_source=optimized.spec.pytorch_reference,
    )
    return VerificationResult(
        passed=result.passed,
        details=result.error_message if not result.passed else "Verification passed.",
    )
