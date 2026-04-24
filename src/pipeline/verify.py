"""Post-optimization verification.

After the search loop picks a best node, we recompile the winner and
re-run the 5-stage correctness gate — belt-and-braces check before the
result is declared green. Mirrors the Coder's correctness tool but is
driven from the pipeline layer, not the SDK tool loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.runtime.events import emit

if TYPE_CHECKING:
    from src.eval.correctness import ComparisonPolicy
    from src.kernels.kernel import Kernel


@dataclass
class VerificationResult:
    """Result of post-optimization verification."""

    passed: bool
    details: str = ""


def verify_optimized_kernel(
    optimized: Kernel,
    *,
    reference_fn: Callable[..., Any],
    input_generator: Callable[[int], tuple],
    policy: ComparisonPolicy | None = None,
    cache_dir: Path | None = None,
) -> VerificationResult:
    """Re-verify the best kernel found by the search against the PyTorch reference.

    Compiles ``optimized`` afresh and runs the 5-stage correctness gate.
    Returns a ``VerificationResult`` describing the outcome — compile
    failures surface as ``passed=False`` with a compile-phrased detail
    string.
    """
    from src.eval.correctness import verify_correctness
    from src.kernels.compiler import compile_kernel

    emit("verify_start")

    compiled = compile_kernel(optimized, cache_dir=cache_dir)
    if not compiled.success:
        details = f"Compilation failed:\n{compiled.error_message}"
        emit("verify_done", passed=False, detail_short=details[:200])
        return VerificationResult(passed=False, details=details)

    result = verify_correctness(
        candidate_fn=compiled.compiled_fn,
        reference_fn=reference_fn,
        input_generator=input_generator,
        policy=policy,
    )
    details = result.error_message if not result.passed else "Verification passed."
    emit("verify_done", passed=bool(result.passed), detail_short=str(details)[:200])
    return VerificationResult(passed=result.passed, details=details)
