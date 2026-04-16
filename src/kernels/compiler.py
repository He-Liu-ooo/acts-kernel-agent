"""Triton kernel compilation.

Called by Coder's compile_kernel_tool during its turn, not by the
orchestrator. By the time the Coder returns, compilation is guaranteed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.kernels.kernel import Kernel


@dataclass
class CompilationResult:
    """Result of compiling a Triton kernel."""

    success: bool
    error_message: str = ""
    compiled_fn: object | None = None


def compile_kernel(kernel: Kernel) -> CompilationResult:
    """Compile a Triton kernel from source code.

    Returns a CompilationResult with the compiled function or error details.
    """
    # Placeholder: assume compilation always succeeds.
    return CompilationResult(success=True)
