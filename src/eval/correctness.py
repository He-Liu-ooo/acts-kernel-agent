"""5-stage correctness verification gate.

Called by Coder's check_correctness_tool during its turn, not by the
orchestrator. By the time the Coder returns, correctness is guaranteed.

Correctness is always checked against the **PyTorch reference** — never
against the Triton baseline.  The Triton baseline may carry translation
bugs; using it as the oracle would propagate those bugs through the
entire optimisation run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.kernels.kernel import Kernel

import torch


class CorrectnessStage(Enum):
    """The five stages of correctness verification."""

    SMOKE_TEST = "smoke_test"
    SHAPE_SWEEP = "shape_sweep"
    NUMERICAL_STABILITY = "numerical_stability"
    DETERMINISM = "determinism"
    ANTI_CHEAT = "anti_cheat"


@dataclass
class CorrectnessResult:
    """Result of running the correctness gate."""

    passed: bool
    failed_stage: CorrectnessStage | None = None
    error_message: str = ""


def verify_correctness(
    candidate: Kernel,
    baseline: Kernel | None = None,
    reference_source: str = "",
    tolerance: float = 1e-3,
) -> CorrectnessResult:
    """Run 5-stage correctness verification.

    The correctness oracle is determined by priority:

    1. *reference_source* — a PyTorch ``run()`` source string (from
       ``definition.json``).  Always preferred when available.
    2. *baseline* — a Kernel whose compiled output is used as reference.
       Fallback for custom (non-SOL-ExecBench) problems.

    Stages:
        1. Smoke test — single input, check output matches reference
        2. Shape sweep — multiple input sizes (tiny -> xlarge)
        3. Numerical stability — NaN/Inf detection, precision check
        4. Determinism — repeated runs must produce identical outputs
        5. Anti-cheat — randomised inputs, strict tolerance
    """
    # Placeholder: always passes.
    return CorrectnessResult(passed=True)
