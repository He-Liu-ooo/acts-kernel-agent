"""Experience dataclass for optimization memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.eval.types import BottleneckType


@dataclass
class ActionRecord:
    """Record of an action applied during optimization."""

    action_id: str
    tier: int
    name: str
    parameters: dict[str, str] = field(default_factory=dict)


@dataclass
class Experience:
    """A single optimization experience stored in memory.

    Records what was tried, what happened, and on what hardware.
    No kernel code stored — only summaries.
    """

    kernel_type: str
    action_applied: ActionRecord
    metrics: dict[str, float] = field(default_factory=dict)  # latency, sol_score
    speedup: float = 0.0
    reviewer_summary: str = ""
    bottleneck_before: BottleneckType = BottleneckType.BALANCED
    hardware: str = ""
    success: bool = False
