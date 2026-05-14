"""Action registry and tier system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import HardwareSpec


class ActionTier(IntEnum):
    """Action tiers ordered by risk/reward."""

    SIZING = 1
    MEMORY = 2
    COMPUTE = 3
    ADVANCED = 4
    ARCH_SPECIFIC = 5
    KERNEL_SPECIFIC = 6


@dataclass(frozen=True)
class Action:
    """A structured optimization action record.

    ``preconditions: list[str]`` is LLM-visible documentation rendered into
    the Planner's system prompt — advisory natural-language descriptions of
    when the technique is appropriate. ``min_compute_capability`` is the
    one structured enforcement gate consulted by ``list_applicable``.
    """

    id: str
    tier: ActionTier
    name: str
    description: str
    applicable_to: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)
    guidance: str = ""
    anti_patterns: list[str] = field(default_factory=list)
    expected_impact: str = ""
    min_compute_capability: float | None = None


class ActionRegistry:
    """Complete catalog of all optimization actions, built once at startup.

    The Planner does not search this directly. The orchestrator calls
    list_applicable() to filter by kernel type and hardware, then injects
    the filtered subset into the Planner's prompt context.
    """

    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        """Register an action in the registry."""
        self._actions[action.id] = action

    def get(self, action_id: str) -> Action:
        """Look up an action by ID."""
        return self._actions[action_id]

    def list_by_tier(self, tier: ActionTier) -> list[Action]:
        """Return all actions in a given tier."""
        return [a for a in self._actions.values() if a.tier == tier]

    def list_applicable(
        self,
        kernel_type: str,
        *,
        hardware: "HardwareSpec | None" = None,
    ) -> list[Action]:
        """Return actions applicable to the given kernel type + hardware.

        Hardware gating uses ``Action.min_compute_capability``. When the
        action declares a minimum and the supplied ``hardware`` either is
        ``None`` or has ``compute_capability == 0.0`` (unknown), the
        action is excluded — default-deny on hardware-gated surfaces.
        """
        results = []
        for action in self._actions.values():
            type_match = not action.applicable_to or kernel_type in action.applicable_to
            if type_match and _hardware_meets(action, hardware):
                results.append(action)
        return sorted(results, key=lambda a: a.tier)


def _hardware_meets(action: "Action", hardware: "HardwareSpec | None") -> bool:
    """Return True iff ``hardware`` satisfies ``action``'s structured gate.

    Unknown hardware (None or cc=0.0) is **deny** — the original failure
    mode was a Hopper-only action shipping on Ada because no one enforced
    the gate; default-deny on missing info preserves that intent.
    """
    if action.min_compute_capability is None:
        return True
    if hardware is None or hardware.compute_capability == 0.0:
        return False
    return hardware.compute_capability >= action.min_compute_capability


def build_default_registry() -> ActionRegistry:
    """Build the registry populated with all built-in actions."""
    from src.actions import (
        tier1_sizing,
        tier2_memory,
        tier3_compute,
        tier4_advanced,
        tier5_arch,
        tier6_specific,
    )

    registry = ActionRegistry()
    for module in [
        tier1_sizing, tier2_memory, tier3_compute,
        tier4_advanced, tier5_arch, tier6_specific,
    ]:
        for action in module.all_actions():
            registry.register(action)
    return registry
