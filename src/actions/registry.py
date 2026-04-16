"""Action registry and tier system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


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
    """A structured optimization action record."""

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


class ActionRegistry:
    """Complete catalog of all optimization actions, built once at startup.

    The Planner does not search this directly. The orchestrator calls
    list_applicable() to filter by kernel type and bottleneck, then injects
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
        bottleneck: str,
    ) -> list[Action]:
        """Return actions applicable to the given kernel type and bottleneck."""
        results = []
        for action in self._actions.values():
            type_match = not action.applicable_to or kernel_type in action.applicable_to
            precond_match = not action.preconditions or bottleneck in action.preconditions
            if type_match and precond_match:
                results.append(action)
        return sorted(results, key=lambda a: a.tier)


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
