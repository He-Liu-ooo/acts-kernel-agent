"""Tier 5 actions — architecture-specific optimizations (H100/A100)."""

from __future__ import annotations

from src.actions.registry import Action, ActionTier


def h100_tma_loads() -> Action:
    """Action: use H100 Tensor Memory Accelerator for async loads."""
    return Action(
        id="t5_h100_tma",
        tier=ActionTier.ARCH_SPECIFIC,
        name="H100 TMA Loads",
        description="Use H100 Tensor Memory Accelerator for async global-to-shared loads.",
        preconditions=["compute_capability >= 9.0"],
        guidance="Placeholder guidance.",
        anti_patterns=["Mostly blocked in Triton V1."],
    )


def h100_wgmma() -> Action:
    """Action: use H100 warp-group MMA instructions."""
    return Action(
        id="t5_h100_wgmma",
        tier=ActionTier.ARCH_SPECIFIC,
        name="H100 WGMMA",
        description="Use H100 warp-group MMA instructions for higher throughput.",
        preconditions=["compute_capability >= 9.0"],
        guidance="Placeholder guidance.",
        anti_patterns=["Mostly blocked in Triton V1."],
    )


def a100_cp_async() -> Action:
    """Action: use A100 cp.async for global-to-shared copies."""
    return Action(
        id="t5_a100_cp_async",
        tier=ActionTier.ARCH_SPECIFIC,
        name="A100 cp.async",
        description="Use A100 cp.async for asynchronous global-to-shared memory copies.",
        preconditions=["compute_capability >= 8.0"],
        guidance="Placeholder guidance.",
    )


def hopper_cluster_launch() -> Action:
    """Action: use Hopper cluster launch for SM cooperation."""
    return Action(
        id="t5_hopper_cluster",
        tier=ActionTier.ARCH_SPECIFIC,
        name="Hopper Cluster Launch",
        description="Use Hopper cluster launch for cross-SM cooperation.",
        preconditions=["compute_capability >= 9.0"],
        guidance="Placeholder guidance.",
        anti_patterns=["Mostly blocked in Triton V1."],
    )


def all_actions() -> list[Action]:
    """Return all Tier 5 actions."""
    return [h100_tma_loads(), h100_wgmma(), a100_cp_async(), hopper_cluster_launch()]
