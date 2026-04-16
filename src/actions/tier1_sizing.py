"""Tier 1 actions — block/grid sizing and occupancy tuning."""

from __future__ import annotations

from src.actions.registry import Action, ActionTier


def block_size_tuning() -> Action:
    """Action: tune BLOCK_SIZE_M/N/K for optimal occupancy."""
    return Action(
        id="t1_block_size_tuning",
        tier=ActionTier.SIZING,
        name="Block Size Tuning",
        description="Tune BLOCK_SIZE_M/N/K for optimal occupancy and cache utilization.",
        parameters={"block_size": "32-256"},
        guidance="Placeholder guidance.",
        expected_impact="10-30% latency reduction on sub-optimal block sizes.",
    )


def grid_shape_optimization() -> Action:
    """Action: optimize grid launch dimensions."""
    return Action(
        id="t1_grid_shape",
        tier=ActionTier.SIZING,
        name="Grid Shape Optimization",
        description="Optimize grid launch dimensions for better SM utilization.",
        guidance="Placeholder guidance.",
        expected_impact="5-20% latency reduction.",
    )


def occupancy_maximization() -> Action:
    """Action: maximize SM occupancy via resource balancing."""
    return Action(
        id="t1_occupancy",
        tier=ActionTier.SIZING,
        name="Occupancy Maximization",
        description="Maximize SM occupancy by balancing registers, shared memory, and block size.",
        guidance="Placeholder guidance.",
        expected_impact="10-25% latency reduction on occupancy-limited kernels.",
    )


def all_actions() -> list[Action]:
    """Return all Tier 1 actions."""
    return [block_size_tuning(), grid_shape_optimization(), occupancy_maximization()]
