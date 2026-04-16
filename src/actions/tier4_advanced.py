"""Tier 4 actions — advanced structural optimizations."""

from __future__ import annotations

from src.actions.registry import Action, ActionTier


def split_k_decomposition() -> Action:
    """Action: split the K dimension across multiple thread blocks."""
    return Action(
        id="t4_split_k",
        tier=ActionTier.ADVANCED,
        name="Split-K Decomposition",
        description="Split the K dimension across multiple thread blocks for better parallelism.",
        guidance="Placeholder guidance.",
        expected_impact="10-40% on large-K matmuls with low SM utilization.",
    )


def persistent_kernel() -> Action:
    """Action: convert to persistent kernel (single-wave launch)."""
    return Action(
        id="t4_persistent",
        tier=ActionTier.ADVANCED,
        name="Persistent Kernel",
        description="Convert to persistent kernel with single-wave launch.",
        guidance="Placeholder guidance.",
        anti_patterns=["Awkward in Triton — limited control over thread persistence."],
    )


def warp_specialization() -> Action:
    """Action: assign different warps to different roles."""
    return Action(
        id="t4_warp_spec",
        tier=ActionTier.ADVANCED,
        name="Warp Specialization",
        description="Assign different warps to producer/consumer roles.",
        guidance="Placeholder guidance.",
        anti_patterns=["Not possible in Triton V1."],
    )


def stream_k() -> Action:
    """Action: stream-K work partitioning for load balancing."""
    return Action(
        id="t4_stream_k",
        tier=ActionTier.ADVANCED,
        name="Stream-K",
        description="Stream-K work partitioning for better load balancing.",
        guidance="Placeholder guidance.",
    )


def all_actions() -> list[Action]:
    """Return all Tier 4 actions."""
    return [split_k_decomposition(), persistent_kernel(), warp_specialization(), stream_k()]
