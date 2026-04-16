"""Tier 2 actions — memory optimization."""

from __future__ import annotations

from src.actions.registry import Action, ActionTier


def shared_memory_tiling() -> Action:
    """Action: tile data through shared memory for reuse."""
    return Action(
        id="t2_shared_memory_tiling",
        tier=ActionTier.MEMORY,
        name="Shared Memory Tiling",
        description="Tile data through shared memory for reuse across threads.",
        preconditions=["memory_bound"],
        guidance="Placeholder guidance.",
        expected_impact="2-5x speedup on memory-bound kernels with data reuse.",
    )


def global_memory_coalescing() -> Action:
    """Action: ensure coalesced global memory access patterns."""
    return Action(
        id="t2_coalescing",
        tier=ActionTier.MEMORY,
        name="Global Memory Coalescing",
        description="Ensure coalesced global memory access patterns.",
        preconditions=["memory_bound"],
        guidance="Placeholder guidance.",
        expected_impact="2-10x on uncoalesced access patterns.",
    )


def register_caching() -> Action:
    """Action: cache frequently accessed values in registers."""
    return Action(
        id="t2_register_caching",
        tier=ActionTier.MEMORY,
        name="Register Caching",
        description="Cache frequently accessed values in registers to reduce memory traffic.",
        preconditions=["memory_bound"],
        guidance="Placeholder guidance.",
    )


def prefetching() -> Action:
    """Action: software prefetching via num_stages pipelining."""
    return Action(
        id="t2_prefetching",
        tier=ActionTier.MEMORY,
        name="Prefetching",
        description="Software prefetching via Triton num_stages pipelining.",
        preconditions=["memory_bound"],
        parameters={"num_stages": "2-5"},
        guidance="Placeholder guidance.",
    )


def bank_conflict_resolution() -> Action:
    """Action: resolve shared memory bank conflicts."""
    return Action(
        id="t2_bank_conflict",
        tier=ActionTier.MEMORY,
        name="Bank Conflict Resolution",
        description="Resolve shared memory bank conflicts via padding or access reordering.",
        preconditions=["memory_bound"],
        guidance="Placeholder guidance.",
    )


def all_actions() -> list[Action]:
    """Return all Tier 2 actions."""
    return [
        shared_memory_tiling(), global_memory_coalescing(),
        register_caching(), prefetching(), bank_conflict_resolution(),
    ]
