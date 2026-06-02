"""Tier 2 actions — memory optimization."""

from __future__ import annotations

from src.actions.registry import Action, ActionTier


def shared_memory_tiling() -> Action:
    """Action: tile data through shared memory for reuse."""
    return Action(
        id="t2_shared_memory_tiling",
        tier=ActionTier.MEMORY,
        name="Shared Memory Tiling",
        description=(
            "Increase reuse of frequently-loaded operands so each is read once "
            "per block and reused across the inner loop."
        ),
        preconditions=["memory_bound"],
        guidance=(
            "In Triton, shared-memory tiling is IMPLICIT — load block tiles with "
            "`tl.load` and feed them to `tl.dot` (the compiler stages the operands "
            "through shared memory for you), then tune `num_stages` (software "
            "pipelining) and `BLOCK_M`/`BLOCK_N`/`BLOCK_K` for overlap and reuse. "
            "Triton has NO explicit shared-memory allocation API — `tl.static_shared_memory` "
            "and `tl.static_shared` do not exist and will fail to compile. Tile size is "
            "bounded by shared memory per SM divided by concurrent blocks — oversizing "
            "collapses occupancy."
        ),
        anti_patterns=[
            "Tiling data with no reuse — pure copy overhead, no speedup.",
            "Tile size that leaves only one block per SM resident.",
        ],
        expected_impact="Often substantial when data has reuse; near-zero on streaming kernels.",
    )


def global_memory_coalescing() -> Action:
    """Action: ensure coalesced global memory access patterns."""
    return Action(
        id="t2_coalescing",
        tier=ActionTier.MEMORY,
        name="Global Memory Coalescing",
        description="Ensure coalesced global memory access patterns.",
        preconditions=["memory_bound"],
        guidance=(
            "Adjacent threads in a warp must read adjacent addresses. Strided or transposed access "
            "patterns issue separate transactions per thread, wasting bandwidth. Restructure the index "
            "computation, swap the loop nest, or stage through shared memory to coalesce. Verify with "
            "the profiler's `gld_efficiency` (should be ≥80%)."
        ),
        anti_patterns=[
            "Coalescing without measuring — pretty access patterns aren't always faster.",
            "Transposing in global memory instead of staging through shared memory.",
        ],
        expected_impact="Often large on memory-bound kernels with uncoalesced loads; nothing otherwise.",
    )


def register_caching() -> Action:
    """Action: cache frequently accessed values in registers."""
    return Action(
        id="t2_register_caching",
        tier=ActionTier.MEMORY,
        name="Register Caching",
        description="Cache frequently accessed values in registers to reduce memory traffic.",
        preconditions=["memory_bound"],
        guidance=(
            "Hoist values out of inner loops into per-thread registers when the same address is hit "
            "many times. Triton handles this automatically for scalars; for small vectors, store the "
            "loaded values in a `tl.zeros`-initialized accumulator and reuse. Watch register pressure — "
            "spilling to local memory is worse than the original load."
        ),
        anti_patterns=[
            "Hoisting large arrays into registers — causes spilling, regressive.",
            "Caching values that are only read once.",
        ],
        expected_impact="Typically modest; can be substantial when the same address is hit many times.",
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
        guidance=(
            "Pass `num_stages=N` to `triton.autotune` or the Triton kernel decorator to overlap the "
            "next iteration's loads with the current iteration's compute. Start with `num_stages=2`; "
            "increase only if shared memory budget allows. Each stage roughly doubles the per-block "
            "shared memory footprint."
        ),
        anti_patterns=[
            "Increasing num_stages until shared memory overflows — drops occupancy to 1 block/SM.",
            "Setting num_stages on kernels with no obvious load/compute overlap (e.g., elementwise).",
        ],
        expected_impact="Typically small-to-moderate; matters most on inner loops with cheap compute per load.",
    )


def bank_conflict_resolution() -> Action:
    """Action: resolve shared memory bank conflicts."""
    return Action(
        id="t2_bank_conflict",
        tier=ActionTier.MEMORY,
        name="Bank Conflict Resolution",
        description="Resolve shared memory bank conflicts via padding or access reordering.",
        preconditions=["memory_bound"],
        guidance=(
            "Multiple threads in a warp hitting the same shared-memory bank serialize. The classic fix "
            "for power-of-two strides is to pad the inner dimension by one element so adjacent rows "
            "land on different banks. Triton's swizzling (`tl.swizzle2d`) handles this for matmul-shaped "
            "tiles. NCU's `shared_load_transactions_per_request` reveals conflicts."
        ),
        anti_patterns=[
            "Padding without verifying conflicts exist — wastes shared memory budget.",
            "Hand-coding swizzles when `tl.dot` with a standard tile shape already does it.",
        ],
        expected_impact="Typically small; rarely the dominant factor unless shared memory is the bottleneck.",
    )


def all_actions() -> list[Action]:
    """Return all Tier 2 actions."""
    return [
        shared_memory_tiling(), global_memory_coalescing(),
        register_caching(), prefetching(), bank_conflict_resolution(),
    ]
