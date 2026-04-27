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
        guidance=(
            "Pick block sizes that fit the SM register/shared-memory budget while "
            "leaving headroom for ≥2 concurrent blocks per SM. Powers of two are "
            "the safe default; for reductions, a block that covers the full row "
            "in one pass eliminates inter-block reduction but caps occupancy. "
            "Sweep candidates by halving from the current value first — a too-large "
            "block almost always hurts more than a too-small one."
        ),
        anti_patterns=[
            "Picking BLOCK_SIZE = problem dimension on small inputs — kills occupancy.",
            "Tuning block size before the kernel is correct — masks bugs as 'wrong shape'.",
        ],
        expected_impact="Typically modest; occasionally large when occupancy is the binding constraint.",
    )


def grid_shape_optimization() -> Action:
    """Action: optimize grid launch dimensions."""
    return Action(
        id="t1_grid_shape",
        tier=ActionTier.SIZING,
        name="Grid Shape Optimization",
        description="Optimize grid launch dimensions for better SM utilization.",
        guidance=(
            "Total program count should be a multiple of SM count, ideally 2–4× SM count "
            "to amortize tail effects from uneven block runtimes. For 2D grids on matmul-shaped "
            "problems, prefer block-swizzled launch order (group N consecutive program_ids on "
            "the same row) to improve L2 reuse. Don't bother with grid swizzling on small problems "
            "where one wave saturates."
        ),
        anti_patterns=[
            "Launching fewer programs than SM count — leaves SMs idle.",
            "Hand-rolling complex grid swizzles before profiling shows L2 misses dominate.",
        ],
        expected_impact="Typically small; matters most on matmul-shaped kernels with poor L2 reuse.",
    )


def occupancy_maximization() -> Action:
    """Action: maximize SM occupancy via resource balancing."""
    return Action(
        id="t1_occupancy",
        tier=ActionTier.SIZING,
        name="Occupancy Maximization",
        description="Maximize SM occupancy by balancing registers, shared memory, and block size.",
        guidance=(
            "If `num_warps` × per-warp-registers exceeds the SM register file, occupancy drops to one "
            "block per SM and latency-hiding collapses. Try lowering num_warps (4 → 2) or splitting a "
            "fat kernel into smaller ones. Shared memory per block × concurrent-blocks must fit in the "
            "SM's shared memory pool — check the profiler's 'achieved occupancy' metric before "
            "speculating about this lever."
        ),
        anti_patterns=[
            "Maximizing occupancy on compute-bound kernels — useless and often regressive.",
            "Reducing num_warps without checking whether the kernel is occupancy-limited.",
        ],
        expected_impact="High-variance; large on occupancy-limited memory-bound kernels, near-zero otherwise.",
    )


def all_actions() -> list[Action]:
    """Return all Tier 1 actions."""
    return [block_size_tuning(), grid_shape_optimization(), occupancy_maximization()]
