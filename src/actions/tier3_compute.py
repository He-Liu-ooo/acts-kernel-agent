"""Tier 3 actions — compute optimization."""

from __future__ import annotations

from src.actions.registry import Action, ActionTier


def tf32_accumulation() -> Action:
    """Action: use TF32 for faster FP32 accumulation."""
    return Action(
        id="t3_tf32",
        tier=ActionTier.COMPUTE,
        name="TF32 Accumulation",
        description="Use TF32 for faster FP32 accumulation on Ampere+.",
        preconditions=["compute_bound"],
        guidance="Placeholder guidance.",
    )


def mixed_precision() -> Action:
    """Action: mixed-precision computation (FP16/BF16 compute, FP32 accum)."""
    return Action(
        id="t3_mixed_precision",
        tier=ActionTier.COMPUTE,
        name="Mixed Precision",
        description="Mixed-precision: FP16/BF16 compute with FP32 accumulation.",
        preconditions=["compute_bound"],
        guidance="Placeholder guidance.",
        expected_impact="Up to 2x throughput on tensor-core-eligible ops.",
    )


def fused_operations() -> Action:
    """Action: fuse multiple operations into a single kernel."""
    return Action(
        id="t3_fused_ops",
        tier=ActionTier.COMPUTE,
        name="Fused Operations",
        description="Fuse multiple operations into a single kernel to reduce launch overhead and memory traffic.",
        guidance="Placeholder guidance.",
    )


def vectorized_loads() -> Action:
    """Action: use vectorized memory loads (tl.load with wider types)."""
    return Action(
        id="t3_vectorized_loads",
        tier=ActionTier.COMPUTE,
        name="Vectorized Loads",
        description="Use vectorized memory loads for higher bandwidth utilization.",
        preconditions=["compute_bound"],
        guidance="Placeholder guidance.",
    )


def loop_unrolling() -> Action:
    """Action: unroll reduction or iteration loops."""
    return Action(
        id="t3_loop_unroll",
        tier=ActionTier.COMPUTE,
        name="Loop Unrolling",
        description="Unroll reduction or iteration loops to reduce branch overhead.",
        preconditions=["compute_bound"],
        guidance="Placeholder guidance.",
    )


def all_actions() -> list[Action]:
    """Return all Tier 3 actions."""
    return [
        tf32_accumulation(), mixed_precision(), fused_operations(),
        vectorized_loads(), loop_unrolling(),
    ]
