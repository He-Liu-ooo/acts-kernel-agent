"""Tier 6 actions — kernel-type-specific optimizations."""

from __future__ import annotations

from src.actions.registry import Action, ActionTier


def welford_online_stats() -> Action:
    """Action: use Welford's algorithm for numerically stable online stats."""
    return Action(
        id="t6_welford",
        tier=ActionTier.KERNEL_SPECIFIC,
        name="Welford Online Stats",
        description="Use Welford's algorithm for numerically stable online mean/variance.",
        applicable_to=["layernorm", "reduction"],
        guidance="Placeholder guidance.",
    )


def online_softmax() -> Action:
    """Action: online softmax (single-pass, no separate max reduction)."""
    return Action(
        id="t6_online_softmax",
        tier=ActionTier.KERNEL_SPECIFIC,
        name="Online Softmax",
        description="Online softmax: single-pass computation without separate max reduction.",
        applicable_to=["softmax", "attention"],
        guidance="Placeholder guidance.",
    )


def causal_mask_skip() -> Action:
    """Action: skip computation for masked positions in causal attention."""
    return Action(
        id="t6_causal_mask",
        tier=ActionTier.KERNEL_SPECIFIC,
        name="Causal Mask Skip",
        description="Skip computation for masked positions in causal attention.",
        applicable_to=["attention"],
        guidance="Placeholder guidance.",
    )


def flash_attention_tiling() -> Action:
    """Action: FlashAttention-style tiling for fused QKV attention."""
    return Action(
        id="t6_flash_attn",
        tier=ActionTier.KERNEL_SPECIFIC,
        name="FlashAttention Tiling",
        description="FlashAttention-style tiling for fused QKV with O(N) memory.",
        applicable_to=["attention"],
        guidance="Placeholder guidance.",
        expected_impact="2-4x on long-sequence attention.",
    )


def all_actions() -> list[Action]:
    """Return all Tier 6 actions."""
    return [welford_online_stats(), online_softmax(), causal_mask_skip(), flash_attention_tiling()]
