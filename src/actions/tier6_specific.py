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
        guidance=(
            "**When**: layernorm-style kernels computing mean and variance over a long reduction axis "
            "where the naive sum-of-squares accumulator drifts in fp16/bf16, or where the kernel needs "
            "to fuse normalization stats with another reduction in one pass.\n"
            "**How**: Welford's recurrence updates `(mean, M2)` per element, where M2 is the running "
            "sum of squared deviations from the current mean: \n"
            "  delta = x - mean; mean += delta / n; M2 += delta * (x - mean)\n"
            "Variance = M2 / (n - 1). Tree-reduce the (mean, M2, n) triples across threads using the "
            "Chan parallel-merge formula.\n"
            "**Verify**: numerical error vs. the reference layernorm should drop sharply on long axes "
            "(>4096) where the naive approach loses precision.\n"
            "**Limits**: on short reduction axes the naive two-pass approach (sum, then sum-of-squares "
            "with the known mean) is faster and equally precise. Welford is preferred mainly for the "
            "single-pass property, not for raw speed."
        ),
        anti_patterns=[
            "Welford on short axes (<256) — slower than naive two-pass with no precision benefit.",
            "Forgetting the parallel-merge step — per-thread Welford without tree reduction is sequential.",
        ],
        expected_impact="Typically modest; matters most when fusion or numerical stability is the binding constraint.",
    )


def online_softmax() -> Action:
    """Action: online softmax (single-pass, no separate max reduction)."""
    return Action(
        id="t6_online_softmax",
        tier=ActionTier.KERNEL_SPECIFIC,
        name="Online Softmax",
        description="Online softmax: single-pass computation without separate max reduction.",
        applicable_to=["softmax", "attention"],
        guidance=(
            "**When**: softmax or softmax-fused kernels (attention) where the standard three-pass "
            "implementation (max → exp-sum → divide) costs three full passes over the input.\n"
            "**How**: Maintain `(running_max, running_sum)` and update both as each new element is seen:\n"
            "  new_max = max(running_max, x);\n"
            "  running_sum = running_sum * exp(running_max - new_max) + exp(x - new_max);\n"
            "  running_max = new_max\n"
            "Final softmax is `exp(x_i - running_max) / running_sum`. Reduces three passes to two "
            "(stats pass + normalize pass) and enables the FlashAttention pattern (see t6_flash_attn).\n"
            "**Verify**: bench against the three-pass reference; correctness via the standard "
            "softmax tolerance gate.\n"
            "**Limits**: requires careful tree-reduction to merge per-thread (max, sum) pairs across "
            "the warp/block. The merge formula has the same shape as Welford's — re-scale the smaller "
            "max's sum before adding."
        ),
        anti_patterns=[
            "Forgetting the rescale step in the parallel merge — produces silent numerical errors.",
            "Online softmax on tiny reduction axes — overhead exceeds the saved pass.",
        ],
        expected_impact="Often substantial in attention; modest in standalone softmax.",
    )


def causal_mask_skip() -> Action:
    """Action: skip computation for masked positions in causal attention."""
    return Action(
        id="t6_causal_mask",
        tier=ActionTier.KERNEL_SPECIFIC,
        name="Causal Mask Skip",
        description="Skip computation for masked positions in causal attention.",
        applicable_to=["attention"],
        guidance=(
            "**When**: causal (decoder) attention where roughly half the QK^T elements are masked to "
            "-inf and softmax-zeroed; computing them is pure waste.\n"
            "**How**: Restructure the per-block loop so that blocks entirely below the diagonal are "
            "skipped (`if block_n_start > block_m_end: continue`), and blocks straddling the diagonal "
            "use a per-element mask. Pure-below-diagonal blocks halve the QK^T compute on average.\n"
            "**Verify**: bench against the un-skipped causal kernel; the win scales with sequence "
            "length (longer sequences → more fully-masked blocks).\n"
            "**Limits**: only applies to causal/lower-triangular masks — sliding-window or "
            "block-sparse masks need their own skip logic. The diagonal-block branch adds register "
            "pressure that can hurt occupancy on small blocks."
        ),
        anti_patterns=[
            "Applying causal-skip to bidirectional attention — silently produces wrong results.",
            "Skipping at element granularity instead of block granularity — branch-divergent, slow.",
        ],
        expected_impact="Often substantial on long-sequence causal attention (~2× theoretical max).",
    )


def flash_attention_tiling() -> Action:
    """Action: FlashAttention-style tiling for fused QKV attention."""
    return Action(
        id="t6_flash_attn",
        tier=ActionTier.KERNEL_SPECIFIC,
        name="FlashAttention Tiling",
        description="FlashAttention-style tiling for fused QKV with O(N) memory.",
        applicable_to=["attention"],
        guidance=(
            "**When**: attention kernels where the N×N attention matrix doesn't fit in shared memory "
            "(typical for sequences ≥1024) or where memory bandwidth is the binding constraint.\n"
            "**How**: Tile both Q and KV. The outer loop iterates over Q tiles; the inner loop streams "
            "KV tiles through shared memory, accumulating partial output and online-softmax stats "
            "(see t6_online_softmax) per Q tile. The N×N matrix is never materialized; per-tile partial "
            "results merge via the online-softmax rescale rule. Memory drops from O(N²) to O(N).\n"
            "**Verify**: bench against the standard QKV-then-softmax-then-V kernel sequence; "
            "verify against the unfused reference at the kernel's configured tolerance.\n"
            "**Limits**: implementation complexity is high — the online stats merge and the per-tile "
            "rescale are easy to get wrong. Triton's reference flash-attention kernel is the canonical "
            "starting point. For very short sequences (<512) the overhead can exceed the savings."
        ),
        anti_patterns=[
            "FlashAttention on short sequences — overhead exceeds the savings.",
            "Tiling Q only without tiling KV — defeats the O(N) memory property.",
        ],
        expected_impact="Often substantial on long-sequence attention; modest-to-negative on short.",
    )


def all_actions() -> list[Action]:
    """Return all Tier 6 actions."""
    return [welford_online_stats(), online_softmax(), causal_mask_skip(), flash_attention_tiling()]
