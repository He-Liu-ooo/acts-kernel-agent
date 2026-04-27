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
        guidance=(
            "**When**: matmul-shaped problems where M*N is small relative to SM count (so the standard "
            "M×N grid leaves SMs idle) but K is large. Typical case: skinny matmuls in attention "
            "(M=batch×seq, N=head_dim, K=seq).\n"
            "**How**: Launch a 3D grid (M_blocks, N_blocks, K_split). Each (m, n, k_split) block "
            "computes a partial sum over its K-slice; a second kernel (or atomics) reduces partials "
            "into the final output. Tune `SPLIT_K` between 2 and 16 — larger splits hurt the reduction "
            "pass and can collapse L2 reuse on the K-strided inputs.\n"
            "**Verify**: NCU `sm__warps_active.avg.pct_of_peak_sustained_active` should rise; the "
            "reduction kernel's overhead should be <10% of the matmul time.\n"
            "**Limits**: requires either an atomic-add reduction (numerically nondeterministic) or a "
            "second kernel launch (overhead). For tall-skinny matmuls (large M, small K), use the "
            "standard non-split decomposition instead."
        ),
        anti_patterns=[
            "Splitting K when M×N already saturates SM count — strictly regressive.",
            "SPLIT_K so large that the reduction pass dominates the matmul.",
        ],
        expected_impact="High-variance; large on small-MN matmuls with large K, none on standard shapes.",
    )


def persistent_kernel() -> Action:
    """Action: convert to persistent kernel (single-wave launch)."""
    return Action(
        id="t4_persistent",
        tier=ActionTier.ADVANCED,
        name="Persistent Kernel",
        description="Convert to persistent kernel with single-wave launch.",
        guidance=(
            "**When**: launch overhead is a measurable fraction of kernel runtime, OR the problem has "
            "many small tiles where per-block scheduler latency dominates.\n"
            "**How**: Launch exactly `SM_count` blocks; each block loops over a queue of work-tile IDs "
            "computed from its `program_id`. The inner loop typically iterates `total_tiles / SM_count` "
            "times. Reduces grid scheduling overhead from O(num_tiles) to O(SM_count).\n"
            "**Verify**: bench against the non-persistent baseline; the win is in launch and dispatch "
            "latency, not in arithmetic throughput.\n"
            "**Limits**: Triton's high-level abstraction makes manual persistence awkward — there's no "
            "first-class persistent-block primitive, so the work-queue is hand-rolled with `program_id` "
            "arithmetic. For most kernels the gain doesn't justify the complexity."
        ),
        anti_patterns=[
            "Awkward in Triton — limited first-class support for persistent threads.",
            "Adding persistence to a kernel where launch overhead is <5% of runtime.",
        ],
        expected_impact="Typically small; matters most on very-short-running kernels with high launch overhead.",
    )


def warp_specialization() -> Action:
    """Action: assign different warps to different roles."""
    return Action(
        id="t4_warp_spec",
        tier=ActionTier.ADVANCED,
        name="Warp Specialization",
        description="Assign different warps to producer/consumer roles.",
        guidance=(
            "**When**: kernel has clearly-separable producer (loads / address computation) and consumer "
            "(compute / accumulation) phases that could overlap if scheduled on independent warps.\n"
            "**How**: At the SASS level, dedicate some warps to issuing loads while others execute "
            "tensor-core math. CUTLASS does this with explicit `cute::async_copy` + named-barrier "
            "synchronization between warp groups.\n"
            "**Verify**: NCU `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` and "
            "`sm__pipe_tensor_cycles_active...` should both show high utilization in the same time window.\n"
            "**Limits**: Triton V1 does not expose warp-level scheduling primitives — there is no way to "
            "pin code to specific warps within a block. This action is essentially blocked until "
            "Triton V2's warp-group support lands. Listed for completeness; do not select on Triton V1."
        ),
        anti_patterns=[
            "Not possible in Triton V1 — no warp-level scheduling primitives exposed.",
            "Selecting this action on H100/A100 thinking compute capability alone enables it.",
        ],
        expected_impact="Blocked on Triton V1; potentially substantial on producer-consumer-shaped kernels once enabled.",
    )


def stream_k() -> Action:
    """Action: stream-K work partitioning for load balancing."""
    return Action(
        id="t4_stream_k",
        tier=ActionTier.ADVANCED,
        name="Stream-K",
        description="Stream-K work partitioning for better load balancing.",
        guidance=(
            "**When**: matmul where the standard tile decomposition leaves a 'tail' of partial work on "
            "some SMs while others are idle (problem dimensions not divisible by tile dimensions × SM count).\n"
            "**How**: Each SM picks up a contiguous range of K-iterations from a flat work queue rather "
            "than being assigned a fixed (M, N, K) sub-tile. Partial sums merge via atomics or a "
            "lightweight reduction pass. Eliminates the tail by construction — every SM stays busy "
            "until the queue is empty.\n"
            "**Verify**: NCU `sm__warps_active...sustained_active` should approach 100%; per-SM runtime "
            "variance should drop sharply.\n"
            "**Limits**: requires a global work counter (atomic) or pre-computed work queue, both with "
            "overhead that only pays off when the standard decomposition leaves substantial idle time. "
            "Implementing in Triton is non-trivial — most production stream-K kernels are CUTLASS."
        ),
        anti_patterns=[
            "Stream-K on perfectly-divisible matmul shapes — no load imbalance to fix, pure overhead.",
            "Using stream-K on small matmuls where one wave already saturates — no benefit.",
        ],
        expected_impact="High-variance; substantial on tail-bound matmuls, near-zero on standard-shaped ones.",
    )


def all_actions() -> list[Action]:
    """Return all Tier 4 actions."""
    return [split_k_decomposition(), persistent_kernel(), warp_specialization(), stream_k()]
