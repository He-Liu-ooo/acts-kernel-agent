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
        min_compute_capability=9.0,
        guidance=(
            "**When**: H100 (compute capability 9.0+) kernels where the load phase serializes against "
            "compute and you've already exhausted standard prefetching/coalescing.\n"
            "**How**: TMA is a dedicated DMA-style copy engine that issues bulk async loads from global "
            "to shared memory without consuming warp-issue slots. CUTLASS exposes it as "
            "`cute::SM90_TMA_LOAD`; PTX-level access is `cp.async.bulk.tensor`.\n"
            "**Verify**: NCU `tma_active_cycles_pct` (or equivalent on the H100 metric tree) should "
            "be non-zero; warp-issue cycles previously spent on loads should drop.\n"
            "**Limits**: Triton V1 does not expose TMA primitives — there is no `tl.tma_load` or "
            "equivalent. This action is essentially blocked until Triton V2 lands TMA support. "
            "Listed for completeness; do not select on Triton V1."
        ),
        anti_patterns=[
            "Mostly blocked in Triton V1 — no first-class TMA primitive exposed.",
            "Selecting this on a pre-Hopper GPU — silently impossible.",
        ],
        expected_impact="Blocked on Triton V1; potentially substantial on H100 once enabled.",
    )


def h100_wgmma() -> Action:
    """Action: use H100 warp-group MMA instructions."""
    return Action(
        id="t5_h100_wgmma",
        tier=ActionTier.ARCH_SPECIFIC,
        name="H100 WGMMA",
        description="Use H100 warp-group MMA instructions for higher throughput.",
        preconditions=["compute_capability >= 9.0"],
        min_compute_capability=9.0,
        guidance=(
            "**When**: H100 matmul/conv kernels where the standard `tl.dot` is the binding constraint "
            "and tensor-core utilization can still rise.\n"
            "**How**: WGMMA (`wgmma.mma_async.sync.aligned`) operates on a full warp-group (128 threads) "
            "as a unit, asynchronously, with a separate completion barrier — letting the scheduler "
            "issue more independent tensor-core ops per cycle than the older `mma.sync` allowed. "
            "CUTLASS 3.x exposes this via `cute::SM90_64x*x*_F32F16F16_SS`.\n"
            "**Verify**: NCU `tensor_active_cycles_pct` should rise toward 100%; "
            "`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` should approach the H100 "
            "advertised peak.\n"
            "**Limits**: Triton V1 does not expose WGMMA — `tl.dot` lowers to mma.sync, not wgmma. "
            "Triton V2 nightly has experimental wgmma support but is unstable. Do not select on "
            "Triton V1."
        ),
        anti_patterns=[
            "Mostly blocked in Triton V1 — `tl.dot` does not lower to wgmma.",
            "Selecting on memory-bound kernels — wgmma doesn't address the bottleneck.",
        ],
        expected_impact="Blocked on Triton V1; potentially substantial on H100 compute-bound kernels once enabled.",
    )


def a100_cp_async() -> Action:
    """Action: use A100 cp.async for global-to-shared copies."""
    return Action(
        id="t5_a100_cp_async",
        tier=ActionTier.ARCH_SPECIFIC,
        name="A100 cp.async",
        description="Use A100 cp.async for asynchronous global-to-shared memory copies.",
        preconditions=["compute_capability >= 8.0"],
        min_compute_capability=8.0,
        guidance=(
            "**When**: A100 (compute capability 8.0+) kernels where the global-to-shared copy phase is "
            "serializing against compute and prefetching alone is insufficient.\n"
            "**How**: `cp.async` issues an async global→shared copy that completes via "
            "`cp.async.commit_group` + `cp.async.wait_group`, freeing the warp to do compute in the "
            "meantime. Triton's `num_stages` pipelining lowers to cp.async automatically on A100+; "
            "explicit `cp.async` PTX is rarely needed.\n"
            "**Verify**: bench against the same kernel with `num_stages=1`; NCU "
            "`l1tex__t_sectors_pipe_lsu_mem_global_op_ld...` should overlap with compute cycles.\n"
            "**Limits**: in practice this action is a re-statement of `t2_prefetching` for A100 — "
            "Triton's `num_stages` already exercises cp.async under the hood. Select this only when "
            "explicit PTX-level control is needed (rare)."
        ),
        anti_patterns=[
            "Selecting this when `t2_prefetching` already covers the use case via `num_stages`.",
            "Manually inlining cp.async PTX in Triton — fragile and rarely faster than `num_stages`.",
        ],
        expected_impact="Typically modest beyond what `num_stages` already provides.",
    )


def hopper_cluster_launch() -> Action:
    """Action: use Hopper cluster launch for SM cooperation."""
    return Action(
        id="t5_hopper_cluster",
        tier=ActionTier.ARCH_SPECIFIC,
        name="Hopper Cluster Launch",
        description="Use Hopper cluster launch for cross-SM cooperation.",
        preconditions=["compute_capability >= 9.0"],
        min_compute_capability=9.0,
        guidance=(
            "**When**: H100 kernels where blocks need to cooperate across SMs — sharing data via "
            "distributed shared memory (DSMEM) instead of round-tripping through global memory.\n"
            "**How**: Launch a thread-block cluster (2/4/8/16 blocks) with `cudaLaunchKernelEx` + "
            "`cudaLaunchAttributeClusterDimension`. Blocks within a cluster can read each other's "
            "shared memory via DSMEM and synchronize with cluster-level barriers.\n"
            "**Verify**: NCU `sm__cluster_active_cycles_pct` should be non-zero; cross-SM data movement "
            "previously hitting global memory should now hit DSMEM.\n"
            "**Limits**: Triton V1 does not expose cluster-launch primitives — `triton.jit` always "
            "launches single-SM blocks. Listed for completeness; do not select on Triton V1."
        ),
        anti_patterns=[
            "Mostly blocked in Triton V1 — no cluster-launch primitive exposed.",
            "Selecting on kernels with no cross-SM data sharing — pure overhead.",
        ],
        expected_impact="Blocked on Triton V1; niche on H100 even when enabled (most kernels don't need cross-SM cooperation).",
    )


def all_actions() -> list[Action]:
    """Return all Tier 5 actions."""
    return [h100_tma_loads(), h100_wgmma(), a100_cp_async(), hopper_cluster_launch()]
