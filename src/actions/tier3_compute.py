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
        guidance=(
            "**When**: FP32 matmul/conv on Ampere or newer (compute capability ≥8.0) where the kernel "
            "is compute-bound and tensor-core eligible.\n"
            "**How**: Pass `allow_tf32=True` to `tl.dot` or set the global flag "
            "`torch.backends.cuda.matmul.allow_tf32 = True` upstream. TF32 keeps the FP32 exponent range "
            "but truncates the mantissa to 10 bits inside the tensor core, then accumulates in FP32.\n"
            "**Verify**: NCU `tensor_pipe_active_cycles_pct` should jump from 0 to >50%; "
            "absolute error stays within 1e-3 for typical activation/weight magnitudes.\n"
            "**Limits**: tail-precision-sensitive workloads (gradient accumulation in long chains, "
            "iterative solvers) can drift; verify against the FP32 reference per workload."
        ),
        anti_patterns=[
            "Enabling TF32 on pre-Ampere hardware — silently falls back to FP32, no speedup.",
            "Using TF32 on accumulation-sensitive ops (Kahan summation, iterative refinement).",
        ],
        expected_impact="Often substantial on tensor-core-eligible FP32 matmuls; nothing on memory-bound or non-matmul kernels.",
    )


def mixed_precision() -> Action:
    """Action: mixed-precision computation (FP16/BF16 compute, FP32 accum)."""
    return Action(
        id="t3_mixed_precision",
        tier=ActionTier.COMPUTE,
        name="Mixed Precision",
        description="Mixed-precision: FP16/BF16 compute with FP32 accumulation.",
        preconditions=["compute_bound"],
        guidance=(
            "**When**: kernel inputs are FP32 but the math could be done in lower precision without "
            "loss (most ML inference and training-forward kernels).\n"
            "**How**: Cast inputs to bf16 (or fp16 if dynamic range permits) at load, pass through "
            "`tl.dot(..., out_dtype=tl.float32)` so the accumulator stays FP32, cast back at store. "
            "BF16 is preferred for training-adjacent code (same exponent range as FP32, no overflow); "
            "FP16 has more mantissa bits but narrow dynamic range.\n"
            "**Verify**: NCU `tensor_pipe_active_cycles_pct` should rise; per-element error against the "
            "FP32 reference must stay within the kernel's tolerance.\n"
            "**Limits**: bf16 ULP at unit magnitude is ~7.8e-3 — kernels with tighter tolerance "
            "expectations will fail correctness; correctness gates run at atol=1e-2/rtol=1e-2 by default."
        ),
        anti_patterns=[
            "Accumulating in fp16/bf16 — drift compounds across the inner loop, fails correctness.",
            "Mixed precision on kernels already memory-bound — saves nothing, costs cast overhead.",
        ],
        expected_impact="Often substantial on tensor-core-eligible matmul/conv; modest on elementwise.",
    )


def fused_operations() -> Action:
    """Action: fuse multiple operations into a single kernel."""
    return Action(
        id="t3_fused_ops",
        tier=ActionTier.COMPUTE,
        name="Fused Operations",
        description="Fuse multiple operations into a single kernel to reduce launch overhead and memory traffic.",
        guidance=(
            "**When**: a sequence of small kernels round-trips intermediate results through global memory "
            "(e.g., GELU after matmul, RMSNorm after residual-add, softmax after scale).\n"
            "**How**: Inline the second op's compute inside the first kernel before the final `tl.store`. "
            "Verify the fused kernel's register/shared-memory budget still allows ≥2 blocks per SM — "
            "fusion that crashes occupancy is regressive.\n"
            "**Verify**: bench the fused kernel against the baseline pair; the win comes from saving "
            "the intermediate's bandwidth, so the larger the intermediate tensor, the bigger the gain.\n"
            "**Limits**: don't fuse across different launch shapes (e.g., reduction + elementwise with "
            "different grid dims) — usually requires synthetic padding that costs more than it saves."
        ),
        anti_patterns=[
            "Fusing into a kernel that's already register-pressured — spills negate the win.",
            "Fusing ops with different bottleneck classes (memory + compute) — masks the dominant cost.",
        ],
        expected_impact="Often substantial when intermediates are large; modest when intermediates fit in cache.",
    )


def vectorized_loads() -> Action:
    """Action: use vectorized memory loads (tl.load with wider types)."""
    return Action(
        id="t3_vectorized_loads",
        tier=ActionTier.COMPUTE,
        name="Vectorized Loads",
        description="Use vectorized memory loads for higher bandwidth utilization.",
        preconditions=["compute_bound"],
        guidance=(
            "**When**: kernel issues many small (≤4-byte) loads per thread; the memory subsystem is "
            "underutilized despite high request count.\n"
            "**How**: Reshape the load pattern so each thread reads a contiguous vector "
            "(2/4/8 fp32 or 4/8/16 fp16 elements), then operate on the vector. Triton handles this "
            "automatically when the pointer arithmetic is contiguous and the block shape is a multiple "
            "of the vector width.\n"
            "**Verify**: NCU `gld_transactions_per_request` should drop (fewer transactions, more bytes "
            "per transaction); `gld_throughput` should rise toward DRAM peak.\n"
            "**Limits**: misaligned base pointers fall back to scalar loads silently — confirm the input "
            "tensor is allocated with appropriate alignment (PyTorch tensors usually are)."
        ),
        anti_patterns=[
            "Vectorizing without checking alignment — silent fallback to scalar.",
            "Vectorizing on kernels already at peak bandwidth — no win, possible block-size misalignment regression.",
        ],
        expected_impact="Typically modest; matters most on memory-bound kernels with under-wide loads.",
    )


def loop_unrolling() -> Action:
    """Action: unroll reduction or iteration loops."""
    return Action(
        id="t3_loop_unroll",
        tier=ActionTier.COMPUTE,
        name="Loop Unrolling",
        description="Unroll reduction or iteration loops to reduce branch overhead.",
        preconditions=["compute_bound"],
        guidance=(
            "**When**: tight inner loops with predictable trip count and cheap per-iter compute "
            "(e.g., reduction across a small dimension, fused-multiply-add chains).\n"
            "**How**: Add `@triton.jit(do_not_specialize=...)` annotations or factor the loop into a "
            "sequence of explicit `tl.dot`/elementwise ops over fixed-shape tiles. Triton's compiler "
            "unrolls loops with constexpr bounds automatically; the manual lever matters when the bound "
            "is runtime-known but small.\n"
            "**Verify**: NCU `inst_executed` should drop relative to the rolled version; "
            "`branch_efficiency` should approach 100%.\n"
            "**Limits**: aggressive unrolling inflates register pressure linearly with unroll factor — "
            "watch for occupancy collapse and instruction-cache thrashing on very long unrolled loops."
        ),
        anti_patterns=[
            "Unrolling a memory-bound loop — saves branch cost but spends register budget for nothing.",
            "Manually unrolling what Triton already unrolls (constexpr-bound loops).",
        ],
        expected_impact="Typically small; rarely the dominant factor on its own.",
    )


def all_actions() -> list[Action]:
    """Return all Tier 3 actions."""
    return [
        tf32_accumulation(), mixed_precision(), fused_operations(),
        vectorized_loads(), loop_unrolling(),
    ]
