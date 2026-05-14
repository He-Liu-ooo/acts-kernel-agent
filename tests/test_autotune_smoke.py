"""Tier-2 GPU smoke: ``@triton.autotune`` burn-in behaviour end-to-end (A1 PR 1).

Validates that ``benchmark_kernel`` + the new burn-in step in
``_time_workload`` correctly fires Triton autotune, populates the JIT
cache, and produces deterministic timing across a second invocation.
Requires real CUDA + ``triton``; runs only in the Tier-2 venv
(``~/.venvs/acts_run_venv``).
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu]


@pytest.fixture
def autotune_matmul_source() -> str:
    return '''
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((offs_k + _k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=((offs_k + _k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def kernel_fn(A, B):
    M, K = A.shape
    K2, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
    )
    return C
'''


def test_autotune_burn_in_populates_cache(autotune_matmul_source):
    """First invocation triggers autotune; second uses the cached winner."""
    import time
    import torch

    from src.kernels.compiler import compile_kernel
    from src.kernels.kernel import Kernel, KernelSpec, KernelType

    spec = KernelSpec(name="matmul_test", kernel_type=KernelType.MATMUL, entrypoint="kernel_fn")
    k = Kernel(spec=spec, source_code=autotune_matmul_source, triton_kernel_name="matmul_kernel")
    assert len(k.autotune_configs) == 4
    assert k.autotune_keys == ["M", "N", "K"]

    compiled = compile_kernel(k)
    assert compiled.success, compiled.error_message
    # A1 PR 1: compile_kernel now surfaces the Triton Autotuner instance
    # alongside the host entrypoint, so benchmark_kernel can diff the
    # cache around each workload. The host wrapper (compiled.compiled_fn)
    # doesn't carry the cache; the Autotuner (compiled.triton_autotuner)
    # does.
    assert compiled.triton_autotuner is not None, (
        "triton_autotuner not resolved — compile_kernel did not find "
        "module.matmul_kernel or the wrapper has no .cache attribute"
    )

    A = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    B = torch.randn(256, 256, device="cuda", dtype=torch.float16)

    t0 = time.perf_counter()
    compiled.compiled_fn(A, B)
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    compiled.compiled_fn(A, B)
    torch.cuda.synchronize()
    second_call_s = time.perf_counter() - t1

    # First call includes autotune (4 compiles + 4 microbench runs);
    # second is just the cached winner. Sanity bound: first call is
    # at least 5x slower than the cache-hit. Loose because compile time
    # varies across Triton versions / GPUs.
    assert first_call_s > 5 * second_call_s, (
        f"autotune did not appear to fire — first={first_call_s:.3f}s, "
        f"second={second_call_s:.3f}s"
    )

    # The Autotuner's cache should have at least one entry now.
    cache = compiled.triton_autotuner.cache
    assert cache, "Autotuner cache empty after burn-in invocation"
    assert len(cache) >= 1


def test_benchmark_kernel_with_autotune_burn_in_then_timed(autotune_matmul_source):
    """End-to-end: benchmark_kernel produces a stable median across two runs."""
    import torch
    from sol_execbench.core.data import Workload

    from src.config import ACTSConfig
    from src.eval.benchmark import benchmark_kernel
    from src.kernels.kernel import Kernel, KernelSpec, KernelType

    spec = KernelSpec(name="matmul_test", kernel_type=KernelType.MATMUL, entrypoint="kernel_fn")
    k = Kernel(spec=spec, source_code=autotune_matmul_source, triton_kernel_name="matmul_kernel")
    cfg = ACTSConfig(warmup_runs=5, timed_runs=10)

    def gen(seed: int) -> tuple:
        # Use a positive deterministic seed even for the reserved
        # burn-in seed of -1; the kernel doesn't care about seed value
        # beyond reproducibility of the input tensors.
        torch.manual_seed(seed if seed >= 0 else 12345)
        return (
            torch.randn(256, 256, device="cuda", dtype=torch.float16),
            torch.randn(256, 256, device="cuda", dtype=torch.float16),
        )

    wl = Workload.model_validate({"uuid": "wl-test", "axes": {"M": 256, "N": 256, "K": 256}, "inputs": {}})

    result1 = benchmark_kernel(k, cfg, workloads=[wl], input_generators=[gen])
    result2 = benchmark_kernel(k, cfg, workloads=[wl], input_generators=[gen])

    # Both runs should agree to within 2x. Both pay their own per-call
    # burn-in (separate _time_workload entries) but inherit Triton's
    # module-level JIT cache, so the second autotune is essentially
    # cache-hit.
    ratio = result2.median_latency_us / result1.median_latency_us
    assert 0.5 < ratio < 2.0, f"unstable median across runs: {ratio:.2f}x"
