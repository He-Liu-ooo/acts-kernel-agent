"""Smoke-test operator-supplied baseline for SOL 069_rms_norm.

Minimal valid Triton kernel — single @triton.jit def, no @triton.autotune
(loaded with triton_baseline_enforce_autotune=false). Used by
tests/test_optimize_operator_baseline_gpu.py to verify the
load_operator_baseline path against a live SOL problem. Not optimized;
the test asserts the operator path RUNS, not that it wins.

Op shape: fused residual-add + RMSNorm. Hidden size is const 8192 per
the problem's definition.json. Reference is bf16 storage with fp32
intermediate math; this kernel matches that convention.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_residual_kernel(
    x_ptr, r_ptr, w_ptr, y_ptr,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + r

    var = tl.sum(s * s, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = s * rstd

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (w * normed).to(tl.bfloat16)
    tl.store(y_ptr + row * N + cols, y, mask=mask)


def kernel_fn(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    *prefix, N = hidden_states.shape
    M = 1
    for d in prefix:
        M *= d
    x = hidden_states.reshape(M, N)
    r = residual.reshape(M, N)
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    rmsnorm_residual_kernel[(M,)](x, r, weight, y, eps, N, BLOCK)
    return y.reshape(hidden_states.shape)
