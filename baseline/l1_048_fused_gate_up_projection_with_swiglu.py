import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_gate_up_kernel(
    x_ptr, gate_proj_ptr, up_proj_ptr, output_ptr,
    M, N, K,
    stride_x0, stride_x1,
    stride_gp0, stride_gp1,
    stride_up0, stride_up1,
    stride_out0, stride_out1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    x_ptrs = x_ptr + rm[:, None] * stride_x0 + rk[None, :] * stride_x1
    gp_ptrs = gate_proj_ptr + rn[:, None] * stride_gp0 + rk[None, :] * stride_gp1
    up_ptrs = up_proj_ptr + rn[:, None] * stride_up0 + rk[None, :] * stride_up1
    
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        x_mask = (rm[:, None] < M) & (rk[None, :] < K - k)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        gp_mask = (rn[:, None] < N) & (rk[None, :] < K - k)
        gp = tl.load(gp_ptrs, mask=gp_mask, other=0.0)
        
        up_mask = (rn[:, None] < N) & (rk[None, :] < K - k)
        up = tl.load(up_ptrs, mask=up_mask, other=0.0)
        
        acc_gate += tl.dot(x, tl.trans(gp))
        acc_up += tl.dot(x, tl.trans(up))
        
        x_ptrs += BLOCK_K * stride_x1
        gp_ptrs += BLOCK_K * stride_gp1
        up_ptrs += BLOCK_K * stride_up1
    
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (acc_gate + 0.044715 * acc_gate * acc_gate * acc_gate)
    activated_gate = acc_gate * tl.sigmoid(2.0 * inner)
    
    result = activated_gate * acc_up
    
    out_ptrs = output_ptr + rm[:, None] * stride_out0 + rn[None, :] * stride_out1
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, result, mask=out_mask)


def kernel_fn(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, hidden_size = x.shape
    intermediate_size, hidden_size_gate = gate_proj.shape
    
    M = batch_size * seq_len
    N = intermediate_size
    K = hidden_size
    
    x_2d = x.reshape(M, K)
    
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    
    fused_gate_up_kernel[grid](
        x_2d, gate_proj, up_proj, output,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        gate_proj.stride(0), gate_proj.stride(1),
        up_proj.stride(0), up_proj.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output.reshape(batch_size, seq_len, intermediate_size)
