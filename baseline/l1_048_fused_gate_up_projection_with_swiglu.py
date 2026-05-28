import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_gate_up_gelu_kernel(
    x_ptr, gate_proj_ptr, up_proj_ptr, out_ptr,
    M, N, K,
    stride_x_m, stride_x_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_out_m, stride_out_n,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    num_m_tiles = (M + BLOCK_M - 1) // BLOCK_M
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    total_tiles = num_m_tiles * num_n_tiles

    tile_idx = pid
    while tile_idx < total_tiles:
        tile_m = tile_idx // num_n_tiles
        tile_n = tile_idx % num_n_tiles

        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        gate_ptrs = gate_proj_ptr + offs_n[:, None] * stride_gate_n + offs_k[None, :] * stride_gate_k
        up_ptrs = up_proj_ptr + offs_n[:, None] * stride_up_n + offs_k[None, :] * stride_up_k

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
            w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K - k)

            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            gate = tl.load(gate_ptrs, mask=w_mask, other=0.0)
            up = tl.load(up_ptrs, mask=w_mask, other=0.0)

            gate_acc += tl.dot(x, tl.trans(gate))
            up_acc += tl.dot(x, tl.trans(up))

            x_ptrs += BLOCK_K * stride_x_k
            gate_ptrs += BLOCK_K * stride_gate_k
            up_ptrs += BLOCK_K * stride_up_k

        # GELU-tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # sqrt(2/pi) ~= 0.7978845608028654
        inner = 0.7978845608028654 * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
        exp2x = tl.exp(2.0 * inner)
        tanh_inner = (exp2x - 1.0) / (exp2x + 1.0)
        activated_gate = 0.5 * gate_acc * (1.0 + tanh_inner)

        result = activated_gate * up_acc
        result = result.to(DTYPE)

        out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, result, mask=out_mask)

        tile_idx += num_pids


def kernel_fn(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, hidden_size = x.shape
    intermediate_size = gate_proj.shape[0]

    M = batch_size * seq_len
    N = intermediate_size
    K = hidden_size

    x_2d = x.reshape(M, K)

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    if x.dtype == torch.float16:
        DTYPE = tl.float16
    elif x.dtype == torch.bfloat16:
        DTYPE = tl.bfloat16
    else:
        DTYPE = tl.float32

    num_persistent_blocks = 284
    grid = (num_persistent_blocks,)

    fused_gate_up_gelu_kernel[grid](
        x_2d, gate_proj, up_proj, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        gate_proj.stride(0), gate_proj.stride(1),
        up_proj.stride(0), up_proj.stride(1),
        out.stride(0), out.stride(1),
        DTYPE=DTYPE,
    )

    return out.reshape(batch_size, seq_len, intermediate_size)
