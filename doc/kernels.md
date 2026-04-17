# Kernels — `src/kernels/`

Kernel abstraction and Triton compilation.

## Data Model

### KernelType

Enum of known kernel archetypes. Used by `MemoryRetriever` to filter past experiences by type.

Values: `MATMUL`, `SOFTMAX`, `LAYERNORM`, `ATTENTION`, `REDUCTION`, `ELEMENTWISE`, `CUSTOM`.

### KernelSpec

Static metadata about the kernel *problem* — stays the same across all optimization versions of the same kernel.

| Field | Type | Used by |
|-------|------|---------|
| `name` | str | Logging, reports |
| `kernel_type` | KernelType | Memory retrieval filtering |
| `flop_count` | int | `roofline.py` for T_SOL derivation |
| `memory_bytes` | int | `roofline.py` for T_SOL derivation |
| `input_shapes` | list[dict] | `correctness.py` for test input generation |

### Kernel

A single *version*: source code + Triton tuning parameters. Every search tree node holds one `Kernel`.

| Field | Type | Description |
|-------|------|-------------|
| `spec` | KernelSpec | Shared across all versions |
| `source_code` | str | Triton source code |
| `num_warps` | int | Triton num_warps parameter |
| `num_stages` | int | Triton num_stages for pipelining |
| `block_size` | dict[str,int] | Block dimensions (e.g., BLOCK_M, BLOCK_N) |

Read `source_code` directly — it's the full Triton source string.

## Compiler — `compiler.py`

Called by the Coder's `compile_kernel_tool` during its turn (not by the orchestrator).

- `compile_kernel(kernel) -> CompilationResult`: Compile Triton source. Returns success/fail + compiled function or error message.

By the time the Coder returns, compilation is guaranteed.

## Starters — `starters/`

Factory functions creating baseline `Kernel` instances for common operations. These are the root nodes of the search tree.

| File | Function | FLOPs estimate |
|------|----------|---------------|
| `matmul.py` | `make_matmul_kernel(M, N, K)` | 2MNK |
| `softmax.py` | `make_softmax_kernel(rows, cols)` | 5 * rows * cols |
| `layernorm.py` | `make_layernorm_kernel(batch, hidden)` | 5 * batch * hidden |
| `attention.py` | `make_attention_kernel(batch, heads, seq_len, head_dim)` | 4 * B * H * S^2 * D |
