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
| `entrypoint` | str | Callable name the compiler resolves via `getattr` (default `"kernel_fn"`). Overridable for fused ops where the launchable symbol is a host wrapper. |

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

Called by the Coder's `compile_kernel_tool` during its turn, by `eval/benchmark.py::_compile_entrypoint` before timing, by `eval/profiler.py::profile_kernel` to materialize the kernel for NCU's subprocess, and by `pipeline/verify.py` post-search.

- `compile_kernel(kernel, cache_dir=None) -> CompilationResult`: Source-hash-keyed file-backed import. Writes source to `<cache_dir>/<name>_<hash>.py` (hash = `sha256(source)[:12]`), loads via `importlib.util.spec_from_file_location` + `exec_module`, resolves `kernel.spec.entrypoint` via `getattr`. Returns `success`, `compiled_fn`, `error_message`, and `source_path` (carries real filenames into tracebacks so the Coder can self-correct). Defaults to `DEFAULT_CACHE_DIR = Path(".acts_cache/compiled")`.

### `sys.modules` short-circuit

The module name is pinned by the source hash (`acts_compiled_<name>_<hash>`), so identical source always resolves to the same `sys.modules` entry. Before writing to disk, `compile_kernel` checks `sys.modules.get(module_name)`: if the module is already loaded and the cache file still exists, it returns the cached callable without re-executing `exec_module`. Failed loads eagerly pop the half-built module from `sys.modules` so a second attempt on the same (buggy) source re-runs the loader and surfaces the same error rather than returning a zombie.

This collapses the three repeat-compile vectors that hit during real runs: (a) the Coder's correctness tool compiling the same source twice in one turn, (b) `baseline_generator`'s post-verify recompile after `translate()` returns, (c) `pipeline/verify`'s post-search re-verify, (d) Phase C's re-profile across N workloads.

### Parse-time vs launch-time errors

Parse-time errors (syntax, imports, missing/non-callable entrypoint) surface as `success=False`. Triton's `@triton.jit` specialization is lazy — shape/dtype-dependent compile errors surface later in `eval/correctness.py` or on the first kernel launch inside `eval/benchmark.py`.

## Starters — `starters/`

Factory functions creating baseline `Kernel` instances for common operations. These are the root nodes of the search tree.

| File | Function | FLOPs estimate |
|------|----------|---------------|
| `matmul.py` | `make_matmul_kernel(M, N, K)` | 2MNK |
| `softmax.py` | `make_softmax_kernel(rows, cols)` | 5 * rows * cols |
| `layernorm.py` | `make_layernorm_kernel(batch, hidden)` | 5 * batch * hidden |
| `attention.py` | `make_attention_kernel(batch, heads, seq_len, head_dim)` | 4 * B * H * S^2 * D |
