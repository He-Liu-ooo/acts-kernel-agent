# Kernels — `src/kernels/`

Kernel abstraction and Triton compilation.

## Data Model

### KernelType

Enum of known kernel archetypes. Used by `MemoryRetriever` to filter past experiences by type.

Values: `MATMUL`, `GEMM`, `SOFTMAX`, `LAYERNORM`, `RMSNORM`, `ATTENTION`, `GQA`, `MOE`, `EMBEDDING`, `LINEAR`, `FUSED_BLOCK`, `MLP`, `CONV`, `SSM`, `REDUCTION`, `ELEMENTWISE`, `CUSTOM`. SOL-ExecBench `op_type` strings (`gemm`, `rmsnorm`, `gqa`, `moe`, …) map to these via `_OP_TYPE_TO_KERNEL_TYPE` in `src/pipeline/optimize.py`.

### KernelSpec

Static metadata about the kernel *problem* — stays the same across all optimization versions of the same kernel.

| Field | Type | Used by |
|-------|------|---------|
| `name` | str | Logging, reports |
| `kernel_type` | KernelType | Memory retrieval filtering |
| `flop_count` | int | `roofline.py` for T_SOL derivation (left at `0` on the SOL path — SOLAR + `compute_roofline_inputs(definition, workload)` derive FLOPs from workload axes; only the placeholder starters populate it directly) |
| `memory_bytes` | int | `roofline.py` for T_SOL derivation (same SOL-vs-placeholder split as `flop_count`) |
| `input_shapes` | list[dict] | `correctness.py` test input generation for placeholder starters; on the SOL path this carries `definition.const_axes` only (variable + expr axes resolve from each `Workload`, not the spec) |
| `definition_path` | `Path \| None` | SOL-ExecBench `definition.json` path. Threaded into the profiler subprocess driver so it can re-load the problem and rebuild the (unpicklable) input generator. `None` on the placeholder path. |
| `pytorch_reference` | str | PyTorch `run()` source from `definition.json` — the correctness oracle. Empty string for placeholder starters. |
| `t_sol_us` | `float \| None` | SOLAR-derived hardware bound, populated at problem-load time when SOLAR returns a result. `None` when SOLAR is unavailable or soft-fails (caller falls back to `compute_roofline()`). |
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
| `triton_kernel_name` | str | Bare name of the `@triton.jit` device function the profiler filters NCU on. Defaults to `""` for hand-written starters / test fixtures — the profiler's priority chain (`Kernel.triton_kernel_name` → source-regex → `spec.entrypoint`) handles the empty case via the fallback. Coder-produced kernels populate it via the `submit_kernel` tool's Pydantic-validated argument. |
| `dps` | bool | Destination-passing-style flag. `True` means the host wrapper takes pre-allocated outputs after the inputs (`def kernel_fn(x, y, out)`) and the benchmark / correctness loops allocate buffers via `allocate_outputs(definition, workload, device)` and thread them through. `False` (default) means the wrapper returns its outputs as the function's return value. Default preserves back-compat with hand-written starters and pre-`dps` checkpoint round-trips. |

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

Factory functions producing **placeholder/metadata-only** `Kernel` instances for common operations. Each factory builds a `KernelSpec` (name, kernel_type, flop_count, memory_bytes, input_shapes) and returns a `Kernel` whose `source_code` is a literal comment stub like `"# placeholder matmul kernel"` — they are *not* compilable Triton baselines. Their real role is the placeholder CLI smoke path (`pipeline/optimize.py`'s `_load_placeholder` constructs `make_matmul_kernel(1024, 1024, 1024)` so the scaffold runs end-to-end without a model / SOL problem); on the SOL path the search tree's root kernel comes from `generate_triton_baseline`'s LLM-driven PyTorch→Triton port, not from these starters.

| File | Function | FLOPs estimate |
|------|----------|---------------|
| `matmul.py` | `make_matmul_kernel(M, N, K)` | 2MNK |
| `softmax.py` | `make_softmax_kernel(rows, cols)` | 5 * rows * cols |
| `layernorm.py` | `make_layernorm_kernel(batch, hidden)` | 5 * batch * hidden |
| `attention.py` | `make_attention_kernel(batch, heads, seq_len, head_dim)` | 4 * B * H * S^2 * D |
