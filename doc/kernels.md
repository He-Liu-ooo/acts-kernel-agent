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

A single *version*: source code + autotune metadata parsed from it. Every search tree node holds one `Kernel`.

| Field | Type | Description |
|-------|------|-------------|
| `spec` | KernelSpec | Shared across all versions |
| `source_code` | str | Triton source code |
| `triton_kernel_name` | str | Bare name of the `@triton.jit` device function the profiler filters NCU on. Defaults to `""` for hand-written starters / test fixtures — the profiler's priority chain (`Kernel.triton_kernel_name` → source-regex → `spec.entrypoint`) handles the empty case via the fallback. Coder-produced kernels populate it via the `submit_kernel` tool's Pydantic-validated argument. |
| `dps` | bool | Destination-passing-style flag. `True` means the host wrapper takes pre-allocated outputs after the inputs (`def kernel_fn(x, y, out)`) and the benchmark / correctness loops allocate buffers via `allocate_outputs(definition, workload, device)` and thread them through. `False` (default) means the wrapper returns its outputs as the function's return value. Default preserves back-compat with hand-written starters and pre-`dps` checkpoint round-trips. |
| `autotune_configs` | list[dict] | Parsed at `__post_init__` from the `@triton.autotune(configs=[...])` decorator in `source_code` via stdlib `ast` (no Triton import — works in Tier-1 torchless venv). Each entry: `{"kwargs": {...}, "num_warps": int, "num_stages": int}`. Empty list when the source has no `@triton.autotune` (hand-written starters, test fixtures, legacy kernels). Parsing is lenient; the validator in `src/agents/coder.py::KernelCodeOutput` is what enforces presence for Coder-emitted source. |
| `autotune_keys` | list[str] | Arg-name strings parsed from the same decorator's `key=[...]` arg in the same AST pass (e.g., `["M", "N", "K"]` for matmul). Retained as kernel metadata and validated as non-empty for Coder output; winner attribution reads Triton's cache deltas directly instead of resolving these names against SOL axes. |
| `autotune_winner` | dict[str, dict] | Populated post-bench by the orchestrator from `BenchmarkResult.autotune_winner_per_workload`, keyed by `workload.uuid` with values matching the `autotune_configs` schema. Empty dict until the first benchmark with successful winner attribution; stays empty when cache-delta introspection soft-fails. |

Read `source_code` directly — it's the full Triton source string.

#### Legacy checkpoint migration

`Kernel.from_legacy_dict(data)` reconstructs a `Kernel` from a pre-A1 checkpoint dict by wrapping the old `(num_warps, num_stages, block_size)` triple into a single-entry `autotune_configs` list (`autotune_keys=[]`, `autotune_winner={}`). Used by `TreeNode` deserialization when legacy field keys are detected; new checkpoints never write the legacy schema. Both call sites — and the new-format path in `tree._deserialize_node` — share `KernelSpec.from_dict(data)` for the spec rebuild so the codec lives in one place.

#### `render_condensed_source(representative_workload_uuid=None) -> str`

Used by `src/search/orchestrator.py` to produce condensed parent source for the Planner and Reviewer prompts. The Coder uses `source_code` verbatim because it must edit the decorator block (the reframed `t1_block_size_tuning` action widens the autotune sweep).

Replaces the `@triton.autotune` decorator block with a single-line comment of the form:

```text
# autotune: BLOCK_M ∈ {64,128,256}, num_warps ∈ {4,8}, num_stages ∈ {2,3,4}, key=[M,N,K]
```

When `representative_workload_uuid` is supplied and present in `autotune_winner`, a second comment line is appended:

```text
# winner (representative wl): BLOCK_M=128, num_warps=4, num_stages=3
```

Falls back to verbatim `source_code` (entire return) when: source has no `@triton.autotune` decorator; AST parse fails; `autotune_configs` is empty; or the decorator's line span can't be located. All fallbacks degrade silently — the LLM still gets the kernel source.

The method is paired with the module-level `_find_autotune_decorator_span(source, triton_kernel_name)` AST helper, which returns the decorator's 1-indexed `(start_lineno, end_lineno)` span or `None`. Two format helpers (`_render_autotune_summary`, `_render_autotune_winner`) produce the comment lines from parsed config + winner data. A third helper `_flatten_autotune_config(cfg) -> dict` lifts the parser's nested `{"kwargs": {...}, "num_warps": int, "num_stages": int}` shape into a flat dict so `_render_autotune_winner` and the Coder's `autotune_exclude` validator (in `src/agents/coder.py`, see doc/agents.md) agree on what "flat" means.

`_autotune_span: tuple[int, int] | None` is a private cached span populated in `__post_init__` alongside `autotune_configs`/`autotune_keys` so `render_condensed_source` reads the decorator's source-line range from the cache instead of re-parsing.

## Compiler — `compiler.py`

Called by the Coder's `compile_kernel_tool` during its turn, by `eval/benchmark.py::_compile_entrypoint` before timing, by `eval/profiler.py::profile_kernel` to materialize the kernel for NCU's subprocess, and by `pipeline/verify.py` post-search.

- `compile_kernel(kernel, cache_dir=None) -> CompilationResult`: Source-hash-keyed file-backed import. Writes source to `<cache_dir>/<name>_<hash>.py` (hash = `sha256(source)[:12]`), loads via `importlib.util.spec_from_file_location` + `exec_module`, resolves `kernel.spec.entrypoint` via `getattr`. Returns `success`, `compiled_fn`, `error_message`, `source_path` (carries real filenames into tracebacks so the Coder can self-correct), and `triton_autotuner`. Defaults to `DEFAULT_CACHE_DIR = Path(".acts_cache/compiled")`.

### `triton_autotuner` resolution

`CompilationResult.triton_autotuner` holds the `@triton.autotune`-wrapped JIT kernel object when the source declares one, so the orchestrator can introspect Triton's per-config cache post-bench to populate `Kernel.autotune_winner`. The `_resolve_triton_autotuner(module, kernel)` helper returns `module.<kernel.triton_kernel_name>` only when that attribute resolves to an object exposing a `.cache` attribute (i.e. is wrapped in `@triton.autotune`); it returns `None` for legacy starters / fixtures with empty `triton_kernel_name`, for bare `@triton.jit` kernels (no `.cache`), and for any attribute-resolution failure. Best-effort: failures degrade to `None` so the success path is unaffected.

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
