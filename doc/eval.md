# Eval — `src/eval/`

Correctness verification, benchmarking, profiling, roofline analysis, and SOL scoring. Entirely deterministic (no LLM).

## Eval Harness Split

The eval harness is split across two call sites:

### Coder-Side (via `function_tool`)

Run inside the Coder agent's turn. By the time the Coder returns, the kernel is compiled and correct.

| Module | Tool | Purpose |
|--------|------|---------|
| `compiler.py` (in `kernels/`) | `compile_kernel_tool` | Triton compilation |
| `correctness.py` + `anti_cheat.py` | `check_correctness_tool` | 5-stage correctness gate |

### Problem-Load (once per problem, Phase A)

Run once at startup before the search loop. Results are constant for the entire optimization.

| Module | Purpose |
|--------|---------|
| `roofline.py` | T_SOL derivation + once-per-run bottleneck classification (`classify_run`) consumed by retriever / planner / reviewer every iteration |

### Orchestrator-Side (after Coder returns, every iteration)

Run by the orchestrator. Never part of the Coder's tool loop — prevents the LLM from gaming benchmark numbers.

| Module | Purpose |
|--------|---------|
| `benchmark.py` | Latency measurement via CUDA events |
| `profiler.py` | Analytical roofline metrics + curated NCU signals (per-iter diagnostics; bottleneck classification is NOT re-derived per iter) |
| `scorer.py` | SOL score computation (using static T_SOL from roofline.py) |

## 5-Stage Correctness Gate — `correctness.py`

| Stage | What | Seeds | Tolerance | On failure |
|-------|------|-------|-----------|------------|
| 1. Smoke test | Single input, output matches oracle | 42 | `atol/rtol` (default `1e-2`, mirrors SOL-ExecBench) | Coder self-corrects |
| 2. Shape sweep | N trials with varying seeds | `0..n_sweep_trials-1` (default 5) | `atol/rtol` | Coder self-corrects |
| 3. Numerical stability | Match oracle **and** output finite (no NaN/Inf) | 7 | `atol/rtol` | Coder self-corrects |
| 4. Determinism | Match oracle **and** two runs on identical input are bitwise-equal | 11 | `atol/rtol` | Coder self-corrects |
| 5. Anti-cheat | Randomized inputs under strict tolerance | `1000..1000+n_anti_cheat_trials-1` (default 3) | `strict_atol=1e-5`, `strict_rtol=1e-4` | Coder self-corrects |

Stages short-circuit on first failure — a failing `CorrectnessResult` carries `failed_stage: CorrectnessStage` and `error_message`. Any failure triggers the Coder's self-correction loop (up to `max_debug_retries`); budget exhaustion marks the branch dead.

Stages 3 and 4 fuse the oracle compare with their domain check so a seed-7 or seed-11 wrong answer can't slip past by passing a pure finite-check or self-equality check.

### `ComparisonPolicy` Protocol

Tensor comparison is abstracted behind a `ComparisonPolicy` protocol (`compare`, `contains_non_finite`, `bitwise_equal`) so the module imports torch-free. Tests inject a scalar-backed policy; production uses `TorchComparisonPolicy`:

- When `sol_execbench` is importable, delegates to `sol_execbench.core.bench.correctness.compute_error_stats` with `ToleranceSpec(max_atol, max_rtol)` — `required_matched_ratio` is left at SOL's default (0.99 = 1% slack) so bf16 quantization outliers don't reject a mathematically correct kernel. Element-wise pass condition: `|output - reference| <= max_atol + max_rtol * |reference|`. This gives matched-ratio tolerance, separate NaN/Inf flags, and a hard max-error cap "for free."
- Falls back to a local `torch.allclose` check when SOL-ExecBench is absent (keeps the module usable for non-SOL benchmarks).

### `verify_correctness` Contract

```python
verify_correctness(
    candidate_fn, reference_fn, input_generator,
    *, policy=None, atol=1e-2, rtol=1e-2,
    strict_atol=1e-5, strict_rtol=1e-4,
    n_sweep_trials=5, n_anti_cheat_trials=3,
) -> CorrectnessResult
```

`input_generator(seed) -> tuple` produces fresh args for each trial. The Coder's correctness tool iterates over the **full** input-generator list (one per selected workload) and short-circuits on the first failing workload so the Coder sees the offending workload index and stage.

### `anti_cheat.py`

Currently a placeholder (`generate_randomized_inputs`, `strict_tolerance_check`) marked as `skeleton` in PROCESS.md. The correctness-level anti-cheat surface is handled by Stage 5 above; the performance-level `reward_hack_suspect` flag is surfaced by `scorer.py`. The `anti_cheat.py` surface will fill in when the threat model becomes non-empty (see PROCESS.md → Deferred Improvements → "Reward-hack detection").

## Benchmark — `benchmark.py`

Measures kernel latency using CUDA events. Called by the orchestrator after the Coder returns a compiled, correct kernel; not part of the Coder's tool loop.

### Per-iteration protocol

Each timed iteration runs: `prepare → flush_l2 → record_start → kernel_fn(*args) → record_end → finalize_ms`. L2 is flushed **before** `record_start` so the kernel sees a cold cache and the flush is excluded from the measurement (KernelBench convention). Inputs are regenerated per iter outside the timing window so in-place kernels don't see degenerate inputs on later iterations.

### `BenchmarkTimer` Protocol

The timer is an injectable `Protocol` (`prepare` / `flush_l2` / `record_start` / `record_end` / `finalize_ms`). Production uses `_TorchCudaTimer` — `torch.cuda.Event` pairs plus a 256MB int64 L2-thrash tensor. Tests inject a `RecordingTimer` that returns a scripted elapsed sequence so dispatch / aggregation / call-order can be verified without torch.

### Multi-workload contract

`benchmark_kernel` accepts parallel lists `workloads: list[Workload]` and `input_generators: list[Callable[[int], tuple]]` (one generator per workload — the Coder's correctness tool uses the same list, see `inputs.py`). A fresh `BenchmarkTimer` is constructed per workload: a CUDA launch/event fault can leave the stream in a sticky error state, and reusing a timer would turn a workload-local failure into order-dependent false failures on subsequent workloads.

### Aggregation

Per workload: median of the timed samples (first `discard_first` dropped). Across workloads: median-of-medians as the scalar `median_latency_us`, with the full per-workload dict preserved on `BenchmarkResult.per_workload_latency_us`.

### Fail-closed semantics

| Failure | Behavior |
|---------|----------|
| Per-workload launch failure | Record `math.inf` in `per_workload_latency_us`, reason in `workload_errors` |
| Fewer than half the workloads survive | Raise `BenchmarkError` |
| Baseline partial-workload failure (orchestrator) | Abort run — baseline is the SOL-score denominator, partial failures make every downstream child meaningless |
| Child partial-workload failure (orchestrator) | Mark branch `DEAD_END` — branch-local, search continues |

`BenchmarkResult.is_fully_successful` is the property orchestrator checks (True iff `workload_errors` is empty) — call sites never touch the dict directly.

### Empty-workload sentinel

When both `workloads` and `input_generators` are empty (placeholder CLI path, no SOL problem loaded), `benchmark_kernel` returns a 100us sentinel result. Returning 0.0 would collapse `compute_sol_score` to 1.0 and silently fabricate an optimum.

## Inputs — `inputs.py`

Bridges SOL-ExecBench's `Problem` to the pair of callables `verify_correctness` needs.

- `build_reference_fn(source, entrypoint="run") -> Callable` — execs the PyTorch reference source (from `definition.json`'s `reference` field) into an isolated namespace and resolves the entrypoint symbol. Raises `ReferenceLoadError` when the entrypoint is missing or non-callable; `SyntaxError` / `ImportError` from the source propagate unchanged so the real cause is visible. Pure-Python (no torch import) so the module loads in the test venv.
- `build_input_generator(problem, workload, *, device="cuda") -> Callable[[int], tuple]` — wraps `sol_execbench.core.bench.io.gen_inputs`. Validates SOL's pydantic `Definition` + `Workload` models **once** at build time so the per-seed call only pays the RNG reset (`set_seed(seed)`) plus input generation. Lazy-imports torch + sol_execbench so the module stays importable without the GPU stack.

`_problem_to_sol_dict` / `_workload_to_sol_dict` shim ACTS's hand-written dataclasses into SOL's pydantic shape; the "adopt SOL pydantic end-to-end" cleanup is tracked as a Deferred Improvement in PROCESS.md.

## Profiler — `profiler.py`

Per-iteration diagnostic signals for the Reviewer. Two pieces:

- **Analytical (required)** — `_compute_analytical()` derives `AnalyticalMetrics` from `(flops, nbytes, latency_s, HardwareSpec)`: arithmetic intensity, ridge point, achieved TFLOPS + GB/s, pct-of-peak compute + bandwidth. Fails closed with `ProfilerError` on zero-latency, non-positive `nbytes`, negative `flops`, or zero-peak hardware (the orchestrator marks the branch DEAD_END).
- **NCU (best-effort)** — subprocess `ncu --csv --print-metric-name=name --section ...` via a dedicated driver (`_profiler_driver.py`). Extracts curated signals: SM occupancy, L2 hit rate, tensor-core utilization, and the top-2 warp-stall classes with percentages. Failures degrade the result (`ncu=None, degraded=True, degraded_reason=<slug>`) but keep the branch alive — the analytical block still drives the Reviewer's profiling summary.

Returns `ProfilingResult(analytical, ncu, raw_metrics, degraded_reason)`. Bottleneck classification is **not** on `ProfilingResult` — it lives at the run level (see `classify_run` in `roofline.py`) because it's invariant per `(problem, representative workload, hardware)`.

### Curated metric set

Required (a missing one degrades the NCU result with `missing_metric:<name>`):

| Raw NCU metric | Field | Section |
|---|---|---|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | `sm_occupancy_pct` | `Occupancy` |
| `lts__t_sector_hit_rate.pct` | `l2_hit_rate_pct` | `MemoryWorkloadAnalysis` |

Optional (defaults to 0.0 when absent — tensor-core metric is missing on NCU 2025.1.1.0 for pure-memory kernels, so it's demoted to avoid killing memory-bound runs):

| Raw NCU metric | Field | Section |
|---|---|---|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | `tensor_core_util_pct` | `ComputeWorkloadAnalysis` |

Warp stalls are explicitly enumerated (18 reasons) under the prefix `smsp__average_warp_latency_issue_stalled_<reason>.pct` because NCU does not expand wildcards; top-2 stalls (by percentage, ties broken by reason name) populate `warp_stall_dominant`/`warp_stall_runner_up`.

`raw_metrics` preserves the full parsed NCU metric dict so future Reviewer prompts can reference metrics outside the curated set without a code change.

### NCU subprocess driver — `_profiler_driver.py`

NCU wraps a fresh Python subprocess that imports the compiled kernel and launches it **once** (after one warmup). The driver reads a JSON spec file (path passed as its sole argv) with shape:

```json
{
  "kernel_source_path": "<abs path to compiled .py>",
  "entrypoint": "kernel_fn",
  "workload": {"uuid": "...", "axes": {...}, "inputs": {...}},
  "mode": "curated" | "full",
  "problem_dir": "<abs path to SOL problem dir>",  // optional
  "seed": 0                                         // optional
}
```

Input resolution priority: (1) `problem_dir` → `load_problem(dir)` + `build_input_generator(problem, workload)(seed)` (orchestrator path); (2) `module.make_inputs(seed)` if the source exposes it (self-contained kernel convention — primary Tier 2 path); (3) `spec["args"]` as last-resort literal; (4) `()`.

Host-callable resolution: prefers `module.run` (the Triton host wrapper that launches the JIT'd kernel via `fn[grid](...)`), falls back to `module.<entrypoint>`. `spec.entrypoint` is the host-wrapper name — **not** the GPU kernel symbol NCU filters on.

GPU-symbol resolution priority (T4): (1) `Kernel.triton_kernel_name` declared by the Coder via `KernelCodeOutput` and Pydantic-validated against the source's `@triton.jit def` matches; (2) `_extract_triton_kernel_name(source)` regex fallback (first `@triton.jit def`) — used for hand-written starters and test fixtures whose `Kernel.triton_kernel_name` is empty; (3) `kernel.spec.entrypoint` last-ditch (NCU degrades to `no_matching_kernel` rather than crash). The declared-name path is the load-bearing one for fused outputs with multiple `@triton.jit` defs — picking the first via regex would silently mis-profile a helper rather than the dominant kernel.

Subprocess invocation uses `sys.executable` (not bare `"python"`) so the child inherits the venv with torch/triton installed. `TMPDIR` is redirected to a user-scoped `/tmp/<user>_ncu` so `nsight-compute-lock` files owned by other users on shared hosts can't block the run.

### Failure taxonomy

| Reason slug | Cause | Behavior |
|---|---|---|
| `ncu_binary_not_found` | `ncu` not on `$PATH` | Degraded, no cache write |
| `ncu_timeout` | Subprocess exceeded `timeout_s` (default 60s) | Degraded, no cache write |
| `ncu_nonzero_exit:<rc>` | Subprocess returned non-zero | Degraded, no cache write |
| `csv_parse:<kind>` | Parser couldn't find header / columns | Degraded, no cache write |
| `no_matching_kernel` | `--kernel-name regex:` matched no row in the NCU CSV | Degraded |
| `missing_metric:<name>` | Required curated metric absent from CSV | Degraded |
| `stalls_incomplete` | Fewer than 2 stall metrics parsed | Degraded |

Analytical failures raise `ProfilerError` and kill the branch. NCU failures never raise.

### Cache

Source-hash-keyed JSON cache: key = `sha256(source_hash + repr(workload) + mode + kernel_name + _METRIC_SET_VERSION)[:16]`. The resolved `kernel_name` (Coder-declared → regex → entrypoint) participates in the key so two `Kernel` objects with identical source but different declared `triton_kernel_name` values can't alias to one entry — without this, a fused output where the Coder renamed the dominant kernel would silently receive cached metrics NCU collected on a helper jit'd function. `_METRIC_SET_VERSION` is bumped when the curated metric map, stall reasons, parser contract, or *cache-key shape* changes so stale entries are unreachable; the v1→v2 bump (Codex P2 fix, 2026-04-22) was the cache-key-shape change that added `kernel_name`. Writes are atomic (`tempfile.mkstemp` + `os.replace`) and swallow OSError — caching is best-effort, never branch-killing.

### Modes

- `curated` (default) — `--section Occupancy WarpStateStats MemoryWorkloadAnalysis ComputeWorkloadAnalysis` plus the enumerated stall `--metrics`.
- `full` — `--set full` for debug; parser still pulls the curated subset, but `raw_metrics` captures everything NCU emitted.

## Types — `types.py`

Shared eval primitives imported across memory / search / pipeline without pulling in the full `roofline.py` / `profiler.py` modules. Hosts `BottleneckType` (`MEMORY_BOUND`, `COMPUTE_BOUND`, `BALANCED`). Kept in a leaf module so `eval/profiler.py` and `memory/experience.py` can both type-check against it without a circular import.

## Roofline — `roofline.py`

### Paths to T_SOL

Two paths, each returning both T_SOL and bottleneck classification — no hybrid:

1. **SOLAR** (preferred): `derive_t_sol_from_solar()` calls the SOLAR adapter on the PyTorch reference. SOLAR uses its own arch config internally. Returns tight, hardware-grounded T_SOL + bottleneck.
2. **Built-in** (fallback): `compute_roofline()` does `T_SOL = max(FLOPs / peak_compute, bytes / peak_bandwidth)` from `KernelSpec` fields + `HardwareSpec` (loaded from SOLAR arch YAML). Used when SOLAR is not installed.

Both classify the kernel as `MEMORY_BOUND`, `COMPUTE_BOUND`, or `BALANCED`.

### Classification helpers

- `classify_bottleneck(arithmetic_intensity, ridge_point) -> BottleneckType` — shared band (BALANCED within a narrow ratio of the ridge, otherwise MEMORY_BOUND / COMPUTE_BOUND). Every classifier funnels through this so the threshold can't drift between callers.
- `classify_run(hardware, roofline, baseline_spec) -> BottleneckType` — once-per-run classification consumed by retriever / planner / reviewer. Prefers the baseline's shape-derived AI when available; otherwise uses the roofline's AI. Called once by the orchestrator right after roofline resolution.
- `classify_workload(problem, workload, hardware) -> BottleneckType` — per-workload classification for Phase C's `OptimizationReport.winner_per_workload_bottlenecks`. Derives `(flops, nbytes)` from `compute_roofline_inputs` and feeds `classify_bottleneck`. Raises `ValueError` on no-formula ops or zero-peak hardware.

## SOL Score — `scorer.py`

```
S(T_k) = (T_b - T_SOL) / ((T_k - T_SOL) + (T_b - T_SOL))
```

| Condition | Score | Meaning |
|-----------|-------|---------|
| T_k = T_b | 0.5 | Matches baseline |
| T_k = T_SOL | 1.0 | Hardware speed-of-light |
| T_k → ∞ | → 0 | Regression |

### Audit flags

The formula assumes `T_b > T_SOL` and `T_k >= T_SOL`. `ScoreResult` includes two flags for when these are violated (per SOL-ExecBench paper Section 4.3):

- `reward_hack_suspect` (`T_k < T_SOL`): Candidate beats hardware speed-of-light. Raw score > 1.0 is preserved (not clamped) as the signal. Routes to performance-level anti-cheat inspection.
- `calibration_warning` (`T_b <= T_SOL`): Baseline already at limit. Score set to 1.0. May indicate SOLAR bound is too loose.

This is real implemented logic (not a placeholder).
