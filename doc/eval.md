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

| Stage | What | On failure |
|-------|------|------------|
| 1. Smoke test | Single input, output matches baseline | Coder self-corrects |
| 2. Shape sweep | Multiple input sizes (tiny → xlarge) | Coder self-corrects |
| 3. Numerical stability | NaN/Inf detection, precision check | Coder self-corrects |
| 4. Determinism | Repeated runs produce identical outputs | Coder self-corrects |
| 5. Anti-cheat | Randomized inputs, strict tolerance | Coder self-corrects |

Any failure triggers the Coder's self-correction loop (up to `max_debug_retries`). If budget exhausted, branch is marked dead.

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

`build_reference_fn(pytorch_source)` execs SOL-ExecBench's PyTorch reference source and resolves the `run` callable. `build_input_generator(workload)` wraps SOL's `gen_inputs` with seeding so the same iteration index yields deterministic arguments. Both torch + sol_execbench imports are lazy so the module loads in the torch-free test venv.

## Profiler — `profiler.py`

Per-iteration diagnostic signals for the Reviewer. Two pieces:

- **Analytical (required)** — `_compute_analytical()` derives `AnalyticalMetrics` from `(flops, nbytes, latency_s, HardwareSpec)`: arithmetic intensity, ridge point, achieved TFLOPS + GB/s, pct-of-peak compute + bandwidth. Fails closed with `ProfilerError` on zero-latency or missing peaks (the orchestrator marks the branch DEAD_END).
- **NCU (best-effort)** — subprocess `ncu --csv --section ...` via a dedicated driver (`_profiler_driver.py`). Extracts curated signals: SM occupancy, L2 hit rate, tensor-core utilization, and the top-2 warp-stall classes with percentages. Failures degrade the result (`ncu=None, degraded=True, degraded_reason=<slug>`) but keep the branch alive — the analytical block still drives the Reviewer's profiling summary.

Returns `ProfilingResult(analytical, ncu, raw_metrics, degraded_reason)`. Bottleneck classification is **not** on `ProfilingResult` — it lives at the run level (see `classify_run` in `roofline.py`) because it's invariant per `(problem, representative workload, hardware)`.

Source-hash-keyed cache avoids re-running NCU on an identical kernel source.

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
