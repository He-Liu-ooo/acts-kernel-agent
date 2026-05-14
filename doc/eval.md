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

This table covers only the **eval-harness slice** of Phase A. The wider Phase A flow is orchestrated by `pipeline/optimize.py`: the `_load_problem` adapter dispatcher selects a benchmark format (currently SOL), `derive_t_sol_from_solar` fills the T_SOL + bottleneck pair, `validate_hardware_spec` (from `src/config.py`) reconciles the configured `HardwareSpec` with `detect_hardware()`, and the clock-lock primitives from `sol_execbench.core.bench.clock_lock` (`probe_clock_lock_available`, `acquire_clock_lock`) attempt to pin GPU clocks for the run — all before the search loop starts. See `doc/pipeline.md` for the end-to-end Phase A wiring.

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
| 1. Smoke test | Single input, output matches oracle | 42 | `atol/rtol` (default `1e-2`, mirrors SOL-ExecBench) † | Coder self-corrects |
| 2. Shape sweep | N trials with varying seeds | `0..n_sweep_trials-1` (default 5) | `atol/rtol` † | Coder self-corrects |
| 3. Numerical stability | Match oracle **and** output finite (no NaN/Inf) | 7 | `atol/rtol` † | Coder self-corrects |
| 4. Determinism | Match oracle **and** two runs on identical input are bitwise-equal | 11 | `atol/rtol` † | Coder self-corrects |
| 5. Anti-cheat | Randomized inputs under strict tolerance | `1000..1000+n_anti_cheat_trials-1` (default 3) | `strict_atol=1e-5`, `strict_rtol=1e-4` † | Coder self-corrects |

† When `workload.tolerance` is non-None (SOL-ExecBench's `ToleranceSpec`), its `max_atol` / `max_rtol` override **every** stage's tolerance — stages 1–4 *and* the stage 5 strict thresholds — so the gate matches the per-problem spec verbatim. Pass `workload=None` to keep the hardcoded defaults.

Stages short-circuit on first failure — a failing `CorrectnessResult` carries `failed_stage: CorrectnessStage` and `error_message`. Any failure triggers the Coder's self-correction loop (up to `max_debug_retries`); budget exhaustion marks the branch dead.

Stages 3 and 4 fuse the oracle compare with their domain check so a seed-7 or seed-11 wrong answer can't slip past by passing a pure finite-check or self-equality check.

### `ComparisonPolicy` Protocol

Tensor comparison is abstracted behind a `ComparisonPolicy` protocol (`compare`, `contains_non_finite`, `bitwise_equal`) so the module imports torch-free. Tests inject a scalar-backed policy; production uses `TorchComparisonPolicy`:

- Delegates to `sol_execbench.core.bench.correctness.compute_error_stats` with `ToleranceSpec(max_atol, max_rtol)` — `required_matched_ratio` is left at SOL's default (0.99 = 1% slack) so bf16 quantization outliers don't reject a mathematically correct kernel. Element-wise pass condition: `|output - reference| <= max_atol + max_rtol * |reference|`. This gives matched-ratio tolerance, separate NaN/Inf flags, and a hard max-error cap "for free."
- **SOL-ExecBench is mandatory.** `_try_import_sol()` raises `ImportError("TorchComparisonPolicy requires sol_execbench. ...")` on first `compare` call when SOL is absent — there is no `torch.allclose` fallback. A fallback's pass criterion (every element within tolerance) would diverge from SOL's matched-ratio rule and silently hide bf16 outliers in non-SOL test runs, so the policy fails closed instead.

### `verify_correctness` Contract

```python
verify_correctness(
    candidate_fn, reference_fn, input_generator,
    *,
    definition: Definition | None = None,
    kernel: Kernel | None = None,
    workload: Workload | None = None,
    policy=None, atol=1e-2, rtol=1e-2,
    strict_atol=1e-5, strict_rtol=1e-4,
    n_sweep_trials=5, n_anti_cheat_trials=3,
) -> CorrectnessResult
```

`input_generator(seed) -> tuple` produces fresh args for each trial. The Coder's correctness tool iterates over the **full** input-generator list (one per selected workload) and short-circuits on the first failing workload so the Coder sees the offending workload index and stage.

The three SOL-aware kwargs drive the multi-output / DPS flow:

- `definition` — when provided, both candidate and reference outputs flow through SOL's `normalize_outputs` so multi-output (tuple / dict) returns get compared name-by-name under per-name dtypes from `definition.outputs`. When `None`, the gate falls back to comparing raw outputs directly (preserves back-compat with non-SOL benchmarks and the scalar-policy unit tests that drive the gate without a `Definition`).
- `kernel` + `workload` — required when the candidate's host wrapper is destination-passing-style (`kernel.dps=True`). The gate then allocates output buffers per call via `allocate_dps_outputs(definition, workload, device=...)` (which delegates to `sol_execbench.core.bench.io.allocate_outputs`) and invokes `candidate_fn(*inputs, *outputs)`; the filled buffers serve as the candidate's outputs for the per-stage comparison. The reference oracle is always return-by-value — it's the PyTorch `run()` from `definition.json` and is never DPS — so the comparison sides line up via `normalize_outputs`. When `kernel` is `None` or `kernel.dps` is `False`, the gate calls `candidate_fn(*inputs)` and treats the return value as the output (legacy / non-DPS path).

`workload` also carries the per-problem tolerance: when `workload.tolerance` is non-None, its `max_atol` / `max_rtol` override the `atol` / `rtol` kwargs **and** the `strict_atol` / `strict_rtol` kwargs for stage 5. The override is opt-in via workload presence — callers that want the prior tighter anti-cheat behaviour pass `workload=None` and keep the hardcoded defaults.

### `anti_cheat.py`

Real implementation. Coordinates three layers of reward-hack defense:

**1. Correctness-level — Stage 5 of `verify_correctness`.** Randomized inputs at seeds `1000..1000+n_anti_cheat_trials-1` evaluated under strict tolerance — the `strict_atol=1e-5` / `strict_rtol=1e-4` defaults apply when `workload.tolerance` is None; otherwise the workload's `max_atol` / `max_rtol` override them (same override that drives stages 1–4 — see the footnote on the 5-stage table). Catches kernels that pass on canned seeds but diverge on random inputs.

**2. Performance-level — scorer flags.** `compute_sol_score` sets `reward_hack_suspect` when `T_k < T_SOL` (candidate beats hardware speed-of-light) and `calibration_warning` when `T_b <= T_SOL` (baseline already at limit). The orchestrator's `_reward_hack_re_eval` re-runs the candidate under SOL strict tolerance when `score.reward_hack_suspect` fires and emits `reward_hack_confirmed` (then `_kill_branch(reason=DeadReason.REWARD_HACK_CONFIRMED)`) or `reward_hack_cleared`.

**3. Process-level — SOL `reward_hack` detector set.** The `per_iter_anti_cheat(critical_names)` context manager wraps each evaluation:

- *On entry:* `snapshot_critical_functions(namespace, critical_names)` over `vars(torch.cuda.Event)` plus `check_monkey_patch()` — the latter validates SOL's module-load `_ELAPSED_TIME_ADDR` snapshot, fires only if a candidate patched `torch.cuda.Event.elapsed_time` after sol_execbench import. The import-order contract in `pipeline/optimize.py` (SOL imported first) keeps this snapshot trustworthy.
- *On exit (unconditional):* `check_thread_injection(threads_before, threading.active_count())` and `check_eval_integrity(snapshot, namespace)`. Run unconditionally so a body that raised for any reason still gets validated; a tampered run is more important to surface than whatever caused the body to error.

`check_lazy_outputs_after_bench(outputs)` runs after the bench loop and validates that every produced tensor is a strict `type(t) is torch.Tensor` (rejects FakeTensor / lazy proxies). Any `RewardHackDetected` raised by these checks propagates to the orchestrator's `_kill_branch(reason=DeadReason.REWARD_HACK)`.

**Helpers.**

- `AntiCheatContext` — dataclass holding `snapshot: dict[str, int]`, `namespace: MappingProxyType[str, Any]` (read-only view over `torch.cuda.Event`'s namespace, prevents accidental mutation across the per-iter window), and `threads_before: int`.
- `generate_randomized_inputs(input_generator, seed) -> list` — thin wrapper that materializes a tuple from the per-seed generator (used by Stage 5 + the re-eval).
- `strict_tolerance_check(candidate_output, reference_output, *, atol=1e-5, rtol=1e-4) -> bool` — strict-spec wrapper around `compute_error_stats` with `required_matched_ratio=1.0` (zero slack); used by the SOLAR-strict re-eval.

## Benchmark — `benchmark.py`

Measures kernel latency using CUDA events. Called by the orchestrator after the Coder returns a compiled, correct kernel; not part of the Coder's tool loop.

### Per-iteration protocol

Each timed iteration runs: `prepare → flush_l2 → record_start → kernel_fn(*args) → record_end → finalize_ms`. L2 is flushed **before** `record_start` so the kernel sees a cold cache and the flush is excluded from the measurement (KernelBench convention). Inputs are regenerated per iter outside the timing window so in-place kernels don't see degenerate inputs on later iterations.

### Burn-in call (autotune warm-up)

Before the warmup loop, `_time_workload` fires one untimed burn-in call using reserved seed `-1` (`_BURN_IN_SEED = -1` constant in `src/eval/benchmark.py`). This forces `@triton.autotune`'s config sweep + compile + per-config microbench to complete **outside** the timed window, so what flows into the median is cached-winner perf rather than first-call autotune overhead. The burn-in invokes one extra `prepare` (regenerating inputs) ahead of the recorded `[prepare, flush_l2, record_start, record_end, finalize_ms]` per timed iter. Failure inside the burn-in surfaces as a workload-level error on the same fail-closed path as warmup failure — latency = inf in `per_workload_latency_us`, reason recorded in `workload_errors`, half-survivors-or-die gate then fires.

### Autotune winner capture

`benchmark_kernel(..., autotuner=...)` snapshots `autotuner.cache` before each workload and diffs it after `_time_workload` returns. A single new cache entry is attributed to that workload and stored on `BenchmarkResult.autotune_winner_per_workload[workload.uuid]` with the usual `{"kwargs", "num_warps", "num_stages"}` shape. Zero-entry deltas (identical-shape cache hits) are left unattributed; multi-entry deltas log a warning and are skipped.

`benchmark_kernel` calls `autotuner.cache.clear()` once at entry (when an autotuner is supplied and `.cache.clear` is callable; an `AttributeError` flips `autotuner_usable=False` with a WARNING and proceeds without winner attribution). The clear is necessary because the Coder's correctness pass (`check_correctness_tool`) compiles and launches the same kernel before `benchmark_kernel` runs, so Triton's `@triton.autotune` may already have cached entries for our workload shapes — without the clear, the per-workload `set(autotuner.cache) - before` delta would be empty and every winner silently dropped. Tradeoff: correctness-pass autotune work is discarded and re-paid on the first warmup run of each workload, which is acceptable because the warmup loop exists for thermal/clock stabilization anyway.

### Autotune burn-in event

`autotune_burn_in_done` is registered in `runtime/events.py::CORE_EVENT_KINDS` and fired **once per iter** by the orchestrator after a successful bench loop with an Autotuner reference. Payload: `{iter, workload_count, winner_count}` — `workload_count` summarizes the iter's bench loop; `winner_count` is the number of workload UUIDs with captured winners.

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

Bridges SOL `Definition` + `Workload` directly to the pair of callables `verify_correctness` needs (plus the DPS allocator). The legacy ACTS `Problem` / `Workload` dataclasses are gone — SOL's pydantic models flow through unchanged, no shim layer.

- `build_reference_fn(source, entrypoint="run") -> Callable` — execs the PyTorch reference source (from `definition.json`'s `reference` field) into an isolated namespace and resolves the entrypoint symbol. Raises `ReferenceLoadError` when the entrypoint is missing or non-callable; `SyntaxError` / `ImportError` from the source propagate unchanged so the real cause is visible. Pure-Python (no torch import) so the module loads in the test venv.
- `build_input_generator(definition, workload, *, device="cuda", blob_roots: list[Path] | None = None) -> Callable[[int], tuple]` — wraps `sol_execbench.core.bench.io.gen_inputs`. Per-seed call resets RNG (`set_seed(seed)`) and generates fresh inputs. `blob_roots` is forwarded to `sol_execbench.core.bench.io.load_safetensors` when the workload declares any `SafetensorsInput` — blobs are resolved against the roots in order (first existing match wins) and loaded **once at build time**, so the on-disk read is excluded from the per-seed (and therefore per-iter) timing path. Torch and sol_execbench are lazy-imported so the module remains importable without the GPU stack.
- `allocate_dps_outputs(definition, workload, *, device="cuda") -> list` — pre-allocates DPS output buffers for `kernel_fn(*inputs, *outputs)` calls. Resolves the workload's axes against the definition once (`definition.get_resolved_axes_values(workload.axes)`), then delegates to `sol_execbench.core.bench.io.allocate_outputs`. Single source of truth for the DPS allocation shape — used by the correctness gate (`maybe_wrap_dps_candidate`), the benchmark loop, and the `_profiler_driver` NCU subprocess.

## Profiler — `profiler.py`

Per-iteration diagnostic signals for the Reviewer. Two pieces:

- **Analytical (optional)** — `_compute_analytical()` derives `AnalyticalMetrics` from `(flops, nbytes, latency_s, HardwareSpec)`, narrowed to per-iter dynamics only: `achieved_tflops`, `achieved_bandwidth_gb_s`, `pct_peak_compute`, `pct_peak_bandwidth`. Fails closed with `ProfilerError` on zero-latency, non-positive `nbytes`, negative `flops`, or zero-peak hardware (the orchestrator marks the branch DEAD_END). `profile_kernel` skips the analytical computation entirely when `nbytes == 0` (no byte count derivable for this iteration) and the resulting `ProfilingResult.analytical` is `None`; NCU still runs. Run-level invariants `arithmetic_intensity` and `ridge_point` (both in MACs/byte) live on `RooflineResult` (`src/eval/roofline.py`); the Reviewer prompt builder reads AI / ridge from `RooflineResult` and achieved_* / pct_peak_* from `AnalyticalMetrics`.
- **NCU (best-effort)** — subprocess `ncu --csv --print-metric-name=name --section ...` via a dedicated driver (`_profiler_driver.py`). Extracts curated signals: SM occupancy, L2 hit rate, tensor-core utilization, and the top-2 warp-stall classes with percentages. The full parsed metric map is retained as `raw_metrics`; `metric_groups` overlays present/missing status for diagnostic groups (`tensor_core`, `math_pipe`, `memory`, `occupancy`, `scheduler`, `launch`, `stalls`). Failures degrade the result (`ncu=None, degraded=True, degraded_reason=<slug>`) but keep the branch alive — the analytical block (when present) still drives the Reviewer's profiling summary.

Returns `ProfilingResult(analytical, ncu, raw_metrics, metric_groups, degraded_reason, ncu_rep_path)`. `analytical: AnalyticalMetrics | None` — `None` when `nbytes == 0` was passed in; consumers gate on `ProfilingResult.has_analytical` (mirrors `has_ncu`) before reading `achieved_*` fields. `ProfilingResult.make_degraded(analytical, reason)` accepts `analytical=None` so the degraded path works regardless of whether analytical ran. Bottleneck classification is **not** on `ProfilingResult` — it lives at the run level (see `classify_run` in `roofline.py`) because it's invariant per `(problem, representative workload, hardware)`.

### Curated metric set

Required (a missing one degrades the NCU result with `missing_metric:<name>`):

| Raw NCU metric | Field | Section |
|---|---|---|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | `sm_occupancy_pct` | `Occupancy` |
| `lts__t_sector_hit_rate.pct` | `l2_hit_rate_pct` | `MemoryWorkloadAnalysis` |

Optional (absence does NOT degrade — field is typed `float | None` and falls through to renderers as `n/a`):

| Raw NCU metric | Field | Section |
|---|---|---|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | `tensor_core_util_pct: float | None` | `ComputeWorkloadAnalysis` |

The tensor-core metric is also explicitly requested via `--metrics`, not only via `ComputeWorkloadAnalysis`, because some NCU exports omit it from the section output. The counter is optional because not every NCU/GPU/kernel combination emits it (e.g. Ada elementwise kernels); when absent, `tensor_core_util_pct` is `None` and the rest of the curated NCU block (occupancy, L2, stalls) is preserved. Only REQUIRED metrics gate degradation via `missing_metric:<name>`.

The explicit `--metrics` list also requests the stable members of the diagnostic groups above, plus all 18 stall metrics under the prefix `smsp__average_warp_latency_issue_stalled_<reason>.pct` because NCU does not expand wildcards. Top-2 stalls (by percentage, ties broken by reason name) populate `warp_stall_dominant`/`warp_stall_runner_up`.

`raw_metrics` preserves the full parsed NCU metric dict so future Reviewer prompts can reference metrics outside the curated set without a code change. `metric_groups` is generated from that dict with per-metric `{"status": "present", "value": ...}` or `{"status": "missing"}` entries, making missing data distinct from a real zero. Both are persisted to the per-key JSON cache and rehydrated on cache hits, so `Reviewer.query_metric` works the same on a re-profile as on the originating iteration. Legacy cache entries without a `raw` or `groups` field load with `raw_metrics={}` and generated missing-only groups for back-compat.

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

Input resolution priority: (1) `problem_dir` → `src.benchmarks.sol_execbench.load(dir)` + `build_input_generator(definition, workload)(seed)` (orchestrator path); (2) `module.make_inputs(seed)` if the source exposes it (self-contained kernel convention — primary Tier 2 path); (3) `spec["args"]` as last-resort literal; (4) `()`.

Host-callable resolution: prefers `module.run` (the Triton host wrapper that launches the JIT'd kernel via `fn[grid](...)`), falls back to `module.<entrypoint>`. `spec.entrypoint` is the host-wrapper name — **not** the GPU kernel symbol NCU filters on.

GPU-symbol resolution priority (T4): (1) `Kernel.triton_kernel_name` declared by the Coder via `KernelCodeOutput` and Pydantic-validated against the source's `@triton.jit def` matches; (2) `_extract_triton_kernel_name(source)` regex fallback (first `@triton.jit def`) — used for hand-written starters and test fixtures whose `Kernel.triton_kernel_name` is empty; (3) `kernel.spec.entrypoint` last-ditch (NCU degrades to `no_matching_kernel` rather than crash). The declared-name path is the load-bearing one for fused outputs with multiple `@triton.jit` defs — picking the first via regex would silently mis-profile a helper rather than the dominant kernel.

Subprocess invocation uses `sys.executable` (not bare `"python"`) so the child inherits the venv with torch/triton installed. `TMPDIR` is redirected to a user-scoped `/tmp/<user>_ncu` so `nsight-compute-lock` files owned by other users on shared hosts can't block the run.

### Two-subprocess architecture (capture + extract)

NCU 2025.x suppresses the CSV stream from stdout whenever `-o` is set on the profile call — the binary report is written but stdout carries only `==PROF==` banners; no flag combination delivers both binary `.ncu-rep` and CSV in a single invocation (`--log-file` only redirects what would have been printed). The driver therefore runs two subprocesses:

- **Profile call** — existing argv built by `_build_ncu_argv` (with `-f -o <rep_path-without-suffix>`) produces the binary `.ncu-rep` and a banner-only stdout that is discarded.
- **Extract call** — `_extract_ncu_csv` runs `ncu --import <rep_path> --csv --page details --print-metric-name=name`, which extracts the CSV from the binary report (no GPU work, just binary→CSV serialization). Its stdout is handed to `_parse_ncu_csv` with the unchanged signature.

The extract subprocess inherits the same `TMPDIR` env as the capture subprocess (post-Codex-review amendment, via the shared `_ncu_env()` helper) so the user-scoped tmpdir workaround for shared `/tmp` lock-ownership applies symmetrically across both calls.

`ProfilingResult.ncu_rep_path` is populated whenever NCU produced a `.ncu-rep` (success path); the orchestrator's `tree_dump.dump_node` reads it to copy the binary report into the run artefact tree. Cache-hit returns surface the path when the previously-written report is still on disk and `None` otherwise; degraded returns always carry `None`.

See JOURNAL 2026-05-08 entry "NCU `.ncu-rep` capture + CSV extraction — two-subprocess architecture" for the full rationale (rejected single-call alternatives, cost analysis).

### Failure taxonomy

| Reason slug | Cause | Behavior |
|---|---|---|
| `ncu_disabled_via_env` | Operator escape hatch via `ACTS_DISABLE_NCU=1` (truthy: `1` / `true` / `yes` / `on`, case-insensitive) | Degraded, short-circuits before subprocess fork |
| `ncu_skipped:permanently_unavailable:<sig>` | Process-wide cache flip on permanent-failure signatures (`ERR_NVGPUCTRPERM`, driver-init class, counter-permission class) | Degraded, short-circuits subsequent calls (no re-fork) |
| `ncu_binary_not_found` | `ncu` not on `$PATH` | Degraded, no cache write |
| `ncu_timeout` | Subprocess exceeded `timeout_s` (default 60s) | Degraded, no cache write |
| `ncu_nonzero_exit:<rc>` / `ncu_nonzero_exit:<rc>:<sanitized_stderr_first_line>` | Subprocess returned non-zero. When stderr is non-empty, the slug carries a fingerprint suffix (sanitized = slugified, capped 60 chars); empty stderr keeps the bare `ncu_nonzero_exit:<rc>` form | Degraded, no cache write |
| `ncu_import_failed:timeout` | The second subprocess (`ncu --import <rep_path> --csv --page details`) exceeded its timeout. Distinct from `ncu_timeout` (capture call) and from `csv_parse:no_header` (parser couldn't find a header in extract-call stdout) — `ncu_import_failed:*` means "binary `.ncu-rep` was captured but post-processing it via `ncu --import` failed." | Degraded, no cache write |
| `ncu_import_failed:<rc>` | The `ncu --import` subprocess returned non-zero exit code `<rc>`. Same distinction from `ncu_nonzero_exit` (capture call) and `csv_parse:*` (parser-side) as above | Degraded, no cache write |
| `csv_parse:<kind>` | Parser couldn't find header / columns | Degraded, no cache write |
| `no_matching_kernel` | `--kernel-name regex:` matched no row in the NCU CSV | Degraded |
| `missing_metric:<name>` | Required curated metric absent from CSV | Degraded |
| `stalls_incomplete` | Fewer than 2 stall metrics parsed | Degraded |

Every degraded `ProfilingResult` is announced at WARNING level via `logger = logging.getLogger("src.eval.profiler")` so silent failures are greppable from `run.log` alone. The factory `ProfilingResult.make_degraded(analytical, reason)` emits `ncu degraded: <slug>` and is called from each degraded return path in `profile_kernel`; the parser-degraded branch (which preserves `raw_metrics` and so cannot use the factory) emits the same `ncu degraded: <slug>` line inline. This was the regression-guard added 2026-05-08.

**Once-per-process permission cache.** When the first NCU invocation returns a permanent-failure signature (admin-only counters, driver-init failure), the module-level `_NCU_PERMANENTLY_UNAVAILABLE` flag in `src/eval/profiler.py` flips and every subsequent call short-circuits with `ncu_skipped:permanently_unavailable:<sig>` instead of re-forking `ncu`. This avoids paying the subprocess + driver-init cost on every iteration once we know counters are admin-only on this host. Operators can pre-empt the probe entirely by setting `ACTS_DISABLE_NCU=1`, which yields the `ncu_disabled_via_env` slug before any subprocess work.

Analytical failures raise `ProfilerError` and kill the branch. NCU failures never raise.

### Cache

Source-hash-keyed JSON cache: key = `sha256(source_hash + repr(workload) + mode + kernel_name + _METRIC_SET_VERSION)[:16]`. The resolved `kernel_name` (Coder-declared → regex → entrypoint) participates in the key so two `Kernel` objects with identical source but different declared `triton_kernel_name` values can't alias to one entry — without this, a fused output where the Coder renamed the dominant kernel would silently receive cached metrics NCU collected on a helper jit'd function. `_METRIC_SET_VERSION` is bumped when the curated metric map, stall reasons, parser contract, or *cache-key shape* changes so stale entries are unreachable; the v1→v2 bump (Codex P2 fix, 2026-04-22) was the cache-key-shape change that added `kernel_name`. Writes are atomic (`tempfile.mkstemp` + `os.replace`) and swallow OSError — caching is best-effort, never branch-killing.

### Modes

- `curated` (default) — `--section Occupancy WarpStateStats MemoryWorkloadAnalysis ComputeWorkloadAnalysis` plus explicit `--metrics` for tensor-core utilization, stable grouped diagnostics, and the enumerated stall metrics.
- `full` — `--set full` for debug; parser still pulls the curated subset, but `raw_metrics` captures everything NCU emitted.

## Types — `types.py`

Shared eval primitives imported across memory / search / pipeline without pulling in the full `roofline.py` / `profiler.py` modules. Hosts `BottleneckType` (`MEMORY_BOUND`, `COMPUTE_BOUND`, `BALANCED`). Kept in a leaf module so `eval/profiler.py` and `memory/experience.py` can both type-check against it without a circular import.

## Roofline — `roofline.py`

### Paths to T_SOL

Two paths, each returning both T_SOL and bottleneck classification — no hybrid:

1. **SOLAR** (preferred): `derive_t_sol_from_solar(definition, workload, hardware_spec, arch_yaml_path=None)` calls the SOLAR adapter on the PyTorch reference. Returns tight, hardware-grounded T_SOL + bottleneck. `arch_yaml_path` is set when the caller forwards `config.arch_config_path` from the .cfg so SOLAR's arch resolution can pick up an explicit YAML; left `None` triggers the name/fallback resolution path below. On bridge failure (e.g. unresolvable expr axis) the adapter logs a warning and returns `None`, and the caller falls back to the built-in roofline rather than crashing the load path. The returned `RooflineResult.source` is set to `"solar"`.
2. **Built-in** (fallback): `compute_roofline()` does `T_SOL = max(FLOPs / peak_compute, bytes / peak_bandwidth)` from `KernelSpec` fields + `HardwareSpec` (loaded from SOLAR arch YAML). Used when SOLAR is not installed or when the SOLAR path soft-fails. Sets `RooflineResult.source = "builtin"`.

Both classify the kernel as `MEMORY_BOUND`, `COMPUTE_BOUND`, or `BALANCED`.

### SOLAR adapter pipeline — `src/benchmark/solar_adapter.py`

Drives SOLAR via its published Python API (no subprocess) in four stages:

1. `PyTorchProcessor.process_model_file` — extract the PyTorch graph from a synthesized bridge file
2. `PyTorchToEinsum.convert` — convert to einsum representation
3. `EinsumGraphAnalyzer.analyze_graph` — count MACs and memory elements
4. `EinsumGraphPerfModel.predict` — apply the arch YAML's roofline; reads the `fused` section's `runtime_ms`, `bottleneck`, `arithmetic_intensity`, and `memory_bytes` (lifted onto `SolarResult.total_fused_bytes`), plus `perf["workload"]["total_flops"]` (lifted onto `SolarResult.total_flops`). Both count fields default to 0 via `.get(..., 0)` so SOLAR schema drift falls back to shape formulas rather than crashing.

`_write_model_bridge_file` synthesizes a SOLAR-shaped `Model(nn.Module)` + `get_inputs()` from the SOL `Definition` + a representative `Workload`. Handles const, var, and expr axes (fixed-point eval); 0-D tensors via `shape=[]`; and int/bool dtypes via `_tensor_constructor_call` (since `torch.randn` doesn't support those).

**Arch resolution priority** (in `_resolve_arch_config`):

1. Explicit `arch_yaml_path` (forwarded from `config.arch_config_path`)
2. SOLAR-bundled name (`H100_PCIe`, `B200`) — passed through as-is
3. ACTS-supplied YAML lookup via `_lookup_arch_yaml(detected_name)` — the `_ACTS_ARCH_YAMLS` registry and lookup helper now live in `src/config.py` as the single source of truth, consulted by both `solar_adapter._resolve_arch_config` and `config.detect_hardware()` (currently: `RTX6000Ada`, `NVIDIA RTX 6000 Ada Generation`, `placeholder-RTX6000Ada` — all → `configs/arch/RTX6000Ada.yaml`)
4. Fallback: `H100_PCIe` with a `WARNING` log

`SolarResult.bottleneck` is a `BottleneckType` enum (consistent with `RooflineResult.bottleneck`), not a raw SOLAR string.

### Unit policy and precision-aware ridge

Both `RooflineResult.arithmetic_intensity` and `RooflineResult.ridge_point` are in **MACs/byte**, regardless of `source`. The two paths fill them with different fidelity:

- **SOLAR-source** (the normal path): AI is exact — SOLAR's einsum analyzer counts MACs precisely; for kernels with no MACs (e.g. rmsnorm) AI = 0. Ridge_point is **precision-aware**, sourced from `perf["arch"]["ridge_point"]` which SOLAR computes as `mac_per_cycle / dram_byte_per_cycle` using the workload's actual dtype (e.g. `MAC_per_cycle_bf16_tc` for a bf16 workload, **not** `MAC_per_cycle_fp32_sm`). For RTX 6000 Ada, fp32 ridge ≈ 47.5 MACs/byte vs. bf16 ridge ≈ 190 MACs/byte (~4× difference) — same hardware, different ridge per dtype.
- **Built-in source** (fallback for non-SOL problems with explicit `KernelSpec.flop_count` / `memory_bytes`): AI is approximate via `MACs ≈ FLOPs / 2` — exact only for FMA-dominated kernels, over-estimates for elementwise / reduction-heavy kernels. Ridge_point uses FP32 peak (no precision metadata in the built-in path), so tensor-core workloads on the built-in path get an FP32-derived ridge that may misclassify them as compute-bound.

`RooflineResult` fields: `t_sol_us`, `bottleneck`, `source`, `arithmetic_intensity`, `ridge_point`, `total_flops`, `total_fused_bytes`. The two count fields carry SOLAR-derived absolute counts when `source == "solar"` (populated by `derive_t_sol_from_solar` from `SolarResult.total_flops` / `total_fused_bytes`) and stay at 0 on the built-in path; `compute_roofline_inputs` prefers them over its shape-formula fallback when both are positive. The previously documented `peak_achievable_tflops` field has been **removed** (was an unused stub). Bottleneck classification is still `MEMORY_BOUND` / `COMPUTE_BOUND` / `BALANCED`.

### Classification helpers

SOLAR is the sole bottleneck source post-2026-04-28 (see JOURNAL "SOLAR as sole bottleneck source"). The analytical band classifier remains as a fallback for `compute_roofline()` when SOLAR is unavailable, but is never the per-workload classifier in production.

- `classify_bottleneck(arithmetic_intensity, ridge_point) -> BottleneckType` — shared band (BALANCED within a narrow ratio of the ridge, otherwise MEMORY_BOUND / COMPUTE_BOUND). Used inside `compute_roofline()`'s no-SOLAR fallback path so the threshold can't drift.
- `classify_run(hardware, roofline, baseline_spec) -> BottleneckType` — once-per-run classification consumed by retriever / planner / reviewer. Returns `roofline.bottleneck` verbatim when a `RooflineResult` is provided (SOLAR is authoritative); otherwise falls back to `compute_roofline(baseline_spec, hardware)`. Called once by the orchestrator right after roofline resolution.
- Per-workload classification — `report.py::generate_report` calls `derive_t_sol_from_solar` once per selected workload and reads `RooflineResult.bottleneck`. Workloads where SOLAR is unavailable or returns `None` are omitted from `OptimizationReport.winner_per_workload_bottlenecks` rather than fall back to the analytical classifier (the omission is the signal "SOLAR couldn't classify this one"; the analytical formula's per-dtype peak limitation made it the wrong fallback for tensor-core workloads anyway).

`compute_roofline_inputs(definition, workload, *, roofline=None) -> (flops, nbytes)` in `src/benchmark/roofline_shapes.py` is **only** used to feed `_compute_analytical`'s arithmetic-intensity / %peak math at the two `profile_kernel` call sites (orchestrator per-iter + report.py Phase C re-profile). It is **not** used for bottleneck classification. Source preference: SOLAR counts on the optional `roofline` argument (when both `total_flops > 0` and `total_fused_bytes > 0`) → shape-formula fallback dispatched on `definition.op_type` → nbytes-only when only flops can't be derived (fused / `op_type=None` kernels with resolvable shapes; `_compute_analytical` accepts `flops=0`). Returns `(0, 0)` only when nbytes itself is unresolvable — callers then skip analytical entirely (`nbytes == 0` short-circuits the `_compute_analytical` call in `profile_kernel` and leaves `ProfilingResult.analytical=None`). The function's docstring lists the authoritative call sites; a third call site likely indicates a regression that re-routed bottleneck classification away from SOLAR.

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
