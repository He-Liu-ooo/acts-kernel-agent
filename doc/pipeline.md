# Pipeline — `src/pipeline/`

End-to-end optimization entry points.

## optimize.py — Main Entry Point

CLI:

```
python -m src.pipeline.optimize [problem_path] [--run-dir DIR] [--trace-dir DIR] [--reset-clocks]
```

- `problem_path` (positional, optional) — SOL-ExecBench problem directory (contains `definition.json` + `workload.jsonl`), or the literal string `placeholder` for the built-in matmul demo. Default `"placeholder"` preserves the no-LLM smoke path.
- `--run-dir DIR` (optional) — parent directory for per-invocation run artifacts. Defaults to `./runs`. Each invocation creates `<run-dir>/run_<YYYYMMDDTHHMMSS_ffffffZ>/` (see "Run artifacts" below).
- `--trace-dir DIR` (optional) — directory for per-run JSONL trace files capturing every LLM input/output, tool call, and span via `src.agents.trace_processor.JSONLTraceProcessor`. Default is `None`: when omitted, SDK traces land under `<run-dir>/traces/` inside the per-invocation run directory. Passing `--trace-dir <path>` relocates the traces to `<path>`. Passing `--trace-dir=` (empty string) is a kill switch — no capture.
- `--reset-clocks` — operator escape hatch. Resets GPU 0 clocks (`-rgc -i 0` + `-rmc -i 0`) and exits without running any pipeline phase. Use after a SIGKILL / segfault left clocks locked from a prior run; the in-process atexit + signal handler chain (SIGTERM / SIGHUP) covers the normal exit cases.

### Phase A: Load Problem

`optimize()` takes a `problem_path` that is either a SOL-ExecBench problem directory (contains `definition.json` + `workload.jsonl`) or the literal string `"placeholder"` for the built-in matmul demo. SOL mode is the real path; placeholder mode keeps the CLI runnable without an LLM or SOL dependency.

**Adapter dispatch** (`_load_problem`): real-problem paths funnel through a thin adapter dispatcher before reaching the SOL-specific loader. Precedence:

1. `config.benchmark_adapter` — explicit override (`"sol_execbench"`, `"kernelbench"`); unknown values raise `UnknownBenchmarkFormat`. `"kernelbench"` raises `NotImplementedError` until that adapter ships.
2. `definition.json` present → SOL-ExecBench adapter (`_load_sol_problem`).
3. `model.py` present → KernelBench (`NotImplementedError`).
4. otherwise raise `UnknownBenchmarkFormat`.

**SOL mode** (`_load_sol_problem`, dispatched from `_load_problem`):

0. **Fail-fast SOLAR guard**: if `is_solar_available()` returns False (the `solar` Python package not installed in the run venv), `_load_sol_problem` raises `RuntimeError` with the canonical install hint: `uv pip install -e <SOLAR_PATH> --no-deps` + `uv pip install torchview`. SOLAR is REQUIRED for SOL-ExecBench problems — the previous silent fallback to `compute_roofline()` produced `t_sol_us=0.0` on SOL specs (their `flop_count` / `memory_bytes` are zero by construction), corrupting every score.
1. `src.benchmarks.sol_execbench.load(path)` parses the SOL definition + workloads into the pydantic `Definition` + `list[Workload]` pair.
2. `_definition_to_kernel_spec(definition, definition_path)` (in `pipeline/optimize.py`) derives the `KernelSpec` (name, kernel_type, `definition_path`, `pytorch_reference`); `flop_count` / `memory_bytes` stay 0 because SOLAR / `compute_roofline_inputs` derive them from workload axes, not the static definition.
3. `derive_t_sol_from_solar()` produces the roofline result; `spec.t_sol_us` is populated when SOLAR returns data.
4. `select_workloads()` samples `config.benchmark_workload_count` representative workloads.
5. **`generate_triton_baseline()`** (see `baseline_generator.py` below) drives `CoderAgent.translate()` to port the PyTorch reference into a Triton kernel, post-verifying on every selected workload. The returned `Kernel` is the search-tree root.
6. `build_reference_fn()` + `build_input_generator()` produce the oracle + one generator per workload; these are forwarded to `Orchestrator.run()` so Phase B's correctness tool binds to every workload the baseline was verified against.

**Model load** (`_load_model_if_configured`): reads `$ACTS_MODEL_CONFIG` or falls back to `configs/models/deepseek.json`. Gated on SOL mode — placeholder mode intentionally runs with `model=None` so the CLI stays executable without credentials. Without an SDK install or without a model config on disk, returns `None` and every agent stays in no-op mode.

**Placeholder mode** (`_load_placeholder`): loads `make_matmul_kernel(1024, 1024, 1024)` directly; no oracle, no workloads, no roofline. Exercises the scaffold end-to-end only.

### Clock-lock lifecycle

GPU clocks are locked at run start so per-iter latency isn't poisoned by DVFS-driven frequency drift. The lock targets GPU 0 only (the GPU ACTS uses); GPUs 1+ on a multi-GPU host are untouched.

- **Probe** — `probe_clock_lock_available()` (from SOL) checks whether passwordless `sudo nvidia-smi` is configured. The call site does defensive both-shape handling for the bare-`bool` return (today) vs a future tuple-`(bool, str)` return.
- **Preset lookup** — `_resolve_clock_preset(device_name)` consults the ACTS-side `_ACTS_CLOCK_PRESETS` table first (currently `RTX 6000 Ada → ClockPreset(2505, 10001)`) and falls back to SOL's `get_clock_preset` for known datacenter cards.
- **Lock** — `_lock_gpu0_clocks(gpu_mhz, dram_mhz)` runs `nvidia-smi -lgc <m>,<m> -i 0` and `-lmc <m>,<m> -i 0`. GPU 0 only.
- **Verify** — `_verify_gpu0_locked(gpu_mhz, dram_mhz)` first wakes GPU 0 with a tiny torch op (the lock sets the *application* clock, which only manifests when the GPU has work; querying current clock on an idle GPU returns 210 MHz, not the locked target). Then queries `nvidia-smi -i 0` and compares with 50 MHz tolerance.
- **Lifecycle** — lock attempted at run start (after `run_start` event); state stored in `_clock_lock_state` (locked + device_name); cleanup goes through three paths: (a) explicit `finally` in `main()`, (b) `atexit.register(_unlock_clocks_safe)`, (c) signal handlers `_signal_unlock_handler` for `SIGTERM` + `SIGHUP`. Operator escape hatch via `--reset-clocks` covers the SIGKILL / segfault case.
- **Verify-failure semantics** — `_verify_gpu0_locked` returning False or raising triggers `_rollback_partial_lock(device, reason)`, which calls `_unlock_gpu0_clocks` and emits `clock_lock_unavailable` with `reason="verify_failed"` or `reason="verify_raised:<exc>"`. Lock state stays False.
- **Event surface** — `clock_lock_unavailable` reasons go through a `ClockLockReason` `StrEnum` (`OK` / `PROBE_RETURNED_FALSE` / `NO_PRESET` / `LOCK_FAILED` / `VERIFY_FAILED` / `UNKNOWN`); exception-derived reasons are free-form strings (`verify_raised:<exc>`).

### Phase B: Search Loop

Delegates to `Orchestrator.run()`. Runs up to `max_depth` iterations with 3 agents (Planner → Coder → Reviewer). The `reference_fn` + `input_generators` returned by Phase A are forwarded verbatim every iteration so the Coder's correctness tool remains bound to the problem's oracle.

### Phase C: Report

`generate_report(result, ...)` walks the root-to-best path on `result.tree` to build the `technique_trace`, carries the audit flags (`reward_hack_suspect`, `calibration_warning`) off the best node's `ScoreResult`, and unwraps `termination_reason` to a plain string. `render_report` formats the report for the CLI and surfaces audit flags as explicit `[AUDIT]` lines so a flagged run can't be skimmed past.

`optimize()` returns `(SearchResult, OptimizationReport)` — a 2-tuple. The report is built inside `optimize()` (not in `main()`) so the rich Phase A locals (`definition`, `workloads`, `hardware_spec`, `arch_yaml_path`, `blob_roots`) reach `generate_report` directly. `main()` unpacks the tuple and renders.

SOL `Trace` payloads are emitted per evaluation (`trace_emitted` event, built by `Orchestrator._emit_trace`) so per-evaluation environment + correctness + performance records land in `events.jsonl` alongside the score line.

## baseline_generator.py — Triton Baseline Generation

`generate_triton_baseline(definition, spec, *, coder, workloads, max_retries=3, cache_dir=None, policy=None, blob_roots=None) -> Kernel`

Runs at problem-load time. Drives `CoderAgent.translate()` to port the PyTorch reference into Triton, then post-verifies: recompiles the returned source and reruns the 5-stage correctness gate against every workload in *workloads*. The post-verify catches SDK best-effort output when the Coder's turn budget was exhausted. Returns the first candidate that compiles and passes correctness on all workloads.

**Fail-closed contract** — raises `BaselineGenerationError` when:
- `coder is None` or `coder.has_model is False` (no model configured). Search against a fake baseline would silently look like progress, so there is intentionally no stub fallback.
- `max_retries` attempts are exhausted without a verified candidate.

`ValueError` is raised for a caller bug — an empty `workloads` list.

## verify.py — Post-Optimization Verification

Re-runs the correctness gate on the best kernel to confirm results are reproducible. Recompiles the winner, then delegates to `verify_correctness` against the PyTorch reference. Compile failures surface as `passed=False` with a compile-phrased detail string.

`verify_optimized_kernel(optimized, *, reference_fn, input_generator, definition=None, workload=None, policy=None, cache_dir=None) -> VerificationResult`

`definition` + `workload` are required when `optimized.dps` is True so the gate can pre-allocate output buffers via `allocate_outputs(definition, resolved_axes, device)`; both are passed through to `verify_correctness`.

## report.py — Report Generation

`generate_report(result, *, workloads=None, input_generators=None, hardware_spec=None, cache_dir=None, definition=None, definition_path=None, blob_roots=None, arch_yaml_path=None) -> OptimizationReport`

Reads the best node's `ScoreResult` and walks `result.tree.path_to_node(best.id)` to build the root-to-best action sequence. The root's `action_applied` is the empty-string baseline placeholder and is filtered out of the trace. When `best.score is None` (scoring failed), the returned report surfaces only `termination_reason` + `total_iterations` without crashing.

When `workloads` + `hardware_spec` are supplied, `generate_report` iterates the selected workloads once. Per-workload bottlenecks are sourced from SOLAR via `derive_t_sol_from_solar(definition, workload, hardware_spec, arch_yaml_path=...).bottleneck` (SOLAR is authoritative). Workloads where SOLAR returns `None` or where `definition is None` are **omitted** from `winner_per_workload_bottlenecks` rather than falling back to the analytical band classifier (`classify_bottleneck` in `eval/roofline.py` is the analytical band classifier; it is no longer used for per-workload labels). When `input_generators` is also supplied, the same loop re-profiles the winning kernel on each workload into `winner_profiling_per_workload`. The two passes are fused so `(flops, nbytes)` are computed once per workload and shared between SOLAR-call and the re-profile call. `blob_roots` is forwarded into `profile_kernel` so the NCU subprocess driver resolves safetensors-backed inputs against the same root list as the in-process generator (defaults to `[definition_path.parent]` when unspecified).

| Field | Type | Description |
|-------|------|-------------|
| `baseline_latency_us` | float | Starting latency |
| `best_latency_us` | float | Best achieved latency |
| `sol_score` | float | Final SOL score |
| `speedup` | float | Baseline / best |
| `technique_trace` | `list[str]` | Root-to-best action sequence (root baseline filtered out) |
| `bottleneck` | `BottleneckType \| None` | Once-per-run classification, copied verbatim from `SearchResult.run_bottleneck` (produced by `classify_run` in `eval/roofline.py`). `None` on the placeholder path that has no roofline. |
| `winner_per_workload_bottlenecks` | `dict[str, BottleneckType]` | Per-workload classification sourced from SOLAR via `derive_t_sol_from_solar(...).bottleneck`, keyed by `Workload.uuid`. Populated only when `workloads` + `hardware_spec` + `definition` are all provided; workloads where SOLAR returns `None` (or `definition is None`) are **omitted** — no analytical-band fallback. |
| `winner_profiling_per_workload` | `dict[str, ProfilingResult]` | Phase C re-profile of the winning kernel on every selected workload (spec §3.4). Empty when `input_generators` is missing. |
| `remaining_headroom_pct` | float | Distance to hardware limit, `(1 - sol_score) * 100` |
| `total_iterations` | int | Search iterations run |
| `termination_reason` | str | Why search stopped (plain string, unwrapped from `TerminationReason` enum) |
| `reward_hack_suspect` | bool | Propagated from best node's `ScoreResult` — candidate beats `T_SOL` |
| `calibration_warning` | bool | Propagated from best node's `ScoreResult` — baseline already at/below `T_SOL` |

`render_report(report: OptimizationReport) -> str`

Multi-line CLI summary. Skips the scoring block when `baseline_latency_us == 0` so a degenerate run (no scored best node) doesn't print misleading "0.00us / 0.00x" lines. Emits `Bottleneck (run): <label>` when `report.bottleneck` is set, and `Bottleneck (per workload): uuid=label, ...` when the per-workload dict is non-empty (enum values are rendered via `.value` at the string boundary). When `reward_hack_suspect` / `calibration_warning` are set, emits an `[AUDIT]` line per flag so operators scanning the output can't miss a physics-violating or poorly-calibrated result.

When `winner_profiling_per_workload` is populated, a "Winner profile (per workload)" block follows, with one analytical line per workload plus optional NCU lines. If every per-workload profile is degraded with `ncu_binary_not_found` (common on machines without the NCU CLI), the NCU block is suppressed to keep the output tidy.

#### Per-workload latency degraded rendering

- **Source of per-workload latency** — `child.per_workload_latency_us[uuid]` is populated by the orchestrator from `BenchmarkResult.per_workload_latency_us` after the per-iter benchmark.
- **Degraded path** — when a workload's UUID is missing from the dict (or its value is non-finite), `_render_profiling_block` in `src/pipeline/report.py` constructs a sentinel via `_degraded_for_missing_latency` (factory in the same file) — sets a `ProfilingResult` with empty `raw_metrics`, no `ncu`, and `degraded_reason = _DEGRADED_LATENCY_REASON = "per_workload_latency_missing"`.
- **Render** — the workload's analytical line is replaced with `[DEGRADED: missing per-workload latency — analytical metrics suppressed]`. The roofline summary (`AI`, `ridge`, `bottleneck`) still renders since those don't depend on latency.
- **Why** — the previous silent fallback to `aggregate_latency_s` (the run-level candidate latency) paired one workload's `nbytes` with another workload's `latency_s`, producing impossible apparent bandwidths (e.g., `bw 4839.4%`). The degraded path fails closed instead.
- **Known issue (2026-05-02)** — `child.per_workload_latency_us` is currently missing UUIDs for all workloads on the live run (`runs/run_20260502T060558_692431Z`). Three suspects: (a) per-workload bench failure → `inf` sentinel; (b) per-workload UUID never reaching the dict from a partial-failure path in `eval/benchmark.py`; (c) stale checkpoint. Investigation pending — see PROCESS.md "Active phase" queue item 1.

## Running the Pipeline

**Placeholder mode** — the default CLI (`python -m src.pipeline.optimize`, no positional arg) runs the matmul starter without GPU, LLM, or SOL-ExecBench. `main()` wraps its body in a `RunContext` (from `src/runtime/run_context.py`) that owns run-dir creation, logging config, and trace-processor wiring (replaced the removed `_enable_traces_if_possible` helper). It resolves `args.problem_path == "placeholder"`, runs `optimize("placeholder")`, and prints `render_report(generate_report(result))`. No model is loaded — every agent stays in no-op mode, the baseline comes from `make_matmul_kernel`, and with no workloads `benchmark_kernel` returns its 100us sentinel so the report emits a scoring block with baseline == best (speedup 1.00x). This only exercises the scaffold end-to-end; it is not a meaningful search result.

**SOL mode** — pass a SOL-ExecBench problem directory as the positional argument: `python -m src.pipeline.optimize /abs/path/to/sol/problem/` (or from a Python caller: `optimize(problem_path=<sol-dir>)`). Requires `configs/models/<provider>.json` (or `$ACTS_MODEL_CONFIG` pointing at one) and the `openai-agents` SDK installed; `generate_triton_baseline` fails closed otherwise with `BaselineGenerationError`.

### Run artifacts

Every CLI invocation creates a fresh `<run-dir>/run_<YYYYMMDDTHHMMSS_ffffffZ>/` directory (default `./runs/run_<UTC>/`) holding three files:

- `run.log` — human-readable text log of the invocation.
- `events.jsonl` — structured event stream (27 kinds in `CORE_EVENT_KINDS`) emitted by the orchestrator and `RunContext`.
- `traces/acts_trace_<UTC>.jsonl` — SDK per-call records (LLM inputs/outputs, tool calls, spans) written by `JSONLTraceProcessor`. Relocated when `--trace-dir <path>` is passed; absent when `--trace-dir=` disables capture.

The `httpx`, `openai`, and `agents` SDK loggers are silenced to WARNING so `run.log` stays focused on pipeline events.

Live-watch one-liners:

```bash
tail -f runs/run_<UTC>/run.log
```

```bash
tail -f runs/run_<UTC>/events.jsonl | jq -c 'select(.kind | IN("iter_start","score_computed","run_end","baseline_ready","branch_dead_end"))'
```

See `doc/runtime.md` for the full event catalog and the `RunContext` contract.

Phase B runs real CUDA-event benchmarking (`eval/benchmark.py`) end-to-end. `eval/profiler.py` provides analytical roofline metrics (required, fail-closed) plus a best-effort NCU subprocess for curated signals. Phase C populates `winner_per_workload_bottlenecks` whenever `workloads` + `hardware_spec` reach `generate_report`.

### Hardware-spec fallback in `optimize()`

`optimize()` substitutes a populated placeholder `HardwareSpec` whenever the resolved spec has zero peaks — both for the `config is None` path (where `detect_hardware()` may return zeros) and for caller-supplied configs whose peaks are zero. Without this, the orchestrator's fail-fast guard (`peak_flops_fp32 > 0`, `peak_memory_bandwidth_gb_s > 0`) would kill the run before the first iteration. Substitution uses `dataclasses.replace` so the caller's config object is not mutated.

Before substituting, `optimize()` calls `validate_hardware_spec(_PLACEHOLDER_HARDWARE_SPEC, config.hardware)` and logs `WARNING` per mismatch (see `doc/config.md` for the validator's field-by-field comparison rules). This catches the silent miscalibration where a real GPU (e.g. H100) gets the Ada-shaped placeholder substituted on top of it because its peaks were left at zero.

The stand-in `_PLACEHOLDER_HARDWARE_SPEC` mirrors `configs/arch/RTX6000Ada.yaml` so SOLAR (YAML-driven) and the placeholder fallback agree on bottleneck classification. Aligned values: `freq_GHz=2.505`, `MAC_per_cycle_fp32_sm=18_185.0`, `MAC_per_cycle_fp16_tc=72_695.0`, `MAC_per_cycle_bf16_tc=72_695.0`, `DRAM_byte_per_cycle=383.0`, `SRAM_byte_per_cycle=2200.0`, `SRAM_capacity=100_663_296` (96 MiB L2), `DRAM_capacity=51_539_607_552` (48 GiB GDDR6 ECC). The alignment was briefly drifted today (placeholder was unsynced when the YAML was edited from sparse → dense values) and is restored in this same patch. The spec's `name="placeholder-RTX6000Ada"` is also the SOLAR adapter's alias key in `_ACTS_ARCH_YAMLS`, so SOLAR loads `configs/arch/RTX6000Ada.yaml` for this name instead of silently falling back to `H100_PCIe`. The result: the analytical roofline (built from these fixture peaks) and the SOLAR roofline (built from the YAML) stay aligned on the placeholder path. Real runs should load a SOLAR arch YAML for their target GPU.

### Phase A → B threading

`optimize()` forwards `problem.definition_path` as `problem_definition_path` to `Orchestrator.run()`. The profiler's NCU subprocess driver re-loads the problem directory (`definition.json` + `workload.jsonl`) to rebuild the input generator — closures don't pickle across the subprocess boundary. On the placeholder path `problem` is `None` and the profiler falls back to `module.make_inputs` or `spec['args']`.

`_load_sol_problem()` (the SOL adapter dispatched by `_load_problem`) derives the roofline once per problem (not once per workload) and threads three signals into Phase B:

1. **Representative-workload selection** — picks `representative = workloads[len(workloads) // 2]` (median index by selection order) and feeds only that workload into `derive_t_sol_from_solar()`. Full per-workload re-derivation would re-run SOLAR's 4-stage pipeline N times for a number that's roughly invariant across shapes, so the static roofline is computed once on the median workload.
2. **Arch YAML forwarding to SOLAR** — passes `Path(config.arch_config_path)` as `arch_yaml_path` to `derive_t_sol_from_solar()` whenever the `.cfg` sets `[hardware] arch_config_path`; otherwise passes `None`. Without this explicit path the SOLAR adapter falls back to its name-based arch lookup, which silently resolves unknown hardware names to `H100_PCIe`.
3. **`roofline.source` threading** (`"solar"` | `"builtin"`) — propagated alongside `t_sol_us` so downstream consumers (orchestrator scoring, report rendering) can distinguish SOLAR-grounded scores from fallback-grounded ones rather than treating every roofline as equally trustworthy.
