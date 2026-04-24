# Pipeline — `src/pipeline/`

End-to-end optimization entry points.

## optimize.py — Main Entry Point

CLI:

```
python -m src.pipeline.optimize [problem_path] [--run-dir DIR] [--trace-dir DIR]
```

- `problem_path` (positional, optional) — SOL-ExecBench problem directory (contains `definition.json` + `workload.jsonl`), or the literal string `placeholder` for the built-in matmul demo. Default `"placeholder"` preserves the no-LLM smoke path.
- `--run-dir DIR` (optional) — parent directory for per-invocation run artifacts. Defaults to `./runs`. Each invocation creates `<run-dir>/run_<YYYYMMDDTHHMMSS_ffffffZ>/` (see "Run artifacts" below).
- `--trace-dir DIR` (optional) — directory for per-run JSONL trace files capturing every LLM input/output, tool call, and span via `src.agents.trace_processor.JSONLTraceProcessor`. Default is `None`: when omitted, SDK traces land under `<run-dir>/traces/` inside the per-invocation run directory. Passing `--trace-dir <path>` relocates the traces to `<path>`. Passing `--trace-dir=` (empty string) is a kill switch — no capture.

### Phase A: Load Problem

`optimize()` takes a `problem_path` that is either a SOL-ExecBench problem directory (contains `definition.json` + `workload.jsonl`) or the literal string `"placeholder"` for the built-in matmul demo. SOL mode is the real path; placeholder mode keeps the CLI runnable without an LLM or SOL dependency.

**SOL mode** (`_load_sol_execbench`):

1. `load_problem()` parses the SOL definition + workloads.
2. `problem_to_kernel_spec()` derives the `KernelSpec` (name, entrypoint, kernel_type).
3. `derive_t_sol_from_solar()` produces the roofline result; `spec.t_sol_us` is populated when SOLAR returns data.
4. `select_workloads()` samples `config.benchmark_workload_count` representative workloads.
5. **`generate_triton_baseline()`** (see `baseline_generator.py` below) drives `CoderAgent.translate()` to port the PyTorch reference into a Triton kernel, post-verifying on every selected workload. The returned `Kernel` is the search-tree root.
6. `build_reference_fn()` + `build_input_generator()` produce the oracle + one generator per workload; these are forwarded to `Orchestrator.run()` so Phase B's correctness tool binds to every workload the baseline was verified against.

**Model load** (`_load_model_if_configured`): reads `$ACTS_MODEL_CONFIG` or falls back to `configs/models/deepseek.json`. Gated on SOL mode — placeholder mode intentionally runs with `model=None` so the CLI stays executable without credentials. Without an SDK install or without a model config on disk, returns `None` and every agent stays in no-op mode.

**Placeholder mode** (`_load_placeholder`): loads `make_matmul_kernel(1024, 1024, 1024)` directly; no oracle, no workloads, no roofline. Exercises the scaffold end-to-end only.

### Phase B: Search Loop

Delegates to `Orchestrator.run()`. Runs up to `max_depth` iterations with 3 agents (Planner → Coder → Reviewer). The `reference_fn` + `input_generators` returned by Phase A are forwarded verbatim every iteration so the Coder's correctness tool remains bound to the problem's oracle.

### Phase C: Report

`generate_report(result)` walks the root-to-best path on `result.tree` to build the `technique_trace`, carries the audit flags (`reward_hack_suspect`, `calibration_warning`) off the best node's `ScoreResult`, and unwraps `termination_reason` to a plain string. `render_report` formats the report for the CLI and surfaces audit flags as explicit `[AUDIT]` lines so a flagged run can't be skimmed past.

## baseline_generator.py — Triton Baseline Generation

`generate_triton_baseline(problem, spec, *, coder, workloads, max_retries, cache_dir=None, policy=None) -> Kernel`

Runs at problem-load time. Drives `CoderAgent.translate()` to port the PyTorch reference into Triton, then post-verifies: recompiles the returned source and reruns the 5-stage correctness gate against every workload in *workloads*. The post-verify catches SDK best-effort output when the Coder's turn budget was exhausted. Returns the first candidate that compiles and passes correctness on all workloads.

**Fail-closed contract** — raises `BaselineGenerationError` when:
- `coder is None` or `coder.has_model is False` (no model configured). Search against a fake baseline would silently look like progress, so there is intentionally no stub fallback.
- `max_retries` attempts are exhausted without a verified candidate.

`ValueError` is raised for a caller bug — an empty `workloads` list.

## verify.py — Post-Optimization Verification

Re-runs the correctness gate on the best kernel to confirm results are reproducible. Recompiles the winner, then delegates to `verify_correctness` against the PyTorch reference. Compile failures surface as `passed=False` with a compile-phrased detail string.

`verify_optimized_kernel(optimized, *, reference_fn, input_generator, policy=None, cache_dir=None) -> VerificationResult`

## report.py — Report Generation

`generate_report(result, *, workloads=None, input_generators=None, hardware_spec=None, cache_dir=None, problem=None) -> OptimizationReport`

Reads the best node's `ScoreResult` and walks `result.tree.path_to_node(best.id)` to build the root-to-best action sequence. The root's `action_applied` is the empty-string baseline placeholder and is filtered out of the trace. When `best.score is None` (scoring failed), the returned report surfaces only `termination_reason` + `total_iterations` without crashing.

When `workloads` + `hardware_spec` are supplied, `generate_report` iterates the selected workloads once: it calls `classify_bottleneck` to populate `winner_per_workload_bottlenecks`, and (when `input_generators` is also supplied) re-profiles the winning kernel on each workload into `winner_profiling_per_workload`. The two passes are fused so `(flops, nbytes)` are computed once per workload and shared between classification and the re-profile call.

| Field | Type | Description |
|-------|------|-------------|
| `baseline_latency_us` | float | Starting latency |
| `best_latency_us` | float | Best achieved latency |
| `sol_score` | float | Final SOL score |
| `speedup` | float | Baseline / best |
| `technique_trace` | `list[str]` | Root-to-best action sequence (root baseline filtered out) |
| `bottleneck` | `BottleneckType \| None` | Once-per-run classification, copied verbatim from `SearchResult.run_bottleneck` (produced by `classify_run` in `eval/roofline.py`). `None` on the placeholder path that has no roofline. |
| `winner_per_workload_bottlenecks` | `dict[str, BottleneckType]` | Per-workload shape-derived classification (via `classify_workload`) for every selected workload, keyed by `Workload.uuid`. Populated only when `workloads` + `hardware_spec` are provided. Replaces the removed `bottleneck_transitions` (classification is invariant within a run — the per-workload view is the only non-trivial axis left for diagnostics). |
| `winner_profiling_per_workload` | `dict[str, ProfilingResult]` | Phase C re-profile of the winning kernel on every selected workload (spec §3.4). Empty when `input_generators` is missing. |
| `remaining_headroom_pct` | float | Distance to hardware limit, `(1 - sol_score) * 100` |
| `total_iterations` | int | Search iterations run |
| `termination_reason` | str | Why search stopped (plain string, unwrapped from `TerminationReason` enum) |
| `reward_hack_suspect` | bool | Propagated from best node's `ScoreResult` — candidate beats `T_SOL` |
| `calibration_warning` | bool | Propagated from best node's `ScoreResult` — baseline already at/below `T_SOL` |

`render_report(report: OptimizationReport) -> str`

Multi-line CLI summary. Skips the scoring block when `baseline_latency_us == 0` so a degenerate run (no scored best node) doesn't print misleading "0.00us / 0.00x" lines. Emits `Bottleneck (run): <label>` when `report.bottleneck` is set, and `Bottleneck (per workload): uuid=label, ...` when the per-workload dict is non-empty (enum values are rendered via `.value` at the string boundary). When `reward_hack_suspect` / `calibration_warning` are set, emits an `[AUDIT]` line per flag so operators scanning the output can't miss a physics-violating or poorly-calibrated result.

When `winner_profiling_per_workload` is populated, a "Winner profile (per workload)" block follows, with one analytical line per workload plus optional NCU lines. If every per-workload profile is degraded with `ncu_binary_not_found` (common on machines without the NCU CLI), the NCU block is suppressed to keep the output tidy.

## Running the Pipeline

**Placeholder mode** — the default CLI (`python -m src.pipeline.optimize`, no positional arg) runs the matmul starter without GPU, LLM, or SOL-ExecBench. `main()` wraps its body in a `RunContext` (from `src/runtime/run_context.py`) that owns run-dir creation, logging config, and trace-processor wiring (replaced the removed `_enable_traces_if_possible` helper). It resolves `args.problem_path == "placeholder"`, runs `optimize("placeholder")`, and prints `render_report(generate_report(result))`. No model is loaded — every agent stays in no-op mode, the baseline comes from `make_matmul_kernel`, and with no workloads `benchmark_kernel` returns its 100us sentinel so the report emits a scoring block with baseline == best (speedup 1.00x). This only exercises the scaffold end-to-end; it is not a meaningful search result.

**SOL mode** — pass a SOL-ExecBench problem directory as the positional argument: `python -m src.pipeline.optimize /abs/path/to/sol/problem/` (or from a Python caller: `optimize(problem_path=<sol-dir>)`). Requires `configs/models/<provider>.json` (or `$ACTS_MODEL_CONFIG` pointing at one) and the `openai-agents` SDK installed; `generate_triton_baseline` fails closed otherwise with `BaselineGenerationError`.

### Run artifacts

Every CLI invocation creates a fresh `<run-dir>/run_<YYYYMMDDTHHMMSS_ffffffZ>/` directory (default `./runs/run_<UTC>/`) holding three files:

- `run.log` — human-readable text log of the invocation.
- `events.jsonl` — structured event stream (18 kinds) emitted by the orchestrator and `RunContext`.
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

The stand-in `_PLACEHOLDER_HARDWARE_SPEC` mirrors the Tier 1/2 test-fixture values for the RTX 6000 Ada (`freq_GHz=2.5`, `DRAM_byte_per_cycle=384`, `MAC_per_cycle_fp32_sm=12_800`, `MAC_per_cycle_fp16_tc=MAC_per_cycle_bf16_tc=512_000`) so the placeholder run produces representative roofline math. Real runs should load a SOLAR arch YAML for their target GPU.

### Phase A → B threading

`optimize()` forwards `problem.definition_path` as `problem_definition_path` to `Orchestrator.run()`. The profiler's NCU subprocess driver re-loads the problem directory (`definition.json` + `workload.jsonl`) to rebuild the input generator — closures don't pickle across the subprocess boundary. On the placeholder path `problem` is `None` and the profiler falls back to `module.make_inputs` or `spec['args']`.
