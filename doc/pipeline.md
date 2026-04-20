# Pipeline — `src/pipeline/`

End-to-end optimization entry points.

## optimize.py — Main Entry Point

`python -m src.pipeline.optimize`

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

`generate_report(result: SearchResult) -> OptimizationReport`

Reads the best node's `ScoreResult` and walks `result.tree.path_to_node(best.id)` to build the root-to-best action sequence. The root's `action_applied` is the empty-string baseline placeholder and is filtered out of the trace. When `best.score is None` (scoring failed), the returned report surfaces only `termination_reason` + `total_iterations` without crashing.

| Field | Description |
|-------|-------------|
| `baseline_latency_us` | Starting latency |
| `best_latency_us` | Best achieved latency |
| `sol_score` | Final SOL score |
| `speedup` | Baseline / best |
| `technique_trace` | Root-to-best action sequence (root baseline filtered out) |
| `bottleneck_transitions` | Per-iteration bottleneck shift. Stays empty until `eval/profiler.py` lands (GPU-blocked). |
| `remaining_headroom_pct` | Distance to hardware limit, `(1 - sol_score) * 100` |
| `total_iterations` | Search iterations run |
| `termination_reason` | Why search stopped (plain string, unwrapped from `TerminationReason` enum) |
| `reward_hack_suspect` | Propagated from best node's `ScoreResult` — candidate beats `T_SOL` |
| `calibration_warning` | Propagated from best node's `ScoreResult` — baseline already at/below `T_SOL` |

`render_report(report: OptimizationReport) -> str`

Multi-line CLI summary. Skips the scoring block when `baseline_latency_us == 0` so a degenerate run (no scored best node) doesn't print misleading "0.00us / 0.00x" lines. When `reward_hack_suspect` / `calibration_warning` are set, emits an `[AUDIT]` line per flag so operators scanning the output can't miss a physics-violating or poorly-calibrated result.

## Running the Pipeline

**Placeholder mode** — the default CLI (`python -m src.pipeline.optimize`) runs the matmul starter without GPU, LLM, or SOL-ExecBench. `main()` runs `optimize("placeholder")` and prints `render_report(generate_report(result))`. No model is loaded — every agent stays in no-op mode, the baseline comes from `make_matmul_kernel`, and with no workloads `benchmark_kernel` returns its 100us sentinel so the report emits a scoring block with baseline == best (speedup 1.00x). This only exercises the scaffold end-to-end; it is not a meaningful search result.

**SOL mode** — call `optimize(problem_path=<sol-dir>)` with a SOL-ExecBench problem directory. Requires `configs/models/<provider>.json` (or `$ACTS_MODEL_CONFIG` pointing at one) and the `openai-agents` SDK installed; `generate_triton_baseline` fails closed otherwise with `BaselineGenerationError`.

Phase B runs real CUDA-event benchmarking (`eval/benchmark.py`) end-to-end; only `eval/profiler.py` remains a placeholder, so `bottleneck_transitions` stays empty in Phase C reports until it lands.
