# Pipeline — `src/pipeline/`

End-to-end optimization entry points.

## optimize.py — Main Entry Point

CLI (shrunk on 2026-05-11):

```
python -m src.pipeline.optimize [--config FILE] [--run-dir DIR] [--trace-dir DIR]
```

Everything algorithmic + invocation (problem path, GPU index, reset-clocks, beam width, …) lives in a `.cfg` file passed via `--config`. The CLI keeps only three flags — invocation-scoped surfaces that the cfg can't easily own (where artifacts land, which cfg to read).

- `--config FILE` (optional) — path to a libconfig-format ACTS `.cfg` (see `configs/example.cfg`), parsed by `libconf` in `load_config()`. When omitted, `ACTSConfig` dataclass defaults apply (problem=`placeholder`, gpu_index=0, reset_clocks=False) so the no-LLM smoke path stays runnable. The module-top preparse opens the cfg before any CUDA-aware import and reads `hardware.gpu_index` so `CUDA_VISIBLE_DEVICES` lands in time (via `_preparse_config_path` + `_preparse_gpu_index`). A non-existent path raises a clean argparse error in `main()`.
- `--run-dir DIR` (optional) — parent directory for per-invocation run artifacts. Defaults to `./runs`. Each invocation creates `<run-dir>/run_<YYYYMMDDTHHMMSS_ffffffZ>/`.
- `--trace-dir DIR` (optional) — directory for per-run JSONL trace files capturing every LLM input/output, tool call, and span via `src.agents.trace_processor.JSONLTraceProcessor`. Default is `None`: when omitted, SDK traces land under `<run-dir>/traces/`. Passing `--trace-dir <path>` relocates them. Passing `--trace-dir=` (empty string) is a kill switch — no capture.

### cfg-resident fields (formerly CLI)

- `runtime.problem_path` — SOL-ExecBench problem directory or literal `"placeholder"`. Default `"placeholder"`.
- `runtime.reset_clocks` (bool) — operator escape hatch. When `true`, `main()` resets the selected GPU's clocks (`-rgc -i 0` + `-rmc -i 0` on the logical index after `CUDA_VISIBLE_DEVICES` remapping) and exits without running any pipeline phase. Toggle on for one recovery run, then back off. Use after SIGKILL / segfault leaves clocks locked; the in-process atexit + signal handler chain (SIGTERM / SIGHUP) covers normal exits.
- `hardware.gpu_index` — which physical GPU ACTS pins. Read by the module-top preparse to set `CUDA_VISIBLE_DEVICES` before `import sol_execbench`, so SOL (and every downstream CUDA consumer) sees the selected GPU as logical index 0. The two-tier `_validate_gpu_visible` helper runs after preparse to confirm the index exists on the host and the remapped logical GPU is reachable; on the reset-clocks path it skips torch and uses `nvidia-smi --list-gpus -i N` instead.

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
5. **Baseline dispatch** (`_dispatch_baseline`, defined above `_load_problem`): the cfg flag `config.use_operator_baseline` chooses between two paths that both return a verified `Kernel` of identical shape — downstream code is unchanged. `triton_baseline_path` still names WHERE the operator-supplied Triton file lives; `use_operator_baseline` is the toggle that decides whether to read it. When the flag is False (default), `generate_triton_baseline()` (see `baseline_generator.py` below) drives `CoderAgent.translate()` to port the PyTorch reference into a Triton kernel, post-verifying on every selected workload. When the flag is True, `load_operator_baseline()` reads the operator-supplied Triton source from `config.triton_baseline_path` and runs it through the same loader gate sequence (compile → autotune-shape check → DPS routing → correctness over every selected workload) — `CoderAgent.translate()` is bypassed entirely, so SOL mode no longer requires a model when the operator pre-supplies the baseline. The flag/path pair is enforced asymmetrically at cfg load: `use_operator_baseline=True` with an empty `triton_baseline_path` raises, while `use_operator_baseline=False` with a stray path only warns (see `doc/config.md` for the exact rule and the rest of the `triton_baseline_*` cfg fields). The returned `Kernel` is the search-tree root. The imports of `generate_triton_baseline` / `load_operator_baseline` inside `_dispatch_baseline` are **function-local on purpose**: existing pipeline tests patch the canonical `src.benchmark.baseline_generator.{generate_triton_baseline,load_operator_baseline}` attributes, and the deferred import makes the dispatcher pick the patched callables up at call time instead of binding the originals at module import. See `doc/eval.md` for the loader gate sequence.
6. `build_reference_fn()` + `build_input_generator()` produce the oracle + one generator per workload; these are forwarded to `Orchestrator.run()` so Phase B's correctness tool binds to every workload the baseline was verified against.

**Model load** (`_load_model_if_configured`): reads `$ACTS_MODEL_CONFIG` or falls back to `configs/models/deepseek.json`; the JSON's `api_key` field is optional and falls back to `$OPENAI_API_KEY` then `$DEEPSEEK_API_KEY` (`ValueError` if none supply a key), so the key can live in the env rather than the committed JSON. Gated on SOL mode — placeholder mode intentionally runs with `model=None` so the CLI stays executable without credentials. Without an SDK install or without a model config on disk, returns `None` and every agent stays in no-op mode.

**Placeholder mode** (`_load_placeholder`): loads `make_matmul_kernel(1024, 1024, 1024)` directly; no oracle, no workloads, no roofline. Exercises the scaffold end-to-end only.

### Clock-lock lifecycle

GPU clocks are locked at run start so per-iter latency isn't poisoned by DVFS-driven frequency drift. The lock targets the single GPU ACTS pins via `[hardware] gpu_index` in the cfg (default 0); other GPUs on a multi-GPU host are untouched. Because the module-top preparse sets `CUDA_VISIBLE_DEVICES=N` before any CUDA-aware import, the selected GPU is always the logical index 0 from ACTS's perspective, which is what the `nvidia-smi ... -i 0` invocations below address.

- **Prerequisite** — `nvidia-persistenced.service` must be active on the host (`systemctl is-active nvidia-persistenced`). Persistence mode is what makes the GPU hold its `clocks.applications.*` at the lock target between the `-lgc`/`-lmc` issue and the verify read; without it, the GPU drops back to deep idle, idle DRAM falls to ~810 MHz (vs the 10001 MHz target on RTX 6000 Ada), and `_verify_gpu0_locked` rolls the partial lock back with `verify_failed`. See `configs/venvs/3.12.md` for the one-time enable.
- **Probe** — `probe_clock_lock_available()` (from SOL) checks whether passwordless `sudo nvidia-smi` is configured. The call site does defensive both-shape handling for the bare-`bool` return (today) vs a future tuple-`(bool, str)` return.
- **Preset lookup** — `_resolve_clock_preset(device_name)` consults the ACTS-side `_ACTS_CLOCK_PRESETS` table first (currently `RTX 6000 Ada → ClockPreset(2505, 10001)`) and falls back to SOL's `get_clock_preset` for known datacenter cards.
- **Lock** — `_lock_gpu0_clocks(gpu_mhz, dram_mhz)` runs `nvidia-smi -lgc <m>,<m> -i 0` and `-lmc <m>,<m> -i 0`. Targets the selected GPU only (logical index 0 after `CUDA_VISIBLE_DEVICES` remapping by the cfg-driven `gpu_index` preparse).
- **Verify** — `_verify_gpu0_locked(gpu_mhz, dram_mhz)` queries `clocks.applications.{graphics,memory}` directly via `nvidia-smi --query-gpu=... --format=csv,noheader,nounits -i 0` and compares against the targets with 50 MHz tolerance. The applications-clocks fields reflect the *target* set by `-lgc`/`-lmc` regardless of whether the GPU is busy or idle, so no torch wake-op is needed (a previous version woke the GPU with a tiny op to dodge an idle-clock read of 210 MHz from `clocks.current.*`; switching the queried field eliminated that workaround entirely).
- **Lifecycle** — lock attempted at run start (after `run_start` event); state stored in `_clock_lock_state` (locked + device_name); cleanup goes through three paths: (a) explicit `finally` in `main()`, (b) `atexit.register(_unlock_clocks_safe)`, (c) signal handlers `_signal_unlock_handler` for `SIGTERM` + `SIGHUP`. Operator escape hatch via `[runtime] reset_clocks = true` in the cfg covers the SIGKILL / segfault case.
- **Verify-failure semantics** — `_verify_gpu0_locked` returning False or raising triggers `_rollback_partial_lock(device, reason)`, which calls `_unlock_gpu0_clocks` and emits `clock_lock_unavailable` with `reason="verify_failed"` or `reason="verify_raised:<exc>"`. Lock state stays False.
- **Event surface** — `clock_lock_unavailable` reasons go through a `ClockLockReason` `StrEnum` (`OK` / `PROBE_RETURNED_FALSE` / `NO_PRESET` / `LOCK_FAILED` / `VERIFY_FAILED` / `UNKNOWN`); exception-derived reasons are free-form strings (`verify_raised:<exc>`).

### Phase B: Search Loop

Delegates to `Orchestrator.run()`. Runs up to `max_depth` iterations with 3 agents (Planner → Coder → Reviewer). The `reference_fn` + `input_generators` returned by Phase A are forwarded verbatim every iteration so the Coder's correctness tool remains bound to the problem's oracle.

`optimize(..., run_dir: Path | None = None)` accepts a new kwarg threaded by `main()` from `ctx.run_dir` so the orchestrator can locate per-iter worker directories for the bench-subprocess isolation path. `optimize()` derives `ncu_cache_dir = run_dir / "ncu_cache"` and passes both into `Orchestrator.run(..., run_dir=, ncu_cache_dir=)`; the orchestrator in turn forwards `run_dir` to `run_bench_subprocess`, which creates per-iter worker dirs under `<run_dir>/iter_<n>/worker/`. When `run_dir` is None (test paths without a `RunContext`, or the `RunContext` `OSError` fallback), the worker dir falls back to a tempdir hoisted outside the iter loop and registered for `atexit` cleanup. The orchestrator logs a `WARNING` when `bench_use_subprocess=True` but `run_dir is None` — a silent fallback to the in-process bench path would disable CUDA-context isolation without surfacing the regression.

### Phase C: Report

`generate_report(result, ...)` walks the root-to-best path on `result.tree` to build the `technique_trace`, carries the audit flags (`reward_hack_suspect`, `calibration_warning`) off the best node's `ScoreResult`, and unwraps `termination_reason` to a plain string. `render_report` formats the report for the CLI and surfaces audit flags as explicit `[AUDIT]` lines so a flagged run can't be skimmed past.

`optimize()` returns `(SearchResult, OptimizationReport)` — a 2-tuple. The report is built inside `optimize()` (not in `main()`) so the rich Phase A locals (`definition`, `workloads`, `hardware_spec`, `arch_yaml_path`, `blob_roots`) reach `generate_report` directly. `main()` unpacks the tuple and renders.

SOL `Trace` payloads are emitted per evaluation (`trace_emitted` event, built by `Orchestrator._emit_trace`) so per-evaluation environment + correctness + performance records land in `events.jsonl` alongside the score line.

## baseline_generator.py — Triton Baseline Generation

`generate_triton_baseline(definition, spec, *, coder, workloads, max_retries=3, cache_dir=None, policy=None, blob_roots=None) -> Kernel`

Runs at problem-load time. Drives `CoderAgent.translate()` to port the PyTorch reference into Triton, then post-verifies: recompiles the returned source and reruns the 5-stage correctness gate against every workload in *workloads*. The post-verify catches SDK best-effort output when the Coder's turn budget was exhausted. Returns the first candidate that compiles and passes correctness on all workloads.

**Cross-attempt retry memory** — a `list[AttemptFailure]` accumulator (`prior_failures`) grows across the retry loop and is threaded into the next `coder.translate(prior_failures=...)` call, so the LLM sees failures from earlier attempts across `Runner.run` boundaries. Within one `Runner.run` the SDK's typed-item list already feeds tool errors back; across attempts it didn't. Three append sites cover the failure modes: (1) on `ImplementationError` catch, `AttemptFailure(attempt_no, tool_errors=list(exc.tool_errors))`; (2) on post-verify compile failure, a synthetic `AttemptFailure` with `tool_errors=[f"{_POST_VERIFY_COMPILE_FAILED}:\n{compile_error}"]`; (3) on post-verify correctness failure, a synthetic `AttemptFailure` carrying the first failing workload's error as `tool_errors=[f"{_POST_VERIFY_CORRECTNESS_FAILED} on workload N/M:\n{result.error_message}"]`. The Coder's `build_translate_prompt` prepends a "## Prior attempt failures" section listing each `AttemptFailure` under `### Attempt N` headers. To capture the first failing workload's error, the correctness verification loop was restructured from `all(verify_correctness(...).passed for ...)` into an explicit `for` loop with `break` on the first failure — identical semantics (same short-circuit call count on failure, identical on the all-pass path). Spec: `doc/specs/2026-05-13-cross-attempt-memory-design.md` (uncommitted). The in-prompt accumulator currently has no per-error / total size cap; tracked as trigger-gated tech debt in PROCESS.md.

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

The returned `OptimizationReport` also carries `hardware_spec: HardwareSpec | None` (defaults to `None`); `optimize()` populates it from the resolved/substituted spec so the report ships its calibration context end-to-end.

Reads the best node's `ScoreResult` and walks `result.tree.path_to_node(best.id)` to build the root-to-best action sequence. The root's `action_applied` is the empty-string baseline placeholder and is filtered out of the trace. When `best.score is None` (scoring failed), the returned report surfaces only `termination_reason` + `total_iterations` without crashing.

When `workloads` + `hardware_spec` are supplied, `generate_report` iterates the selected workloads once. Per-workload bottlenecks are sourced from SOLAR via `derive_t_sol_from_solar(definition, workload, hardware_spec, arch_yaml_path=...).bottleneck` (SOLAR is authoritative). Workloads where SOLAR returns `None` or where `definition is None` are **omitted** from `winner_per_workload_bottlenecks` rather than falling back to the analytical band classifier (`classify_bottleneck` in `eval/roofline.py` is the analytical band classifier; it is no longer used for per-workload labels). When `input_generators` is also supplied, the same loop re-profiles the winning kernel on each workload into `winner_profiling_per_workload`. **Loop order**: SOLAR runs first per workload, then `_resolve_workload_roofline(definition, w, best.kernel, roofline=solar)` derives `(flops, nbytes)` — SOLAR's `total_flops` / `total_fused_bytes` outrank the shape-formula fallback (same precedence as the orchestrator), and the shape formulas only fire when SOLAR is unavailable or returns `None`. `_resolve_workload_roofline` returns `(0, nbytes)` when only flops can't be derived, and `(0, 0)` only when shape resolution also fails; **there is no `(flops, nbytes)` skip gate** — `profile_kernel` is always called for any selected workload with a valid per-workload latency (`profile_kernel` handles `nbytes=0` internally by emitting a degraded result with `analytical=None`). `blob_roots` is forwarded into `profile_kernel` so the NCU subprocess driver resolves safetensors-backed inputs against the same root list as the in-process generator (defaults to `[definition_path.parent]` when unspecified).

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
| `hardware_spec` | `HardwareSpec \| None` | Resolved hardware spec used for this run (post-placeholder-substitution). `None` when the spec couldn't be derived. Surfaced verbatim by `_render_hardware_spec_block` at the tail of `render_report`. |
| `usage_stats` | `UsageSnapshot \| None` | Per-iter × per-agent LLM call counts + token usage captured via `RunContext`'s usage accumulator. Defaults to `None`; populated by `pipeline/optimize.py::main()` via direct field-assign (`report.usage_stats = ctx.usage_snapshot()`) after `optimize()` returns — `generate_report` does **not** accept a `usage_snapshot` kwarg. `None` and empty-snapshot render identically as the `(no LLM usage captured)` fallback line in `_render_usage_block`. |

`render_report(report: OptimizationReport) -> str`

Multi-line CLI summary. Skips the scoring block when `baseline_latency_us == 0` so a degenerate run (no scored best node) doesn't print misleading "0.00us / 0.00x" lines. Emits `Bottleneck (run): <label>` when `report.bottleneck` is set, and `Bottleneck (per workload): uuid=label, ...` when the per-workload dict is non-empty (enum values are rendered via `.value` at the string boundary). When `reward_hack_suspect` / `calibration_warning` are set, emits an `[AUDIT]` line per flag so operators scanning the output can't miss a physics-violating or poorly-calibrated result.

Between the `[AUDIT]` lines and the `Hardware spec` block, `_render_usage_block(snapshot)` (in `src/pipeline/report.py`) emits a `Resource usage (LLM)` table when `report.usage_stats` is populated: header row `Iter | <agent columns...> | total`, then per-iter rows where each cell is formatted `<calls> (<turns>) / <input>→<output>` with em-dash for empty cells (rows where every agent cell is empty are skipped). Token counts go through `_fmt_tokens` (k/M abbreviation: <1000 exact, <1M one-decimal k, ≥1M one-decimal M). A run-total row closes the table; `of which cached input: X (Y%)` and `of which reasoning output: X (Y%)` lines follow only when those counters are non-zero so non-thinking-model runs aren't cluttered. When `report.usage_stats is None` or the snapshot is empty, the block degrades to a single line: `Resource usage (LLM): (no LLM usage captured)`.

When `winner_profiling_per_workload` is populated, a "Winner profile (per workload)" block follows, with one analytical line per workload plus optional NCU lines. `_render_profiling_block` guards on `ProfilingResult.has_analytical` (property on the dataclass): when `analytical is None` (no byte count was derivable), the achieved-throughput line is replaced with `[analytical unavailable — no byte count]`; NCU rows still render normally when present. If every per-workload profile is degraded with `ncu_binary_not_found` (common on machines without the NCU CLI), the NCU block is suppressed to keep the output tidy.

When `report.hardware_spec is not None`, `render_report` appends a trailing **"Hardware spec"** block emitted by `_render_hardware_spec_block` (in `src/pipeline/report.py`). The block always emits **every** field — including zero-valued ones — so a degraded/partial detection (e.g. a real GPU with peaks left at zero) is visible in the report rather than papered over by selective field rendering. This is the calibration-visibility contract: the operator should be able to see exactly which hardware peaks fed the run's scoring without cross-referencing config files.

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

Every CLI invocation creates a fresh `<run-dir>/run_<YYYYMMDDTHHMMSS_ffffffZ>/` directory (default `./runs/run_<UTC>/`) holding:

- `run.log` — human-readable text log of the invocation.
- `events.jsonl` — structured event stream (kinds enumerated in `runtime/events.CORE_EVENT_KINDS`) emitted by the orchestrator and `RunContext`.
- `traces/acts_trace_<UTC>.jsonl` — SDK per-call records (LLM inputs/outputs, tool calls, spans) written by `JSONLTraceProcessor`. Relocated when `--trace-dir <path>` is passed; absent when `--trace-dir=` disables capture.
- `report.txt` — final `render_report(report)` text persisted alongside the stdout print, plus an appended `=== ACTSConfig (resolved at run start) ===` block (JSON dump of the dataclass `main()` constructed for this invocation). Best-effort write; an `OSError` is logged at `WARNING` and skipped. The terminal print stays focused on results — only the persisted file carries the config dump.
- `usage.json` — machine-readable sidecar with per-iter × per-agent LLM usage. Schema: `{schema_version: 1, columns: [...], by_iter: [...], by_agent: {...}, total: {...}}`, serialized via `dataclasses.asdict` over the snapshot's buckets (no custom dict-builder). Persisted by `pipeline/optimize.py::_write_usage_sidecar(snapshot, run_dir)`, built from `ctx.usage_snapshot()` (see `doc/runtime.md` for the accumulator's design). Best-effort write; an `OSError` is logged at `WARNING` and skipped.
- `tree/` — per-node + tree-level dump (see `doc/runtime.md` "Tree dump" section).

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
