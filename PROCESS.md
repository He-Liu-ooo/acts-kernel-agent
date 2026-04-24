# ACTS — Implementation Status

## Completed

- [x] Reference repo analysis (AccelOpt, Astra, AutoKernel, EvoToolkit)
- [x] 9-paper knowledge base analysis
- [x] Architecture design (search strategy, agent architecture, action library, eval harness, optimization memory, backend choice, hardware handling)
- [x] Directory structure design
- [x] Pipeline flow design
- [x] Project scaffolding (pyproject.toml, src/ skeleton with placeholder modules, pipeline runs end-to-end)

### Implemented during scaffolding (real logic, not placeholders)

- [x] config.py — HardwareSpec (SOLAR arch YAML schema), load_hardware_spec(), load_config(), ACTSConfig
- [x] kernels/kernel.py — Kernel, KernelSpec, KernelType dataclasses
- [x] eval/scorer.py — SOL Score formula + reward_hack_suspect / calibration_warning audit flags
- [x] eval/roofline.py — compute_roofline() (built-in fallback) + derive_t_sol_from_solar() wrapper
- [x] benchmark/problem_loader.py — load_problem(), load_definition(), load_workloads(), problem_to_kernel_spec(), op_type mapping
- [x] benchmark/workload_selector.py — select_workloads() (evenly-spaced sampling by problem size)
- [x] benchmark/solution_formatter.py — format_solution() (SOL-ExecBench solution JSON)
- [x] actions/registry.py — Action dataclass, ActionTier enum, build_default_registry()
- [x] actions/tier1-6 — action definitions (guidance text is placeholder, but structure/metadata is real)
- [x] memory/experience.py — Experience dataclass
- [x] memory/store.py — MemoryStore with save/load/add_experience/query (real JSON persistence)

### Implemented during search (real logic, not placeholders)

- [x] search/tree.py — path_to_node, checkpoint save/load (atomic writes)
- [x] search/beam.py — diversity-aware beam pruning (B2), branch-quality-weighted pruning (B3), configurable diversity (`beam_diversity`)
- [x] search/orchestrator.py — `detect_plateau` wired into search loop, plateau termination
- [x] agents/reviewer.py — Pydantic `ReviewerFeedbackOutput`, `build_user_prompt()`, rule-based fallback with `degraded`/`error_reason` surfacing, configurable `prompt_dir` for future Compute/Memory sub-reviewer split
- [x] prompts/reviewer/ — system.md (diagnostic reasoning) + interpret.md
- [x] agents/llm_backend.py retry hardening — narrow transient catch, exponential backoff with ±25% jitter, named-logger observability
- [x] /simplify sweep across all prior commits — whole-repo review for reuse/quality/efficiency; surgical fixes applied, remaining tech-debt recorded in "Deferred Improvements"

### Implemented during Coder phase (real logic, not placeholders)

- [x] agents/coder.py — tool-using Agent with Pydantic `KernelCodeOutput`, `build_user_prompt()`, `ImplementationError`, turn budget `2*max_debug_retries+2` (= 8 by default — the +2 over 2N reserves the `submit_kernel` tool call and its final plain-text confirmation; see live-GPU-run pre-flight T4 + option-α entries below), temperature 0.0 for determinism. Tools wire to real `compile_kernel` / `verify_correctness` via closure-captured `KernelSpec` + `reference_fn` + **`input_generators` (list, one per selected workload — correctness tool iterates all, short-circuits on first failure)** at call time. Second entry point `translate()` (one-shot PyTorch→Triton port for baseline generation) shares tool wiring with `implement()` via private `_run_tool_agent` helper; `has_model` property for callers that must branch before reaching into internals.
- [x] prompts/coder/ — system.md (prescribed compile-then-correctness workflow, hard rules, anti-patterns, one sanctioned failure mode) + implement.md (user-prompt format doc) + translate.md (baseline-port system prompt: port PyTorch `run` to Triton `kernel_fn`, signature invariance, no precision drop)
- [x] agents/llm_backend.py — added optional `max_turns` kwarg to `run_agent()` (threads SDK tool-loop bound) and `render_kernel_section()` helper (replaces triple-duplicated fence+escape logic in coder/planner/reviewer)
- [x] Planner/Reviewer temperature bumped 0.0 → 0.3 — Coder stays at 0.0 (determinism for code gen), upstream agents get variance for technique exploration / diagnosis wording; strict Pydantic enums still pin schema

### Implemented during baseline-generator phase (real logic, not placeholders)

- [x] benchmark/baseline_generator.py — drives `CoderAgent.translate()`, recompiles the returned source, and reruns the 5-stage correctness gate against every selected workload before accepting a candidate. Post-verify catches SDK best-effort output when the Coder's turn budget is exhausted. Fail-closed: raises typed `BaselineGenerationError` on no-model-configured or retry exhaustion (no stub fallback — search against a fake baseline would look like progress).
- [x] pipeline/optimize.py Phase A — `_load_model_if_configured` reads `$ACTS_MODEL_CONFIG` / `configs/models/deepseek.json` (TOCTOU-safe via try/except), model load gated on SOL mode so placeholder CLI stays runnable. `_load_sol_execbench` now async: calls `generate_triton_baseline` and returns `reference_fn` + the full `input_generators` list so Phase B's correctness tool binds to every selected workload.
- [x] search/orchestrator.py — accepts plural `input_generators` and forwards verbatim to `CoderAgent.implement()` every iteration.

### Implemented during report phase (real logic, not placeholders)

- [x] pipeline/report.py — `generate_report(result)` walks `result.tree.path_to_node(best.id)` to build `technique_trace` (root baseline placeholder filtered out), propagates `reward_hack_suspect` / `calibration_warning` from the best node's `ScoreResult`, unwraps `TerminationReason` to a plain string, and defensively handles a `None` score. `render_report` emits a multi-line CLI summary that skips the scoring block when `baseline_latency_us == 0` and surfaces audit flags as explicit `[AUDIT]` lines. Post-refactor (2026-04-22): `report.bottleneck: BottleneckType | None` comes from `SearchResult.run_bottleneck`; `winner_per_workload_bottlenecks: dict[str, BottleneckType]` is populated by a fused pass over selected workloads when `workloads` + `hardware_spec` are supplied, sharing `(flops, nbytes)` with the Phase C re-profile loop.
- [x] search/orchestrator.py — `SearchResult` gained a `tree: SearchTree` field so Phase C can reconstruct path-derived views without the orchestrator denormalizing upfront; all four `SearchResult` construction sites updated (ALL_DEAD_END / SOL_TARGET / PLATEAU / BUDGET). Lighter-snapshot alternative tracked as a Deferred Improvement.
- [x] pipeline/optimize.py — `main()` now prints `render_report(generate_report(result))`.

### Implemented during benchmark phase (real logic, not placeholders)

- [x] eval/benchmark.py — CUDA-event timing via injectable `BenchmarkTimer` Protocol. Production `_TorchCudaTimer` uses `torch.cuda.Event` pairs + 256MB int64 L2 thrasher; tests inject a scripted `RecordingTimer` so dispatch / aggregation / call-order are verifiable without torch. Multi-workload path takes parallel `workloads` / `input_generators` lists, constructs a fresh timer per workload (CUDA sticky-error isolation), aggregates median-of-medians across workloads and preserves `per_workload_latency_us`. Fail-closed: per-workload launch failures record `inf` + reason; <half survive raises `BenchmarkError`; `BenchmarkResult.is_fully_successful` is the orchestrator's partial-failure check. Empty-workload path returns a 100us sentinel so `compute_sol_score` can't silently collapse to 1.0.
- [x] search/orchestrator.py — baseline partial-workload failure raises `BenchmarkError` (SOL-score denominator must be complete); child partial failure marks branch `DEAD_END` (branch-local); both paths deduplicated through a single `dead_reason` sentinel. Uses `BenchmarkResult.is_fully_successful` instead of reaching into `workload_errors`.

### Implemented during live-GPU-run pre-flight (real logic, not placeholders)

- [x] pipeline/optimize.py — `main()` accepts a positional `problem_path` arg via argparse; default `"placeholder"` preserves the existing CLI smoke-path. (T2)
- [x] agents/coder.py + kernels/kernel.py + eval/profiler.py — `KernelCodeOutput` gains a `triton_kernel_name` field with a `@model_validator(mode="after")` that pulls every `@triton.jit def <name>` out of `source_code` and asserts the declared name matches one of them. `Kernel` carries `triton_kernel_name: str = ""` (defaults preserve back-compat for hand-written starters / pre-T4 checkpoints). `CoderAgent.implement` and `.translate` return `KernelCodeOutput` (orchestrator + baseline_generator thread both fields into the new `Kernel`). `profile_kernel` resolves NCU's `--kernel-name regex:` filter via the priority chain `kernel.triton_kernel_name → _extract_triton_kernel_name(source) → kernel.spec.entrypoint`, removing the silent mis-profile failure mode that source-regex extraction had on fused outputs with multiple `@triton.jit` defs. Coder system + translate prompts gain a Hard Rule documenting the schema. See JOURNAL → "Coder declares `triton_kernel_name` explicitly (T4, 2026-04-22)". (T4)

### Implemented during logger-system phase (real logic, not placeholders)

- [x] src/runtime/{__init__,timefmt,events,run_context}.py — new package. `RunContext.create(root, *, trace_dir=None, capture_traces=True)` owns per-invocation lifecycle: creates `./runs/run_<UTC>/` with microsecond-precision filename (collision-safe for parallel invocations), configures stdlib `FileHandler` + `StreamHandler` on the root logger, silences `httpx`/`openai`/`agents` to WARNING, binds `events.jsonl`, and wires the SDK trace processor under `<run-dir>/traces/` (or an explicit `--trace-dir` override). `close()` is idempotent; `atexit` + `finally` wired so crashed runs still flush. `_NullRunContext` fallback on any OSError during setup keeps `emit()` in logger-only mode. `events.emit(kind, *, iter, **fields)` fans out to both sinks with `never-raise` discipline; 18 `CORE_EVENT_KINDS` in the catalog. `finite_or_none()` sanitizes `inf`/`nan` benchmark sentinels so `events.jsonl` stays RFC-8259 valid. See `doc/runtime.md` for the full reference + JOURNAL → "Logger system before first live GPU run (2026-04-23)" for design rationale.
- [x] pipeline/optimize.py — added `--run-dir` CLI flag (default `./runs`); changed `--trace-dir` default from `"traces"` to `None` (routes through RunContext); deleted `_enable_traces_if_possible` helper (replaced by `RunContext._wire_trace_capture`). `main()` wraps body in RunContext + `atexit.register(ctx.close)`; emits `run_start` at entry and `run_end` at exit (including `termination_reason="ERROR"` on the exception path). New `_is_model_configured()` helper populates `run_start.model_configured` before `optimize()` loads the model.
- [x] search/orchestrator.py — emits 10 iteration-level events + `baseline_ready` per run. Coder outcome events are `coder_submitted` (no pass claim) and `coder_failed(reason)` — the orchestrator can't verify gates from `implement()`'s return value alone, so ground-truth per-tool-call records live in `traces/*.jsonl`. `iter_end.outcome` is `advanced`/`dead_end`/`skipped` (skipped only on Coder failure, no tree mutation). `running_best_score` tracked as a local float (O(N) total) instead of per-iter `tree.best_node()` (was O(N²)). `_per_workload_us` + `_emit_dead_end` helpers dedup the per-workload list-comp and the two-emit pair across 4 dead-end sites.
- [x] benchmark/baseline_generator.py — emits `baseline_attempt` / `baseline_success` / `baseline_failure` in the retry loop.
- [x] pipeline/verify.py — emits `verify_start` / `verify_done` on both return paths (compile-failure + post-compile).
- [x] agents/trace_processor.py — dropped local `_isoformat_utc`, now imports `filename_ts` from `src/runtime/timefmt.py`; `datetime.utcnow()` deprecation fixed.
- [x] .gitignore — `/runs/` added alongside `/traces/`.

## Next Up

Phase A, Phase B, Phase C are all wired end-to-end on real CUDA-event benchmarking + real analytical profiling. GPU is available (NVIDIA RTX 6000 Ada, CUDA 12.8). For what last shipped, see `git log c912e9a..HEAD` + JOURNAL → "Bottleneck classify-once (2026-04-22)".

Candidates (pick one before writing code; design-discussion first where called out):

- **First live GPU run** (integration milestone, not a new module) — highest-value milestone now that every in-loop module is real. Run Phase A → B → C end-to-end against a SOL-ExecBench problem. Observability is now in place (logger phase): `./runs/run_<UTC>/run.log` is tail-able during the run, `events.jsonl` captures 18-kind structured narrative, `traces/acts_trace_<UTC>.jsonl` captures the full SDK per-LLM-call payloads. Likely triggers deferred work — the per-dtype ridge fix (if a tc workload shows up), the `do_bench`-shape timer rewrite (if sync-per-iter cost becomes visible), `detect_hardware()` (if the SOLAR arch YAML path proves inconvenient in practice).

- **`actions/tier{1..6}` real guidance text** (non-GPU) — action-library descriptions are the Planner's fuel. Structure is done (`src/actions/tier*_*.py` ~373 LOC total); the guidance strings are placeholders. Content-heavy (literature synthesis from 9-paper KB + AccelOpt / Astra), high impact on search quality but does not require GPU. Good parallel track to the live-GPU run — disjoint files.

- **`config.py::detect_hardware()`** (non-GPU code, uses GPU at runtime) — currently a placeholder returning a zeroed spec. Wire `torch.cuda` / `pynvml` to populate `HardwareSpec` fields at startup. `pipeline/optimize.py` already has a placeholder-spec fallback that would become a best-effort warning path once this lands. Small feature, useful for ablation runs that skip the SOLAR arch YAML.

- **Codex adversarial review of the most recent PR** — `/codex:adversarial-review` against `d9e6c4b..dd3220a` to catch anything the non-adversarial pass missed. Highest-value targets: the deferred-`child.score` invariant (does any other call-site still assume score is populated the moment the benchmark succeeds?), the fused Phase C loop (is `_resolve_workload_roofline`'s `(0, 0)` contract honored at every call site?), and the `dataclasses.replace` in `optimize.py` (does it actually leave the caller's config untouched in every path?).

- **Multi-turn Reviewer with on-demand profiling queries** — turn the Reviewer from a single-call agent into a tool-using agent (same shape as the Coder) with two query shapes: (A) lookup into `ProfilingResult.raw_metrics` for metrics the initial NCU run already captured (cheap, in-memory); (B) on-demand re-profile with different `--section` / `--metrics` (expensive, ~30s subprocess on RTX 6000 Ada, cache-key expansion). Unlocks richer bottleneck diagnosis when the curated NCU subset (occupancy, L2, tensor-core util, top-2 stalls) doesn't explain the measured behavior. **Design discussion required first** — Reviewer shape change (single Pydantic call → tool loop), turn-budget choice, new failure modes, prompt contract. See JOURNAL → Agents → "Multi-turn Reviewer deferred" for the rationale behind the earlier deferral (briefly: without live-run data on how the curated subset actually fails, the tool risks optimizing for the wrong shape).

- **SOL integration tightening (multi-tier, 2026-04-22 plan)** — replace
  ACTS duplicates with SOL primitives where SOL owns the canonical
  version, while staying cu12.8-compatible and benchmark-agnostic.
  Consolidates four previously-separate Deferred Improvement entries
  (schema adoption, `do_bench` timing, `sol_score` delegation, reward-hack
  detection) into one sequenced plan.

  **Prerequisite env bump** (own first phase): Python 3.12 via deadsnakes
  PPA + torch cu128 wheels + `pip install -e /path/to/SOL-ExecBench
  --no-deps` + pydantic / safetensors / numpy / click / rich / pyyaml.
  Current `/tmp/acts_test_venv` (Python 3.10, no torch) stays for
  torch-less unit tests; the new 3.12 venv is for integration + live-GPU
  runs. Smoke test: `from sol_execbench.core.data import Definition;
  from sol_execbench.core.bench.io import gen_inputs;
  from sol_execbench.core.bench.timing import do_bench`.

  - **Tier 1 — Schema adoption** (trigger **fired** 2026-04-22,
    largest LOC win ~-180). Replace `src/benchmark/problem.py` +
    `problem_loader.py` + `solution_formatter.py` with direct use of
    `sol_execbench.core.data.{Definition, Workload, Solution, Trace}`.
    Drops `_problem_to_sol_dict` / `_workload_to_sol_dict` shims in
    `eval/inputs.py` (SOL pydantic validation stops running twice per
    problem load). `Definition` is itself a benchmark-agnostic kernel
    IR (tensor-in / tensor-out + pure-Python `def run` reference, no
    SOL-ExecBench-specific fields), so adopting it does NOT close off
    KernelBench — KB plugs in via a `Model.forward → def run(...)`
    converter in Tier 5 (init params via SOL's
    `custom_inputs_entrypoint` hook).
  - **Tier 2 — Timing adoption** (trigger: before first real
    multi-workload GPU run). Replace `_TorchCudaTimer` in
    `eval/benchmark.py` with
    `sol_execbench.core.bench.timing.{time_runnable, do_bench}` (syncs
    once at end instead of per-iter; drops the `BenchmarkTimer`
    Protocol's `prepare` / `flush_l2` / `finalize_ms` methods).
    Requires test-seam redesign — swap per-iter Protocol for
    `BenchmarkFn = Callable[[fn, setup], float]` alias; 12 tests in
    `tests/test_benchmark.py` assert per-iter call order and need
    rewrite around the callable. Do this as its own phase with a
    design discussion — not a drop-in change.
  - **Tier 3 — `sol_score` delegation** (bundle with Tier 1, same PR).
    `src/eval/scorer.py::compute_sol_score` becomes a ~5-line wrapper
    around `sol_execbench.sol_score.sol_score(t_k, t_p, t_sol)`,
    layering the existing `reward_hack_suspect` / `calibration_warning`
    audit flags (per SOL-ExecBench paper §4.3) on top. Keeps the
    formula canonical; any future SOL formula revision flows through
    without an ACTS patch.
  - **Tier 4 — Optional reward-hack + clock-lock** (gated on threat
    model — stays deferred today). Wire
    `sol_execbench.core.bench.reward_hack.{check_monkey_patch,
    check_thread_injection, check_result_caching}` into
    `eval/anti_cheat.py` and
    `sol_execbench.core.bench.clock_lock.are_clocks_locked` as a
    startup warning in `pipeline/optimize.py`. Today's anti-cheat is
    strict-tolerance comparison only (`correctness.py` Stage 5) + a
    performance audit flag (`scorer.py`). Trigger: ACTS evaluates
    externally-sourced kernels OR runs on multi-tenant hardware. For
    bounded internal search the threat model is empty — adoption
    would cost CPU with no benefit.
  - **Tier 5 — Benchmark adapter scaffold** (bundle with Tier 1).
    Move SOL-specific loading into `src/benchmarks/sol_execbench/load.py`
    returning `tuple[Definition, list[Workload]]`. Scaffold empty
    `src/benchmarks/kernelbench/` + `src/benchmarks/custom/` dirs so
    the benchmark-agnostic contract is visible. Downstream pipeline
    (orchestrator, `optimize.py`) consumes Definition + Workload
    directly. KernelBench converter is a future phase.

  **Recommended sequencing**: env bump → Tier 1 + Tier 3 + Tier 5
  (one PR, schema work is disjoint from live-run work) → Tier 2
  (own phase with design discussion, before first multi-workload run).
  Tier 4 stays deferred until its trigger fires.

  Full plan + CUDA-12.8 install recipe + bounded-blast-radius table
  (what cu12.8 blocks) in JOURNAL → "SOL integration tightening —
  CUDA 12.8 plan (2026-04-22)".

Recommended order: **first live GPU run** (end-to-end smoke, now observable via `runs/run_<UTC>/`) → action guidance text + `detect_hardware` in parallel (disjoint files) → adversarial review. Multi-turn Reviewer in parallel once its design discussion settles. SOL integration tightening can land in parallel once the env bump is done — Tier 1 (schemas) is disjoint from the live-run work and from action guidance. Defer the remaining Deferred Improvements until their triggers fire during the live run.

Still deferred regardless of GPU: `eval/anti_cheat.py` (threat model empty for bounded internal search; SOL-side wiring described in SOL integration tightening Tier 4 above) and `benchmark/solar_adapter.py` (needs SOLAR package installed).

## Remaining (dependency-ordered)

Items marked `(skeleton)` have interfaces + placeholder logic that keeps the pipeline runnable. Items marked `(done)` have real implementations. Unmarked items need real implementation.

### Phase 1: Foundation

- [x] config.py (done) — detect_hardware() is placeholder (deferred — YAML loading covers the primary path; now implementable via torch.cuda/pynvml since GPU is available)
- [x] kernels/kernel.py (done) — dataclasses complete
- [x] kernels/compiler.py (done) — file-backed importlib load (`spec_from_file_location` + `exec_module`), hash-keyed cache path, resolves `KernelSpec.entrypoint` via `getattr`. GPU-side Triton specialization still happens at launch time in correctness/benchmark runs.

### Phase 2: Evaluation Harness

- [x] eval/correctness.py (done) — 5-stage gate (smoke → shape-sweep → numerical stability → determinism → anti-cheat) with short-circuit failure attribution. Injectable `ComparisonPolicy` (torch-free at import); `TorchComparisonPolicy` delegates to `sol_execbench.compute_error_stats` when installed, falls back to `torch.allclose` otherwise.
- [x] eval/inputs.py (done) — `build_reference_fn` (exec PyTorch reference source, resolve `run`) + `build_input_generator` (wraps SOL's `gen_inputs` with seeding). Torch + sol_execbench lazy-imported.
- [x] eval/benchmark.py (done) — CUDA-event timing via injectable `BenchmarkTimer` Protocol; multi-workload parallel-list contract with fresh-timer-per-workload isolation; fail-closed on partial-workload failures (<half survive → `BenchmarkError`; `is_fully_successful` property on result); 100us sentinel on empty-workload path.
- [x] eval/profiler.py (done) — hybrid analytical roofline (required, fail-closed) + curated NCU subprocess (best-effort, degrades on failure). Representative workload per iteration; Phase C re-profiles the winner on every selected workload. Source-hash-keyed cache. Tier 1 fake-ncu unit tests + Tier 2 `@pytest.mark.gpu` real-GPU tests (`tests/test_profiler_gpu.py`). Per-iter signals feed the Reviewer; run-level classification comes from `classify_run` (see JOURNAL.md → "Bottleneck classify-once (2026-04-22)").
- [x] eval/roofline.py (done) — two clean paths: SOLAR (T_SOL + bottleneck together) or built-in fallback. solar_adapter.py placeholder returns synthetic data until SOLAR is installed.
- [x] eval/scorer.py (done) — SOL Score with audit flags per SOL-ExecBench paper Section 4.3
- [ ] eval/anti_cheat.py (skeleton) — two surfaces: correctness-level (input randomization, precision checks) + performance-level (T_k < T_SOL flagging from scorer)

### Phase 3: Actions & Memory

- [x] memory/experience.py (done) — Experience dataclass
- [x] memory/store.py (done) — JSON persistence with save/load
- [x] memory/retriever.py (done) — scored retrieval: kernel-type + hardware filtering, bottleneck + success + speedup scoring, reserved failure slots. Pure Python, no GPU.
- [x] actions/registry.py (done) — registry + tier system
- [ ] actions/tier1-6 (skeleton) — action definitions exist but guidance text is placeholder

### Phase 4: Agents & Prompts

- [x] agents/llm_backend.py (done) — OpenAI Agents SDK integration: ModelConfig, create_model(), run_agent() with retry (narrow transient catch + exponential backoff w/ jitter), make_run_config()
- [x] prompts/planner/system.md (done) — bottleneck→technique mapping tables, gain ranges, anti-patterns, decision rules
- [x] prompts/planner/technique_select.md (done) — documents user prompt format
- [x] prompts/coder/ (done) — system.md (prescribed workflow, hard rules, one sanctioned failure mode) + implement.md (user-prompt format)
- [x] prompts/reviewer/ (done) — system.md (diagnostic reasoning) + interpret.md
- [x] agents/planner.py (done) — Pydantic output_type, build_user_prompt(), PlanningError, technique validation
- [x] agents/coder.py (done) — tool-using Agent, Pydantic `KernelCodeOutput`, `ImplementationError`, `_max_turns = 2*config.max_debug_retries + 2` (= 8 by default; +2 over 2N covers the `submit_kernel` tool call + final plain-text confirmation), placeholder tools until compiler/correctness land
- [x] agents/reviewer.py (done) — Pydantic ReviewerFeedbackOutput, build_user_prompt, rule-based fallback (`degraded`/`error_reason`), configurable `prompt_dir`

### Phase 5: Search

- [x] search/tree.py (done) — tree state, path_to_node, checkpoint save/load (atomic)
- [x] search/beam.py (done) — beam pruning (B3 quality-weighted + B2 diversity-aware, configurable), epsilon-greedy selection
- [x] search/orchestrator.py (done) — real control flow + real agents + real CUDA-event benchmarking + real analytical profiling. Fail-closed baseline check (aborts run on partial-workload failure); branch-local `DEAD_END` on child partial failure, profile failure, or missing representative latency. Post-refactor (2026-04-22): calls `classify_run` once after roofline resolution, threads `run_bottleneck` into retriever / planner / reviewer / `SearchResult`; commits `child.score` + `per_workload_latency_us` only after the profile DEAD_END gauntlet clears.

### Phase 6: Pipeline & Integration

- [x] pipeline/optimize.py Phase A (done) — real two-path load, roofline, workload selection, model-configured `CoderAgent`, and fail-closed `generate_triton_baseline`. Phase B runs real CUDA-event benchmarking + real analytical profiling. Post-refactor (2026-04-22): placeholder hardware substitution also applies to caller-supplied zero-peak configs (not just `config is None`) via `dataclasses.replace`.
- [x] pipeline/verify.py (done) — recompiles the winner and reruns the 5-stage correctness gate against the PyTorch reference; compile failures surface as `passed=False` with a compile-phrased detail string
- [x] pipeline/report.py (done) — `generate_report` + `render_report`; trace via `result.tree.path_to_node`; propagates `reward_hack_suspect` / `calibration_warning`; surfaces run-level `bottleneck` (from `SearchResult.run_bottleneck`) and `winner_per_workload_bottlenecks` (via `classify_workload` on every selected workload, fused with the Phase C re-profile pass)
- [x] benchmark/problem_loader.py (done)
- [x] benchmark/baseline_generator.py (done) — `generate_triton_baseline` drives `CoderAgent.translate` + post-verifies on every selected workload; `BaselineGenerationError` on no-model / retry exhaustion.
- [x] benchmark/workload_selector.py (done)
- [x] benchmark/solution_formatter.py (done)
- [ ] benchmark/solar_adapter.py (skeleton) — returns synthetic data. Needs SOLAR installed.

### Future (Post-V1)
- [ ] Multi-objective optimization (power, energy-latency product)
- [ ] CUDA C++ backend (V2)
- [ ] Embedding-based memory retrieval
- [ ] Context-adaptive agent specialization
- [ ] Reviewer Knowledge Base architecture
- [ ] Parallel kernel candidate generation (Coder produces N candidates per plan)
- [ ] Multi-technique planning (Planner selects multiple complementary techniques)

## Deferred Improvements

Tech-debt items surfaced by review passes but not yet worth fixing. Each has
a **trigger** — the signal to act. If you find yourself reaching for one of
these before its trigger fires, re-read the trigger first.

- [ ] **Per-dtype peak in `_compute_analytical()` ridge** — profiler currently
  uses `hardware_spec.peak_flops_fp32` regardless of workload dtype
  (`src/eval/profiler.py:186`). For tensor-core workloads (fp16/bf16) the
  real ridge is much higher, so `classify_workload()` mislabels tc-heavy
  workloads as compute-bound when they're actually memory-bound (or vice
  versa). Search loop is unaffected (it uses SOLAR's run-level label, not
  the analytical per-workload one), so the impact is confined to Phase C
  diagnostic accuracy in `OptimizationReport.winner_per_workload_bottlenecks`.
  Fix requires plumbing `Workload.dtype` (or kernel-inspected dtype) into
  the helper plus a `peak_for_dtype(hw, dtype)` lookup against
  `HardwareSpec.MAC_per_cycle_{fp32_sm, fp16_tc, bf16_tc}`.
  *Trigger*: first Phase C report on a tc workload that shows a
  classification disagreeing with NCU's `tensor_core_util_pct`, OR the
  first SOL run whose per-workload labels look obviously wrong relative
  to the kernel's known regime. Don't pre-fix — current SOLAR stub
  (`solar_adapter.py:69-77`) means run-level labels are also fake, so
  fixing per-workload accuracy in isolation has no consumer yet.

- [ ] **`MemoryStore.add()` batched flush** — currently rewrites the full
  JSON on every add (O(N²) write bytes per session). Split into
  in-memory `add()` + explicit `flush()` at iteration boundaries.
  *Trigger*: first end-to-end run where the store grows past ~500
  experiences, OR if the rewrite shows up in a profile.

- [ ] **Tree serialization via `dataclasses.asdict`** (partial) —
  `_serialize_profiling` was switched to `asdict` during the
  bottleneck-classify-once /simplify pass (2026-04-22), which removed
  the hand-rolled per-field mirror for `AnalyticalMetrics` + `NCUMetrics`.
  `_serialize_kernel` / `_serialize_score` still hand-roll their dicts
  and carry the drift risk — `ScoreResult`'s `.get("reward_hack_suspect",
  False)` back-compat hook shows the shape of the problem. A shared
  helper with enum/Path coercion would collapse the remainder.
  *Trigger*: the next time a field is added to `TreeNode`, `Kernel`,
  `KernelSpec`, or `ScoreResult`. Don't pre-refactor — checkpoint
  back-compat risk isn't worth paying proactively.

- [ ] **Subprocess-isolated correctness / benchmark (Tier 3)** —
  SOL-ExecBench's `driver/templates/eval_driver.py` + `ProblemPackager`
  runs each submission in a fresh subprocess so kernel crashes, OOMs,
  or monkey-patch attempts don't take down the harness. Our Coder
  self-corrects in a tight in-process loop (`compile_kernel` +
  `verify_correctness` run inline per tool call).
  *Trigger*: if we start seeing real kernel crashes that kill the
  orchestrator process, or if we ever accept externally-sourced kernel
  code (reward-hack threat model). In-process is faster while the
  search is internal and bounded — don't pay subprocess per-call
  latency to solve a problem we don't have yet.

- [ ] **Coder failure surfacing at the orchestrator** — today
  `ImplementationError` (transient retry exhaustion) and SDK
  `MaxTurnsExceeded` (tool-loop budget exhaustion) both unwind
  `Orchestrator.run()`. Design intent: catch at the orchestrator
  boundary and mark the branch dead/degraded so one bad branch does
  not take down the search run.
  *Trigger*: same increment as above — once compiler/correctness are
  real, the orchestrator starts seeing genuine Coder failures, and
  "mark branch dead" has a concrete meaning.

- [ ] **`CorrectnessContext` dataclass to replace triple-kwarg
  threading** — `CoderAgent.implement()`, `CoderAgent.translate()`, and
  `Orchestrator.run()` all accept `kernel_spec` + `reference_fn` +
  `input_generators` (a list, one entry per selected workload) as three
  kwargs that are jointly required when a model is configured. The
  tri-state "all-or-none" validation is parameter sprawl. A small
  `CorrectnessContext(kernel_spec, reference_fn, input_generators)`
  dataclass would collapse the trio to one parameter at every call
  site and make the "bound oracle for this problem" concept explicit.
  Side benefit: `pipeline/optimize.py::_load_sol_execbench` and
  `benchmark/baseline_generator.py::generate_triton_baseline` currently
  each call `build_reference_fn` + `build_input_generator` once,
  running SOL pydantic validation twice per problem load. Threading
  one `CorrectnessContext` through instead of rebuilding inside the
  generator drops the duplicate validation pass.
  *Trigger*: the "baseline_generator constructs its own context"
  trigger has fired (as of the Codex-review fix round — see JOURNAL).
  Defer until a fourth field needs to travel alongside the trio
  (e.g., `device`, `tolerance_override`, or a per-problem `atol`),
  then do both the type cleanup and the dup-build fix in one pass.

- [ ] **`SearchResult.tree` → lighter path snapshot** —
  Phase C currently gets the full `SearchTree` on `SearchResult` so
  `pipeline/report.py::generate_report` can walk the root-to-best path
  for `technique_trace`. Keeping the tree around is cheap for the
  one-shot CLI path (GC'd when `main()` returns) but retains every
  node's generated source — non-best branches included — until the
  caller releases `SearchResult`. It also makes Phase C import-coupled
  to `SearchTree`, which is more surface than it needs. A lighter
  snapshot — precompute `best_path: list[TreeNode]` (or just
  `technique_trace: list[str]`) in `Orchestrator` and drop the tree
  reference — would shrink the retained footprint and narrow the
  abstraction.
  *Trigger*: when ACTS runs in a long-lived or batch context
  (server, multi-problem batch driver) where `SearchResult` outlives a
  single run, OR when tree retention shows up in a memory profile.
  Not today — the CLI caller is ephemeral, and keeping the tree lets
  future report views (per-iter SOL curve, tree depth histogram) grow
  without another orchestrator round.

- [ ] **Parallel beam expansion via `asyncio.gather`** —
  `Orchestrator.run()` currently expands one frontier node per
  iteration: select → plan → implement → benchmark → review. Each
  iteration is bounded by three sequential LLM calls (Planner, Coder,
  Reviewer). Beam width ≥ k opens the door to `asyncio.gather`-ing
  the top-k frontier picks per iteration — amortizing LLM latency
  across the beam.
  *Trigger*: when wallclock per iteration becomes the dominant cost
  in a real run (not today — search is LLM-latency-bound only once
  the full pipeline runs end-to-end). Design pass required before
  implementation: serial expansion is load-bearing for `beam_prune`
  + `MemoryStore.add()` + checkpoint writes, all of which assume
  single-writer semantics on the tree. See JOURNAL → Search →
  "Serial beam expansion" for the rationale to keep it serial today.

### Skipped (decisions, not tech debt)

- **Tier action files → YAML catalog**: `src/actions/tier{1..6}*.py` are
  mostly data (~280 LOC). Moving to YAML would trade away type-checking,
  IDE refactor support, and import-time error detection for slightly
  fewer lines. Only worth it if non-developers need to edit actions —
  which isn't the case. Keep as Python.
