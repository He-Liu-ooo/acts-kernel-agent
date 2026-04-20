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

- [x] agents/coder.py — tool-using Agent with Pydantic `KernelCodeOutput`, `build_user_prompt()`, `ImplementationError`, turn budget `2*max_debug_retries+1` (= 7 by default), temperature 0.0 for determinism. Tools wire to real `compile_kernel` / `verify_correctness` via closure-captured `KernelSpec` + `reference_fn` + **`input_generators` (list, one per selected workload — correctness tool iterates all, short-circuits on first failure)** at call time. Second entry point `translate()` (one-shot PyTorch→Triton port for baseline generation) shares tool wiring with `implement()` via private `_run_tool_agent` helper; `has_model` property for callers that must branch before reaching into internals.
- [x] prompts/coder/ — system.md (prescribed compile-then-correctness workflow, hard rules, anti-patterns, one sanctioned failure mode) + implement.md (user-prompt format doc) + translate.md (baseline-port system prompt: port PyTorch `run` to Triton `kernel_fn`, signature invariance, no precision drop)
- [x] agents/llm_backend.py — added optional `max_turns` kwarg to `run_agent()` (threads SDK tool-loop bound) and `render_kernel_section()` helper (replaces triple-duplicated fence+escape logic in coder/planner/reviewer)
- [x] Planner/Reviewer temperature bumped 0.0 → 0.3 — Coder stays at 0.0 (determinism for code gen), upstream agents get variance for technique exploration / diagnosis wording; strict Pydantic enums still pin schema

### Implemented during baseline-generator phase (real logic, not placeholders)

- [x] benchmark/baseline_generator.py — drives `CoderAgent.translate()`, recompiles the returned source, and reruns the 5-stage correctness gate against every selected workload before accepting a candidate. Post-verify catches SDK best-effort output when the Coder's turn budget is exhausted. Fail-closed: raises typed `BaselineGenerationError` on no-model-configured or retry exhaustion (no stub fallback — search against a fake baseline would look like progress).
- [x] pipeline/optimize.py Phase A — `_load_model_if_configured` reads `$ACTS_MODEL_CONFIG` / `configs/models/deepseek.json` (TOCTOU-safe via try/except), model load gated on SOL mode so placeholder CLI stays runnable. `_load_sol_execbench` now async: calls `generate_triton_baseline` and returns `reference_fn` + the full `input_generators` list so Phase B's correctness tool binds to every selected workload.
- [x] search/orchestrator.py — accepts plural `input_generators` and forwards verbatim to `CoderAgent.implement()` every iteration.

### Implemented during report phase (real logic, not placeholders)

- [x] pipeline/report.py — `generate_report(result)` walks `result.tree.path_to_node(best.id)` to build `technique_trace` (root baseline placeholder filtered out), propagates `reward_hack_suspect` / `calibration_warning` from the best node's `ScoreResult`, unwraps `TerminationReason` to a plain string, and defensively handles a `None` score. `render_report` emits a multi-line CLI summary that skips the scoring block when `baseline_latency_us == 0` and surfaces audit flags as explicit `[AUDIT]` lines. `bottleneck_transitions` stays empty pending `eval/profiler.py` (GPU-blocked).
- [x] search/orchestrator.py — `SearchResult` gained a `tree: SearchTree` field so Phase C can reconstruct path-derived views without the orchestrator denormalizing upfront; all four `SearchResult` construction sites updated (ALL_DEAD_END / SOL_TARGET / PLATEAU / BUDGET). Lighter-snapshot alternative tracked as a Deferred Improvement.
- [x] pipeline/optimize.py — `main()` now prints `render_report(generate_report(result))`.

## Next Up

Phase A, Phase B, and Phase C are wired end-to-end with a real (LLM-driven) Coder, and the doc set has been reconciled against the post-report-wiring src tree (pipeline.md / search.md). The remaining non-GPU work:

- **`actions/tier{1..6}` real guidance text** — action-library descriptions are the Planner's fuel. Structure is done; the guidance strings are placeholders. Content-heavy (literature synthesis), high impact on search quality.

GPU-bound items (`eval/benchmark.py`, `eval/profiler.py`) are blocked on hardware and remain out of scope for this environment.

## Remaining (dependency-ordered)

Items marked `(skeleton)` have interfaces + placeholder logic that keeps the pipeline runnable. Items marked `(done)` have real implementations. Unmarked items need real implementation.

### Phase 1: Foundation

- [x] config.py (done) — detect_hardware() is placeholder (deferred — YAML loading covers the primary path)
- [x] kernels/kernel.py (done) — dataclasses complete
- [x] kernels/compiler.py (done) — file-backed importlib load (`spec_from_file_location` + `exec_module`), hash-keyed cache path, resolves `KernelSpec.entrypoint` via `getattr`. GPU-side Triton specialization still happens at launch time in correctness/benchmark runs.

### Phase 2: Evaluation Harness

- [x] eval/correctness.py (done) — 5-stage gate (smoke → shape-sweep → numerical stability → determinism → anti-cheat) with short-circuit failure attribution. Injectable `ComparisonPolicy` (torch-free at import); `TorchComparisonPolicy` delegates to `sol_execbench.compute_error_stats` when installed, falls back to `torch.allclose` otherwise.
- [x] eval/inputs.py (done) — `build_reference_fn` (exec PyTorch reference source, resolve `run`) + `build_input_generator` (wraps SOL's `gen_inputs` with seeding). Torch + sol_execbench lazy-imported.
- [ ] eval/benchmark.py (skeleton) — latency measurement. Needs compiler.py + GPU.
- [ ] eval/profiler.py (skeleton) — NCU integration + per-iteration bottleneck classification. Needs GPU. Note: orchestrator must call per candidate and feed dynamic classification to retriever/reviewer/planner (see JOURNAL.md "Dynamic bottleneck reclassification")
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
- [x] agents/coder.py (done) — tool-using Agent, Pydantic `KernelCodeOutput`, `ImplementationError`, `_MAX_TURNS=7` (see Deferred: config wiring), placeholder tools until compiler/correctness land
- [x] agents/reviewer.py (done) — Pydantic ReviewerFeedbackOutput, build_user_prompt, rule-based fallback (`degraded`/`error_reason`), configurable `prompt_dir`

### Phase 5: Search

- [x] search/tree.py (done) — tree state, path_to_node, checkpoint save/load (atomic)
- [x] search/beam.py (done) — beam pruning (B3 quality-weighted + B2 diversity-aware, configurable), epsilon-greedy selection
- [ ] search/orchestrator.py (skeleton) — real control flow + real agents; still calls placeholder `eval/benchmark.py` / `eval/profiler.py` for latency and bottleneck classification

### Phase 6: Pipeline & Integration

- [x] pipeline/optimize.py Phase A (done) — real two-path load, roofline, workload selection, model-configured `CoderAgent`, and fail-closed `generate_triton_baseline`. Phase B still bounded by placeholder `eval/benchmark.py` / `eval/profiler.py`.
- [x] pipeline/verify.py (done) — recompiles the winner and reruns the 5-stage correctness gate against the PyTorch reference; compile failures surface as `passed=False` with a compile-phrased detail string
- [x] pipeline/report.py (done) — `generate_report` + `render_report`; trace via `result.tree.path_to_node`; propagates `reward_hack_suspect` / `calibration_warning`; `bottleneck_transitions` stays empty pending profiler
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

- [ ] **Thread `BottleneckType` end-to-end** — move the enum out of
  `src/eval/roofline.py` into a shared types module, change
  `Experience.bottleneck_before/after` and `MemoryRetriever.retrieve` to
  take the enum, drop the `.value` string hops in orchestrator/retriever.
  Purely a typing change; prevents a `"memory_bound"` vs `"memory-bound"`
  drift bug.
  *Trigger*: before memory is first exercised with a real, scored
  retrieval run — earlier if a second bottleneck-valued site is added.

- [ ] **`MemoryStore.add()` batched flush** — currently rewrites the full
  JSON on every add (O(N²) write bytes per session). Split into
  in-memory `add()` + explicit `flush()` at iteration boundaries.
  *Trigger*: first end-to-end run where the store grows past ~500
  experiences, OR if the rewrite shows up in a profile.

- [ ] **Tree serialization via `dataclasses.asdict`** —
  `src/search/tree.py` has ~100 LOC of hand-rolled `_serialize_*` /
  `_deserialize_*`. A shared helper with enum/Path coercion would
  collapse it to ~20 LOC and remove the 3-place change cost when a
  dataclass field is added (the `.get("reward_hack_suspect", False)` on
  line 228 already shows the drift cost).
  *Trigger*: the next time a field is added to `TreeNode`, `Kernel`,
  `KernelSpec`, or `ScoreResult`. Don't pre-refactor — checkpoint
  back-compat risk isn't worth paying proactively.

- [ ] **Adopt SOL-ExecBench `Definition` / `Workload` pydantic models
  end-to-end (Tier 2)** — today ACTS parses SOL definition.json /
  workload.jsonl into hand-written dataclasses (`src/benchmark/problem.py`),
  and `src/eval/inputs.py` round-trips them to dicts to feed SOL's
  `gen_inputs`. Replacing ACTS's `Problem` / `Workload` with SOL's
  pydantic models would drop the round-trip, the duplicated schema,
  and the `_problem_to_sol_dict` / `_workload_to_sol_dict` shims.
  Also applies to `Trace` / `Correctness` / `Performance` in
  `solution_formatter.py`.
  *Trigger*: when `benchmark/baseline_generator.py` lands and starts
  passing definitions through the full pipeline. At that point the
  duplicated schema has to travel further and the shim cost shows up;
  refactor then. Keep in mind future KernelBench support — the Problem
  abstraction may need to stay benchmark-agnostic, with SOL's pydantic
  as one backend rather than the universal type.

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

- [ ] **Reward-hack detection (Tier 3)** —
  `sol_execbench.core.bench.reward_hack` catches monkey-patches of
  torch primitives, thread injection, lazy/deferred outputs, and
  critical-function tampering. Our current anti-cheat is strict-tolerance
  comparison only (`eval/correctness.py` Stage 5) plus a
  performance-side audit flag in `eval/scorer.py`.
  *Trigger*: when the agent loop runs against a multi-tenant surface
  or accepts code from outside the controlled search. For a bounded
  internal search the threat model is empty — adding these checks now
  would cost CPU and add nothing.

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

- [ ] **`sys.modules` compile cache in `kernels/compiler.py`** —
  `compile_kernel` writes `<stem>.py` to the cache dir and calls
  `spec.loader.exec_module()` unconditionally, even when `stem`
  (source hash prefix) already resolves in `sys.modules`. Three
  repeat-compile vectors share this cost: (a) the Coder's correctness
  tool compiles the candidate; (b) within a single SDK turn loop the
  same source can be handed to compile + correctness back-to-back, or
  to correctness twice if the model re-invokes it on unchanged source;
  (c) `baseline_generator`'s post-verify pass recompiles after
  `translate()` returns, and `pipeline/verify` recompiles the winner
  post-search — all guaranteed identical hash, guaranteed cache hit,
  currently re-executed. Short-circuit via `sys.modules.get(module_name)`
  + `getattr(module, entrypoint)` would eliminate every repeat.
  *Trigger*: when a real Triton compile shows up in a profile — likely
  as soon as `eval/benchmark.py` lands and live SOL problems run. Skip
  until then; the Python-level file write + exec_module pair is cheap
  at current scale, and adding the cache introduces a "stale module in
  sys.modules after reload" failure mode that we'd have to reason about.

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
