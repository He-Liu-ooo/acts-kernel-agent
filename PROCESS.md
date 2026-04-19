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

- [x] agents/coder.py — tool-using Agent with Pydantic `KernelCodeOutput`, `build_user_prompt()`, `ImplementationError`, `_MAX_TURNS` derived from `ACTSConfig.max_debug_retries` (= 2×n+1, default 7), temperature 0.0 for determinism. Tools wire to real `compile_kernel` / `verify_correctness` via closure-captured `KernelSpec` + `reference_fn` + `input_generator` at `implement()` call time.
- [x] prompts/coder/ — system.md (prescribed compile-then-correctness workflow, hard rules, anti-patterns, one sanctioned failure mode: emit last-compiled source on budget exhaustion) + implement.md (user-prompt format doc)
- [x] agents/llm_backend.py — added optional `max_turns` kwarg to `run_agent()` (threads SDK tool-loop bound) and `render_kernel_section()` helper (replaces triple-duplicated fence+escape logic in coder/planner/reviewer)
- [x] Planner/Reviewer temperature bumped 0.0 → 0.3 — Coder stays at 0.0 (determinism for code gen), upstream agents get variance for technique exploration / diagnosis wording; strict Pydantic enums still pin schema

## Next Up

### benchmark/baseline_generator.py — PyTorch-to-Triton one-shot translation

Uses the Coder to translate a problem's PyTorch reference into a Triton baseline at problem load time. Now that `kernels/compiler.py`, `eval/correctness.py`, and the Coder tool wiring are real, the remaining work is:

1. Build the per-problem `KernelSpec` + `reference_fn` + `input_generator` from a loaded `Problem` (helpers live in `src/eval/inputs.py`).
2. Drive `CoderAgent.implement()` with a translation plan (Planner-free — this is a fixed "translate PyTorch `run()` to Triton `kernel_fn`" task).
3. Retry up to `ACTSConfig.max_baseline_retries` attempts; skip the problem if all fail (per `definition.json` → baseline contract).

### Orchestrator-side Coder failure handling (deferred — see JOURNAL)

Currently `ImplementationError` propagates out of `Orchestrator.run()`. Design intent: mark branch dead/degraded and continue. Deferred until the orchestrator's per-iteration Coder call site lands real eval/scoring. Tracked here so it isn't forgotten.

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
- [ ] search/orchestrator.py (skeleton) — has real control flow but calls placeholder agents/eval

### Phase 6: Pipeline & Integration

- [ ] pipeline/optimize.py (skeleton) — has real Phase A flow (two load paths, roofline, workload selection) but calls placeholder baseline generator
- [x] pipeline/verify.py (done) — recompiles the winner and reruns the 5-stage correctness gate against the PyTorch reference; compile failures surface as `passed=False` with a compile-phrased detail string
- [ ] pipeline/report.py (skeleton) — report generation
- [x] benchmark/problem_loader.py (done)
- [ ] benchmark/baseline_generator.py (skeleton) — Triton baseline generation. Needs Coder agent.
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
  threading** — `CoderAgent.implement()` and `Orchestrator.run()` both
  accept `kernel_spec` + `reference_fn` + `input_generator` as three
  Optional kwargs that are jointly required when a model is configured.
  The tri-state "all-or-none" validation is parameter sprawl. A small
  `CorrectnessContext(kernel_spec, reference_fn, input_generator)`
  dataclass would collapse the trio to one parameter at both call
  sites and make the "bound oracle for this problem" concept explicit
  in the type system.
  *Trigger*: when a fourth field needs to travel alongside the trio
  (e.g., `device`, `tolerance_override`, or a per-problem `atol`),
  OR when `benchmark/baseline_generator.py` starts constructing its
  own context — whichever comes first. Pre-refactor cost is low but
  the restructure touches `coder.py`, `orchestrator.py`, and their
  tests, so pay it when there's a concrete second motivation.

- [ ] **`sys.modules` compile cache in `kernels/compiler.py`** —
  `compile_kernel` writes `<stem>.py` to the cache dir and calls
  `spec.loader.exec_module()` unconditionally, even when `stem`
  (source hash prefix) already resolves in `sys.modules`. In the
  search loop, the Coder's correctness tool compiles each candidate,
  then `verify_optimized_kernel` recompiles the winner post-loop —
  guaranteed identical hash, guaranteed cache hit, currently executed
  twice. Short-circuit via `sys.modules.get(module_name)` +
  `getattr(module, entrypoint)` would eliminate the double compile.
  *Trigger*: when a real benchmark shows the double-compile in a
  profile, OR when we add a third in-process compile site
  (e.g., benchmark.py runs the winner once more on GPU). Skip until
  then — the file write + exec_module pair is cheap at current scale
  and adding the cache introduces a new "stale module in sys.modules
  after a reload" failure mode that we'd have to reason about.

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
