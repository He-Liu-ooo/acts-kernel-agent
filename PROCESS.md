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

- [x] agents/coder.py — tool-using Agent with Pydantic `KernelCodeOutput`, `build_user_prompt()`, `ImplementationError`, hardcoded `_MAX_TURNS=7` turn budget (maps to "3 compile+correctness tries"), temperature 0.0 for determinism. Tools `compile_kernel_tool` / `check_correctness_tool` are placeholder stubs returning success strings — real wiring lands with `kernels/compiler.py` + `eval/correctness.py`.
- [x] prompts/coder/ — system.md (prescribed compile-then-correctness workflow, hard rules, anti-patterns, one sanctioned failure mode: emit last-compiled source on budget exhaustion) + implement.md (user-prompt format doc)
- [x] agents/llm_backend.py — added optional `max_turns` kwarg to `run_agent()` (threads SDK tool-loop bound) and `render_kernel_section()` helper (replaces triple-duplicated fence+escape logic in coder/planner/reviewer)
- [x] Planner/Reviewer temperature bumped 0.0 → 0.3 — Coder stays at 0.0 (determinism for code gen), upstream agents get variance for technique exploration / diagnosis wording; strict Pydantic enums still pin schema

## Next Up

### benchmark/baseline_generator.py — PyTorch-to-Triton one-shot translation

Uses the Coder to translate a problem's PyTorch reference into a Triton baseline at problem load time. Depends on:

- `kernels/compiler.py` (real Triton compilation) — unlocks the Coder's compile tool.
- `eval/correctness.py` (5-stage correctness gate) — unlocks the Coder's correctness tool and verifies the generated baseline against the PyTorch reference.

Order of work: `kernels/compiler.py` → `eval/correctness.py` → wire Coder tool stubs to real calls → `benchmark/baseline_generator.py`.

### Orchestrator-side Coder failure handling (deferred — see JOURNAL)

Currently `ImplementationError` propagates out of `Orchestrator.run()`. Design intent: mark branch dead/degraded and continue. Deferred until the orchestrator's per-iteration Coder call site lands real eval/scoring. Tracked here so it isn't forgotten.

## Remaining (dependency-ordered)

Items marked `(skeleton)` have interfaces + placeholder logic that keeps the pipeline runnable. Items marked `(done)` have real implementations. Unmarked items need real implementation.

### Phase 1: Foundation

- [x] config.py (done) — detect_hardware() is placeholder (deferred — YAML loading covers the primary path)
- [x] kernels/kernel.py (done) — dataclasses complete
- [ ] kernels/compiler.py (skeleton) — Triton compilation. Needs GPU for real tests.

### Phase 2: Evaluation Harness

- [ ] eval/correctness.py (skeleton) — 5-stage correctness verification. Needs compiler.py + GPU.
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
- [ ] pipeline/verify.py (skeleton) — post-optimization verification
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

- [ ] **Wire `_MAX_TURNS` through config** — Coder currently hardcodes
  the SDK turn budget as `_MAX_TURNS = 7` (derived from "3 compile+
  correctness tries"). The existing `ACTSConfig.max_debug_retries=3`
  captures the same intent at the user-facing level but is not yet
  read by CoderAgent. Config wiring ties the two together:
  `_MAX_TURNS = 2 * config.max_debug_retries + 1`.
  *Trigger*: when `kernels/compiler.py` and `eval/correctness.py` are
  wired into the Coder's tools — the same increment that turns the
  placeholder stubs into real compile/correctness calls should accept
  `ACTSConfig` on `CoderAgent` and drop the module constant.

- [ ] **Coder failure surfacing at the orchestrator** — today
  `ImplementationError` (transient retry exhaustion) and SDK
  `MaxTurnsExceeded` (tool-loop budget exhaustion) both unwind
  `Orchestrator.run()`. Design intent: catch at the orchestrator
  boundary and mark the branch dead/degraded so one bad branch does
  not take down the search run.
  *Trigger*: same increment as above — once compiler/correctness are
  real, the orchestrator starts seeing genuine Coder failures, and
  "mark branch dead" has a concrete meaning.

### Skipped (decisions, not tech debt)

- **Tier action files → YAML catalog**: `src/actions/tier{1..6}*.py` are
  mostly data (~280 LOC). Moving to YAML would trade away type-checking,
  IDE refactor support, and import-time error detection for slightly
  fewer lines. Only worth it if non-developers need to edit actions —
  which isn't the case. Keep as Python.
