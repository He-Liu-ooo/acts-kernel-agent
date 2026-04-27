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

- [x] src/runtime/{__init__,timefmt,events,run_context}.py — new package. `RunContext.create(root, *, trace_dir=None, capture_traces=True)` owns per-invocation lifecycle: creates `./runs/run_<UTC>/` with microsecond-precision filename (collision-safe for parallel invocations), configures stdlib `FileHandler` + `StreamHandler` on the root logger, silences `httpx`/`openai`/`agents` to WARNING, binds `events.jsonl`, and wires the SDK trace processor under `<run-dir>/traces/` (or an explicit `--trace-dir` override). `close()` is idempotent; `atexit` + `finally` wired so crashed runs still flush. `_NullRunContext` fallback on any OSError during setup keeps `emit()` in logger-only mode. `events.emit(kind, *, iter, **fields)` fans out to both sinks with `never-raise` discipline; 19 `CORE_EVENT_KINDS` in the catalog. `finite_or_none()` sanitizes `inf`/`nan` benchmark sentinels so `events.jsonl` stays RFC-8259 valid. See `doc/runtime.md` for the full reference + JOURNAL → "Logger system before first live GPU run (2026-04-23)" for design rationale.
- [x] pipeline/optimize.py — added `--run-dir` CLI flag (default `./runs`); changed `--trace-dir` default from `"traces"` to `None` (routes through RunContext); deleted `_enable_traces_if_possible` helper (replaced by `RunContext._wire_trace_capture`). `main()` wraps body in RunContext + `atexit.register(ctx.close)`; emits `run_start` at entry and `run_end` at exit (including `termination_reason="ERROR"` on the exception path). New `_is_model_configured()` helper populates `run_start.model_configured` before `optimize()` loads the model.
- [x] search/orchestrator.py — emits 10 iteration-level events + `baseline_ready` per run. Coder outcome events are `coder_submitted` (no pass claim) and `coder_failed(reason)` — the orchestrator can't verify gates from `implement()`'s return value alone, so ground-truth per-tool-call records live in `traces/*.jsonl`. `iter_end.outcome` is `advanced`/`dead_end`/`skipped` (skipped only on Coder failure, no tree mutation). `running_best_score` tracked as a local float (O(N) total) instead of per-iter `tree.best_node()` (was O(N²)). `_per_workload_us` + `_emit_dead_end` helpers dedup the per-workload list-comp and the two-emit pair across 4 dead-end sites.
- [x] benchmark/baseline_generator.py — emits `baseline_attempt` / `baseline_success` / `baseline_failure` in the retry loop.
- [x] pipeline/verify.py — emits `verify_start` / `verify_done` on both return paths (compile-failure + post-compile).
- [x] agents/trace_processor.py — dropped local `_isoformat_utc`, now imports `filename_ts` from `src/runtime/timefmt.py`; `datetime.utcnow()` deprecation fixed.
- [x] .gitignore — `/runs/` added alongside `/traces/`.

### Implemented during planner/reviewer submit-tool migration phase (real logic, not placeholders)

- [x] agents/llm_backend.py — added `SUBMIT_OK_SENTINEL` constant + `format_submit_validation_error(exc)` helper. Both Coder/Planner/Reviewer submit-tool factories now share one source of truth for the OK string the LLM must emit on success and for the per-field validation-error rendering on failure. Replaces three near-identical inlined formatters.
- [x] agents/coder.py — refactored to import `SUBMIT_OK_SENTINEL` + `format_submit_validation_error` from `llm_backend` (no behavior change; coder remains the canonical option-α reference).
- [x] agents/planner.py — full option-α migration. New private `_make_submit_plan_tool(captured_output)` factory returns a closure-bound `@function_tool` that calls `_validate_and_convert(payload)` and pushes the validated `PlannerOutput` into the captured slot, returning `SUBMIT_OK_SENTINEL` on success or `format_submit_validation_error(...)` on Pydantic failure. `run()` builds an `Agent` with the tool registered, calls `Runner.run(..., max_turns=4)`, and on `MaxTurnsExceeded` falls back to the captured output if the LLM submitted before exhausting turns; raises `PlanningError` only if no valid submission was captured. `has_model` property gates external callers on `_SDK_AVAILABLE`. Submit-tool fields use Pydantic defaults so the LLM can omit optional fields without tool-call rejection.
- [x] agents/reviewer.py — symmetric option-α migration. `_make_submit_review_tool` factory mirrors planner's; `run()` uses `max_turns=4` + `MaxTurnsExceeded → captured-output recovery → degraded fallback` (degraded path tags `error_reason=max_turns_exceeded` or `missing_submit_review` instead of raising). `has_model` property added. Submit-tool fields use Pydantic defaults.
- [x] search/orchestrator.py — `PlanningError` catch site added next to existing `ImplementationError` catch; both increment `parent.consecutive_agent_failures`; successful child commit resets the counter to 0. Emits `planner_failed(reason)` event on the planner-failure path (mirrors `coder_failed`).
- [x] search/tree.py — quarantine concept added: `TreeNode.consecutive_agent_failures: int` (default 0), module constant `QUARANTINE_THRESHOLD = 2`, `frontier()` filters out nodes at/above the threshold so chronically-failing branches stop attracting selection. `best_node()` deliberately still considers them. Serialize/deserialize updated; legacy checkpoints default the counter to 0.
- [x] runtime/events.py — `planner_failed` added to `CORE_EVENT_KINDS` (count 18 → 19). Symmetric with `coder_failed`.
- [x] prompts/planner/system.md + prompts/reviewer/system.md — appended `## Submission` section to each, documenting the `submit_plan` / `submit_review` tool contract (must call exactly once, expected field shapes, `OK` sentinel behavior).
- [x] agents/planner.py + agents/reviewer.py — `function_tool(_make_submit_*_tool(...), strict_mode=False)` on both submit-tool registrations. The SDK's strict-schema validator rejects `dict[str, X]` (`params` on planner, `metric_deltas` on reviewer) with the same `additionalProperties should not be set` UserError that originally killed the Pydantic-output path; `strict_mode=False` bypasses the pre-flight check while keeping Pydantic validation inside the tool body. Validated end-to-end on the first live GPU run (rmsnorm, 2026-04-26, runs/run_20260426T152032_091547Z/). Tests widen 20 existing `function_tool` lambda patches to accept kwargs + add `test_submit_tool_registered_with_strict_mode_false` regression guard per agent. See JOURNAL → "Strict-mode opt-out for submit-tool dict params (2026-04-26)".
- [x] agents/reviewer.py + prompts/reviewer/system.md — multi-turn Variant A: optional `query_metric` tool gated by `ACTSConfig.reviewer_metric_queries` (default False); registers alongside `submit_review` with `max_turns=6` (was 4); prompt grows `## Available raw metrics (queryable)` menu listing `ProfilingResult.raw_metrics` keys; tool returns `{name: stringified-float | "[unknown]" | "[no data]"}`; new `reviewer_metric_query` event (CORE_EVENT_KINDS 19→20). `strict_mode=False` opt-out reused for the same SDK reason as the submit-tool dict params (JOURNAL 2026-04-26). See JOURNAL → "Multi-turn Reviewer (Variant A): on-demand metric queries (2026-04-27)".

## Next Up

Phase A, Phase B, Phase C are all wired end-to-end on real CUDA-event benchmarking + real analytical profiling. GPU is available (NVIDIA RTX 6000 Ada, CUDA 12.8). **First live GPU run completed 2026-04-26** (rmsnorm, run dir `runs/run_20260426T152032_091547Z/`) — full Phase A → B → C, plateau termination after 3 iterations, no failures or quarantines. **V1 completion cleared 2026-04-27** (action library guidance, real `detect_hardware()`, SOLAR adapter integration; second live GPU run validated SOLAR T_SOL=0.282µs for rmsnorm batch=16). **Multi-turn Reviewer (Variant A) shipped 2026-04-27** — opt-in `query_metric` tool behind `ACTSConfig.reviewer_metric_queries` (default off). Validation milestone for the multi-turn capability is a third live GPU run with the flag on.

### Active phase — SOL integration (firing next, in order)

Ordered sequence — design discussion happens before each phase fires, code lands one phase at a time. **Scope expansion 2026-04-27**: SOL integration phase now adopts every applicable in-process SOL primitive. Tier 4 (reward-hack + clock-lock) PROMOTED from optional → required and lands in this phase, not in anti-cheat. Two new tiers added: output handling (Tier 6), safetensors loading (Tier 7). **Tier 8 (subprocess-isolated evaluation) was briefly in scope and then deferred same-day** — inline and subprocess have functional parity for ACTS's actual use case; subprocess's crash-recovery and tampering-isolation benefits don't justify the per-eval overhead under the current threat model. See JOURNAL → "SOL integration scope refinement — Tier 8 (subprocess) deferred (2026-04-27)" for the rationale and trigger conditions, and Deferred Improvements below for the entry.

1. **SOL integration tightening (2026-04-22 plan + 2026-04-27 scope expansion + Tier 8 deferral)** — see expanded Backlog entry below for the full 7-tier plan (Tiers 1–7; Tier 8 deferred). Sub-phase sequence (option B, 3 PRs): env bump (Python 3.12 + cu128 + `pip install -e SOL-ExecBench`) → library primitives mega-PR (Tier 1 + Tier 3 + Tier 4 + Tier 5 + Tier 6 + Tier 7, all in-process) → Tier 2 (timing adoption with `do_bench` + memory pool, own phase with test-seam design discussion). Each sub-phase gets a design pass per CLAUDE.md before code lands.

2. **`eval/anti_cheat.py` orchestration layer** — collapses to a thin wrapper now that SOL Tier 4 lands the process-level reward-hack primitives directly. `anti_cheat.py` becomes the coordination point: routes the `reward_hack_suspect` / `calibration_warning` flags from `scorer.py` (performance-level surface), composes Stage 5 randomized-input checks from `correctness.py` (correctness-level surface), and exposes the SOL `reward_hack` snapshot/check pair (process-level surface) as a context manager the orchestrator wraps each evaluation with. Skeleton-level placeholders (`generate_randomized_inputs`, `strict_tolerance_check`) finalize here. Design discussion required first — orchestrator-side handler shape for the audit flags (mark branch dead vs warn-only vs human-review queue), check fire timing (per-iter vs startup vs final-report). **No code yet** until design lands.

3. **Backward-kernel SOLAR support** — see existing Backlog entry below. Sequenced after anti-cheat so the schema decisions (parse `op_type` suffix vs. add `Problem.kind` field) ride on the SOL Tier 1 schema-adoption work that lands in phase 1 above, rather than landing as a one-off.

### Backlog (post-V1-completion)

- **Codex adversarial review of the most recent PR** — `/codex:adversarial-review` against `d9e6c4b..dd3220a` to catch anything the non-adversarial pass missed. Highest-value targets: the deferred-`child.score` invariant (does any other call-site still assume score is populated the moment the benchmark succeeds?), the fused Phase C loop (is `_resolve_workload_roofline`'s `(0, 0)` contract honored at every call site?), and the `dataclasses.replace` in `optimize.py` (does it actually leave the caller's config untouched in every path?).

- **Variant B — `reprofile(sections, metrics) -> ProfilingResult`** (multi-turn Reviewer next step). On-demand `ncu` subprocess re-run with caller-specified `--section` / `--metrics`; ~30s per call; cache-key expansion. *Trigger*: a real run where the LLM consistently asks `query_metric` for keys *not* in `raw_metrics`, AND the curated NCU section is genuinely the wrong section for the bottleneck shape.

- **`request_workload_variant(workload_idx)` — re-bench against a different selected workload** (multi-turn Reviewer further extension). Pulls in `BenchmarkTimer` + `input_generators[idx]` re-entry. *Trigger*: a real run where Phase C reveals per-workload disagreement with the iteration's chosen representative workload, and the Reviewer would have benefited from re-bench data mid-review.

- **`ACTSConfig.reviewer_max_turns` operator-tunable**. Today the multi-turn budget is fixed at 6. *Trigger*: a real run where the fixed `max_turns=6` budget shows pressure (LLM consistently busts on legitimate paths, not pathological ones).

- **Hard cap on `query_metric` invocations per review** (independent of turn budget). *Trigger*: a real run showing pathological query loops within the turn budget.

- **`prompt_dir`-based Compute / Memory Reviewer split** consuming the same `query_metric` tool. *Trigger*: a run where one specialty class of bottleneck consistently warrants a different fetch heuristic than the other.

- **SOL integration tightening (2026-04-22 plan + 2026-04-27 scope
  expansion)** — replace ACTS duplicates with SOL primitives wherever
  SOL owns the canonical version, while staying cu12.8-compatible and
  benchmark-agnostic. The 2026-04-22 plan covered five tiers (schemas,
  `do_bench` timing, `sol_score`, optional reward-hack + clock-lock,
  benchmark adapter). The 2026-04-27 scope expansion (user direction)
  promoted Tier 4 from optional → required, expanded its surface to
  the full reward-hack detector set + active clock locking, and added
  two new tiers (output handling, safetensors loading). Tier 8
  (subprocess-isolated evaluation) was briefly in scope and then
  deferred same-day after a functional-parity analysis showed inline
  and subprocess produce identical results for ACTS's use case while
  subprocess adds ~200–500ms per evaluation — the previously-separate
  "Subprocess-isolated correctness / benchmark" Deferred entry stays
  deferred, now with the trigger conditions tightened (see Deferred
  Improvements below). See JOURNAL → "SOL integration scope refinement
  — Tier 8 (subprocess) deferred (2026-04-27)" for the full rationale.

  **Prerequisite env bump** (own first sub-phase): Python 3.12 via
  deadsnakes PPA + torch cu128 wheels + `pip install -e /path/to/SOL-ExecBench
  --no-deps` + pydantic / safetensors / numpy / click / rich / pyyaml.
  Current `/tmp/acts_test_venv` (Python 3.10, no torch) stays for
  torch-less unit tests; the new 3.12 venv is for integration + live-GPU
  runs. Smoke test: `from sol_execbench.core.data import Definition;
  from sol_execbench.core.bench.io import gen_inputs;
  from sol_execbench.core.bench.timing import do_bench`.

  **In-process library primitives** (Tiers 1–7 below — adopt SOL
  primitives without changing ACTS's process model). Tier 8
  (subprocess-isolated evaluation) was a Group 2 architectural change
  briefly in scope; it has been deferred — see Deferred Improvements
  below.

  - **Tier 1 — Schema adoption** (trigger **fired** 2026-04-22,
    largest LOC win ~-180). Replace `src/benchmark/problem.py` +
    `problem_loader.py` + `solution_formatter.py` with direct use of
    `sol_execbench.core.data.{Definition, Workload, Solution, Trace}`
    plus all input variants (`RandomInput` / `ScalarInput` /
    `SafetensorsInput` / `CustomInput` / `ToleranceSpec`), all
    solution types (`SourceFile` / `BuildSpec` / `CompileOptions` /
    `SupportedLanguages` / `SupportedHardware` / `SupportedBindings`),
    and all trace types (`Correctness` / `Performance` / `Environment`
    / `EvaluationStatus` / `Evaluation`). Drops `_problem_to_sol_dict`
    / `_workload_to_sol_dict` shims in `eval/inputs.py`. Pulls in
    `core/utils.py::env_snapshot` + `hardware_from_device` (needed to
    populate `Trace.environment`). `Definition` is a benchmark-agnostic
    kernel IR — KernelBench plugs in via Tier 5 converter, not via a
    parallel Problem abstraction.
  - **Tier 2 — Timing adoption + per-iter memory pool** (trigger: before
    next multi-workload GPU run). Replace `_TorchCudaTimer` in
    `eval/benchmark.py` with `sol_execbench.core.bench.timing.
    {time_runnable, do_bench, clone_args}` (syncs once at end vs
    per-iter; drops `BenchmarkTimer` Protocol's `prepare` / `flush_l2`
    / `finalize_ms`). Wire `sol_execbench.core.bench.io.
    ShiftingMemoryPoolAllocator` into `do_bench`'s `setup` callback —
    advances `data_ptr` per iteration to defeat result-caching kernels
    keyed on tensor `id()`. Test-seam redesign: swap per-iter Protocol
    for `BenchmarkFn = Callable[[fn, setup], float]` alias; 12 tests
    in `tests/test_benchmark.py` need rewrite. Own phase with design
    discussion — not a drop-in change.
  - **Tier 3 — `sol_score` delegation** (bundle with Tier 1, same PR).
    `src/eval/scorer.py::compute_sol_score` becomes a ~5-line wrapper
    around `sol_execbench.sol_score.sol_score(t_k, t_p, t_sol)`,
    layering the existing `reward_hack_suspect` / `calibration_warning`
    audit flags (per SOL-ExecBench paper §4.3) on top.
  - **Tier 4 — Reward-hack defense + active clock locking** (PROMOTED
    2026-04-27 from optional to required, scope expanded). Wire the
    full `sol_execbench.core.bench.reward_hack` detector set into
    `eval/anti_cheat.py`: `check_monkey_patch` (catches torch primitive
    rebinding), `check_thread_injection`, `check_lazy_outputs`
    (catches lazy/deferred outputs that look like cached results —
    note: 2026-04-22 entry mis-named this `check_result_caching`),
    `snapshot_critical_functions` + `check_eval_integrity` (catches
    namespace tampering between snapshot and check). For active clock
    locking go beyond warn-only adoption: drive clocks via
    `sol_execbench.core.bench.clock_lock.{lock_clocks, verify_clocks,
    unlock_clocks, probe_clock_lock_available}` plus `BenchmarkConfig`
    + `device_config.get_clock_preset(device_name)` (preset table per
    GPU model). Requires `sudo nvidia-smi --lock-gpu-clocks`; falls
    back to warn-only on permission denial. Removes boost-clock
    variance as a real source of timing noise on Ada / H100. **Design
    implication**: `reward_hack._ELAPSED_TIME_ADDR` snapshots
    `torch.cuda.Event.elapsed_time`'s `id()` at SOL module-load time,
    so `pipeline/optimize.py::main` must `import sol_execbench` before
    any candidate kernel touches torch.
  - **Tier 5 — Benchmark adapter scaffold** (bundle with Tier 1).
    Move SOL-specific loading into `src/benchmarks/sol_execbench/load.py`
    returning `tuple[Definition, list[Workload]]`. Scaffold empty
    `src/benchmarks/kernelbench/` + `src/benchmarks/custom/` dirs so
    the benchmark-agnostic contract is visible. Downstream pipeline
    (orchestrator, `optimize.py`) consumes Definition + Workload
    directly. KernelBench converter is a future phase.
  - **Tier 6 — Output handling** (NEW 2026-04-27). Adopt
    `sol_execbench.core.bench.io.{normalize_outputs, allocate_outputs}`.
    `normalize_outputs` handles tuple / dict / scalar return shapes
    uniformly; `allocate_outputs` pre-allocates DPS
    (destination-passing-style) output buffers. ACTS today assumes
    single-tensor outputs throughout the 5-stage gate and benchmark
    loop; this unlocks SOL problems with multi-output or DPS kernels.
    Touches `eval/correctness.py` (gate) and `eval/benchmark.py`
    (timing loop).
  - **Tier 7 — Safetensors input loading** (NEW 2026-04-27). Adopt
    `sol_execbench.core.bench.io.load_safetensors` for workloads whose
    `workload.jsonl` references safetensors blobs (some SOL problems
    carry frozen weight tensors via safetensors paths instead of random
    init). Without this, those problems error at input generation.
    Touches `eval/inputs.py::build_input_generator` — detect
    `SafetensorsInput` entries in the workload, pre-load via
    `load_safetensors` once at problem-load, thread the loaded tensor
    dict through the per-trial generator.

  **Recommended sub-phase sequencing** (option B, 3 PRs total):
  env bump → library primitives mega-PR (Tier 1 + Tier 3 + Tier 4 +
  Tier 5 + Tier 6 + Tier 7, all in-process) → Tier 2 (own phase with
  design discussion on test seam, before next multi-workload run).
  The mega-PR clusters all schema / scoring / defenses / adapter /
  output / safetensors work — they share the SOL pydantic foundation
  laid by Tier 1 and review better as one coherent surface than as
  three smaller PRs.

  Full IN/OUT decision + bounded-blast-radius table for what cu12.8
  blocks in JOURNAL → "SOL integration tightening — CUDA 12.8 plan
  (2026-04-22)" + "SOL integration scope expansion — adopt every
  applicable primitive (2026-04-27)".

SOL integration tightening can land in parallel with the active V1-completion phase once the env bump is done — Tier 1 (schemas) is disjoint from action-guidance work and from `detect_hardware()`/`solar_adapter` work. Defer the remaining Deferred Improvements until their triggers fire during the next live run (after the V1-completion phase lands).

`eval/anti_cheat.py` is no longer indefinitely deferred — sequenced as phase 2 of the active queue above. Under the 2026-04-27 scope expansion (option B), SOL Tier 4 (the full `reward_hack` detector set + active clock locking) lands in SOL integration phase 1, NOT anti-cheat phase 2. Anti-cheat phase 2 collapses to a thin orchestration layer over the already-wired SOL primitives: routes audit flags, finalizes the skeleton functions (`generate_randomized_inputs`, `strict_tolerance_check`), exposes the SOL `reward_hack` snapshot/check pair as a per-iteration context manager.

## Remaining (dependency-ordered)

Items marked `(skeleton)` have interfaces + placeholder logic that keeps the pipeline runnable. Items marked `(done)` have real implementations. Unmarked items need real implementation.

### Phase 1: Foundation

- [x] config.py (done) — `HardwareSpec` (SOLAR YAML schema), `load_hardware_spec()`, `load_config()`, `ACTSConfig`. `detect_hardware()` wires `torch.cuda.get_device_properties(0)` for runtime-knowable fields; per-precision tables still need the SOLAR arch YAML. `validate_hardware_spec()` catches wrong-YAML-vs-actual-GPU mismatches at config-load + pre-placeholder-substitution time (DRAM/SRAM/freq with 10% tolerance, warn-don't-raise).
- [x] kernels/kernel.py (done) — dataclasses complete
- [x] kernels/compiler.py (done) — file-backed importlib load (`spec_from_file_location` + `exec_module`), hash-keyed cache path, resolves `KernelSpec.entrypoint` via `getattr`. GPU-side Triton specialization still happens at launch time in correctness/benchmark runs.

### Phase 2: Evaluation Harness

- [x] eval/correctness.py (done) — 5-stage gate (smoke → shape-sweep → numerical stability → determinism → anti-cheat) with short-circuit failure attribution. Injectable `ComparisonPolicy` (torch-free at import); `TorchComparisonPolicy` delegates to `sol_execbench.compute_error_stats` when installed, falls back to `torch.allclose` otherwise.
- [x] eval/inputs.py (done) — `build_reference_fn` (exec PyTorch reference source, resolve `run`) + `build_input_generator` (wraps SOL's `gen_inputs` with seeding). Torch + sol_execbench lazy-imported.
- [x] eval/benchmark.py (done) — CUDA-event timing via injectable `BenchmarkTimer` Protocol; multi-workload parallel-list contract with fresh-timer-per-workload isolation; fail-closed on partial-workload failures (<half survive → `BenchmarkError`; `is_fully_successful` property on result); 100us sentinel on empty-workload path.
- [x] eval/profiler.py (done) — hybrid analytical roofline (required, fail-closed) + curated NCU subprocess (best-effort, degrades on failure). Representative workload per iteration; Phase C re-profiles the winner on every selected workload. Source-hash-keyed cache. Tier 1 fake-ncu unit tests + Tier 2 `@pytest.mark.gpu` real-GPU tests (`tests/test_profiler_gpu.py`). Per-iter signals feed the Reviewer; run-level classification comes from `classify_run` (see JOURNAL.md → "Bottleneck classify-once (2026-04-22)").
- [x] eval/roofline.py (done) — two clean paths: SOLAR (`derive_t_sol_from_solar` accepts `arch_yaml_path`, returns `RooflineResult(source="solar")`) or built-in `compute_roofline()` fallback (`source="builtin"`). `SolarResult.bottleneck` typed as `BottleneckType` enum (no string round-trip).
- [x] eval/scorer.py (done) — SOL Score with audit flags per SOL-ExecBench paper Section 4.3
- [ ] eval/anti_cheat.py (skeleton) — two surfaces: correctness-level (input randomization, precision checks) + performance-level (T_k < T_SOL flagging from scorer)

### Phase 3: Actions & Memory

- [x] memory/experience.py (done) — Experience dataclass
- [x] memory/store.py (done) — JSON persistence with save/load
- [x] memory/retriever.py (done) — scored retrieval: kernel-type + hardware filtering, bottleneck + success + speedup scoring, reserved failure slots. Pure Python, no GPU.
- [x] actions/registry.py (done) — registry + tier system
- [x] actions/tier1-6 (done) — action definitions + real `guidance` / `anti_patterns` / `expected_impact` text synthesized from the 9-paper KB + AccelOpt / Astra / autokernel / cuda-optimized-skill / evotoolkit catalogs (2026-04-27). `expected_impact` is qualitative-only and `anti_patterns` is sparse-but-grounded — both intentional, with re-open triggers in Backlog → "Action library KB refinement".

### Phase 4: Agents & Prompts

- [x] agents/llm_backend.py (done) — OpenAI Agents SDK integration: ModelConfig, create_model(), run_agent() with retry (narrow transient catch + exponential backoff w/ jitter), make_run_config()
- [x] prompts/planner/system.md (done) — bottleneck→technique mapping tables, gain ranges, anti-patterns, decision rules
- [x] prompts/planner/technique_select.md (done) — documents user prompt format
- [x] prompts/coder/ (done) — system.md (prescribed workflow, hard rules, one sanctioned failure mode) + implement.md (user-prompt format)
- [x] prompts/reviewer/ (done) — system.md (diagnostic reasoning) + interpret.md
- [x] agents/planner.py (done) — tool-using Agent with `submit_plan` tool (option α, mirrors Coder), Pydantic `PlannerOutput`, `build_user_prompt()`, `PlanningError`, technique validation, `max_turns=4`, `_make_submit_plan_tool` factory + `_validate_and_convert` helper, `has_model` property gated on `_SDK_AVAILABLE`. Static `parse_plan` removed (dead after submit-tool migration).
- [x] agents/coder.py (done) — tool-using Agent, Pydantic `KernelCodeOutput`, `ImplementationError`, `_max_turns = 2*config.max_debug_retries + 2` (= 8 by default; +2 over 2N covers the `submit_kernel` tool call + final plain-text confirmation), placeholder tools until compiler/correctness land
- [x] agents/reviewer.py (done) — tool-using Agent with `submit_review` tool (option α, mirrors Coder/Planner), Pydantic `ReviewerFeedbackOutput`, `build_user_prompt`, rule-based fallback (`degraded`/`error_reason` — tags expanded to include `max_turns_exceeded` and `missing_submit_review` alongside existing `llm_retries_exhausted`), configurable `prompt_dir`, `max_turns=4`, `has_model` property gated on `_SDK_AVAILABLE`. Static `parse_feedback` removed (dead after submit-tool migration).

### Phase 5: Search

- [x] search/tree.py (done) — tree state, path_to_node, checkpoint save/load (atomic). `TreeNode` carries `consecutive_agent_failures: int` (default 0); module constant `QUARANTINE_THRESHOLD = 2` defines the cutoff. `frontier()` excludes any node at or above the threshold so dead-weight branches stop attracting selection; `best_node()` intentionally still considers them (a quarantined parent may still hold the run's best score). Serialize/deserialize updated; legacy checkpoints default the counter to 0.
- [x] search/beam.py (done) — beam pruning (B3 quality-weighted + B2 diversity-aware, configurable), epsilon-greedy selection
- [x] search/orchestrator.py (done) — real control flow + real agents + real CUDA-event benchmarking + real analytical profiling. Fail-closed baseline check (aborts run on partial-workload failure); branch-local `DEAD_END` on child partial failure, profile failure, or missing representative latency. Post-refactor (2026-04-22): calls `classify_run` once after roofline resolution, threads `run_bottleneck` into retriever / planner / reviewer / `SearchResult`; commits `child.score` + `per_workload_latency_us` only after the profile DEAD_END gauntlet clears. Submit-tool migration phase (2026-04-26): added `PlanningError` catch site mirroring `ImplementationError`; both catches increment `parent.consecutive_agent_failures` (parent quarantine accounting); successful `tree.add_child(...)` resets the parent's counter to 0. Emits new `planner_failed` event alongside `coder_failed`.

### Phase 6: Pipeline & Integration

- [x] pipeline/optimize.py Phase A (done) — real two-path load, roofline, workload selection, model-configured `CoderAgent`, and fail-closed `generate_triton_baseline`. Phase B runs real CUDA-event benchmarking + real analytical profiling. Post-refactor (2026-04-22): placeholder hardware substitution also applies to caller-supplied zero-peak configs (not just `config is None`) via `dataclasses.replace`. V1-completion (2026-04-27): forwards `Path(config.arch_config_path)` to `derive_t_sol_from_solar`, picks median-workload as static-roofline representative, threads `roofline.source` through, and runs `validate_hardware_spec` before placeholder substitution.
- [x] pipeline/verify.py (done) — recompiles the winner and reruns the 5-stage correctness gate against the PyTorch reference; compile failures surface as `passed=False` with a compile-phrased detail string
- [x] pipeline/report.py (done) — `generate_report` + `render_report`; trace via `result.tree.path_to_node`; propagates `reward_hack_suspect` / `calibration_warning`; surfaces run-level `bottleneck` (from `SearchResult.run_bottleneck`) and `winner_per_workload_bottlenecks` (via `classify_workload` on every selected workload, fused with the Phase C re-profile pass)
- [x] benchmark/problem_loader.py (done)
- [x] benchmark/baseline_generator.py (done) — `generate_triton_baseline` drives `CoderAgent.translate` + post-verifies on every selected workload; `BaselineGenerationError` on no-model / retry exhaustion.
- [x] benchmark/workload_selector.py (done)
- [x] benchmark/solution_formatter.py (done)
- [x] benchmark/solar_adapter.py (done) — drives SOLAR's 4-stage Python pipeline (`PyTorchProcessor` → `PyTorchToEinsum` → `EinsumGraphAnalyzer` → `EinsumGraphPerfModel`). Bridge synthesizes a SOLAR-shaped `Model` from `Problem` + representative `Workload` (handles const + var + expr axes via fixed-point eval, 0-D tensors, int/bool dtypes). Arch resolution: explicit `arch_yaml_path` > SOLAR-bundled name (H100_PCIe, B200) > ACTS-supplied YAML (`_ACTS_ARCH_YAMLS`, includes `placeholder-RTX6000Ada` alias) > H100_PCIe with WARNING. `configs/arch/RTX6000Ada.yaml` hand-authored. Forward-only — backward-pass kernels deferred (see Backlog → "Backward-kernel SOLAR support").

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

- [ ] **Subprocess-isolated evaluation (SOL Tier 8 — deferred 2026-04-27)** —
  adopt `sol_execbench.driver.problem_packager.ProblemPackager` +
  `sol_execbench.driver.templates.eval_driver._make_eval` +
  `sol_execbench.core.bench.utils.make_eval` so each candidate
  evaluation runs in a fresh subprocess. Pulls in
  `core/utils.py::redirect_stdio_to_file` + `flush_stdio_streams`
  and `core/bench/utils.py::_read_log_file` for stdout / stderr
  capture. Briefly in active scope under the 2026-04-27 SOL
  integration scope expansion; deferred same-day after the
  inline-vs-subprocess functional-parity analysis. Inline and
  subprocess produce identical results for ACTS's success path
  (compile, 5-stage gate, latency, score, NCU). Subprocess unlocks
  operational benefits — clean kernel-crash recovery (vs sticky CUDA
  context state inline), GPU-memory isolation (vs explicit
  `empty_cache` inline), cross-iteration global-state isolation
  (vs Tier 4's named-function-only checks), tampering robustness
  against state-based reward hacks (weakref caches, import-time
  hooks) that escape Tier 4. Cost is real: ~200–500ms per eval
  for fork + Python startup + `import torch / triton /
  sol_execbench`, ×~5–10 candidates × ~20–50 iters × dozens of
  problems per benchmarking sweep. For ACTS today (own LLM, bounded
  internal search, Triton-only, single-tenant dev box), Tier 4's
  in-process `reward_hack` detector set + the 5-stage gate cover the
  realistic threat surface; Triton-post-gate kernel crashes are rare
  enough that inline DEAD_END handling is functional.
  *Trigger A*: ACTS evaluates externally-sourced kernels (KernelBench
  external solutions, RL-discovered kernels, anything not generated
  by our own Coder) — the threat-model assumption of "well-prompted
  internal LLM, narrow API" no longer holds and isolation becomes
  load-bearing.
  *Trigger B*: ACTS runs on multi-tenant GPU hardware where another
  tenant could attempt cross-process tampering.
  *Trigger C*: A live run shows real, frequent kernel crashes that
  disrupt the orchestrator (>1% of evaluations, or any case where
  inline DEAD_END recovery requires manual intervention).
  See JOURNAL → "SOL integration scope refinement — Tier 8
  (subprocess) deferred (2026-04-27)" for the full rationale.

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

- [ ] **Test helper consolidation** — `_simulate_plan_submission`
  (`tests/test_planner.py`) and `_simulate_review_submission`
  (`tests/test_reviewer.py`) are near-identical (~30 lines each):
  both pull the `submit_*` tool out of the patched `Agent`, invoke it
  with a payload, and assert the captured output. Could share a
  parametrized helper in `tests/conftest.py` that takes the Pydantic
  Output class + module path and returns the simulator.
  *Trigger*: a third `_simulate_*_submission` helper appears (e.g., a
  fourth agent migrates to the submit-tool pattern), OR shared
  retry/captured-output assertions diverge between the two test files
  (the duplication itself is fine while only two copies exist).

- [ ] **Action library KB refinement** — initial action guidance text
  authored 2026-04-27 with intentional limitations recorded in JOURNAL
  → "Initial guidance authoring decisions (2026-04-27)". Two known
  gaps: (a) `expected_impact` is qualitative-only ("typically modest",
  "high-variance") instead of numeric ranges, because no real `T_SOL`
  data exists yet to calibrate (SOLAR adapter is still synthetic);
  (b) `anti_patterns` is sparse — populated only where upstream repos
  (AccelOpt / Astra / autokernel / cuda-optimized-skill / evotoolkit)
  gave explicit warnings, left empty otherwise. Hand-fabricating
  anti-patterns from imagined failures risked anchoring the Planner on
  non-existent hazards, so the deliberate choice was sparse-but-grounded.
  *Trigger*: (a) reopens once the SOLAR adapter lands and real T_SOL is
  available — at that point `expected_impact` ranges can be calibrated
  against observed `T_k / T_SOL` distributions across actions; (b)
  reopens once ≥10 live runs accumulate enough failed-kernel
  `Experience` records in `MemoryStore` to ground anti-patterns from
  actual ACTS failure modes (not from imagined ones).

- [ ] **Backward-kernel SOLAR support** — current `solar_adapter.py`
  bridge synthesizes a forward-only `Model` (`get_inputs()` only); SOLAR
  ships `BackwardProcessor` (`solar/graph/backward_processor.py`) which
  sets `requires_grad=True`, runs forward, calls `backward()`, and
  extracts the gradient graph for stages 2-4. Bridge would need two more
  synthesized helpers (`get_loss_fn`, `get_target` — SOLAR doesn't care
  about loss semantics, only that the gradient graph builds) and a
  branch in `derive_t_sol` selecting `BackwardProcessor` vs
  `PyTorchProcessor` for stage 1. Stages 2-4 + result parsing unchanged.
  Open schema decision: **how do we identify backward problems?** Two
  options — (A) parse `Problem.op_type` for `_backward` suffix (cheap, no
  schema change, risks convention drift); (B) add explicit
  `Problem.kind: Literal["forward","backward"]` field (cleaner, also
  needed by `eval/correctness.py` for grad-comparison and by the
  baseline generator for backward-Triton baselines). Recommend (B) —
  backward is a multi-module feature that touches schema + correctness +
  inputs + baseline generator + adapter, deserves its own brainstorming
  round rather than a one-off solar_adapter patch. *Trigger*: when ACTS
  first exercises a backward problem in SOL-ExecBench (likely the first
  attention-backward or layernorm-backward kernel).

- [ ] **`_SDK_AVAILABLE` patch fixture** — ~17 LLM-path tests across
  `tests/test_planner.py` + `tests/test_reviewer.py` repeat the same
  `with (patch("src.agents.{planner,reviewer}._SDK_AVAILABLE", True),
  patch(".Agent"), patch(".function_tool", side_effect=lambda f: f),
  ...)` block. A `@pytest.fixture` returning the patch bundle as a
  context manager would dedupe ~75 lines and remove a recurring
  miss-one-patch failure mode (forgetting `function_tool` in
  particular silently turns the tool into a `FunctionTool` wrapper
  that breaks the simulator).
  *Trigger*: when adding the next agent that needs the same
  SDK-availability mock (a third copy makes the fixture obviously
  worth it), OR if the boilerplate proves to mask test failures.

### Skipped (decisions, not tech debt)

- **Tier action files → YAML catalog**: `src/actions/tier{1..6}*.py` are
  mostly data (~280 LOC). Moving to YAML would trade away type-checking,
  IDE refactor support, and import-time error detection for slightly
  fewer lines. Only worth it if non-developers need to edit actions —
  which isn't the case. Keep as Python.
