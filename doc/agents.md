# Agents — `src/agents/`

3 LLM agents built on the OpenAI Agents SDK. No separate Debugger — the Coder handles self-correction via tools.

## Architecture

```
Planner (single-call, no tools)
    → Coder (tool-using: compile + correctness)
        → [orchestrator-side eval]
            → Reviewer (single-call, no tools)
```

The deterministic orchestrator controls all flow. Agents are stateless — they receive context, make one LLM call (or one tool-loop for Coder), and return structured output.

## Planner — `planner.py`

**Role**: Analyzes profiling data + optimization memory, selects technique from action library, produces structured plan.

**SDK pattern**: Single-call with Pydantic structured output. `Agent(name="Planner", instructions=..., model=..., output_type=OptimizationPlanOutput)` → `Runner.run()`. The SDK enforces the output schema — the LLM must return valid JSON matching `OptimizationPlanOutput`.

**Output models**:
- `OptimizationPlanOutput` (Pydantic) — schema sent to the LLM via `output_type`. Fields: `tier` (int), `technique` (str), `params` (dict), `target_region` (str), `rationale` (str).
- `OptimizationPlan` (dataclass) — internal representation used by the rest of the codebase. Converted from `OptimizationPlanOutput` via `_output_to_plan()`.

**Prompt assembly**: `build_user_prompt()` (static method) assembles the user prompt from runtime data. Sections: Current kernel (with backtick escaping), **Run context** (once-per-run bottleneck via shared `render_run_context()` helper — omitted when no `bottleneck` is supplied), Profiling summary, Past experiences (with action parameters), Available actions, Search tree context, Reviewer feedback. Empty sections are omitted.

**Input** (via orchestrator): kernel source, profiling summary (from Reviewer), past experiences (from MemoryRetriever), available actions (from ActionRegistry), tree context, `bottleneck: BottleneckType | None` (the run-level classification; threaded verbatim from `SearchResult.run_bottleneck`).

**Output**: `OptimizationPlan` — `{tier, technique, params, target_region, rationale}`.

**Error handling**: `PlanningError` is raised when: (1) `run_agent()` returns `None` (all retries exhausted), or (2) the LLM returns a technique not in `available_actions`. Without a model configured, returns a default plan (no LLM call).

**Validation**: If `available_actions` is non-empty, the selected technique must be in the list. This prevents the LLM from hallucinating technique IDs.

**System prompt** (`prompts/planner/system.md`): Bottleneck→technique mapping tables (memory_bound, compute_bound, balanced), expected gains by tier, experience interpretation guide, 7 anti-patterns, 6 decision rules.

**Model choice**: Strongest reasoning model (planning quality is the bottleneck).

## Coder — `coder.py`

**Role**: Implements the Planner's plan into kernel code. Self-corrects compilation and correctness errors via tools, then emits its final answer via a submit-tool call.

**SDK pattern**: Tool-using, no `output_type=`. `Agent(name="Coder", instructions=..., model=..., tools=[compile_kernel_tool, check_correctness_tool, submit_kernel])` → `Runner.run()` with an internal tool loop bounded by `max_turns`. The submit tool replaces the `output_type=Pydantic` enforcement that the SDK used to translate to a `response_format=json_schema` API field — that field is rejected by reasoning-model providers (DeepSeek-reasoner, etc.), so the Coder routes its structured submission through a tool call instead. Tool-call schemas are universally supported across OpenAI-compatible providers; same Pydantic validator runs inside the tool body, preserving T4's "validation failure → in-loop tool retry" guarantee verbatim.

**Output model**:
- `KernelCodeOutput` (Pydantic) — `source_code: str` plus `triton_kernel_name: str`, with a `@model_validator(mode="after")` that pulls every `@triton.jit def <name>` out of `source_code` and asserts the declared name is one of them. Validation runs inside `submit_kernel`'s tool body; on failure the tool returns the validator's error string and the SDK hands it back to the LLM as the tool-call response, prompting an in-loop retry within the existing turn budget. The `triton_kernel_name` field is consumed downstream by `eval/profiler.py` as the highest-priority source for NCU's `--kernel-name regex:` filter — explicit declaration removes the silent mis-profiling failure mode that source-regex extraction had on fused kernels with multiple `@triton.jit` defs.

**Tools** (built per call via `_make_compile_tool` / `_make_correctness_tool` / `_make_submit_tool`, then wrapped with `@function_tool` at Agent-construction time so the factories stay unit-testable without the SDK):
- `compile_kernel_tool(source_code)` → calls `kernels/compiler.py::compile_kernel`. Success returns an entrypoint confirmation; failure returns the full compiler traceback so the Coder can read the error and fix it.
- `check_correctness_tool(source_code)` → recompiles the candidate (compile is cheap) and runs `eval/correctness.py::verify_correctness` against **every** input generator bound at call time, short-circuiting on the first failure. Returns a human-readable pass/fail message that names the failing workload index and stage. Compile failures are surfaced before attempting correctness so the Coder gets the cheaper error first.
- `submit_kernel(source_code, triton_kernel_name)` → instantiates `KernelCodeOutput(...)` (triggers the cross-field validator) and stores the validated output in a per-call captured dict. Success returns a sentinel string instructing the LLM to emit a one-word confirmation so the SDK loop terminates; validation failure returns the error string for an in-loop retry. After `_run_tool_agent` returns, an empty captured dict raises `ImplementationError("Coder did not call submit_kernel ...")` — fail-loud rather than silently treating a missing submission as a degraded best-effort.

**Per-call binding**: both tools are closures over `(kernel_spec, reference_fn, input_generators)` captured when `implement()` or `translate()` is invoked. A fresh `Agent` is constructed per call — cheap (object construction, no network) and keeps the oracle bound to the right problem.

**Prompt assembly**: `build_user_prompt()` (static method) assembles the user prompt from the kernel source and the `OptimizationPlan`. Sections: Current kernel (via shared `render_kernel_section()` helper — backticks escaped), Optimization plan (tier, technique, optional params, target region, rationale). **Reviewer feedback is intentionally not included** — the Planner has already consumed it and distilled its conclusions into the plan, so the Coder works from the plan only.

**Turn budget**: `self._max_turns = 2 * config.max_debug_retries + 2` — derived at construction time from `ACTSConfig`. Derivation: `max_debug_retries` compile+correctness tries × 2 tool turns per cycle + 1 `submit_kernel` tool call + 1 final plain-text confirmation. Default config (`max_debug_retries=3`) gives 8. The one sanctioned failure mode, spelled out in `system.md`, is calling `submit_kernel` with the last version that compiled cleanly when the budget runs out without a green correctness run.

**`has_model` property**: `True` when the agent is backed by a real LLM. Callers that must branch before reaching into internals (e.g., `baseline_generator` fail-closing without a model) use this instead of touching `_model` directly.

**Entry points**:

`implement(kernel_source, plan, *, kernel_spec, reference_fn, input_generators) -> KernelCodeOutput` — the per-iteration call from the orchestrator. Applies the plan to the current kernel; `kernel_spec` + `reference_fn` + `input_generators` (one generator per selected workload) are jointly required when a model is configured — all three are captured by the tool closures. Without a model configured, returns a `KernelCodeOutput.model_construct(source_code=<unchanged>, triton_kernel_name="")` stub (validation skipped — no LLM produced the source, so the empty kernel name signals to the profiler to use its regex fallback).

`translate(*, reference_source, kernel_spec, reference_fn, input_generators) -> KernelCodeOutput` — one-shot PyTorch→Triton port used at problem-load time by `benchmark/baseline_generator.py`. Drives the same tool-loop as `implement()` (shared via `_run_tool_agent`) but under a dedicated system prompt (`prompts/coder/translate.md`) that emphasizes signature invariance and no precision drop. Callers post-verify after translation because the SDK may emit a degraded best-effort when the turn budget is exhausted. No no-op fallback — raises `ImplementationError` when no model is configured, since there is no sensible from-scratch port without an LLM.

**Translate prompt assembly**: `build_translate_prompt(reference_source, kernel_spec)` is a separate static method (backticks in the reference are escaped, target kernel name/entrypoint/kernel-type are surfaced) so the PyTorch→Triton port prompt is distinct from the per-iteration `implement()` prompt format.

**Output**: `KernelCodeOutput` carrying both `source_code` and the validated `triton_kernel_name`; downstream callers (orchestrator, baseline_generator) thread both fields into the new `Kernel`. **May be a degraded best-effort** when the SDK tool loop exhausts `max_turns` without a green correctness run — downstream verification/scoring (or, for `translate()`, the caller's post-verify pass) handles that case.

**Error handling**: `ImplementationError` is raised when `run_agent()` returns `None` (transient retry exhaustion), when `translate()` is called without a model, or when `implement()` is called with a model but missing correctness context (`kernel_spec` / `reference_fn` / a non-empty `input_generators`). **Deferred**: orchestrator-side handling of `ImplementationError` and SDK `MaxTurnsExceeded` — see `PROCESS.md` → Deferred Improvements.

**Temperature**: 0.0. Determinism is load-bearing for code generation — variance in kernel code is almost always noise, not creativity. Planner/Reviewer use 0.3 for exactly the opposite reason (see Planner / Reviewer sections).

**System prompt** (`prompts/coder/system.md`): prescribed 6-step workflow (apply change → compile → correctness → submit → confirm), hard rules (signature invariance, `triton_kernel_name` matches source, one focused change, no benchmarking, no bypassing correctness, no invented APIs, no precision drop, no stray imports), anti-patterns (rewrites, correctness-before-compile, snippets, prose in submitted source, multi-change after failure, calling more tools after submit). The hard rule on correctness bypassing explicitly defines the single legal failure-mode submission so the prompt and the `KernelCodeOutput` schema never contradict each other.

**Model choice**: Strong code + reasoning model.

## Reviewer — `reviewer.py`

**Role**: Interprets eval results into structured feedback. Acts as intelligent filter between raw profiling data and the Planner.

**SDK pattern**: Single-call with Pydantic structured output, same shape as the Planner. `Agent(name="Reviewer", instructions=..., model=..., output_type=ReviewerFeedbackOutput)` → `Runner.run()`. Strict Pydantic validation on `bottleneck_classification` (`Literal["memory_bound", "compute_bound", "balanced"]`) and `branch_quality` (`BranchQuality` enum) surfaces hallucinated values as retry-worthy errors inside `run_agent`.

**Output models**:
- `ReviewerFeedbackOutput` (Pydantic) — schema sent to the LLM via `output_type`.
- `ReviewerFeedback` (dataclass) — internal representation. Adds `degraded: bool` and `error_reason: str` so the orchestrator can surface/halt when a run came from retry exhaustion rather than a healthy LLM call. `BranchQuality` is a `str`-subclass enum defined in this module.

**Prompt assembly**: `build_user_prompt()` (static method) — sections: Current kernel (with backtick escaping), **Run context** (once-per-run bottleneck via shared `render_run_context()` helper — always present for the Reviewer, since it's invoked only after the orchestrator has a `run_bottleneck`), Profiling summary, Scoring (SOL score, headroom %), optional Search tree context, optional Knowledge base context. Empty sections are omitted.

**Input** (via orchestrator): `kernel_source`, `profiling_summary`, `sol_score`, `headroom_pct`, `bottleneck: BottleneckType` (once-per-run classification from `classify_run`), `tree_context=""` (root-to-child trajectory from `SearchTree.render_path`), `kb_context=""` (reserved for future Reviewer KB), `prev_sol_score=None`, `profiling: ProfilingResult | None = None` (renders the analytical + NCU blocks in the profiling summary when supplied).

**Rule-based fallback**: `rule_based_feedback()` derives feedback from the sol delta alone when no LLM is configured **or** when `run_agent` returns `None` (all retries exhausted). In the retry-exhausted path the result is stamped `degraded=True, error_reason="llm_retries_exhausted"` and the orchestrator logs a warning — distinguishing it from the expected "no-LLM" configuration.

**Branch quality values**: `promising`, `blocked_potential`, `plateau`, `dead_end`.

**Specialization hook**: `prompt_dir` is a constructor parameter, so a future Compute-Reviewer / Memory-Reviewer split can swap in specialized system prompts without subclassing.

**Model choice**: Can be cheaper model (analysis is easier than planning).

## Why Debugger Was Merged Into Coder

With the Coder having compile + correctness tools, a separate Debugger was redundant. A compilation error that previously took 3 LLM calls (Coder → Debugger → Coder) now resolves in one Coder call with an internal tool loop. Failed branches are handled by the tree search (pruning), not by escalating to a separate agent.

See JOURNAL.md "Debugger merged into Coder (2026-04-13)" for full rationale.
