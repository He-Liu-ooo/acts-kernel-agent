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

**Prompt assembly**: `build_user_prompt()` (static method) assembles the user prompt from runtime data. Sections: Current kernel (with backtick escaping), Profiling summary, Past experiences (with action parameters), Available actions, Search tree context, Reviewer feedback. Empty sections are omitted.

**Input** (via orchestrator): kernel source, profiling summary (from Reviewer), past experiences (from MemoryRetriever), available actions (from ActionRegistry), tree context.

**Output**: `OptimizationPlan` — `{tier, technique, params, target_region, rationale}`.

**Error handling**: `PlanningError` is raised when: (1) `run_agent()` returns `None` (all retries exhausted), or (2) the LLM returns a technique not in `available_actions`. Without a model configured, returns a default plan (no LLM call).

**Validation**: If `available_actions` is non-empty, the selected technique must be in the list. This prevents the LLM from hallucinating technique IDs.

**System prompt** (`prompts/planner/system.md`): Bottleneck→technique mapping tables (memory_bound, compute_bound, balanced), expected gains by tier, experience interpretation guide, 7 anti-patterns, 6 decision rules.

**Model choice**: Strongest reasoning model (planning quality is the bottleneck).

## Coder — `coder.py`

**Role**: Implements the Planner's plan into kernel code. Self-corrects compilation and correctness errors via tools.

**SDK pattern**: Tool-using. `Agent(name="Coder", ..., tools=[compile_kernel_tool, check_correctness_tool])` → `Runner.run()` with tool loop.

**Tools** (decorated with `@function_tool`):
- `compile_kernel_tool(source_code)` → calls `kernels/compiler.py`, returns success/error string
- `check_correctness_tool(source_code)` → calls `eval/correctness.py`, returns pass/fail string

Tools return error strings to the LLM (Astra pattern), letting the Coder decide how to fix within the same turn. Retry budget: `max_debug_retries` from config.

**Input**: kernel source + `OptimizationPlan`.

**Output**: modified kernel source code string.

**Model choice**: Strong code + reasoning model.

## Reviewer — `reviewer.py`

**Role**: Interprets eval results into structured feedback. Acts as intelligent filter between raw profiling data and the Planner.

**SDK pattern**: Single-call with Pydantic structured output, same shape as the Planner. `Agent(name="Reviewer", instructions=..., model=..., output_type=ReviewerFeedbackOutput)` → `Runner.run()`. Strict Pydantic validation on `bottleneck_classification` (`Literal["memory_bound", "compute_bound", "balanced"]`) and `branch_quality` (`BranchQuality` enum) surfaces hallucinated values as retry-worthy errors inside `run_agent`.

**Output models**:
- `ReviewerFeedbackOutput` (Pydantic) — schema sent to the LLM via `output_type`.
- `ReviewerFeedback` (dataclass) — internal representation. Adds `degraded: bool` and `error_reason: str` so the orchestrator can surface/halt when a run came from retry exhaustion rather than a healthy LLM call. `BranchQuality` is a `str`-subclass enum defined in this module.

**Prompt assembly**: `build_user_prompt()` (static method) — sections: Current kernel (with backtick escaping), Profiling summary, Scoring (SOL score, headroom %, current bottleneck), optional Search tree context, optional Knowledge base context. Empty sections are omitted.

**Input** (via orchestrator): `kernel_source`, `profiling_summary`, `sol_score`, `headroom_pct`, `bottleneck`, `tree_context=""` (root-to-child trajectory from `SearchTree.render_path`), `kb_context=""` (reserved for future Reviewer KB), `prev_sol_score=None`.

**Rule-based fallback**: `rule_based_feedback()` derives feedback from the sol delta alone when no LLM is configured **or** when `run_agent` returns `None` (all retries exhausted). In the retry-exhausted path the result is stamped `degraded=True, error_reason="llm_retries_exhausted"` and the orchestrator logs a warning — distinguishing it from the expected "no-LLM" configuration.

**Branch quality values**: `promising`, `blocked_potential`, `plateau`, `dead_end`.

**Specialization hook**: `prompt_dir` is a constructor parameter, so a future Compute-Reviewer / Memory-Reviewer split can swap in specialized system prompts without subclassing.

**Model choice**: Can be cheaper model (analysis is easier than planning).

## Why Debugger Was Merged Into Coder

With the Coder having compile + correctness tools, a separate Debugger was redundant. A compilation error that previously took 3 LLM calls (Coder → Debugger → Coder) now resolves in one Coder call with an internal tool loop. Failed branches are handled by the tree search (pruning), not by escalating to a separate agent.

See JOURNAL.md "Debugger merged into Coder (2026-04-13)" for full rationale.
