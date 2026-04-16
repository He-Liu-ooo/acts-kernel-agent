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

**SDK pattern**: Single-call. `Agent(name="Planner", instructions=..., model=...)` → `Runner.run()`.

**Input** (via orchestrator): kernel source, profiling summary (from Reviewer), past experiences (from MemoryRetriever), available actions (from ActionRegistry), tree context.

**Output**: `OptimizationPlan` — `{tier, technique, params, target_region, rationale}`.

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

## Reviewer (Evaluator) — `evaluator.py`

**Role**: Interprets eval results into structured feedback. Acts as intelligent filter between raw profiling data and the Planner.

**SDK pattern**: Single-call. `Agent(name="Reviewer", instructions=..., model=...)` → `Runner.run()`.

**Input**: kernel source, profiling summary, SOL score, headroom %, bottleneck classification.

**Output**: `ReviewerFeedback` — `{outcome, metric_deltas, bottleneck_classification, bottleneck_diagnosis, suggestions, branch_quality, conditional_assessment}`.

**Branch quality values**: `promising`, `blocked_potential`, `plateau`, `dead_end`.

**Model choice**: Can be cheaper model (analysis is easier than planning).

## Why Debugger Was Merged Into Coder

With the Coder having compile + correctness tools, a separate Debugger was redundant. A compilation error that previously took 3 LLM calls (Coder → Debugger → Coder) now resolves in one Coder call with an internal tool loop. Failed branches are handled by the tree search (pruning), not by escalating to a separate agent.

See JOURNAL.md "Debugger merged into Coder (2026-04-13)" for full rationale.
