# Implementation — User Prompt Format

The user prompt is assembled programmatically by `CoderAgent.build_user_prompt()` from the current kernel source and the Planner's `OptimizationPlan`. Reviewer feedback is **not** included — the Planner already consumes it and distills its conclusions into the plan.

## Sections

```
## Current kernel
<kernel source code in a Python code block; triple-backticks in the source are escaped>

## Optimization plan
- Tier: <N>
- Technique: <technique_id>
- Params: <param>=<val>, <param>=<val>, ...
  (omitted when plan.params is empty)
- Target region: <region string>
- Rationale: <1-2 sentences from the Planner>
```

## Field provenance

| Section             | Source                                          |
|---------------------|-------------------------------------------------|
| Current kernel      | `Kernel.source_code` of the parent tree node    |
| Tier                | `OptimizationPlan.tier`                         |
| Technique           | `OptimizationPlan.technique`                    |
| Params              | `OptimizationPlan.params` (rendered `k=v, ...`) |
| Target region       | `OptimizationPlan.target_region`                |
| Rationale           | `OptimizationPlan.rationale`                    |

## Output contract

The Coder's response is validated against the `KernelCodeOutput` Pydantic schema:

- `source_code` (str): the complete modified kernel source.

Schema violations raise inside `run_agent` and are retried transparently. If all retries are exhausted, `CoderAgent.implement()` raises `ImplementationError` — the orchestrator surfaces this as a dead branch.
