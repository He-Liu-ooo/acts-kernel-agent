# Technique Selection — User Prompt Format

The user prompt is assembled programmatically by `PlannerAgent.build_user_prompt()`. This file documents the expected sections. Each section is included only when data is available.

## Sections

```
## Current kernel
<kernel source code in a Python code block>

## Run context
- Bottleneck: <memory_bound | compute_bound | balanced>

## Profiling summary
<key metrics from the profiler — pct_peak_*, arithmetic_intensity, NCU signals>

## Past experiences
- <action_name> (tier <N>) [<param>=<val>, ...]: <success|failure>, speedup <X>x, bottleneck_before <label>
- ...
(Parameters are included when present, omitted when empty)

## Available actions
- <action_id_1>
- <action_id_2>
- ...

## Search tree context
<iteration number, depth, parent performance — omitted on first iteration>

## Reviewer feedback
<reviewer's diagnosis and suggestions — omitted on first iteration>
```

`Run context` carries the once-per-run bottleneck (`classify_run`). It is stable across iterations because the problem, representative workload, and hardware do not change within a run — so the Planner can rely on it without having to re-derive it from the per-iter profiling metrics. Past experiences no longer carry a `bottleneck_after` field (experiences store only the pre-iteration classification).
