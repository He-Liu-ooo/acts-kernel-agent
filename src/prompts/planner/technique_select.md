# Technique Selection — User Prompt Format

The user prompt is assembled programmatically by `PlannerAgent.build_user_prompt()`. This file documents the expected sections. Each section is included only when data is available.

## Sections

```
## Current kernel
<kernel source code in a Python code block>

## Profiling summary
<bottleneck classification and key metrics from the profiler/reviewer>

## Past experiences
- <action_name> (tier <N>) [<param>=<val>, ...]: <success|failure>, speedup <X>x, bottleneck <before> -> <after>
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
