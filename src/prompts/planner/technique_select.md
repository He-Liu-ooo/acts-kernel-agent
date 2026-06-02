# Technique Selection — User Prompt Format

The user prompt is assembled programmatically by `PlannerAgent.build_user_prompt()`. This file documents the expected sections. Each section is included only when data is available.

## Sections

```
## Current kernel
<kernel source code in a Python code block>

## Run context
- Bottleneck (this run): <memory_bound | compute_bound | balanced>

## Profiling summary
<key metrics from the profiler — pct_peak_*, arithmetic_intensity (MACs/byte, run-level invariant), NCU signals>

## Past optimization lessons
Below are past optimization lessons retrieved from similar kernels. Use them
as inspiration, not directives — the current kernel and profile take
precedence.

[L1] **<title>**  (scope: edge|run, speedup: <X>x, arch: <hardware_arch>)
<prose lesson body — no code>

Before:
```
<changed-region snippet from the slower kernel>
```

After:
```
<changed-region snippet from the faster kernel>
```

[L2] ...
(Block omitted entirely when no lessons are retrieved.)

## Available actions
- <action_id_1>
- <action_id_2>
- ...

## Search tree context
<iteration number, depth, parent performance — omitted on first iteration>

## Reviewer feedback
<reviewer's diagnosis and suggestions — omitted on first iteration>
```

`Run context` carries the once-per-run bottleneck (`classify_run`). It is stable across iterations because the problem, representative workload, and hardware do not change within a run — so the Planner can rely on it without having to re-derive it from the per-iter profiling metrics.

The `Past optimization lessons` block surfaces distilled (title + prose + before/after snippet) rows retrieved by `MemoryRetriever.sample(kernel_type, hardware_arch)`. The Retriever filters by kernel type, prefers same-arch rows, then samples `top_k` weighted by `speedup ** α` (with replacement). Two `scope` values: `edge` (per-iter improving edge captured by the producer) and `run` (cumulative baseline → best-of-run lesson at run end). No profile data — the Planner reasons against the live profile in the `Profiling summary` block above.
