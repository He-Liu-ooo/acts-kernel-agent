# Interpretation — User Prompt Format

The user prompt is assembled programmatically by `ReviewerAgent.build_user_prompt()`. This file documents the expected sections. Each optional section is included only when data is available.

## Sections

```
## Current kernel
<kernel source code in a Python code block; triple backticks in the source are escaped>

## Run context
- Bottleneck: <memory_bound | compute_bound | balanced>

## Profiling summary
### Analytical (roofline)
- arithmetic_intensity: <float, 3 decimals> FLOP/byte
- ridge_point: <float, 3 decimals> FLOP/byte
- achieved: <TFLOPS, 2 decimals> TFLOPS / <GB/s, 2 decimals> GB/s
- pct_peak: compute <pct, 1 decimal>% · bw <pct, 1 decimal>%

### NCU (curated)       # omitted when NCU degraded or unavailable
- sm_occupancy: <pct, 1 decimal>%
- l2_hit_rate: <pct, 1 decimal>%
- tensor_core_util: <pct, 1 decimal>%
- top stalls: <dominant> (<pct>%), <runner_up> (<pct>%)

[DEGRADED: NCU unavailable — reason=<slug>]   # only when profiling.degraded and ncu is None

## Scoring
- SOL score: <float, 3 decimals>
- Headroom: <float, 1 decimal>%

## Search tree context
<iteration depth, parent SOL score, sibling outcomes — omitted on first iteration>

## Knowledge base context
<retrieved Reviewer-KB entries (metric-triggered or action-triggered) — omitted when no KB wired up>
```

The `Run context` section carries the once-per-run bottleneck from `classify_run` — invariant per (problem, representative workload, hardware). It is deliberately separated from the `Profiling summary` block so the Reviewer never conflates per-iter runtime metrics with the ground-truth classification.

When the orchestrator calls `review(..., profiling=None)` (e.g., analytical failure prior to the reviewer stage, or a test harness that omits profiling), the Profiling summary section is replaced with `[no profiling data — profile_kernel did not run]` so the LLM can see the absence explicitly rather than inferring from silence.

## Notes

- **Section order is stable.** Kernel → Profiling → Scoring → Tree → KB. The system prompt assumes this order when it says "above" or "below".
- **Optional sections are dropped, not blanked.** Their absence means "no data", not "empty".
- **Prev SOL score is not in the prompt.** It is supplied to `review()` as a Python arg (`prev_sol_score`) and consumed by the rule-based fallback only. The LLM infers trajectory from tree context.
- **No raw hardware spec.** Hardware characteristics are baked into the profiling metrics and the pre-computed SOL score; the Reviewer reasons about the kernel relative to its ridge point, not about specific GPUs.

## Future sub-agent split

When the Reviewer splits into Compute-Reviewer and Memory-Reviewer (see JOURNAL "context-adaptive agent specialization"), each sub-agent will receive the same prompt structure but with a focused `system.md`. The user-prompt format in this file stays unchanged.
