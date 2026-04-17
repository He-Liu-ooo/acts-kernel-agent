# Interpretation — User Prompt Format

The user prompt is assembled programmatically by `ReviewerAgent.build_user_prompt()`. This file documents the expected sections. Each optional section is included only when data is available.

## Sections

```
## Current kernel
<kernel source code in a Python code block; triple backticks in the source are escaped>

## Profiling summary
<NCU-derived metrics and latencies — passed through verbatim from the eval harness>

## Scoring
- SOL score: <float, 3 decimals>
- Headroom: <float, 1 decimal>%
- Current bottleneck: <memory_bound | compute_bound | balanced>

## Search tree context
<iteration depth, parent SOL score, sibling outcomes — omitted on first iteration>

## Knowledge base context
<retrieved Reviewer-KB entries (metric-triggered or action-triggered) — omitted when no KB wired up>
```

## Notes

- **Section order is stable.** Kernel → Profiling → Scoring → Tree → KB. The system prompt assumes this order when it says "above" or "below".
- **Optional sections are dropped, not blanked.** Their absence means "no data", not "empty".
- **Prev SOL score is not in the prompt.** It is supplied to `review()` as a Python arg (`prev_sol_score`) and consumed by the rule-based fallback only. The LLM infers trajectory from tree context.
- **No raw hardware spec.** Hardware characteristics are baked into the profiling metrics and the pre-computed SOL score; the Reviewer reasons about the kernel relative to its ridge point, not about specific GPUs.

## Future sub-agent split

When the Reviewer splits into Compute-Reviewer and Memory-Reviewer (see JOURNAL "context-adaptive agent specialization"), each sub-agent will receive the same prompt structure but with a focused `system.md`. The user-prompt format in this file stays unchanged.
