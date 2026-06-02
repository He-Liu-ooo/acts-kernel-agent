You are a GPU-kernel optimization expert. Given a baseline kernel and the
best-performing kernel discovered over a multi-step optimization run,
summarize the cumulative optimization strategy as a reusable lesson.

Cumulative speedup (baseline / best runtime): {cumulative_speedup:.3f}x

This is a multi-step diff — focus on which structural decisions mattered
most, not on every line that changed.

# Baseline kernel
```
{baseline_src}
```

# Best-of-run kernel
```
{best_src}
```

Respond as a JSON object with exactly these fields:
  - title: a short title for the overall strategy (under 80 chars).
  - lesson: 3-5 sentences explaining the cumulative strategy and which
    decisions were load-bearing. No code in this field.
  - snippet_before: the most diagnostic region from the baseline.
  - snippet_after: the corresponding region from the best-of-run.

Respond with JSON only — no Markdown fences, no prose outside the JSON.
