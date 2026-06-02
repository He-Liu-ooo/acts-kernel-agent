You are a GPU-kernel optimization expert. Given a slower kernel and a faster
kernel that differ by a single optimization, summarize the change as a
reusable one-step lesson.

Speedup (slow / fast runtime): {speedup:.3f}x
Action applied: {action_name} (tier {action_tier})

# Slow kernel
```
{parent_src}
```

# Fast kernel
```
{child_src}
```

Respond as a JSON object with exactly these fields:
  - title: a short title for the optimization (under 80 chars).
  - lesson: 2-4 sentences explaining what changed and why it's faster. No code.
  - snippet_before: the changed region from the slow kernel only (not the whole file).
  - snippet_after: the changed region from the fast kernel only (not the whole file).

If the two kernels are identical or the difference is trivial, set title to
exactly "No optimization found" and leave the other fields empty.
Respond with JSON only — no Markdown fences, no prose outside the JSON.
