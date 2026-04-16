# Memory — `src/memory/`

Persistent cross-task optimization memory. Stores distilled experiences (no kernel code), both successes and failures.

## Experience — `experience.py`

Dataclass recording what was tried, what happened, and on what hardware.

| Field | Type | Description |
|-------|------|-------------|
| `kernel_type` | str | Kernel archetype (e.g., "matmul") |
| `action_applied` | ActionRecord | Action ID, tier, name, parameters |
| `metrics` | dict | latency, sol_score |
| `speedup` | float | Baseline / candidate latency |
| `reviewer_summary` | str | Reviewer's distilled feedback |
| `bottleneck_before` | str | Bottleneck before action |
| `bottleneck_after` | str | Bottleneck after action |
| `hardware` | str | GPU name |
| `success` | bool | Whether the action improved performance |

No kernel code stored — only summaries.

## MemoryStore — `store.py`

JSON file backend. Simple, git-friendly, human-readable.

- `load()`: Read experiences from disk.
- `save()`: Persist to disk.
- `add(experience)`: Append and persist.
- `all() -> list[Experience]`: Return all stored.

This is real implemented logic (JSON serialization via `dataclasses.asdict`).

## MemoryRetriever — `retriever.py`

Retrieves relevant past experiences for the Planner.

Strategy:
1. Filter by `kernel_type`
2. Rank by bottleneck relevance (exact match first)
3. Select top-K (configured by `optimization_memory_top_k`, default 5)

Injected into Planner only — not Coder (has the plan), not Reviewer (evaluates independently).

This is real implemented logic.
