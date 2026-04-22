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
| `bottleneck_before` | `BottleneckType` | Run-level classification (from `classify_run`) at the time the action was tried |
| `hardware` | str | GPU name |
| `success` | bool | Whether the action improved performance |

No kernel code stored — only summaries. There is no `bottleneck_after` — classification is invariant per `(problem, representative workload, hardware)` within a run, so a pre/post pair on the same experience would always hold the same value. See JOURNAL → "Bottleneck classify-once (2026-04-22)".

## MemoryStore — `store.py`

JSON file backend. Simple, git-friendly, human-readable.

- `load()`: Read experiences from disk.
- `save()`: Persist to disk.
- `add(experience)`: Append and persist.
- `all() -> list[Experience]`: Return all stored.

This is real implemented logic (JSON serialization via `dataclasses.asdict`).

## MemoryRetriever — `retriever.py`

Retrieves relevant past experiences for the Planner.

### Retrieval pipeline

1. **Filter by `kernel_type`** — exact match
2. **Hardware filter** — if `hardware` is provided, prefer same-hardware experiences. Falls back to cross-hardware if fewer than `top_k` same-hardware matches exist.
3. **Score and rank** — each experience gets a relevance score:
   - Bottleneck exact match: +10
   - Success: +3
   - Speedup: +min(speedup, 5.0)
   - Tiebreaker: speedup (higher first)
4. **Select top-K with reserved failure slots** — successes and failures are ranked independently, then merged:
   - `top_k >= 3`: reserves `max(1, top_k // 3)` slots for failures
   - `top_k == 2`: reserves 1 slot for failures (if both pools exist)
   - `top_k == 1`: no reservation — returns the single highest-scored experience
   - If either pool is empty, its slots are given to the other pool

### Interface

```python
MemoryRetriever(store: MemoryStore, top_k: int = 5)
    .retrieve(
        kernel_type: str,
        current_bottleneck: BottleneckType,
        hardware: str = "",
    ) -> list[Experience]
```

`hardware` is optional — the orchestrator currently omits it (skeleton phase). When the orchestrator gets its real implementation, it will pass the hardware identifier from config.

Injected into Planner only — not Coder (has the plan), not Reviewer (evaluates independently).
