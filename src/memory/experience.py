"""Experience dataclass for optimization memory.

A single distilled lesson produced by the Summarizer from one improving
(parent → child) edge in the search tree (``scope="edge"``) or from the
baseline → best-of-run pair at run end (``scope="run"``). See
``doc/memory.md`` and ``doc/specs/2026-05-24-optimization-memory-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ActionRecord:
    """Record of an action applied during optimization. Kept verbatim from v1."""

    action_id: str
    tier: int
    name: str
    parameters: dict[str, str] = field(default_factory=dict)


@dataclass
class Experience:
    """A single distilled optimization lesson stored in opt-mem.

    Produced by the Summarizer; consumed by the Planner via the Retriever.
    No profile data, no full kernel source — only the distilled lesson +
    the changed-region snippets.
    """

    row_id: str
    schema_version: int
    kernel_type: str
    hardware_arch: str
    scope: Literal["edge", "run"]
    speedup: float
    # ``None`` when ``scope == "run"`` — the cumulative G3 row distills a
    # multi-step trajectory; no single action was "applied." For
    # ``scope == "edge"`` rows the producer always populates a real
    # ``ActionRecord``. Invariant: ``scope == "run" ⇒ action_applied is None``.
    action_applied: ActionRecord | None
    title: str
    lesson: str
    snippet_before: str
    snippet_after: str
    provenance: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    # Deterministic applicability signature = run bottleneck + action params
    # (e.g. "compute_bound | BLOCK_N=32"); run-scope rows carry bottleneck
    # only. Keys dedup (same technique + same condition collapse; different
    # conditions are preserved) and is surfaced to the Planner as
    # "applies when: ...". See doc/specs/2026-06-02-optmem-dedup-design.md.
    condition: str = ""


def _format_condition(bottleneck, action: "ActionRecord | None") -> str:
    """Deterministic applicability signature: ``"<bottleneck> | k=v, k=v"``.

    ``bottleneck`` may be a ``BottleneckType`` (``.value``), a str, or None.
    Run-scope rows pass ``action=None`` → bottleneck only. Params are sorted
    for a stable key. Co-located with ``dedup_key``/``dedup_best`` because the
    condition is part of an Experience's identity; ``producer.py`` re-exports
    it. (Legacy-row backfill in ``store.py`` also calls this.)"""
    parts: list[str] = []
    if bottleneck:
        parts.append(getattr(bottleneck, "value", bottleneck))
    if action is not None and action.parameters:
        params = ", ".join(f"{k}={v}" for k, v in sorted(action.parameters.items()))
        parts.append(params)
    return " | ".join(parts)


def dedup_key(exp: "Experience") -> tuple:
    """Identity for dedup: same technique + same condition collapse.

    Run-scope rows (no action) use the sentinel ``"∅"`` for the action
    component so they key only on (kernel, arch, run, condition=bottleneck)."""
    action_id = exp.action_applied.action_id if exp.action_applied is not None else "∅"
    return (exp.kernel_type, exp.hardware_arch, exp.scope, action_id, exp.condition)


def dedup_best(rows: list["Experience"]) -> list["Experience"]:
    """Collapse rows sharing a ``dedup_key`` to the highest-speedup row
    (ties → most recent ``created_at``). Preserves first-seen key order."""
    best: dict[tuple, "Experience"] = {}
    for e in rows:
        k = dedup_key(e)
        cur = best.get(k)
        if cur is None or (e.speedup, e.created_at) > (cur.speedup, cur.created_at):
            best[k] = e
    return list(best.values())
