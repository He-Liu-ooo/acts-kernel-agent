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
