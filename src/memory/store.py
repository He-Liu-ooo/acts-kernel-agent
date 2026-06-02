"""JSONL append-only storage backend for optimization memory.

One Experience per line. Append-only, crash-safe per-row (flush after each
``write``). See ``doc/specs/2026-05-24-optimization-memory-design.md`` §6
and §11.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from src.memory.experience import ActionRecord, Experience

logger = logging.getLogger(__name__)

KNOWN_VERSION = 1

# Canonical fields of a v1 row. Missing values are tolerated (defaulted)
# but warned once per missing-field-name per ``load()`` call. The
# ``schema_version`` field is excluded — it's the migration pivot itself,
# and silently treating its absence as v1 is the intended behaviour.
_CANONICAL_FIELDS = (
    "row_id",
    "kernel_type",
    "hardware_arch",
    "scope",
    "speedup",
    "action_applied",
    "title",
    "lesson",
    "snippet_before",
    "snippet_after",
    "provenance",
    "created_at",
)


class MemoryStore:
    """Append-only JSONL store for distilled optimization lessons."""

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._experiences: list[Experience] = []

    def load(self) -> None:
        """Read all rows into the in-memory cache. Idempotent.

        Missing file → empty store (no error). Tolerant of unknown / missing
        fields. Skips rows whose ``schema_version`` is greater than
        ``KNOWN_VERSION``. Skips malformed JSON lines. Each missing field
        name produces one warn per load, not one per occurrence.
        """
        self._experiences = []
        if not self._store_path.exists():
            return
        warned_missing: set[str] = set()
        warned_future_version = False
        for lineno, raw in enumerate(self._store_path.read_text().splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "skipping malformed JSON line %d in %s: %s",
                    lineno,
                    self._store_path,
                    exc,
                )
                continue
            if not isinstance(d, dict):
                logger.warning(
                    "skipping line %d in %s: expected a JSON object, got %s",
                    lineno,
                    self._store_path,
                    type(d).__name__,
                )
                continue
            # ``int(...)`` coerces a stringly-typed ``"1"`` and rejects
            # ``null`` / ``"v2"`` cleanly. Without this coerce, a row with
            # ``"schema_version": "2"`` raises TypeError on the ``>``
            # comparison below — escaping ``load()`` entirely (the
            # surrounding except only catches ``json.JSONDecodeError``)
            # and aborting the whole store read, defeating the documented
            # "skip rows with schema_version > KNOWN_VERSION" tolerance.
            try:
                version = int(d.get("schema_version", KNOWN_VERSION))
            except (TypeError, ValueError):
                logger.warning(
                    "skipping line %d in %s: non-integer schema_version=%r",
                    lineno,
                    self._store_path,
                    d.get("schema_version"),
                )
                continue
            if version > KNOWN_VERSION:
                if not warned_future_version:
                    logger.warning(
                        "skipping row with schema_version=%d (this binary knows up to %d) in %s",
                        version,
                        KNOWN_VERSION,
                        self._store_path,
                    )
                    warned_future_version = True
                continue
            for required in _CANONICAL_FIELDS:
                if required not in d and required not in warned_missing:
                    logger.warning(
                        "row in %s missing field %r — defaulting; suppressing further warnings for this field",
                        self._store_path,
                        required,
                    )
                    warned_missing.add(required)
            # A non-finite-or-non-positive ``speedup`` (NaN / inf / 0 /
            # negative) decodes as valid JSON but later detonates the
            # retriever: ``random.choices()`` raises "Total of weights must
            # be finite". Reject it here. ``bool`` is excluded because
            # ``isinstance(True, int)`` is True and a bool speedup is junk.
            sp = d.get("speedup")
            if (
                not isinstance(sp, (int, float))
                or isinstance(sp, bool)
                or not math.isfinite(sp)
                or sp <= 0
            ):
                logger.warning(
                    "skipping row at line %d in %s: invalid speedup=%r",
                    lineno,
                    self._store_path,
                    sp,
                )
                continue
            # A non-string ``title`` / ``lesson`` decodes cleanly but later
            # crashes the planner's ``_neutralize_prompt_markdown``, which
            # calls ``.splitlines()`` on these fields.
            if not isinstance(d.get("title"), str) or not isinstance(
                d.get("lesson"), str
            ):
                logger.warning(
                    "skipping row at line %d in %s: non-string title/lesson",
                    lineno,
                    self._store_path,
                )
                continue
            try:
                self._experiences.append(_row_to_experience(d))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "skipping unparseable row at line %d in %s: %s",
                    lineno,
                    self._store_path,
                    exc,
                )

    def add(self, experience: Experience) -> None:
        """Append a single row to the JSONL file. Crash-safe per row."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_experience_to_dict(experience))
        with self._store_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
        self._experiences.append(experience)

    def add_many(self, experiences: Iterable[Experience]) -> None:
        """Append multiple rows in one file open/close. Flush per row.

        The in-memory cache is extended row-by-row INSIDE the write loop
        so that a mid-batch IOError leaves the cache consistent with the
        bytes that actually reached disk. Extending only after the loop
        (the prior shape) would lose track of rows partially written on
        disk on a crash, silently desyncing ``all()`` from the on-disk
        state.
        """
        rows = list(experiences)
        if not rows:
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        with self._store_path.open("a", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(_experience_to_dict(e)) + "\n")
                f.flush()
                self._experiences.append(e)

    def all(self) -> list[Experience]:
        """Return a copy of the in-memory cache."""
        return list(self._experiences)


def _experience_to_dict(exp: Experience) -> dict:
    return asdict(exp)


def _row_to_experience(d: dict) -> Experience:
    # ``action_applied`` is ``None`` for ``scope == "run"`` rows (G3
    # cumulative lessons have no single applied action). An explicit
    # ``null`` in the JSON or a missing key both load as ``None``;
    # the missing-key warn already fires in ``load()`` via
    # ``_CANONICAL_FIELDS``, so no extra signaling here.
    action_raw = d.get("action_applied")
    if action_raw is None:
        action: ActionRecord | None = None
    else:
        action = ActionRecord(
            action_id=action_raw.get("action_id", ""),
            tier=int(action_raw.get("tier", 0)),
            name=action_raw.get("name", ""),
            parameters=action_raw.get("parameters", {}) or {},
        )
    return Experience(
        row_id=d.get("row_id", ""),
        schema_version=d.get("schema_version", KNOWN_VERSION),
        kernel_type=d.get("kernel_type", ""),
        hardware_arch=d.get("hardware_arch", ""),
        scope=d.get("scope", "edge"),
        speedup=float(d.get("speedup", 1.0)),
        action_applied=action,
        title=d.get("title", ""),
        lesson=d.get("lesson", ""),
        snippet_before=d.get("snippet_before", ""),
        snippet_after=d.get("snippet_after", ""),
        provenance=d.get("provenance", {}) or {},
        created_at=d.get("created_at", ""),
    )
