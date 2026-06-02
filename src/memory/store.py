"""JSONL storage backend for optimization memory.

One Experience per line, consolidated by dedup key on write (atomic
whole-file rewrite) and on load: keeps the highest-speedup row per
``(kernel, arch, scope, action, condition)`` (ties → most recent
``created_at``). No longer strictly append-only — each ``add``/``add_many``
merges then rewrites the whole file atomically (tmp + ``os.replace``), which
preserves crash-safety while collapsing redundant lessons. See
``doc/specs/2026-05-24-optimization-memory-design.md`` §6 and §11 and
``doc/specs/2026-06-02-optmem-dedup-design.md``. The write path re-reads
the on-disk store and merges before rewriting, so a write-only store (one
that never called ``load()``) cannot truncate prior lessons. Forward-compat
rows (``schema_version > KNOWN_VERSION``) this binary cannot parse are
carried through the rewrite verbatim, so an older binary cannot delete a
newer binary's lessons.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from src.memory.experience import (
    ActionRecord,
    Experience,
    _format_condition,
    dedup_best,
)

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
    """Dedup-consolidated JSONL store for distilled optimization lessons."""

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._experiences: list[Experience] = []

    def load(self) -> None:
        """Read all rows into the in-memory cache. Idempotent.

        Missing file → empty store (no error). Tolerant of unknown / missing
        fields. Does not surface rows whose ``schema_version`` is greater than
        ``KNOWN_VERSION`` (they stay on disk, carried through any later
        rewrite — see ``_parse``). Skips malformed JSON lines. Each missing
        field name produces one warn per load, not one per occurrence.
        """
        text = self._store_path.read_text() if self._store_path.exists() else ""
        exps, _ = self._parse(text)
        self._experiences = dedup_best(exps)

    def _parse(self, text: str) -> tuple[list[Experience], list[str]]:
        """Parse + validate raw JSONL ``text`` into valid ``Experience`` rows.

        Returns ``(experiences, passthrough_future_raw_lines)``. The second
        element holds the verbatim raw lines of rows skipped *only* for the
        forward-compat reason (valid JSON dict with ``schema_version >
        KNOWN_VERSION``); the write path carries these through the rewrite
        unchanged so an older binary cannot delete a newer binary's lessons.
        Malformed-JSON / non-dict / non-integer-schema_version / bad-speedup /
        bad-title-lesson / bad-identity rows are junk and are NOT preserved.

        Shared by ``load()`` and the write path so both apply the identical
        tolerant guards (malformed-JSON skip, non-dict skip, schema_version
        coerce/skip, missing-field warn, speedup guard, title/lesson
        type-guard, non-string identity-field skip, non-string condition drop,
        per-row parse try/except). Each missing-field name (and a future
        schema_version) warns once per call. Caller is responsible for any
        ``dedup_best`` consolidation.
        """
        parsed: list[Experience] = []
        passthrough: list[str] = []
        warned_missing: set[str] = set()
        warned_future_version = False
        for lineno, raw in enumerate(text.splitlines(), start=1):
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
            # comparison below — escaping the parse entirely (the
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
                        "preserving row with schema_version=%d (this binary knows up to %d) in %s through rewrite",
                        version,
                        KNOWN_VERSION,
                        self._store_path,
                    )
                    warned_future_version = True
                # Forward-compat: keep the verbatim raw line so the write path
                # carries it through the whole-file rewrite unchanged. This is
                # the ONLY skip reason that preserves the row — junk rows below
                # are dropped, not passed through.
                passthrough.append(raw)
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
            # A non-string dedup-key identity field (``kernel_type`` /
            # ``hardware_arch`` / ``scope``, or ``action_applied.action_id``)
            # makes ``dedup_key`` an unhashable tuple member and aborts the
            # entire load/merge. Unlike ``condition`` (recoverable → drop to
            # ""), a bad identity field is unrecoverable junk — skip the row.
            action_applied = d.get("action_applied")
            action_id = (
                action_applied.get("action_id")
                if isinstance(action_applied, dict)
                else action_applied
            )
            bad_identity = any(
                field in d and not isinstance(d[field], str)
                for field in ("kernel_type", "hardware_arch", "scope")
            ) or (action_applied is not None and not isinstance(action_id, str))
            if bad_identity:
                logger.warning(
                    "skipping row at line %d in %s: non-string dedup-key identity field",
                    lineno,
                    self._store_path,
                )
                continue
            # A non-string ``condition`` (hand-edited / malicious JSONL)
            # would reach ``dedup_key()`` as an unhashable member and abort
            # the read. Unlike title/lesson it is recoverable: drop the bad
            # value to "" and KEEP the row (condition is not
            # correctness-load-bearing). This guarantees ``dedup_key`` only
            # ever sees a hashable string.
            if "condition" in d and not isinstance(d["condition"], str):
                logger.warning(
                    "row at line %d in %s: non-string condition=%r — dropping to \"\"",
                    lineno,
                    self._store_path,
                    d["condition"],
                )
                d["condition"] = ""
            try:
                parsed.append(_row_to_experience(d))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "skipping unparseable row at line %d in %s: %s",
                    lineno,
                    self._store_path,
                    exc,
                )
        return parsed, passthrough

    def _rewrite(
        self, rows: list[Experience], passthrough_lines: Iterable[str] = ()
    ) -> None:
        """Atomically replace the JSONL with ``rows`` (write tmp + os.replace),
        preserving crash-safety now that writes are whole-file compactions.

        ``passthrough_lines`` are verbatim raw lines of forward-compat rows
        (``schema_version > KNOWN_VERSION``) that this binary cannot parse;
        they are written through unchanged so the rewrite cannot delete a
        newer binary's lessons.

        The temp file carries a per-pid suffix so a stale ``.tmp`` left by a
        crashed run cannot be mistaken for this run's scratch (cheap hygiene,
        not a concurrency mechanism — a single writer per store is assumed)."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._store_path.with_suffix(
            self._store_path.suffix + f".{os.getpid()}.tmp"
        )
        with tmp.open("w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(_experience_to_dict(e)) + "\n")
            for line in passthrough_lines:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._store_path)

    def _disk_rows(self) -> tuple[list[Experience], list[str]]:
        """Current on-disk rows + forward-compat passthrough lines (``([], [])``
        if the file is missing). The write path merges the experiences in
        before rewriting so a write-only store (one that never called
        ``load()``) cannot truncate prior lessons, and carries the passthrough
        lines through verbatim so future-version rows survive compaction."""
        if not self._store_path.exists():
            return [], []
        return self._parse(self._store_path.read_text())

    def add(self, experience: Experience) -> None:
        """Add one row. Thin wrapper over ``add_many`` — see it for the
        read-merge + forward-compat-passthrough behavior."""
        self.add_many([experience])

    def add_many(self, experiences: Iterable[Experience]) -> None:
        """Add rows, merging by dedup key (keep best speedup), then rewrite.

        Re-reads the on-disk store and merges it in first, so write-only mode
        (no prior ``load()``) cannot truncate existing lessons. Forward-compat
        future-version rows are carried through the rewrite verbatim."""
        rows = list(experiences)
        if not rows:
            return
        disk_exps, passthrough = self._disk_rows()
        merged = dedup_best(disk_exps + self._experiences + rows)
        self._rewrite(merged, passthrough)
        self._experiences = merged

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
    # A true legacy row (no ``condition`` key) is backfilled with a
    # params-only condition so edge rows sharing an action_id but differing
    # in params (e.g. BLOCK_N=32 vs 64) keep distinct dedup keys and do not
    # collapse on migration ("" when there are no params). New rows always
    # serialize ``condition`` and take the str branch untouched; a non-string
    # value was already warned + dropped to "" in ``_parse``.
    if "condition" not in d:
        condition = _format_condition(None, action)
    elif isinstance(d["condition"], str):
        condition = d["condition"]
    else:
        condition = ""
    # ``created_at`` is a ``dedup_best`` tie-breaker (``(speedup, created_at)``),
    # not identity. A non-string value (hand-edited JSONL) makes the tuple
    # comparison raise ``TypeError`` and aborts the whole load/merge. Coerce it
    # to "" — keeps the otherwise-valid row; "" sorts before any real ISO
    # timestamp, an acceptable order for a corrupt row.
    created_at = d.get("created_at", "")
    if not isinstance(created_at, str):
        logger.warning(
            "row %r: non-string created_at=%r — coercing to \"\"",
            d.get("row_id", ""),
            created_at,
        )
        created_at = ""
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
        created_at=created_at,
        condition=condition,
    )
