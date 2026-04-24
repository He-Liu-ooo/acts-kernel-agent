"""Shared UTC timestamp formatters for filenames and event payloads.

Two surfaces, two formats, one source of truth:

- ``filename_ts()`` — ``YYYYMMDDTHHMMSS_ffffffZ`` — used for per-run
  directory and trace file names. Microsecond precision so concurrent
  invocations (parallel CI, ablation scripts sharing ``--run-dir``) get
  distinct paths; sortable; filename-safe (``:`` is illegal on FAT/Win).
- ``iso_ts()`` — ``YYYY-MM-DDTHH:MM:SSZ`` — used for the ``ts`` field in
  JSONL event records. Standard ISO 8601, second precision.

Both use ``datetime.now(timezone.utc)`` instead of the deprecated
``datetime.utcnow()``.
"""
from __future__ import annotations

from datetime import datetime, timezone


def filename_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def iso_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
