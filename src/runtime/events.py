"""ACTS-narrative event emission.

One ``emit()`` call fans out to two sinks: stdlib ``logger.info`` (human
text -> stderr + run.log) and a module-level JSONL file handle
(structured -> events.jsonl). Unbound => JSONL write is skipped; the
logger line still goes out. See ``doc/specs/2026-04-22-logger-system-design.md``
§5.2 for full semantics.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any, IO

from src.runtime.timefmt import iso_ts

logger = logging.getLogger(__name__)

CORE_EVENT_KINDS: frozenset[str] = frozenset({
    "run_start", "baseline_attempt", "baseline_success", "baseline_failure",
    "baseline_ready", "iter_start", "planner_selected",
    # ``coder_submitted`` marks ``implement()`` returning a kernel — it
    # does NOT prove the compile/correctness tools ran or passed.
    # Ground-truth per-tool-call records are in ``traces/*.jsonl``.
    # ``coder_failed`` covers any ``ImplementationError`` cause
    # (compile, correctness, turn-budget, missing ``submit_kernel``).
    "coder_submitted", "coder_failed",
    "bench_done", "profile_done", "score_computed",
    "reviewer_feedback", "branch_dead_end", "iter_end", "verify_start",
    "verify_done", "run_end",
})

# ``iter_end.outcome`` values. Kept as string constants (not an enum) so
# emit payloads stay trivially JSON-serializable and callers don't pay an
# import tax. Typos are caught via `CORE_EVENT_KINDS`-style review, not
# the type system.
ITER_ADVANCED = "advanced"
ITER_DEAD_END = "dead_end"
ITER_SKIPPED = "skipped"

_events_fh: IO[str] | None = None
_lock = threading.Lock()


def finite_or_none(x: float | int | None) -> float | None:
    """Map non-finite floats (``inf``/``-inf``/``nan``) to ``None``.

    ``BenchmarkResult.per_workload_latency_us`` uses ``math.inf`` as a
    launch-failure sentinel; forwarding that verbatim produces the
    non-standard ``Infinity`` token in ``events.jsonl``, which RFC-8259
    parsers reject.
    """
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def bind(fh: IO[str]) -> None:
    """Register a file handle for JSONL writes. Called by ``RunContext.create``."""
    global _events_fh
    with _lock:
        _events_fh = fh


def unbind() -> None:
    """Clear the registered file handle. Called by ``RunContext.close`` before FH close."""
    global _events_fh
    with _lock:
        _events_fh = None


def _compact_json(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, default=str, separators=(",", ":"))
    except Exception:
        return "{}"


def emit(kind: str, *, iter: int | None = None, **fields: Any) -> None:
    """Emit a narrative event to both sinks. Never raises."""
    if kind not in CORE_EVENT_KINDS:
        try:
            logger.warning("unknown event kind: %s", kind)
        except Exception:
            pass
    # Skip the serialize-for-log work if nobody at INFO is listening.
    if logger.isEnabledFor(logging.INFO):
        # Merge ``iter`` into the log payload so ``run.log`` shows the
        # iteration for iter-scoped events; omitted when ``iter is None``
        # so per-iter greps aren't polluted with a null key.
        log_payload: dict[str, Any] = (
            {"iter": iter, **fields} if iter is not None else dict(fields)
        )
        try:
            logger.info("%s %s", kind, _compact_json(log_payload))
        except Exception:
            pass
    fh = _events_fh
    if fh is None:
        return
    # Build the record (including timestamp) outside the write lock so
    # contending threads don't serialize on datetime formatting.
    try:
        record = {"ts": iso_ts(), "kind": kind, "iter": iter, **fields}
        payload = json.dumps(record, default=str) + "\n"
    except Exception:
        return
    try:
        with _lock:
            fh.write(payload)
    except Exception:
        pass
