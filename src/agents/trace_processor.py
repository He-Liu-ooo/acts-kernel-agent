"""Local JSONL trace capture for the OpenAI Agents SDK.

Without this the SDK's default tracing path ships every span to
``api.openai.com/v1/traces/ingest``, which is unreachable for non-OpenAI
providers (DeepSeek key → 401) and routes prompts/responses through a
third-party service we have no business using.

This module provides ``JSONLTraceProcessor``, a ``TracingProcessor``
implementation that writes one newline-delimited JSON record per trace /
span event to a per-run file under a configurable directory. A helper
``enable_local_trace_capture`` registers an instance with the SDK
(``set_trace_processors``) so it fully replaces the default OpenAI
exporter — no traces leave the host. Returns the processor for atexit
registration / inspection.

Captured fields (per record):
- ``event``: ``"trace_end"`` | ``"span_end"``.
- ``trace_id`` / ``span_id`` / ``parent_id``: linkage.
- ``started_at`` / ``ended_at``: ISO timestamps.
- ``span_data``: full ``SpanData.export()`` payload — includes the LLM
  ``input`` / ``output`` arrays, ``model``, ``model_config``, ``usage``
  for ``GenerationSpanData``; tool name / arguments / result for
  ``FunctionSpanData``; etc.
- ``error``: populated when the span ends with an error set.
- ``metadata``: trace-level metadata (workflow tags etc.) on ``trace_end``.

Trace ``start`` events are intentionally not recorded — start-only data
duplicates fields that arrive on ``end`` (timestamps, name) and would
double the file size for no diagnostic value.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from agents.tracing.processor_interface import TracingProcessor
except ModuleNotFoundError:  # pragma: no cover — Tier 1 venv has no SDK
    class TracingProcessor:  # type: ignore[no-redef]
        """SDK-absent stand-in. Concrete tests subclass this directly."""


def _isoformat_utc() -> str:
    """Filename-safe UTC timestamp; ``:`` is illegal on FAT/Win and noisy
    in shell completions everywhere else."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


class JSONLTraceProcessor(TracingProcessor):
    """Serialize SDK trace + span events to a per-run JSONL file.

    Thread-safe: the SDK's processor interface guarantees calls from
    arbitrary threads, so writes go through a lock. File handles use
    line-buffering so ``force_flush`` and post-shutdown reads see complete
    records without an explicit ``flush()`` per write.

    Late events (received after ``shutdown``) are silently dropped — the
    SDK can fire a final span notification from a background worker after
    main-thread cleanup, and a noisy "I/O on closed file" secondary
    failure during teardown is worse than losing one diagnostic record.
    """

    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path: Path = out_dir / f"acts_trace_{_isoformat_utc()}.jsonl"
        # ``buffering=1`` = line-buffered, so each ``write(line)`` is
        # visible to readers (e.g., ``tail -f``) without an explicit flush.
        self._fh = self.path.open("w", buffering=1)
        self._lock = threading.Lock()
        self._closed = False

    # ── TracingProcessor interface ─────────────────────────────────────

    def on_trace_start(self, trace: Any) -> None:
        # Trace start carries no data the end event doesn't also carry —
        # skip the write to halve file size.
        return

    def on_trace_end(self, trace: Any) -> None:
        record: dict[str, Any] = {
            "event": "trace_end",
            "trace_id": getattr(trace, "trace_id", None),
            "name": getattr(trace, "name", None),
            "started_at": getattr(trace, "started_at", None),
            "ended_at": getattr(trace, "ended_at", None),
            "metadata": dict(getattr(trace, "metadata", None) or {}),
        }
        self._write(record)

    def on_span_start(self, span: Any) -> None:
        # Span start = no data yet (input/output land at end). Same logic
        # as trace_start: skip to halve file size.
        return

    def on_span_end(self, span: Any) -> None:
        span_data = getattr(span, "span_data", None)
        exported: dict[str, Any] | None = None
        if span_data is not None and hasattr(span_data, "export"):
            try:
                exported = span_data.export()
            except Exception as exc:  # noqa: BLE001 — capture-best-effort
                exported = {"export_error": f"{type(exc).__name__}: {exc}"}

        record: dict[str, Any] = {
            "event": "span_end",
            "span_id": getattr(span, "span_id", None),
            "trace_id": getattr(span, "trace_id", None),
            "parent_id": getattr(span, "parent_id", None),
            "started_at": getattr(span, "started_at", None),
            "ended_at": getattr(span, "ended_at", None),
            "span_data": exported,
            "error": getattr(span, "error", None),
        }
        self._write(record)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass

    def force_flush(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._fh.flush()
            except Exception:  # noqa: BLE001
                pass

    # ── internals ──────────────────────────────────────────────────────

    def _write(self, record: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._fh.write(json.dumps(record, default=str))
                self._fh.write("\n")
            except Exception:  # noqa: BLE001 — never let trace I/O kill a run
                pass


def enable_local_trace_capture(out_dir: Path) -> JSONLTraceProcessor:
    """Register a ``JSONLTraceProcessor`` as the SDK's only trace processor.

    Replaces (not augments) the default OpenAI exporter so traces stay
    on-host. Returns the processor so the caller can inspect ``.path``
    or wire ``shutdown`` into atexit. Raises ``RuntimeError`` if the SDK
    isn't installed — callers that may run in the Tier 1 venv should
    gate on SDK availability before calling.
    """
    try:
        from agents import set_trace_processors
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "openai-agents SDK not installed; trace capture requires the "
            "GPU venv (see auto-memory ``reference_test_venv.md``)."
        ) from exc

    processor = JSONLTraceProcessor(out_dir=out_dir)
    set_trace_processors([processor])
    return processor
