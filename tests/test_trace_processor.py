"""Tests for src.agents.trace_processor — local JSONL trace capture.

The processor receives Trace + Span notifications from the OpenAI Agents
SDK and writes one JSON record per event to a per-run JSONL file. This
is our diagnostic surface for "what did the LLM actually say in each
turn" — without it, all we see in logs is HTTP status codes.

These tests run torch-free: the processor itself is pure file I/O over
the SDK's trace abstractions, so we can stand in for ``Trace`` / ``Span``
with simple namespaces. The ``set_trace_processors`` registration helper
is exercised separately under an SDK-installed marker so Tier 1 stays
green without ``agents`` available.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.trace_processor import JSONLTraceProcessor


def _fake_trace(
    trace_id: str = "trace_abc",
    name: str = "Coder",
    metadata: dict | None = None,
) -> SimpleNamespace:
    """Stand-in for ``agents.tracing.traces.Trace``. The processor only reads
    ``trace_id``, ``name``, ``started_at``, ``ended_at`` (all string-typed)
    and a ``metadata`` mapping — no abstract-method machinery needed.

    ``metadata`` defaults to ``{"workflow": "implement"}``; pass an explicit
    dict (e.g. ``{"iter": 1, "agent": "coder"}``) to drive the
    UsageAccumulator's (iter, agent) bucket resolution.
    """
    return SimpleNamespace(
        trace_id=trace_id,
        name=name,
        started_at="2026-04-22T17:30:00",
        ended_at="2026-04-22T17:30:30",
        metadata={"workflow": "implement"} if metadata is None else metadata,
    )


def _fake_generation_span(
    span_id: str = "span_1",
    parent_id: str | None = None,
    trace_id: str = "trace_abc",
    usage: object = None,
    span_type: str = "generation",
) -> SimpleNamespace:
    """Stand-in for an LLM generation span with full input/output capture.
    Mirrors the shape ``GenerationSpanData.export()`` returns + the parent
    ``Span`` envelope fields the processor wraps each export in.

    ``usage`` overrides the default ``{"prompt_tokens": 1024,
    "completion_tokens": 128}`` payload so tests can exercise malformed
    or Responses-API-shaped usage dicts. ``span_type`` allows
    non-generation shapes (e.g. ``"function"``).
    """
    if usage is None:
        usage = {"prompt_tokens": 1024, "completion_tokens": 128}
    exported = {
        "type": span_type,
        "input": [{"role": "user", "content": "compile this kernel"}],
        "output": [{"role": "assistant", "content": "tool_call: compile"}],
        "model": "deepseek-reasoner",
        "model_config": {"temperature": 0.0},
        "usage": usage,
    }
    span_data = SimpleNamespace(export=lambda: exported)
    return SimpleNamespace(
        span_id=span_id,
        trace_id=trace_id,
        parent_id=parent_id,
        started_at="2026-04-22T17:30:01",
        ended_at="2026-04-22T17:30:05",
        span_data=span_data,
        error=None,
    )


# ── file lifecycle ─────────────────────────────────────────────────────


def test_processor_creates_per_run_jsonl_file(tmp_path: Path):
    """A fresh processor opens a single JSONL file under ``out_dir``. Path
    is exposed via ``.path`` so callers can surface it in run reports."""
    proc = JSONLTraceProcessor(out_dir=tmp_path)
    try:
        assert proc.path.parent == tmp_path
        assert proc.path.suffix == ".jsonl"
        assert proc.path.exists()
    finally:
        proc.shutdown()


def test_processor_writes_no_records_before_any_event(tmp_path: Path):
    """Empty file when no traces/spans have fired — important so a
    crash-before-first-trace produces a 0-byte file, not stale data."""
    proc = JSONLTraceProcessor(out_dir=tmp_path)
    proc.shutdown()
    assert proc.path.read_text() == ""


# ── trace event capture ──────────────────────────────────────────────────


def test_on_trace_end_writes_one_record(tmp_path: Path):
    """``trace_end`` lands as one JSONL record with the trace's identifying
    fields. ``trace_start`` is intentionally not recorded — start-only
    info is redundant with end (timestamps + name are on end too)."""
    proc = JSONLTraceProcessor(out_dir=tmp_path)
    trace = _fake_trace(trace_id="t-1", name="Coder")
    proc.on_trace_start(trace)
    proc.on_trace_end(trace)
    proc.shutdown()

    lines = proc.path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "trace_end"
    assert record["trace_id"] == "t-1"
    assert record["name"] == "Coder"


def test_on_span_end_writes_full_generation_data(tmp_path: Path):
    """A generation span end record must carry the full LLM input/output
    + model + usage — that's the diagnostic surface for "what did the
    Coder actually say on turn N"."""
    proc = JSONLTraceProcessor(out_dir=tmp_path)
    span = _fake_generation_span(span_id="s-1")
    proc.on_span_start(span)
    proc.on_span_end(span)
    proc.shutdown()

    lines = proc.path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["event"] == "span_end"
    assert record["span_id"] == "s-1"
    assert record["trace_id"] == "trace_abc"
    assert record["span_data"]["type"] == "generation"
    assert record["span_data"]["model"] == "deepseek-reasoner"
    assert "compile this kernel" in record["span_data"]["input"][0]["content"]
    assert record["span_data"]["usage"]["prompt_tokens"] == 1024


def test_span_with_error_captures_error_field(tmp_path: Path):
    """When a span ends with ``error`` populated (tool failure, validation
    miss, etc.), the record must surface it so post-mortem analysis can
    correlate failures to the LLM turn that produced them."""
    proc = JSONLTraceProcessor(out_dir=tmp_path)
    span = _fake_generation_span()
    span.error = {"message": "submit_kernel FAILED", "data": {"name": "claimed"}}
    proc.on_span_end(span)
    proc.shutdown()

    record = json.loads(proc.path.read_text().strip().splitlines()[0])
    assert record["error"] == {
        "message": "submit_kernel FAILED",
        "data": {"name": "claimed"},
    }


def test_records_are_jsonl_one_per_event(tmp_path: Path):
    """Mixed trace + span events serialize as one JSON record per line.
    Format guarantee — downstream tooling (jq, pandas.read_json(lines=True))
    expects strict newline-delimited JSON with no nested arrays."""
    proc = JSONLTraceProcessor(out_dir=tmp_path)
    trace = _fake_trace()
    span_a = _fake_generation_span(span_id="s-a")
    span_b = _fake_generation_span(span_id="s-b", parent_id="s-a")

    proc.on_trace_start(trace)
    proc.on_span_end(span_a)
    proc.on_span_end(span_b)
    proc.on_trace_end(trace)
    proc.shutdown()

    lines = proc.path.read_text().strip().splitlines()
    assert len(lines) == 3  # 2 spans + 1 trace_end (trace_start not recorded)

    # Every line must parse standalone — no multi-line records.
    parsed = [json.loads(line) for line in lines]
    assert [p["event"] for p in parsed] == ["span_end", "span_end", "trace_end"]
    assert parsed[0]["span_id"] == "s-a"
    assert parsed[1]["span_id"] == "s-b"
    assert parsed[1]["parent_id"] == "s-a"


# ── force_flush + shutdown idempotence ──────────────────────────────────


def test_force_flush_makes_records_visible_to_concurrent_readers(tmp_path: Path):
    """``force_flush`` must drain the buffered writer so a separate process
    (e.g., a tail watcher) sees records as they're produced. Without this,
    a long-running orchestrator's first ~hundred turns could be invisible
    until shutdown."""
    proc = JSONLTraceProcessor(out_dir=tmp_path)
    proc.on_span_end(_fake_generation_span())
    proc.force_flush()

    # File must contain the record without a shutdown call.
    assert proc.path.read_text().strip() != ""
    proc.shutdown()


def test_shutdown_is_idempotent(tmp_path: Path):
    """Double-shutdown must not raise — useful when both an atexit hook and
    explicit cleanup fire on the same processor."""
    proc = JSONLTraceProcessor(out_dir=tmp_path)
    proc.shutdown()
    proc.shutdown()  # must not raise


def test_event_after_shutdown_is_silently_ignored(tmp_path: Path):
    """A late span event arriving after shutdown (race between SDK worker
    threads and main-thread cleanup) must not crash. We drop the event
    rather than reopen the file — silent loss is preferable to a noisy
    secondary failure during teardown."""
    proc = JSONLTraceProcessor(out_dir=tmp_path)
    proc.shutdown()
    # Should not raise even though the file handle is closed.
    proc.on_span_end(_fake_generation_span())
    proc.on_trace_end(_fake_trace())


# ── UsageAccumulator tap (Task 3) ───────────────────────────────────────


from src.runtime.usage import UsageSnapshot


class TestUsageAccumulatorTap:
    def test_generation_spans_routed_to_accumulator(self, tmp_path):
        proc = JSONLTraceProcessor(out_dir=tmp_path)
        try:
            usage = {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }
            span = _fake_generation_span(
                span_id="s1", trace_id="t1", usage=usage,
            )
            proc.on_span_end(span)
            proc.on_trace_end(_fake_trace(
                trace_id="t1", metadata={"iter": 1, "agent": "coder"},
            ))
            snap = proc.snapshot()
            assert isinstance(snap, UsageSnapshot)
            bucket = snap.by_iter_agent[(1, "coder")]
            assert bucket.invocations == 1
            assert bucket.turns == 1
            assert bucket.input_tokens == 100
            assert bucket.output_tokens == 20
        finally:
            proc.shutdown()

    def test_non_generation_spans_ignored_by_accumulator(self, tmp_path):
        proc = JSONLTraceProcessor(out_dir=tmp_path)
        try:
            span = _fake_generation_span(
                span_id="s1", trace_id="t1", span_type="function",
            )
            proc.on_span_end(span)
            proc.on_trace_end(_fake_trace(
                trace_id="t1", metadata={"iter": 1, "agent": "coder"},
            ))
            snap = proc.snapshot()
            # No generation spans → trace_close still credits an
            # invocation? Spec says invocations counts traces; but the
            # whole point is to count model usage. We DO count the
            # invocation (the trace was tagged + closed) but `turns` and
            # token fields stay zero.
            bucket = snap.by_iter_agent.get((1, "coder"))
            assert bucket is not None
            assert bucket.invocations == 1
            assert bucket.turns == 0
            assert bucket.input_tokens == 0
        finally:
            proc.shutdown()

    def test_snapshot_empty_when_no_traces(self, tmp_path):
        proc = JSONLTraceProcessor(out_dir=tmp_path)
        try:
            snap = proc.snapshot()
            assert snap.is_empty
        finally:
            proc.shutdown()

    def test_shutdown_does_not_lose_already_resolved_buckets(self, tmp_path):
        proc = JSONLTraceProcessor(out_dir=tmp_path)
        usage = {"input_tokens": 50, "output_tokens": 5,
                 "input_tokens_details": {}, "output_tokens_details": {}}
        proc.on_span_end(_fake_generation_span(
            span_id="s1", trace_id="t1", usage=usage,
        ))
        proc.on_trace_end(_fake_trace(
            trace_id="t1", metadata={"iter": 1, "agent": "planner"},
        ))
        proc.shutdown()
        # Snapshot still accessible after shutdown.
        snap = proc.snapshot()
        assert snap.by_iter_agent[(1, "planner")].input_tokens == 50


# ── Fix 1: usage tap must never raise from inside SDK callbacks ────────


class TestUsageTapNeverRaises:
    """Discipline: trace I/O must never kill a run. The accumulator tap in
    on_span_end / on_trace_end is wrapped to swallow any version-skew or
    malformed-payload exception from `_parse_span_usage`."""

    def test_malformed_generation_usage_does_not_raise(self, tmp_path):
        proc = JSONLTraceProcessor(out_dir=tmp_path)
        try:
            # `usage` is a string, not a dict — the older parser would have
            # raised AttributeError on `.get(...)`.
            span = _fake_generation_span(
                span_id="s1", trace_id="t1", usage="not a dict",
            )
            # Must not raise.
            proc.on_span_end(span)
            # File write should also have happened (record on disk).
            # Sanity: snapshot still accessible.
            _ = proc.snapshot()
        finally:
            proc.shutdown()

    def test_malformed_trace_metadata_does_not_raise(self, tmp_path):
        proc = JSONLTraceProcessor(out_dir=tmp_path)
        try:
            # `iter` is a non-int string; the accumulator's int() cast in
            # on_trace_close would have raised.
            trace = _fake_trace(
                trace_id="t1", metadata={"iter": "not an int", "agent": "coder"},
            )
            # Must not raise.
            proc.on_trace_end(trace)
            _ = proc.snapshot()
        finally:
            proc.shutdown()
