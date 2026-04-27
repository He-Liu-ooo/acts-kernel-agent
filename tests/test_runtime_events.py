"""Tests for src/runtime/events.py — event catalog membership."""

from __future__ import annotations

from src.runtime import events


def test_reviewer_metric_query_in_core_event_kinds():
    """Multi-turn Reviewer's per-call metric-fetch event is part of the
    canonical catalog; emit() warns on unknown kinds, so this prevents
    a typo regression."""
    assert "reviewer_metric_query" in events.CORE_EVENT_KINDS
