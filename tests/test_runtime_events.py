"""Tests for src/runtime/events.py — event catalog membership."""

from __future__ import annotations

from src.runtime import events


def test_reviewer_metric_query_in_core_event_kinds():
    """Multi-turn Reviewer's per-call metric-fetch event is part of the
    canonical catalog; emit() warns on unknown kinds, so this prevents
    a typo regression."""
    assert "reviewer_metric_query" in events.CORE_EVENT_KINDS


def test_sibling_context_and_repeated_pathway_event_kinds_registered():
    """Sibling-aware Planner/Reviewer contracts add two catalog entries
    (see doc/specs/2026-05-13-sibling-aware-agent-contracts-design.md):
    one fires on every Planner/Reviewer call that consumed a non-empty
    sibling_context block; the other fires when the Reviewer dead-ends
    on an action that a prior sibling already regressed on. emit() warns
    on unknown kinds, so missing either entry would silently downgrade
    these signals to log noise."""
    from src.runtime.events import CORE_EVENT_KINDS
    assert "sibling_context_rendered" in CORE_EVENT_KINDS
    assert "repeated_pathway_dead_end" in CORE_EVENT_KINDS
