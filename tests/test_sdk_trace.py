"""Tier-1 unit tests for src/runtime/sdk_trace.py — no SDK, no torch.

Covers the two branches of ``trace_span``: SDK-absent (nullcontext
fallback for Tier-1) and SDK-present (delegate to ``agents.trace`` with
properly shaped metadata). The AgentLabel coercion rule — read
``.value`` rather than relying on ``str(member)`` because Python 3.10's
stdlib Enum overrides ``__str__`` to ``"AgentLabel.X"`` — is the
load-bearing detail and gets its own test.
"""

from __future__ import annotations

import contextlib

import pytest

from src.runtime import sdk_trace
from src.runtime.sdk_trace import _coerce_agent_label, trace_span
from src.runtime.usage import AgentLabel


class TestTraceSpanSDKAbsent:
    """When the openai-agents SDK isn't installed (Tier-1 venv) the helper
    must degrade to ``contextlib.nullcontext`` so orchestrator / baseline
    code stays harness-agnostic."""

    def test_returns_nullcontext_when_sdk_missing(self, monkeypatch):
        monkeypatch.setattr(sdk_trace, "_agents_trace", None)
        cm = trace_span("acts_iter", iter_no=0, agent=AgentLabel.PLANNER)
        # nullcontext supports `with`-statement with no side effect.
        assert isinstance(cm, contextlib.nullcontext)
        with cm:
            pass  # no-op; just confirms the contract holds


class TestTraceSpanSDKPresent:
    """When the SDK is present the helper must forward ``workflow_name``
    and a properly shaped metadata dict to ``agents.trace``."""

    def _install_fake(self, monkeypatch):
        captured: list[dict] = []

        def fake_trace(*, workflow_name, metadata):
            captured.append({"workflow_name": workflow_name, "metadata": metadata})
            return contextlib.nullcontext()

        monkeypatch.setattr(sdk_trace, "_agents_trace", fake_trace)
        return captured

    def test_metadata_carries_iter_and_agent_as_string(self, monkeypatch):
        captured = self._install_fake(monkeypatch)
        with trace_span("acts_iter", iter_no=3, agent=AgentLabel.CODER):
            pass
        assert len(captured) == 1
        assert captured[0]["workflow_name"] == "acts_iter"
        assert captured[0]["metadata"] == {"iter": 3, "agent": "coder"}

    def test_extra_kwargs_flatten_into_metadata(self, monkeypatch):
        # Baseline path uses ``attempt=N`` as an extra metadata key.
        captured = self._install_fake(monkeypatch)
        with trace_span(
            "acts_baseline",
            iter_no=0,
            agent=AgentLabel.CODER_TRANSLATE,
            attempt=2,
        ):
            pass
        assert captured[0]["metadata"] == {
            "iter": 0,
            "agent": "coder-translate",
            "attempt": 2,
        }

    def test_agent_label_coder_translate_stringifies_to_value(self, monkeypatch):
        # Load-bearing: the resource accumulator reads the metadata
        # ``agent`` field as a plain string. AgentLabel.CODER_TRANSLATE
        # must render as ``"coder-translate"`` (with hyphen), not as
        # ``"AgentLabel.CODER_TRANSLATE"`` which is what Python 3.10's
        # stdlib enum ``__str__`` produces. ``_coerce_agent_label`` reads
        # ``.value`` for enum members.
        captured = self._install_fake(monkeypatch)
        with trace_span(
            "acts_baseline",
            iter_no=0,
            agent=AgentLabel.CODER_TRANSLATE,
        ):
            pass
        assert captured[0]["metadata"]["agent"] == "coder-translate"

    def test_bare_string_agent_passes_through(self, monkeypatch):
        # Callers may still pass a bare string for ad-hoc agents not in
        # the canonical enum; the helper coerces via ``str(agent)`` in
        # that path. Backward-compat for any future caller that isn't
        # ready to switch to the enum.
        captured = self._install_fake(monkeypatch)
        with trace_span("acts_iter", iter_no=1, agent="custom-agent"):
            pass
        assert captured[0]["metadata"]["agent"] == "custom-agent"


class TestCoerceAgentLabel:
    """Direct unit test of the coercion helper — `trace_span` is the
    primary consumer but any future event emitter can reuse it."""

    @pytest.mark.parametrize("member,expected", [
        (AgentLabel.PLANNER, "planner"),
        (AgentLabel.CODER, "coder"),
        (AgentLabel.CODER_TRANSLATE, "coder-translate"),
        (AgentLabel.REVIEWER, "reviewer"),
    ])
    def test_enum_member_yields_value(self, member, expected):
        assert _coerce_agent_label(member) == expected

    def test_bare_string_passes_through(self):
        assert _coerce_agent_label("custom") == "custom"
