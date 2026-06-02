"""Tests for SummarizerAgent — mocked LLM, structured JSON output."""

from __future__ import annotations

import contextlib
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.experience import ActionRecord
from src.memory.summarizer import SummarizerAgent, SummarizerResult
from src.runtime.usage import AgentLabel


@pytest.mark.asyncio
async def test_summarize_edge_round_trip(monkeypatch):
    response = json.dumps({
        "title": "Removing double buffering",
        "lesson": "The slow kernel had unnecessary double buffering.",
        "snippet_before": "a_regs_cur = ...",
        "snippet_after": "a = tl.load(...)",
    })
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="deepseek-chat")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value=response))

    out = await agent.summarize(
        parent_src="def kernel(): a_regs_cur = ...",
        child_src="def kernel(): a = tl.load(...)",
        speedup=1.96,
        action=ActionRecord(action_id="simplify_pipeline", tier=3, name="simplify"),
    )
    assert isinstance(out, SummarizerResult)
    assert out.title == "Removing double buffering"
    assert out.snippet_before.startswith("a_regs_cur")
    assert out.snippet_after.startswith("a = tl.load")


@pytest.mark.asyncio
async def test_summarize_run_uses_cumulative_prompt(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_run(prompt: str, *, iter_no: int = 0) -> str:
        captured["text"] = prompt
        return json.dumps({
            "title": "T", "lesson": "L",
            "snippet_before": "a", "snippet_after": "b",
        })

    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", fake_run)

    await agent.summarize_run(
        baseline_src="def baseline(): ...",
        best_src="def best(): ...",
        cumulative_speedup=3.2,
    )
    text = captured["text"].lower()
    assert "cumulative" in text or "multi-step" in text


@pytest.mark.asyncio
async def test_summarize_returns_none_on_malformed_json(monkeypatch, caplog):
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value="not json at all"))
    with caplog.at_level("WARNING"):
        out = await agent.summarize(
            parent_src="a", child_src="b", speedup=1.5,
            action=ActionRecord(action_id="a", tier=1, name="n"),
        )
    assert out is None
    assert any(
        "json" in r.message.lower() or "parse" in r.message.lower()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_summarize_returns_none_on_no_optimization_found(monkeypatch):
    response = json.dumps({
        "title": "No optimization found", "lesson": "",
        "snippet_before": "", "snippet_after": "",
    })
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value=response))
    out = await agent.summarize(
        parent_src="x", child_src="x", speedup=1.0,
        action=ActionRecord(action_id="a", tier=1, name="n"),
    )
    assert out is None


@pytest.mark.asyncio
async def test_summarize_returns_none_on_empty_snippets(monkeypatch):
    response = json.dumps({
        "title": "T", "lesson": "L",
        "snippet_before": "", "snippet_after": "b",
    })
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value=response))
    out = await agent.summarize(
        parent_src="x", child_src="y", speedup=1.5,
        action=ActionRecord(action_id="a", tier=1, name="n"),
    )
    assert out is None


@pytest.mark.asyncio
async def test_summarize_returns_none_on_identical_snippets(monkeypatch):
    response = json.dumps({
        "title": "T", "lesson": "L",
        "snippet_before": "x = 1", "snippet_after": "x = 1",
    })
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value=response))
    out = await agent.summarize(
        parent_src="x", child_src="y", speedup=1.5,
        action=ActionRecord(action_id="a", tier=1, name="n"),
    )
    assert out is None


@pytest.mark.asyncio
async def test_summarize_returns_none_on_runner_yielding_none(monkeypatch, caplog):
    """Regression for ultra-review finding: ``Runner.run`` can yield
    ``final_output=None`` (max_turns exhausted, no submit). ``json.loads(None)``
    raises ``TypeError`` (not ``JSONDecodeError``); the prior narrow except
    let it escape into the caller. The explicit None-guard must catch it.
    """
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value=None))
    with caplog.at_level("WARNING"):
        out = await agent.summarize(
            parent_src="x", child_src="y", speedup=1.5,
            action=ActionRecord(action_id="a", tier=1, name="n"),
        )
    assert out is None
    assert any("non-string" in r.message.lower() or "nonetype" in r.message.lower()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_summarize_returns_none_on_non_object_json(monkeypatch, caplog):
    """Regression for ultra-review finding: a valid JSON list/scalar would
    crash ``data.get('title')`` with ``AttributeError`` before the
    isinstance guard was added. Documented contract is warn + None."""
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value='["a", "b"]'))
    with caplog.at_level("WARNING"):
        out = await agent.summarize(
            parent_src="x", child_src="y", speedup=1.5,
            action=ActionRecord(action_id="a", tier=1, name="n"),
        )
    assert out is None
    assert any("non-object" in r.message.lower() or "list" in r.message.lower()
               for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("title", 5),
        ("title", ["x"]),
        ("lesson", {"a": 1}),
        ("lesson", 3.14),
    ],
)
async def test_summarize_returns_none_on_non_string_title_lesson(
    monkeypatch, caplog, field, bad_value
):
    """Regression for Codex P2: a valid JSON object with a truthy non-string
    ``title``/``lesson`` (e.g. ``{"title": 5}``) passes ``... or ""`` then
    crashes ``.strip()`` with ``AttributeError`` — violating the never-raise
    contract. Must warn + return None instead."""
    payload = {
        "title": "T",
        "lesson": "L",
        "snippet_before": "a = 1",
        "snippet_after": "b = 2",
    }
    payload[field] = bad_value
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value=json.dumps(payload)))
    with caplog.at_level("WARNING"):
        out = await agent.summarize(
            parent_src="x", child_src="y", speedup=1.5,
            action=ActionRecord(action_id="a", tier=1, name="n"),
        )
    assert out is None
    assert any("non-string" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_summarize_returns_none_on_non_string_snippet(monkeypatch, caplog):
    """A truthy non-string snippet would flow downstream as a non-string;
    reject it the same way for consistency with title/lesson."""
    response = json.dumps({
        "title": "T",
        "lesson": "L",
        "snippet_before": ["a = 1"],
        "snippet_after": "b = 2",
    })
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value=response))
    with caplog.at_level("WARNING"):
        out = await agent.summarize(
            parent_src="x", child_src="y", speedup=1.5,
            action=ActionRecord(action_id="a", tier=1, name="n"),
        )
    assert out is None
    assert any("non-string" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_summarize_accepts_null_title_lesson_coerced_to_empty(monkeypatch):
    """Positive control: an explicit ``null`` (or missing) title/lesson is
    tolerated and coerces to ``""`` — only non-None non-str is rejected.
    The row still returns because snippets are valid + distinct."""
    response = json.dumps({
        "title": None,
        "lesson": None,
        "snippet_before": "a = 1",
        "snippet_after": "b = 2",
    })
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value=response))
    out = await agent.summarize(
        parent_src="x", child_src="y", speedup=1.5,
        action=ActionRecord(action_id="a", tier=1, name="n"),
    )
    assert isinstance(out, SummarizerResult)
    assert out.title == ""
    assert out.lesson == ""


@pytest.mark.asyncio
async def test_summarize_rejects_snippet_with_four_backticks(monkeypatch, caplog):
    """Defense in depth for the planner-prompt fence-escape vector.

    The Planner wraps snippets in 4-backtick fences. A snippet containing
    ``\\`\\`\\`\\``` or more would close the fence and bleed into prose.
    The summarizer must reject such rows at write time so they never reach
    the store.
    """
    response = json.dumps({
        "title": "T",
        "lesson": "L",
        "snippet_before": "x = 1\n```` markdown ````\nx = 2",
        "snippet_after": "y = 1",
    })
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value=response))
    with caplog.at_level("WARNING"):
        out = await agent.summarize(
            parent_src="x", child_src="y", speedup=1.5,
            action=ActionRecord(action_id="a", tier=1, name="n"),
        )
    assert out is None
    assert any("backtick" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_summarize_accepts_snippet_with_three_backticks(monkeypatch):
    """The 4-backtick guard must not reject snippets with at most 3
    consecutive backticks — those are legitimate Triton-source docstring
    / comment artifacts that the Planner can safely contain in a
    4-backtick fence.
    """
    response = json.dumps({
        "title": "T",
        "lesson": "L",
        "snippet_before": '"""```\ndocstring with triple backticks\n"""',
        "snippet_after": "x = 2",
    })
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", AsyncMock(return_value=response))
    out = await agent.summarize(
        parent_src="x", child_src="y", speedup=1.5,
        action=ActionRecord(action_id="a", tier=1, name="n"),
    )
    assert out is not None
    assert "```" in out.snippet_before
    assert "````" not in out.snippet_before


@pytest.mark.asyncio
async def test_summarize_returns_none_on_llm_exception(monkeypatch, caplog):
    async def boom(prompt: str) -> str:
        raise RuntimeError("simulated network error")

    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")
    monkeypatch.setattr(agent, "_run", boom)
    with caplog.at_level("WARNING"):
        out = await agent.summarize(
            parent_src="x", child_src="y", speedup=1.5,
            action=ActionRecord(action_id="a", tier=1, name="n"),
        )
    assert out is None
    assert any("summariz" in r.message.lower() for r in caplog.records)


# === Usage-trace tagging (Codex P2: summarizer tokens dropped) ==================


def test_agent_label_summarizer_exists():
    """The summarizer needs its own AgentLabel so its trace metadata carries
    a recognized ``agent`` tag — without it UsageAccumulator drops the trace
    and usage.json / the report omit summarizer tokens."""
    assert AgentLabel.SUMMARIZER == "summarizer"
    assert AgentLabel.SUMMARIZER.value == "summarizer"


def _install_run_seam(monkeypatch, *, captured: dict):
    """Patch the SDK seams ``_run`` reaches so the real ``_run`` body runs in
    the torchless Tier-1 venv (no ``agents`` SDK). Captures the ``trace_span``
    call kwargs so a test can assert iter + agent tagging.
    """
    import src.agents.llm_backend as llm_backend
    import src.runtime.sdk_trace as sdk_trace

    @contextlib.contextmanager
    def fake_trace_span(workflow_name, *, iter_no, agent, **extra):
        captured["workflow_name"] = workflow_name
        captured["iter_no"] = iter_no
        captured["agent"] = agent
        yield

    async def fake_run_agent(agent, prompt, *, run_config, max_turns):
        captured["ran"] = True
        return types.SimpleNamespace(
            final_output=json.dumps({
                "title": "T", "lesson": "L",
                "snippet_before": "a = 1", "snippet_after": "b = 2",
            })
        )

    # Patch on the already-imported module objects (not via string paths),
    # so monkeypatch never re-runs a ``from agents import ...`` that the
    # fake-``agents`` stub below would break.
    monkeypatch.setattr(sdk_trace, "trace_span", fake_trace_span)
    monkeypatch.setattr(llm_backend, "make_run_config", lambda **k: object())
    monkeypatch.setattr(llm_backend, "run_agent", fake_run_agent)

    # Stub the heavy ``agents`` SDK import at the top of _run last, after the
    # seam patches above resolved against the real (absent-SDK) modules.
    fake_agents = types.ModuleType("agents")
    fake_agents.Agent = lambda *a, **k: MagicMock(name="Agent")
    fake_agents.trace = lambda *a, **k: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "agents", fake_agents)


@pytest.mark.asyncio
async def test_summarize_wraps_run_in_summarizer_trace_span(monkeypatch):
    """``summarize`` must run the LLM call inside a ``trace_span`` tagged
    with the caller's iter and ``AgentLabel.SUMMARIZER`` so the tokens are
    bucketed instead of dropped."""
    captured: dict = {}
    _install_run_seam(monkeypatch, captured=captured)
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")

    out = await agent.summarize(
        parent_src="x", child_src="y", speedup=1.5,
        action=ActionRecord(action_id="a", tier=1, name="n"),
        iter_no=7,
    )
    assert isinstance(out, SummarizerResult)
    assert captured["ran"] is True
    assert captured["workflow_name"] == "acts_summarizer"
    assert captured["iter_no"] == 7
    assert captured["agent"] == AgentLabel.SUMMARIZER


@pytest.mark.asyncio
async def test_summarize_run_uses_iter_zero_trace_span(monkeypatch):
    """The run-scope (G3) summary has no live iter; it must still be tagged
    (``iter_no=0``, the baseline/translate out-of-loop convention) so the
    trace is not dropped."""
    captured: dict = {}
    _install_run_seam(monkeypatch, captured=captured)
    agent = SummarizerAgent(model=MagicMock(), summarizer_model_name="m")

    out = await agent.summarize_run(
        baseline_src="def b(): ...", best_src="def c(): ...",
        cumulative_speedup=3.0,
    )
    assert isinstance(out, SummarizerResult)
    assert captured["iter_no"] == 0
    assert captured["agent"] == AgentLabel.SUMMARIZER
