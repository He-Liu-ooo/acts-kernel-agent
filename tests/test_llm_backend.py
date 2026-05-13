"""Tests for agents/llm_backend.py — retry semantics only.

Model/SDK wiring is exercised indirectly via the agent tests; this file
isolates the transient-vs-permanent distinction and the backoff schedule.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from src.agents.llm_backend import run_agent


class _Transient(Exception):
    """Stand-in for an openai RateLimitError / APITimeoutError."""


class _Permanent(Exception):
    """Stand-in for auth / schema / programmer-bug errors."""


@pytest.mark.asyncio
async def test_transient_error_retries_then_succeeds():
    """First call raises _Transient, second succeeds — result returned,
    no retries left unused."""
    calls = {"n": 0}

    async def fake_run(*_a, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Transient("rate limited")
        return "ok"

    with patch("src.agents.llm_backend.Runner", create=True) as mock_runner:
        mock_runner.run.side_effect = fake_run
        with patch("src.agents.llm_backend.asyncio.sleep", return_value=None):
            result = await run_agent(
                agent=None, prompt="",
                max_retries=3, initial_delay=0.0,
                retriable=(_Transient,),
            )

    assert result == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_transient_exhausts_retries_returns_none(caplog):
    """All attempts raise _Transient — must return None and log a warning
    that names the exception class so the orchestrator can diagnose it."""
    async def always_transient(*_a, **_kw):
        raise _Transient("still rate-limited")

    caplog.set_level(logging.WARNING, logger="src.agents.llm_backend")
    with patch("src.agents.llm_backend.Runner", create=True) as mock_runner:
        mock_runner.run.side_effect = always_transient
        with patch("src.agents.llm_backend.asyncio.sleep", return_value=None):
            result = await run_agent(
                agent=None, prompt="",
                max_retries=3, initial_delay=0.0,
                retriable=(_Transient,),
            )

    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("retries exhausted" in r.getMessage() for r in warnings)
    assert any("_Transient" in r.getMessage() for r in warnings)


@pytest.mark.asyncio
async def test_permanent_error_propagates_without_retry():
    """Non-retriable errors (auth/schema/bug) must raise, NOT be retried.
    Retrying a 401 wastes wall-clock and hides the real cause."""
    calls = {"n": 0}

    async def always_permanent(*_a, **_kw):
        calls["n"] += 1
        raise _Permanent("invalid api key")

    with patch("src.agents.llm_backend.Runner", create=True) as mock_runner:
        mock_runner.run.side_effect = always_permanent
        with patch("src.agents.llm_backend.asyncio.sleep", return_value=None):
            with pytest.raises(_Permanent):
                await run_agent(
                    agent=None, prompt="",
                    max_retries=3, initial_delay=0.0,
                    retriable=(_Transient,),  # _Permanent is NOT in this tuple
                )

    assert calls["n"] == 1, "Permanent errors must not trigger retries"


@pytest.mark.asyncio
async def test_backoff_is_exponential_with_jitter():
    """Sleep schedule must grow geometrically (1× → 2× → 4× of initial_delay)
    with ±25% jitter. Use a fixed base so we can assert on the bands."""
    sleep_calls: list[float] = []

    async def always_transient(*_a, **_kw):
        raise _Transient("transient")

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    with patch("src.agents.llm_backend.Runner", create=True) as mock_runner:
        mock_runner.run.side_effect = always_transient
        with patch("src.agents.llm_backend.asyncio.sleep", side_effect=fake_sleep):
            await run_agent(
                agent=None, prompt="",
                max_retries=4, initial_delay=1.0,
                retriable=(_Transient,),
            )

    # 4 attempts → 3 sleeps (no sleep after the last, it gives up).
    assert len(sleep_calls) == 3
    # Jitter is ±25% around base = 1, 2, 4.
    assert 0.75 <= sleep_calls[0] <= 1.25
    assert 1.50 <= sleep_calls[1] <= 2.50
    assert 3.00 <= sleep_calls[2] <= 5.00


# ── submit-tool helpers (shared by all submit_* tool factories) ───────────


def test_submit_ok_sentinel_is_terminal_instruction():
    """The sentinel must instruct the LLM to emit a brief plain-text reply
    and stop calling tools — that's what terminates the SDK loop cleanly.
    Drift in this string across agents would silently break the loop in
    one place but not another."""
    from src.agents.llm_backend import SUBMIT_OK_SENTINEL

    text = SUBMIT_OK_SENTINEL.lower()
    assert "submitted" in text
    assert "plain-text" in text or "plain text" in text
    assert "no further tool" in text or "no other tool" in text


def test_format_submit_validation_error_contains_tool_name_and_exc_message():
    """Standard error string returned by submit_* tools on Pydantic
    validation failure. The SDK hands this back to the LLM as the
    tool-call response so the model can self-correct in-loop."""
    from pydantic import BaseModel, Field, ValidationError

    from src.agents.llm_backend import format_submit_validation_error

    class Foo(BaseModel):
        x: int = Field(ge=0)

    try:
        Foo(x=-1)
    except ValidationError as exc:
        msg = format_submit_validation_error("submit_foo", exc)
        assert msg.startswith("submit_foo FAILED:")
        assert "x" in msg  # Pydantic includes the offending field name
    else:
        raise AssertionError("ValidationError should have been raised")


# ── load_model_config: api_key sourcing ───────────────────────────────


def test_load_model_config_uses_literal_api_key_when_present(tmp_path, monkeypatch):
    """A JSON-supplied api_key wins over the env var — preserves the
    pre-existing path so old configs (deepseek.json with a hardcoded key)
    keep working unchanged."""
    import json as _json

    from src.agents.llm_backend import load_model_config

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({
        "model": "m", "url": "https://x.example/v1", "api_key": "sk-literal",
    }))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-should-be-ignored")

    result = load_model_config(cfg)
    assert result.api_key == "sk-literal"


def test_load_model_config_falls_back_to_env_when_api_key_absent(tmp_path, monkeypatch):
    """When the JSON omits ``api_key`` entirely, the loader pulls from
    ``$OPENAI_API_KEY`` — this is the recommended path for committed
    configs (e.g. configs/models/kimi.json) so the secret never lands
    in the working tree."""
    import json as _json

    from src.agents.llm_backend import load_model_config

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({"model": "m", "url": "https://x.example/v1"}))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    result = load_model_config(cfg)
    assert result.api_key == "sk-from-env"


def test_load_model_config_falls_back_to_env_when_api_key_empty_string(tmp_path, monkeypatch):
    """Empty-string api_key is treated as 'not supplied' — covers the
    config-with-placeholder case ('api_key': '') that might land if a
    user clears the field manually."""
    import json as _json

    from src.agents.llm_backend import load_model_config

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({
        "model": "m", "url": "https://x.example/v1", "api_key": "",
    }))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    result = load_model_config(cfg)
    assert result.api_key == "sk-from-env"


def test_load_model_config_raises_when_neither_source_has_key(tmp_path, monkeypatch):
    """No literal key + no env var → ValueError that names both options
    so the operator knows how to recover. Auth errors at request time
    are confusing; failing here at load-time is the better failure mode."""
    import json as _json

    from src.agents.llm_backend import load_model_config

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({"model": "m", "url": "https://x.example/v1"}))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_model_config(cfg)


# ── force_temperature override ────────────────────────────────────────


def test_force_temperature_in_config_overrides_per_agent_value(tmp_path, monkeypatch):
    """When the model config carries ``force_temperature``, the value
    must override whatever the agent passes to ``make_run_config`` —
    required for Kimi-K2 / o1 / DeepSeek-Reasoner which reject temp ≠ 1.0."""
    import json as _json

    from src.agents import llm_backend as lb

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({
        "model": "kimi-k2.6", "url": "https://api.moonshot.cn/v1",
        "api_key": "sk-x", "force_temperature": 1.0,
    }))
    saved = lb._FORCE_TEMPERATURE
    try:
        result = lb.load_model_config(cfg)
        assert result.force_temperature == 1.0

        # The Coder pins 0.0 normally; with the override active that must
        # become 1.0 when synthesizing the RunConfig.
        with patch.object(lb, "ModelSettings", create=True) as ms, \
             patch.object(lb, "RunConfig", create=True):
            lb.make_run_config(temperature=0.0)
            ms.assert_called_once()
            assert ms.call_args.kwargs["temperature"] == 1.0
    finally:
        lb._FORCE_TEMPERATURE = saved


def test_no_force_temperature_preserves_per_agent_value(tmp_path, monkeypatch):
    """When ``force_temperature`` is absent from the JSON, ``make_run_config``
    must use the value the agent passes — Planner/Reviewer at 0.3, Coder at
    0.0. The override stays None on this load path."""
    import json as _json

    from src.agents import llm_backend as lb

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({
        "model": "m", "url": "https://x.example/v1", "api_key": "sk-x",
    }))
    saved = lb._FORCE_TEMPERATURE
    try:
        # Prime the module global with a stale value to make sure the load
        # path clears it when the new config has no force_temperature.
        lb._FORCE_TEMPERATURE = 1.0
        result = lb.load_model_config(cfg)
        assert result.force_temperature is None
        assert lb._FORCE_TEMPERATURE is None

        with patch.object(lb, "ModelSettings", create=True) as ms, \
             patch.object(lb, "RunConfig", create=True):
            lb.make_run_config(temperature=0.3)
            assert ms.call_args.kwargs["temperature"] == 0.3
    finally:
        lb._FORCE_TEMPERATURE = saved


# ── max_tokens override ───────────────────────────────────────────────


def test_max_tokens_in_config_overrides_default(tmp_path, monkeypatch):
    """``max_tokens`` from the model config must override
    ``make_run_config``'s 4096 default — required for long-context models
    like Kimi-K2 (256k) where the default would truncate legitimate output."""
    import json as _json

    from src.agents import llm_backend as lb

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({
        "model": "kimi-k2.6", "url": "https://api.moonshot.cn/v1",
        "api_key": "sk-x", "max_tokens": 262144,
    }))
    saved = lb._MAX_TOKENS_OVERRIDE
    try:
        result = lb.load_model_config(cfg)
        assert result.max_tokens == 262144

        with patch.object(lb, "ModelSettings", create=True) as ms, \
             patch.object(lb, "RunConfig", create=True):
            lb.make_run_config()  # Default max_tokens=4096
            ms.assert_called_once()
            assert ms.call_args.kwargs["max_tokens"] == 262144
    finally:
        lb._MAX_TOKENS_OVERRIDE = saved


def test_no_max_tokens_in_config_preserves_default(tmp_path, monkeypatch):
    """When ``max_tokens`` is absent from the JSON, the
    ``make_run_config(max_tokens=4096)`` default applies — and a stale
    module global from a previous load is cleared."""
    import json as _json

    from src.agents import llm_backend as lb

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({
        "model": "m", "url": "https://x.example/v1", "api_key": "sk-x",
    }))
    saved = lb._MAX_TOKENS_OVERRIDE
    try:
        lb._MAX_TOKENS_OVERRIDE = 262144  # Stale carryover
        result = lb.load_model_config(cfg)
        assert result.max_tokens is None
        assert lb._MAX_TOKENS_OVERRIDE is None

        with patch.object(lb, "ModelSettings", create=True) as ms, \
             patch.object(lb, "RunConfig", create=True):
            lb.make_run_config()
            assert ms.call_args.kwargs["max_tokens"] == 4096
    finally:
        lb._MAX_TOKENS_OVERRIDE = saved


# ── reasoning_effort + extra_body overrides ────────────────────────────


def test_reasoning_effort_in_config_threads_through_model_settings(tmp_path, monkeypatch):
    """When ``reasoning_effort`` is in the JSON, ``make_run_config`` must
    pass ``ModelSettings(reasoning=Reasoning(effort=...))`` so the SDK
    forwards it as the ``reasoning_effort`` kwarg to the provider — required
    for DeepSeek-v4-pro thinking mode and other reasoning-tier models."""
    import json as _json

    from src.agents import llm_backend as lb

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({
        "model": "deepseek-v4-pro",
        "url": "https://api.deepseek.com/v1",
        "api_key": "sk-x",
        "reasoning_effort": "high",
    }))
    saved = lb._REASONING_EFFORT_OVERRIDE
    try:
        result = lb.load_model_config(cfg)
        assert result.reasoning_effort == "high"
        assert lb._REASONING_EFFORT_OVERRIDE == "high"

        with patch.object(lb, "ModelSettings", create=True) as ms, \
             patch.object(lb, "RunConfig", create=True):
            lb.make_run_config()
            reasoning_arg = ms.call_args.kwargs["reasoning"]
            assert reasoning_arg is not None
            # Reasoning object carries the effort string.
            assert getattr(reasoning_arg, "effort", None) == "high"
    finally:
        lb._REASONING_EFFORT_OVERRIDE = saved


def test_no_reasoning_effort_keeps_model_settings_reasoning_unset(tmp_path, monkeypatch):
    """Without ``reasoning_effort`` in the JSON, ``make_run_config`` must
    pass ``reasoning=None`` so non-reasoning models (chat-tier) aren't
    forced into a mode they don't support."""
    import json as _json

    from src.agents import llm_backend as lb

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({
        "model": "deepseek-chat", "url": "https://api.deepseek.com/v1", "api_key": "sk-x",
    }))
    saved = lb._REASONING_EFFORT_OVERRIDE
    try:
        lb._REASONING_EFFORT_OVERRIDE = "high"  # Stale carryover
        result = lb.load_model_config(cfg)
        assert result.reasoning_effort is None
        assert lb._REASONING_EFFORT_OVERRIDE is None

        with patch.object(lb, "ModelSettings", create=True) as ms, \
             patch.object(lb, "RunConfig", create=True):
            lb.make_run_config()
            assert ms.call_args.kwargs["reasoning"] is None
    finally:
        lb._REASONING_EFFORT_OVERRIDE = saved


def test_extra_body_in_config_threads_through_model_settings(tmp_path, monkeypatch):
    """When ``extra_body`` is in the JSON (provider-specific extensions
    like DeepSeek's ``thinking`` dict), it must be passed through to
    ``ModelSettings`` so the openai SDK forwards it on every request."""
    import json as _json

    from src.agents import llm_backend as lb

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({
        "model": "deepseek-v4-pro",
        "url": "https://api.deepseek.com/v1",
        "api_key": "sk-x",
        "extra_body": {"thinking": {"type": "enabled"}},
    }))
    saved = lb._EXTRA_BODY_OVERRIDE
    try:
        result = lb.load_model_config(cfg)
        assert result.extra_body == {"thinking": {"type": "enabled"}}
        assert lb._EXTRA_BODY_OVERRIDE == {"thinking": {"type": "enabled"}}

        with patch.object(lb, "ModelSettings", create=True) as ms, \
             patch.object(lb, "RunConfig", create=True):
            lb.make_run_config()
            assert ms.call_args.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    finally:
        lb._EXTRA_BODY_OVERRIDE = saved


def test_load_model_config_falls_back_to_deepseek_env_var(tmp_path, monkeypatch):
    """Third api_key fallback: $DEEPSEEK_API_KEY. Mirrors the DeepSeek SDK
    snippet's own env-var convention; tried after JSON literal and
    $OPENAI_API_KEY are absent / empty."""
    import json as _json

    from src.agents import llm_backend as lb

    cfg = tmp_path / "model.json"
    cfg.write_text(_json.dumps({
        "model": "deepseek-v4-pro", "url": "https://api.deepseek.com/v1",
        # No api_key in JSON.
    }))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-deepseek-env")
    result = lb.load_model_config(cfg)
    assert result.api_key == "sk-from-deepseek-env"
