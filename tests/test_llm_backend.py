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
