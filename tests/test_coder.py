"""Tests for agents/coder.py — Coder agent with tool-using LLM loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.coder import (
    CoderAgent,
    ImplementationError,
    KernelCodeOutput,
)
from src.agents.planner import OptimizationPlan


# ── Pydantic output model ──────────────────────────────────────────────


def test_output_model_accepts_valid_data():
    """KernelCodeOutput parses valid code."""
    out = KernelCodeOutput(source_code="@triton.jit\ndef k(): pass")
    assert out.source_code.startswith("@triton.jit")


def test_output_model_requires_source_code():
    """KernelCodeOutput rejects missing source_code."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KernelCodeOutput()  # type: ignore[call-arg]


# ── prompt assembly ─────────────────────────────────────────────────────


def test_build_user_prompt_contains_all_sections():
    """The assembled user prompt includes the kernel source and all plan fields."""
    plan = OptimizationPlan(
        tier=2,
        technique="t2_shared_memory_tiling",
        params={"tile_m": "64", "tile_n": "64"},
        target_region="inner loop",
        rationale="Reduce DRAM traffic via tiling.",
    )
    prompt = CoderAgent.build_user_prompt(
        kernel_source="@triton.jit\ndef matmul_kernel(): ...",
        plan=plan,
    )
    assert "@triton.jit" in prompt
    assert "Tier: 2" in prompt
    assert "t2_shared_memory_tiling" in prompt
    assert "tile_m=64" in prompt
    assert "tile_n=64" in prompt
    assert "inner loop" in prompt
    assert "Reduce DRAM traffic" in prompt


def test_build_user_prompt_omits_empty_params():
    """Params section is omitted when plan.params is empty."""
    plan = OptimizationPlan(
        tier=1,
        technique="t1_occupancy",
        target_region="launch config",
        rationale="Increase occupancy.",
    )
    prompt = CoderAgent.build_user_prompt(kernel_source="def k(): pass", plan=plan)
    assert "Params:" not in prompt


def test_build_user_prompt_escapes_backticks_in_kernel_source():
    """Triple backticks in kernel source are escaped so the fence stays closed."""
    plan = OptimizationPlan(tier=1, technique="t1", rationale="x")
    source = 'def kernel():\n    """```python\n    fake section\n    ```"""\n    pass'
    prompt = CoderAgent.build_user_prompt(kernel_source=source, plan=plan)
    sections = prompt.split("## ")
    kernel_section = [s for s in sections if s.startswith("Current kernel")][0]
    assert "```python\nfake section\n```" not in kernel_section


# ── tool stubs ──────────────────────────────────────────────────────────


def test_compile_tool_placeholder_returns_success_string():
    """The placeholder compile tool returns a non-error string so the LLM
    can proceed. Real compilation lands when kernels/compiler.py does."""
    from src.agents.coder import _compile_kernel_tool

    result = _compile_kernel_tool("@triton.jit\ndef k(): pass")
    assert "success" in result.lower()


def test_correctness_tool_placeholder_returns_pass_string():
    """Placeholder correctness tool returns a pass-ish string."""
    from src.agents.coder import _check_correctness_tool

    result = _check_correctness_tool("@triton.jit\ndef k(): pass")
    assert "pass" in result.lower()


# ── implement() — placeholder path (no model) ───────────────────────────


@pytest.mark.asyncio
async def test_implement_without_model_returns_source_unchanged():
    """Without a configured model, implement() is a no-op — returns input source."""
    agent = CoderAgent(model=None)
    plan = OptimizationPlan(tier=1, technique="t1_occupancy")
    src = "@triton.jit\ndef k(): ..."
    out = await agent.implement(kernel_source=src, plan=plan)
    assert out == src


# ── implement() — LLM path (mocked) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_implement_calls_llm_and_returns_modified_source():
    """With a model, implement() calls the LLM and returns final_output.source_code."""
    mock_output = KernelCodeOutput(source_code="@triton.jit\ndef k(): return 42")
    mock_result = MagicMock()
    mock_result.final_output = mock_output

    with (
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
    ):
        mock_run.return_value = mock_result

        agent = CoderAgent(model=None)
        agent._agent = MagicMock()  # non-None triggers LLM path

        plan = OptimizationPlan(
            tier=1,
            technique="t1_block_size_tuning",
            params={"block_size": "128"},
            target_region="main loop",
            rationale="Bigger tile => more reuse.",
        )
        result = await agent.implement(kernel_source="original source", plan=plan)

    assert result == "@triton.jit\ndef k(): return 42"
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_implement_raises_on_llm_failure():
    """If run_agent returns None (retries exhausted), raise ImplementationError."""
    with (
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
    ):
        mock_run.return_value = None

        agent = CoderAgent(model=None)
        agent._agent = MagicMock()

        plan = OptimizationPlan(tier=1, technique="t1")
        with pytest.raises(ImplementationError, match="LLM"):
            await agent.implement(kernel_source="src", plan=plan)


@pytest.mark.asyncio
async def test_implement_passes_max_turns_to_bound_self_correction():
    """Coder passes max_turns=7 so the tool loop can't run unbounded.

    Budget: 3 compile+correctness cycles (2 tool turns each) + 1 final output.
    """
    mock_output = KernelCodeOutput(source_code="modified")
    mock_result = MagicMock()
    mock_result.final_output = mock_output

    with (
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
    ):
        mock_run.return_value = mock_result

        agent = CoderAgent(model=None)
        agent._agent = MagicMock()

        plan = OptimizationPlan(tier=1, technique="t1")
        await agent.implement(kernel_source="src", plan=plan)

    kwargs = mock_run.await_args.kwargs
    assert kwargs.get("max_turns") == 7


@pytest.mark.asyncio
async def test_implement_uses_zero_temperature():
    """Coder runs with temperature=0.0 — deterministic code generation."""
    mock_output = KernelCodeOutput(source_code="modified")
    mock_result = MagicMock()
    mock_result.final_output = mock_output

    with (
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config") as mock_cfg,
    ):
        mock_run.return_value = mock_result
        mock_cfg.return_value = None

        agent = CoderAgent(model=None)
        agent._agent = MagicMock()

        plan = OptimizationPlan(tier=1, technique="t1")
        await agent.implement(kernel_source="src", plan=plan)

    mock_cfg.assert_called_once_with(temperature=0.0)
