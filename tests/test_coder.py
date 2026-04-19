"""Tests for agents/coder.py — Coder agent with tool-using LLM loop.

These tests exercise the tool factories and the `implement()` flow
without requiring `torch` or the OpenAI Agents SDK. The factory
closures delegate to `src.kernels.compiler.compile_kernel` (already
covered by `test_compiler.py`) and `src.eval.correctness.verify_correctness`
(covered by `test_correctness.py`), so tests here focus on wiring and
error-string shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.coder import (
    CoderAgent,
    ImplementationError,
    KernelCodeOutput,
    _make_compile_tool,
    _make_correctness_tool,
)
from src.agents.planner import OptimizationPlan
from src.config import ACTSConfig
from tests.conftest import (
    ScalarPolicy as _ScalarPolicy,
    make_kernel_spec as _make_spec,
    scalar_gen as _gen,
    scalar_ref as _ref,
)


# ── Pydantic output model ──────────────────────────────────────────────


def test_output_model_accepts_valid_data():
    out = KernelCodeOutput(source_code="@triton.jit\ndef k(): pass")
    assert out.source_code.startswith("@triton.jit")


def test_output_model_requires_source_code():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KernelCodeOutput()  # type: ignore[call-arg]


# ── prompt assembly ─────────────────────────────────────────────────────


def test_build_user_prompt_contains_all_sections():
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
    plan = OptimizationPlan(
        tier=1,
        technique="t1_occupancy",
        target_region="launch config",
        rationale="Increase occupancy.",
    )
    prompt = CoderAgent.build_user_prompt(kernel_source="def k(): pass", plan=plan)
    assert "Params:" not in prompt


def test_build_user_prompt_escapes_backticks_in_kernel_source():
    plan = OptimizationPlan(tier=1, technique="t1", rationale="x")
    source = 'def kernel():\n    """```python\n    fake section\n    ```"""\n    pass'
    prompt = CoderAgent.build_user_prompt(kernel_source=source, plan=plan)
    sections = prompt.split("## ")
    kernel_section = [s for s in sections if s.startswith("Current kernel")][0]
    assert "```python\nfake section\n```" not in kernel_section


# ── compile tool factory ────────────────────────────────────────────────


def test_compile_tool_factory_returns_callable(tmp_path):
    tool = _make_compile_tool(_make_spec(), cache_dir=tmp_path)
    assert callable(tool)


def test_compile_tool_reports_success_on_good_source(tmp_path):
    tool = _make_compile_tool(_make_spec(), cache_dir=tmp_path)
    msg = tool("def kernel_fn(x):\n    return x + 1\n")
    assert "success" in msg.lower()
    assert "kernel_fn" in msg  # entrypoint surfaced so Coder knows what it resolved


def test_compile_tool_reports_error_on_syntax_error(tmp_path):
    tool = _make_compile_tool(_make_spec(), cache_dir=tmp_path)
    msg = tool("def kernel_fn(: invalid\n")
    assert "fail" in msg.lower() or "error" in msg.lower()
    assert "SyntaxError" in msg


def test_compile_tool_reports_error_on_missing_entrypoint(tmp_path):
    tool = _make_compile_tool(_make_spec(entrypoint="run"), cache_dir=tmp_path)
    msg = tool("def kernel_fn(x): return x\n")  # wrong symbol name
    assert "run" in msg  # the missing entrypoint name


# ── correctness tool factory ────────────────────────────────────────────


def test_correctness_tool_factory_returns_callable(tmp_path):
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generator=_gen,
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
    )
    assert callable(tool)


def test_correctness_tool_reports_compile_error_without_running_correctness(tmp_path):
    """If the candidate source won't compile, surface that — don't try to run it."""
    calls = {"ref": 0}

    def ref(x):
        calls["ref"] += 1
        return x * 2.0

    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=ref,
        input_generator=_gen,
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
    )
    msg = tool("def kernel_fn(: broken\n")
    assert "compile" in msg.lower()
    assert calls["ref"] == 0  # reference was never invoked


def test_correctness_tool_reports_success_on_matching_candidate(tmp_path):
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generator=_gen,
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
    )
    msg = tool("def kernel_fn(x):\n    return x * 2.0\n")
    assert "pass" in msg.lower()


def test_correctness_tool_reports_failure_stage_on_mismatch(tmp_path):
    """Failure messages surface the failed stage so the Coder can diagnose."""
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generator=_gen,
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
    )
    msg = tool("def kernel_fn(x):\n    return x * 3.0\n")
    assert "fail" in msg.lower()
    assert "smoke_test" in msg  # first-stage failure for a uniformly-wrong candidate


# ── implement() — placeholder path (no model) ───────────────────────────


@pytest.mark.asyncio
async def test_implement_without_model_returns_source_unchanged():
    agent = CoderAgent(model=None)
    plan = OptimizationPlan(tier=1, technique="t1_occupancy")
    src = "@triton.jit\ndef k(): ..."
    out = await agent.implement(
        kernel_source=src,
        plan=plan,
        kernel_spec=_make_spec(),
        reference_fn=_ref,
        input_generator=_gen,
    )
    assert out == src


# ── implement() — LLM path (mocked) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_implement_calls_llm_and_returns_modified_source():
    """With a model, implement() builds the Agent with bound tools and runs it."""
    mock_output = KernelCodeOutput(source_code="@triton.jit\ndef k(): return 42")
    mock_result = MagicMock()
    mock_result.final_output = mock_output

    with (
        patch("src.agents.coder.Agent") as mock_agent_cls,
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = mock_result

        agent = CoderAgent(model=MagicMock())
        plan = OptimizationPlan(
            tier=1,
            technique="t1_block_size_tuning",
            params={"block_size": "128"},
            target_region="main loop",
            rationale="Bigger tile => more reuse.",
        )
        result = await agent.implement(
            kernel_source="original source",
            plan=plan,
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generator=_gen,
        )

    assert result == "@triton.jit\ndef k(): return 42"
    mock_run.assert_awaited_once()
    # Agent was constructed with exactly two tools (compile + correctness).
    kwargs = mock_agent_cls.call_args.kwargs
    assert len(kwargs["tools"]) == 2


@pytest.mark.asyncio
async def test_implement_raises_on_llm_failure():
    """If run_agent returns None (retries exhausted), raise ImplementationError."""
    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = None

        agent = CoderAgent(model=MagicMock())
        plan = OptimizationPlan(tier=1, technique="t1")
        with pytest.raises(ImplementationError, match="LLM"):
            await agent.implement(
                kernel_source="src",
                plan=plan,
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generator=_gen,
            )


@pytest.mark.asyncio
async def test_implement_passes_default_max_turns_when_no_config():
    """No config → default ACTSConfig.max_debug_retries=3 → max_turns = 2*3+1 = 7."""
    mock_result = MagicMock()
    mock_result.final_output = KernelCodeOutput(source_code="ok")

    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = mock_result

        agent = CoderAgent(model=MagicMock())
        await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generator=_gen,
        )

    assert mock_run.await_args.kwargs.get("max_turns") == 7


@pytest.mark.asyncio
async def test_implement_max_turns_derived_from_config():
    """max_debug_retries=5 → max_turns = 2*5+1 = 11."""
    mock_result = MagicMock()
    mock_result.final_output = KernelCodeOutput(source_code="ok")

    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = mock_result

        agent = CoderAgent(model=MagicMock(), config=ACTSConfig(max_debug_retries=5))
        await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generator=_gen,
        )

    assert mock_run.await_args.kwargs.get("max_turns") == 11


@pytest.mark.asyncio
async def test_implement_uses_zero_temperature():
    """Coder runs with temperature=0.0 — deterministic code generation."""
    mock_result = MagicMock()
    mock_result.final_output = KernelCodeOutput(source_code="ok")

    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config") as mock_cfg,
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = mock_result
        mock_cfg.return_value = None

        agent = CoderAgent(model=MagicMock())
        await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generator=_gen,
        )

    mock_cfg.assert_called_once_with(temperature=0.0)
