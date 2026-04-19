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
        input_generators=[_gen],
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
        input_generators=[_gen],
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
        input_generators=[_gen],
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
        input_generators=[_gen],
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
    )
    msg = tool("def kernel_fn(x):\n    return x * 3.0\n")
    assert "fail" in msg.lower()
    assert "smoke_test" in msg  # first-stage failure for a uniformly-wrong candidate


def test_correctness_tool_empty_generators_raises():
    """Building a correctness tool with no workloads is a contract violation."""
    with pytest.raises(ValueError, match="generator"):
        _make_correctness_tool(
            _make_spec(),
            reference_fn=_ref,
            input_generators=[],
            policy=_ScalarPolicy(),
        )


def test_correctness_tool_iterates_all_generators_and_reports_first_failure(tmp_path):
    """Tool must run each generator until one fails — its output tells the Coder
    which workload broke so retries can actually correct multi-workload bugs."""
    from src.eval.correctness import CorrectnessResult, CorrectnessStage

    gens = [MagicMock(name=f"gen_{i}") for i in range(3)]
    results = [
        CorrectnessResult(passed=True, max_abs_error=0.0),
        CorrectnessResult(
            passed=False,
            failed_stage=CorrectnessStage.NUMERICAL_STABILITY,
            error_message="numerical mismatch",
            max_abs_error=1.0,
        ),
    ]
    with patch("src.agents.coder.verify_correctness", side_effect=results) as mock_verify:
        tool = _make_correctness_tool(
            _make_spec(),
            reference_fn=_ref,
            input_generators=gens,
            cache_dir=tmp_path,
        )
        msg = tool("def kernel_fn(x):\n    return x * 2.0\n")

    assert mock_verify.call_count == 2  # short-circuit after first failure
    assert "workload 2" in msg.lower()
    assert "numerical_stability" in msg


def test_correctness_tool_reports_success_when_all_generators_pass(tmp_path):
    """All workloads clean → single success message (not one per workload)."""
    from src.eval.correctness import CorrectnessResult

    gens = [MagicMock(name=f"gen_{i}") for i in range(3)]
    with patch(
        "src.agents.coder.verify_correctness",
        return_value=CorrectnessResult(passed=True, max_abs_error=0.0),
    ) as mock_verify:
        tool = _make_correctness_tool(
            _make_spec(),
            reference_fn=_ref,
            input_generators=gens,
            cache_dir=tmp_path,
        )
        msg = tool("def kernel_fn(x):\n    return x * 2.0\n")

    assert mock_verify.call_count == 3
    assert "pass" in msg.lower()


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
        input_generators=[_gen],
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
            input_generators=[_gen],
        )

    assert result == "@triton.jit\ndef k(): return 42"
    mock_run.assert_awaited_once()
    # Agent was constructed with exactly two tools (compile + correctness).
    kwargs = mock_agent_cls.call_args.kwargs
    assert len(kwargs["tools"]) == 2


@pytest.mark.asyncio
async def test_implement_raises_when_input_generators_missing_or_empty():
    """A model-backed implement() cannot bind its correctness tool without at least
    one generator — refuse fast so Phase B doesn't silently score broken children."""
    agent = CoderAgent(model=MagicMock())
    plan = OptimizationPlan(tier=1, technique="t1")

    with pytest.raises(ImplementationError, match="input_generators"):
        await agent.implement(
            kernel_source="src",
            plan=plan,
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=None,
        )

    with pytest.raises(ImplementationError, match="input_generators"):
        await agent.implement(
            kernel_source="src",
            plan=plan,
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[],
        )


@pytest.mark.asyncio
async def test_implement_binds_all_generators_to_correctness_tool():
    """Phase B must bind every selected workload's generator to the correctness
    tool — else a kernel that passes workload 1 but breaks 2..N slips through."""
    gens = [MagicMock(name=f"gen_{i}") for i in range(3)]
    mock_result = MagicMock(final_output=KernelCodeOutput(source_code="ok"))

    captured = {}
    def capture_factory(*args, **kwargs):
        captured["input_generators"] = kwargs["input_generators"]
        return lambda src: "passed"

    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
        patch("src.agents.coder._make_correctness_tool", side_effect=capture_factory),
    ):
        mock_run.return_value = mock_result
        agent = CoderAgent(model=MagicMock())
        await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=gens,
        )

    assert captured["input_generators"] is gens


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
                input_generators=[_gen],
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
            input_generators=[_gen],
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
            input_generators=[_gen],
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
            input_generators=[_gen],
        )

    mock_cfg.assert_called_once_with(temperature=0.0)


# ── has_model property ─────────────────────────────────────────────────


def test_has_model_reflects_configuration():
    """baseline_generator branches on has_model — must be a public, stable signal."""
    assert CoderAgent(model=None).has_model is False
    assert CoderAgent(model=MagicMock()).has_model is True


# ── translate() — PyTorch→Triton one-shot port ─────────────────────────


@pytest.mark.asyncio
async def test_translate_without_model_raises():
    """translate() has no sensible no-op fallback — a model is required."""
    agent = CoderAgent(model=None)
    with pytest.raises(ImplementationError, match="model"):
        await agent.translate(
            reference_source="def run(x):\n    return x * 2.0\n",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
        )


@pytest.mark.asyncio
async def test_translate_builds_agent_with_two_tools_and_returns_source():
    """translate() constructs a fresh Agent with compile + correctness tools."""
    mock_output = KernelCodeOutput(source_code="@triton.jit\ndef kernel_fn(x): pass")
    mock_result = MagicMock(final_output=mock_output)

    with (
        patch("src.agents.coder.Agent") as mock_agent_cls,
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = mock_result
        agent = CoderAgent(model=MagicMock())
        result = await agent.translate(
            reference_source="def run(x):\n    return x * 2.0\n",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
        )

    assert result == "@triton.jit\ndef kernel_fn(x): pass"
    mock_run.assert_awaited_once()
    kwargs = mock_agent_cls.call_args.kwargs
    assert len(kwargs["tools"]) == 2  # compile + correctness


@pytest.mark.asyncio
async def test_translate_user_prompt_contains_reference_and_entrypoint():
    """Prompt must surface the source to translate and the target entrypoint."""
    mock_result = MagicMock(final_output=KernelCodeOutput(source_code="ok"))
    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = mock_result
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x):\n    return x * 2.0\n",
            kernel_spec=_make_spec(entrypoint="my_kernel"),
            reference_fn=_ref,
            input_generators=[_gen],
        )

    prompt = mock_run.await_args.args[1]
    assert "def run(x)" in prompt
    assert "my_kernel" in prompt


@pytest.mark.asyncio
async def test_translate_uses_distinct_translate_instructions():
    """translate() loads translate.md — separate from the optimize system.md."""
    mock_result = MagicMock(final_output=KernelCodeOutput(source_code="ok"))
    with (
        patch("src.agents.coder.Agent") as mock_agent_cls,
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = mock_result
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x): return x",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
        )

    instructions = mock_agent_cls.call_args.kwargs["instructions"]
    # translate.md is the from-scratch port prompt; must surface both dialects.
    assert "Triton" in instructions
    assert "PyTorch" in instructions
    # Must NOT carry optimize-mode framing that contradicts the translation task.
    assert "one focused change" not in instructions.lower()


@pytest.mark.asyncio
async def test_translate_raises_on_llm_failure():
    """run_agent returning None (retries exhausted) → ImplementationError."""
    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = None
        agent = CoderAgent(model=MagicMock())
        with pytest.raises(ImplementationError, match="LLM"):
            await agent.translate(
                reference_source="def run(x): return x",
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
            )


@pytest.mark.asyncio
async def test_translate_uses_zero_temperature():
    """Like implement(), translate() pins temperature=0.0 for determinism."""
    mock_result = MagicMock(final_output=KernelCodeOutput(source_code="ok"))
    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config") as mock_cfg,
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = mock_result
        mock_cfg.return_value = None
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x): return x",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
        )

    mock_cfg.assert_called_once_with(temperature=0.0)
