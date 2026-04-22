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


# ── test helpers for the option-α submit-tool flow ─────────────────────
#
# The Coder used to expose its final answer via Pydantic ``output_type=``
# enforcement; option α (DeepSeek-reasoner compatibility) routes the same
# (source, name) pair through a ``submit_kernel`` tool call instead. To
# simulate the LLM calling that tool inside the SDK loop, tests pair a
# synthetic ``Agent`` factory that captures the constructed tools list
# with a synthetic ``run_agent`` that finds ``submit_kernel`` in that
# list and invokes it directly.


def _simulate_submission(source_code: str, triton_kernel_name: str):
    """Return ``(capture_agent, fake_run_agent)`` patch side-effects that
    simulate the LLM calling ``submit_kernel(source_code, triton_kernel_name)``
    once during the agent run.

    The captured tools list survives across both side-effects via closure
    so the second side-effect can find ``submit_kernel`` even though the
    test patches ``Agent`` itself out (so ``agent.tools`` on the mock isn't
    populated).
    """
    captured_tools: list[list] = []

    def capture_agent(*args, **kwargs):
        captured_tools.append(kwargs.get("tools", []))
        return MagicMock()

    async def fake_run_agent(agent, prompt, **kwargs):
        for tool in captured_tools[-1]:
            if getattr(tool, "__name__", "") == "submit_kernel":
                tool(
                    source_code=source_code,
                    triton_kernel_name=triton_kernel_name,
                )
                break
        # The SDK's ``RunResult.final_output`` is a plain text confirmation
        # under option α — coder.py reads from the captured submission, not
        # from ``result.final_output``, so its content is irrelevant.
        return MagicMock(final_output="done")

    return capture_agent, fake_run_agent


_VALID_SOURCE = "@triton.jit\ndef k(): pass"
_VALID_NAME = "k"


# ── Pydantic output model ──────────────────────────────────────────────


def test_output_model_accepts_valid_data():
    out = KernelCodeOutput(
        source_code="@triton.jit\ndef k(): pass",
        triton_kernel_name="k",
    )
    assert out.source_code.startswith("@triton.jit")
    assert out.triton_kernel_name == "k"


def test_output_model_requires_source_code():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KernelCodeOutput()  # type: ignore[call-arg]


def test_output_model_requires_triton_kernel_name():
    """T4: Coder must declare which @triton.jit kernel NCU should profile.
    Empty / missing triton_kernel_name is a Pydantic validation failure
    so the SDK's tool loop retries within the existing turn budget."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KernelCodeOutput(source_code="@triton.jit\ndef k(): pass")  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="required"):
        KernelCodeOutput(
            source_code="@triton.jit\ndef k(): pass",
            triton_kernel_name="",
        )


def test_output_model_rejects_kernel_name_not_in_source():
    """The declared kernel name must literally appear in source_code as
    ``@triton.jit\\ndef <name>``. Mismatch → silent NCU mis-profile in
    production, so we surface it as a validation failure."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="not found"):
        KernelCodeOutput(
            source_code="@triton.jit\ndef actual_name(): pass",
            triton_kernel_name="claimed_name",
        )


def test_output_model_rejects_source_with_no_triton_jit():
    """The Coder writes Triton kernels — pure-PyTorch source means it
    skipped its job. Reject before the kernel reaches the orchestrator."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="@triton.jit"):
        KernelCodeOutput(
            source_code="def run(x):\n    return x * 2.0\n",
            triton_kernel_name="run",
        )


def test_output_model_accepts_multiple_jit_defs_with_matching_name():
    """Fused kernels can declare ``@triton.jit`` helpers alongside the main
    kernel. The Coder picks the dominant-work kernel; we only verify the
    declared name is one of the jit'd defs (not necessarily the first)."""
    src = (
        "@triton.jit\ndef _epilogue(): pass\n"
        "@triton.jit\ndef main_kernel(): pass\n"
    )
    out = KernelCodeOutput(source_code=src, triton_kernel_name="main_kernel")
    assert out.triton_kernel_name == "main_kernel"

    # The helper is also a valid choice — Coder is the source of truth.
    out2 = KernelCodeOutput(source_code=src, triton_kernel_name="_epilogue")
    assert out2.triton_kernel_name == "_epilogue"


def test_output_model_jit_decorator_with_args_recognized():
    """``@triton.jit(do_not_specialize=...)`` should still match — the
    regex tolerates decorator arguments."""
    src = "@triton.jit(do_not_specialize=['n'])\ndef tuned_kernel(n): pass"
    out = KernelCodeOutput(source_code=src, triton_kernel_name="tuned_kernel")
    assert out.triton_kernel_name == "tuned_kernel"


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
    assert isinstance(out, KernelCodeOutput)
    assert out.source_code == src
    # No-model placeholder path can't declare a kernel name (no LLM to ask) —
    # downstream profiler falls back to source-regex extraction when this is empty.
    assert out.triton_kernel_name == ""


# ── implement() — LLM path (mocked) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_implement_calls_llm_and_returns_modified_source():
    """With a model, implement() builds the Agent with bound tools and runs it.
    The Coder emits its answer by calling submit_kernel, not via output_type."""
    capture_agent, fake_run = _simulate_submission(
        source_code="@triton.jit\ndef k(): pass",
        triton_kernel_name="k",
    )

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent) as mock_agent_cls,
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run

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

    assert isinstance(result, KernelCodeOutput)
    assert result.source_code == "@triton.jit\ndef k(): pass"
    assert result.triton_kernel_name == "k"
    mock_run.assert_awaited_once()
    # Agent gets compile + correctness + submit_kernel tools (3) and is built
    # without ``output_type=`` so the SDK doesn't request response_format=json_schema.
    kwargs = mock_agent_cls.call_args.kwargs
    assert len(kwargs["tools"]) == 3
    assert "output_type" not in kwargs
    assert any(
        getattr(t, "__name__", "") == "submit_kernel" for t in kwargs["tools"]
    )


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
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)

    captured = {}
    def capture_factory(*args, **kwargs):
        captured["input_generators"] = kwargs["input_generators"]
        return lambda src: "passed"

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
        patch("src.agents.coder._make_correctness_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run
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
    """No config → default ACTSConfig.max_debug_retries=3 → max_turns = 2*3+2 = 8.
    The +2 (vs. the historical +1) reserves one turn for ``submit_kernel`` and
    one for the final plain-text confirmation that terminates the SDK loop."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run

        agent = CoderAgent(model=MagicMock())
        await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
        )

    assert mock_run.await_args.kwargs.get("max_turns") == 8


@pytest.mark.asyncio
async def test_implement_max_turns_derived_from_config():
    """max_debug_retries=5 → max_turns = 2*5+2 = 12 (+2 for submit + confirm)."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run

        agent = CoderAgent(model=MagicMock(), config=ACTSConfig(max_debug_retries=5))
        await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
        )

    assert mock_run.await_args.kwargs.get("max_turns") == 12


@pytest.mark.asyncio
async def test_implement_uses_zero_temperature():
    """Coder runs with temperature=0.0 — deterministic code generation."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config") as mock_cfg,
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
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


@pytest.mark.asyncio
async def test_implement_raises_when_agent_terminates_without_submitting():
    """Option-α invariant: if the LLM exits the tool loop without ever calling
    submit_kernel, we have no Coder output. Raising ImplementationError lets
    the caller surface the failure rather than silently treating an empty
    submission as a degraded best-effort."""
    # Capture-agent path (no fake_run side_effect that calls submit_kernel) —
    # mock_run returns a normal RunResult but the captured dict stays empty.
    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = MagicMock(final_output="done")

        agent = CoderAgent(model=MagicMock())
        with pytest.raises(ImplementationError, match="submit_kernel"):
            await agent.implement(
                kernel_source="src",
                plan=OptimizationPlan(tier=1, technique="t1"),
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
            )


# ── option γ: MaxTurnsExceeded handling ────────────────────────────────


@pytest.mark.asyncio
async def test_implement_converts_max_turns_exceeded_to_implementation_error():
    """Option γ invariant: SDK ``MaxTurnsExceeded`` (raised mid-tool-loop
    when the LLM burns through the budget without ever submitting) must
    be converted to ``ImplementationError`` at the Coder boundary so the
    orchestrator / baseline_generator catch sites work uniformly. Without
    this conversion, the SDK exception propagates straight out of
    ``optimize()`` and aborts the entire run instead of dead-ending one
    branch (live-GPU-run #2 trigger, 2026-04-22)."""
    from src.agents.coder import MaxTurnsExceeded

    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = MaxTurnsExceeded("Max turns (8) exceeded")

        agent = CoderAgent(model=MagicMock())
        with pytest.raises(ImplementationError, match="turn budget"):
            await agent.implement(
                kernel_source="src",
                plan=OptimizationPlan(tier=1, technique="t1"),
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
            )


@pytest.mark.asyncio
async def test_implement_returns_partial_output_when_max_turns_after_submission():
    """If the LLM submitted a valid kernel before the SDK loop hit max_turns
    (e.g., it kept calling tools after submit despite the system-prompt rule),
    treat that submission as the answer rather than raising. The kernel was
    Pydantic-validated when submit_kernel ran; the run merely went over budget."""
    from src.agents.coder import MaxTurnsExceeded

    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)

    async def submit_then_exhaust(agent, prompt, **kwargs):
        # First simulate a successful submission, then raise as if the
        # SDK kept spinning after submit and burned the budget.
        await fake_run(agent, prompt, **kwargs)
        raise MaxTurnsExceeded("Max turns exceeded")

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = submit_then_exhaust

        agent = CoderAgent(model=MagicMock())
        result = await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
        )

    assert isinstance(result, KernelCodeOutput)
    assert result.source_code == _VALID_SOURCE
    assert result.triton_kernel_name == _VALID_NAME


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
async def test_translate_builds_agent_with_three_tools_and_returns_source():
    """translate() constructs a fresh Agent with compile + correctness +
    submit tools and returns the captured Coder submission."""
    capture_agent, fake_run = _simulate_submission(
        source_code="@triton.jit\ndef kernel_fn(x): pass",
        triton_kernel_name="kernel_fn",
    )

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent) as mock_agent_cls,
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
        agent = CoderAgent(model=MagicMock())
        result = await agent.translate(
            reference_source="def run(x):\n    return x * 2.0\n",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
        )

    assert isinstance(result, KernelCodeOutput)
    assert result.source_code == "@triton.jit\ndef kernel_fn(x): pass"
    assert result.triton_kernel_name == "kernel_fn"
    mock_run.assert_awaited_once()
    kwargs = mock_agent_cls.call_args.kwargs
    assert len(kwargs["tools"]) == 3  # compile + correctness + submit
    assert "output_type" not in kwargs


@pytest.mark.asyncio
async def test_translate_user_prompt_contains_reference_and_entrypoint():
    """Prompt must surface the source to translate and the target entrypoint."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)
    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
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
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)
    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent) as mock_agent_cls,
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
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
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)
    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config") as mock_cfg,
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
        mock_cfg.return_value = None
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x): return x",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
        )

    mock_cfg.assert_called_once_with(temperature=0.0)


# ── submit-tool factory (option α) ─────────────────────────────────────


def test_make_submit_tool_captures_valid_output():
    """Direct unit test of the submit-tool factory: a valid (source, name)
    pair populates the captured dict and returns the success sentinel."""
    from src.agents.coder import _make_submit_tool

    captured: dict = {}
    submit = _make_submit_tool(captured)
    msg = submit(
        source_code="@triton.jit\ndef k(): pass",
        triton_kernel_name="k",
    )
    assert "submitted" in msg.lower()
    assert "output" in captured
    assert isinstance(captured["output"], KernelCodeOutput)
    assert captured["output"].source_code == "@triton.jit\ndef k(): pass"
    assert captured["output"].triton_kernel_name == "k"


def test_make_submit_tool_returns_validation_error_string_on_mismatch():
    """A name-not-in-source mismatch must NOT raise — instead the tool returns
    the error string so the SDK hands it back to the LLM as the tool-call
    response, prompting an in-loop retry within the existing turn budget."""
    from src.agents.coder import _make_submit_tool

    captured: dict = {}
    submit = _make_submit_tool(captured)
    msg = submit(
        source_code="@triton.jit\ndef actual_name(): pass",
        triton_kernel_name="claimed_name",
    )
    assert "FAILED" in msg
    assert "claimed_name" in msg
    # On failure the captured dict stays empty so coder.py raises
    # ImplementationError after the run if the LLM never recovered.
    assert "output" not in captured


def test_make_submit_tool_returns_error_when_source_lacks_triton_jit():
    """The Coder writes Triton kernels — pure-PyTorch source is rejected
    by the same Pydantic validator the old output_type= path used."""
    from src.agents.coder import _make_submit_tool

    captured: dict = {}
    submit = _make_submit_tool(captured)
    msg = submit(
        source_code="def run(x): return x * 2.0",
        triton_kernel_name="run",
    )
    assert "FAILED" in msg
    assert "@triton.jit" in msg
    assert "output" not in captured
