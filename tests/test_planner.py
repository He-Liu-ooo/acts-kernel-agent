"""Tests for agents/planner.py — Planner agent with structured LLM output."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.planner import OptimizationPlan, PlannerAgent, PlanningError


# ── Pydantic output model ──────────────────────────────────────────────


def test_output_model_accepts_valid_data():
    """OptimizationPlanOutput parses valid JSON-like data."""
    from src.agents.planner import OptimizationPlanOutput

    out = OptimizationPlanOutput(
        tier=2,
        technique="shared_memory_caching",
        params={"cache_size": "64KB"},
        target_region="inner loop",
        rationale="Memory-bound bottleneck — caching reduces global loads.",
    )
    assert out.tier == 2
    assert out.technique == "shared_memory_caching"
    assert out.params == {"cache_size": "64KB"}


def test_output_model_rejects_bad_tier():
    """OptimizationPlanOutput rejects non-integer tier."""
    from pydantic import ValidationError

    from src.agents.planner import OptimizationPlanOutput

    with pytest.raises(ValidationError):
        OptimizationPlanOutput(
            tier="not_an_int",
            technique="tile_sizes",
            params={},
            target_region="",
            rationale="",
        )


def test_output_model_defaults():
    """OptimizationPlanOutput uses empty defaults for optional fields."""
    from src.agents.planner import OptimizationPlanOutput

    out = OptimizationPlanOutput(tier=1, technique="block_size_tuning")
    assert out.params == {}
    assert out.target_region == ""
    assert out.rationale == ""


# ── prompt assembly ─────────────────────────────────────────────────────


def test_build_user_prompt_contains_all_sections():
    """The assembled user prompt includes kernel source, profiling,
    experiences, available actions, and reviewer feedback."""
    from src.eval.types import BottleneckType
    from src.memory.experience import ActionRecord, Experience

    agent = PlannerAgent(model=None)
    experiences = [
        Experience(
            kernel_type="matmul",
            action_applied=ActionRecord(action_id="tile_sizes", tier=1, name="tile_sizes"),
            speedup=1.3,
            bottleneck_before=BottleneckType.MEMORY_BOUND,
            success=True,
            hardware="H100",
        ),
    ]
    prompt = agent.build_user_prompt(
        kernel_source="@triton.jit\ndef matmul_kernel(): ...",
        profiling_summary="Memory bound: 78% DRAM util, 22% compute",
        past_experiences=experiences,
        available_actions=["tile_sizes", "shared_memory_caching"],
        tree_context="Iteration 3, depth 2, parent speedup 1.2x",
        reviewer_feedback="Try reducing global memory loads.",
    )
    assert "@triton.jit" in prompt
    assert "Memory bound" in prompt
    assert "tile_sizes" in prompt
    assert "shared_memory_caching" in prompt
    assert "Iteration 3" in prompt
    assert "reducing global memory loads" in prompt
    assert "1.3" in prompt  # experience speedup


def test_build_user_prompt_includes_experience_parameters():
    """Past experiences include action parameters so the Planner can
    distinguish failed parameterizations from untried ones."""
    from src.eval.types import BottleneckType
    from src.memory.experience import ActionRecord, Experience

    agent = PlannerAgent(model=None)
    experiences = [
        Experience(
            kernel_type="matmul",
            action_applied=ActionRecord(
                action_id="t1_block_size_tuning",
                tier=1,
                name="Block Size Tuning",
                parameters={"block_size": "128"},
            ),
            speedup=0.9,
            bottleneck_before=BottleneckType.MEMORY_BOUND,
            success=False,
            hardware="H100",
        ),
    ]
    prompt = agent.build_user_prompt(
        kernel_source="def kernel(): pass",
        profiling_summary="Memory bound",
        past_experiences=experiences,
        available_actions=["t1_block_size_tuning"],
    )
    assert "block_size=128" in prompt


def test_build_user_prompt_omits_empty_sections():
    """Reviewer feedback and tree context are omitted when empty."""
    agent = PlannerAgent(model=None)
    prompt = agent.build_user_prompt(
        kernel_source="def kernel(): pass",
        profiling_summary="Compute bound",
        past_experiences=[],
        available_actions=["block_size_tuning"],
    )
    assert "Reviewer" not in prompt
    assert "Search tree" not in prompt


# ── plan() with mocked LLM ─────────────────────────────────────────────


@pytest.mark.asyncio
def _simulate_plan_submission(**fields):
    """Test helper: returns (capture_factory, fake_run) that together
    simulate a submit_plan tool call inside Runner.run. Mirrors
    tests/test_coder.py::_simulate_submission. Use both via patches:

        capture_factory, fake_run = _simulate_plan_submission(tier=3, ...)
        with (
            patch("src.agents.planner._SDK_AVAILABLE", True),
            patch("src.agents.planner.Agent"),
            patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
            patch("src.agents.planner.make_run_config", return_value=None),
            patch("src.agents.planner.function_tool", side_effect=lambda f: f),
            patch("src.agents.planner._make_submit_plan_tool", side_effect=capture_factory),
        ):
            mock_run.side_effect = fake_run
            ...
    """
    from src.agents.planner import OptimizationPlanOutput, _make_submit_plan_tool

    captured_holder: list[dict] = []

    def capture_factory(captured_dict: dict):
        captured_holder.append(captured_dict)
        return _make_submit_plan_tool(captured_dict)

    async def fake_run(agent, prompt, **kwargs):
        assert captured_holder, "factory should have been called by plan()"
        captured_holder[0]["output"] = OptimizationPlanOutput(**fields)
        return MagicMock(final_output="done")

    return capture_factory, fake_run


@pytest.mark.asyncio
async def test_plan_calls_llm_and_returns_parsed_plan():
    """With a model configured, plan() calls the LLM through the submit_plan
    tool path and parses captured output."""
    capture_factory, fake_run = _simulate_plan_submission(
        tier=3,
        technique="warp_specialization",
        params={"num_warps": "8"},
        target_region="reduction loop",
        rationale="Compute-bound — split work across warps.",
    )

    with (
        patch("src.agents.planner._SDK_AVAILABLE", True),
        patch("src.agents.planner.Agent"),
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config", return_value=None),
        patch("src.agents.planner.function_tool", side_effect=lambda f: f),
        patch("src.agents.planner._make_submit_plan_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run

        agent = PlannerAgent(model=MagicMock())
        plan = await agent.plan(
            kernel_source="@triton.jit\ndef kernel(): ...",
            profiling_summary="Compute bound: 85% ALU util",
            past_experiences=[],
            available_actions=["warp_specialization"],
        )

    assert isinstance(plan, OptimizationPlan)
    assert plan.tier == 3
    assert plan.technique == "warp_specialization"
    assert plan.params == {"num_warps": "8"}
    assert plan.rationale == "Compute-bound — split work across warps."
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_plan_raises_on_llm_failure():
    """If run_agent returns None (all retries exhausted), raise PlanningError."""
    with (
        patch("src.agents.planner._SDK_AVAILABLE", True),
        patch("src.agents.planner.Agent"),
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config", return_value=None),
        patch("src.agents.planner.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = None

        agent = PlannerAgent(model=MagicMock())

        with pytest.raises(PlanningError, match="LLM"):
            await agent.plan(
                kernel_source="def kernel(): pass",
                profiling_summary="Unknown",
                past_experiences=[],
                available_actions=["block_size_tuning"],
            )


@pytest.mark.asyncio
async def test_plan_uses_nonzero_temperature():
    """Planner runs with temperature=0.3 — variance in technique exploration."""
    capture_factory, fake_run = _simulate_plan_submission(
        tier=1, technique="block_size_tuning",
    )

    with (
        patch("src.agents.planner._SDK_AVAILABLE", True),
        patch("src.agents.planner.Agent"),
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config") as mock_cfg,
        patch("src.agents.planner.function_tool", side_effect=lambda f: f),
        patch("src.agents.planner._make_submit_plan_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run
        mock_cfg.return_value = None

        agent = PlannerAgent(model=MagicMock())
        await agent.plan(
            kernel_source="def k(): pass",
            profiling_summary="Memory bound",
            past_experiences=[],
            available_actions=["block_size_tuning"],
        )

    mock_cfg.assert_called_once_with(temperature=0.3)


@pytest.mark.asyncio
async def test_plan_rejects_hallucinated_technique():
    """If the LLM returns a technique not in available_actions, raise PlanningError."""
    capture_factory, fake_run = _simulate_plan_submission(
        tier=1,
        technique="hallucinated_technique",
        rationale="I made this up.",
    )

    with (
        patch("src.agents.planner._SDK_AVAILABLE", True),
        patch("src.agents.planner.Agent"),
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config", return_value=None),
        patch("src.agents.planner.function_tool", side_effect=lambda f: f),
        patch("src.agents.planner._make_submit_plan_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run

        agent = PlannerAgent(model=MagicMock())
        with pytest.raises(PlanningError, match="hallucinated_technique"):
            await agent.plan(
                kernel_source="def kernel(): pass",
                profiling_summary="Unknown",
                past_experiences=[],
                available_actions=["block_size_tuning", "grid_shape"],
            )


# ── prompt escaping ─────────────────────────────────────────────────────


def test_backticks_in_kernel_source_are_escaped():
    """Triple backticks in kernel source don't break the prompt fence."""
    agent = PlannerAgent(model=None)
    source = 'def kernel():\n    """```python\n    fake section\n    ```"""\n    pass'
    prompt = agent.build_user_prompt(
        kernel_source=source,
        profiling_summary="Compute bound",
        past_experiences=[],
        available_actions=["block_size_tuning"],
    )
    # The kernel section should be self-contained — count opening/closing fences
    # The prompt should not have an unmatched fence that breaks structure
    sections = prompt.split("## ")
    kernel_section = [s for s in sections if s.startswith("Current kernel")][0]
    # Backticks in the source must be escaped so the fence stays closed
    assert "```python\nfake section\n```" not in kernel_section


# ── plan() without model ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_without_model_returns_default():
    """Without a model, plan() returns a default plan (no LLM call)."""
    agent = PlannerAgent(model=None)
    plan = await agent.plan(
        kernel_source="def kernel(): pass",
        profiling_summary="Unknown",
        past_experiences=[],
        available_actions=[],
    )
    assert isinstance(plan, OptimizationPlan)
    assert plan.tier == 1
    assert plan.technique == "block_size_tuning"


# ── _make_submit_plan_tool — direct unit tests ────────────────────────────


def test_make_submit_plan_tool_captures_valid_output():
    """A valid plan populates the captured dict and returns the success sentinel."""
    from src.agents.llm_backend import SUBMIT_OK_SENTINEL
    from src.agents.planner import OptimizationPlanOutput, _make_submit_plan_tool

    captured: dict = {}
    submit = _make_submit_plan_tool(captured)
    msg = submit(
        tier=1,
        technique="block_size_tuning",
        params={"block_size": "128"},
        target_region="main loop",
        rationale="memory-bound — increase tile",
    )
    assert msg == SUBMIT_OK_SENTINEL
    assert "output" in captured
    assert isinstance(captured["output"], OptimizationPlanOutput)
    assert captured["output"].technique == "block_size_tuning"
    assert captured["output"].params == {"block_size": "128"}


def test_make_submit_plan_tool_returns_validation_error_string_on_invalid_tier():
    """A non-int tier (uncoerceable string) returns the error string so the
    SDK hands it back to the LLM for in-loop retry."""
    from src.agents.planner import _make_submit_plan_tool

    captured: dict = {}
    submit = _make_submit_plan_tool(captured)
    msg = submit(
        tier="not_an_int",  # type: ignore[arg-type]
        technique="x",
        params={},
        target_region="",
        rationale="",
    )
    assert msg.startswith("submit_plan FAILED:")
    assert "output" not in captured


def test_make_submit_plan_tool_handles_empty_params_dict():
    """Empty params={} must capture cleanly — defends against the
    dict[str, str] field being the original SDK breakage point."""
    from src.agents.planner import OptimizationPlanOutput, _make_submit_plan_tool

    captured: dict = {}
    submit = _make_submit_plan_tool(captured)
    msg = submit(
        tier=2,
        technique="vectorize",
        params={},
        target_region="",
        rationale="",
    )
    assert "output" in captured
    assert isinstance(captured["output"], OptimizationPlanOutput)
    assert captured["output"].params == {}


# ── plan() MaxTurnsExceeded + missing-submit handling ─────────────────────


@pytest.mark.asyncio
async def test_plan_raises_when_loop_terminates_without_submitting():
    """If the LLM exits the tool loop without ever calling submit_plan,
    plan() must raise PlanningError so the orchestrator can skip the
    iteration cleanly. Mocked Runner.run returns normally but captured
    dict stays empty."""
    with (
        patch("src.agents.planner._SDK_AVAILABLE", True),
        patch("src.agents.planner.Agent"),
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config", return_value=None),
        patch("src.agents.planner.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = MagicMock(final_output="done")

        agent = PlannerAgent(model=MagicMock())
        with pytest.raises(PlanningError, match="submit_plan"):
            await agent.plan(
                kernel_source="src",
                profiling_summary="",
                past_experiences=[],
                available_actions=[],
            )


@pytest.mark.asyncio
async def test_plan_converts_max_turns_exceeded_to_planning_error():
    """SDK MaxTurnsExceeded with empty captured must convert to
    PlanningError so the orchestrator's catch site (added in Task 3)
    treats it as a branch-local skip rather than a run-fatal exception."""
    from src.agents.planner import MaxTurnsExceeded

    with (
        patch("src.agents.planner._SDK_AVAILABLE", True),
        patch("src.agents.planner.Agent"),
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config", return_value=None),
        patch("src.agents.planner.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = MaxTurnsExceeded("Max turns (4) exceeded")

        agent = PlannerAgent(model=MagicMock())
        with pytest.raises(PlanningError, match="turn budget"):
            await agent.plan(
                kernel_source="src",
                profiling_summary="",
                past_experiences=[],
                available_actions=[],
            )


@pytest.mark.asyncio
async def test_plan_returns_partial_output_when_max_turns_after_submission():
    """If the LLM submitted a valid plan before the SDK loop hit max_turns,
    return the captured submission rather than raising. Mirrors Coder's
    `test_implement_returns_partial_output_when_max_turns_after_submission`.
    """
    from src.agents.planner import (
        MaxTurnsExceeded,
        OptimizationPlanOutput,
        _make_submit_plan_tool,
    )

    captured_holder: list[dict] = []

    def _capture_factory(captured_dict: dict):
        captured_holder.append(captured_dict)
        return _make_submit_plan_tool(captured_dict)

    with (
        patch("src.agents.planner._SDK_AVAILABLE", True),
        patch("src.agents.planner.Agent"),
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config", return_value=None),
        patch("src.agents.planner.function_tool", side_effect=lambda f: f),
        patch("src.agents.planner._make_submit_plan_tool", side_effect=_capture_factory),
    ):
        async def _side_effect(*args, **kwargs):
            assert captured_holder, "factory should have been called by plan()"
            captured_holder[0]["output"] = OptimizationPlanOutput(
                tier=2, technique="vectorize", params={}, target_region="", rationale="r",
            )
            raise MaxTurnsExceeded("Max turns (4) exceeded")

        mock_run.side_effect = _side_effect

        agent = PlannerAgent(model=MagicMock())
        plan = await agent.plan(
            kernel_source="src",
            profiling_summary="",
            past_experiences=[],
            available_actions=[],
        )

    assert isinstance(plan, OptimizationPlan)
    assert plan.tier == 2
    assert plan.technique == "vectorize"


@pytest.mark.asyncio
async def test_plan_recovers_from_first_invalid_submit_within_turn_budget():
    """Validation-retry budget: first submit_plan call returns FAILED (invalid
    payload), the LLM corrects on the second call, the agent emits the
    plain-text confirmation, and plan() returns cleanly without going
    through MaxTurnsExceeded recovery. max_turns=4 must reserve room for
    this exact path: turn 1 invalid + turn 2 corrected + turn 3 confirmation."""
    from src.agents.planner import (
        OptimizationPlanOutput,
        _make_submit_plan_tool,
    )

    captured_holder: list[dict] = []

    def _capture_factory(captured_dict: dict):
        captured_holder.append(captured_dict)
        return _make_submit_plan_tool(captured_dict)

    with (
        patch("src.agents.planner._SDK_AVAILABLE", True),
        patch("src.agents.planner.Agent"),
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config", return_value=None),
        patch("src.agents.planner.function_tool", side_effect=lambda f: f),
        patch("src.agents.planner._make_submit_plan_tool", side_effect=_capture_factory),
    ):
        async def _side_effect(*args, **kwargs):
            # Simulate the LLM's two-turn validation retry: first invalid
            # (no capture), then valid (populates captured), then a clean
            # confirmation that ends the SDK loop without raising.
            assert captured_holder, "factory should have been called by plan()"
            captured_holder[0]["output"] = OptimizationPlanOutput(
                tier=1,
                technique="block_size_tuning",
                params={"block_size": "128"},
                target_region="loop",
                rationale="recovered after one validation retry",
            )
            return MagicMock(final_output="done")

        mock_run.side_effect = _side_effect

        agent = PlannerAgent(model=MagicMock())
        plan = await agent.plan(
            kernel_source="src",
            profiling_summary="",
            past_experiences=[],
            available_actions=[],
        )

    assert isinstance(plan, OptimizationPlan)
    assert plan.technique == "block_size_tuning"
    assert "recovered after one validation retry" in plan.rationale


# ── SDK-absent fallback (regression guard) ────────────────────────────────


@pytest.mark.asyncio
async def test_plan_returns_default_when_sdk_absent_even_with_model_arg():
    """SDK-absent + non-None model arg must NOT take the LLM tool path —
    Agent and function_tool are None in that environment, so calling
    them raises TypeError. The pre-migration constructor gated _agent
    creation on `model is not None and _SDK_AVAILABLE`; the migration
    must preserve the equivalent fallback via has_model."""
    with patch("src.agents.planner._SDK_AVAILABLE", False):
        agent = PlannerAgent(model=MagicMock())
        plan = await agent.plan(
            kernel_source="def k(): pass",
            profiling_summary="",
            past_experiences=[],
            available_actions=[],
        )
    assert isinstance(plan, OptimizationPlan)
    assert plan.tier == 1
    assert plan.technique == "block_size_tuning"  # _DEFAULT_PLAN


def test_planner_has_model_false_when_sdk_absent():
    """``has_model`` must reflect both ``self._model is not None`` AND
    SDK availability. Without this gate, an SDK-absent test environment
    that injects a model stub flows into the tool path and crashes."""
    with patch("src.agents.planner._SDK_AVAILABLE", False):
        agent = PlannerAgent(model=MagicMock())
    assert agent.has_model is False


# ── _make_submit_plan_tool — defaulted Pydantic fields ────────────────────


def test_make_submit_plan_tool_omits_optional_fields_uses_pydantic_defaults():
    """``OptimizationPlanOutput`` defaults ``params={}``, ``target_region=""``,
    ``rationale=""``. The tool signature must mark these optional so the
    SDK doesn't reject a tool call that omits them — the LLM might emit
    only ``tier`` + ``technique`` for a minimal plan, which the old
    ``output_type=`` path accepted verbatim."""
    from src.agents.llm_backend import SUBMIT_OK_SENTINEL
    from src.agents.planner import OptimizationPlanOutput, _make_submit_plan_tool

    captured: dict = {}
    submit = _make_submit_plan_tool(captured)
    msg = submit(tier=2, technique="vectorize")  # only required fields
    assert msg == SUBMIT_OK_SENTINEL
    assert "output" in captured
    assert isinstance(captured["output"], OptimizationPlanOutput)
    assert captured["output"].params == {}
    assert captured["output"].target_region == ""
    assert captured["output"].rationale == ""
