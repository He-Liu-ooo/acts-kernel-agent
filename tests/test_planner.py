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
    from src.memory.experience import ActionRecord, Experience

    agent = PlannerAgent(model=None)
    experiences = [
        Experience(
            kernel_type="matmul",
            action_applied=ActionRecord(action_id="tile_sizes", tier=1, name="tile_sizes"),
            speedup=1.3,
            bottleneck_before="memory_bound",
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
            bottleneck_before="memory_bound",
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
async def test_plan_calls_llm_and_returns_parsed_plan():
    """With a model configured, plan() calls the LLM and parses output."""
    from src.agents.planner import OptimizationPlanOutput

    mock_output = OptimizationPlanOutput(
        tier=3,
        technique="warp_specialization",
        params={"num_warps": "8"},
        target_region="reduction loop",
        rationale="Compute-bound — split work across warps.",
    )

    mock_result = MagicMock()
    mock_result.final_output = mock_output

    with (
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config", return_value=None),
    ):
        mock_run.return_value = mock_result

        # SDK not installed in test env — bypass constructor, set _agent directly.
        agent = PlannerAgent(model=None)
        agent._agent = MagicMock()  # non-None triggers LLM path

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
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config", return_value=None),
    ):
        mock_run.return_value = None

        agent = PlannerAgent(model=None)
        agent._agent = MagicMock()

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
    from src.agents.planner import OptimizationPlanOutput

    mock_output = OptimizationPlanOutput(tier=1, technique="block_size_tuning")
    mock_result = MagicMock()
    mock_result.final_output = mock_output

    with (
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config") as mock_cfg,
    ):
        mock_run.return_value = mock_result
        mock_cfg.return_value = None

        agent = PlannerAgent(model=None)
        agent._agent = MagicMock()

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
    from src.agents.planner import OptimizationPlanOutput

    mock_output = OptimizationPlanOutput(
        tier=1,
        technique="hallucinated_technique",
        rationale="I made this up.",
    )
    mock_result = MagicMock()
    mock_result.final_output = mock_output

    with (
        patch("src.agents.planner.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.planner.make_run_config", return_value=None),
    ):
        mock_run.return_value = mock_result

        agent = PlannerAgent(model=None)
        agent._agent = MagicMock()

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
