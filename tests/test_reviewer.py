"""Tests for agents/reviewer.py — Reviewer agent with structured LLM output."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.reviewer import (
    BranchQuality,
    ReviewerAgent,
    ReviewerFeedback,
)


# ── Pydantic output model ──────────────────────────────────────────────


def test_output_model_accepts_valid_data():
    """ReviewerFeedbackOutput parses valid JSON-like data."""
    from src.agents.reviewer import ReviewerFeedbackOutput

    out = ReviewerFeedbackOutput(
        outcome="improved",
        metric_deltas={"sol_score": 0.08},
        bottleneck_classification="memory_bound",
        bottleneck_diagnosis="DRAM bandwidth at 82%, still dominated by global loads.",
        suggestions=["Try shared-memory tiling."],
        branch_quality=BranchQuality.PROMISING,
        conditional_assessment="If tiling lands, expect compute-bound shift.",
    )
    assert out.outcome == "improved"
    assert out.branch_quality is BranchQuality.PROMISING
    assert out.bottleneck_classification == "memory_bound"


def test_output_model_rejects_bad_bottleneck():
    """bottleneck_classification is strict — rejects values outside the enum set."""
    from pydantic import ValidationError

    from src.agents.reviewer import ReviewerFeedbackOutput

    with pytest.raises(ValidationError):
        ReviewerFeedbackOutput(
            outcome="improved",
            bottleneck_classification="something_invented",
            branch_quality=BranchQuality.PROMISING,
        )


def test_output_model_rejects_bad_branch_quality():
    """branch_quality is strict — only BranchQuality enum values allowed."""
    from pydantic import ValidationError

    from src.agents.reviewer import ReviewerFeedbackOutput

    with pytest.raises(ValidationError):
        ReviewerFeedbackOutput(
            outcome="improved",
            bottleneck_classification="memory_bound",
            branch_quality="not_a_valid_quality",
        )


def test_output_model_accepts_free_form_outcome():
    """outcome is non-strict — unusual strings are accepted as-is."""
    from src.agents.reviewer import ReviewerFeedbackOutput

    out = ReviewerFeedbackOutput(
        outcome="partially_improved",  # not in canonical set, but accepted
        bottleneck_classification="balanced",
        branch_quality=BranchQuality.PROMISING,
    )
    assert out.outcome == "partially_improved"


def test_output_model_defaults():
    """ReviewerFeedbackOutput uses empty defaults for optional fields."""
    from src.agents.reviewer import ReviewerFeedbackOutput

    out = ReviewerFeedbackOutput(
        outcome="neutral",
        bottleneck_classification="compute_bound",
        branch_quality=BranchQuality.PLATEAU,
    )
    assert out.metric_deltas == {}
    assert out.suggestions == []
    assert out.bottleneck_diagnosis == ""
    assert out.conditional_assessment == ""


# ── prompt assembly ─────────────────────────────────────────────────────


def test_build_user_prompt_contains_all_sections():
    """The assembled user prompt includes kernel source, profiling, score,
    headroom, bottleneck, tree context, and KB context."""
    agent = ReviewerAgent(model=None)
    prompt = agent.build_user_prompt(
        kernel_source="@triton.jit\ndef matmul_kernel(): ...",
        profiling_summary="DRAM: 78%, ALU: 22%, L2 hit: 55%",
        sol_score=0.62,
        headroom_pct=38.0,
        bottleneck="memory_bound",
        tree_context="Iteration 3, depth 2, parent SOL 0.55",
        kb_context="Pattern: low L2 hit + high DRAM util -> check blocking.",
    )
    assert "@triton.jit" in prompt
    assert "DRAM: 78%" in prompt
    assert "0.62" in prompt
    assert "38" in prompt  # headroom
    assert "memory_bound" in prompt
    assert "Iteration 3" in prompt
    assert "low L2 hit" in prompt


def test_build_user_prompt_omits_empty_optional_sections():
    """tree_context and kb_context sections are omitted when empty."""
    agent = ReviewerAgent(model=None)
    prompt = agent.build_user_prompt(
        kernel_source="def kernel(): pass",
        profiling_summary="Compute bound: 85% ALU",
        sol_score=0.71,
        headroom_pct=29.0,
        bottleneck="compute_bound",
    )
    assert "Search tree" not in prompt
    assert "Knowledge base" not in prompt


def test_build_user_prompt_escapes_backticks_in_kernel_source():
    """Triple backticks in kernel source are escaped so the fence stays closed."""
    agent = ReviewerAgent(model=None)
    source = 'def kernel():\n    """```python\n    fake section\n    ```"""\n    pass'
    prompt = agent.build_user_prompt(
        kernel_source=source,
        profiling_summary="Compute bound",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck="compute_bound",
    )
    sections = prompt.split("## ")
    kernel_section = [s for s in sections if s.startswith("Current kernel")][0]
    assert "```python\nfake section\n```" not in kernel_section


# ── review() with mocked LLM ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_review_calls_llm_and_returns_parsed_feedback():
    """With a model configured, review() calls the LLM and parses output."""
    from src.agents.reviewer import ReviewerFeedbackOutput

    mock_output = ReviewerFeedbackOutput(
        outcome="improved",
        metric_deltas={"sol_score": 0.08, "latency_ms": -0.4},
        bottleneck_classification="memory_bound",
        bottleneck_diagnosis="Global loads still dominate; L2 hit rose to 68%.",
        suggestions=["Increase block_k.", "Prefetch A."],
        branch_quality=BranchQuality.PROMISING,
        conditional_assessment="Two more tiling steps should hit compute-bound.",
    )
    mock_result = MagicMock()
    mock_result.final_output = mock_output

    with (
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
    ):
        mock_run.return_value = mock_result
        agent = ReviewerAgent(model=None)
        agent._agent = MagicMock()  # non-None triggers LLM path

        feedback = await agent.review(
            kernel_source="@triton.jit\ndef kernel(): ...",
            profiling_summary="DRAM 78%",
            sol_score=0.62,
            headroom_pct=38.0,
            bottleneck="memory_bound",
        )

    assert isinstance(feedback, ReviewerFeedback)
    assert feedback.outcome == "improved"
    assert feedback.bottleneck_classification == "memory_bound"
    assert feedback.branch_quality is BranchQuality.PROMISING
    assert feedback.metric_deltas == {"sol_score": 0.08, "latency_ms": -0.4}
    assert "L2 hit rose to 68%" in feedback.bottleneck_diagnosis


@pytest.mark.asyncio
async def test_review_uses_nonzero_temperature():
    """Reviewer runs with temperature=0.3 — variance in diagnosis wording;
    the strict enum fields (branch_quality, bottleneck_classification) stay pinned."""
    from src.agents.reviewer import ReviewerFeedbackOutput

    mock_output = ReviewerFeedbackOutput(
        outcome="neutral",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.BLOCKED_POTENTIAL,
    )
    mock_result = MagicMock()
    mock_result.final_output = mock_output

    with (
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config") as mock_cfg,
    ):
        mock_run.return_value = mock_result
        mock_cfg.return_value = None

        agent = ReviewerAgent(model=None)
        agent._agent = MagicMock()

        await agent.review(
            kernel_source="def k(): pass",
            profiling_summary="DRAM 60%",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck="memory_bound",
        )

    mock_cfg.assert_called_once_with(temperature=0.3)
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_review_passes_tree_and_kb_context_to_prompt():
    """tree_context and kb_context provided to review() reach the user prompt."""
    from src.agents.reviewer import ReviewerFeedbackOutput

    mock_output = ReviewerFeedbackOutput(
        outcome="neutral",
        bottleneck_classification="balanced",
        branch_quality=BranchQuality.PLATEAU,
    )
    mock_result = MagicMock()
    mock_result.final_output = mock_output

    with (
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
    ):
        mock_run.return_value = mock_result
        agent = ReviewerAgent(model=None)
        agent._agent = MagicMock()

        await agent.review(
            kernel_source="def kernel(): pass",
            profiling_summary="Balanced",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck="balanced",
            tree_context="Depth 4, sibling SOL 0.48",
            kb_context="Entry: plateau often indicates warp-schedule stall.",
        )

    # Inspect the prompt that was actually sent to the LLM.
    sent_prompt = mock_run.await_args.args[1]
    assert "Depth 4" in sent_prompt
    assert "plateau often indicates warp-schedule stall" in sent_prompt


# ── rule-based fallback ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_review_without_model_returns_rule_based_fallback():
    """Without a model, review() returns a rule-based feedback (no LLM call)."""
    agent = ReviewerAgent(model=None)
    feedback = await agent.review(
        kernel_source="def kernel(): pass",
        profiling_summary="Unknown",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck="memory_bound",
    )
    assert isinstance(feedback, ReviewerFeedback)
    assert feedback.bottleneck_classification == "memory_bound"
    # Diagnosis should indicate rule-based origin.
    assert "rule" in feedback.bottleneck_diagnosis.lower()


@pytest.mark.asyncio
async def test_review_falls_back_to_rules_when_llm_returns_none():
    """When run_agent returns None (all retries exhausted), review() falls back
    to rule-based feedback — it does NOT raise."""
    with (
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
    ):
        mock_run.return_value = None
        agent = ReviewerAgent(model=None)
        agent._agent = MagicMock()  # pretend LLM is configured

        feedback = await agent.review(
            kernel_source="def kernel(): pass",
            profiling_summary="Compute bound: 85% ALU",
            sol_score=0.66,
            headroom_pct=34.0,
            bottleneck="compute_bound",
            prev_sol_score=0.60,
        )

    assert isinstance(feedback, ReviewerFeedback)
    assert feedback.bottleneck_classification == "compute_bound"
    assert "rule" in feedback.bottleneck_diagnosis.lower()


@pytest.mark.asyncio
async def test_llm_failure_is_flagged_degraded():
    """run_agent returning None means retries exhausted — feedback must be
    flagged so the orchestrator can distinguish it from an expected fallback."""
    with (
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
    ):
        mock_run.return_value = None
        agent = ReviewerAgent(model=None)
        agent._agent = MagicMock()

        feedback = await agent.review(
            kernel_source="def kernel(): pass",
            profiling_summary="...",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck="balanced",
            prev_sol_score=0.5,
        )

    assert feedback.degraded is True
    assert feedback.error_reason == "llm_retries_exhausted"
    assert "degraded" in feedback.bottleneck_diagnosis.lower()


@pytest.mark.asyncio
async def test_no_model_configured_is_not_degraded():
    """When no model is configured, the rule-based path is expected operation,
    not a degraded state — the orchestrator should not alarm on this."""
    agent = ReviewerAgent(model=None)
    feedback = await agent.review(
        kernel_source="def kernel(): pass",
        profiling_summary="...",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck="balanced",
    )
    assert feedback.degraded is False
    assert feedback.error_reason == ""


@pytest.mark.parametrize(
    "sol_score, prev_sol_score, headroom_pct, expected_outcome, expected_branch",
    [
        # improved + high headroom -> promising
        (0.62, 0.50, 38.0, "improved", BranchQuality.PROMISING),
        # improved + low headroom -> plateau
        (0.90, 0.85, 10.0, "improved", BranchQuality.PLATEAU),
        # neutral -> blocked_potential
        (0.50, 0.50, 50.0, "neutral", BranchQuality.BLOCKED_POTENTIAL),
        # regressed -> dead_end
        (0.40, 0.55, 60.0, "regressed", BranchQuality.DEAD_END),
    ],
)
def test_rule_based_feedback_branch_quality(
    sol_score, prev_sol_score, headroom_pct, expected_outcome, expected_branch
):
    """Rule-based fallback maps (sol_delta, headroom) to (outcome, branch_quality)
    per the spec in the design discussion."""
    from src.agents.reviewer import rule_based_feedback

    feedback = rule_based_feedback(
        sol_score=sol_score,
        prev_sol_score=prev_sol_score,
        headroom_pct=headroom_pct,
        bottleneck="memory_bound",
    )
    assert feedback.outcome == expected_outcome
    assert feedback.branch_quality is expected_branch
    assert feedback.bottleneck_classification == "memory_bound"


def test_rule_based_feedback_handles_missing_prev_score():
    """Without a prev_sol_score, rule-based fallback cannot compute delta —
    treats outcome as neutral."""
    from src.agents.reviewer import rule_based_feedback

    feedback = rule_based_feedback(
        sol_score=0.5,
        prev_sol_score=None,
        headroom_pct=50.0,
        bottleneck="balanced",
    )
    assert feedback.outcome == "neutral"
    assert feedback.branch_quality is BranchQuality.BLOCKED_POTENTIAL


# ── prompt_dir customization (enables future sub-agent split) ──────────


def test_custom_prompt_dir_is_used(tmp_path):
    """ReviewerAgent accepts a custom prompt_dir so future Compute/Memory
    sub-agents can load their own system prompts without subclassing."""
    # Only constructed path is validated when model=None — no file read.
    custom_dir = tmp_path / "compute"
    custom_dir.mkdir()
    agent = ReviewerAgent(model=None, prompt_dir=custom_dir)
    assert agent._prompt_dir == custom_dir


def test_default_prompt_dir_points_to_reviewer():
    """Default prompt_dir is prompts/reviewer/."""
    agent = ReviewerAgent(model=None)
    assert agent._prompt_dir.name == "reviewer"
    assert agent._prompt_dir.parent.name == "prompts"


# ── parse_feedback utility ─────────────────────────────────────────────


def test_parse_feedback_converts_output_to_dataclass():
    """parse_feedback() converts a RunResult.final_output (Pydantic) to a
    ReviewerFeedback dataclass."""
    from src.agents.reviewer import ReviewerFeedbackOutput

    mock_output = ReviewerFeedbackOutput(
        outcome="regressed",
        metric_deltas={"sol_score": -0.05},
        bottleneck_classification="memory_bound",
        bottleneck_diagnosis="Spill rate spiked.",
        suggestions=["Revert last action."],
        branch_quality=BranchQuality.DEAD_END,
    )
    mock_result = MagicMock()
    mock_result.final_output = mock_output

    feedback = ReviewerAgent.parse_feedback(mock_result)
    assert isinstance(feedback, ReviewerFeedback)
    assert feedback.outcome == "regressed"
    assert feedback.branch_quality is BranchQuality.DEAD_END
    assert feedback.suggestions == ["Revert last action."]
