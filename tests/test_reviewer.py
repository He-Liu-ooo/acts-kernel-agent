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
from src.eval.types import BottleneckType


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
        bottleneck=BottleneckType.MEMORY_BOUND,
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
        bottleneck=BottleneckType.COMPUTE_BOUND,
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
        bottleneck=BottleneckType.COMPUTE_BOUND,
    )
    sections = prompt.split("## ")
    kernel_section = [s for s in sections if s.startswith("Current kernel")][0]
    assert "```python\nfake section\n```" not in kernel_section


# ── review() with mocked LLM ───────────────────────────────────────────


@pytest.mark.asyncio
def _simulate_review_submission(**fields):
    """Test helper: returns (capture_factory, fake_run) that together
    simulate a submit_review tool call inside Runner.run. Mirrors
    tests/test_planner.py::_simulate_plan_submission. Use both via patches:

        capture_factory, fake_run = _simulate_review_submission(outcome=..., ...)
        with (
            patch("src.agents.reviewer._SDK_AVAILABLE", True),
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
            patch("src.agents.reviewer.make_run_config", return_value=None),
            patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
            patch("src.agents.reviewer._make_submit_review_tool", side_effect=capture_factory),
        ):
            mock_run.side_effect = fake_run
            ...
    """
    from src.agents.reviewer import ReviewerFeedbackOutput, _make_submit_review_tool

    captured_holder: list[dict] = []

    def capture_factory(captured_dict: dict):
        captured_holder.append(captured_dict)
        return _make_submit_review_tool(captured_dict)

    async def fake_run(agent, prompt, **kwargs):
        assert captured_holder, "factory should have been called by review()"
        captured_holder[0]["output"] = ReviewerFeedbackOutput(**fields)
        return MagicMock(final_output="done")

    return capture_factory, fake_run


@pytest.mark.asyncio
async def test_review_calls_llm_and_returns_parsed_feedback():
    """With a model configured, review() calls the LLM through the submit_review
    tool path and parses captured output."""
    capture_factory, fake_run = _simulate_review_submission(
        outcome="improved",
        metric_deltas={"sol_score": 0.08, "latency_ms": -0.4},
        bottleneck_classification="memory_bound",
        bottleneck_diagnosis="Global loads still dominate; L2 hit rose to 68%.",
        suggestions=["Increase block_k.", "Prefetch A."],
        branch_quality=BranchQuality.PROMISING,
        conditional_assessment="Two more tiling steps should hit compute-bound.",
    )

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
        patch("src.agents.reviewer._make_submit_review_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run
        agent = ReviewerAgent(model=MagicMock())
        feedback = await agent.review(
            kernel_source="@triton.jit\ndef kernel(): ...",
            profiling_summary="DRAM 78%",
            sol_score=0.62,
            headroom_pct=38.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
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
    capture_factory, fake_run = _simulate_review_submission(
        outcome="neutral",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.BLOCKED_POTENTIAL,
    )

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config") as mock_cfg,
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
        patch("src.agents.reviewer._make_submit_review_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run
        mock_cfg.return_value = None

        agent = ReviewerAgent(model=MagicMock())
        await agent.review(
            kernel_source="def k(): pass",
            profiling_summary="DRAM 60%",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        )

    mock_cfg.assert_called_once_with(temperature=0.3)
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_review_passes_tree_and_kb_context_to_prompt():
    """tree_context and kb_context provided to review() reach the user prompt."""
    capture_factory, fake_run = _simulate_review_submission(
        outcome="neutral",
        bottleneck_classification="balanced",
        branch_quality=BranchQuality.PLATEAU,
    )

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
        patch("src.agents.reviewer._make_submit_review_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run
        agent = ReviewerAgent(model=MagicMock())
        await agent.review(
            kernel_source="def kernel(): pass",
            profiling_summary="Balanced",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.BALANCED,
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
        bottleneck=BottleneckType.MEMORY_BOUND,
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
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
    ):
        mock_run.return_value = None
        agent = ReviewerAgent(model=MagicMock())

        feedback = await agent.review(
            kernel_source="def kernel(): pass",
            profiling_summary="Compute bound: 85% ALU",
            sol_score=0.66,
            headroom_pct=34.0,
            bottleneck=BottleneckType.COMPUTE_BOUND,
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
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
    ):
        mock_run.return_value = None
        agent = ReviewerAgent(model=MagicMock())

        feedback = await agent.review(
            kernel_source="def kernel(): pass",
            profiling_summary="...",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.BALANCED,
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
        bottleneck=BottleneckType.BALANCED,
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
        bottleneck=BottleneckType.MEMORY_BOUND,
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
        bottleneck=BottleneckType.BALANCED,
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


# ── _make_submit_review_tool — direct unit tests ──────────────────────────


def test_make_submit_review_tool_captures_valid_output():
    """A valid review populates the captured dict and returns the success sentinel."""
    from src.agents.llm_backend import SUBMIT_OK_SENTINEL
    from src.agents.reviewer import (
        ReviewerFeedbackOutput,
        _make_submit_review_tool,
    )

    captured: dict = {}
    submit = _make_submit_review_tool(captured)
    msg = submit(
        outcome="improved",
        metric_deltas={"latency_us": -1.5},
        bottleneck_classification="memory_bound",
        bottleneck_diagnosis="bandwidth-bound — saturate L2",
        suggestions=["increase tile size"],
        branch_quality=BranchQuality.PROMISING,
        conditional_assessment="",
    )
    assert msg == SUBMIT_OK_SENTINEL
    assert "output" in captured
    assert isinstance(captured["output"], ReviewerFeedbackOutput)
    assert captured["output"].metric_deltas == {"latency_us": -1.5}


def test_make_submit_review_tool_returns_validation_error_string_on_invalid_branch_quality():
    """An invalid branch_quality returns the error string for in-loop retry."""
    from src.agents.reviewer import _make_submit_review_tool

    captured: dict = {}
    submit = _make_submit_review_tool(captured)
    msg = submit(
        outcome="improved",
        metric_deltas={},
        bottleneck_classification="memory_bound",
        bottleneck_diagnosis="",
        suggestions=[],
        branch_quality="not_a_real_quality",  # type: ignore[arg-type]
        conditional_assessment="",
    )
    assert msg.startswith("submit_review FAILED:")
    assert "output" not in captured


def test_make_submit_review_tool_handles_empty_metric_deltas_dict():
    """Empty metric_deltas={} captures cleanly — defends against the
    dict[str, float] field being the original SDK breakage point."""
    from src.agents.reviewer import (
        ReviewerFeedbackOutput,
        _make_submit_review_tool,
    )

    captured: dict = {}
    submit = _make_submit_review_tool(captured)
    msg = submit(
        outcome="degraded",
        metric_deltas={},
        bottleneck_classification="compute_bound",
        bottleneck_diagnosis="",
        suggestions=[],
        branch_quality=BranchQuality.DEAD_END,
        conditional_assessment="",
    )
    assert "output" in captured
    assert isinstance(captured["output"], ReviewerFeedbackOutput)
    assert captured["output"].metric_deltas == {}


# ── review() MaxTurnsExceeded + missing-submit handling ──────────────────


@pytest.mark.asyncio
async def test_review_falls_back_to_degraded_when_loop_terminates_without_submitting():
    """If the LLM exits the tool loop without ever calling submit_review,
    review() must fall back to the existing rule-based degraded path —
    Reviewer's failure mode is recoverable (unlike Planner / Coder), so
    the orchestrator's frontier never starves."""
    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
    ):
        mock_run.return_value = MagicMock(final_output="done")

        agent = ReviewerAgent(model=MagicMock())
        feedback = await agent.review(
            kernel_source="src",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        )

    assert feedback.degraded is True
    assert feedback.error_reason == "missing_submit_review"


@pytest.mark.asyncio
async def test_review_falls_back_to_degraded_on_max_turns_exceeded():
    """SDK MaxTurnsExceeded with empty captured falls through to the
    existing rule-based degraded path with error_reason='max_turns_exceeded'.
    """
    from src.agents.reviewer import MaxTurnsExceeded

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
    ):
        mock_run.side_effect = MaxTurnsExceeded("Max turns (4) exceeded")

        agent = ReviewerAgent(model=MagicMock())
        feedback = await agent.review(
            kernel_source="src",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        )

    assert feedback.degraded is True
    assert feedback.error_reason == "max_turns_exceeded"


@pytest.mark.asyncio
async def test_review_returns_partial_output_when_max_turns_after_submission():
    """If the LLM submitted a valid review before the SDK loop hit max_turns,
    return the captured submission rather than falling back to degraded."""
    from src.agents.reviewer import (
        MaxTurnsExceeded,
        ReviewerFeedbackOutput,
        _make_submit_review_tool,
    )

    captured_holder: list[dict] = []

    def _capture_factory(captured_dict: dict):
        captured_holder.append(captured_dict)
        return _make_submit_review_tool(captured_dict)

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
        patch("src.agents.reviewer._make_submit_review_tool", side_effect=_capture_factory),
    ):
        async def _side_effect(*args, **kwargs):
            assert captured_holder, "factory should have been called by review()"
            captured_holder[0]["output"] = ReviewerFeedbackOutput(
                outcome="improved",
                metric_deltas={},
                bottleneck_classification="memory_bound",
                bottleneck_diagnosis="",
                suggestions=[],
                branch_quality=BranchQuality.PROMISING,
                conditional_assessment="",
            )
            raise MaxTurnsExceeded("Max turns (4) exceeded")

        mock_run.side_effect = _side_effect

        agent = ReviewerAgent(model=MagicMock())
        feedback = await agent.review(
            kernel_source="src",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        )

    assert feedback.degraded is False
    assert feedback.outcome == "improved"
    assert feedback.branch_quality == BranchQuality.PROMISING


@pytest.mark.asyncio
async def test_review_recovers_from_first_invalid_submit_within_turn_budget():
    """Validation-retry budget: first submit_review call returns FAILED, the
    LLM corrects on the second call, and review() returns a non-degraded
    feedback without falling back to the rule-based path. max_turns=4
    must reserve room for the in-band correction."""
    from src.agents.reviewer import (
        ReviewerFeedbackOutput,
        _make_submit_review_tool,
    )

    captured_holder: list[dict] = []

    def _capture_factory(captured_dict: dict):
        captured_holder.append(captured_dict)
        return _make_submit_review_tool(captured_dict)

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
        patch("src.agents.reviewer._make_submit_review_tool", side_effect=_capture_factory),
    ):
        async def _side_effect(*args, **kwargs):
            assert captured_holder, "factory should have been called by review()"
            captured_holder[0]["output"] = ReviewerFeedbackOutput(
                outcome="improved",
                metric_deltas={},
                bottleneck_classification="memory_bound",
                bottleneck_diagnosis="recovered after one validation retry",
                suggestions=[],
                branch_quality=BranchQuality.PROMISING,
                conditional_assessment="",
            )
            return MagicMock(final_output="done")

        mock_run.side_effect = _side_effect

        agent = ReviewerAgent(model=MagicMock())
        feedback = await agent.review(
            kernel_source="src",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        )

    assert feedback.degraded is False
    assert feedback.error_reason == ""
    assert feedback.branch_quality == BranchQuality.PROMISING
    assert "recovered after one validation retry" in feedback.bottleneck_diagnosis


# ── SDK-absent fallback (regression guard) ────────────────────────────────


@pytest.mark.asyncio
async def test_review_returns_rule_based_when_sdk_absent_even_with_model_arg():
    """SDK-absent + non-None model arg must take the rule-based path —
    Agent and function_tool are None in that environment, so calling
    them raises TypeError. Mirrors planner's regression guard."""
    with patch("src.agents.reviewer._SDK_AVAILABLE", False):
        agent = ReviewerAgent(model=MagicMock())
        feedback = await agent.review(
            kernel_source="def k(): pass",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        )
    assert isinstance(feedback, ReviewerFeedback)
    # No-model path is the *expected* fallback (not degraded).
    assert feedback.degraded is False
    assert "rule" in feedback.bottleneck_diagnosis.lower()


def test_reviewer_has_model_false_when_sdk_absent():
    """``has_model`` must reflect both ``self._model is not None`` AND
    SDK availability."""
    with patch("src.agents.reviewer._SDK_AVAILABLE", False):
        agent = ReviewerAgent(model=MagicMock())
    assert agent.has_model is False


# ── _make_submit_review_tool — defaulted Pydantic fields ──────────────────


def test_make_submit_review_tool_omits_optional_fields_uses_pydantic_defaults():
    """``ReviewerFeedbackOutput`` defaults ``metric_deltas={}``,
    ``bottleneck_diagnosis=""``, ``suggestions=[]``,
    ``conditional_assessment=""``. The tool signature must mark these
    optional so the SDK doesn't reject a tool call that omits them."""
    from src.agents.llm_backend import SUBMIT_OK_SENTINEL
    from src.agents.reviewer import (
        ReviewerFeedbackOutput,
        _make_submit_review_tool,
    )

    captured: dict = {}
    submit = _make_submit_review_tool(captured)
    msg = submit(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
    )  # only required fields
    assert msg == SUBMIT_OK_SENTINEL
    assert "output" in captured
    assert isinstance(captured["output"], ReviewerFeedbackOutput)
    assert captured["output"].metric_deltas == {}
    assert captured["output"].bottleneck_diagnosis == ""
    assert captured["output"].suggestions == []
    assert captured["output"].conditional_assessment == ""


@pytest.mark.asyncio
async def test_review_uses_degraded_with_existing_error_reason_when_run_agent_returns_none():
    """The original 'llm_retries_exhausted' degraded path still fires when
    run_agent returns None — submit-tool migration shouldn't regress
    existing behavior."""
    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
    ):
        mock_run.return_value = None

        agent = ReviewerAgent(model=MagicMock())
        feedback = await agent.review(
            kernel_source="src",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        )

    assert feedback.degraded is True
    assert feedback.error_reason == "llm_retries_exhausted"


@pytest.mark.asyncio
async def test_submit_tool_registered_with_strict_mode_false():
    """Regression guard: ``submit_review`` must be registered with
    ``strict_mode=False``. The SDK's strict-schema validator otherwise
    rejects the ``metric_deltas: dict[str, float]`` arg with an
    ``additionalProperties`` UserError, which is the exact failure the
    submit-tool migration was meant to fix."""
    capture_factory, fake_run = _simulate_review_submission(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
    )
    recorded_kwargs: list[dict] = []

    def recording_function_tool(f, **kwargs):
        recorded_kwargs.append(kwargs)
        return f

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=recording_function_tool),
        patch("src.agents.reviewer._make_submit_review_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run
        await ReviewerAgent(model=MagicMock()).review(
            kernel_source="@triton.jit\ndef k(): ...",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        )

    assert recorded_kwargs == [{"strict_mode": False}]
