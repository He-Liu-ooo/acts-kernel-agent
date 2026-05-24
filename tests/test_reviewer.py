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


# ── review() — multi-turn (flag on) ────────────────────────────────────


@pytest.mark.asyncio
async def test_review_flag_on_registers_both_tools_with_strict_mode_false():
    """When reviewer_metric_queries=True, both submit_review AND query_metric
    must be registered, both with strict_mode=False. Same SDK strict-schema
    trap as the planner submit-tool dict params."""
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
            reviewer_metric_queries=True,
            iter_idx=2,
        )

    assert len(recorded_kwargs) == 2
    assert all(kw == {"strict_mode": False} for kw in recorded_kwargs)


@pytest.mark.asyncio
async def test_review_flag_on_uses_max_turns_6():
    """The multi-turn path budgets max_turns=6; flag-off keeps max_turns=4."""
    capture_factory, fake_run = _simulate_review_submission(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
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
        await ReviewerAgent(model=MagicMock()).review(
            kernel_source="def k(): pass",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
            reviewer_metric_queries=True,
            iter_idx=0,
        )

    mock_run.assert_awaited_once()
    assert mock_run.await_args.kwargs.get("max_turns") == 6


@pytest.mark.asyncio
async def test_review_flag_off_still_uses_max_turns_4():
    """Regression guard: with the flag off (default), the existing
    max_turns=4 path is unchanged."""
    capture_factory, fake_run = _simulate_review_submission(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
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
        await ReviewerAgent(model=MagicMock()).review(
            kernel_source="def k(): pass",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        )

    assert mock_run.await_args.kwargs.get("max_turns") == 4


@pytest.mark.asyncio
async def test_review_max_turns_kwarg_overrides_flag_off_default():
    """review(max_turns=N, reviewer_metric_queries=False) → run_agent
    receives N, overriding the hardcoded 4."""
    capture_factory, fake_run = _simulate_review_submission(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
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
        await ReviewerAgent(model=MagicMock()).review(
            kernel_source="def k(): pass",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
            max_turns=7,
        )

    assert mock_run.await_args.kwargs.get("max_turns") == 7


@pytest.mark.asyncio
async def test_review_max_turns_kwarg_overrides_flag_on_default():
    """review(max_turns=N, reviewer_metric_queries=True) → run_agent
    receives N, overriding the hardcoded 6."""
    capture_factory, fake_run = _simulate_review_submission(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
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
        await ReviewerAgent(model=MagicMock()).review(
            kernel_source="def k(): pass",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
            reviewer_metric_queries=True,
            max_turns=10,
        )

    assert mock_run.await_args.kwargs.get("max_turns") == 10


@pytest.mark.asyncio
async def test_review_max_turns_none_preserves_4_or_6_toggle():
    """Regression guard: omitting max_turns keeps today's 4/6 conditional
    intact (verified by the two existing tests above)."""
    # Covered by:
    #   test_review_flag_off_still_uses_max_turns_4  (None + off → 4)
    #   test_review_flag_on_uses_max_turns_6         (None + on  → 6)
    # This sentinel exists so a future refactor that drops the toggle
    # without updating those tests at least surfaces the intent here.
    pass


@pytest.mark.asyncio
async def test_review_flag_on_threads_raw_metrics_into_prompt_menu():
    """End-to-end: review(reviewer_metric_queries=True, profiling=...)
    builds the prompt with the menu populated from profiling.raw_metrics."""
    from src.eval.profiler import AnalyticalMetrics, ProfilingResult

    profiling = ProfilingResult(
        analytical=AnalyticalMetrics(
            achieved_tflops=5.0, achieved_bandwidth_gb_s=200.0,
            pct_peak_compute=0.4, pct_peak_bandwidth=0.5,
        ),
        ncu=None,
        raw_metrics={"sm__a.avg": 1.0, "sm__b.avg": 2.0},
    )
    capture_factory, fake_run = _simulate_review_submission(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
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
        await ReviewerAgent(model=MagicMock()).review(
            kernel_source="def k(): pass",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
            profiling=profiling,
            reviewer_metric_queries=True,
            iter_idx=1,
        )

    sent_prompt = mock_run.await_args.args[1]
    assert "## Available raw metrics (queryable)" in sent_prompt
    assert "- sm__a.avg" in sent_prompt
    assert "- sm__b.avg" in sent_prompt


@pytest.mark.asyncio
async def test_review_flag_on_two_queries_plus_invalid_submit_busts_budget():
    """Budget regression: with the flag on, the worst-case path of
    `2 query_metric calls + invalid submit_review + corrected submit_review`
    requires more than `max_turns=6` turns. The prompt caps fetches at one
    per review precisely to preserve submit-retry headroom; this test
    confirms the degraded fallback fires cleanly when an LLM ignores the
    heuristic and walks the budget-bust path, rather than raising or
    returning a partially-formed feedback."""
    from src.agents.reviewer import MaxTurnsExceeded

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
    ):
        # SDK reports `Max turns (6) exceeded` after the LLM walked
        # query → response → query → response → invalid submit (validation
        # error response was the 6th turn) — no captured submission.
        mock_run.side_effect = MaxTurnsExceeded("Max turns (6) exceeded")

        feedback = await ReviewerAgent(model=MagicMock()).review(
            kernel_source="src",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
            reviewer_metric_queries=True,
            iter_idx=0,
        )

    # Must degrade cleanly to rule-based, with the existing tag — no new
    # error_reason values, no exception escape.
    assert feedback.degraded is True
    assert feedback.error_reason == "max_turns_exceeded"
    assert feedback.bottleneck_classification == "memory_bound"


@pytest.mark.asyncio
async def test_review_flag_on_max_turns_exceeded_no_capture_degrades():
    """Flag on, MaxTurnsExceeded with no submit captured → existing degraded
    fallback fires with error_reason='max_turns_exceeded' (no new tag)."""
    from src.agents.reviewer import MaxTurnsExceeded

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
    ):
        mock_run.side_effect = MaxTurnsExceeded("Max turns (6) exceeded")

        feedback = await ReviewerAgent(model=MagicMock()).review(
            kernel_source="src",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
            reviewer_metric_queries=True,
            iter_idx=0,
        )

    assert feedback.degraded is True
    assert feedback.error_reason == "max_turns_exceeded"


# ── build_user_prompt — metric menu append (multi-turn flag on) ────────


def _profiling_with(raw_metrics: dict[str, float] | None) -> "ProfilingResult":
    """Test helper: build a minimal ProfilingResult with the given raw_metrics.
    The analytical block is filler — the menu-rendering tests only exercise
    the raw_metrics-derived menu, not the profiling-summary section."""
    from src.eval.profiler import AnalyticalMetrics, ProfilingResult

    return ProfilingResult(
        analytical=AnalyticalMetrics(
            achieved_tflops=5.0, achieved_bandwidth_gb_s=200.0,
            pct_peak_compute=0.4, pct_peak_bandwidth=0.5,
        ),
        ncu=None,
        raw_metrics=raw_metrics or {},
    )


def test_build_user_prompt_no_menu_when_flag_off():
    """Default behavior (flag off): no menu section, regardless of
    raw_metrics. Existing flag-off tests already serve as a regression
    guard for the single-call default path."""
    agent = ReviewerAgent(model=None)
    prompt = agent.build_user_prompt(
        kernel_source="def k(): pass",
        profiling_summary="",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck=BottleneckType.MEMORY_BOUND,
        profiling=_profiling_with({"foo": 1.0, "bar": 2.0}),  # provided but flag off
    )
    assert "Available raw metrics" not in prompt


def test_build_user_prompt_menu_when_flag_on_and_raw_metrics_present():
    """Flag on + non-empty raw_metrics: menu lists keys alphabetically."""
    agent = ReviewerAgent(model=None)
    prompt = agent.build_user_prompt(
        kernel_source="def k(): pass",
        profiling_summary="",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck=BottleneckType.MEMORY_BOUND,
        reviewer_metric_queries=True,
        profiling=_profiling_with({"sm__b": 1.0, "sm__a": 2.0}),
    )
    assert "## Available raw metrics (queryable)" in prompt
    section = prompt.split("## Available raw metrics (queryable)")[1]
    assert section.index("- sm__a") < section.index("- sm__b")


def test_build_user_prompt_lists_metric_groups_when_available():
    """Flag on + metric_groups: prompt shows grouped inventory so Reviewer
    can query high-signal groups instead of guessing exact NCU names."""
    from src.eval.profiler import AnalyticalMetrics, ProfilingResult

    profiling = ProfilingResult(
        analytical=AnalyticalMetrics(
            achieved_tflops=5.0,
            achieved_bandwidth_gb_s=200.0,
            pct_peak_compute=0.4,
            pct_peak_bandwidth=0.5,
        ),
        raw_metrics={"foo": 1.0},
        metric_groups={
            "tensor_core": {
                "tc.present": {"status": "present", "value": 1.0},
                "tc.missing": {"status": "missing"},
            },
            "memory": {
                "mem.present": {"status": "present", "value": 2.0},
            },
        },
    )
    agent = ReviewerAgent(model=None)
    prompt = agent.build_user_prompt(
        kernel_source="def k(): pass",
        profiling_summary="",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck=BottleneckType.MEMORY_BOUND,
        reviewer_metric_queries=True,
        profiling=profiling,
    )
    assert "## Available metric groups (queryable)" in prompt
    assert "- tensor_core: 1 present, 1 missing" in prompt
    assert "- memory: 1 present, 0 missing" in prompt


def test_build_user_prompt_menu_degraded_notice_when_raw_metrics_empty():
    """Flag on + empty raw_metrics: degraded-state notice replaces the list."""
    agent = ReviewerAgent(model=None)
    prompt = agent.build_user_prompt(
        kernel_source="def k(): pass",
        profiling_summary="",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck=BottleneckType.MEMORY_BOUND,
        reviewer_metric_queries=True,
        profiling=_profiling_with({}),
    )
    assert "## Available raw metrics (queryable)" in prompt
    assert "[no NCU data" in prompt
    assert 'query_metric will return "[no data]"' in prompt


def test_build_user_prompt_menu_degraded_notice_when_profiling_none():
    """Flag on + profiling=None: same degraded notice as empty raw_metrics."""
    agent = ReviewerAgent(model=None)
    prompt = agent.build_user_prompt(
        kernel_source="def k(): pass",
        profiling_summary="",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck=BottleneckType.MEMORY_BOUND,
        reviewer_metric_queries=True,
        profiling=None,
    )
    assert "[no NCU data" in prompt


@pytest.mark.asyncio
async def test_review_partial_ncu_degradation_still_exposes_raw_metrics():
    """Invariant: `ProfilingResult.degraded` and `raw_metrics` are
    independent surfaces. A partial NCU parse failure sets
    `degraded_reason` while still leaving `raw_metrics` populated with
    whatever was successfully extracted. The multi-turn path must:

    1. show the menu with real keys (not the degraded notice), and
    2. return real values from `query_metric`.

    The system prompt's degraded-state guidance must point at the menu
    (visible to the LLM) — not at the abstract `degraded` flag — so the
    LLM doesn't skip the tool exactly when the only-partial-data case
    fires (the failure shape this feature is meant to recover from)."""
    from src.eval.profiler import AnalyticalMetrics, ProfilingResult
    from src.agents.reviewer import _make_query_metric_tool

    profiling = ProfilingResult(
        analytical=AnalyticalMetrics(
            achieved_tflops=5.0, achieved_bandwidth_gb_s=200.0,
            pct_peak_compute=0.4, pct_peak_bandwidth=0.5,
        ),
        ncu=None,                                       # curated NCU dataclass un-built
        raw_metrics={"sm__warps_active.avg.pct": 0.62}, # raw still populated
        degraded_reason="parse_partial",                # degraded flag set
    )
    assert profiling.degraded is True
    assert profiling.raw_metrics  # non-empty

    # 1. Menu rendering: real keys visible, NOT the degraded notice.
    capture_factory, fake_run = _simulate_review_submission(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
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
        await ReviewerAgent(model=MagicMock()).review(
            kernel_source="def k(): pass",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
            profiling=profiling,
            reviewer_metric_queries=True,
            iter_idx=1,
        )

    sent_prompt = mock_run.await_args.args[1]
    assert "## Available raw metrics (queryable)" in sent_prompt
    assert "- sm__warps_active.avg.pct" in sent_prompt
    # The degraded-notice must NOT appear when raw_metrics is non-empty.
    assert "[no NCU data" not in sent_prompt

    # 2. Tool body returns real values, not "[no data]".
    tool = _make_query_metric_tool(
        raw_metrics=profiling.raw_metrics, iter_idx=1
    )
    out = tool(names=["sm__warps_active.avg.pct"])
    assert out == {"sm__warps_active.avg.pct": "0.62"}


# ── _make_query_metric_tool — direct unit tests ────────────────────────


def test_make_query_metric_tool_all_known_names():
    """All-known names → dict with stringified float values."""
    from src.agents.reviewer import _make_query_metric_tool

    raw = {"sm__warps_active.avg.pct": 0.62, "smsp__cycles_active.sum": 1234.5}
    tool = _make_query_metric_tool(raw_metrics=raw, iter_idx=3)
    out = tool(names=["sm__warps_active.avg.pct", "smsp__cycles_active.sum"])
    assert out == {
        "sm__warps_active.avg.pct": "0.62",
        "smsp__cycles_active.sum": "1234.5",
    }


def test_make_query_metric_tool_group_query_returns_status_values():
    from src.agents.reviewer import _make_query_metric_tool

    raw = {"foo": 1.0}
    groups = {
        "tensor_core": {
            "tc.present": {"status": "present", "value": 12.5},
            "tc.missing": {"status": "missing"},
        }
    }
    tool = _make_query_metric_tool(
        raw_metrics=raw,
        iter_idx=0,
        metric_groups=groups,
    )

    out = tool(names=["group:tensor_core"])

    assert out == {
        "group:tensor_core.tc.present": "present: 12.5",
        "group:tensor_core.tc.missing": "missing",
    }


def test_make_query_metric_tool_all_unknown_names():
    """All-unknown names → all '[unknown]'."""
    from src.agents.reviewer import _make_query_metric_tool

    raw = {"foo": 1.0}
    tool = _make_query_metric_tool(raw_metrics=raw, iter_idx=0)
    out = tool(names=["bar", "baz"])
    assert out == {"bar": "[unknown]", "baz": "[unknown]"}


def test_make_query_metric_tool_partial_unknown():
    """Mixed known/unknown → mixed dict; one tool call serves both."""
    from src.agents.reviewer import _make_query_metric_tool

    raw = {"foo": 1.0}
    tool = _make_query_metric_tool(raw_metrics=raw, iter_idx=0)
    out = tool(names=["foo", "bar"])
    assert out == {"foo": "1.0", "bar": "[unknown]"}


def test_make_query_metric_tool_raw_metrics_none():
    """raw_metrics=None → all '[no data]' (NCU was degraded this iter)."""
    from src.agents.reviewer import _make_query_metric_tool

    tool = _make_query_metric_tool(raw_metrics=None, iter_idx=0)
    out = tool(names=["foo", "bar"])
    assert out == {"foo": "[no data]", "bar": "[no data]"}


def test_make_query_metric_tool_raw_metrics_empty_dict():
    """raw_metrics={} → all '[no data]' (treated identically to None)."""
    from src.agents.reviewer import _make_query_metric_tool

    tool = _make_query_metric_tool(raw_metrics={}, iter_idx=0)
    out = tool(names=["foo"])
    assert out == {"foo": "[no data]"}


def test_make_query_metric_tool_empty_names_list():
    """Empty names=[] → empty dict; defensive against degenerate LLM call."""
    from src.agents.reviewer import _make_query_metric_tool

    tool = _make_query_metric_tool(raw_metrics={"foo": 1.0}, iter_idx=0)
    assert tool(names=[]) == {}


def test_make_query_metric_tool_emits_event_per_call():
    """Tool body emits `reviewer_metric_query` event with iter, count, names[:8]."""
    from src.agents.reviewer import _make_query_metric_tool

    raw = {"foo": 1.0}
    tool = _make_query_metric_tool(raw_metrics=raw, iter_idx=5)

    with patch("src.agents.reviewer.events_emit") as mock_emit:
        tool(names=["foo", "bar"])

    mock_emit.assert_called_once()
    call_kwargs = mock_emit.call_args.kwargs
    call_args = mock_emit.call_args.args
    assert call_args[0] == "reviewer_metric_query"
    assert call_kwargs["iter"] == 5
    assert call_kwargs["count"] == 2
    assert call_kwargs["names"] == ["foo", "bar"]


def test_make_query_metric_tool_truncates_names_in_event_to_first_8():
    """names cap at 8 in the event payload — keeps events.jsonl bounded."""
    from src.agents.reviewer import _make_query_metric_tool

    tool = _make_query_metric_tool(raw_metrics={}, iter_idx=0)
    long_names = [f"m{i}" for i in range(12)]

    with patch("src.agents.reviewer.events_emit") as mock_emit:
        tool(names=long_names)

    call_kwargs = mock_emit.call_args.kwargs
    assert call_kwargs["count"] == 12  # full count
    assert call_kwargs["names"] == long_names[:8]  # truncated list


def test_make_query_metric_tool_non_list_names_returns_error_dict():
    """`strict_mode=False` means the SDK doesn't pre-validate `names`. If
    the model emits a bare string, the tool body MUST NOT iterate it
    char-by-char (silent garbage); it must return a recoverable error
    dict so the LLM can self-correct in-loop."""
    from src.agents.reviewer import _make_query_metric_tool

    tool = _make_query_metric_tool(raw_metrics={"foo": 1.0}, iter_idx=0)
    out = tool(names="foo")  # bare string — the failure mode this guard pins
    assert "_error" in out
    assert "list" in out["_error"].lower()
    assert "str" in out["_error"].lower()  # mentions actual type received


def test_make_query_metric_tool_none_names_returns_error_dict():
    """`names=None` must not raise — return a recoverable error dict."""
    from src.agents.reviewer import _make_query_metric_tool

    tool = _make_query_metric_tool(raw_metrics={"foo": 1.0}, iter_idx=0)
    out = tool(names=None)  # type: ignore[arg-type]
    assert "_error" in out
    assert "list" in out["_error"].lower()


def test_make_query_metric_tool_non_string_elements_are_coerced():
    """Element-level type drift (e.g. int names) is coerced via str();
    lookup proceeds against the stringified key. No exception, no garbage."""
    from src.agents.reviewer import _make_query_metric_tool

    tool = _make_query_metric_tool(raw_metrics={"42": 1.0}, iter_idx=0)
    out = tool(names=[42])  # type: ignore[list-item]
    assert out == {"42": "1.0"}


def test_make_query_metric_tool_invalid_input_does_not_emit_event():
    """Bad-input early-return path skips event emission — events.jsonl
    only records well-formed query attempts."""
    from src.agents.reviewer import _make_query_metric_tool

    tool = _make_query_metric_tool(raw_metrics={"foo": 1.0}, iter_idx=0)

    with patch("src.agents.reviewer.events_emit") as mock_emit:
        tool(names="foo")  # bare string

    mock_emit.assert_not_called()


# ── Prompt-leak guard: ``query_metric`` mention must mirror tool registration ──
#
# Regression for the live-run failure
#   ``ModelBehaviorError: Tool query_metric not found in agent Reviewer``
# The cause was that the static ``system.md`` documented a ``query_metric``
# tool unconditionally; the LLM rationally tried to call it even when the
# multi-turn flag was off and the tool was never registered. Symmetry between
# prompt mention and tool registration is now invariant: flag-off => no
# mention; flag-on => mention.


def test_reviewer_prompt_omits_metric_menu_when_flag_off():
    """Default path (``reviewer_metric_queries=False``): neither the user
    prompt nor the system instructions mention ``query_metric`` or
    ``Available raw metrics``. Mirrors the actual tool-registration gate
    in ``review()`` — flag off, only ``submit_review`` is registered, so
    nothing prompt-side may advertise the absent tool."""
    # 1. SDK-absent path (model=None) — the system file is not loaded, but
    #    user prompt assembly must still gate the menu.
    agent_no_model = ReviewerAgent(model=None)
    user_prompt_no_model = agent_no_model.build_user_prompt(
        kernel_source="def k(): pass",
        profiling_summary="",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck=BottleneckType.MEMORY_BOUND,
        # Flag defaults to False — explicit here for documentation.
        reviewer_metric_queries=False,
    )
    assert "query_metric" not in user_prompt_no_model
    assert "Available raw metrics" not in user_prompt_no_model

    # 2. SDK-present path (model is set) — the static system prompt is
    #    loaded from disk in ``__init__``. With the flag off, ``review()``
    #    threads ``_instructions_base`` (NOT the metric-queries addendum)
    #    into the Agent. ``_instructions_base`` itself must be clean.
    with patch("src.agents.reviewer._SDK_AVAILABLE", True):
        agent_with_model = ReviewerAgent(model=MagicMock())
    assert "query_metric" not in agent_with_model._instructions_base
    assert "Available raw metrics" not in agent_with_model._instructions_base
    # Back-compat: ``_instructions`` (legacy attr) defaults to the base —
    # the same string used when the flag is off.
    assert agent_with_model._instructions == agent_with_model._instructions_base


def test_reviewer_prompt_includes_metric_menu_when_flag_on():
    """Flag on (``reviewer_metric_queries=True``): both the user prompt
    menu AND the system addendum mention ``query_metric``. Symmetric with
    the tool registration in ``review()`` — flag on, both
    ``submit_review`` and ``query_metric`` are registered, so prompt-side
    mention is correct (and required for the LLM to use the tool)."""
    agent = ReviewerAgent(model=None)
    user_prompt = agent.build_user_prompt(
        kernel_source="def k(): pass",
        profiling_summary="",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck=BottleneckType.MEMORY_BOUND,
        reviewer_metric_queries=True,
        profiling=_profiling_with({"sm__a": 1.0, "sm__b": 2.0}),
    )
    assert "## Available raw metrics (queryable)" in user_prompt

    # System-prompt side: the addendum file must exist and mention
    # ``query_metric`` so the LLM knows the tool is callable. Loaded
    # lazily via ``__init__`` only when SDK + model are present.
    with patch("src.agents.reviewer._SDK_AVAILABLE", True):
        agent_with_model = ReviewerAgent(model=MagicMock())
    assert "query_metric" in agent_with_model._instructions_metric_queries


@pytest.mark.asyncio
async def test_review_threads_metric_queries_addendum_into_agent_instructions_when_flag_on():
    """End-to-end: with the flag on, ``review()`` must construct the
    Agent with instructions that DO mention ``query_metric`` (the
    addendum is appended to the base). With the flag off, the Agent's
    instructions must NOT mention it — symmetry guard against the
    prompt-leak regression sneaking back in."""
    capture_factory, fake_run = _simulate_review_submission(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
    )
    agent_kwargs_seen: list[dict] = []

    def recording_agent(**kwargs):
        agent_kwargs_seen.append(kwargs)
        return MagicMock()

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent", side_effect=recording_agent),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
        patch("src.agents.reviewer._make_submit_review_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run
        # Flag OFF — instructions must be clean.
        await ReviewerAgent(model=MagicMock()).review(
            kernel_source="def k(): pass",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        )

    assert len(agent_kwargs_seen) == 1
    instructions_off = agent_kwargs_seen[0]["instructions"]
    assert "query_metric" not in instructions_off
    assert "Available raw metrics" not in instructions_off

    # Flag ON — addendum must be appended, ``query_metric`` mentioned.
    capture_factory, fake_run = _simulate_review_submission(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
    )
    agent_kwargs_seen.clear()
    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent", side_effect=recording_agent),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
        patch("src.agents.reviewer._make_submit_review_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run
        await ReviewerAgent(model=MagicMock()).review(
            kernel_source="def k(): pass",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
            reviewer_metric_queries=True,
            iter_idx=0,
        )

    assert len(agent_kwargs_seen) == 1
    instructions_on = agent_kwargs_seen[0]["instructions"]
    assert "query_metric" in instructions_on


# ── Defensive ``ModelBehaviorError`` catch ─────────────────────────────


@pytest.mark.asyncio
async def test_review_handles_model_behavior_error_gracefully():
    """Defense-in-depth: if the SDK raises ``ModelBehaviorError`` (e.g. a
    future prompt-leak regression that mentions a tool the orchestrator
    didn't register), ``review()`` must degrade to rule-based feedback
    rather than letting the exception unwind ``Orchestrator.run()`` and
    abort the whole optimization run. Mirrors the existing
    ``MaxTurnsExceeded`` pattern."""
    from src.agents.reviewer import ModelBehaviorError

    with (
        patch("src.agents.reviewer._SDK_AVAILABLE", True),
        patch("src.agents.reviewer.Agent"),
        patch("src.agents.reviewer.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.reviewer.make_run_config", return_value=None),
        patch("src.agents.reviewer.function_tool", side_effect=lambda f, **kw: f),
    ):
        mock_run.side_effect = ModelBehaviorError(
            "Tool query_metric not found in agent Reviewer"
        )

        agent = ReviewerAgent(model=MagicMock())
        feedback = await agent.review(
            kernel_source="def k(): pass",
            profiling_summary="",
            sol_score=0.5,
            headroom_pct=50.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
            prev_sol_score=0.45,
        )

    assert isinstance(feedback, ReviewerFeedback)
    assert feedback.degraded is True
    assert feedback.error_reason.startswith("model_behavior_error")
    assert feedback.bottleneck_classification == "memory_bound"


@pytest.mark.asyncio
async def test_review_returns_partial_output_on_model_behavior_error_after_submission():
    """If a valid submit_review landed before the SDK raised
    ``ModelBehaviorError`` (e.g. the LLM submitted, then made a stray
    call to an unregistered tool), prefer the captured submission —
    same precedence rule as ``MaxTurnsExceeded``."""
    from src.agents.reviewer import (
        ModelBehaviorError,
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
                bottleneck_diagnosis="captured before stray call",
                suggestions=[],
                branch_quality=BranchQuality.PROMISING,
                conditional_assessment="",
            )
            raise ModelBehaviorError("Tool query_metric not found in agent Reviewer")

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
    assert "captured before stray call" in feedback.bottleneck_diagnosis


def test_reviewer_prompt_includes_sibling_section_when_provided():
    from src.agents.reviewer import ReviewerAgent
    from src.eval.types import BottleneckType

    sibling_text = "- t1_block_size_tuning {BLOCK_N:32}: SOL 0.434 (Δ -0.071), regressed, blocked_potential"
    prompt = ReviewerAgent.build_user_prompt(
        kernel_source="def k(): pass",
        profiling_summary="summary",
        sol_score=0.42,
        headroom_pct=58.0,
        bottleneck=BottleneckType.COMPUTE_BOUND,
        sibling_context=sibling_text,
    )
    assert "## Siblings already tried from this parent" in prompt
    assert sibling_text in prompt
    # Order: after Search tree context (when present), before Knowledge base context
    # Since tree_context="" here, just confirm presence + that KB section
    # would come after if it existed:
    assert "## Siblings already tried" in prompt


def test_reviewer_prompt_omits_sibling_section_when_empty():
    from src.agents.reviewer import ReviewerAgent
    from src.eval.types import BottleneckType

    prompt = ReviewerAgent.build_user_prompt(
        kernel_source="def k(): pass",
        profiling_summary="summary",
        sol_score=0.5,
        headroom_pct=50.0,
        bottleneck=BottleneckType.COMPUTE_BOUND,
        sibling_context="",
    )
    assert "## Siblings already tried" not in prompt
