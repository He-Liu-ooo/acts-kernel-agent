"""Reviewer agent — interprets eval results into structured feedback.

Single-call agent (no tools). Uses OpenAI Agents SDK Agent + Runner.run with a
Pydantic output_type. Falls back to rule-based feedback when the LLM is
unavailable or its call fails after all retries.

Designed for future split into Compute-Reviewer and Memory-Reviewer sub-agents:
`prompt_dir` is a constructor parameter, so a specialized instance is one call
away without subclassing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

try:
    from agents import Agent, OpenAIChatCompletionsModel

    _SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _SDK_AVAILABLE = False

if TYPE_CHECKING:
    from agents import Agent, OpenAIChatCompletionsModel, RunResult

from src.agents.llm_backend import make_run_config, render_kernel_section, run_agent

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "reviewer"

# Canonical bottleneck labels (matches roofline + orchestrator usage).
BottleneckLabel = Literal["memory_bound", "compute_bound", "balanced"]

# Thresholds for rule-based fallback (kept small so tiny numerical noise
# doesn't flip outcomes).
_SOL_DELTA_EPSILON = 0.02
_HEADROOM_PLATEAU_PCT = 20.0


# ── Branch quality enum ────────────────────────────────────────────────


class BranchQuality(str, Enum):
    """Reviewer's assessment of a search tree branch."""

    PROMISING = "promising"
    BLOCKED_POTENTIAL = "blocked_potential"
    PLATEAU = "plateau"
    DEAD_END = "dead_end"


# ── Pydantic output model (schema sent to LLM via output_type) ─────────


class ReviewerFeedbackOutput(BaseModel):
    """Structured output schema enforced on the LLM response."""

    outcome: str  # non-strict: accept free-form strings (e.g. "partially_improved")
    metric_deltas: dict[str, float] = Field(default_factory=dict)
    bottleneck_classification: BottleneckLabel  # strict literal
    bottleneck_diagnosis: str = ""
    suggestions: list[str] = Field(default_factory=list)
    branch_quality: BranchQuality  # strict enum
    conditional_assessment: str = ""


# ── Plain dataclass used internally ────────────────────────────────────


@dataclass
class ReviewerFeedback:
    """Structured feedback from the Reviewer agent.

    *degraded* is True when this feedback did not come from a healthy LLM
    call — e.g., all retries exhausted, schema validation failed repeatedly.
    The orchestrator should surface this (log, halt, or down-weight the
    signal) so a broken reviewer doesn't silently drive search decisions.
    *error_reason* is a short machine-readable tag ("llm_retries_exhausted",
    etc.) — empty when not degraded.
    """

    outcome: str
    metric_deltas: dict[str, float] = field(default_factory=dict)
    bottleneck_classification: str = ""
    bottleneck_diagnosis: str = ""
    suggestions: list[str] = field(default_factory=list)
    branch_quality: BranchQuality = BranchQuality.PROMISING
    conditional_assessment: str = ""
    degraded: bool = False
    error_reason: str = ""


def _output_to_feedback(out: ReviewerFeedbackOutput) -> ReviewerFeedback:
    """Convert Pydantic output to internal dataclass."""
    return ReviewerFeedback(
        outcome=out.outcome,
        metric_deltas=dict(out.metric_deltas),
        bottleneck_classification=out.bottleneck_classification,
        bottleneck_diagnosis=out.bottleneck_diagnosis,
        suggestions=list(out.suggestions),
        branch_quality=out.branch_quality,
        conditional_assessment=out.conditional_assessment,
    )


# ── Rule-based fallback ────────────────────────────────────────────────


def rule_based_feedback(
    sol_score: float,
    prev_sol_score: float | None,
    headroom_pct: float,
    bottleneck: str,
    degraded: bool = False,
    error_reason: str = "",
) -> ReviewerFeedback:
    """Derive feedback from raw metrics when the LLM is unavailable.

    Mapping:
      sol_delta > +epsilon             -> "improved"
        + headroom > 20%               -> PROMISING
        + headroom <= 20%              -> PLATEAU
      sol_delta < -epsilon              -> "regressed" -> DEAD_END
      otherwise (incl. missing prev)   -> "neutral"    -> BLOCKED_POTENTIAL

    *degraded* / *error_reason* mark the result when it comes from an
    LLM failure rather than an expected configuration (no model).
    """
    if prev_sol_score is None:
        outcome = "neutral"
        branch = BranchQuality.BLOCKED_POTENTIAL
        delta = 0.0
    else:
        delta = sol_score - prev_sol_score
        if delta > _SOL_DELTA_EPSILON:
            outcome = "improved"
            branch = (
                BranchQuality.PROMISING
                if headroom_pct > _HEADROOM_PLATEAU_PCT
                else BranchQuality.PLATEAU
            )
        elif delta < -_SOL_DELTA_EPSILON:
            outcome = "regressed"
            branch = BranchQuality.DEAD_END
        else:
            outcome = "neutral"
            branch = BranchQuality.BLOCKED_POTENTIAL

    diagnosis = (
        f"Rule-based fallback — LLM degraded ({error_reason})."
        if degraded
        else "Rule-based fallback — LLM unavailable."
    )
    return ReviewerFeedback(
        outcome=outcome,
        metric_deltas={"sol_score": delta} if prev_sol_score is not None else {},
        bottleneck_classification=bottleneck,
        bottleneck_diagnosis=diagnosis,
        branch_quality=branch,
        degraded=degraded,
        error_reason=error_reason,
    )


# ── Agent ──────────────────────────────────────────────────────────────


class ReviewerAgent:
    """Interprets evaluation results and produces structured feedback.

    Acts as intelligent filter between raw profiling data and the Planner.
    Single-call, no tools — receives all eval data in the prompt.

    `prompt_dir` is configurable so future Compute-Reviewer / Memory-Reviewer
    sub-agents can load specialized system prompts without subclassing.
    """

    def __init__(
        self,
        model: OpenAIChatCompletionsModel | None = None,
        prompt_dir: Path = PROMPT_DIR,
    ) -> None:
        self._prompt_dir = prompt_dir
        if model is not None and _SDK_AVAILABLE:
            self._agent: Agent | None = Agent(
                name="Reviewer",
                instructions=(prompt_dir / "system.md").read_text(),
                model=model,
                output_type=ReviewerFeedbackOutput,
            )
        else:
            self._agent = None

    # ── prompt assembly ─────────────────────────────────────────────

    @staticmethod
    def build_user_prompt(
        kernel_source: str,
        profiling_summary: str,
        sol_score: float,
        headroom_pct: float,
        bottleneck: str,
        tree_context: str = "",
        kb_context: str = "",
    ) -> str:
        """Assemble the user prompt from runtime data."""
        sections: list[str] = []

        sections.append(render_kernel_section(kernel_source))
        sections.append("## Profiling summary\n" + profiling_summary)
        sections.append(
            "## Scoring\n"
            f"- SOL score: {sol_score:.3f}\n"
            f"- Headroom: {headroom_pct:.1f}%\n"
            f"- Current bottleneck: {bottleneck}"
        )

        if tree_context:
            sections.append("## Search tree context\n" + tree_context)

        if kb_context:
            sections.append("## Knowledge base context\n" + kb_context)

        return "\n\n".join(sections)

    # ── main entry point ────────────────────────────────────────────

    async def review(
        self,
        kernel_source: str,
        profiling_summary: str,
        sol_score: float,
        headroom_pct: float,
        bottleneck: str,
        tree_context: str = "",
        kb_context: str = "",
        prev_sol_score: float | None = None,
    ) -> ReviewerFeedback:
        """Interpret eval results into structured Reviewer feedback.

        Falls back to rule-based feedback if no model is configured or if the
        LLM call exhausts all retries. Strict Pydantic validation on the
        output makes hallucinated bottleneck/branch_quality values surface as
        retry-worthy errors inside `run_agent`.
        """
        if self._agent is None:
            return rule_based_feedback(
                sol_score=sol_score,
                prev_sol_score=prev_sol_score,
                headroom_pct=headroom_pct,
                bottleneck=bottleneck,
            )

        prompt = self.build_user_prompt(
            kernel_source=kernel_source,
            profiling_summary=profiling_summary,
            sol_score=sol_score,
            headroom_pct=headroom_pct,
            bottleneck=bottleneck,
            tree_context=tree_context,
            kb_context=kb_context,
        )
        result = await run_agent(
            self._agent,
            prompt,
            run_config=make_run_config(temperature=0.3),
        )
        if result is None:
            # LLM call exhausted retries — do NOT mask as ordinary fallback.
            # The orchestrator uses `degraded` to surface/halt on broken runs.
            return rule_based_feedback(
                sol_score=sol_score,
                prev_sol_score=prev_sol_score,
                headroom_pct=headroom_pct,
                bottleneck=bottleneck,
                degraded=True,
                error_reason="llm_retries_exhausted",
            )

        return _output_to_feedback(result.final_output)

    @staticmethod
    def parse_feedback(result: RunResult) -> ReviewerFeedback:
        """Parse the LLM's final_output into ReviewerFeedback."""
        return _output_to_feedback(result.final_output)
