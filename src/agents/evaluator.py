"""Evaluator (Reviewer) agent — interprets eval results into structured feedback.

Single-call agent (no tools). Uses OpenAI Agents SDK Agent + Runner.run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from agents import Agent, OpenAIChatCompletionsModel

    _SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _SDK_AVAILABLE = False

if TYPE_CHECKING:
    from agents import Agent, OpenAIChatCompletionsModel, RunResult

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "reviewer"


class BranchQuality(Enum):
    """Reviewer's assessment of a search tree branch."""

    PROMISING = "promising"
    BLOCKED_POTENTIAL = "blocked_potential"
    PLATEAU = "plateau"
    DEAD_END = "dead_end"


@dataclass
class ReviewerFeedback:
    """Structured feedback from the Reviewer agent."""

    outcome: str  # "improved", "regressed", "neutral"
    metric_deltas: dict[str, float] = field(default_factory=dict)
    bottleneck_classification: str = ""
    bottleneck_diagnosis: str = ""
    suggestions: list[str] = field(default_factory=list)
    branch_quality: BranchQuality = BranchQuality.PROMISING
    conditional_assessment: str = ""


class EvaluatorAgent:
    """Interprets evaluation results and produces structured feedback.

    Acts as intelligent filter between raw profiling data and the Planner.
    Single-call, no tools — receives all eval data in the prompt.
    """

    def __init__(self, model: OpenAIChatCompletionsModel | None = None) -> None:
        if model is not None and _SDK_AVAILABLE:
            self._agent = Agent(
                name="Reviewer",
                instructions=(PROMPT_DIR / "system.md").read_text(),
                model=model,
            )
        else:
            self._agent = None

    async def review(
        self,
        kernel_source: str,
        profiling_summary: str,
        sol_score: float,
        headroom_pct: float,
        bottleneck: str,
    ) -> ReviewerFeedback:
        """Interpret eval results into structured Reviewer feedback.

        Builds the user prompt from profiling data, SOL score, and
        bottleneck classification, then runs a single LLM call.
        """
        # Placeholder: return neutral feedback without calling the LLM.
        return ReviewerFeedback(
            outcome="neutral",
            bottleneck_classification=bottleneck,
            bottleneck_diagnosis="Placeholder — no LLM configured.",
            branch_quality=BranchQuality.PROMISING,
        )

    @staticmethod
    def parse_feedback(result: RunResult) -> ReviewerFeedback:
        """Parse the LLM's final_output into ReviewerFeedback."""
        # Placeholder: return neutral feedback from any result.
        return ReviewerFeedback(
            outcome="neutral",
            bottleneck_diagnosis="Parsed from LLM output (placeholder).",
        )
