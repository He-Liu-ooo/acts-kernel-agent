"""Planner agent — analyzes profiling data + memory, produces structured plan.

Single-call agent (no tools). Uses OpenAI Agents SDK Agent + Runner.run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from agents import Agent, OpenAIChatCompletionsModel

    _SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _SDK_AVAILABLE = False

if TYPE_CHECKING:
    from agents import Agent, OpenAIChatCompletionsModel, RunResult

    from src.memory.experience import Experience

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "planner"


@dataclass
class OptimizationPlan:
    """Structured plan output from the Planner agent."""

    tier: int
    technique: str
    params: dict[str, str] = field(default_factory=dict)
    target_region: str = ""
    rationale: str = ""


class PlannerAgent:
    """Selects optimization technique from action library based on
    profiling data, past experiences, and Reviewer feedback.

    Single-call, no tools — the orchestrator provides all context.
    """

    def __init__(self, model: OpenAIChatCompletionsModel | None = None) -> None:
        if model is not None and _SDK_AVAILABLE:
            self._agent = Agent(
                name="Planner",
                instructions=(PROMPT_DIR / "system.md").read_text(),
                model=model,
            )
        else:
            self._agent = None

    async def plan(
        self,
        kernel_source: str,
        profiling_summary: str,
        past_experiences: list[Experience],
        available_actions: list[str],
        tree_context: str = "",
        reviewer_feedback: str | None = None,
    ) -> OptimizationPlan:
        """Generate a structured optimization plan for the next iteration.

        Builds the user prompt from profiling summary, past experiences,
        available actions, and optional reviewer feedback, then runs
        a single LLM call via Runner.run.
        """
        # Placeholder: return a default Tier 1 plan without calling the LLM.
        return OptimizationPlan(
            tier=1,
            technique="block_size_tuning",
            params={"block_size": "128"},
            target_region="main kernel loop",
            rationale="Placeholder — no LLM configured.",
        )

    @staticmethod
    def parse_plan(result: RunResult) -> OptimizationPlan:
        """Parse the LLM's final_output into an OptimizationPlan."""
        # Placeholder: return default plan from any result.
        return OptimizationPlan(
            tier=1,
            technique="block_size_tuning",
            rationale="Parsed from LLM output (placeholder).",
        )
