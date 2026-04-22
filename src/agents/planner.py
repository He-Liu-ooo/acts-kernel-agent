"""Planner agent — analyzes profiling data + memory, produces structured plan.

Single-call agent (no tools). Uses OpenAI Agents SDK Agent + Runner.run
with Pydantic output_type for structured output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

try:
    from agents import Agent, OpenAIChatCompletionsModel

    _SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _SDK_AVAILABLE = False

if TYPE_CHECKING:
    from agents import Agent, OpenAIChatCompletionsModel, RunResult

    from src.eval.types import BottleneckType
    from src.memory.experience import Experience

from src.agents.llm_backend import (
    make_run_config,
    render_kernel_section,
    render_run_context,
    run_agent,
)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "planner"


# ── Pydantic output model ──────────────────────────────────────────────


class OptimizationPlanOutput(BaseModel):
    """Structured output schema sent to the LLM via output_type."""

    tier: int
    technique: str
    params: dict[str, str] = {}
    target_region: str = ""
    rationale: str = ""


# ── Plain dataclass used internally ────────────────────────────────────


@dataclass
class OptimizationPlan:
    """Structured plan output from the Planner agent."""

    tier: int
    technique: str
    params: dict[str, str] = field(default_factory=dict)
    target_region: str = ""
    rationale: str = ""


def _output_to_plan(out: OptimizationPlanOutput) -> OptimizationPlan:
    """Convert Pydantic output to internal dataclass."""
    return OptimizationPlan(
        tier=out.tier,
        technique=out.technique,
        params=dict(out.params),
        target_region=out.target_region,
        rationale=out.rationale,
    )


class PlanningError(Exception):
    """Raised when the Planner cannot produce a valid plan."""


_DEFAULT_PLAN = OptimizationPlan(
    tier=1,
    technique="block_size_tuning",
    params={"block_size": "128"},
    target_region="main kernel loop",
    rationale="Placeholder — no LLM configured.",
)


class PlannerAgent:
    """Selects optimization technique from action library based on
    profiling data, past experiences, and Reviewer feedback.

    Single-call, no tools — the orchestrator provides all context.
    """

    def __init__(self, model: OpenAIChatCompletionsModel | None = None) -> None:
        if model is not None and _SDK_AVAILABLE:
            self._agent: Agent | None = Agent(
                name="Planner",
                instructions=(PROMPT_DIR / "system.md").read_text(),
                model=model,
                output_type=OptimizationPlanOutput,
            )
        else:
            self._agent = None

    # ── prompt assembly ─────────────────────────────────────────────

    @staticmethod
    def build_user_prompt(
        kernel_source: str,
        profiling_summary: str,
        past_experiences: list[Experience],
        available_actions: list[str],
        tree_context: str = "",
        reviewer_feedback: str | None = None,
        bottleneck: BottleneckType | None = None,
    ) -> str:
        """Assemble the user prompt from runtime data.

        ``bottleneck`` (when set) is rendered as a dedicated
        "## Run context" section so the Planner sees the once-per-run
        classification up front instead of having to reparse it from a
        profiling summary.
        """
        sections: list[str] = []

        sections.append(render_kernel_section(kernel_source))
        if bottleneck is not None:
            sections.append(render_run_context(bottleneck))
        sections.append("## Profiling summary\n" + profiling_summary)

        if past_experiences:
            lines = []
            for exp in past_experiences:
                status = "success" if exp.success else "failure"
                params = ", ".join(
                    f"{k}={v}" for k, v in exp.action_applied.parameters.items()
                )
                params_str = f" [{params}]" if params else ""
                lines.append(
                    f"- {exp.action_applied.name} (tier {exp.action_applied.tier}){params_str}: "
                    f"{status}, speedup {exp.speedup}x, "
                    f"bottleneck_before {exp.bottleneck_before.value}"
                )
            sections.append("## Past experiences\n" + "\n".join(lines))

        sections.append(
            "## Available actions\n" + "\n".join(f"- {a}" for a in available_actions)
        )

        if tree_context:
            sections.append("## Search tree context\n" + tree_context)

        if reviewer_feedback:
            sections.append("## Reviewer feedback\n" + reviewer_feedback)

        return "\n\n".join(sections)

    # ── main entry point ────────────────────────────────────────────

    async def plan(
        self,
        kernel_source: str,
        profiling_summary: str,
        past_experiences: list[Experience],
        available_actions: list[str],
        tree_context: str = "",
        reviewer_feedback: str | None = None,
        bottleneck: BottleneckType | None = None,
    ) -> OptimizationPlan:
        """Generate a structured optimization plan for the next iteration."""
        if self._agent is None:
            return _DEFAULT_PLAN

        prompt = self.build_user_prompt(
            kernel_source=kernel_source,
            profiling_summary=profiling_summary,
            past_experiences=past_experiences,
            available_actions=available_actions,
            tree_context=tree_context,
            reviewer_feedback=reviewer_feedback,
            bottleneck=bottleneck,
        )
        result = await run_agent(
            self._agent,
            prompt,
            run_config=make_run_config(temperature=0.3),
        )
        if result is None:
            raise PlanningError("LLM call failed after all retries.")

        plan = _output_to_plan(result.final_output)

        if available_actions and plan.technique not in available_actions:
            raise PlanningError(
                f"LLM selected technique '{plan.technique}' "
                f"not in available actions: {available_actions}"
            )

        return plan

    @staticmethod
    def parse_plan(result: RunResult) -> OptimizationPlan:
        """Parse the LLM's final_output into an OptimizationPlan."""
        return _output_to_plan(result.final_output)
