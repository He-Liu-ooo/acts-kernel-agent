"""Planner agent — analyzes profiling data + memory, produces structured plan.

Single-call agent (no tools). Uses OpenAI Agents SDK Agent + Runner.run
with Pydantic output_type for structured output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel, ValidationError

try:
    from agents import Agent, MaxTurnsExceeded, OpenAIChatCompletionsModel, function_tool

    _SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    Agent = None  # type: ignore[assignment]
    function_tool = None  # type: ignore[assignment]

    class MaxTurnsExceeded(Exception):  # type: ignore[no-redef]
        """SDK-absent test stand-in. The real exception lives in ``agents``."""

    _SDK_AVAILABLE = False

if TYPE_CHECKING:
    from agents import Agent, OpenAIChatCompletionsModel

    from src.eval.types import BottleneckType
    from src.memory.experience import Experience

from src.agents.llm_backend import (
    SUBMIT_OK_SENTINEL,
    format_submit_validation_error,
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


def _make_submit_plan_tool(captured: dict) -> Callable[..., str]:
    """Build a submit tool that captures the LLM's final ``OptimizationPlanOutput``.

    Mirrors ``coder._make_submit_tool``: runs Pydantic validation in the
    tool body, stores the validated output on success, returns the
    standard error string on failure (which the SDK hands back to the
    LLM for in-loop retry within the turn budget).
    """

    def submit_plan(
        tier: int,
        technique: str,
        params: dict[str, str],
        target_region: str,
        rationale: str,
    ) -> str:
        try:
            captured["output"] = OptimizationPlanOutput(
                tier=tier,
                technique=technique,
                params=params,
                target_region=target_region,
                rationale=rationale,
            )
        except ValidationError as exc:
            return format_submit_validation_error("submit_plan", exc)
        return SUBMIT_OK_SENTINEL

    return submit_plan


def _validate_and_convert(
    out: OptimizationPlanOutput, available_actions: list[str]
) -> OptimizationPlan:
    """Convert + enforce the available-actions guard. Kept separate from
    ``_output_to_plan`` so the conversion can run on captured submit-tool
    output without re-implementing the guard."""
    plan = _output_to_plan(out)
    if available_actions and plan.technique not in available_actions:
        raise PlanningError(
            f"LLM selected technique '{plan.technique}' "
            f"not in available actions: {available_actions}"
        )
    return plan


class PlannerAgent:
    """Selects optimization technique from action library based on
    profiling data, past experiences, and Reviewer feedback.

    Single-call, no tools — the orchestrator provides all context.
    """

    def __init__(self, model: OpenAIChatCompletionsModel | None = None) -> None:
        self._model = model
        if model is not None and _SDK_AVAILABLE:
            self._instructions = (PROMPT_DIR / "system.md").read_text()
        else:
            self._instructions = ""

    @property
    def has_model(self) -> bool:
        """True when the agent is backed by a real LLM."""
        return self._model is not None

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
        """Generate a structured optimization plan for the next iteration.

        Submits via a ``submit_plan`` tool call so the SDK never sends a
        ``response_format=json_schema`` (which DeepSeek-reasoner rejects and
        which the SDK's strict-schema validator rejects on
        ``params: dict[str, str]``). Pydantic validation still runs inside
        the tool body. Mirrors Coder's option α / γ failure contract.
        """
        if not self.has_model:
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

        captured: dict = {}
        submit_tool = function_tool(_make_submit_plan_tool(captured))
        agent = Agent(
            name="Planner",
            instructions=self._instructions,
            model=self._model,
            tools=[submit_tool],
        )

        try:
            result = await run_agent(
                agent,
                prompt,
                run_config=make_run_config(temperature=0.3),
                max_turns=2,
            )
        except MaxTurnsExceeded as exc:
            if "output" in captured:
                return _validate_and_convert(captured["output"], available_actions)
            raise PlanningError(
                "Planner exhausted turn budget (2) without calling submit_plan."
            ) from exc

        if result is None:
            raise PlanningError("LLM call failed after all retries.")
        if "output" not in captured:
            raise PlanningError(
                "Planner did not call submit_plan before terminating — "
                "no final plan was emitted."
            )
        return _validate_and_convert(captured["output"], available_actions)
