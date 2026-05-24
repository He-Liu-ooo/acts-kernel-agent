"""Planner agent — analyzes profiling data + memory, produces structured plan.

Uses the OpenAI Agents SDK with a single ``submit_plan`` tool: the LLM
calls ``submit_plan`` with the structured payload, which is validated by
the matching Pydantic model inside the tool body before being returned
to the caller.
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
    """Structured output schema validated on the submit_plan tool payload."""

    tier: int
    technique: str
    params: dict[str, str] = {}
    target_region: str = ""
    rationale: str = ""
    # Each dict is a partial-match pattern; the Coder validator rejects
    # any submitted @triton.autotune Config whose listed keys all match.
    # Top-level field (not inside ``params``) so the existing strict-mode
    # str-only ``params`` workaround stays intact.
    autotune_exclude: list[dict[str, int]] = []


# ── Plain dataclass used internally ────────────────────────────────────


@dataclass
class OptimizationPlan:
    """Structured plan output from the Planner agent."""

    tier: int
    technique: str
    params: dict[str, str] = field(default_factory=dict)
    target_region: str = ""
    rationale: str = ""
    autotune_exclude: list[dict[str, int]] = field(default_factory=list)


def _output_to_plan(out: OptimizationPlanOutput) -> OptimizationPlan:
    """Convert Pydantic output to internal dataclass."""
    return OptimizationPlan(
        tier=out.tier,
        technique=out.technique,
        params=dict(out.params),
        target_region=out.target_region,
        rationale=out.rationale,
        autotune_exclude=list(out.autotune_exclude),
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

    # Optionality on params/target_region/rationale mirrors
    # ``OptimizationPlanOutput``'s Pydantic defaults — without it the
    # SDK rejects tool calls that omit any of these fields, even though
    # the Pydantic model itself would have filled the defaults.
    def submit_plan(
        tier: int,
        technique: str,
        params: dict[str, str] | None = None,
        target_region: str = "",
        rationale: str = "",
        autotune_exclude: list[dict[str, int]] | None = None,
    ) -> str:
        try:
            captured["output"] = OptimizationPlanOutput(
                tier=tier,
                technique=technique,
                params=params or {},
                target_region=target_region,
                rationale=rationale,
                autotune_exclude=autotune_exclude or [],
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

    The orchestrator provides all context in the prompt; the agent's
    only tool is ``submit_plan`` (no compile / correctness tools — those
    belong to the Coder).
    """

    def __init__(self, model: OpenAIChatCompletionsModel | None = None) -> None:
        self._model = model
        if model is not None and _SDK_AVAILABLE:
            self._instructions = (PROMPT_DIR / "system.md").read_text()
        else:
            self._instructions = ""

    @property
    def has_model(self) -> bool:
        """True when the agent is backed by a real LLM AND the SDK is
        importable. Both are required: ``Agent`` and ``function_tool``
        are ``None`` in SDK-absent test environments, so ``plan()`` would
        crash with ``TypeError`` if it took the LLM path with a model
        stub but no real SDK behind it.
        """
        return self._model is not None and _SDK_AVAILABLE

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
        sibling_context: str = "",
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

        if sibling_context:
            sections.append(
                "## Siblings already tried from this parent\n" + sibling_context
            )

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
        sibling_context: str = "",
        max_turns: int | None = None,
    ) -> OptimizationPlan:
        """Generate a structured optimization plan for the next iteration.

        Submits via a ``submit_plan`` tool call so the SDK never sends a
        ``response_format=json_schema`` (which DeepSeek-reasoner rejects and
        which the SDK's strict-schema validator rejects on
        ``params: dict[str, str]``). Pydantic validation still runs inside
        the tool body. Failure contract: a missing submission raises
        ``PlanningError``; if the turn budget is exhausted but a valid
        submission was already captured, that captured output is returned.
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
            sibling_context=sibling_context,
        )

        captured: dict = {}
        # ``strict_mode=False``: the SDK's strict-schema validator rejects
        # ``dict[str, str]`` (the ``params`` arg) with the same
        # ``additionalProperties`` error that originally killed the
        # ``output_type=Pydantic`` path. Pydantic validation still runs
        # inside the tool body, so end-to-end type safety is preserved;
        # malformed payloads bounce through the in-loop retry budget.
        submit_tool = function_tool(_make_submit_plan_tool(captured), strict_mode=False)
        agent = Agent(
            name="Planner",
            instructions=self._instructions,
            model=self._model,
            tools=[submit_tool],
        )

        # Turn budget: 2*N + 2 with N=1 in-band validation retry. Reserves
        # turns for: 1 invalid submit + 1 corrected submit + 1 confirmation,
        # plus the +1 buffer the SDK needs to land confirmation cleanly.
        # Without the retry budget, a single Pydantic validation slip
        # downgrades to MaxTurnsExceeded; the captured-output recovery
        # below would then have to catch it implicitly. cfg-tunable via
        # ACTSConfig.planner_max_turns (None preserves the default of 4).
        effective_max_turns = 4 if max_turns is None else max_turns
        try:
            result = await run_agent(
                agent,
                prompt,
                run_config=make_run_config(temperature=0.3),
                max_turns=effective_max_turns,
            )
        except MaxTurnsExceeded as exc:
            if "output" in captured:
                return _validate_and_convert(captured["output"], available_actions)
            raise PlanningError(
                f"Planner exhausted turn budget ({effective_max_turns}) "
                "without calling submit_plan."
            ) from exc

        if result is None:
            raise PlanningError("LLM call failed after all retries.")
        if "output" not in captured:
            raise PlanningError(
                "Planner did not call submit_plan before terminating — "
                "no final plan was emitted."
            )
        return _validate_and_convert(captured["output"], available_actions)
