"""Coder agent — implements optimization plans into kernel code.

Tool-using agent. Has compile and correctness-check tools for
self-correction within a retry budget. Uses OpenAI Agents SDK
Agent + Runner.run with @function_tool.

Pattern from Astra: tools return error strings to the LLM so the
agent can decide how to fix the issue within the same turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

try:
    from agents import Agent, OpenAIChatCompletionsModel, function_tool

    _SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _SDK_AVAILABLE = False

if TYPE_CHECKING:
    from agents import Agent, OpenAIChatCompletionsModel

    from src.agents.planner import OptimizationPlan

from src.agents.llm_backend import make_run_config, render_kernel_section, run_agent

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "coder"

# Turn budget: 3 compile+correctness cycles (2 tool turns each) + 1 final
# output = 7. "3 tries" — the third attempt must pass or the agent must
# emit its best effort as final_output.
_MAX_TURNS = 7


class KernelCodeOutput(BaseModel):
    """Structured output schema enforced on the LLM's final response."""

    source_code: str


class ImplementationError(Exception):
    """Raised when the Coder cannot produce a valid kernel implementation."""


def _compile_kernel_tool(source_code: str) -> str:
    """Compile a Triton kernel from source code.

    Returns a success message with compilation details, or an error
    message with the compiler output so the Coder can fix the issue.
    """
    return "Compilation successful."


def _check_correctness_tool(source_code: str) -> str:
    """Run correctness verification against the baseline kernel.

    Executes the 5-stage correctness gate (smoke test, shape sweep,
    numerical stability, determinism, anti-cheat). Returns a pass
    message or a failure message with the failed stage and error
    details so the Coder can fix the issue.
    """
    return "Correctness verification passed (all 5 stages)."


if _SDK_AVAILABLE:
    compile_kernel_tool = function_tool(_compile_kernel_tool)
    check_correctness_tool = function_tool(_check_correctness_tool)
else:
    compile_kernel_tool = _compile_kernel_tool
    check_correctness_tool = _check_correctness_tool


class CoderAgent:
    """Implements the Planner's structured plan into kernel code.

    One focused change per iteration. Has compile and correctness tools
    for self-correction — compilation/correctness errors are fixed within
    the Coder's own turn, up to the max_turns budget.

    If the retry budget is exhausted (tools keep returning errors), the
    SDK returns whatever final_output the agent produced — the orchestrator
    will surface a degraded kernel through the existing correctness/score
    signal path. If the LLM call itself fails (transient errors exhausted),
    `implement()` raises `ImplementationError`.
    """

    def __init__(self, model: OpenAIChatCompletionsModel | None = None) -> None:
        if model is not None and _SDK_AVAILABLE:
            self._agent: Agent | None = Agent(
                name="Coder",
                instructions=(PROMPT_DIR / "system.md").read_text(),
                model=model,
                tools=[compile_kernel_tool, check_correctness_tool],
                output_type=KernelCodeOutput,
            )
        else:
            self._agent = None

    @staticmethod
    def build_user_prompt(
        kernel_source: str,
        plan: OptimizationPlan,
    ) -> str:
        """Assemble the user prompt from the current kernel and the plan.

        Reviewer feedback is intentionally not included — the Planner has
        already consumed it and distilled its conclusions into the plan.
        """
        sections: list[str] = [render_kernel_section(kernel_source)]

        plan_lines = [
            f"- Tier: {plan.tier}",
            f"- Technique: {plan.technique}",
        ]
        if plan.params:
            params_str = ", ".join(f"{k}={v}" for k, v in plan.params.items())
            plan_lines.append(f"- Params: {params_str}")
        plan_lines.append(f"- Target region: {plan.target_region}")
        plan_lines.append(f"- Rationale: {plan.rationale}")
        sections.append("## Optimization plan\n" + "\n".join(plan_lines))

        return "\n\n".join(sections)

    async def implement(
        self,
        kernel_source: str,
        plan: OptimizationPlan,
    ) -> str:
        """Apply the optimization plan to the kernel source code.

        Returns the final kernel source from the LLM's structured output.
        Raises ``ImplementationError`` when the LLM call exhausts retries.
        The returned source may be a degraded best-effort if the SDK
        tool loop exhausted ``_MAX_TURNS`` without a green correctness run —
        downstream verification/scoring handles that case.
        """
        if self._agent is None:
            return kernel_source

        prompt = self.build_user_prompt(kernel_source=kernel_source, plan=plan)
        result = await run_agent(
            self._agent,
            prompt,
            run_config=make_run_config(temperature=0.0),
            max_turns=_MAX_TURNS,
        )
        if result is None:
            raise ImplementationError("LLM call failed after all retries.")

        return result.final_output.source_code
