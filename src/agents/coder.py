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

try:
    from agents import Agent, OpenAIChatCompletionsModel, function_tool

    _SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _SDK_AVAILABLE = False

if TYPE_CHECKING:
    from agents import Agent, OpenAIChatCompletionsModel

    from src.agents.planner import OptimizationPlan
    from src.kernels.kernel import Kernel

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "coder"


# ---------------------------------------------------------------------------
# Tools — SDK auto-generates JSON schemas from type hints + docstrings.
# Tools return strings: success messages or error details for self-correction.
# When SDK is not installed, tools are plain functions (not decorated).
# ---------------------------------------------------------------------------

def _compile_kernel_tool(source_code: str) -> str:
    """Compile a Triton kernel from source code.

    Returns a success message with compilation details, or an error
    message with the compiler output so the Coder can fix the issue.
    """
    # Placeholder: always succeeds.
    return "Compilation successful."


def _check_correctness_tool(source_code: str) -> str:
    """Run correctness verification against the baseline kernel.

    Executes the 5-stage correctness gate (smoke test, shape sweep,
    numerical stability, determinism, anti-cheat). Returns a pass
    message or a failure message with the failed stage and error
    details so the Coder can fix the issue.
    """
    # Placeholder: always passes.
    return "Correctness verification passed (all 5 stages)."


# Apply SDK decorator when available
if _SDK_AVAILABLE:
    compile_kernel_tool = function_tool(_compile_kernel_tool)
    check_correctness_tool = function_tool(_check_correctness_tool)
else:
    compile_kernel_tool = _compile_kernel_tool
    check_correctness_tool = _check_correctness_tool


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CoderAgent:
    """Implements the Planner's structured plan into kernel code.

    One focused change per iteration. Has compile and correctness tools
    for self-correction — compilation/correctness errors are fixed within
    the Coder's own turn, up to max_debug_retries attempts.

    If the retry budget is exhausted (tools keep returning errors), the
    SDK returns the final_output and the orchestrator marks the branch dead.
    """

    def __init__(self, model: OpenAIChatCompletionsModel | None = None) -> None:
        if model is not None and _SDK_AVAILABLE:
            self._agent = Agent(
                name="Coder",
                instructions=(PROMPT_DIR / "system.md").read_text(),
                model=model,
                tools=[compile_kernel_tool, check_correctness_tool],
            )
        else:
            self._agent = None

    async def implement(
        self,
        kernel_source: str,
        plan: OptimizationPlan,
    ) -> str:
        """Apply the optimization plan to the kernel source code.

        The Coder receives the current kernel source and the structured
        plan, generates optimized code, then uses compile_kernel_tool
        and check_correctness_tool to verify. On failure, the SDK's
        tool loop lets the Coder self-correct.

        Returns the final kernel source code string (from final_output).
        """
        # Placeholder: return source unchanged without calling the LLM.
        return kernel_source
