"""Coder agent — implements optimization plans into kernel code.

Tool-using agent. Has compile and correctness-check tools for
self-correction within a retry budget. Uses OpenAI Agents SDK
Agent + Runner.run with @function_tool.

The two tools close over per-problem context (KernelSpec, reference_fn,
input_generator) captured at ``implement()`` call time. A fresh Agent
is built per call — cheap (object construction, no network) and keeps
the tool closures bound to the right oracle.

Error strings follow Astra's pattern: tools return failure messages
so the agent can self-correct within the same turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

try:
    from agents import Agent, function_tool
except ModuleNotFoundError:  # pragma: no cover
    Agent = None  # type: ignore[assignment]
    function_tool = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from agents import OpenAIChatCompletionsModel as _Model

    from src.agents.planner import OptimizationPlan

from src.agents.llm_backend import make_run_config, render_kernel_section, run_agent
from src.config import ACTSConfig
from src.eval.correctness import ComparisonPolicy, verify_correctness
from src.kernels.compiler import compile_kernel
from src.kernels.kernel import Kernel, KernelSpec

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "coder"


class KernelCodeOutput(BaseModel):
    """Structured output schema enforced on the LLM's final response."""

    source_code: str


class ImplementationError(Exception):
    """Raised when the Coder cannot produce a valid kernel implementation."""


# ── tool factories ──────────────────────────────────────────────────────
#
# Each factory returns a raw callable `(source_code: str) -> str`. The
# SDK's `function_tool` wrapper is applied at Agent-construction time
# inside `implement()` so the factories remain unit-testable without the
# SDK installed.


def _make_compile_tool(
    kernel_spec: KernelSpec,
    cache_dir: Path | None = None,
) -> Callable[[str], str]:
    """Build a compile tool bound to a specific KernelSpec.

    The tool wraps ``kernels.compiler.compile_kernel``. Success returns
    a short confirmation; failure returns the full compiler traceback so
    the Coder can read the error and fix it.
    """

    def compile_kernel_tool(source_code: str) -> str:
        kernel = Kernel(spec=kernel_spec, source_code=source_code)
        result = compile_kernel(kernel, cache_dir=cache_dir)
        if result.success:
            return (
                f"Compilation successful (entrypoint: '{kernel_spec.entrypoint}')."
            )
        return f"Compilation FAILED:\n{result.error_message}"

    return compile_kernel_tool


def _make_correctness_tool(
    kernel_spec: KernelSpec,
    reference_fn: Callable[..., Any],
    input_generator: Callable[[int], tuple],
    *,
    cache_dir: Path | None = None,
    policy: ComparisonPolicy | None = None,
) -> Callable[[str], str]:
    """Build a correctness tool bound to a KernelSpec + oracle + input generator.

    The tool recompiles the submitted source (compile is cheap; tools
    are independent), runs the 5-stage gate, and returns a human-readable
    pass/fail message. Compile failures are surfaced before attempting
    correctness so the Coder gets the cheaper error first.
    """

    def check_correctness_tool(source_code: str) -> str:
        kernel = Kernel(spec=kernel_spec, source_code=source_code)
        compiled = compile_kernel(kernel, cache_dir=cache_dir)
        if not compiled.success:
            return (
                "Correctness aborted — candidate failed to compile:\n"
                f"{compiled.error_message}"
            )
        result = verify_correctness(
            candidate_fn=compiled.compiled_fn,
            reference_fn=reference_fn,
            input_generator=input_generator,
            policy=policy,
        )
        if result.passed:
            return (
                f"Correctness verification passed (all 5 stages, "
                f"max_abs_error={result.max_abs_error:.3e})."
            )
        stage = result.failed_stage.value if result.failed_stage else "unknown"
        return (
            f"Correctness FAILED at stage [{stage}]:\n{result.error_message}"
        )

    return check_correctness_tool


class CoderAgent:
    """Implements the Planner's structured plan into kernel code.

    One focused change per iteration. Has compile and correctness tools
    for self-correction — compilation/correctness errors are fixed within
    the Coder's own turn, up to a config-derived turn budget.

    Turn budget: ``2 * config.max_debug_retries + 1`` — each retry cycle
    is one compile call + one correctness call, plus one final output
    turn. Default config gives 7 (= 2×3 + 1).

    If the retry budget is exhausted (tools keep returning errors), the
    SDK returns whatever final_output the agent produced — the orchestrator
    will surface a degraded kernel through the existing correctness/score
    signal path. If the LLM call itself fails (transient errors exhausted),
    ``implement()`` raises ``ImplementationError``.
    """

    def __init__(
        self,
        model: _Model | None = None,
        *,
        config: ACTSConfig | None = None,
    ) -> None:
        self._model = model
        cfg = config or ACTSConfig()
        self._max_turns = 2 * cfg.max_debug_retries + 1
        self._instructions = (
            (PROMPT_DIR / "system.md").read_text() if model is not None else ""
        )

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
        *,
        kernel_spec: KernelSpec | None = None,
        reference_fn: Callable[..., Any] | None = None,
        input_generator: Callable[[int], tuple] | None = None,
    ) -> str:
        """Apply the optimization plan to the kernel source code.

        Returns the final kernel source from the LLM's structured output.
        Raises ``ImplementationError`` when the LLM call exhausts retries
        or when the correctness context (``kernel_spec`` / ``reference_fn``
        / ``input_generator``) is missing while a model is configured.

        The returned source may be a degraded best-effort if the SDK
        tool loop exhausted the turn budget without a green correctness
        run — downstream verification/scoring handles that case.
        """
        if self._model is None:
            return kernel_source

        if kernel_spec is None or reference_fn is None or input_generator is None:
            raise ImplementationError(
                "LLM-driven Coder requires kernel_spec, reference_fn, and "
                "input_generator — its tools are bound to these at call time."
            )

        compile_tool = function_tool(_make_compile_tool(kernel_spec))
        correctness_tool = function_tool(
            _make_correctness_tool(
                kernel_spec,
                reference_fn=reference_fn,
                input_generator=input_generator,
            )
        )
        agent = Agent(
            name="Coder",
            instructions=self._instructions,
            model=self._model,
            tools=[compile_tool, correctness_tool],
            output_type=KernelCodeOutput,
        )

        prompt = self.build_user_prompt(kernel_source=kernel_source, plan=plan)
        result = await run_agent(
            agent,
            prompt,
            run_config=make_run_config(temperature=0.0),
            max_turns=self._max_turns,
        )
        if result is None:
            raise ImplementationError("LLM call failed after all retries.")

        return result.final_output.source_code
