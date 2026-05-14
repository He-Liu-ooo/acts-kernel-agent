"""Coder agent — implements optimization plans into kernel code.

Tool-using agent. Has compile, correctness-check, and submit tools for
self-correction and final emission within a retry budget. Uses OpenAI
Agents SDK Agent + Runner.run with @function_tool.

The compile + correctness tools close over per-problem context
(KernelSpec, reference_fn, input_generators) captured at ``implement()``
call time. The submit tool closes over a per-call captured dict so the
LLM's structured submission flows back to the caller. A fresh Agent is
built per call — cheap (object construction, no network) and keeps the
tool closures bound to the right oracle and the right capture slot.

Error strings follow Astra's pattern: tools return failure messages
so the agent can self-correct within the same turn. Submission goes
through a ``submit_kernel`` tool call rather than a Pydantic
``output_type=`` schema because reasoning-model providers
(DeepSeek-reasoner, etc.) reject the ``response_format=json_schema``
request the SDK derives from ``output_type=``; tool-call schemas are
universally supported. The Pydantic ``KernelCodeOutput`` validator
still runs — invoked by the submit tool body — preserving the
"validation failure → in-loop tool retry" guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ValidationError, model_validator

try:
    from agents import Agent, MaxTurnsExceeded, function_tool
except ModuleNotFoundError:  # pragma: no cover
    Agent = None  # type: ignore[assignment]
    function_tool = None  # type: ignore[assignment]

    class MaxTurnsExceeded(Exception):  # type: ignore[no-redef]
        """SDK-absent test stand-in. The real exception lives in ``agents``."""

if TYPE_CHECKING:
    from agents import OpenAIChatCompletionsModel as _Model

    from src.agents.planner import OptimizationPlan

from src.agents.llm_backend import (
    SUBMIT_OK_SENTINEL,
    format_submit_validation_error,
    make_run_config,
    render_kernel_section,
    run_agent,
)
from src.config import ACTSConfig
from src.eval.correctness import ComparisonPolicy, verify_correctness
from src.eval.profiler import triton_kernel_names_in
from src.kernels.compiler import compile_kernel
from src.kernels.kernel import Kernel, KernelSpec

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "coder"


class KernelCodeOutput(BaseModel):
    """Structured output schema validated on the submit_kernel tool payload.

    ``triton_kernel_name`` is the bare name of the ``@triton.jit`` device
    function the profiler should filter NCU on. Cross-validated against
    ``source_code``: must appear in the source as ``@triton.jit def
    <name>``. When the kernel is fused (multiple ``@triton.jit`` defs),
    the Coder names the one performing the dominant work — picking the
    wrong one silently mis-profiles the branch.
    """

    source_code: str
    triton_kernel_name: str
    # Destination-passing-style flag — see ``Kernel.dps``. The Coder sets
    # this to True when the emitted host wrapper takes pre-allocated output
    # buffers as positional args after the inputs; defaults to False so
    # legacy translate paths and tests that build outputs via ``return``
    # still validate. Heavy DPS-shape validation happens at first call
    # site (TypeError surfaces a mismatch); the schema only carries the
    # contract bit.
    dps: bool = False

    @model_validator(mode="after")
    def _triton_kernel_name_matches_source(self) -> "KernelCodeOutput":
        names = triton_kernel_names_in(self.source_code)
        if not names:
            raise ValueError(
                "source_code must define at least one ``@triton.jit def`` — "
                "the Coder writes Triton kernels, not bare PyTorch.",
            )
        if not self.triton_kernel_name:
            raise ValueError(
                "triton_kernel_name is required and must match one of the "
                f"@triton.jit defs in source: {names}",
            )
        if self.triton_kernel_name not in names:
            raise ValueError(
                f"triton_kernel_name={self.triton_kernel_name!r} not found in "
                f"source as ``@triton.jit def {self.triton_kernel_name}``. "
                f"Source defines: {names}",
            )
        return self

    @model_validator(mode="after")
    def _autotune_decorator_well_formed(self) -> "KernelCodeOutput":
        """Enforce the @triton.autotune contract on Coder-emitted source (A1).

        Rules:
          1. Source must contain a ``@triton.autotune`` decorator directly
             above the ``@triton.jit def`` matching ``triton_kernel_name``.
          2. The decorator's ``configs=[...]`` must have >=4 entries, each
             a ``triton.Config(...)`` call.
          3. The decorator's ``key=[...]`` must be present and non-empty.

        Stdlib ``ast`` only — no Triton import. The Coder-side tool loop
        catches the ValidationError and retries within the existing turn
        budget; the LLM sees the message and produces a corrected source.
        """
        import ast as _ast

        try:
            tree = _ast.parse(self.source_code)
        except SyntaxError as exc:
            raise ValueError(f"source_code does not parse: {exc}") from exc

        target_fn: _ast.FunctionDef | None = None
        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name == self.triton_kernel_name:
                target_fn = node
                break
        if target_fn is None:
            # Already caught by _triton_kernel_name_matches_source; defensive.
            return self

        autotune_dec: _ast.Call | None = None
        for dec in target_fn.decorator_list:
            if not isinstance(dec, _ast.Call):
                continue
            attr = ""
            if isinstance(dec.func, _ast.Attribute):
                attr = dec.func.attr
            elif isinstance(dec.func, _ast.Name):
                attr = dec.func.id
            if attr == "autotune":
                autotune_dec = dec
                break

        if autotune_dec is None:
            raise ValueError(
                "@triton.autotune decorator required directly above "
                f"@triton.jit def {self.triton_kernel_name}. Every "
                "Coder-emitted kernel must autotune; see "
                "prompts/coder/system.md."
            )

        configs_list: _ast.List | None = None
        key_list: _ast.List | None = None
        for kw in autotune_dec.keywords:
            if kw.arg == "configs" and isinstance(kw.value, _ast.List):
                configs_list = kw.value
            elif kw.arg == "key" and isinstance(kw.value, _ast.List):
                key_list = kw.value

        if configs_list is None:
            raise ValueError(
                "@triton.autotune must pass configs= as a list literal of "
                "triton.Config(...) calls."
            )

        n_configs = sum(
            1 for c in configs_list.elts
            if isinstance(c, _ast.Call) and (
                (isinstance(c.func, _ast.Attribute) and c.func.attr == "Config")
                or (isinstance(c.func, _ast.Name) and c.func.id == "Config")
            )
        )
        if n_configs < 4:
            raise ValueError(
                f"@triton.autotune.configs must have at least 4 triton.Config "
                f"entries; got {n_configs}. Closing the parameter-axis gap to "
                "the autotuned Triton baseline requires a real sweep."
            )

        if key_list is None or len(key_list.elts) == 0:
            raise ValueError(
                "@triton.autotune must pass a non-empty key=[...] list of "
                "shape-arg names (e.g. key=['M','N','K']). Empty key= means "
                "Triton autotunes once and reuses across every shape — the "
                "bug we started with."
            )

        return self


@dataclass(frozen=True)
class AttemptFailure:
    """One failed baseline-generation attempt's tool-error trace.

    Captured by ``baseline_generator.generate_triton_baseline`` after each
    ``ImplementationError`` (and after post-translate compile / correctness
    failures) and threaded into the next ``translate()`` call so the Coder's
    user prompt can surface what didn't work in prior attempts.

    ``tool_errors`` is chronological — the order the SDK loop fired the
    FAILED tool returns during the attempt. Empty list when the attempt
    terminated without invoking any tool (reasoning-content truncation
    pathology). See ``doc/specs/2026-05-13-cross-attempt-memory-design.md``.
    """

    attempt_no: int  # 1-indexed, matches emit() event payloads
    tool_errors: list[str] = field(default_factory=list)


class ImplementationError(Exception):
    """Raised when the Coder cannot produce a valid kernel implementation.

    ``tool_errors`` carries the chronological list of FAILED strings the
    compile / correctness tools returned during the SDK loop. Empty when
    no tool calls happened (reasoning-content truncation). Carried out so
    ``baseline_generator`` can thread it into the next attempt's prompt
    via ``AttemptFailure``.
    """

    def __init__(self, message: str, *, tool_errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.tool_errors = tool_errors or []


# ── tool factories ──────────────────────────────────────────────────────
#
# Each factory returns a raw callable `(source_code: str) -> str`. The
# SDK's `function_tool` wrapper is applied at Agent-construction time
# inside `implement()` so the factories remain unit-testable without the
# SDK installed.


def _record_failure(error_log: list[str] | None, msg: str) -> str:
    """Append *msg* to *error_log* (when supplied) and return it.

    Shared by every tool factory's FAILED return branch so the
    "if error_log is not None: error_log.append(msg)" pattern lives in
    one place. ``_run_tool_agent`` binds the same ``tool_errors`` list
    to every factory, so each tool's failure rides out via
    ``ImplementationError.tool_errors`` for cross-attempt memory.
    """
    if error_log is not None:
        error_log.append(msg)
    return msg


def _make_compile_tool(
    kernel_spec: KernelSpec,
    cache_dir: Path | None = None,
    *,
    error_log: list[str] | None = None,
) -> Callable[[str], str]:
    """Build a compile tool bound to a specific KernelSpec.

    The tool wraps ``kernels.compiler.compile_kernel``. Success returns
    a short confirmation; failure returns the full compiler traceback so
    the Coder can read the error and fix it.

    When *error_log* is supplied, every FAILED return string is appended
    to it in-place. ``_run_tool_agent`` uses this to capture cross-turn
    errors so they can ride out as ``ImplementationError.tool_errors``
    for cross-attempt memory. Success returns are not logged — they
    are not failures to remember.
    """

    def compile_kernel_tool(source_code: str) -> str:
        kernel = Kernel(spec=kernel_spec, source_code=source_code)
        result = compile_kernel(kernel, cache_dir=cache_dir)
        if result.success:
            return (
                f"Compilation successful (entrypoint: '{kernel_spec.entrypoint}')."
            )
        return _record_failure(
            error_log,
            f"Compilation FAILED:\n{result.error_message}",
        )

    return compile_kernel_tool


def _make_correctness_tool(
    kernel_spec: KernelSpec,
    reference_fn: Callable[..., Any],
    input_generators: list[Callable[[int], tuple]],
    *,
    cache_dir: Path | None = None,
    policy: ComparisonPolicy | None = None,
    definition: Any | None = None,
    workloads: list[Any] | None = None,
    error_log: list[str] | None = None,
) -> Callable[..., str]:
    """Build a correctness tool bound to a KernelSpec + oracle + workload generators.

    The tool recompiles the submitted source (compile is cheap; tools
    are independent), runs the 5-stage gate against *every* generator in
    order, and returns a human-readable pass/fail message. Short-circuits
    on the first failing workload so the Coder sees exactly which one
    broke — so retries can actually correct cross-workload bugs instead
    of reproducing the same kernel when only the primary workload was
    exercised. Compile failures are surfaced before attempting
    correctness so the Coder gets the cheaper error first.

    *workloads* runs parallel to *input_generators*; when supplied with
    ``dps=True`` from the LLM tool call, each workload's resolved axes
    feed ``allocate_outputs`` so DPS host wrappers
    (``def kernel_fn(x, out_a, out_b)``) can be checked against the
    PyTorch oracle that returns its outputs by value. A length mismatch
    is a contract bug at the factory level (raised eagerly).

    When *error_log* is supplied, every FAILED / aborted return string is
    appended to it in-place. ``_run_tool_agent`` uses this to capture
    cross-turn errors so they can ride out as
    ``ImplementationError.tool_errors`` for cross-attempt memory.
    Success returns are not logged.
    """
    if not input_generators:
        raise ValueError(
            "correctness tool requires at least one input generator — "
            "got an empty list.",
        )
    if workloads is not None and len(workloads) != len(input_generators):
        raise ValueError(
            f"workloads ({len(workloads)}) and input_generators "
            f"({len(input_generators)}) must be the same length so each "
            f"DPS allocate_outputs gets the right resolved axes."
        )

    def check_correctness_tool(source_code: str, dps: bool = False) -> str:
        kernel = Kernel(spec=kernel_spec, source_code=source_code, dps=dps)
        compiled = compile_kernel(kernel, cache_dir=cache_dir)
        if not compiled.success:
            return _record_failure(
                error_log,
                "Correctness aborted — candidate failed to compile:\n"
                f"{compiled.error_message}",
            )
        total = len(input_generators)
        max_err = 0.0
        for idx, gen in enumerate(input_generators):
            wl = workloads[idx] if workloads is not None else None
            result = verify_correctness(
                candidate_fn=compiled.compiled_fn,
                reference_fn=reference_fn,
                input_generator=gen,
                definition=definition,
                kernel=kernel,
                workload=wl,
                policy=policy,
            )
            if not result.passed:
                stage = result.failed_stage.value if result.failed_stage else "unknown"
                return _record_failure(
                    error_log,
                    f"Correctness FAILED on workload {idx + 1}/{total} "
                    f"at stage [{stage}]:\n{result.error_message}",
                )
            max_err = max(max_err, result.max_abs_error)
        return (
            f"Correctness verification passed on all {total} workloads "
            f"(5 stages each, max_abs_error={max_err:.3e})."
        )

    return check_correctness_tool


def _make_submit_tool(
    captured: dict,
    *,
    error_log: list[str] | None = None,
) -> Callable[[str, str], str]:
    """Build a submit tool that captures the LLM's final ``KernelCodeOutput``.

    The tool runs the ``KernelCodeOutput`` Pydantic validator inside the
    tool body. On success it stores the validated output in
    ``captured["output"]`` and returns a sentinel string instructing the
    LLM to emit a one-word confirmation so the SDK tool loop terminates.
    On validation failure it returns the error string — the SDK will hand
    that back to the LLM as the tool-call response, prompting an in-loop
    retry within the existing turn budget.

    *captured* is a dict (not a single-element list / nullable variable)
    because tests construct one per call and assert via ``"output" in captured``.

    *error_log* mirrors the compile / correctness tools: when supplied,
    every validation-failure return string is appended to it in-place so
    cross-attempt memory captures the actual reason an attempt failed
    when the failure happens at submit time. Without this, the
    prior-failures section would render the misleading
    "no tool errors recorded" placeholder for an attempt that DID
    invoke ``submit_kernel`` but had its payload rejected.
    """

    def submit_kernel(
        source_code: str,
        triton_kernel_name: str,
        dps: bool = False,
    ) -> str:
        try:
            captured["output"] = KernelCodeOutput(
                source_code=source_code,
                triton_kernel_name=triton_kernel_name,
                dps=dps,
            )
        except ValidationError as exc:
            return _record_failure(
                error_log,
                format_submit_validation_error("submit_kernel", exc),
            )
        return SUBMIT_OK_SENTINEL

    return submit_kernel


class CoderAgent:
    """Implements the Planner's structured plan into kernel code.

    One focused change per iteration. Has compile and correctness tools
    for self-correction — compilation/correctness errors are fixed within
    the Coder's own turn, up to a config-derived turn budget.

    Turn budget: ``2 * config.max_debug_retries + 2`` — each retry cycle
    is one compile call + one correctness call, plus one ``submit_kernel``
    tool call plus one final plain-text confirmation. Default config gives
    8 (= 2×3 + 2).

    If the turn budget is exhausted, ``implement()`` raises
    ``ImplementationError`` unless a Pydantic-valid ``submit_kernel``
    payload was already captured before the budget ran out — in which
    case that captured output is returned. If the LLM call itself fails
    (transient errors exhausted), ``implement()`` also raises
    ``ImplementationError``.
    """

    def __init__(
        self,
        model: _Model | None = None,
        *,
        config: ACTSConfig | None = None,
    ) -> None:
        self._model = model
        cfg = config or ACTSConfig()
        self._max_turns = 2 * cfg.max_debug_retries + 2
        if model is not None:
            self._instructions = (PROMPT_DIR / "system.md").read_text()
            self._translate_instructions = (PROMPT_DIR / "translate.md").read_text()
        else:
            self._instructions = ""
            self._translate_instructions = ""

    @property
    def has_model(self) -> bool:
        """True when the agent is backed by a real LLM."""
        return self._model is not None

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

    async def _run_tool_agent(
        self,
        *,
        agent_name: str,
        instructions: str,
        prompt: str,
        kernel_spec: KernelSpec,
        reference_fn: Callable[..., Any],
        input_generators: list[Callable[[int], tuple]],
        definition: Any | None = None,
        workloads: list[Any] | None = None,
    ) -> KernelCodeOutput:
        # Shared across all three tool factories so every FAILED return
        # rides out via ``ImplementationError.tool_errors`` for the
        # baseline_generator's cross-attempt memory.
        tool_errors: list[str] = []
        compile_tool = function_tool(
            _make_compile_tool(kernel_spec, error_log=tool_errors)
        )
        correctness_tool = function_tool(
            _make_correctness_tool(
                kernel_spec,
                reference_fn=reference_fn,
                input_generators=input_generators,
                definition=definition,
                workloads=workloads,
                error_log=tool_errors,
            )
        )
        captured: dict = {}
        submit_tool = function_tool(
            _make_submit_tool(captured, error_log=tool_errors)
        )
        agent = Agent(
            name=agent_name,
            instructions=instructions,
            model=self._model,
            tools=[compile_tool, correctness_tool, submit_tool],
        )
        # SDK ``MaxTurnsExceeded`` is converted to ``ImplementationError`` so
        # callers (orchestrator iteration loop, baseline_generator retry loop)
        # have a single typed failure to catch. If the LLM submitted a valid
        # kernel before burning the budget, return that — the run merely went
        # over budget after the answer was already in hand.
        try:
            result = await run_agent(
                agent,
                prompt,
                run_config=make_run_config(temperature=0.0),
                max_turns=self._max_turns,
            )
        except MaxTurnsExceeded as exc:
            if "output" in captured:
                return captured["output"]
            raise ImplementationError(
                f"Coder exhausted turn budget ({self._max_turns}) without "
                "calling submit_kernel.",
                tool_errors=tool_errors,
            ) from exc
        if result is None:
            raise ImplementationError(
                "LLM call failed after all retries.",
                tool_errors=tool_errors,
            )
        if "output" not in captured:
            raise ImplementationError(
                "Coder did not call submit_kernel before terminating — "
                "no final kernel was emitted.",
                tool_errors=tool_errors,
            )
        return captured["output"]

    async def implement(
        self,
        kernel_source: str,
        plan: OptimizationPlan,
        *,
        kernel_spec: KernelSpec | None = None,
        reference_fn: Callable[..., Any] | None = None,
        input_generators: list[Callable[[int], tuple]] | None = None,
        definition: Any | None = None,
        workloads: list[Any] | None = None,
    ) -> KernelCodeOutput:
        """Apply the optimization plan to the kernel source code.

        Returns the structured Coder output (``source_code`` plus a
        Pydantic-validated ``triton_kernel_name``) so the caller can
        thread the declared kernel symbol into a fresh ``Kernel`` for
        downstream profiling. If the turn budget is exhausted, the
        captured ``submit_kernel`` payload is returned when one exists
        (so a valid late submission isn't discarded); otherwise
        ``ImplementationError`` is raised. In the no-model placeholder
        path the method returns a stub ``KernelCodeOutput`` with the
        unchanged source and an empty kernel-name (validation skipped
        via ``model_construct``); the orchestrator's profiler-resolution
        chain falls back to source-regex extraction in that case.
        Raises ``ImplementationError`` when the LLM call exhausts retries
        or when the correctness context is missing while a model is configured.
        """
        if self._model is None:
            return KernelCodeOutput.model_construct(
                source_code=kernel_source,
                triton_kernel_name="",
                dps=False,
            )

        if kernel_spec is None or reference_fn is None or not input_generators:
            raise ImplementationError(
                "LLM-driven Coder requires kernel_spec, reference_fn, and a "
                "non-empty input_generators list — its tools are bound to "
                "these at call time."
            )

        return await self._run_tool_agent(
            agent_name="Coder",
            instructions=self._instructions,
            prompt=self.build_user_prompt(kernel_source=kernel_source, plan=plan),
            kernel_spec=kernel_spec,
            reference_fn=reference_fn,
            input_generators=input_generators,
            definition=definition,
            workloads=workloads,
        )

    @staticmethod
    def build_translate_prompt(
        reference_source: str,
        kernel_spec: KernelSpec,
        *,
        prior_failures: Sequence[AttemptFailure] = (),
    ) -> str:
        """Assemble the user prompt for a one-shot PyTorch→Triton port.

        When *prior_failures* is non-empty, prepends a "## Prior attempt
        failures" section listing each attempt's tool errors so the model
        can avoid repeating the same kernel + same error across
        ``Runner.run`` boundaries. Threaded by
        ``baseline_generator.generate_triton_baseline`` after each
        ``ImplementationError`` catch. Empty default = no section rendered
        (backward-compatible for callers that don't supply it). See
        ``doc/specs/2026-05-13-cross-attempt-memory-design.md``.
        """
        safe_reference = reference_source.replace("```", r"\`\`\`")
        sections: list[str] = []

        if prior_failures:
            intro = (
                "Previous baseline-generation attempts for this same PyTorch "
                "reference hit the following tool errors. Each \"Attempt N\" "
                "block lists the errors in the order they fired during that "
                "attempt. The same error class recurring across attempts "
                "indicates a persistent issue — try a structurally different "
                "solution rather than re-applying the same approach."
            )
            attempt_blocks: list[str] = []
            for af in prior_failures:
                if af.tool_errors:
                    bullets = "\n".join(f"- {e}" for e in af.tool_errors)
                else:
                    bullets = (
                        "- (no tool errors recorded — agent terminated without "
                        "invoking compile / correctness / submit; likely a "
                        "reasoning-content budget issue)"
                    )
                attempt_blocks.append(f"### Attempt {af.attempt_no}\n{bullets}")
            sections.append(
                "## Prior attempt failures\n\n"
                + intro
                + "\n\n"
                + "\n\n".join(attempt_blocks)
            )

        sections.append("## PyTorch reference\n```python\n" + safe_reference + "\n```")
        sections.append(
            "## Target kernel\n"
            f"- Name: {kernel_spec.name}\n"
            f"- Entrypoint: {kernel_spec.entrypoint}\n"
            f"- Kernel type: {kernel_spec.kernel_type.value}"
        )
        return "\n\n".join(sections)

    async def translate(
        self,
        *,
        reference_source: str,
        kernel_spec: KernelSpec,
        reference_fn: Callable[..., Any],
        input_generators: list[Callable[[int], tuple]],
        definition: Any | None = None,
        workloads: list[Any] | None = None,
        prior_failures: Sequence[AttemptFailure] = (),
    ) -> KernelCodeOutput:
        """Port a PyTorch reference into a Triton kernel in one agent run.

        Used at problem-load time by ``benchmark.baseline_generator``.
        Returns the structured Coder output so the baseline ``Kernel``
        carries the Pydantic-validated ``triton_kernel_name`` from the
        moment it enters the search tree. Callers post-verify after
        translation because the in-loop correctness tool only runs on
        whatever workloads the Coder chose to call it with; a final
        oracle check on the submitted kernel is still the caller's
        responsibility. Raises ``ImplementationError`` when no model is
        configured, when the LLM call exhausts its retries, or when the
        turn budget is exhausted with no captured submission.

        *prior_failures* — when non-empty, ``build_translate_prompt``
        prepends a "## Prior attempt failures" section so the model can
        see what didn't work in earlier baseline-generation attempts for
        this same reference. Threaded by
        ``baseline_generator.generate_triton_baseline`` after each
        ``ImplementationError`` catch. Default empty tuple = no section
        rendered.
        """
        if self._model is None:
            raise ImplementationError(
                "translate() requires a configured model — there is no "
                "sensible no-op fallback for a from-scratch port."
            )

        return await self._run_tool_agent(
            agent_name="Coder-Translator",
            instructions=self._translate_instructions,
            prompt=self.build_translate_prompt(
                reference_source=reference_source,
                kernel_spec=kernel_spec,
                prior_failures=prior_failures,
            ),
            kernel_spec=kernel_spec,
            reference_fn=reference_fn,
            input_generators=input_generators,
            definition=definition,
            workloads=workloads,
        )
