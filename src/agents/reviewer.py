"""Reviewer agent — interprets eval results into structured feedback.

Uses the OpenAI Agents SDK with a ``submit_review`` tool: the LLM emits
its structured payload through that tool, and Pydantic validation runs
inside the tool body. An optional ``query_metric`` tool (gated by
``ACTSConfig.reviewer_metric_queries``) lets the LLM pull specific raw
NCU metrics for the current iteration before submitting. Falls back to
rule-based feedback when the LLM is unavailable or its call fails after
all retries.

Designed for future split into Compute-Reviewer and Memory-Reviewer sub-agents:
`prompt_dir` is a constructor parameter, so a specialized instance is one call
away without subclassing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

try:
    from agents import (
        Agent,
        MaxTurnsExceeded,
        ModelBehaviorError,
        OpenAIChatCompletionsModel,
        function_tool,
    )

    _SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    Agent = None  # type: ignore[assignment]
    function_tool = None  # type: ignore[assignment]

    class MaxTurnsExceeded(Exception):  # type: ignore[no-redef]
        """SDK-absent test stand-in. The real exception lives in ``agents``."""

    class ModelBehaviorError(Exception):  # type: ignore[no-redef]
        """SDK-absent test stand-in. The real exception lives in ``agents``."""

    _SDK_AVAILABLE = False

if TYPE_CHECKING:
    from agents import Agent, OpenAIChatCompletionsModel

from src.agents.llm_backend import (
    SUBMIT_OK_SENTINEL,
    format_submit_validation_error,
    make_run_config,
    render_kernel_section,
    render_run_context,
    run_agent,
)
# Re-exported under an alias so unit tests can patch the emit call site
# inside this module without monkey-patching the runtime package.
from src.runtime.events import emit as events_emit

if TYPE_CHECKING:
    from src.eval.profiler import ProfilingResult
    from src.eval.roofline import RooflineResult
    from src.eval.types import BottleneckType

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


# ── Pydantic output model (validated on submit_review payload) ─────────


class ReviewerFeedbackOutput(BaseModel):
    """Structured output schema validated on the submit_review tool payload."""

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


def _render_review_for_planner(fb: "ReviewerFeedback | None") -> str | None:
    """Render the curated Reviewer subset for the next Planner call.

    Returns None when fb is None so the Planner's prompt-section guard
    (``if reviewer_feedback:``) omits the section entirely.

    Renders four high-signal fields: outcome, bottleneck_diagnosis,
    suggestions, conditional_assessment. Skipped: metric_deltas (Planner
    has profiling), bottleneck_classification (Planner has run_bottleneck),
    branch_quality (search-engine concern).
    See ``doc/specs/2026-05-10-planner-receives-reviewer-feedback-design.md``.
    """
    if fb is None:
        return None
    lines = [f"Outcome: {fb.outcome}"]
    if fb.bottleneck_diagnosis:
        lines.append(f"Diagnosis: {fb.bottleneck_diagnosis}")
    if fb.suggestions:
        lines.append("Suggestions: " + "; ".join(fb.suggestions))
    if fb.conditional_assessment:
        lines.append(f"What would unlock progress: {fb.conditional_assessment}")
    return "\n".join(lines)


def render_profiling_summary(
    profiling: "ProfilingResult | None",
    roofline: "RooflineResult | None" = None,
) -> str:
    """Render a ``ProfilingResult`` into the two-block text the Reviewer
    prompt expects. Used both by ``ReviewerAgent.build_user_prompt`` and
    by ``pipeline.report.render_report`` so the format is consistent
    across the reviewer and the final operator-facing summary.

    ``arithmetic_intensity`` and ``ridge_point`` are sourced from
    ``roofline`` (run-level invariants); ``achieved_*`` and ``pct_peak_*``
    come from ``profiling.analytical`` (per-iter dynamics). When
    ``roofline`` is ``None`` the AI / ridge_point lines are omitted —
    callers that have a roofline should pass it.

    When ``profiling`` is ``None``, returns a one-line notice so the
    Reviewer knows it's operating without profile data (rather than
    silently reusing stale context).
    """
    if profiling is None:
        return "[no profiling data — profile_kernel did not run]"

    lines: list[str] = []
    if profiling.has_analytical or roofline is not None:
        lines.append("### Analytical (roofline)")
        if roofline is not None:
            lines.extend([
                f"- arithmetic_intensity: {roofline.arithmetic_intensity:.3f} MACs/byte",
                f"- ridge_point: {roofline.ridge_point:.3f} MACs/byte",
            ])
        if profiling.has_analytical:
            a = profiling.analytical
            lines.extend([
                f"- achieved: {a.achieved_tflops:.2f} TFLOPS / {a.achieved_bandwidth_gb_s:.2f} GB/s",
                f"- pct_peak: compute {a.pct_peak_compute * 100:.1f}% · bw {a.pct_peak_bandwidth * 100:.1f}%",
            ])
        else:
            lines.append("- achieved / pct_peak: [unavailable — no byte count]")

    ncu = profiling.ncu
    if ncu is not None:
        tc = (
            "n/a" if ncu.tensor_core_util_pct is None
            else f"{ncu.tensor_core_util_pct:.1f}%"
        )
        lines.extend([
            "",
            "### NCU (curated)",
            f"- sm_occupancy: {ncu.sm_occupancy_pct:.1f}%",
            f"- l2_hit_rate: {ncu.l2_hit_rate_pct:.1f}%",
            f"- tensor_core_util: {tc}",
            (
                f"- top stalls: {ncu.warp_stall_dominant} "
                f"({ncu.warp_stall_dominant_pct:.1f}%), "
                f"{ncu.warp_stall_runner_up} "
                f"({ncu.warp_stall_runner_up_pct:.1f}%)"
            ),
        ])
    elif profiling.degraded:
        # Surface the failure reason so the reviewer doesn't infer "NCU said
        # there's no issue" from silence. Matches the DEGRADED notice in
        # pipeline/report.py::render_report.
        lines.extend([
            "",
            f"[DEGRADED: NCU unavailable — reason={profiling.degraded_reason or 'unknown'}]",
        ])

    return "\n".join(lines)


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
    bottleneck: BottleneckType,
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
        bottleneck_classification=bottleneck.value,
        bottleneck_diagnosis=diagnosis,
        branch_quality=branch,
        degraded=degraded,
        error_reason=error_reason,
    )


def _make_submit_review_tool(captured: dict) -> Callable[..., str]:
    """Build a submit tool that captures the LLM's final ``ReviewerFeedbackOutput``.

    Mirrors ``coder._make_submit_tool`` and ``planner._make_submit_plan_tool``:
    runs Pydantic validation in the tool body, stores the validated output
    on success, returns the standard error string on failure (which the
    SDK hands back to the LLM for in-loop retry).

    Tool parameter types are plain ``str`` for ``bottleneck_classification``
    and ``branch_quality`` rather than the enums, because the SDK's
    ``function_tool`` derives the JSON schema from Python annotations and
    enum support is shaky across providers. The Pydantic model accepts
    the string and validates it against the enum internally.
    """

    # Required vs optional mirrors ``ReviewerFeedbackOutput``'s Pydantic
    # defaults — required: outcome, bottleneck_classification, branch_quality.
    # Without optional defaults here the SDK rejects tool calls that omit
    # any of these fields, even though the Pydantic model itself would
    # have filled the defaults.
    def submit_review(
        outcome: str,
        bottleneck_classification: str,
        branch_quality: str,
        metric_deltas: dict[str, float] | None = None,
        bottleneck_diagnosis: str = "",
        suggestions: list[str] | None = None,
        conditional_assessment: str = "",
    ) -> str:
        try:
            captured["output"] = ReviewerFeedbackOutput(
                outcome=outcome,
                metric_deltas=metric_deltas or {},
                bottleneck_classification=bottleneck_classification,
                bottleneck_diagnosis=bottleneck_diagnosis,
                suggestions=suggestions or [],
                branch_quality=branch_quality,
                conditional_assessment=conditional_assessment,
            )
        except ValidationError as exc:
            return format_submit_validation_error("submit_review", exc)
        return SUBMIT_OK_SENTINEL

    return submit_review


# ── Multi-turn fetch tool (gated by ACTSConfig.reviewer_metric_queries) ─


def _make_query_metric_tool(
    raw_metrics: dict[str, float] | None,
    iter_idx: int,
    metric_groups: dict[str, dict[str, dict[str, str | float]]] | None = None,
) -> Callable[[list[str]], dict[str, str]]:
    """Closure-captures this iteration's raw_metrics + iter index.

    Output is a string-valued dict so float values and the
    ``"[unknown]"`` / ``"[no data]"`` sentinels coexist without type
    confusion in the LLM's view. ``raw_metrics is None`` and
    ``raw_metrics == {}`` are treated identically (both signal an
    absent NCU dump for this iteration).

    The tool registers with ``strict_mode=False``, so the SDK does
    NOT pre-validate ``names``; this body does that itself, returning
    a recoverable ``{"_error": ...}`` dict on shape drift instead of
    iterating a bare string char-by-char and emitting garbage events.
    """

    def query_metric(names: list[str]) -> dict[str, str]:
        if not isinstance(names, list):
            return {
                "_error": (
                    f"`names` must be a list[str]; got "
                    f"{type(names).__name__}. Retry with a list of "
                    f"metric names from the menu."
                )
            }
        coerced = [str(n) for n in names]
        events_emit(
            "reviewer_metric_query",
            iter=iter_idx,
            count=len(coerced),
            names=coerced[:8],
        )
        out: dict[str, str] = {}
        groups = metric_groups or {}
        for name in coerced:
            group_name = name.removeprefix("group:")
            if name.startswith("group:") or (
                name in groups and name not in (raw_metrics or {})
            ):
                group = groups.get(group_name)
                if group is None:
                    out[name] = "[unknown group]"
                    continue
                for metric_name, status in group.items():
                    state = str(status.get("status", "missing"))
                    if state == "present" and "value" in status:
                        out[f"group:{group_name}.{metric_name}"] = (
                            f"present: {status['value']}"
                        )
                    else:
                        out[f"group:{group_name}.{metric_name}"] = state
                continue
            if not raw_metrics:
                out[name] = "[no data]"
            else:
                out[name] = (
                    f"{raw_metrics[name]}" if name in raw_metrics else "[unknown]"
                )
        return out

    return query_metric


# ── Agent ──────────────────────────────────────────────────────────────


class ReviewerAgent:
    """Interprets evaluation results and produces structured feedback.

    Acts as intelligent filter between raw profiling data and the Planner.
    The default path is submit-only: the agent is given a ``submit_review``
    tool and receives all eval data in the prompt. When
    ``ACTSConfig.reviewer_metric_queries`` is enabled, it additionally gets
    a ``query_metric`` tool to pull specific raw NCU metrics for the
    current iteration before submitting.

    `prompt_dir` is configurable so future Compute-Reviewer / Memory-Reviewer
    sub-agents can load specialized system prompts without subclassing.
    """

    def __init__(
        self,
        model: OpenAIChatCompletionsModel | None = None,
        prompt_dir: Path = PROMPT_DIR,
    ) -> None:
        self._model = model
        self._prompt_dir = prompt_dir
        if model is not None and _SDK_AVAILABLE:
            # ``system.md`` is the always-on submit-only contract — it must
            # NOT mention ``query_metric``, otherwise the LLM rationally
            # tries to call a tool that's unregistered (the live-run
            # ``ModelBehaviorError: Tool query_metric not found`` regression).
            # ``system_metric_queries.md`` is the optional addendum
            # appended in ``review()`` only when the matching tool is
            # registered, keeping prompt mention symmetric with tool
            # registration.
            self._instructions_base = (prompt_dir / "system.md").read_text()
            self._instructions_metric_queries = (
                prompt_dir / "system_metric_queries.md"
            ).read_text()
            # Back-compat: callers / tests that read ``_instructions``
            # directly see the submit-only path (the multi-turn flag is
            # off by default).
            self._instructions = self._instructions_base
        else:
            self._instructions_base = ""
            self._instructions_metric_queries = ""
            self._instructions = ""

    @property
    def has_model(self) -> bool:
        """True when the agent is backed by a real LLM AND the SDK is
        importable. See planner.PlannerAgent.has_model for rationale.
        """
        return self._model is not None and _SDK_AVAILABLE

    # ── prompt assembly ─────────────────────────────────────────────

    @staticmethod
    def build_user_prompt(
        kernel_source: str,
        profiling_summary: str,
        sol_score: float,
        headroom_pct: float,
        bottleneck: BottleneckType,
        tree_context: str = "",
        kb_context: str = "",
        profiling: "ProfilingResult | None" = None,
        roofline: "RooflineResult | None" = None,
        reviewer_metric_queries: bool = False,
        sibling_context: str = "",
    ) -> str:
        """Assemble the user prompt from runtime data.

        When ``profiling`` is supplied, it takes precedence over the raw
        ``profiling_summary`` string — the analytical + NCU blocks (and
        any degradation notice) are rendered from the dataclass so the
        orchestrator doesn't have to stringify the result itself. The
        multi-turn ``## Available raw metrics`` menu is also derived from
        ``profiling.raw_metrics`` when ``reviewer_metric_queries=True``.
        """
        sections: list[str] = []

        sections.append(render_kernel_section(kernel_source))
        sections.append(render_run_context(bottleneck))
        if profiling is not None:
            sections.append(
                "## Profiling summary\n"
                + render_profiling_summary(profiling, roofline)
            )
        else:
            sections.append("## Profiling summary\n" + profiling_summary)
        sections.append(
            "## Scoring\n"
            f"- SOL score: {sol_score:.3f}\n"
            f"- Headroom: {headroom_pct:.1f}%"
        )

        if tree_context:
            sections.append("## Search tree context\n" + tree_context)

        if sibling_context:
            sections.append(
                "## Siblings already tried from this parent\n" + sibling_context
            )

        if kb_context:
            sections.append("## Knowledge base context\n" + kb_context)

        if reviewer_metric_queries:
            raw_metrics = profiling.raw_metrics if profiling is not None else None
            metric_groups = (
                profiling.metric_groups if profiling is not None else None
            )
            if raw_metrics:
                if metric_groups:
                    group_lines = ["## Available metric groups (queryable)"]
                    for group_name in sorted(metric_groups):
                        group = metric_groups[group_name]
                        present = sum(
                            1
                            for status in group.values()
                            if status.get("status") == "present"
                        )
                        missing = len(group) - present
                        group_lines.append(
                            f"- {group_name}: {present} present, {missing} missing"
                        )
                    group_lines.append(
                        'Query a full group with query_metric(["group:<name>"]).'
                    )
                    sections.append("\n".join(group_lines))
                menu_lines = ["## Available raw metrics (queryable)"]
                menu_lines.extend(f"- {k}" for k in sorted(raw_metrics))
                sections.append("\n".join(menu_lines))
            else:
                sections.append(
                    "## Available raw metrics (queryable)\n"
                    "[no NCU data — profiling degraded; "
                    "query_metric will return \"[no data]\" for every requested name]"
                )

        return "\n\n".join(sections)

    # ── main entry point ────────────────────────────────────────────

    async def review(
        self,
        kernel_source: str,
        profiling_summary: str,
        sol_score: float,
        headroom_pct: float,
        bottleneck: BottleneckType,
        tree_context: str = "",
        kb_context: str = "",
        prev_sol_score: float | None = None,
        profiling: "ProfilingResult | None" = None,
        roofline: "RooflineResult | None" = None,
        reviewer_metric_queries: bool = False,
        iter_idx: int = 0,
        sibling_context: str = "",
        max_turns: int | None = None,
    ) -> ReviewerFeedback:
        """Interpret eval results into structured Reviewer feedback.

        Submits via a ``submit_review`` tool call so the SDK never sends a
        ``response_format=json_schema`` (which DeepSeek-reasoner rejects and
        which the SDK's strict-schema validator rejects on
        ``metric_deltas: dict[str, float]``). Pydantic validation still
        runs inside the tool body.

        Two distinct fallback paths to ``rule_based_feedback``:

        - **No model configured** (``has_model`` is False): returns
          rule-based feedback with ``degraded=False`` — this is the
          expected stub mode, not a failure.
        - **LLM-failure fallback** (``run_agent`` returns None, or
          MaxTurnsExceeded fires with no captured submission, or the
          loop ends cleanly without calling submit_review): returns
          rule-based feedback with ``degraded=True`` and a short
          ``error_reason`` tag, so the orchestrator can surface the
          broken signal instead of silently trusting it.

        The rule-based fallback's bottleneck label is the caller-provided
        ``bottleneck`` — the once-per-run classification, invariant
        across iterations (see ``classify_run``).
        """
        if not self.has_model:
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
            profiling=profiling,
            roofline=roofline,
            reviewer_metric_queries=reviewer_metric_queries,
            sibling_context=sibling_context,
        )

        captured: dict = {}
        # ``strict_mode=False`` on both tools: the SDK's strict-schema
        # validator rejects ``dict[str, X]`` and ``list[str]`` arg shapes —
        # see JOURNAL "Strict-mode opt-out for submit-tool dict params
        # (2026-04-26)" for the rationale. Pydantic / inline validation
        # inside each tool body preserves end-to-end type safety.
        submit_tool = function_tool(_make_submit_review_tool(captured), strict_mode=False)
        tools: list = [submit_tool]
        default_max_turns = 4
        # System prompt mentions of ``query_metric`` are conditional on the
        # same flag that gates tool registration — see ``__init__`` for the
        # rationale (live-run ``Tool query_metric not found`` regression).
        instructions = self._instructions_base
        if reviewer_metric_queries:
            raw_metrics = profiling.raw_metrics if profiling is not None else None
            metric_groups = profiling.metric_groups if profiling is not None else None
            query_tool = function_tool(
                _make_query_metric_tool(
                    raw_metrics=raw_metrics,
                    iter_idx=iter_idx,
                    metric_groups=metric_groups,
                ),
                strict_mode=False,
            )
            tools.append(query_tool)
            default_max_turns = 6
            instructions = (
                self._instructions_base + "\n\n" + self._instructions_metric_queries
            )
        # cfg-tunable override (ACTSConfig.reviewer_max_turns). None
        # preserves the 4-or-6 toggle above so existing behavior is
        # unchanged; non-None overrides both branches uniformly.
        effective_max_turns = default_max_turns if max_turns is None else max_turns
        agent = Agent(
            name="Reviewer",
            instructions=instructions,
            model=self._model,
            tools=tools,
        )

        def _degraded(reason: str) -> ReviewerFeedback:
            return rule_based_feedback(
                sol_score=sol_score,
                prev_sol_score=prev_sol_score,
                headroom_pct=headroom_pct,
                bottleneck=bottleneck,
                degraded=True,
                error_reason=reason,
            )

        # Turn budget: by default ``max_turns`` is 4 (submit-only) or 6
        # (multi-turn, query_metric enabled). The submit-only path
        # (2*N+2 with N=1) reserves room for one invalid submit +
        # corrected submit + confirmation, so a single Pydantic slip
        # self-corrects in-loop instead of degrading to rule-based. The
        # multi-turn path adds room for one query_metric fetch. The cfg
        # override (``ACTSConfig.reviewer_max_turns``) supersedes both
        # branches when set.
        try:
            result = await run_agent(
                agent,
                prompt,
                run_config=make_run_config(temperature=0.3),
                max_turns=effective_max_turns,
            )
        except MaxTurnsExceeded:
            if "output" in captured:
                return _output_to_feedback(captured["output"])
            return _degraded("max_turns_exceeded")
        except ModelBehaviorError as exc:
            # Defense-in-depth: the SDK raises ``ModelBehaviorError`` when
            # the LLM emits a tool call for a name that isn't registered
            # (e.g. a future prompt-leak regression that mentions a tool
            # the orchestrator did not wire up). Without this catch the
            # exception unwinds ``Orchestrator.run()`` and aborts the
            # whole optimization run; with it, this iteration degrades
            # to rule-based feedback and the run continues. If a partial
            # submission was captured before the bad tool call, prefer
            # that — same precedence rule as ``MaxTurnsExceeded``.
            if "output" in captured:
                return _output_to_feedback(captured["output"])
            reason = f"model_behavior_error:{type(exc).__name__}"
            return _degraded(reason)

        if result is None:
            return _degraded("llm_retries_exhausted")
        if "output" not in captured:
            return _degraded("missing_submit_review")
        return _output_to_feedback(captured["output"])
