"""Reviewer agent — interprets eval results into structured feedback.

Single-call agent (no tools). Uses OpenAI Agents SDK Agent + Runner.run with a
Pydantic output_type. Falls back to rule-based feedback when the LLM is
unavailable or its call fails after all retries.

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


# ── Pydantic output model (schema sent to LLM via output_type) ─────────


class ReviewerFeedbackOutput(BaseModel):
    """Structured output schema enforced on the LLM response."""

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


def render_profiling_summary(profiling: "ProfilingResult | None") -> str:
    """Render a ``ProfilingResult`` into the two-block text the Reviewer
    prompt expects. Used both by ``ReviewerAgent.build_user_prompt`` and
    by ``pipeline.report.render_report`` so the format is consistent
    across the reviewer and the final operator-facing summary.

    When ``profiling`` is ``None``, returns a one-line notice so the
    Reviewer knows it's operating without profile data (rather than
    silently reusing stale context).
    """
    if profiling is None:
        return "[no profiling data — profile_kernel did not run]"

    a = profiling.analytical
    lines = [
        "### Analytical (roofline)",
        f"- arithmetic_intensity: {a.arithmetic_intensity:.3f} FLOP/byte",
        f"- ridge_point: {a.ridge_point:.3f} FLOP/byte",
        f"- achieved: {a.achieved_tflops:.2f} TFLOPS / {a.achieved_bandwidth_gb_s:.2f} GB/s",
        f"- pct_peak: compute {a.pct_peak_compute * 100:.1f}% · bw {a.pct_peak_bandwidth * 100:.1f}%",
    ]

    ncu = profiling.ncu
    if ncu is not None:
        lines.extend([
            "",
            "### NCU (curated)",
            f"- sm_occupancy: {ncu.sm_occupancy_pct:.1f}%",
            f"- l2_hit_rate: {ncu.l2_hit_rate_pct:.1f}%",
            f"- tensor_core_util: {ncu.tensor_core_util_pct:.1f}%",
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
        if not raw_metrics:
            return {n: "[no data]" for n in coerced}
        return {
            n: (f"{raw_metrics[n]}" if n in raw_metrics else "[unknown]")
            for n in coerced
        }

    return query_metric


# ── Agent ──────────────────────────────────────────────────────────────


class ReviewerAgent:
    """Interprets evaluation results and produces structured feedback.

    Acts as intelligent filter between raw profiling data and the Planner.
    Single-call, no tools — receives all eval data in the prompt.

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
            self._instructions = (prompt_dir / "system.md").read_text()
        else:
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
        reviewer_metric_queries: bool = False,
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
            sections.append("## Profiling summary\n" + render_profiling_summary(profiling))
        else:
            sections.append("## Profiling summary\n" + profiling_summary)
        sections.append(
            "## Scoring\n"
            f"- SOL score: {sol_score:.3f}\n"
            f"- Headroom: {headroom_pct:.1f}%"
        )

        if tree_context:
            sections.append("## Search tree context\n" + tree_context)

        if kb_context:
            sections.append("## Knowledge base context\n" + kb_context)

        if reviewer_metric_queries:
            raw_metrics = profiling.raw_metrics if profiling is not None else None
            if raw_metrics:
                menu_lines = ["## Available raw metrics (queryable)"]
                menu_lines.extend(f"- {k}" for k in sorted(raw_metrics))
                sections.append("\n".join(menu_lines))
            else:
                sections.append(
                    "## Available raw metrics (queryable)\n"
                    "[no NCU data — profiling degraded; "
                    "query_metric will return empty]"
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
        reviewer_metric_queries: bool = False,
        iter_idx: int = 0,
    ) -> ReviewerFeedback:
        """Interpret eval results into structured Reviewer feedback.

        Submits via a ``submit_review`` tool call so the SDK never sends a
        ``response_format=json_schema`` (which DeepSeek-reasoner rejects and
        which the SDK's strict-schema validator rejects on
        ``metric_deltas: dict[str, float]``). Pydantic validation still
        runs inside the tool body.

        Falls back to rule-based degraded feedback when no model is
        configured, when run_agent returns None (transient retries
        exhausted), when MaxTurnsExceeded fires with no captured
        submission, or when the loop ends cleanly without calling
        submit_review. The rule-based fallback's bottleneck label is the
        caller-provided ``bottleneck`` — the once-per-run classification,
        invariant across iterations (see ``classify_run``).
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
            reviewer_metric_queries=reviewer_metric_queries,
        )

        captured: dict = {}
        # ``strict_mode=False`` on both tools: the SDK's strict-schema
        # validator rejects ``dict[str, X]`` and ``list[str]`` arg shapes —
        # see JOURNAL "Strict-mode opt-out for submit-tool dict params
        # (2026-04-26)" for the rationale. Pydantic / inline validation
        # inside each tool body preserves end-to-end type safety.
        submit_tool = function_tool(_make_submit_review_tool(captured), strict_mode=False)
        tools: list = [submit_tool]
        max_turns = 4
        if reviewer_metric_queries:
            raw_metrics = profiling.raw_metrics if profiling is not None else None
            query_tool = function_tool(
                _make_query_metric_tool(raw_metrics=raw_metrics, iter_idx=iter_idx),
                strict_mode=False,
            )
            tools.append(query_tool)
            max_turns = 6
        agent = Agent(
            name="Reviewer",
            instructions=self._instructions,
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

        # Turn budget: ``max_turns`` is 4 (single-call) or 6 (multi-turn).
        # The single-call path (2*N+2 with N=1) reserves room for one
        # invalid submit + corrected submit + confirmation, so a single
        # Pydantic slip self-corrects in-loop instead of degrading to
        # rule-based. The multi-turn path adds room for one fetch.
        try:
            result = await run_agent(
                agent,
                prompt,
                run_config=make_run_config(temperature=0.3),
                max_turns=max_turns,
            )
        except MaxTurnsExceeded:
            if "output" in captured:
                return _output_to_feedback(captured["output"])
            return _degraded("max_turns_exceeded")

        if result is None:
            return _degraded("llm_retries_exhausted")
        if "output" not in captured:
            return _degraded("missing_submit_review")
        return _output_to_feedback(captured["output"])
