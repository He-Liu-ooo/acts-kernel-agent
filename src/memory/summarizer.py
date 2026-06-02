"""Summarizer agent — distills a (parent, child, speedup) edge into a lesson.

Two prompt variants:

* :meth:`SummarizerAgent.summarize` — edge-scope, one-step optimization
  (AccelOpt-faithful tone)
* :meth:`SummarizerAgent.summarize_run` — run-scope, cumulative multi-step
  strategy summary

Failure modes are bounded: malformed JSON, ``"No optimization found"``,
empty / identical snippets, or LLM exceptions all yield ``None`` with a
single warn log. The summarizer never raises to the caller — opt-mem
write is best-effort by design.

See ``doc/specs/2026-05-24-optimization-memory-design.md`` §6 + §9.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.memory.experience import ActionRecord

logger = logging.getLogger(__name__)

# Match the convention used by Planner / Coder / Reviewer
# (``src/agents/<agent>.py`` reads ``src/prompts/<agent>/system.md``).
# Summarizer's prompts are user-prompt templates with ``{field}``
# placeholders rather than agent instructions — same on-disk location,
# slightly different load path (read into module-level constants at
# import time so the templates can still be ``str.format``-ed).
_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "summarizer"
_EDGE_PROMPT_TEMPLATE = (_PROMPT_DIR / "edge.md").read_text()
_RUN_PROMPT_TEMPLATE = (_PROMPT_DIR / "run.md").read_text()


@dataclass
class SummarizerResult:
    title: str
    lesson: str
    snippet_before: str
    snippet_after: str


class SummarizerAgent:
    """Wraps the shared LLM model to produce structured lessons."""

    def __init__(self, model, summarizer_model_name: str) -> None:
        self._model = model
        self._model_name = summarizer_model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def summarize(
        self,
        parent_src: str,
        child_src: str,
        speedup: float,
        action: ActionRecord,
        *,
        iter_no: int = 0,
    ) -> SummarizerResult | None:
        prompt = _EDGE_PROMPT_TEMPLATE.format(
            speedup=speedup,
            action_name=action.name,
            action_tier=action.tier,
            parent_src=parent_src,
            child_src=child_src,
        )
        return await self._call_and_parse(prompt, iter_no=iter_no)

    async def summarize_run(
        self,
        baseline_src: str,
        best_src: str,
        cumulative_speedup: float,
        *,
        iter_no: int = 0,
    ) -> SummarizerResult | None:
        prompt = _RUN_PROMPT_TEMPLATE.format(
            cumulative_speedup=cumulative_speedup,
            baseline_src=baseline_src,
            best_src=best_src,
        )
        return await self._call_and_parse(prompt, iter_no=iter_no)

    async def _call_and_parse(
        self, prompt: str, *, iter_no: int = 0
    ) -> SummarizerResult | None:
        try:
            raw = await self._run(prompt, iter_no=iter_no)
        except Exception as exc:  # noqa: BLE001
            logger.warning("summarizer LLM call failed: %s", exc)
            return None
        # Runner.run can yield ``final_output=None`` (max_turns exhausted
        # without a final assistant message). ``json.loads(None)`` raises
        # ``TypeError``, not ``JSONDecodeError`` — guard explicitly so the
        # documented "warn-and-return-None" contract holds.
        if raw is None or not isinstance(raw, str):
            logger.warning(
                "summarizer returned non-string final_output (%r); rejecting row",
                type(raw).__name__,
            )
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "summarizer returned non-JSON response: %s; raw=%r",
                exc, raw[:200],
            )
            return None
        # A valid-JSON-but-wrong-shape response (list, scalar, null) would
        # crash ``data.get(...)`` with ``AttributeError`` — reject before
        # field access so the documented never-raise contract holds.
        if not isinstance(data, dict):
            logger.warning(
                "summarizer returned non-object JSON (%s); rejecting row",
                type(data).__name__,
            )
            return None
        # A valid-JSON object with a truthy non-string field — e.g.
        # ``{"title": 5}`` or ``{"snippet_before": ["x"]}`` — would pass the
        # ``... or ""`` guard (truthy) and then crash ``.strip()`` with
        # ``AttributeError`` (title/lesson) or flow downstream as a non-string
        # (snippets). Reject before field coercion so the documented
        # never-raise contract holds. ``None``/missing is still tolerated and
        # coerces to ``""`` below — only a non-None non-str is rejected.
        title_raw = data.get("title")
        lesson_raw = data.get("lesson")
        snippet_before_raw = data.get("snippet_before")
        snippet_after_raw = data.get("snippet_after")
        if any(
            v is not None and not isinstance(v, str)
            for v in (title_raw, lesson_raw, snippet_before_raw, snippet_after_raw)
        ):
            logger.warning(
                "summarizer returned non-string field "
                "(title=%r lesson=%r snippet_before=%r snippet_after=%r); "
                "rejecting row",
                type(title_raw).__name__, type(lesson_raw).__name__,
                type(snippet_before_raw).__name__, type(snippet_after_raw).__name__,
            )
            return None
        title = (title_raw or "").strip()
        lesson = (lesson_raw or "").strip()
        snippet_before = snippet_before_raw or ""
        snippet_after = snippet_after_raw or ""
        if title == "No optimization found":
            logger.warning("summarizer reported no optimization found")
            return None
        if not snippet_before or not snippet_after:
            logger.warning("summarizer produced empty snippet; rejecting row")
            return None
        if snippet_before == snippet_after:
            logger.warning(
                "summarizer produced identical before/after snippets; rejecting row"
            )
            return None
        # Defense in depth against fence escape in the Planner prompt.
        # ``planner._render_past_experiences`` wraps snippets in 4-backtick
        # fences; a snippet containing ``≥ 4`` consecutive backticks could
        # still close the fence and bleed into surrounding prose. Reject
        # the row outright — a kernel needing 4+ backticks in its source
        # is a strong indicator the summarizer extracted Markdown rather
        # than code anyway. Three-backtick docstrings / comments pass.
        if "````" in snippet_before or "````" in snippet_after:
            logger.warning(
                "summarizer produced snippet with 4+ consecutive backticks; "
                "rejecting row to prevent Planner-prompt fence escape"
            )
            return None
        return SummarizerResult(
            title=title,
            lesson=lesson,
            snippet_before=snippet_before,
            snippet_after=snippet_after,
        )

    async def _run(self, prompt: str, *, iter_no: int = 0) -> str | None:
        """Call the underlying model; returns the raw ``final_output`` string.

        Isolated as a single seam so tests can monkeypatch it without
        touching prompt-building or parsing logic. Production wiring goes
        through ``llm_backend.run_agent`` (retry + jittered exponential
        backoff on transient errors) + ``make_run_config`` (forced
        temperature for reasoning models, ``max_tokens`` overrides,
        provider-specific ``extra_body``) — mirrors ``planner.py`` /
        ``reviewer.py`` so the summarizer doesn't silently bypass the
        retry/config plumbing every other agent goes through. Returns
        ``None`` when retries are exhausted (``_call_and_parse`` already
        handles None per the documented warn-and-return-None contract).

        The ``run_agent`` call is wrapped in a ``trace_span`` tagged with
        ``iter`` + ``AgentLabel.SUMMARIZER`` so ``UsageAccumulator`` buckets
        the summarizer's tokens — a trace lacking both ``iter`` and
        ``agent`` metadata is dropped, so usage.json / the report would
        otherwise omit summarizer tokens entirely.
        """
        from agents import Agent  # local import: SDK is heavy

        from src.agents.llm_backend import make_run_config, run_agent
        from src.runtime.sdk_trace import trace_span
        from src.runtime.usage import AgentLabel

        agent = Agent(
            name="summarizer",
            instructions="You are a GPU-kernel optimization expert.",
            model=self._model,
        )
        # max_turns=2 covers the no-tool single-call shape (one assistant
        # turn + one safety turn) — the summarizer has no tools to loop
        # through, so a tight budget keeps a runaway reasoning model from
        # burning tokens. temperature=0.3 mirrors Planner/Reviewer; the
        # forced-temperature override in make_run_config takes precedence
        # on reasoning models.
        with trace_span(
            "acts_summarizer", iter_no=iter_no, agent=AgentLabel.SUMMARIZER,
        ):
            result = await run_agent(
                agent,
                prompt,
                run_config=make_run_config(temperature=0.3),
                max_turns=2,
            )
        if result is None:
            return None
        return result.final_output
