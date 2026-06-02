"""Resource-usage accumulator and snapshot types.

Live `UsageAccumulator` (Task 2) is owned by `JSONLTraceProcessor` and
buffers SDK `generation` spans by `trace_id`, resolving the
`(iter, agent)` bucket when the trace closes. `UsageSnapshot.snapshot()`
returns a frozen view consumed by the report layer.

Pure-Python — no SDK / torch imports — so Tier-1 mocked tests can
exercise the accumulator without the openai-agents SDK installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class AgentLabel(str, Enum):
    """Canonical agent labels used in trace metadata + usage accounting.

    Subclassing ``str`` keeps backward compatibility: existing string-keyed
    dicts and JSON-serializable metadata both work transparently
    (``AgentLabel.CODER == "coder"`` and
    ``hash(AgentLabel.CODER) == hash("coder")``). Pre-3.11 idiom (vs.
    ``enum.StrEnum``) — Tier-1 venv is Python 3.10.
    """

    PLANNER = "planner"
    CODER = "coder"
    CODER_TRANSLATE = "coder-translate"
    REVIEWER = "reviewer"
    SUMMARIZER = "summarizer"


# Canonical column order for the rendered table + `usage.json#columns`.
# Observed agents not in this tuple sort alphabetically after these.
# Derived from `AgentLabel` so the enum is the single source of truth.
_CANONICAL_AGENT_ORDER: tuple[str, ...] = tuple(m.value for m in AgentLabel)


@dataclass(frozen=True)
class UsageBucket:
    """Per-(iter, agent) accumulation of model usage. Frozen for snapshot safety.

    `invocations` counts distinct trace_ids tagged with this (iter, agent);
    `turns` counts generation spans (one per model round-trip).
    """

    invocations: int = 0
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    def __add__(self, other: "UsageBucket") -> "UsageBucket":
        return UsageBucket(
            invocations=self.invocations + other.invocations,
            turns=self.turns + other.turns,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_output_tokens=(
                self.reasoning_output_tokens + other.reasoning_output_tokens
            ),
        )

    @property
    def is_zero(self) -> bool:
        return self == UsageBucket()


@dataclass(frozen=True)
class UsageSnapshot:
    """Frozen rollup of an accumulator's state, safe to consume while the
    live accumulator may still be receiving teardown events.

    `columns` is the canonical render order — `_CANONICAL_AGENT_ORDER`
    head plus any other observed agents sorted alphabetically.
    """

    by_iter_agent: Mapping[tuple[int, str], UsageBucket]
    by_iter: Mapping[int, UsageBucket]
    by_agent: Mapping[str, UsageBucket]
    total: UsageBucket
    columns: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return self.total.is_zero


def _fmt_tokens(n: int) -> str:
    """Render a token count with k/M abbreviation: <1000=exact, <1M=one-decimal k, else one-decimal M."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


@dataclass
class _PendingSpan:
    """One buffered generation-span delta awaiting trace-close.

    Holds a pre-parsed `UsageBucket` (turns=1, invocations=0) instead of
    the raw usage dict so `on_trace_close` is pure dict-merge work.
    """

    delta: UsageBucket


def _safe_int(val: object) -> int:
    """Coerce `val` to int, returning 0 on TypeError/ValueError.

    Defensive against version-skewed providers that may send non-numeric
    strings (e.g. ``"unknown"``) or unexpected types in usage fields.
    """
    if val is None:
        return 0
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _pick_token_count(usage: dict, primary: str, fallback: str) -> int:
    """Read a token count preferring `primary` over `fallback`. Returns 0
    if neither key has a numeric value. Used to bridge between the SDK's
    new Responses-API shape (input_tokens/output_tokens) and the legacy
    Chat-Completions shape (prompt_tokens/completion_tokens) — both occur
    in real provider responses, depending on SDK version + provider.

    Precedence: if `primary` is present (even when zero or non-numeric),
    `fallback` is not consulted. This keeps the new-canonical name
    authoritative when a mixed/buggy provider emits both.
    """
    if primary in usage:
        return _safe_int(usage.get(primary))
    return _safe_int(usage.get(fallback, 0))


def _parse_span_usage(usage: object) -> UsageBucket:
    """Convert an SDK `generation` span `usage` dict into a per-span
    `UsageBucket` delta. Missing keys default to 0; the SDK does not
    guarantee every field is present on every provider.

    Defensive: returns `UsageBucket(turns=1)` for non-dict / malformed
    inputs rather than raising. Called from inside the SDK trace
    callback path where exceptions would violate the never-raise
    discipline of the trace processor.

    Key naming: prefers the Responses-API canonical names
    (`input_tokens` / `output_tokens`) but falls back to the legacy
    Chat-Completions names (`prompt_tokens` / `completion_tokens`)
    when the new names are absent — both shapes appear in real
    SDK exports depending on version + provider.
    """
    if not isinstance(usage, dict):
        return UsageBucket(turns=1)
    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    output_details = usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}
    return UsageBucket(
        invocations=0,
        turns=1,
        input_tokens=_pick_token_count(usage, "input_tokens", "prompt_tokens"),
        output_tokens=_pick_token_count(usage, "output_tokens", "completion_tokens"),
        cached_input_tokens=_safe_int(input_details.get("cached_tokens", 0)),
        reasoning_output_tokens=_safe_int(output_details.get("reasoning_tokens", 0)),
    )


def _order_columns(observed: set[str]) -> tuple[str, ...]:
    """Order observed agent labels: canonical head first, then alphabetical tail."""
    head = tuple(a for a in _CANONICAL_AGENT_ORDER if a in observed)
    tail = tuple(sorted(observed - set(_CANONICAL_AGENT_ORDER)))
    return head + tail


class UsageAccumulator:
    """Live mutable accumulator owned by `JSONLTraceProcessor`.

    Thread-safety: callers must hold the processor's existing lock when
    calling `on_generation_span` / `on_trace_close` / `snapshot`. The
    accumulator does not own a lock — piggybacking the processor's
    avoids double-locking when both methods touch the same state in
    the SDK's processor callbacks.

    Trace closures are resolved at `on_trace_close`. Generation spans
    that arrive before their trace closes are buffered in `_pending`;
    spans whose trace never closes are dropped at the next snapshot.
    Spans arriving AFTER their trace's close (SDK callbacks can fire
    from arbitrary worker threads) are credited directly against the
    already-closed bucket via `_closed_traces`.
    """

    def __init__(self) -> None:
        self._by_iter_agent: dict[tuple[int, str], UsageBucket] = {}
        self._pending: dict[str, list[_PendingSpan]] = {}
        # trace_id -> (iter, agent) for already-closed traces, so a late
        # span_end can still credit the right bucket. Tuple (not the full
        # metadata dict) keeps the value immutable + small.
        self._closed_traces: dict[str, tuple[int, str]] = {}

    def on_generation_span(self, trace_id: str, usage: dict | None) -> None:
        """Route a generation-span delta to its `trace_id`.

        - If the trace has already closed (late span_end from a worker
          thread), credit the delta directly against the closed
          (iter, agent) bucket. No invocation bump — close already did
          that.
        - Otherwise buffer in `_pending` for resolution at trace-close.
        """
        delta = _parse_span_usage(usage)
        closed_key = self._closed_traces.get(trace_id)
        if closed_key is not None:
            self._by_iter_agent[closed_key] = (
                self._by_iter_agent.get(closed_key, UsageBucket()) + delta
            )
            return
        self._pending.setdefault(trace_id, []).append(_PendingSpan(delta=delta))

    def on_trace_close(self, trace_id: str, metadata: dict) -> None:
        """Resolve buffered spans for `trace_id` and credit the
        `(iter, agent)` bucket. Idempotent: a duplicate close on the
        same `trace_id` is a no-op.
        """
        if trace_id in self._closed_traces:
            # Duplicate close — already credited. Drop any buffer too so
            # a stray span doesn't double-credit on a third close.
            self._pending.pop(trace_id, None)
            return
        # Need both `iter` and `agent` to attribute. Without either, drop
        # the spans rather than attribute to a phantom bucket. (Do NOT
        # record in _closed_traces — we want a future tagged close on
        # the same trace_id to still work, and we have no key to credit
        # late spans against anyway.)
        iter_no = metadata.get("iter")
        agent = metadata.get("agent")
        if iter_no is None or agent is None:
            self._pending.pop(trace_id, None)
            return
        try:
            key = (int(iter_no), str(agent))
        except (TypeError, ValueError):
            # Malformed metadata (e.g. iter is non-numeric string). Drop
            # the spans rather than crash from inside the trace callback.
            self._pending.pop(trace_id, None)
            return
        spans = self._pending.pop(trace_id, [])
        # One invocation per resolved trace, regardless of span count.
        delta = UsageBucket(invocations=1, turns=0)
        for s in spans:
            delta = delta + s.delta
        self._by_iter_agent[key] = self._by_iter_agent.get(key, UsageBucket()) + delta
        self._closed_traces[trace_id] = key

    def snapshot(self) -> UsageSnapshot:
        """Return a frozen snapshot; subsequent mutations don't affect the returned snapshot."""
        by_iter_agent = dict(self._by_iter_agent)  # copy for freezing
        by_iter: dict[int, UsageBucket] = {}
        by_agent: dict[str, UsageBucket] = {}
        total = UsageBucket()
        for (iter_no, agent), bucket in by_iter_agent.items():
            by_iter[iter_no] = by_iter.get(iter_no, UsageBucket()) + bucket
            by_agent[agent] = by_agent.get(agent, UsageBucket()) + bucket
            total = total + bucket
        return UsageSnapshot(
            by_iter_agent=by_iter_agent,
            by_iter=by_iter,
            by_agent=by_agent,
            total=total,
            columns=_order_columns({agent for (_, agent) in self._by_iter_agent.keys()}),
        )
