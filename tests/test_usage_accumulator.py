"""Tier-1 unit tests for src/runtime/usage.py — no SDK, no torch."""

from __future__ import annotations

import pytest

from src.runtime.usage import (
    AgentLabel,
    UsageBucket,
    UsageSnapshot,
    _CANONICAL_AGENT_ORDER,
    _fmt_tokens,
    _parse_span_usage,
)


class TestUsageBucket:
    def test_default_is_zero(self):
        b = UsageBucket()
        assert b.invocations == 0
        assert b.turns == 0
        assert b.input_tokens == 0
        assert b.output_tokens == 0
        assert b.cached_input_tokens == 0
        assert b.reasoning_output_tokens == 0
        assert b.is_zero is True

    def test_non_zero_detected(self):
        assert UsageBucket(invocations=1).is_zero is False
        assert UsageBucket(turns=1).is_zero is False
        assert UsageBucket(input_tokens=1).is_zero is False

    def test_addition_sums_each_field(self):
        a = UsageBucket(
            invocations=1, turns=2, input_tokens=100, output_tokens=20,
            cached_input_tokens=30, reasoning_output_tokens=5,
        )
        b = UsageBucket(
            invocations=2, turns=3, input_tokens=200, output_tokens=40,
            cached_input_tokens=50, reasoning_output_tokens=10,
        )
        s = a + b
        assert s == UsageBucket(
            invocations=3, turns=5, input_tokens=300, output_tokens=60,
            cached_input_tokens=80, reasoning_output_tokens=15,
        )

    def test_addition_is_immutable(self):
        a = UsageBucket(invocations=1)
        b = UsageBucket(invocations=2)
        _ = a + b
        assert a.invocations == 1
        assert b.invocations == 2

    def test_frozen(self):
        b = UsageBucket()
        with pytest.raises(Exception):  # FrozenInstanceError or dataclass-equivalent
            b.invocations = 99  # type: ignore[misc]


class TestFmtTokens:
    @pytest.mark.parametrize("n,expected", [
        (0, "0"),
        (1, "1"),
        (42, "42"),
        (999, "999"),
        (1000, "1.0k"),
        (1234, "1.2k"),
        (27_400, "27.4k"),
        (999_999, "1000.0k"),  # boundary: <1M still rendered with k
        (1_000_000, "1.0M"),
        (1_500_000, "1.5M"),
    ])
    def test_boundaries(self, n, expected):
        assert _fmt_tokens(n) == expected


class TestUsageSnapshot:
    def test_empty_snapshot_is_empty(self):
        snap = UsageSnapshot(
            by_iter_agent={}, by_iter={}, by_agent={},
            total=UsageBucket(), columns=(),
        )
        assert snap.is_empty is True

    def test_non_empty_snapshot(self):
        snap = UsageSnapshot(
            by_iter_agent={(0, "coder"): UsageBucket(invocations=1)},
            by_iter={0: UsageBucket(invocations=1)},
            by_agent={"coder": UsageBucket(invocations=1)},
            total=UsageBucket(invocations=1),
            columns=("coder",),
        )
        assert snap.is_empty is False

    def test_canonical_agent_order_constant(self):
        # The constant is the source of truth for column ordering — guard it.
        assert _CANONICAL_AGENT_ORDER == (
            "planner", "coder", "coder-translate", "reviewer", "summarizer",
        )


from src.runtime.usage import UsageAccumulator


def _usage(
    input_tokens: int = 0, output_tokens: int = 0,
    cached: int = 0, reasoning: int = 0,
) -> dict:
    """Build a fake SDK usage dict matching `generation` span shape."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }


class TestUsageAccumulator:
    def test_happy_path_three_spans_one_trace(self):
        acc = UsageAccumulator()
        acc.on_generation_span("t1", _usage(input_tokens=100, output_tokens=10))
        acc.on_generation_span("t1", _usage(input_tokens=200, output_tokens=20))
        acc.on_generation_span("t1", _usage(input_tokens=300, output_tokens=30))
        acc.on_trace_close("t1", {"iter": 1, "agent": "coder"})
        snap = acc.snapshot()
        bucket = snap.by_iter_agent[(1, "coder")]
        assert bucket.invocations == 1
        assert bucket.turns == 3
        assert bucket.input_tokens == 600
        assert bucket.output_tokens == 60

    def test_two_traces_same_iter_agent(self):
        acc = UsageAccumulator()
        acc.on_generation_span("t1", _usage(input_tokens=100, output_tokens=10))
        acc.on_trace_close("t1", {"iter": 1, "agent": "coder"})
        acc.on_generation_span("t2", _usage(input_tokens=200, output_tokens=20))
        acc.on_generation_span("t2", _usage(input_tokens=300, output_tokens=30))
        acc.on_trace_close("t2", {"iter": 1, "agent": "coder"})
        bucket = acc.snapshot().by_iter_agent[(1, "coder")]
        assert bucket.invocations == 2
        assert bucket.turns == 3
        assert bucket.input_tokens == 600
        assert bucket.output_tokens == 60

    def test_out_of_order_arrival(self):
        # span_end can fire before trace_end (trace closes when its
        # containing `with` block exits, after inner spans). Buffer +
        # resolve at trace-close is the only correct order.
        acc = UsageAccumulator()
        acc.on_generation_span("t1", _usage(input_tokens=50, output_tokens=5))
        # Trace not yet closed — bucket should not exist
        assert (1, "coder") not in acc.snapshot().by_iter_agent
        acc.on_trace_close("t1", {"iter": 1, "agent": "coder"})
        bucket = acc.snapshot().by_iter_agent[(1, "coder")]
        assert bucket.turns == 1
        assert bucket.input_tokens == 50

    def test_duplicate_trace_close_idempotent(self):
        # Defensive — SDK has shown late-arriving events.
        acc = UsageAccumulator()
        acc.on_generation_span("t1", _usage(input_tokens=100, output_tokens=10))
        acc.on_trace_close("t1", {"iter": 1, "agent": "coder"})
        acc.on_trace_close("t1", {"iter": 1, "agent": "coder"})  # duplicate
        bucket = acc.snapshot().by_iter_agent[(1, "coder")]
        assert bucket.invocations == 1
        assert bucket.turns == 1

    def test_orphan_spans_dropped(self):
        # Trace never closes (crash mid-iter). Spans for that trace stay
        # in the pending buffer and are not attributed.
        acc = UsageAccumulator()
        acc.on_generation_span("orphan", _usage(input_tokens=100, output_tokens=10))
        acc.on_generation_span("good", _usage(input_tokens=50, output_tokens=5))
        acc.on_trace_close("good", {"iter": 1, "agent": "planner"})
        snap = acc.snapshot()
        assert (1, "orphan") not in snap.by_iter_agent
        # Only the closed trace is in the snapshot.
        assert list(snap.by_iter_agent.keys()) == [(1, "planner")]

    def test_missing_metadata_dropped(self):
        # Trace closed but metadata lacks iter / agent (e.g., baseline
        # translate before we wrapped it, or an unrelated SDK trace).
        # Spans are dropped rather than attributed to (None, None).
        acc = UsageAccumulator()
        acc.on_generation_span("t1", _usage(input_tokens=100, output_tokens=10))
        acc.on_trace_close("t1", {})  # no iter/agent
        snap = acc.snapshot()
        assert snap.by_iter_agent == {}
        assert snap.total.is_zero

    def test_sub_buckets_accumulate(self):
        acc = UsageAccumulator()
        acc.on_generation_span(
            "t1",
            _usage(input_tokens=100, output_tokens=20, cached=30, reasoning=10),
        )
        acc.on_generation_span(
            "t1",
            _usage(input_tokens=200, output_tokens=40, cached=50, reasoning=20),
        )
        acc.on_trace_close("t1", {"iter": 1, "agent": "coder"})
        bucket = acc.snapshot().by_iter_agent[(1, "coder")]
        assert bucket.cached_input_tokens == 80
        assert bucket.reasoning_output_tokens == 30
        # Sub-buckets are <= their parent.
        assert bucket.cached_input_tokens <= bucket.input_tokens
        assert bucket.reasoning_output_tokens <= bucket.output_tokens

    def test_snapshot_is_frozen_against_live_mutation(self):
        acc = UsageAccumulator()
        acc.on_generation_span("t1", _usage(input_tokens=100, output_tokens=10))
        acc.on_trace_close("t1", {"iter": 1, "agent": "coder"})
        snap_a = acc.snapshot()
        # Live accumulator gets more activity after the snapshot.
        acc.on_generation_span("t2", _usage(input_tokens=999, output_tokens=99))
        acc.on_trace_close("t2", {"iter": 2, "agent": "reviewer"})
        # Snapshot A must not have grown.
        assert snap_a.by_iter_agent.get((2, "reviewer")) is None
        assert snap_a.total.input_tokens == 100

    def test_by_iter_and_by_agent_are_consistent(self):
        acc = UsageAccumulator()
        acc.on_generation_span("t1", _usage(input_tokens=100, output_tokens=10))
        acc.on_trace_close("t1", {"iter": 1, "agent": "planner"})
        acc.on_generation_span("t2", _usage(input_tokens=200, output_tokens=20))
        acc.on_trace_close("t2", {"iter": 1, "agent": "coder"})
        acc.on_generation_span("t3", _usage(input_tokens=300, output_tokens=30))
        acc.on_trace_close("t3", {"iter": 2, "agent": "coder"})
        snap = acc.snapshot()
        # Row sums equal column sums equal grand total.
        row_sum = snap.by_iter[1] + snap.by_iter[2]
        col_sum = snap.by_agent["planner"] + snap.by_agent["coder"]
        assert row_sum == snap.total
        assert col_sum == snap.total
        assert snap.total.input_tokens == 600
        assert snap.total.invocations == 3
        assert snap.total.turns == 3

    def test_columns_pin_canonical_order(self):
        acc = UsageAccumulator()
        # Add agents in non-canonical order.
        for tid, agent in [
            ("ta", "reviewer"),
            ("tb", "zebra-agent"),       # unknown -> alphabetical tail
            ("tc", "coder-translate"),
            ("td", "planner"),
            ("te", "coder"),
        ]:
            acc.on_generation_span(tid, _usage(input_tokens=1))
            acc.on_trace_close(tid, {"iter": 0, "agent": agent})
        snap = acc.snapshot()
        # Canonical head, then alphabetical tail.
        assert snap.columns == (
            "planner", "coder", "coder-translate", "reviewer", "zebra-agent",
        )

    # ── Fix 2: late spans against closed traces ────────────────────────
    def test_late_span_after_trace_close_is_credited(self):
        # SDK callbacks come from arbitrary worker threads; a generation
        # span_end can arrive AFTER its trace's trace_end. The accumulator
        # must credit the late span to the already-closed bucket without
        # double-counting the invocation.
        acc = UsageAccumulator()
        acc.on_trace_close("t1", {"iter": 1, "agent": "coder"})
        # Late span arrives after the close.
        acc.on_generation_span("t1", _usage(input_tokens=100, output_tokens=10))
        bucket = acc.snapshot().by_iter_agent[(1, "coder")]
        assert bucket.invocations == 1  # close already counted it
        assert bucket.turns == 1
        assert bucket.input_tokens == 100
        assert bucket.output_tokens == 10

    def test_orphan_spans_still_dropped_when_trace_never_closes(self):
        # Regression guard: the close-tracking refactor must not change
        # the never-closes-drop behavior. Spans for a never-closed trace
        # stay in _pending and never get credited.
        acc = UsageAccumulator()
        acc.on_generation_span("orphan", _usage(input_tokens=100, output_tokens=10))
        snap = acc.snapshot()
        assert snap.by_iter_agent == {}
        assert snap.total.is_zero

    def test_multiple_late_spans_on_same_closed_trace(self):
        # Three late spans against an already-closed trace all credit the
        # bucket; invocations remains 1.
        acc = UsageAccumulator()
        acc.on_trace_close("t1", {"iter": 2, "agent": "reviewer"})
        acc.on_generation_span("t1", _usage(input_tokens=10, output_tokens=1))
        acc.on_generation_span("t1", _usage(input_tokens=20, output_tokens=2))
        acc.on_generation_span("t1", _usage(input_tokens=30, output_tokens=3))
        bucket = acc.snapshot().by_iter_agent[(2, "reviewer")]
        assert bucket.invocations == 1
        assert bucket.turns == 3
        assert bucket.input_tokens == 60
        assert bucket.output_tokens == 6


# ── Fix 1: _parse_span_usage robustness ────────────────────────────────


class TestParseSpanUsageRobust:
    """Version-skewed providers can return non-dict / non-numeric payloads.
    `_parse_span_usage` must return a valid UsageBucket(turns=1) rather
    than raising — it's called from the SDK trace callback path."""

    def test_none_usage_yields_empty_turn(self):
        assert _parse_span_usage(None) == UsageBucket(turns=1)

    def test_list_usage_yields_empty_turn(self):
        # Some providers have returned `usage: []` on streaming completion.
        assert _parse_span_usage([]) == UsageBucket(turns=1)  # type: ignore[arg-type]

    def test_non_numeric_input_tokens_yields_zero(self):
        bucket = _parse_span_usage({"input_tokens": "not a number"})
        assert bucket == UsageBucket(turns=1)

    def test_non_dict_input_details_yields_zero_subbucket(self):
        bucket = _parse_span_usage({"input_tokens_details": "broken"})
        assert bucket == UsageBucket(turns=1)

    def test_well_formed_usage_passes_through(self):
        bucket = _parse_span_usage({"input_tokens": 100, "output_tokens": 20})
        assert bucket.input_tokens == 100
        assert bucket.output_tokens == 20
        assert bucket.cached_input_tokens == 0
        assert bucket.reasoning_output_tokens == 0
        assert bucket.turns == 1


# ── Codex fix: Chat-Completions key fallback ───────────────────────────


class TestPromptCompletionFallback:
    """The SDK's GenerationSpanData.export()['usage'] can carry either the
    new Responses-API key naming (`input_tokens`/`output_tokens`) or the
    legacy Chat-Completions key naming (`prompt_tokens`/`completion_tokens`),
    depending on SDK version + provider. `_parse_span_usage` must read both
    shapes, preferring the new names when present."""

    def test_chat_completions_shape_only(self):
        # Legacy key naming — the only shape present. Falls back cleanly.
        bucket = _parse_span_usage(
            {"prompt_tokens": 1024, "completion_tokens": 128}
        )
        assert bucket == UsageBucket(turns=1, input_tokens=1024, output_tokens=128)

    def test_both_shapes_new_wins(self):
        # Defensive: a mixed/buggy provider could emit both shapes. The
        # Responses-API names take precedence — legacy is a fallback only.
        bucket = _parse_span_usage(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "prompt_tokens": 999,
                "completion_tokens": 999,
            }
        )
        assert bucket == UsageBucket(turns=1, input_tokens=100, output_tokens=20)

    def test_new_shape_with_sub_buckets_still_parsed(self):
        # Sub-bucket parsing must not regress when the fallback path exists.
        bucket = _parse_span_usage(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 30},
            }
        )
        assert bucket.input_tokens == 100
        assert bucket.output_tokens == 20
        assert bucket.cached_input_tokens == 30
        assert bucket.reasoning_output_tokens == 0
        assert bucket.turns == 1

    def test_legacy_shape_with_zero_values(self):
        # Fallback must not invent counts — zero stays zero.
        bucket = _parse_span_usage(
            {"prompt_tokens": 0, "completion_tokens": 0}
        )
        assert bucket == UsageBucket(turns=1)

    def test_mixed_presence_only_prompt_tokens(self):
        # Only `prompt_tokens` is present; `completion_tokens` is absent
        # AND the new `output_tokens` is absent — output stays 0 because
        # there's nothing to fall back to.
        bucket = _parse_span_usage({"prompt_tokens": 500})
        assert bucket == UsageBucket(turns=1, input_tokens=500)


# ── AgentLabel enum ────────────────────────────────────────────────────


class TestAgentLabel:
    """``AgentLabel`` is a ``str``-subclass enum so producer-side code
    (trace metadata writers) can use typed members while consumer-side
    code (accumulator + render + sidecar) keeps reading plain strings.
    Backward-compatibility is the load-bearing contract — verify the
    str / hash interchange explicitly, and pin the canonical order
    derivation."""

    def test_str_equality_with_bare_string(self):
        # str-subclass: AgentLabel.CODER compares == "coder".
        assert AgentLabel.CODER == "coder"
        assert AgentLabel.PLANNER == "planner"
        assert AgentLabel.CODER_TRANSLATE == "coder-translate"
        assert AgentLabel.REVIEWER == "reviewer"

    def test_hash_matches_bare_string(self):
        # Dict-key interchangeability: a dict keyed with the bare string
        # must accept an AgentLabel lookup, and vice versa.
        assert hash(AgentLabel.CODER) == hash("coder")
        d = {"coder": 42}
        assert d[AgentLabel.CODER] == 42
        d2 = {AgentLabel.PLANNER: 7}
        assert d2["planner"] == 7

    def test_all_four_members_exist_with_right_values(self):
        # Exact member set + value strings. The hyphen in CODER_TRANSLATE
        # is load-bearing — `_CANONICAL_AGENT_ORDER` + rendered tables
        # depend on the "coder-translate" string.
        assert AgentLabel.PLANNER.value == "planner"
        assert AgentLabel.CODER.value == "coder"
        assert AgentLabel.CODER_TRANSLATE.value == "coder-translate"
        assert AgentLabel.REVIEWER.value == "reviewer"
        assert AgentLabel.SUMMARIZER.value == "summarizer"
        assert {m.name for m in AgentLabel} == {
            "PLANNER", "CODER", "CODER_TRANSLATE", "REVIEWER", "SUMMARIZER",
        }

    def test_canonical_order_derived_from_enum(self):
        # The constant must be the enum's iteration order in value form.
        # Keeps a single source of truth for column ordering.
        assert _CANONICAL_AGENT_ORDER == tuple(m.value for m in AgentLabel)
        assert _CANONICAL_AGENT_ORDER == (
            "planner", "coder", "coder-translate", "reviewer", "summarizer",
        )

    def test_value_attribute_is_plain_string(self):
        # `.value` is the plain string form trace-metadata writers feed
        # to the SDK — `_coerce_agent_label` in `sdk_trace.py` reads
        # this rather than `str(member)` because Python 3.10's stdlib
        # `Enum.__str__` returns "AgentLabel.X" instead of the value.
        assert AgentLabel.CODER_TRANSLATE.value == "coder-translate"
        assert AgentLabel.REVIEWER.value == "reviewer"

    def test_str_concat_yields_value(self):
        # Str-subclass behavior: concatenating with "" forces the str
        # base class's coercion, yielding the underlying value. This is
        # the property that keeps AgentLabel members slot into
        # dict[str, ...] keys + JSON-encoded contexts transparently.
        assert AgentLabel.CODER + "" == "coder"
        assert "" + AgentLabel.PLANNER == "planner"
