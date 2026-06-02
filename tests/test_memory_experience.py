"""Tests for the redesigned Experience dataclass (F2 schema)."""

from __future__ import annotations

from src.memory.experience import ActionRecord, Experience


def test_experience_has_required_fields():
    e = Experience(
        row_id="r_abc123",
        schema_version=1,
        kernel_type="matmul",
        hardware_arch="RTX6000Ada",
        scope="edge",
        speedup=1.96,
        action_applied=ActionRecord(action_id="a1", tier=3, name="vectorize"),
        title="Removing double buffering",
        lesson="The slow kernel double-buffered unnecessarily...",
        snippet_before="acc = tl.zeros(...)",
        snippet_after="acc = tl.zeros(...)\n# simplified",
        provenance={
            "run_id": "run-1",
            "parent_node_id": "n0",
            "child_node_id": "n1",
            "summarizer_model": "deepseek-chat",
        },
        created_at="2026-05-24T15:30:00Z",
    )
    assert e.scope in ("edge", "run")
    assert e.schema_version == 1
    assert e.speedup > 1.0
    assert e.action_applied.tier == 3


def test_action_record_kept_verbatim_from_v1():
    a = ActionRecord(action_id="a1", tier=2, name="tile")
    assert a.parameters == {}
    a2 = ActionRecord(
        action_id="a2", tier=2, name="tile", parameters={"BLOCK_M": "64"}
    )
    assert a2.parameters == {"BLOCK_M": "64"}


def test_experience_scope_literal_is_edge_or_run():
    e = Experience(
        row_id="r1",
        schema_version=1,
        kernel_type="matmul",
        hardware_arch="RTX6000Ada",
        scope="run",
        speedup=3.2,
        # ``scope == "run"`` rows carry ``action_applied=None`` — no
        # single action was applied to produce the cumulative trajectory.
        action_applied=None,
        title="...",
        lesson="...",
        snippet_before="...",
        snippet_after="...",
        provenance={},
        created_at="",
    )
    assert e.scope == "run"
    assert e.action_applied is None


def test_run_scope_action_applied_invariant():
    """Regression: ``scope == "run"`` ⇒ ``action_applied is None``.
    The producer enforces this at write time; the dataclass tolerates
    any value (Optional types are not runtime-enforced), but downstream
    consumers branching on ``action_applied is None`` rely on the
    contract holding for rows the producer actually writes."""
    from src.memory.experience import Experience

    edge = Experience(
        row_id="r_edge", schema_version=1, kernel_type="matmul",
        hardware_arch="RTX6000Ada", scope="edge", speedup=1.5,
        action_applied=ActionRecord(action_id="a1", tier=2, name="tile"),
        title="t", lesson="l", snippet_before="a", snippet_after="b",
        provenance={}, created_at="",
    )
    assert edge.action_applied is not None
    run_ = Experience(
        row_id="r_run", schema_version=1, kernel_type="matmul",
        hardware_arch="RTX6000Ada", scope="run", speedup=3.0,
        action_applied=None,
        title="t", lesson="l", snippet_before="a", snippet_after="b",
        provenance={}, created_at="",
    )
    assert run_.action_applied is None


from src.memory.experience import Experience, ActionRecord, dedup_key, dedup_best


def _exp(*, scope="edge", action_id="t1_grid_shape", condition="compute_bound",
         speedup=1.2, created_at="2026-06-02T00:00:00+00:00", row_id="r0") -> Experience:
    action = None if scope == "run" else ActionRecord(
        action_id=action_id, tier=1, name=action_id, parameters={})
    return Experience(
        row_id=row_id, schema_version=1, kernel_type="matmul", hardware_arch="RTX6000Ada",
        scope=scope, speedup=speedup, action_applied=action, title="t", lesson="l",
        snippet_before="", snippet_after="", provenance={}, created_at=created_at,
        condition=condition,
    )


def test_condition_field_defaults_empty():
    e = Experience(
        row_id="r", schema_version=1, kernel_type="m", hardware_arch="a", scope="edge",
        speedup=1.1, action_applied=None, title="t", lesson="l",
        snippet_before="", snippet_after="")
    assert e.condition == ""


def test_dedup_key_edge_uses_action_id_and_condition():
    e = _exp(action_id="t1_grid_shape", condition="compute_bound")
    assert dedup_key(e) == ("matmul", "RTX6000Ada", "edge", "t1_grid_shape", "compute_bound")


def test_dedup_key_run_uses_sentinel_for_missing_action():
    e = _exp(scope="run", condition="compute_bound")
    assert dedup_key(e) == ("matmul", "RTX6000Ada", "run", "∅", "compute_bound")


def test_dedup_best_keeps_highest_speedup():
    lo = _exp(speedup=1.2, row_id="lo")
    hi = _exp(speedup=1.5, row_id="hi")
    out = dedup_best([lo, hi])
    assert len(out) == 1 and out[0].row_id == "hi"


def test_dedup_best_tie_breaks_on_recency():
    old = _exp(speedup=1.5, created_at="2026-06-01T00:00:00+00:00", row_id="old")
    new = _exp(speedup=1.5, created_at="2026-06-02T00:00:00+00:00", row_id="new")
    out = dedup_best([old, new])
    assert len(out) == 1 and out[0].row_id == "new"


def test_dedup_best_preserves_distinct_conditions():
    a = _exp(condition="compute_bound", row_id="a")
    b = _exp(condition="memory_bound", row_id="b")
    out = dedup_best([a, b])
    assert {e.row_id for e in out} == {"a", "b"}


# --- condition formatting (hosted here, re-exported from producer) --------

from src.memory.experience import _format_condition


def test_format_condition_bottleneck_and_params():
    a = ActionRecord("t1_grid_shape", 1, "t1_grid_shape", {"BLOCK_N": "32"})
    assert _format_condition("compute_bound", a) == "compute_bound | BLOCK_N=32"


def test_format_condition_bottleneck_only_when_action_none():
    assert _format_condition("compute_bound", None) == "compute_bound"


def test_format_condition_sorts_params():
    a = ActionRecord("t", 1, "t", {"b": "2", "a": "1"})
    assert _format_condition("memory_bound", a) == "memory_bound | a=1, b=2"
