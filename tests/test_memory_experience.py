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
