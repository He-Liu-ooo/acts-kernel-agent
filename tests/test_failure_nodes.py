"""Tier-1 unit tests for failure-node collapse.

Covers: DeadReason enum, TreeNode.failure_details list field,
SearchTree.add_failure_summary, render_siblings failure rendering with
dedup/ordering/cap (still byte-identical when reasons match across
candidates), consumer split, backward-compat checkpoint load synthesis.

Spec: doc/specs/2026-05-18-failure-node-collapse-design.md
Plan: doc/plans/2026-05-18-failure-node-collapse-plan.md
"""
from __future__ import annotations


def test_dead_reason_coder_failed_member_exists():
    """``DeadReason.CODER_FAILED`` joins the enum with stable string value."""
    from src.runtime.events import DeadReason
    assert DeadReason.CODER_FAILED.value == "coder_failed"
    # Member is a string subclass (per the DeadReason docstring) — JSON-
    # serializes as its value so events.jsonl + checkpoint stay consistent.
    assert isinstance(DeadReason.CODER_FAILED, str)
    assert DeadReason.CODER_FAILED == "coder_failed"


def test_failure_summary_added_event_kind_registered():
    """``failure_summary_added`` is a known event kind so emit() won't warn."""
    from src.runtime.events import CORE_EVENT_KINDS
    assert "failure_summary_added" in CORE_EVENT_KINDS


# ── Task 2: TreeNode.failure_reason field + serialization ────────────────────

def _spec():
    """Module-local minimal KernelSpec for failure-node tests."""
    from src.kernels.kernel import KernelSpec, KernelType
    return KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)


def test_tree_node_failure_details_defaults_to_none():
    """``failure_details`` field defaults to None on success / live nodes."""
    from src.search.tree import TreeNode
    from src.kernels.kernel import Kernel
    node = TreeNode(id=0, kernel=Kernel(spec=_spec(), source_code=""))
    assert node.failure_details is None


def test_tree_node_failure_details_round_trips_through_serialize():
    """A summary node carrying ``failure_details`` survives serialize→deserialize."""
    import pytest
    pytest.importorskip("sol_execbench", reason="checkpoint load imports ScoreResult chain")
    import tempfile
    from pathlib import Path
    from src.search.tree import FailureDetail, SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    fn = tree.add_failure_summary(
        parent_id=root.id,
        action_applied="t1_block_size_tuning",
        action_params={"BLOCK_K": 32},
        iter_no=3,
        failure_details=[
            FailureDetail(0, "autotune burn-in failed: cudaErrorInvalidAddressSpace", True),
            FailureDetail(1, "Coder exhausted turn budget", False),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt.json"
        tree.save(ckpt)
        loaded = SearchTree.load(ckpt)

    reloaded = loaded.get_node(fn.id)
    assert reloaded.failure_details is not None
    assert len(reloaded.failure_details) == 2
    assert reloaded.failure_details[0].reason == "autotune burn-in failed: cudaErrorInvalidAddressSpace"
    assert reloaded.failure_details[0].has_kernel_source is True
    assert reloaded.failure_details[1].has_kernel_source is False


def test_legacy_checkpoint_failure_reason_synthesizes_failure_details():
    """Pre-collapse checkpoint with ``failure_reason: str`` synthesizes a
    1-element ``failure_details`` list on load.

    ``has_kernel_source=False`` unconditionally on legacy synthesis: the
    legacy on-disk layout has ``kernel.py`` at the flat path
    ``tree/node_<id>/kernel.py``, not at ``cand_0/kernel.py``; the flag
    must not lie about the new-layout presence.
    """
    import pytest
    pytest.importorskip("sol_execbench", reason="checkpoint load imports ScoreResult chain")
    import json
    import tempfile
    from pathlib import Path
    from src.search.tree import SearchTree
    legacy = {
        "next_id": 1,
        "nodes": {
            "0": {
                "id": 0, "parent_id": None, "children_ids": [],
                "kernel": {"spec": {"name": "t", "kernel_type": "elementwise",
                                    "flop_count": 0, "memory_bytes": 0,
                                    "input_shapes": [],
                                    "definition_path": None,
                                    "pytorch_reference": "", "t_sol_us": 1.0},
                           "source_code": "# crashed",
                           "triton_kernel_name": "",
                           "dps": False,
                           "autotune_configs": [],
                           "autotune_keys": [],
                           "autotune_winner": {}},
                "score": None, "branch_quality": "dead_end",
                "action_applied": "t1_block_size_tuning",
                "action_params": {"BLOCK_K": 32},
                "depth": 1, "profiling": None,
                "per_workload_latency_us": None,
                "consecutive_agent_failures": 0,
                "iter_no": 5, "last_review": None,
                "dead_reason": "coder_failed",
                "failure_reason": "autotune burn-in failed: cudaErrorInvalidAddressSpace",
            },
        },
    }
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt.json"
        ckpt.write_text(json.dumps(legacy))
        tree = SearchTree.load(ckpt)
    n = tree.get_node(0)
    assert n.failure_details is not None
    assert len(n.failure_details) == 1
    assert n.failure_details[0].candidate_idx == 0
    assert n.failure_details[0].reason == "autotune burn-in failed: cudaErrorInvalidAddressSpace"
    assert n.failure_details[0].has_kernel_source is False


# ── Task 3: ACTSConfig.failure_sibling_cap field ──────────────────────────────

def test_failure_sibling_cap_defaults_to_8():
    """New ACTSConfig field defaults to 8."""
    from src.config import ACTSConfig
    cfg = ACTSConfig()
    assert cfg.failure_sibling_cap == 8


def test_failure_sibling_cap_zero_means_uncapped():
    """A cap of 0 is the documented uncapped sentinel."""
    from src.config import ACTSConfig
    cfg = ACTSConfig(failure_sibling_cap=0)
    assert cfg.failure_sibling_cap == 0


# ── Task 4: add_failure_child + frontier/best_node exclusion ──────────────────

def test_add_failure_summary_field_assignments():
    """``add_failure_summary`` produces a node with the spec'd field shape."""
    from src.runtime.events import DeadReason
    from src.agents.reviewer import BranchQuality
    from src.search.tree import FailureDetail, SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    failure = tree.add_failure_summary(
        parent_id=root.id,
        action_applied="t1_block_size_tuning",
        action_params={"BLOCK_K": 32},
        iter_no=6,
        failure_details=[
            FailureDetail(0, "autotune burn-in failed: cudaErrorInvalidAddressSpace", True),
        ],
    )
    assert failure.score is None
    assert failure.last_review is None
    assert failure.branch_quality == BranchQuality.DEAD_END
    assert failure.dead_reason == DeadReason.CODER_FAILED
    assert failure.children_ids == []
    # Summary nodes always have kernel=None (per-candidate kernels live
    # on disk under tree/node_<id>/cand_<idx>/).
    assert failure.kernel is None
    assert failure.failure_details is not None
    assert len(failure.failure_details) == 1
    assert failure.failure_details[0].reason == "autotune burn-in failed: cudaErrorInvalidAddressSpace"
    assert failure.failure_details[0].has_kernel_source is True
    assert failure.action_applied == "t1_block_size_tuning"
    assert failure.action_params == {"BLOCK_K": 32}
    assert failure.parent_id == root.id
    assert failure.depth == 1
    assert failure.iter_no == 6
    # Parent's children_ids is updated so sibling rendering finds it.
    assert failure.id in root.children_ids


def test_add_failure_summary_records_turn_exhaust_with_has_kernel_source_false():
    """Turn-exhaust path has no submitted kernel; FailureDetail.has_kernel_source
    records the bit so postmortem layout matches on-disk truth."""
    from src.search.tree import FailureDetail, SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    failure = tree.add_failure_summary(
        parent_id=root.id,
        action_applied="t3_loop_unroll",
        action_params=None,
        iter_no=7,
        failure_details=[
            FailureDetail(0, "Coder exhausted turn budget (8) without calling submit_kernel.", False),
        ],
    )
    assert failure.kernel is None
    assert failure.action_params is None
    assert failure.failure_details[0].has_kernel_source is False


def test_frontier_excludes_failure_summary_nodes():
    """Summary nodes have branch_quality == DEAD_END → ``frontier()`` skips."""
    from src.search.tree import FailureDetail, SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    tree.add_failure_summary(
        parent_id=root.id,
        action_applied="t1", action_params=None, iter_no=1,
        failure_details=[FailureDetail(0, "boom", False)],
    )
    frontier_ids = [n.id for n in tree.frontier()]
    # Root is expandable, summary is not.
    assert root.id in frontier_ids
    assert all(n.dead_reason is None for n in tree.frontier())


def test_best_node_excludes_failure_summary_nodes_even_when_only_dead_end():
    """``best_node()`` filters on ``_eligible_for_best``; CODER_FAILED is
    excluded so a summary-only tree falls back to root."""
    from src.search.tree import FailureDetail, SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    tree.add_failure_summary(
        parent_id=root.id,
        action_applied="t1", action_params=None, iter_no=1,
        failure_details=[FailureDetail(0, "boom", False)],
    )
    assert tree.best_node().id == root.id


# ── Task 5: render_siblings failure rendering ──────────────────────────────────

def _add_failure(tree, parent_id, action, params, reason, iter_no, cand_idx=0):
    """Helper: build a failure-summary node carrying ONE candidate detail.

    Migrated from the legacy per-candidate ``add_failure_child`` API to
    the collapse-era ``add_failure_summary`` API; the test contract
    (one ``_add_failure`` call ⇒ one failure rendered by
    ``render_siblings``) is preserved because the flatten-then-dedup
    path treats a 1-element ``failure_details`` list identically to a
    legacy single-failure node. Tests calling this K times to simulate
    K candidate failures end up with K summary nodes; the
    render-time dedup on ``(action, params, reason)`` collapses them to
    a single ``FAILED ×K`` line just as before."""
    from src.search.tree import FailureDetail
    return tree.add_failure_summary(
        parent_id=parent_id,
        action_applied=action,
        action_params=params,
        iter_no=iter_no,
        failure_details=[FailureDetail(cand_idx, reason, True)],
    )


def test_render_siblings_planner_includes_failure_lines():
    """Planner consumer sees FAILED lines after success lines."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    _add_failure(tree, root.id, "t1_block_size_tuning", {"BLOCK_K": 32},
                 "autotune burn-in failed: cudaErrorInvalidAddressSpace",
                 iter_no=6, cand_idx=0)
    rendered = tree.render_siblings(root.id, consumer="planner")
    assert "FAILED" in rendered
    assert "t1_block_size_tuning" in rendered
    assert "BLOCK_K:32" in rendered
    assert "cudaErrorInvalidAddressSpace" in rendered


def test_render_siblings_reviewer_omits_failure_lines():
    """Reviewer consumer keeps the success-only render — per spec."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    _add_failure(tree, root.id, "t1", {"BLOCK_K": 32}, "boom",
                 iter_no=1, cand_idx=0)
    rendered = tree.render_siblings(root.id, consumer="reviewer")
    assert "FAILED" not in rendered


def test_render_siblings_default_consumer_is_planner():
    """Existing call sites passing no consumer kwarg get the failure-
    inclusive render — soft-default to planner-style behavior."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    _add_failure(tree, root.id, "t1", None, "boom", iter_no=1, cand_idx=0)
    rendered = tree.render_siblings(root.id)
    assert "FAILED" in rendered


def test_render_siblings_dedup_collapses_identical_failure_tuples():
    """4 failures sharing (action, params, reason) render as 1 line + ×4."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    for cand_idx in range(4):
        _add_failure(tree, root.id, "t1", {"BLOCK_K": 32},
                     "autotune burn-in failed: cudaErrorInvalidAddressSpace",
                     iter_no=6, cand_idx=cand_idx)
    rendered = tree.render_siblings(root.id, consumer="planner")
    failure_lines = [l for l in rendered.splitlines() if "FAILED" in l]
    assert len(failure_lines) == 1
    assert "×4" in failure_lines[0]


def test_render_siblings_no_dedup_when_params_differ():
    """Same reason but different params → separate lines."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    _add_failure(tree, root.id, "t1", {"BLOCK_K": 32},
                 "autotune burn-in failed: cudaErrorInvalidAddressSpace",
                 iter_no=6, cand_idx=0)
    _add_failure(tree, root.id, "t1", {"BLOCK_K": 64},
                 "autotune burn-in failed: cudaErrorInvalidAddressSpace",
                 iter_no=7, cand_idx=0)
    rendered = tree.render_siblings(root.id, consumer="planner")
    failure_lines = [l for l in rendered.splitlines() if "FAILED" in l]
    assert len(failure_lines) == 2
    assert "BLOCK_K:32" in rendered and "BLOCK_K:64" in rendered


def test_render_siblings_no_dedup_when_reasons_differ():
    """Same (action, params) but different reasons → separate lines."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    _add_failure(tree, root.id, "t1", {"BLOCK_K": 32}, "reason A",
                 iter_no=6, cand_idx=0)
    _add_failure(tree, root.id, "t1", {"BLOCK_K": 32}, "reason B",
                 iter_no=6, cand_idx=1)
    rendered = tree.render_siblings(root.id, consumer="planner")
    failure_lines = [l for l in rendered.splitlines() if "FAILED" in l]
    assert len(failure_lines) == 2


def test_render_siblings_cap_truncates_keeping_most_recent():
    """failure_cap=2 with 5 distinct failure groups → keep latest 2."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    # Distinct dedup groups, with monotone iter_no for clean tail-N.
    for i in range(5):
        _add_failure(tree, root.id, "t1", {"i": i}, f"reason {i}",
                     iter_no=i + 1, cand_idx=0)
    rendered = tree.render_siblings(root.id, consumer="planner", failure_cap=2)
    failure_lines = [l for l in rendered.splitlines() if "FAILED" in l]
    assert len(failure_lines) == 2
    # Latest two groups should be reasons 3 and 4 (iter_no 4 and 5).
    assert "reason 4" in rendered
    assert "reason 3" in rendered
    assert "reason 0" not in rendered
    assert "3 earlier failures omitted" in rendered


def test_render_siblings_cap_keeps_recurring_group_by_latest_occurrence():
    """Cap-tail must order groups by *latest* occurrence, not first.

    Codex adversarial review (2026-05-17) flagged that first-occurrence
    ordering silently evicts a recurring signature when enough unique
    signatures appear between the first and latest occurrence — exactly
    the repeated pattern the Planner is supposed to react to.

    Setup: signature A appears at iter 1 + iter 10 (recurs latest);
    signatures B, C, D each appear once at iters 2-4. failure_cap=2 →
    must keep A (latest at iter 10) and D (latest at iter 4), not
    B and C (which would be kept under buggy first-seen ordering).
    """
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    # A first
    _add_failure(tree, root.id, "t1", {"BLOCK": "A"}, "reason A",
                 iter_no=1, cand_idx=0)
    # B, C, D — each unique
    _add_failure(tree, root.id, "t1", {"BLOCK": "B"}, "reason B",
                 iter_no=2, cand_idx=0)
    _add_failure(tree, root.id, "t1", {"BLOCK": "C"}, "reason C",
                 iter_no=3, cand_idx=0)
    _add_failure(tree, root.id, "t1", {"BLOCK": "D"}, "reason D",
                 iter_no=4, cand_idx=0)
    # A recurs latest
    _add_failure(tree, root.id, "t1", {"BLOCK": "A"}, "reason A",
                 iter_no=10, cand_idx=0)

    rendered = tree.render_siblings(root.id, consumer="planner", failure_cap=2)
    failure_lines = [l for l in rendered.splitlines() if "FAILED" in l]
    assert len(failure_lines) == 2
    # A must survive because its latest occurrence (iter 10) is newest.
    assert "reason A" in rendered
    # D must survive — latest occurrence iter 4, newer than B/C.
    assert "reason D" in rendered
    # B and C must be evicted (latest occurrence iter 2/3).
    assert "reason B" not in rendered
    assert "reason C" not in rendered
    # A's count must reflect both occurrences.
    a_line = next(l for l in failure_lines if "reason A" in l)
    assert "×2" in a_line


def test_render_siblings_cap_zero_means_uncapped():
    """failure_cap=0 renders every failure with no omission preface."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    for i in range(10):
        _add_failure(tree, root.id, "t1", {"i": i}, f"reason {i}",
                     iter_no=i, cand_idx=0)
    rendered = tree.render_siblings(root.id, consumer="planner", failure_cap=0)
    failure_lines = [l for l in rendered.splitlines() if "FAILED" in l]
    assert len(failure_lines) == 10
    assert "earlier failures omitted" not in rendered


def test_render_siblings_ordering_success_before_failure():
    """Success siblings render before failure siblings in the output."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel
    from src.agents.reviewer import BranchQuality, ReviewerFeedback

    tree = SearchTree()
    root = tree.add_root(Kernel(spec=_spec(), source_code=""))
    # One success sibling.
    success = tree.add_child(
        root.id, Kernel(spec=_spec(), source_code=""),
        action_applied="t1_block_size_tuning",
        action_params={"BLOCK_K": 64},
        iter_no=1,
    )

    from dataclasses import dataclass

    @dataclass
    class _StubScore:
        sol_score: float
    success.score = _StubScore(sol_score=0.55)
    success.branch_quality = BranchQuality.BLOCKED_POTENTIAL
    success.last_review = ReviewerFeedback(
        outcome="improved", bottleneck_classification=None,
        branch_quality=BranchQuality.BLOCKED_POTENTIAL,
        metric_deltas={}, bottleneck_diagnosis="", suggestions=[],
        conditional_assessment="",
    )
    root.score = _StubScore(sol_score=0.50)
    # One failure sibling.
    _add_failure(tree, root.id, "t1_occupancy", {"num_warps": 8}, "boom",
                 iter_no=2, cand_idx=0)
    rendered = tree.render_siblings(root.id, consumer="planner")
    idx_success = rendered.index("SOL 0.550")
    idx_failed = rendered.index("FAILED")
    assert idx_success < idx_failed


# ── Task 10: legacy checkpoint round-trip end-to-end ──────────────────────────

def test_legacy_checkpoint_round_trip_renders_success_only():
    """End-to-end: load a pre-feature checkpoint (success-only tree),
    render siblings → no FAILED lines for either consumer. Backward-compat
    regression guard for in-flight runs that started before this feature
    landed."""
    import pytest
    pytest.importorskip("sol_execbench", reason="checkpoint load imports ScoreResult chain")
    import json
    import tempfile
    from pathlib import Path
    from src.search.tree import SearchTree

    legacy = {
        "next_id": 2,
        "nodes": {
            "0": {
                "id": 0, "parent_id": None, "children_ids": [1],
                "kernel": {"spec": {"name": "t", "kernel_type": "elementwise",
                                    "flop_count": 0, "memory_bytes": 0,
                                    "input_shapes": [],
                                    "definition_path": None,
                                    "pytorch_reference": "", "t_sol_us": 1.0},
                           "source_code": "",
                           "triton_kernel_name": "",
                           "dps": False,
                           "autotune_configs": [],
                           "autotune_keys": [],
                           "autotune_winner": {}},
                "score": {"sol_score": 0.50, "baseline_latency_us": 100.0,
                          "candidate_latency_us": 100.0, "t_sol_us": 50.0,
                          "speedup": 1.0, "reward_hack_suspect": False,
                          "calibration_warning": False},
                "branch_quality": None,
                "action_applied": "", "action_params": None,
                "depth": 0, "profiling": None,
                "per_workload_latency_us": None,
                "consecutive_agent_failures": 0,
                "iter_no": -1, "last_review": None, "dead_reason": None,
            },
            "1": {
                "id": 1, "parent_id": 0, "children_ids": [],
                "kernel": {"spec": {"name": "t", "kernel_type": "elementwise",
                                    "flop_count": 0, "memory_bytes": 0,
                                    "input_shapes": [],
                                    "definition_path": None,
                                    "pytorch_reference": "", "t_sol_us": 1.0},
                           "source_code": "",
                           "triton_kernel_name": "",
                           "dps": False,
                           "autotune_configs": [],
                           "autotune_keys": [],
                           "autotune_winner": {}},
                "score": {"sol_score": 0.55, "baseline_latency_us": 100.0,
                          "candidate_latency_us": 90.0, "t_sol_us": 50.0,
                          "speedup": 1.11, "reward_hack_suspect": False,
                          "calibration_warning": False},
                "branch_quality": "blocked_potential",
                "action_applied": "t1_block_size_tuning",
                "action_params": {"BLOCK_K": 64},
                "depth": 1, "profiling": None,
                "per_workload_latency_us": None,
                "consecutive_agent_failures": 0,
                "iter_no": 1,
                "last_review": {
                    "outcome": "improved",
                    "bottleneck_classification": None,
                    "branch_quality": "blocked_potential",
                    "metric_deltas": {},
                    "bottleneck_diagnosis": "",
                    "suggestions": [],
                    "conditional_assessment": "",
                    "degraded": False,
                    "error_reason": "",
                },
                "dead_reason": None,
                # NOTE: no failure_reason key — legacy on-disk shape.
            },
        },
    }
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt.json"
        ckpt.write_text(json.dumps(legacy))
        tree = SearchTree.load(ckpt)
    rendered_planner = tree.render_siblings(0, consumer="planner")
    assert "SOL 0.550" in rendered_planner
    assert "FAILED" not in rendered_planner
    rendered_reviewer = tree.render_siblings(0, consumer="reviewer")
    assert "FAILED" not in rendered_reviewer
    # Legacy success children have no failure_details set.
    assert tree.get_node(1).failure_details is None
