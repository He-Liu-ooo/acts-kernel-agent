"""Focused tests for ``src/search/tree.py`` fields + checkpoint round-trip.

Broader SearchTree behavior (frontier / best_node / serialize) lives in
``tests/test_search.py``; this file collects the targeted assertions for
the ``iter_no`` field added by the search-tree-recording feature.
"""
from __future__ import annotations


def test_iter_no_threaded_through_add_child():
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    root_kernel = Kernel(spec=spec, source_code="")
    child_kernel = Kernel(spec=spec, source_code="")
    tree = SearchTree()
    root = tree.add_root(root_kernel)
    child = tree.add_child(root.id, child_kernel, "tiling", iter_no=3)
    assert child.iter_no == 3
    assert root.iter_no == -1  # root default


def test_add_child_persists_action_params():
    """``add_child`` accepts an ``action_params`` kwarg and stores it on the
    new node so sibling rendering can distinguish e.g. BLOCK_N=32 from 16."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=spec, source_code=""))
    child = tree.add_child(
        root.id,
        Kernel(spec=spec, source_code=""),
        action_applied="t1_block_size_tuning",
        action_params={"BLOCK_N": 32},
        iter_no=1,
    )
    assert child.action_params == {"BLOCK_N": 32}
    # Root has no action_params (no action produced it).
    assert root.action_params is None


def test_add_child_action_params_defaults_to_none():
    """When ``add_child`` is called without ``action_params``, the field is
    ``None`` — keeps legacy call sites (and root) interpretable."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=spec, source_code=""))
    child = tree.add_child(
        root.id,
        Kernel(spec=spec, source_code=""),
        action_applied="t3_tf32",
        iter_no=2,
    )
    assert child.action_params is None


def _make_score(sol_score: float):
    """Helper: minimal score stub carrying just the field render_siblings reads.

    Avoids importing ``src.eval.scorer`` because its module-top
    ``import sol_execbench.sol_score`` makes the real ScoreResult
    Tier-2-only — the render_siblings code only touches ``.sol_score``.
    """
    from dataclasses import dataclass
    @dataclass
    class _StubScore:
        sol_score: float
    return _StubScore(sol_score=sol_score)


def _make_sibling(tree, parent_id, action, params, sol_score, iter_no):
    """Helper: add a child with score + last_review + branch_quality so the
    sibling-rendering tests can drive every formatting branch."""
    from src.agents.reviewer import BranchQuality, ReviewerFeedback
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    child = tree.add_child(
        parent_id,
        Kernel(spec=spec, source_code=""),
        action_applied=action,
        iter_no=iter_no,
        action_params=params,
    )
    child.score = _make_score(sol_score)
    child.branch_quality = BranchQuality.BLOCKED_POTENTIAL
    child.last_review = ReviewerFeedback(
        outcome="regressed" if sol_score < 0.5 else "neutral",
    )
    return child


def test_render_siblings_empty_when_no_children():
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=spec, source_code=""))
    assert tree.render_siblings(root.id) == ""


def test_render_siblings_one_child_no_exclude():
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=spec, source_code=""))
    root.score = _make_score(0.5)
    _make_sibling(tree, root.id, "t1_block_size_tuning", {"BLOCK_N": 32}, 0.43, 1)

    out = tree.render_siblings(root.id)
    assert "t1_block_size_tuning" in out
    assert "BLOCK_N" in out and "32" in out
    assert "0.430" in out
    # Delta is negative (regressed from parent 0.5):
    assert "-0.07" in out
    assert "regressed" in out
    assert "blocked_potential" in out


def test_render_siblings_excludes_specified_id():
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=spec, source_code=""))
    root.score = _make_score(0.5)
    c1 = _make_sibling(tree, root.id, "t1_block_size_tuning", {"BLOCK_N": 32}, 0.43, 1)
    c2 = _make_sibling(tree, root.id, "t3_tf32", None, 0.51, 2)

    out_excl_c1 = tree.render_siblings(root.id, exclude_id=c1.id)
    assert "t1_block_size_tuning" not in out_excl_c1
    assert "t3_tf32" in out_excl_c1

    out_excl_c2 = tree.render_siblings(root.id, exclude_id=c2.id)
    assert "t3_tf32" not in out_excl_c2
    assert "t1_block_size_tuning" in out_excl_c2


def test_render_siblings_sentinels_on_missing_fields():
    """Still-scoring sibling renders with sentinels rather than being skipped."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=spec, source_code=""))
    # Child without score, without last_review, without branch_quality
    tree.add_child(
        root.id,
        Kernel(spec=spec, source_code=""),
        action_applied="t3_tf32",
        iter_no=1,
    )
    out = tree.render_siblings(root.id)
    assert "t3_tf32" in out
    assert "SOL n/a" in out
    assert "no review yet" in out
    assert "(unscored)" in out


def test_failure_detail_dataclass_fields():
    """FailureDetail carries (candidate_idx, reason, has_kernel_source)."""
    from src.search.tree import FailureDetail
    fd = FailureDetail(candidate_idx=2, reason="boom", has_kernel_source=True)
    assert fd.candidate_idx == 2
    assert fd.reason == "boom"
    assert fd.has_kernel_source is True


def test_treenode_failure_details_default_none_failure_reason_removed():
    """Live nodes have failure_details=None; legacy failure_reason field is gone."""
    from src.search.tree import TreeNode
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    node = TreeNode(id=0, kernel=Kernel(spec=spec, source_code=""))
    assert node.failure_details is None
    # legacy field removed; surfacing as AttributeError when accessed
    assert "failure_reason" not in TreeNode.__dataclass_fields__


def _make_baseline_tree():
    """Tier-1-safe fixture: tree with root only. Mirrors the local helpers."""
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    baseline = Kernel(spec=spec, source_code="")
    tree = SearchTree()
    tree.add_root(baseline)
    return tree, spec


def test_add_failure_summary_happy_path():
    from src.search.tree import FailureDetail
    from src.runtime.events import DeadReason
    from src.agents.reviewer import BranchQuality
    tree, _ = _make_baseline_tree()
    details = [
        FailureDetail(candidate_idx=0, reason="boom0", has_kernel_source=True),
        FailureDetail(candidate_idx=1, reason="boom1", has_kernel_source=False),
        FailureDetail(candidate_idx=2, reason="boom2", has_kernel_source=True),
    ]
    fn = tree.add_failure_summary(
        parent_id=0,
        action_applied="t1_block_size_tuning",
        action_params={"BLOCK_M": 64},
        iter_no=7,
        failure_details=details,
    )
    assert fn.kernel is None
    assert fn.score is None
    assert fn.branch_quality == BranchQuality.DEAD_END
    assert fn.dead_reason == DeadReason.CODER_FAILED
    assert fn.iter_no == 7
    assert fn.action_applied == "t1_block_size_tuning"
    assert fn.action_params == {"BLOCK_M": 64}
    assert len(fn.failure_details) == 3
    assert fn.failure_details[0].reason == "boom0"
    assert fn.parent_id == 0
    assert fn.id in tree.get_node(0).children_ids


def test_add_failure_summary_frontier_excludes_it():
    from src.search.tree import FailureDetail
    tree, _ = _make_baseline_tree()
    tree.add_failure_summary(
        parent_id=0,
        action_applied="t1_block_size_tuning",
        action_params=None,
        iter_no=1,
        failure_details=[FailureDetail(0, "boom", False)],
    )
    frontier_ids = [n.id for n in tree.frontier()]
    assert frontier_ids == [0]  # root only; summary excluded as DEAD_END


def test_add_failure_summary_rejects_empty_details():
    import pytest
    tree, _ = _make_baseline_tree()
    with pytest.raises(ValueError, match="failure_details must be non-empty"):
        tree.add_failure_summary(
            parent_id=0,
            action_applied="t1_block_size_tuning",
            action_params=None,
            iter_no=1,
            failure_details=[],
        )


def test_add_failure_child_removed():
    """Per-candidate add_failure_child API is removed; only add_failure_summary remains."""
    tree, _ = _make_baseline_tree()
    assert not hasattr(tree, "add_failure_child")


def test_render_siblings_flattens_summary_same_reason():
    """Summary node with 4 same-reason details renders as `FAILED ×4` —
    byte-identical to the old K-separate-nodes output for the common case."""
    from src.search.tree import FailureDetail
    tree, _ = _make_baseline_tree()
    tree.get_node(0).score = _make_score(0.5)
    details = [
        FailureDetail(i, "RewardHackDetected: torch.cuda.Event", True)
        for i in range(4)
    ]
    tree.add_failure_summary(
        parent_id=0,
        action_applied="t2_register_caching",
        action_params={"BLOCK_M": 64},
        iter_no=5,
        failure_details=details,
    )
    out = tree.render_siblings(parent_id=0, consumer="planner")
    assert "t2_register_caching" in out
    assert "×4" in out
    assert "RewardHackDetected: torch.cuda.Event" in out


def test_render_siblings_summary_distinct_reasons_renders_each():
    """Summary with K distinct reasons renders K separate FAILED lines."""
    from src.search.tree import FailureDetail
    tree, _ = _make_baseline_tree()
    tree.get_node(0).score = _make_score(0.5)
    details = [
        FailureDetail(0, "ImplementationError: turn exhaust", False),
        FailureDetail(1, "EntrypointBinding: name mismatch", True),
        FailureDetail(2, "BenchmarkError: OOM", True),
        FailureDetail(3, "CUDA sticky-state", True),
    ]
    tree.add_failure_summary(
        parent_id=0,
        action_applied="t1_block_size_tuning",
        action_params=None,
        iter_no=3,
        failure_details=details,
    )
    out = tree.render_siblings(parent_id=0, consumer="planner")
    for reason in [
        "ImplementationError: turn exhaust",
        "EntrypointBinding: name mismatch",
        "BenchmarkError: OOM",
        "CUDA sticky-state",
    ]:
        assert reason in out
    assert out.count("FAILED") == 4


def test_render_siblings_mixed_winner_and_summary():
    """Winner sibling in success block; summary failures in FAILED block."""
    from src.search.tree import FailureDetail
    tree, _ = _make_baseline_tree()
    tree.get_node(0).score = _make_score(0.5)
    _make_sibling(tree, 0, "t1_block_size_tuning", None, 0.6, 2)
    tree.add_failure_summary(
        parent_id=0,
        action_applied="t1_block_size_tuning",
        action_params={"BLOCK_M": 32},
        iter_no=2,
        failure_details=[FailureDetail(0, "boom", True), FailureDetail(1, "boom", True)],
    )
    out = tree.render_siblings(parent_id=0, consumer="planner")
    assert "SOL 0.600" in out
    assert "×2" in out
    # Success block precedes FAILED block.
    assert out.index("SOL 0.600") < out.index("FAILED")


def test_load_legacy_checkpoint_synthesizes_failure_details():
    """Pre-collapse checkpoints with ``failure_reason: str`` load with
    failure_details synthesized as ``[FailureDetail(0, <legacy>, False)]``.

    ``has_kernel_source=False`` unconditionally: legacy on-disk layout
    has ``kernel.py`` at the flat path, not at ``cand_0/kernel.py``.
    """
    import pytest
    pytest.importorskip("sol_execbench", reason="checkpoint load imports ScoreResult chain")
    from pathlib import Path
    from src.search.tree import SearchTree
    fixture = Path(__file__).parent / "fixtures" / "legacy_checkpoint_failure_nodes.json"
    tree = SearchTree.load(fixture)
    # Root: no failure
    assert tree.get_node(0).failure_details is None
    # Node 1: legacy failure_reason with kernel attached
    n1 = tree.get_node(1)
    assert n1.failure_details is not None
    assert len(n1.failure_details) == 1
    fd1 = n1.failure_details[0]
    assert fd1.candidate_idx == 0
    assert fd1.reason == "BenchmarkError: OOM on workload uuid-A"
    assert fd1.has_kernel_source is False  # legacy: see synth comment
    # Node 2: legacy turn-exhaust (kernel=None)
    n2 = tree.get_node(2)
    assert n2.failure_details is not None
    assert n2.failure_details[0].reason == "ImplementationError: Coder exhausted turns"
    assert n2.failure_details[0].has_kernel_source is False


def test_round_trip_new_format_failure_details(tmp_path):
    """Save → load round-trip preserves the new failure_details list shape."""
    import pytest
    pytest.importorskip("sol_execbench", reason="checkpoint load imports ScoreResult chain")
    from src.search.tree import FailureDetail, SearchTree
    tree, _ = _make_baseline_tree()
    tree.add_failure_summary(
        parent_id=0,
        action_applied="t1_block_size_tuning",
        action_params={"BLOCK_M": 64},
        iter_no=3,
        failure_details=[
            FailureDetail(0, "boom0", True),
            FailureDetail(1, "boom1", False),
        ],
    )
    ckpt = tmp_path / "ckpt.json"
    tree.save(ckpt)
    loaded = SearchTree.load(ckpt)
    fd_list = loaded.get_node(1).failure_details
    assert len(fd_list) == 2
    assert fd_list[0].candidate_idx == 0 and fd_list[0].reason == "boom0" and fd_list[0].has_kernel_source is True
    assert fd_list[1].candidate_idx == 1 and fd_list[1].reason == "boom1" and fd_list[1].has_kernel_source is False


def test_regressed_sibling_actions_skips_summary_nodes():
    """Summary nodes have score=None and are naturally excluded."""
    from src.search.tree import FailureDetail
    tree, _ = _make_baseline_tree()
    tree.get_node(0).score = _make_score(0.5)
    tree.add_failure_summary(
        parent_id=0,
        action_applied="t1_block_size_tuning",
        action_params=None,
        iter_no=1,
        failure_details=[FailureDetail(0, "boom", True)],
    )
    assert tree.regressed_sibling_actions(parent_id=0) == []


def test_legacy_checkpoint_load_defaults_iter_no(tmp_path):
    """Pre-iter_no checkpoints round-trip with iter_no = -1."""
    import json
    from src.search.tree import SearchTree
    legacy = {
        "next_id": 1,
        "nodes": {
            "0": {
                "id": 0, "parent_id": None, "children_ids": [],
                "action_applied": "", "depth": 0,
                "branch_quality": None, "score": None,
                "kernel": {
                    "spec": {"name": "t", "kernel_type": "elementwise",
                             "flop_count": 0, "memory_bytes": 0,
                             "input_shapes": [], "definition_path": None,
                             "pytorch_reference": "", "t_sol_us": 1.0},
                    "source_code": "", "num_warps": 4, "num_stages": 2,
                    "block_size": 128, "triton_kernel_name": "", "dps": False,
                },
                "profiling": None, "per_workload_latency_us": None,
                "consecutive_agent_failures": 0,
                # iter_no intentionally missing
            },
        },
    }
    p = tmp_path / "tree.json"
    p.write_text(json.dumps(legacy))
    tree = SearchTree.load(p)
    assert tree.get_node(0).iter_no == -1
