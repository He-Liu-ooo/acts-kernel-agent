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
