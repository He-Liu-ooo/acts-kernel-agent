"""Tests for src/runtime/tree_dump.py."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_bind_creates_dir(tmp_path):
    from src.runtime import tree_dump
    target = tmp_path / "tree"
    tree_dump.bind(target)
    try:
        assert target.is_dir()
    finally:
        tree_dump.unbind()


def test_unbind_clears_state(tmp_path):
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    tree_dump.unbind()
    # After unbind, dump_node is a no-op even on bound-shaped input.
    # We only assert the rebound idempotency here.
    tree_dump.bind(tmp_path / "tree2")
    tree_dump.unbind()


def test_bind_idempotent(tmp_path):
    from src.runtime import tree_dump
    target = tmp_path / "tree"
    tree_dump.bind(target)
    tree_dump.bind(target)  # second call should not raise
    tree_dump.unbind()


def _make_node(node_id=1, parent_id=0, iter_no=1,
               action="tiling", with_score=True, with_profiling=True):
    from src.agents.reviewer import BranchQuality
    from src.eval.profiler import AnalyticalMetrics, NCUMetrics, ProfilingResult
    from src.eval.scorer import ScoreResult
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    from src.search.tree import TreeNode
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    kernel = Kernel(spec=spec, source_code="def kernel_fn(): pass\n")
    score = None
    if with_score:
        score = ScoreResult(sol_score=0.5, baseline_latency_us=100.0,
                            candidate_latency_us=50.0, t_sol_us=25.0,
                            speedup=2.0, reward_hack_suspect=False,
                            calibration_warning=False)
    profiling = None
    if with_profiling:
        a = AnalyticalMetrics(achieved_tflops=10.0, achieved_bandwidth_gb_s=500.0,
                              pct_peak_compute=0.3, pct_peak_bandwidth=0.7)
        n = NCUMetrics(sm_occupancy_pct=80.0, l2_hit_rate_pct=70.0,
                       tensor_core_util_pct=0.0,
                       warp_stall_dominant="long_scoreboard", warp_stall_dominant_pct=40.0,
                       warp_stall_runner_up="lg_throttle", warp_stall_runner_up_pct=20.0)
        profiling = ProfilingResult(analytical=a, ncu=n,
                                    raw_metrics={"sm__cycles_active.avg": 12345.0})
    return TreeNode(id=node_id, kernel=kernel, parent_id=parent_id,
                    action_applied=action, iter_no=iter_no,
                    branch_quality=BranchQuality.PROMISING if with_score else None,
                    score=score, profiling=profiling, depth=1)


def test_dump_node_writes_kernel_and_meta(tmp_path):
    import json
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        node = _make_node()
        tree_dump.dump_node(node, iter_no=1, ncu_rep_src=None)
        node_dir = tmp_path / "tree" / "node_1"
        assert (node_dir / "kernel.py").read_text() == "def kernel_fn(): pass\n"
        meta = json.loads((node_dir / "meta.json").read_text())
        assert meta["id"] == 1
        assert meta["iter_no"] == 1
        assert meta["action_applied"] == "tiling"
        assert meta["branch_quality"] == "promising"
        assert meta["score"]["sol_score"] == 0.5
        assert meta["analytical"]["pct_peak_compute"] == 0.3
        ncu_json = json.loads((node_dir / "ncu.json").read_text())
        assert ncu_json["sm__cycles_active.avg"] == 12345.0
        assert not (node_dir / "ncu.ncu-rep").exists()
    finally:
        tree_dump.unbind()


def test_render_profiling_for_planner_handles_analytical_none():
    """Per the a+b decoupling (2026-05-13), the Planner's lightweight
    profile renderer must guard ``profiling.analytical is None`` before
    reading ``pct_peak_*`` fields. Pre-fix this raised AttributeError
    inside the orchestrator's iter-1 expansion when the baseline (or any
    parent) was profiled with nbytes=0 — aborted the whole search before
    Planner/Reviewer ever ran.

    Codex adversarial review Finding #1 (high)."""
    from src.eval.profiler import NCUMetrics, ProfilingResult
    from src.search.orchestrator import _render_profiling_for_planner

    ncu_only = ProfilingResult(
        analytical=None,
        ncu=NCUMetrics(
            sm_occupancy_pct=8.3, l2_hit_rate_pct=42.0,
            tensor_core_util_pct=0.0,
            warp_stall_dominant="long_scoreboard",
            warp_stall_dominant_pct=85.0,
            warp_stall_runner_up="wait", warp_stall_runner_up_pct=10.0,
        ),
        raw_metrics={},
    )
    # Must not raise. NCU lines still appear; pct_peak_* lines omitted.
    rendered = _render_profiling_for_planner(ncu_only)
    assert "pct_peak_compute" not in rendered
    assert "pct_peak_bandwidth" not in rendered
    assert "sm_occupancy=8.3%" in rendered
    assert "dominant_stall=long_scoreboard" in rendered


def test_render_profiling_for_planner_full_path_unchanged():
    """Regression guard: when analytical IS present, the renderer's
    output keeps the same shape it always had (pct_peak_compute /
    pct_peak_bandwidth lead the summary)."""
    from src.eval.profiler import AnalyticalMetrics, ProfilingResult
    from src.search.orchestrator import _render_profiling_for_planner

    p = ProfilingResult(
        analytical=AnalyticalMetrics(
            achieved_tflops=1.0, achieved_bandwidth_gb_s=100.0,
            pct_peak_compute=0.1, pct_peak_bandwidth=0.5,
        ),
        ncu=None,
        raw_metrics={},
    )
    rendered = _render_profiling_for_planner(p)
    assert "pct_peak_compute=10.0%" in rendered
    assert "pct_peak_bandwidth=50.0%" in rendered


def test_dump_node_handles_analytical_none(tmp_path):
    """Per the a+b decoupling (2026-05-13), ``ProfilingResult.analytical``
    can be ``None`` when the per-iter byte count was 0 (SOLAR + shape
    formulas both failed). ``_build_meta`` must serialize ``analytical``
    as JSON-null instead of crashing on ``asdict(None)`` — the surrounding
    ``dump_node`` only catches ``OSError`` so a TypeError here would
    abort the run mid-profile. NCU data still rides through.

    Codex adversarial review Finding #2 (high)."""
    import json
    from src.agents.reviewer import BranchQuality
    from src.eval.profiler import NCUMetrics, ProfilingResult
    from src.eval.scorer import ScoreResult
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    from src.runtime import tree_dump
    from src.search.tree import TreeNode

    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    kernel = Kernel(spec=spec, source_code="def kernel_fn(): pass\n")
    ncu_only = ProfilingResult(
        analytical=None,
        ncu=NCUMetrics(
            sm_occupancy_pct=12.3, l2_hit_rate_pct=45.6,
            tensor_core_util_pct=0.0,
            warp_stall_dominant="long_scoreboard",
            warp_stall_dominant_pct=80.0,
            warp_stall_runner_up="wait", warp_stall_runner_up_pct=10.0,
        ),
        raw_metrics={"foo": 1.0},
    )
    node = TreeNode(id=1, kernel=kernel, parent_id=0,
                    action_applied="x", iter_no=1,
                    branch_quality=BranchQuality.PROMISING,
                    score=ScoreResult(sol_score=0.5, baseline_latency_us=100.0,
                                      candidate_latency_us=100.0, t_sol_us=25.0,
                                      speedup=1.0, reward_hack_suspect=False,
                                      calibration_warning=False),
                    profiling=ncu_only, depth=1)

    tree_dump.bind(tmp_path / "tree")
    try:
        tree_dump.dump_node(node, iter_no=1, ncu_rep_src=None)
        meta = json.loads((tmp_path / "tree" / "node_1" / "meta.json").read_text())
        assert meta["analytical"] is None
        # NCU still serialized via ncu.json so the Reviewer keeps its signal.
        ncu_json = json.loads((tmp_path / "tree" / "node_1" / "ncu.json").read_text())
        assert ncu_json["foo"] == 1.0
    finally:
        tree_dump.unbind()


def test_dump_node_skips_ncu_when_degraded(tmp_path):
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        node = _make_node(with_profiling=False)
        tree_dump.dump_node(node, iter_no=1, ncu_rep_src=None)
        node_dir = tmp_path / "tree" / "node_1"
        assert (node_dir / "kernel.py").exists()
        assert (node_dir / "meta.json").exists()
        assert not (node_dir / "ncu.json").exists()
        assert not (node_dir / "ncu.ncu-rep").exists()
    finally:
        tree_dump.unbind()


def test_dump_node_copies_ncu_rep(tmp_path):
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    src = tmp_path / "src.ncu-rep"
    src.write_bytes(b"\x00\x01\x02NCURPT")
    try:
        node = _make_node()
        tree_dump.dump_node(node, iter_no=1, ncu_rep_src=src)
        copy = tmp_path / "tree" / "node_1" / "ncu.ncu-rep"
        assert copy.read_bytes() == b"\x00\x01\x02NCURPT"
    finally:
        tree_dump.unbind()


def test_dump_node_unbound_is_noop(tmp_path):
    from src.runtime import tree_dump
    # Not bound.
    node = _make_node()
    tree_dump.dump_node(node, iter_no=1, ncu_rep_src=None)
    # No exception, no files anywhere under tmp_path.
    assert not any(tmp_path.iterdir())


def test_dump_node_oserror_is_swallowed(tmp_path, monkeypatch):
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        def boom(*_a, **_k):
            raise OSError("disk full")
        monkeypatch.setattr(Path, "write_text", boom)
        node = _make_node()
        # Must not raise.
        tree_dump.dump_node(node, iter_no=1, ncu_rep_src=None)
    finally:
        tree_dump.unbind()


def test_node_summary_scored_node():
    from src.runtime.tree_dump import _node_summary
    node = _make_node()
    s = _node_summary(node, is_best=True)
    assert s["id"] == 1
    assert s["iter_no"] == 1
    assert s["action"] == "tiling"
    assert s["branch_quality"] == "promising"
    assert s["sol_score"] == 0.5
    assert s["speedup"] == 2.0
    assert s["is_best"] is True
    assert s["is_dead"] is False


def test_node_summary_unscored_node():
    from src.runtime.tree_dump import _node_summary
    node = _make_node(with_score=False, with_profiling=False)
    s = _node_summary(node, is_best=False)
    assert s["sol_score"] is None
    assert s["speedup"] is None
    assert s["branch_quality"] is None
    assert s["is_best"] is False


def test_node_summary_dead_end_node():
    from src.runtime.tree_dump import _node_summary
    from src.agents.reviewer import BranchQuality
    node = _make_node()
    node.branch_quality = BranchQuality.DEAD_END
    s = _node_summary(node, is_best=False)
    assert s["is_dead"] is True
    assert s["branch_quality"] == "dead_end"


def _make_tree_two_levels():
    """Tree:  0 (baseline) -> 1 (tiling, score 0.4) -> 2 (vectorize, score 0.65 ← best)."""
    from src.search.tree import SearchTree
    tree = SearchTree()
    n0 = _make_node(node_id=0, parent_id=None, iter_no=0,
                    action="", with_score=False, with_profiling=False)
    # Inject the prebuilt node into the tree.
    tree._nodes[0] = n0
    tree._next_id = 1
    n1 = _make_node(node_id=1, parent_id=0, iter_no=1, action="tiling")
    n1.score.sol_score = 0.4
    tree._nodes[1] = n1
    n0.children_ids.append(1)
    tree._next_id = 2
    n2 = _make_node(node_id=2, parent_id=1, iter_no=2, action="vectorize")
    n2.score.sol_score = 0.65
    n2.depth = 2
    tree._nodes[2] = n2
    n1.children_ids.append(2)
    tree._next_id = 3
    return tree


def test_finalize_tree_writes_five_files(tmp_path):
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        tree = _make_tree_two_levels()
        tree_dump.finalize_tree(tree)
        for name in (
            "index.json",
            "tree.txt",
            "tree.dot",
            "tree.mmd",
            "tree.preview.md",
        ):
            assert (tmp_path / "tree" / name).exists(), name
        preview = (tmp_path / "tree" / "tree.preview.md").read_text()
        mmd = (tmp_path / "tree" / "tree.mmd").read_text()
        assert preview.startswith("```mermaid\n")
        assert preview.rstrip().endswith("```")
        assert mmd in preview
    finally:
        tree_dump.unbind()


def test_finalize_tree_marks_best(tmp_path):
    import json
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        tree = _make_tree_two_levels()
        tree_dump.finalize_tree(tree)
        idx = json.loads((tmp_path / "tree" / "index.json").read_text())
        assert idx["best_node_id"] == 2
        assert idx["total_nodes"] == 3
        best = next(n for n in idx["nodes"] if n["id"] == 2)
        assert best["is_best"] is True
    finally:
        tree_dump.unbind()


def test_finalize_tree_unbound_is_noop(tmp_path):
    from src.runtime import tree_dump
    tree = _make_tree_two_levels()
    tree_dump.finalize_tree(tree)  # not bound; must not raise.
    assert not any(tmp_path.iterdir())


def test_finalize_tree_oserror_is_swallowed(tmp_path, monkeypatch):
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        def boom(*_a, **_k):
            raise OSError("disk full")
        monkeypatch.setattr(Path, "write_text", boom)
        tree = _make_tree_two_levels()
        tree_dump.finalize_tree(tree)  # must not raise.
    finally:
        tree_dump.unbind()


def test_render_ascii_includes_all_node_ids(tmp_path):
    from src.runtime.tree_dump import _render_ascii
    tree = _make_tree_two_levels()
    text = _render_ascii(tree, best_id=2)
    assert "[0]" in text and "[1]" in text and "[2]" in text


def test_render_ascii_marks_best():
    from src.runtime.tree_dump import _render_ascii
    tree = _make_tree_two_levels()
    text = _render_ascii(tree, best_id=2)
    assert "★" in text
    # Star appears on the line containing node 2.
    star_line = next(line for line in text.splitlines() if "★" in line)
    assert "[2]" in star_line


def test_render_ascii_shows_dead_end_status():
    from src.runtime.tree_dump import _render_ascii
    from src.agents.reviewer import BranchQuality
    tree = _make_tree_two_levels()
    tree.get_node(1).branch_quality = BranchQuality.DEAD_END
    tree.get_node(1).score = None  # dead-end may have no score
    text = _render_ascii(tree, best_id=2)
    line1 = next(line for line in text.splitlines() if "[1]" in line)
    assert "DEAD_END" in line1


def test_render_dot_has_digraph_header():
    from src.runtime.tree_dump import _render_dot
    tree = _make_tree_two_levels()
    text = _render_dot(tree, best_id=2)
    assert text.startswith("digraph search_tree")
    assert text.rstrip().endswith("}")


def test_render_dot_colors_by_branch_quality():
    from src.runtime.tree_dump import _render_dot
    from src.agents.reviewer import BranchQuality
    tree = _make_tree_two_levels()
    tree.get_node(1).branch_quality = BranchQuality.DEAD_END
    text = _render_dot(tree, best_id=2)
    # Best (node 2) gets the dark-green; dead-end (node 1) gets red.
    assert "#88e088" in text  # best
    assert "#f7c8c8" in text  # dead_end


def test_render_dot_emits_edges():
    from src.runtime.tree_dump import _render_dot
    tree = _make_tree_two_levels()
    text = _render_dot(tree, best_id=2)
    assert "n0 -> n1" in text
    assert "n1 -> n2" in text


def test_render_mermaid_has_graph_header():
    from src.runtime.tree_dump import _render_mermaid
    tree = _make_tree_two_levels()
    text = _render_mermaid(tree, best_id=2)
    assert text.startswith("graph TD")


def test_render_mermaid_emits_classdefs():
    from src.runtime.tree_dump import _render_mermaid
    tree = _make_tree_two_levels()
    text = _render_mermaid(tree, best_id=2)
    for cls in ("promising", "plateau", "blocked_potential",
                "dead_end", "neutral", "best"):
        assert f"classDef {cls}" in text


def test_render_mermaid_emits_edges():
    from src.runtime.tree_dump import _render_mermaid
    tree = _make_tree_two_levels()
    text = _render_mermaid(tree, best_id=2)
    assert "n0 --> n1" in text
    assert "n1 --> n2" in text


def test_dump_node_failure_detail_in_meta(tmp_path):
    """When ``failure_detail`` is set + the node carries ``dead_reason``,
    meta.json carries ``failure_detail`` at the top level and ``dead_reason``
    (via late-bound fields). Replaces the old ``failure: {reason, detail}``
    block — ``dead_reason`` now owns the categorical axis."""
    import json
    from src.agents.reviewer import BranchQuality
    from src.runtime import tree_dump
    from src.runtime.events import DeadReason
    tree_dump.bind(tmp_path / "tree")
    try:
        node = _make_node(with_score=False, with_profiling=False)
        node.branch_quality = BranchQuality.DEAD_END
        node.dead_reason = DeadReason.CUDA_ERROR
        tree_dump.dump_node(
            node, iter_no=2, ncu_rep_src=None,
            failure_detail="illegal memory access",
        )
        meta = json.loads((tmp_path / "tree" / "node_1" / "meta.json").read_text())
        assert meta["dead_reason"] == "cuda_error"
        assert meta["failure_detail"] == "illegal memory access"
        assert "failure" not in meta, (
            "old nested failure block must be gone — dead_reason + "
            "failure_detail at top level replace it"
        )
    finally:
        tree_dump.unbind()


def test_dump_node_no_failure_detail_on_advance_path(tmp_path):
    """``failure_detail`` is absent from meta.json on the advance path
    (no kill-site prose to record)."""
    import json
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        node = _make_node()
        tree_dump.dump_node(node, iter_no=1, ncu_rep_src=None)
        meta = json.loads((tmp_path / "tree" / "node_1" / "meta.json").read_text())
        assert "failure_detail" not in meta
        assert "failure" not in meta  # legacy nested key gone too
    finally:
        tree_dump.unbind()


def test_dump_node_dead_reason_without_failure_detail(tmp_path):
    """A DEAD_END node with a categorical reason but no kill-site prose
    (e.g., beam-pruned or Reviewer-judged) — meta carries ``dead_reason``
    without ``failure_detail``."""
    import json
    from src.agents.reviewer import BranchQuality
    from src.runtime import tree_dump
    from src.runtime.events import DeadReason
    tree_dump.bind(tmp_path / "tree")
    try:
        node = _make_node(with_score=False, with_profiling=False)
        node.branch_quality = BranchQuality.DEAD_END
        node.dead_reason = DeadReason.BEAM_PRUNED
        tree_dump.dump_node(node, iter_no=2, ncu_rep_src=None)
        meta = json.loads((tmp_path / "tree" / "node_1" / "meta.json").read_text())
        assert meta["dead_reason"] == "beam_pruned"
        assert "failure_detail" not in meta
    finally:
        tree_dump.unbind()


def test_finalize_tree_rewrites_evicted_node_branch_quality(tmp_path):
    """A node committed as PROMISING, then beam-evicted to DEAD_END,
    has its streamed meta.json rewritten to DEAD_END at finalize time."""
    import json
    from src.agents.reviewer import BranchQuality
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        tree = _make_tree_two_levels()
        # Stream-dump nodes 1 and 2 with their initial PROMISING status.
        for n_id in (1, 2):
            tree_dump.dump_node(
                tree.get_node(n_id), iter_no=tree.get_node(n_id).iter_no,
                ncu_rep_src=None,
            )
        # Now simulate beam eviction of node 1 (sets DEAD_END after streaming).
        tree.get_node(1).branch_quality = BranchQuality.DEAD_END
        tree_dump.finalize_tree(tree)
        meta1 = json.loads((tmp_path / "tree" / "node_1" / "meta.json").read_text())
        meta2 = json.loads((tmp_path / "tree" / "node_2" / "meta.json").read_text())
        assert meta1["branch_quality"] == "dead_end"
        assert meta2["branch_quality"] == "promising"  # not evicted
    finally:
        tree_dump.unbind()


def test_finalize_tree_preserves_failure_detail_on_rewrite(tmp_path):
    """A node already streamed with ``failure_detail`` (from
    ``_kill_branch``) keeps it after ``finalize_tree``'s rewrite — the
    rewrite only refreshes the late-bound fields, and ``failure_detail``
    is not one of them. ``dead_reason`` is late-bound (and may also have
    been set at stream time), so it round-trips through the late-bound
    refresh path."""
    import json
    from src.agents.reviewer import BranchQuality
    from src.runtime import tree_dump
    from src.runtime.events import DeadReason
    tree_dump.bind(tmp_path / "tree")
    try:
        tree = _make_tree_two_levels()
        node1 = tree.get_node(1)
        node1.branch_quality = BranchQuality.DEAD_END
        node1.dead_reason = DeadReason.CUDA_ERROR
        tree_dump.dump_node(
            node1, iter_no=node1.iter_no, ncu_rep_src=None,
            failure_detail="illegal memory access",
        )
        tree_dump.finalize_tree(tree)
        meta1 = json.loads((tmp_path / "tree" / "node_1" / "meta.json").read_text())
        assert meta1["branch_quality"] == "dead_end"
        assert meta1["dead_reason"] == "cuda_error"
        assert meta1["failure_detail"] == "illegal memory access"
    finally:
        tree_dump.unbind()


def test_finalize_tree_refreshes_late_bound_score_and_pwl(tmp_path):
    """A node dumped before its score and per_workload_latency_us were
    assigned (the orchestrator's root-baseline path) gets those fields
    refreshed from the in-memory tree at finalize_tree time. Mirrors the
    branch_quality refresh pattern; addresses the late-binding regression
    class structurally."""
    import json
    from src.eval.scorer import ScoreResult
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        tree = _make_tree_two_levels()
        # Stream-dump node 1 *before* score / pwl are bound, mirroring the
        # orchestrator's "dump root, then score" sequence.
        node1 = tree.get_node(1)
        node1.score = None
        node1.per_workload_latency_us = None
        tree_dump.dump_node(node1, iter_no=node1.iter_no, ncu_rep_src=None)
        meta1_pre = json.loads((tmp_path / "tree" / "node_1" / "meta.json").read_text())
        assert meta1_pre["score"] is None
        assert meta1_pre["per_workload_latency_us"] == {}
        # Now assign the late-bound fields and finalize.
        node1.score = ScoreResult(
            sol_score=0.42, baseline_latency_us=100.0,
            candidate_latency_us=50.0, t_sol_us=21.0, speedup=2.0,
            reward_hack_suspect=False, calibration_warning=False,
        )
        node1.per_workload_latency_us = {"wkl-a": 50.0, "wkl-b": 60.0}
        tree_dump.finalize_tree(tree)
        meta1_post = json.loads((tmp_path / "tree" / "node_1" / "meta.json").read_text())
        assert meta1_post["score"]["sol_score"] == 0.42
        assert meta1_post["score"]["speedup"] == 2.0
        assert meta1_post["per_workload_latency_us"] == {
            "wkl-a": 50.0, "wkl-b": 60.0,
        }
    finally:
        tree_dump.unbind()


def test_finalize_tree_refreshes_late_bound_children_ids(tmp_path):
    """A node dumped before its children were attached (the orchestrator's
    early root-dump path) gets ``children_ids`` refreshed from the in-memory
    tree at finalize_tree time. Same late-binding regression class as
    score / per_workload_latency_us; ensures node_0/meta.json agrees with
    index.json after iters add children."""
    import json
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        # Build a single-node tree and dump the root *before* any children
        # exist, mirroring orchestrator's "dump root, then iterate" sequence.
        from src.search.tree import SearchTree
        tree = SearchTree()
        n0 = _make_node(node_id=0, parent_id=None, iter_no=0,
                        action="", with_score=False, with_profiling=False)
        tree._nodes[0] = n0
        tree._next_id = 1
        tree_dump.dump_node(n0, iter_no=n0.iter_no, ncu_rep_src=None)
        meta0_pre = json.loads((tmp_path / "tree" / "node_0" / "meta.json").read_text())
        assert meta0_pre["children_ids"] == []
        # Now attach two children (mirrors add_child) and finalize.
        n1 = _make_node(node_id=1, parent_id=0, iter_no=1, action="tiling")
        n2 = _make_node(node_id=2, parent_id=0, iter_no=2, action="vectorize")
        tree._nodes[1] = n1
        tree._nodes[2] = n2
        n0.children_ids.extend([1, 2])
        tree._next_id = 3
        tree_dump.finalize_tree(tree)
        meta0_post = json.loads((tmp_path / "tree" / "node_0" / "meta.json").read_text())
        assert meta0_post["children_ids"] == [1, 2]
    finally:
        tree_dump.unbind()


def test_finalize_tree_skips_nodes_with_no_meta_json(tmp_path):
    """Nodes whose meta.json was never streamed (e.g., root) are
    silently skipped by the rewrite loop. No exception, no spurious file."""
    from src.runtime import tree_dump
    tree_dump.bind(tmp_path / "tree")
    try:
        tree = _make_tree_two_levels()
        # No dump_node calls — only finalize_tree.
        tree_dump.finalize_tree(tree)
        # Root node 0 has no streamed meta.json; finalize_tree must not
        # have created node_0/meta.json (we don't recover/synthesize).
        assert not (tmp_path / "tree" / "node_0" / "meta.json").exists()
    finally:
        tree_dump.unbind()
