"""Tests for search/ — tree state management, beam pruning, orchestration."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.reviewer import BranchQuality
from src.eval.scorer import ScoreResult
from src.kernels.kernel import Kernel, KernelSpec, KernelType
from src.search.tree import SearchTree


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_kernel(name: str = "test") -> Kernel:
    return Kernel(
        spec=KernelSpec(name=name, kernel_type=KernelType.MATMUL),
        source_code="# placeholder",
    )


def _make_score(sol: float) -> ScoreResult:
    return ScoreResult(
        sol_score=sol,
        baseline_latency_us=100.0,
        candidate_latency_us=100.0 - sol * 50.0,
        t_sol_us=50.0,
        speedup=1.0 + sol,
    )


def _build_scored_tree() -> SearchTree:
    """Build a tree with 4 scored nodes for reuse across tests.

    Structure:
        root (0.3, "baseline")
        ├── a (0.6, "tiling")
        │   └── c (0.8, "vectorize")
        └── b (0.5, "unroll")
    """
    tree = SearchTree()
    root = tree.add_root(_make_kernel("root"))
    root.score = _make_score(0.3)
    root.action_applied = "baseline"

    a = tree.add_child(root.id, _make_kernel("a"), "tiling")
    a.score = _make_score(0.6)

    b = tree.add_child(root.id, _make_kernel("b"), "unroll")
    b.score = _make_score(0.5)

    c = tree.add_child(a.id, _make_kernel("c"), "vectorize")
    c.score = _make_score(0.8)

    return tree


# ── path_to_node ─────────────────────────────────────────────────────────────

class TestPathToNode:
    def test_path_to_root(self):
        tree = SearchTree()
        root = tree.add_root(_make_kernel())
        path = tree.path_to_node(root.id)
        assert len(path) == 1
        assert path[0].id == root.id

    def test_path_to_depth_one(self):
        tree = SearchTree()
        root = tree.add_root(_make_kernel())
        child = tree.add_child(root.id, _make_kernel(), "tiling")
        path = tree.path_to_node(child.id)
        assert [n.id for n in path] == [root.id, child.id]

    def test_path_to_depth_three(self):
        tree = _build_scored_tree()
        # root -> a -> c is the path to node c (id=3)
        path = tree.path_to_node(3)
        assert [n.id for n in path] == [0, 1, 3]

    def test_nonexistent_node_raises(self):
        tree = SearchTree()
        tree.add_root(_make_kernel())
        with pytest.raises(KeyError):
            tree.path_to_node(999)


# ── beam pruning: diversity-aware (B2) ───────────────────────────────────────

class TestBeamPruneDiversity:
    """Beam prune should preserve action-type diversity when possible."""

    def test_preserves_minority_action(self):
        """When top scores all come from one action, keep at least one
        from a different action to preserve exploration diversity."""
        from src.search.beam import beam_prune

        tree = SearchTree()
        root = tree.add_root(_make_kernel())
        root.score = _make_score(0.1)
        root.action_applied = "baseline"

        # 3 high-scoring "tiling" nodes
        for i in range(3):
            n = tree.add_child(root.id, _make_kernel(), "tiling")
            n.score = _make_score(0.9 - i * 0.01)

        # 1 lower-scoring "unroll" node
        lone = tree.add_child(root.id, _make_kernel(), "unroll")
        lone.score = _make_score(0.7)

        beam_prune(tree, beam_width=3)

        surviving = [n for n in tree.frontier()]
        surviving_actions = {n.action_applied for n in surviving}
        assert "unroll" in surviving_actions, (
            "Diversity: the lone 'unroll' node should survive pruning"
        )

    def test_score_dominates_at_large_gap(self):
        """Diversity shouldn't save a drastically worse node."""
        from src.search.beam import beam_prune

        tree = SearchTree()
        root = tree.add_root(_make_kernel())
        root.score = _make_score(0.1)
        root.action_applied = "baseline"

        # 3 excellent "tiling" nodes
        for i in range(3):
            n = tree.add_child(root.id, _make_kernel(), "tiling")
            n.score = _make_score(0.95 - i * 0.01)

        # 1 terrible "unroll" node
        bad = tree.add_child(root.id, _make_kernel(), "unroll")
        bad.score = _make_score(0.1)

        beam_prune(tree, beam_width=3)

        surviving = tree.frontier()
        surviving_ids = {n.id for n in surviving}
        assert bad.id not in surviving_ids, (
            "A drastically worse node shouldn't be saved by diversity alone"
        )

    def test_diversity_disabled_skips_rescue(self):
        """When enable_diversity=False, no diversity swap happens."""
        from src.search.beam import beam_prune

        tree = SearchTree()
        root = tree.add_root(_make_kernel())
        root.score = _make_score(0.1)
        root.action_applied = "baseline"

        for i in range(3):
            n = tree.add_child(root.id, _make_kernel(), "tiling")
            n.score = _make_score(0.9 - i * 0.01)

        lone = tree.add_child(root.id, _make_kernel(), "unroll")
        lone.score = _make_score(0.7)

        beam_prune(tree, beam_width=3, enable_diversity=False)

        surviving_ids = {n.id for n in tree.frontier()}
        assert lone.id not in surviving_ids, (
            "With diversity disabled, pure score ranking should prune unroll"
        )

    def test_empty_action_not_rescued(self):
        """Root node with action_applied='' (as created by the orchestrator)
        should not be diversity-rescued over real optimization nodes."""
        from src.search.beam import beam_prune

        tree = SearchTree()
        root = tree.add_root(_make_kernel())
        root.score = _make_score(0.51)
        # root.action_applied is "" by default — matches orchestrator behavior

        # 3 tiling nodes
        for i in range(3):
            n = tree.add_child(root.id, _make_kernel(), "tiling")
            n.score = _make_score(0.90 - i * 0.05)

        # 1 unroll node
        tree.add_child(root.id, _make_kernel(), "unroll")
        tree.get_node(4).score = _make_score(0.79)

        beam_prune(tree, beam_width=3)

        surviving_ids = {n.id for n in tree.frontier()}
        assert root.id not in surviving_ids, (
            "Root with empty action_applied should not be diversity-rescued"
        )


# ── beam pruning: branch-quality weighting (B3) ─────────────────────────────

class TestBeamPruneQuality:
    """PROMISING nodes should survive over PLATEAU nodes at similar scores."""

    def test_promising_survives_over_plateau(self):
        """With quality weighting, a PROMISING node with slightly lower
        raw score should beat a PLATEAU node for the last kept slot."""
        from src.search.beam import beam_prune

        tree = SearchTree()
        root = tree.add_root(_make_kernel())
        root.score = _make_score(0.1)
        root.action_applied = "baseline"

        # Two PLATEAU nodes with higher raw scores
        high = tree.add_child(root.id, _make_kernel(), "tiling")
        high.score = _make_score(0.65)
        high.branch_quality = BranchQuality.PLATEAU

        mid = tree.add_child(root.id, _make_kernel(), "unroll")
        mid.score = _make_score(0.62)
        mid.branch_quality = BranchQuality.PLATEAU

        # PROMISING node with slightly lower raw score than mid
        promising = tree.add_child(root.id, _make_kernel(), "vectorize")
        promising.score = _make_score(0.60)
        promising.branch_quality = BranchQuality.PROMISING

        # beam_width=2 out of 4 nodes.
        # Without quality weighting: high(0.65) + mid(0.62) survive.
        # With quality weighting: promising's bonus should push it past mid.
        beam_prune(tree, beam_width=2)

        surviving_ids = {n.id for n in tree.frontier()}
        assert promising.id in surviving_ids, (
            "PROMISING node should survive over PLATEAU at similar scores"
        )

    def test_score_dominates_over_quality(self):
        """Branch quality shouldn't override a large score advantage."""
        from src.search.beam import beam_prune

        tree = SearchTree()
        root = tree.add_root(_make_kernel())
        root.score = _make_score(0.1)
        root.action_applied = "baseline"

        plateau_good = tree.add_child(root.id, _make_kernel(), "tiling")
        plateau_good.score = _make_score(0.90)
        plateau_good.branch_quality = BranchQuality.PLATEAU

        promising_bad = tree.add_child(root.id, _make_kernel(), "unroll")
        promising_bad.score = _make_score(0.30)
        promising_bad.branch_quality = BranchQuality.PROMISING

        beam_prune(tree, beam_width=2)

        surviving_ids = {n.id for n in tree.frontier()}
        assert plateau_good.id in surviving_ids, (
            "High-scoring PLATEAU should beat low-scoring PROMISING"
        )


# ── beam pruning: no-op when within budget ───────────────────────────────────

class TestBeamPruneNoop:
    def test_no_pruning_when_within_beam(self):
        from src.search.beam import beam_prune

        tree = _build_scored_tree()
        pruned = beam_prune(tree, beam_width=10)
        assert pruned == []
        assert len(tree.frontier()) == 4


# ── tree checkpointing ──────────────────────────────────────────────────────

class TestTreeCheckpoint:
    def test_save_load_roundtrip(self, tmp_path: Path):
        tree = _build_scored_tree()
        save_path = tmp_path / "tree.json"

        tree.save(save_path)
        loaded = SearchTree.load(save_path)

        # Same number of nodes
        assert len(loaded.frontier()) + len([
            n for n in loaded._nodes.values()
            if n.branch_quality == BranchQuality.DEAD_END
        ]) == len(tree._nodes)

    def test_save_load_preserves_scores(self, tmp_path: Path):
        tree = _build_scored_tree()
        save_path = tmp_path / "tree.json"

        tree.save(save_path)
        loaded = SearchTree.load(save_path)

        for nid in tree._nodes:
            orig = tree.get_node(nid)
            restored = loaded.get_node(nid)
            if orig.score is not None:
                assert restored.score is not None
                assert restored.score.sol_score == pytest.approx(orig.score.sol_score)

    def test_save_load_preserves_structure(self, tmp_path: Path):
        tree = _build_scored_tree()
        save_path = tmp_path / "tree.json"

        tree.save(save_path)
        loaded = SearchTree.load(save_path)

        for nid in tree._nodes:
            orig = tree.get_node(nid)
            restored = loaded.get_node(nid)
            assert restored.parent_id == orig.parent_id
            assert restored.children_ids == orig.children_ids
            assert restored.action_applied == orig.action_applied
            assert restored.depth == orig.depth

    def test_save_load_preserves_branch_quality(self, tmp_path: Path):
        tree = _build_scored_tree()
        # Mark one node as dead end
        tree.get_node(2).branch_quality = BranchQuality.DEAD_END
        save_path = tmp_path / "tree.json"

        tree.save(save_path)
        loaded = SearchTree.load(save_path)

        assert loaded.get_node(2).branch_quality == BranchQuality.DEAD_END

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            SearchTree.load(tmp_path / "nonexistent.json")

    def test_next_id_preserved(self, tmp_path: Path):
        """After loading, new nodes should get IDs that don't collide."""
        tree = _build_scored_tree()
        save_path = tmp_path / "tree.json"

        tree.save(save_path)
        loaded = SearchTree.load(save_path)

        new_node = loaded.add_child(0, _make_kernel("new"), "fuse")
        assert new_node.id not in tree._nodes, (
            "New node ID should not collide with existing IDs"
        )


# ── plateau detection ────────────────────────────────────────────────────────

class TestPlateauDetection:
    """Global plateau: best score hasn't improved by more than delta
    over the last `window` iterations."""

    def test_plateau_detected_when_stagnant(self):
        from src.search.orchestrator import detect_plateau

        # 5 iterations, scores barely move
        history = [0.50, 0.51, 0.505, 0.51, 0.508]
        assert detect_plateau(history, window=3, delta=0.01) is True

    def test_plateau_not_detected_when_improving(self):
        from src.search.orchestrator import detect_plateau

        history = [0.50, 0.55, 0.62, 0.70, 0.78]
        assert detect_plateau(history, window=3, delta=0.01) is False

    def test_plateau_not_detected_insufficient_history(self):
        from src.search.orchestrator import detect_plateau

        history = [0.50, 0.51]
        assert detect_plateau(history, window=3, delta=0.01) is False

    def test_plateau_at_exact_boundary(self):
        from src.search.orchestrator import detect_plateau

        # Improvement of exactly delta — NOT plateau (needs to exceed)
        history = [0.50, 0.51, 0.51]
        assert detect_plateau(history, window=3, delta=0.01) is True

    def test_plateau_checks_only_recent_window(self):
        """Early stagnation followed by improvement should not trigger."""
        from src.search.orchestrator import detect_plateau

        history = [0.50, 0.50, 0.50, 0.60, 0.70, 0.80]
        assert detect_plateau(history, window=3, delta=0.01) is False

    def test_empty_history(self):
        from src.search.orchestrator import detect_plateau

        assert detect_plateau([], window=3, delta=0.01) is False


# ── path-context rendering ──────────────────────────────────────────────────

class TestRenderPath:
    """SearchTree.render_path renders the full root-to-node trajectory —
    not just the immediate parent — for Planner / Reviewer prompts."""

    def test_root_only_path(self):
        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = _make_score(0.3)

        ctx = tree.render_path(root.id)
        assert "depth 0" in ctx
        assert "[0] baseline" in ctx
        assert "SOL 0.300" in ctx
        assert "← current" in ctx

    def test_multi_step_path_includes_all_ancestors(self):
        """A depth-3 node's context lists every ancestor action and score."""
        tree = _build_scored_tree()
        # Node 3 is c: root (baseline, 0.3) -> a (tiling, 0.6) -> c (vectorize, 0.8)
        ctx = tree.render_path(3)
        assert "depth 2" in ctx
        assert "[0] baseline" in ctx
        assert "[1] tiling" in ctx
        assert "[2] vectorize" in ctx
        assert "SOL 0.300" in ctx
        assert "SOL 0.600" in ctx
        assert "SOL 0.800" in ctx
        # Only the current node is marked.
        assert ctx.count("← current") == 1
        # Sibling (b, unroll) must not leak into c's path.
        assert "unroll" not in ctx

    def test_path_includes_branch_quality_when_set(self):
        tree = _build_scored_tree()
        tree.get_node(1).branch_quality = BranchQuality.PROMISING
        ctx = tree.render_path(3)
        assert "PROMISING" in ctx


# ── orchestrator → reviewer context threading ───────────────────────────────

@pytest.fixture
def _orch_harness():
    """Reusable single-iteration harness for Orchestrator integration tests.

    Returns a namespace of prebuilt mocks + config. Each test overrides
    only the reviewer's return_value before invoking `orch.run()`.
    """
    from types import SimpleNamespace

    from src.agents.planner import OptimizationPlan
    from src.config import ACTSConfig, HardwareSpec
    from src.eval.benchmark import BenchmarkResult
    from src.eval.roofline import BottleneckType, RooflineResult

    config = ACTSConfig(
        hardware=HardwareSpec(),
        max_depth=1,
        beam_width=3,
        sol_plateau_window=99,  # disable plateau termination
    )

    planner = MagicMock()
    planner.plan = AsyncMock(return_value=OptimizationPlan(
        tier=1, technique="tiling", params={}, target_region="", rationale=""
    ))
    coder = MagicMock()
    coder.implement = AsyncMock(return_value="# child source")
    reviewer = MagicMock()
    reviewer.review = AsyncMock()  # test sets return_value
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=[])

    return SimpleNamespace(
        config=config,
        planner=planner,
        coder=coder,
        reviewer=reviewer,
        retriever=retriever,
        baseline=_make_kernel("root"),
        roofline=RooflineResult(
            t_sol_us=50.0,
            arithmetic_intensity=1.0,
            bottleneck=BottleneckType.MEMORY_BOUND,
        ),
        bench=BenchmarkResult(median_latency_us=100.0, timed_runs=1),
    )


async def _run_orch(h):
    from src.search.orchestrator import Orchestrator

    with patch("src.eval.benchmark.benchmark_kernel", return_value=h.bench):
        orch = Orchestrator(h.config, h.planner, h.coder, h.reviewer, h.retriever)
        await orch.run(h.baseline, workloads=None, roofline=h.roofline)


class TestOrchestratorReviewerContext:
    """Orchestrator must thread real parent SOL + tree context into review()
    so branch_quality is grounded in the actual search state, not defaults."""

    @pytest.mark.asyncio
    async def test_reviewer_receives_real_prev_and_tree_context(self, _orch_harness):
        """Per-iteration review() call gets the parent's SOL score and a
        non-empty tree_context string, not the old defaults."""
        from src.agents.reviewer import ReviewerFeedback

        h = _orch_harness
        h.reviewer.review.return_value = ReviewerFeedback(
            outcome="improved",
            bottleneck_classification="memory_bound",
            branch_quality=BranchQuality.PROMISING,
        )
        await _run_orch(h)

        # The reviewer must have been called with real path context, not defaults.
        assert h.reviewer.review.await_count == 1
        r_kwargs = h.reviewer.review.await_args.kwargs
        assert r_kwargs["prev_sol_score"] is not None, (
            "Orchestrator must pass the parent's SOL score, not leave it None"
        )
        assert r_kwargs["tree_context"] != "", (
            "Orchestrator must pass a non-empty tree_context"
        )
        # Full root-to-current path, not just parent summary.
        assert "Path" in r_kwargs["tree_context"]
        assert "[0] baseline" in r_kwargs["tree_context"], (
            "tree_context should render the root node as the start of the trajectory"
        )
        assert "tiling" in r_kwargs["tree_context"], (
            "tree_context should mention the applied action so the Reviewer can reason about it"
        )
        assert "← current" in r_kwargs["tree_context"], (
            "The current (child) node should be marked in the path"
        )

        # The planner must also get the same root-to-parent trajectory so it
        # can reason about which actions have already been tried on this branch.
        assert h.planner.plan.await_count == 1
        p_kwargs = h.planner.plan.await_args.kwargs
        assert p_kwargs["tree_context"] != ""
        assert "[0] baseline" in p_kwargs["tree_context"]
        # Planner sees the path ending at the parent (no child yet).
        assert "tiling" not in p_kwargs["tree_context"], (
            "At planning time the new action has not been applied yet — "
            "the Planner's path should stop at the parent"
        )

    @pytest.mark.asyncio
    async def test_orchestrator_logs_when_reviewer_is_degraded(self, _orch_harness, caplog):
        """When the reviewer returns degraded=True, the orchestrator must
        surface it via a warning log — silent continuation hides broken runs."""
        import logging

        from src.agents.reviewer import ReviewerFeedback

        h = _orch_harness
        h.reviewer.review.return_value = ReviewerFeedback(
            outcome="neutral",
            bottleneck_classification="balanced",
            branch_quality=BranchQuality.BLOCKED_POTENTIAL,
            degraded=True,
            error_reason="llm_retries_exhausted",
        )

        caplog.set_level(logging.WARNING, logger="src.search.orchestrator")
        await _run_orch(h)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("degraded" in r.getMessage().lower() for r in warnings), (
            "Degraded reviewer feedback must produce a warning log"
        )
        assert any("llm_retries_exhausted" in r.getMessage() for r in warnings)
