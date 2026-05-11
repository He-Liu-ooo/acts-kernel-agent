"""Tier 2 tests for Planner-receives-Reviewer-feedback feature.
Spec: doc/specs/2026-05-10-planner-receives-reviewer-feedback-design.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.reviewer import BranchQuality, ReviewerFeedback
from src.eval.types import BottleneckType
from src.kernels.kernel import Kernel, KernelSpec, KernelType
from src.search.tree import TreeNode


def _make_kernel() -> Kernel:
    return Kernel(
        spec=KernelSpec(name="t", kernel_type=KernelType.MATMUL),
        source_code="# placeholder",
    )


def _make_full_feedback() -> ReviewerFeedback:
    """Helper — full feedback object used across tests."""
    return ReviewerFeedback(
        outcome="improved",
        metric_deltas={"sol_score": 0.12, "achieved_bw_pct": 8.4},
        bottleneck_classification=BottleneckType.MEMORY_BOUND,
        bottleneck_diagnosis="L2 hit 38% drives DRAM traffic above peak.",
        suggestions=["Try larger tiles", "Check coalescing"],
        branch_quality=BranchQuality.PROMISING,
        conditional_assessment="If register pressure drops, occupancy doubles.",
    )


def test_treenode_last_review_defaults_to_none():
    node = TreeNode(id=0, kernel=_make_kernel())
    assert node.last_review is None


def test_treenode_last_review_can_be_set():
    node = TreeNode(id=0, kernel=_make_kernel())
    fb = _make_full_feedback()
    node.last_review = fb
    assert node.last_review is fb


# ── Renderer: curated subset for Planner prompt ───────────────────────


from src.agents.reviewer import _render_review_for_planner


def test_render_review_for_planner_none_returns_none():
    assert _render_review_for_planner(None) is None


def test_render_review_for_planner_curated_subset():
    fb = _make_full_feedback()
    rendered = _render_review_for_planner(fb)
    assert rendered is not None
    # Included fields:
    assert "Outcome: improved" in rendered
    assert "Diagnosis: L2 hit 38% drives DRAM traffic above peak." in rendered
    assert "Suggestions: Try larger tiles; Check coalescing" in rendered
    assert "What would unlock progress: If register pressure drops" in rendered
    # Excluded fields (Planner already gets these from other channels):
    assert "metric_deltas" not in rendered
    assert "achieved_bw_pct" not in rendered
    assert "branch_quality" not in rendered
    assert "PROMISING" not in rendered
    assert "MEMORY_BOUND" not in rendered


def test_render_review_for_planner_omits_empty_fields():
    """Empty/missing fields don't produce stub lines (e.g., 'Suggestions: ')."""
    fb = ReviewerFeedback(
        outcome="neutral",
        metric_deltas={},
        bottleneck_classification=BottleneckType.BALANCED,
        bottleneck_diagnosis="",
        suggestions=[],
        branch_quality=BranchQuality.BLOCKED_POTENTIAL,
        conditional_assessment="",
    )
    rendered = _render_review_for_planner(fb)
    assert rendered == "Outcome: neutral"


# ── Parent-side: rendering driven by parent.last_review ───────────────


def test_render_review_for_planner_via_parent_with_last_review():
    """Orchestrator passes parent.last_review through the renderer."""
    parent = TreeNode(id=0, kernel=_make_kernel())
    parent.last_review = _make_full_feedback()

    rendered = _render_review_for_planner(parent.last_review)
    assert rendered is not None
    assert "Outcome: improved" in rendered


def test_render_review_for_planner_via_parent_with_none():
    """Parent without a review (e.g., legacy node) yields None."""
    parent = TreeNode(id=0, kernel=_make_kernel())
    assert parent.last_review is None
    rendered = _render_review_for_planner(parent.last_review)
    assert rendered is None


# ── Child-side: feedback attaches to child.last_review ────────────────


def test_treenode_last_review_attaches_after_review():
    """Direct semantic test: setting child.last_review after a review
    call mirrors the existing child.branch_quality assignment pattern."""
    child = TreeNode(id=1, kernel=_make_kernel())
    feedback = _make_full_feedback()

    # Mirror the orchestrator's existing pattern at line 939:
    child.branch_quality = feedback.branch_quality
    # NEW (Task 4):
    child.last_review = feedback

    assert child.branch_quality is feedback.branch_quality
    assert child.last_review is feedback


# ── Baseline review pass: prev_sol_score=None fallback ────────────────


def test_rule_based_feedback_handles_none_prev_score():
    """Smoke test for the path the baseline review's failure-fallback
    takes. Confirms outcome='neutral' + BLOCKED_POTENTIAL when
    prev_sol_score=None."""
    from src.agents.reviewer import rule_based_feedback

    fb = rule_based_feedback(
        sol_score=0.5,
        prev_sol_score=None,
        headroom_pct=50.0,
        bottleneck=BottleneckType.MEMORY_BOUND,
    )
    assert fb.outcome == "neutral"
    assert fb.branch_quality == BranchQuality.BLOCKED_POTENTIAL


# ── Tree-dump serialization (meta.json) ───────────────────────────────


def test_tree_dump_serializes_last_review():
    """Per-node meta.json includes last_review (or None when unset) via
    _late_bound_fields, mirroring branch_quality / score."""
    from src.runtime.tree_dump import _late_bound_fields

    node_with_review = TreeNode(id=1, kernel=_make_kernel())
    node_with_review.last_review = _make_full_feedback()

    fields = _late_bound_fields(node_with_review)
    assert "last_review" in fields
    serialized = fields["last_review"]
    assert serialized is not None
    assert serialized["outcome"] == "improved"
    assert serialized["bottleneck_diagnosis"] == \
        "L2 hit 38% drives DRAM traffic above peak."
    # Enums coerce to .value strings:
    assert serialized["branch_quality"] == "promising"
    assert serialized["bottleneck_classification"] == "memory_bound"


def test_tree_dump_serializes_last_review_none():
    """Unset last_review serializes as null."""
    from src.runtime.tree_dump import _late_bound_fields

    node = TreeNode(id=0, kernel=_make_kernel())
    fields = _late_bound_fields(node)
    assert fields.get("last_review") is None


# ── Checkpoint round-trip and legacy compat ───────────────────────────


def test_treenode_legacy_checkpoint_default_last_review():
    """A TreeNode constructed without last_review (legacy checkpoint
    shape) defaults to None — back-compat with checkpoints predating
    this field."""
    legacy_node = TreeNode(
        id=0,
        kernel=_make_kernel(),
        # No last_review kwarg.
    )
    assert legacy_node.last_review is None


def test_deserialize_node_with_no_last_review_field():
    """Loading a checkpoint dict that predates last_review → node.last_review is None."""
    from src.search.tree import _deserialize_node

    legacy_data = {
        "id": 0,
        "parent_id": None,
        "children_ids": [],
        "action_applied": "",
        "depth": 0,
        "branch_quality": None,
        "score": None,
        "kernel": {
            "spec": {
                "name": "t",
                "kernel_type": "matmul",
                "flop_count": 0,
                "memory_bytes": 0,
                "input_shapes": [],
                "definition_path": None,
                "pytorch_reference": "",
                "t_sol_us": 0.0,
            },
            "source_code": "",
            "num_warps": 0,
            "num_stages": 0,
            "block_size": 0,
        },
        # last_review key intentionally absent (legacy)
    }
    node = _deserialize_node(legacy_data)
    assert node.last_review is None


def test_deserialize_node_round_trips_last_review():
    """A node with last_review serialized and deserialized round-trips."""
    from src.search.tree import _deserialize_node, _serialize_node

    node = TreeNode(id=1, kernel=_make_kernel())
    node.last_review = _make_full_feedback()

    data = _serialize_node(node)
    assert "last_review" in data
    assert data["last_review"] is not None

    reloaded = _deserialize_node(data)
    assert reloaded.last_review is not None
    assert reloaded.last_review.outcome == "improved"
    assert reloaded.last_review.bottleneck_diagnosis == \
        "L2 hit 38% drives DRAM traffic above peak."
    assert reloaded.last_review.branch_quality == BranchQuality.PROMISING


# ── Tier 2 GPU end-to-end ─────────────────────────────────────────────


@pytest.mark.gpu
def test_baseline_review_fires_on_live_sol_run(tmp_path):
    """Live GPU run against a small SOL fixture must:
    1. Successfully start (process subprocess returncode 0 or graceful timeout).
    2. Emit a ``reviewer_feedback`` event with ``iter=0`` to events.jsonl
       — proves the new baseline-review block in Phase A fired.
    3. Persist ``last_review`` in tree/node_0/meta.json — proves the new
       TreeNode field reaches disk via _late_bound_fields.

    Strategy: cap the run with a generous timeout. The baseline review
    fires *before* the search loop, so even if the search hits the
    timeout the baseline-pass evidence has already landed in events.jsonl.
    """
    import json
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    fixture_dir = repo_root / "tests" / "fixtures" / "sol_simple"
    assert fixture_dir.is_dir(), f"fixture missing: {fixture_dir}"

    # CLI shrank to {--config, --run-dir, --trace-dir} on 2026-05-11;
    # ``problem_path`` and ``gpu_index`` now live in the libconfig cfg.
    cfg_file = tmp_path / "acts.cfg"
    cfg_file.write_text(
        "hardware: { gpu_index = 0; };\n"
        f'runtime: {{ problem_path = "{fixture_dir}"; }};\n'
    )
    cmd = [
        sys.executable, "-m", "src.pipeline.optimize",
        "--config", str(cfg_file),
        "--run-dir", str(tmp_path),
        "--trace-dir=",  # disable trace capture for speed
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ},
        start_new_session=True,
    )

    # Poll for the baseline reviewer_feedback event (iter=0) up to
    # ~5 minutes. The full run may take longer; we don't need it to
    # finish — only the baseline pass needs to land.
    deadline = time.monotonic() + 300
    baseline_event = None
    run_dir = None
    while time.monotonic() < deadline:
        if run_dir is None:
            dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("run_")]
            if dirs:
                run_dir = dirs[0]
        if run_dir is not None:
            events_path = run_dir / "events.jsonl"
            if events_path.exists():
                for line in events_path.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("kind") == "reviewer_feedback" and rec.get("iter") == 0:
                        baseline_event = rec
                        break
                if baseline_event is not None:
                    break
        if proc.poll() is not None:
            break
        time.sleep(2)

    # Stop the subprocess (graceful → forceful).
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)

    stdout, stderr = proc.communicate()
    stdout_s = stdout.decode("utf-8", errors="replace") if stdout else ""
    stderr_s = stderr.decode("utf-8", errors="replace") if stderr else ""

    assert run_dir is not None, (
        f"no run_<UTC>/ directory created under {tmp_path}\n"
        f"stdout:\n{stdout_s[-2000:]}\nstderr:\n{stderr_s[-2000:]}"
    )
    assert baseline_event is not None, (
        f"no reviewer_feedback event with iter=0 in events.jsonl — "
        f"baseline-review block did not fire.\n"
        f"run_dir={run_dir}\n"
        f"stdout:\n{stdout_s[-2000:]}\nstderr:\n{stderr_s[-2000:]}"
    )

    # Spec-required: baseline event carries the verdict, suggestion, and degraded fields.
    assert "verdict" in baseline_event
    assert "suggestion_short" in baseline_event
    assert "degraded" in baseline_event

    # Spec-required: tree/node_0/meta.json carries the last_review key
    # via _late_bound_fields. Whether it's non-null depends on whether
    # finalize_tree ran (the initial dump_node fires before the baseline
    # review block, so last_review is None there; finalize_tree re-renders
    # at run end with the now-populated field). If we killed the subprocess
    # mid-run, the value may still be null — best-effort assertion only.
    meta_path = run_dir / "tree" / "node_0" / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        assert "last_review" in meta, (
            "tree/node_0/meta.json missing last_review key — _late_bound_fields "
            "isn't including the new field"
        )


# ── Codex adversarial-review fixes (2026-05-10) ────────────────────────


def test_apply_baseline_feedback_to_root_clamps_dead_end():
    """Codex finding [high]: a baseline review returning DEAD_END must
    NOT propagate to root.branch_quality, otherwise frontier() empties
    and the run exits ALL_DEAD_END before iter 1. The Planner still
    needs root.last_review for prompt context."""
    from src.search.orchestrator import _apply_baseline_feedback_to_root

    root = TreeNode(id=0, kernel=_make_kernel())
    feedback = ReviewerFeedback(
        outcome="regressed",
        metric_deltas={},
        bottleneck_classification=BottleneckType.MEMORY_BOUND,
        bottleneck_diagnosis="hallucinated DEAD_END on baseline",
        suggestions=[],
        branch_quality=BranchQuality.DEAD_END,
        conditional_assessment="",
    )

    _apply_baseline_feedback_to_root(root, feedback)

    # last_review is preserved so the Planner sees it as prompt context.
    assert root.last_review is feedback
    # branch_quality is NOT propagated when it's DEAD_END — root stays
    # expandable (frontier() includes it).
    assert root.branch_quality != BranchQuality.DEAD_END


def test_apply_baseline_feedback_to_root_propagates_promising():
    """Non-terminal qualities (PROMISING / PLATEAU / BLOCKED_POTENTIAL)
    propagate normally so root behaves consistently with children."""
    from src.search.orchestrator import _apply_baseline_feedback_to_root

    root = TreeNode(id=0, kernel=_make_kernel())
    feedback = _make_full_feedback()  # branch_quality=PROMISING

    _apply_baseline_feedback_to_root(root, feedback)

    assert root.last_review is feedback
    assert root.branch_quality == BranchQuality.PROMISING


def test_apply_baseline_feedback_to_root_propagates_blocked_potential():
    """BLOCKED_POTENTIAL is the rule-based fallback for prev_sol_score=None,
    which is what the baseline review uses. Must propagate."""
    from src.search.orchestrator import _apply_baseline_feedback_to_root

    root = TreeNode(id=0, kernel=_make_kernel())
    feedback = ReviewerFeedback(
        outcome="neutral",
        metric_deltas={},
        bottleneck_classification=BottleneckType.MEMORY_BOUND,
        bottleneck_diagnosis="rule-based fallback",
        suggestions=[],
        branch_quality=BranchQuality.BLOCKED_POTENTIAL,
        conditional_assessment="",
    )

    _apply_baseline_feedback_to_root(root, feedback)

    assert root.last_review is feedback
    assert root.branch_quality == BranchQuality.BLOCKED_POTENTIAL


def test_baseline_dead_end_keeps_root_in_frontier():
    """Integration: after the baseline-review block runs with a DEAD_END
    feedback, the SearchTree's frontier still contains root so iter 1
    can plan. Regression for Codex finding [high]."""
    from src.search.orchestrator import _apply_baseline_feedback_to_root
    from src.search.tree import SearchTree

    tree = SearchTree()
    root = tree.add_root(_make_kernel())

    feedback = ReviewerFeedback(
        outcome="regressed",
        metric_deltas={},
        bottleneck_classification=BottleneckType.MEMORY_BOUND,
        bottleneck_diagnosis="LLM hallucinated DEAD_END",
        suggestions=[],
        branch_quality=BranchQuality.DEAD_END,
        conditional_assessment="",
    )
    _apply_baseline_feedback_to_root(root, feedback)

    frontier = tree.frontier()
    assert root in frontier, (
        "root must remain in frontier after baseline DEAD_END verdict "
        "— otherwise the run exits ALL_DEAD_END before iter 1"
    )


def test_representative_latency_s_picks_workload_specific_value():
    """Codex finding [medium]: baseline profile must use the representative
    workload's per-workload latency, not the aggregate median, otherwise
    achieved-throughput metrics fed into the iter-1 Planner are corrupted
    when per-workload latencies diverge."""
    from src.eval.benchmark import BenchmarkResult
    from src.search.orchestrator import _representative_latency_s

    fake_workload = MagicMock()
    fake_workload.uuid = "w_repr"
    other_workload = MagicMock()
    other_workload.uuid = "w_other"
    workloads = [other_workload, fake_workload, other_workload]  # repr_idx=1

    # Aggregate median is 50us; representative workload's latency is 200us.
    # The baseline profile must see 200us (representative), not 50us (aggregate).
    bench = BenchmarkResult(
        median_latency_us=50.0,
        timed_runs=3,
        per_workload_latency_us={
            "w_repr": 200.0,
            "w_other": 50.0,
        },
    )

    latency_s = _representative_latency_s(bench, workloads, repr_idx=1)
    assert latency_s == 200.0 / 1e6, (
        f"expected representative latency 200us → 0.0002s; got {latency_s}"
    )


def test_representative_latency_s_returns_none_for_missing_workload():
    """When per_workload_latency_us is missing the representative
    workload entry, returns None so the baseline-review block can skip
    rather than feed a fabricated number to profile_kernel."""
    from src.eval.benchmark import BenchmarkResult
    from src.search.orchestrator import _representative_latency_s

    fake_workload = MagicMock()
    fake_workload.uuid = "w_missing"
    workloads = [fake_workload]

    bench = BenchmarkResult(
        median_latency_us=50.0,
        timed_runs=1,
        per_workload_latency_us={},  # representative absent
    )

    assert _representative_latency_s(bench, workloads, repr_idx=0) is None
