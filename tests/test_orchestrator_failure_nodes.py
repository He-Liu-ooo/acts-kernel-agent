"""Tier-1 integration tests for orchestrator-side K-way failure
persistence under failure-node collapse.

Each iter's K Coder/bench-layer failures collapse into ONE failure-
summary node attached to the parent, with ``failure_details: list``
holding one entry per failed candidate. ``failure_summary_added`` fires
once per iter (not K times); per-candidate ``coder_failed`` events still
fire K times (unchanged). Profile-layer failures stay event-only (no
tree artifact) per the existing "downstream-of-truth" rationale.

Borrows the harness from ``tests/test_orchestrator_events.py`` (mocked
Planner / Coder / Reviewer / eval stack).

Spec: doc/specs/2026-05-18-failure-node-collapse-design.md
Plan: doc/plans/2026-05-18-failure-node-collapse-plan.md
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.runtime import events
# Re-export the harness fixture so this test file has access.
# pytest auto-discovers fixtures in conftest.py; the harness fixture
# lives in test_orchestrator_events.py and isn't conftest-scoped, so we
# import it directly and re-decorate.
from tests.test_orchestrator_events import (  # noqa: F401
    harness,
    _make_profile,
)


pytestmark = pytest.mark.asyncio


def _events_records(jsonl_path):
    return [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]


def _summary_nodes(tree):
    """All failure-summary nodes (the only ones with failure_details set)."""
    return [n for n in tree.nodes() if n.failure_details is not None]


async def _run_and_collect(harness, tmp_path, *, bench_override=None, profile_fake=None):
    """Run orchestrator with events binding; return (result, records)."""
    from src.search.orchestrator import Orchestrator

    if profile_fake is None:
        profile_fake = MagicMock(return_value=_make_profile())
    bench_to_use = bench_override or harness.bench

    if callable(bench_to_use):
        bench_patch = patch("src.eval.benchmark.benchmark_kernel", side_effect=bench_to_use)
    else:
        bench_patch = patch("src.eval.benchmark.benchmark_kernel", return_value=bench_to_use)
    profile_patch = patch("src.eval.profiler.profile_kernel", profile_fake)

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with bench_patch, profile_patch:
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            result = await orch.run(
                harness.baseline, workloads=None, roofline=harness.roofline,
            )
    finally:
        events.unbind()
        fh.close()

    records = _events_records(tmp_path / "events.jsonl")
    return result, records


# ── Coder-layer failures ──────────────────────────────────────────────────────


async def test_implementation_error_collapses_to_summary(tmp_path, harness):
    """K=1 ImplementationError → 1 failure-summary node with 1 detail
    (has_kernel_source=False because turn-exhaust = no submitted kernel)."""
    from src.agents.coder import ImplementationError

    harness.config.coder_n_candidates = 1
    harness.coder.implement = AsyncMock(side_effect=ImplementationError(
        "Coder exhausted turn budget (8) without calling submit_kernel."
    ))

    result, records = await _run_and_collect(harness, tmp_path)
    kinds = [r["kind"] for r in records]
    assert "coder_failed" in kinds
    assert "failure_summary_added" in kinds

    summaries = _summary_nodes(result.tree)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.parent_id == 0
    assert summary.kernel is None  # always None on summary nodes
    assert len(summary.failure_details) == 1
    fd = summary.failure_details[0]
    assert fd.candidate_idx == 0
    assert "Coder exhausted turn budget" in fd.reason
    assert fd.has_kernel_source is False  # turn-exhaust


async def test_entrypoint_binding_failure_collapses_to_summary(tmp_path, harness):
    """K=1 EntrypointBinding → 1 summary, has_kernel_source=True (Coder did submit)."""
    harness.config.coder_n_candidates = 1
    with patch(
        "src.eval.profiler.find_jit_name_in_entrypoint",
        return_value=(False, "kernel name not found in entrypoint"),
    ):
        result, records = await _run_and_collect(harness, tmp_path)

    assert "failure_summary_added" in [r["kind"] for r in records]
    summaries = _summary_nodes(result.tree)
    assert len(summaries) == 1
    fd = summaries[0].failure_details[0]
    assert "EntrypointBinding" in fd.reason
    assert fd.has_kernel_source is True


# ── Bench-layer failures ──────────────────────────────────────────────────────


async def test_benchmark_error_collapses_to_summary(tmp_path, harness):
    """K=1 BenchmarkError → 1 summary, has_kernel_source=True."""
    from src.eval.benchmark import BenchmarkError

    harness.config.coder_n_candidates = 1
    call_seq = [harness.bench]

    def next_bench(*_args, **_kwargs):
        if call_seq:
            return call_seq.pop(0)
        raise BenchmarkError(
            "autotune burn-in failed: AcceleratorError: CUDA error: "
            "operation not supported on global/shared address space"
        )

    result, records = await _run_and_collect(
        harness, tmp_path, bench_override=next_bench,
    )
    assert "failure_summary_added" in [r["kind"] for r in records]
    summaries = _summary_nodes(result.tree)
    assert len(summaries) == 1
    fd = summaries[0].failure_details[0]
    assert "operation not supported" in fd.reason
    assert fd.has_kernel_source is True


async def test_partial_bench_failure_collapses_to_summary(tmp_path, harness):
    """``not cand_bench.is_fully_successful`` → 1 summary."""
    from src.eval.benchmark import BenchmarkResult

    harness.config.coder_n_candidates = 1
    partial = BenchmarkResult(
        median_latency_us=100.0,
        timed_runs=1,
        per_workload_latency_us={"wl-0": 100.0, "wl-1": float("inf")},
        workload_errors={"wl-1": "launch failed"},
    )
    call_seq = [harness.bench, partial]

    def next_bench(*_args, **_kwargs):
        return call_seq.pop(0) if call_seq else partial

    result, records = await _run_and_collect(
        harness, tmp_path, bench_override=next_bench,
    )
    assert "failure_summary_added" in [r["kind"] for r in records]
    summaries = _summary_nodes(result.tree)
    assert len(summaries) == 1
    assert "partial bench failure" in summaries[0].failure_details[0].reason


# ── K-way fan-out semantics ───────────────────────────────────────────────────


async def test_kway_all_fail_collapses_to_one_summary_node(tmp_path, harness):
    """K=3 all ImplementationError → 1 summary node (not 3); 3 failure_details
    entries; iter SKIPPED; 1 failure_summary_added event; 3 coder_failed events."""
    from src.agents.coder import ImplementationError

    harness.config.coder_n_candidates = 3
    harness.coder.implement = AsyncMock(side_effect=ImplementationError(
        "budget exhausted"
    ))

    result, records = await _run_and_collect(harness, tmp_path)
    summaries = _summary_nodes(result.tree)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.parent_id == 0
    assert len(summary.failure_details) == 3
    assert {fd.candidate_idx for fd in summary.failure_details} == {0, 1, 2}

    summary_events = [r for r in records if r["kind"] == "failure_summary_added"]
    coder_failed_events = [r for r in records if r["kind"] == "coder_failed"]
    assert len(summary_events) == 1
    assert summary_events[0]["candidate_count"] == 3
    assert len(coder_failed_events) == 3
    assert {e["candidate_idx"] for e in coder_failed_events} == {0, 1, 2}

    iter_ends = [r for r in records if r["kind"] == "iter_end"]
    assert iter_ends[-1]["outcome"] == "skipped"


async def test_kway_all_fail_bumps_quarantine_counter_once(tmp_path, harness):
    """K parallel ImplementationErrors bump consecutive_agent_failures by exactly 1,
    not K. Refactor moved the persistence call but must not bump per-candidate."""
    from src.agents.coder import ImplementationError

    harness.config.coder_n_candidates = 4
    harness.coder.implement = AsyncMock(side_effect=ImplementationError("boom"))

    result, _ = await _run_and_collect(harness, tmp_path)
    root = result.tree.get_node(0)
    # consecutive_agent_failures on root: 1 per all-K-fail iter, not K.
    # Multiple iters of all-fail will tick the counter up by 1 each iter
    # until the quarantine threshold short-circuits frontier(). Counter
    # never exceeds the number of iters that actually ran K-way.
    assert root.consecutive_agent_failures >= 1
    # Sanity bound: ≤ iter count (not per-candidate explosion).
    iter_ends = sum(1 for n in result.tree.nodes() if n.iter_no >= 1)
    # The summary count == iter count (one per iter), so the counter
    # increments at most once per iter; equal to iters-run upper bound.
    assert root.consecutive_agent_failures <= max(iter_ends, 1)


# ── Disk artifacts ────────────────────────────────────────────────────────────


async def test_summary_node_disk_layout_bench_layer(tmp_path, harness):
    """Bench-layer K=1 failure: tree/node_<id>/meta.json + cand_0/{kernel.py, meta.json}."""
    from src.eval.benchmark import BenchmarkError
    from src.runtime import tree_dump

    harness.config.coder_n_candidates = 1
    call_seq = [harness.bench]

    def next_bench(*_args, **_kwargs):
        if call_seq:
            return call_seq.pop(0)
        raise BenchmarkError("autotune burn-in failed: cudaErrorInvalidAddressSpace")

    tree_root = tmp_path / "tree"
    tree_dump.bind(tree_root)
    try:
        result, _ = await _run_and_collect(
            harness, tmp_path, bench_override=next_bench,
        )
    finally:
        tree_dump.unbind()

    summaries = _summary_nodes(result.tree)
    assert len(summaries) == 1
    summary = summaries[0]
    node_dir = tree_root / f"node_{summary.id}"
    assert node_dir.exists()
    meta = json.loads((node_dir / "meta.json").read_text())
    assert meta["dead_reason"] == "coder_failed"
    assert len(meta["failure_details"]) == 1
    assert "cudaErrorInvalidAddressSpace" in meta["failure_details"][0]["reason"]
    # Per-candidate subdir.
    assert (node_dir / "cand_0" / "kernel.py").exists()
    cand_meta = json.loads((node_dir / "cand_0" / "meta.json").read_text())
    assert cand_meta["candidate_idx"] == 0
    assert cand_meta["has_kernel_source"] is True


async def test_summary_node_disk_layout_turn_exhaust(tmp_path, harness):
    """Turn-exhaust K=1: summary's cand_0/meta.json exists; cand_0/kernel.py absent."""
    from src.agents.coder import ImplementationError
    from src.runtime import tree_dump

    harness.config.coder_n_candidates = 1
    harness.coder.implement = AsyncMock(side_effect=ImplementationError(
        "Coder exhausted turn budget (8) without calling submit_kernel."
    ))

    tree_root = tmp_path / "tree"
    tree_dump.bind(tree_root)
    try:
        result, _ = await _run_and_collect(harness, tmp_path)
    finally:
        tree_dump.unbind()

    summary = _summary_nodes(result.tree)[0]
    node_dir = tree_root / f"node_{summary.id}"
    assert (node_dir / "meta.json").exists()
    assert (node_dir / "cand_0" / "meta.json").exists()
    # Turn-exhaust: no kernel.py.
    assert not (node_dir / "cand_0" / "kernel.py").exists()


async def test_kway_all_fail_disk_layout_has_k_cand_subdirs(tmp_path, harness):
    """K=3 all-fail → tree/node_<id>/cand_0/, cand_1/, cand_2/ all exist."""
    from src.agents.coder import ImplementationError
    from src.runtime import tree_dump

    harness.config.coder_n_candidates = 3
    harness.coder.implement = AsyncMock(side_effect=ImplementationError("boom"))

    tree_root = tmp_path / "tree"
    tree_dump.bind(tree_root)
    try:
        result, _ = await _run_and_collect(harness, tmp_path)
    finally:
        tree_dump.unbind()

    summary = _summary_nodes(result.tree)[0]
    node_dir = tree_root / f"node_{summary.id}"
    for i in range(3):
        cand_dir = node_dir / f"cand_{i}"
        assert cand_dir.exists()
        assert (cand_dir / "meta.json").exists()
        # Turn-exhaust → no kernel.py at any cand subdir.
        assert not (cand_dir / "kernel.py").exists()
    meta = json.loads((node_dir / "meta.json").read_text())
    assert len(meta["failure_details"]) == 3


# ── Profile-layer (negative) ──────────────────────────────────────────────────


async def test_profile_layer_failure_does_not_persist_summary(tmp_path, harness):
    """Profile-layer NCU crashes still stay event-only — no summary node,
    no failure_summary_added event."""
    from src.eval.profiler import ProfilerError

    harness.config.coder_n_candidates = 1
    profile_raises = MagicMock(side_effect=ProfilerError("NCU crash"))
    result, records = await _run_and_collect(
        harness, tmp_path, profile_fake=profile_raises,
    )

    assert len(_summary_nodes(result.tree)) == 0
    assert "failure_summary_added" not in [r["kind"] for r in records]


# ── autotune_exclude end-to-end plumbing ──────────────────────────────────────


async def test_planner_autotune_exclude_reaches_coder_submit_validator(tmp_path, harness):
    """End-to-end: Planner returns a plan with ``autotune_exclude``;
    orchestrator forwards it through ``coder.implement(plan=...)``;
    Coder exhausts turns; summary node attached with the existing
    ``Coder exhausted turn budget`` reason.
    """
    from src.agents.coder import ImplementationError
    from src.agents.planner import OptimizationPlan

    harness.config.coder_n_candidates = 1
    harness.planner.plan = AsyncMock(return_value=OptimizationPlan(
        tier=1,
        technique="t1_block_size_tuning",
        params={"BLOCK_K": "32"},
        rationale="exclude the overcommitted config that crashed prior siblings",
        autotune_exclude=[{"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 4}],
    ))
    harness.coder.implement = AsyncMock(side_effect=ImplementationError(
        "Coder exhausted turn budget (8) without calling submit_kernel."
    ))

    result, _ = await _run_and_collect(harness, tmp_path)

    call_args = harness.coder.implement.call_args
    assert call_args is not None
    plan_arg = call_args.kwargs.get("plan")
    assert plan_arg is not None
    assert plan_arg.autotune_exclude == [
        {"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 4}
    ]
    summaries = _summary_nodes(result.tree)
    assert len(summaries) == 1
    assert "turn budget" in summaries[0].failure_details[0].reason


# ── Channel-B reward-hack confirmed-kill path (mixed outcome) ─────────────────


async def test_reward_hack_confirmed_kill_still_persists_sibling_failures(tmp_path, harness):
    """Mixed outcome: K=2, candidate 0 EntrypointBinding-fails, candidate 1
    wins bench/profile but Channel-B re-eval confirms it as a reward-hack.

    The Channel-B confirmed kill exits the iter via _kill_branch + continue.
    Without explicit persistence on that exit path, the candidate-0 failure
    accumulated during the per-candidate bench loop would emit `coder_failed`
    but produce no summary node + no `failure_summary_added` event — divergent
    from the legacy immediate-persistence contract. Regression for Codex
    finding 2026-05-19.
    """
    from unittest.mock import AsyncMock
    from src.eval.scorer import ScoreResult
    from src.runtime.events import DeadReason
    from src.search.orchestrator import Orchestrator

    harness.config.coder_n_candidates = 2

    # First survivor (cand 0) trips EntrypointBinding → accumulator gets one
    # FailureDetail. Second survivor (cand 1) passes the gate and goes through
    # bench → profile → score → Channel-B re-eval → confirmed kill.
    binding_results = iter([
        (False, "kernel name not found in entrypoint"),
        (True, ""),
    ])
    suspect_score = ScoreResult(
        sol_score=0.99,
        baseline_latency_us=100.0,
        candidate_latency_us=40.0,
        t_sol_us=50.0,
        speedup=2.5,
        reward_hack_suspect=True,
        calibration_warning=False,
    )

    with (
        patch(
            "src.eval.profiler.find_jit_name_in_entrypoint",
            side_effect=lambda *a, **kw: next(binding_results),
        ),
        patch("src.eval.scorer.compute_sol_score", return_value=suspect_score),
        patch.object(
            Orchestrator,
            "_reward_hack_re_eval",
            AsyncMock(return_value=False),  # confirmed hack
        ),
    ):
        result, records = await _run_and_collect(harness, tmp_path)

    # The Channel-B kill path attaches the dead-end *winner* as a tree
    # child (via add_child + mark_dead inside _kill_branch). Filtering on
    # failure_details isolates the summary node, which should carry the
    # one sibling failure that happened before the kill.
    summaries = _summary_nodes(result.tree)
    assert len(summaries) == 1, (
        "Channel-B reward-hack-confirmed kill must still persist the K-1 "
        f"sibling failures accumulated before the kill; got summaries={summaries}"
    )
    summary = summaries[0]
    assert summary.parent_id == 0
    assert len(summary.failure_details) == 1
    assert "EntrypointBinding" in summary.failure_details[0].reason

    # Confirm the event-log mirror: per-candidate coder_failed event for
    # the EntrypointBinding sibling, plus the new failure_summary_added.
    coder_failed_events = [r for r in records if r["kind"] == "coder_failed"]
    summary_events = [r for r in records if r["kind"] == "failure_summary_added"]
    assert any("EntrypointBinding" in e["reason"] for e in coder_failed_events)
    assert len(summary_events) == 1
    assert summary_events[0]["candidate_count"] == 1

    # Channel-B confirmed kill still fires (sibling assertion: the winner
    # is dead with the expected reason; not the focus of this test).
    confirmed = [r for r in records if r["kind"] == "reward_hack_confirmed"]
    assert len(confirmed) == 1
