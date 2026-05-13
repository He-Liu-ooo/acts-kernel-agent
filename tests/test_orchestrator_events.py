"""Integration tests: orchestrator emits the expected event sequence.

Reuses the mock-agents harness pattern from ``test_orchestrator_profiling.py``
but keeps a self-contained copy here — importing across test modules
couples their lifecycles. Tier 1 (no GPU).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import rtx6000_ada_hardware as _rtx6000_ada
from src.agents.coder import KernelCodeOutput
from src.agents.planner import OptimizationPlan
from src.agents.reviewer import BranchQuality, ReviewerFeedback
from src.config import ACTSConfig
from src.eval.benchmark import BenchmarkResult
from src.eval.profiler import (
    AnalyticalMetrics,
    NCUMetrics,
    ProfilingResult,
)
from src.eval.roofline import RooflineResult
from src.eval.types import BottleneckType
from src.kernels.kernel import Kernel, KernelSpec, KernelType
from src.runtime import events


# ── fixtures ────────────────────────────────────────────────────────────


def _make_kernel(name: str = "root") -> Kernel:
    return Kernel(
        spec=KernelSpec(
            name=name,
            kernel_type=KernelType.MATMUL,
            flop_count=1_000_000,
            memory_bytes=100_000,
        ),
        source_code="# placeholder",
    )


def _make_profile() -> ProfilingResult:
    return ProfilingResult(
        analytical=AnalyticalMetrics(
            achieved_tflops=1.0,
            achieved_bandwidth_gb_s=100.0,
            pct_peak_compute=0.1,
            pct_peak_bandwidth=0.5,
        ),
        ncu=NCUMetrics(
            sm_occupancy_pct=72.5,
            l2_hit_rate_pct=45.0,
            tensor_core_util_pct=0.0,
            warp_stall_dominant="long_scoreboard",
            warp_stall_dominant_pct=33.0,
            warp_stall_runner_up="wait",
            warp_stall_runner_up_pct=18.0,
        ),
        raw_metrics={},
        degraded_reason=None,
    )


@pytest.fixture
def harness():
    """Orchestrator harness with mocked agents returning happy-path values."""
    config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=1,
        beam_width=3,
        sol_plateau_window=99,
    )
    planner = MagicMock()
    planner.plan = AsyncMock(return_value=OptimizationPlan(
        tier=3, technique="tiling", params={}, target_region="",
        rationale="reshape loop tiling for better cache reuse",
    ))
    coder = MagicMock()
    coder.implement = AsyncMock(
        return_value=KernelCodeOutput.model_construct(
            source_code="# child source",
            triton_kernel_name="",
        )
    )
    reviewer = MagicMock()
    reviewer.review = AsyncMock(return_value=ReviewerFeedback(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
    ))
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=[])

    bench = BenchmarkResult(median_latency_us=100.0, timed_runs=1)
    baseline = _make_kernel("root")
    roofline = RooflineResult(
        t_sol_us=50.0,
        bottleneck=BottleneckType.MEMORY_BOUND,
    )

    return SimpleNamespace(
        config=config,
        planner=planner,
        coder=coder,
        reviewer=reviewer,
        retriever=retriever,
        bench=bench,
        baseline=baseline,
        roofline=roofline,
    )


async def _run_orch(h, *, bench_override=None, profile_fake=None):
    from src.search.orchestrator import Orchestrator

    if profile_fake is None:
        profile_fake = MagicMock(return_value=_make_profile())
    bench_to_use = bench_override or h.bench
    with (
        patch("src.eval.benchmark.benchmark_kernel", return_value=bench_to_use),
        patch("src.eval.profiler.profile_kernel", profile_fake),
    ):
        orch = Orchestrator(
            h.config, h.planner, h.coder, h.reviewer, h.retriever,
        )
        return await orch.run(
            h.baseline, workloads=None, roofline=h.roofline,
        )


# ── tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_event_sequence(tmp_path, harness):
    """One advanced iteration emits iter-level events in order:
    iter_start → planner_selected → coder_submitted → bench_done →
    profile_done → score_computed → reviewer_feedback → iter_end.
    """
    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        await _run_orch(harness)
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kinds = [r["kind"] for r in records]

    # baseline_ready fires once before the loop body.
    assert kinds.count("baseline_ready") == 1

    # All expected iter-level kinds present.
    for expected in (
        "iter_start", "planner_selected", "coder_submitted",
        "bench_done", "profile_done",
        "score_computed", "reviewer_feedback", "iter_end",
    ):
        assert expected in kinds, f"missing {expected}; got {kinds!r}"

    # ``coder_submitted`` does NOT claim pass — the orchestrator cannot
    # verify the gates from the return value alone; ground truth is in
    # ``traces/*.jsonl``. Guard against a regression that reintroduces
    # a pass field.
    coder_rec = next(r for r in records if r["kind"] == "coder_submitted")
    assert "passed" not in coder_rec

    def idx(kind: str) -> int:
        return kinds.index(kind)

    assert idx("iter_start") < idx("planner_selected")
    assert idx("planner_selected") < idx("coder_submitted")
    assert idx("coder_submitted") < idx("bench_done")
    assert idx("bench_done") < idx("profile_done")
    assert idx("profile_done") < idx("score_computed")
    assert idx("score_computed") < idx("reviewer_feedback")
    assert idx("reviewer_feedback") < idx("iter_end")

    # planner_selected carries technique + tier + rationale_short.
    planner = records[idx("planner_selected")]
    assert planner["technique"] == "tiling"
    assert planner["tier"] == 3
    assert "rationale_short" in planner

    # iter_end on the happy path carries outcome=advanced.
    end = records[idx("iter_end")]
    assert end["outcome"] == "advanced"


@pytest.mark.asyncio
async def test_coder_failure_emits_skipped_not_dead_end(tmp_path, harness):
    """ImplementationError → coder_failed + iter_end(skipped). The branch
    is NOT marked dead (orchestrator soft-skips the iteration without a
    tree mutation), so emitting branch_dead_end + iter_end(dead_end) would
    mis-describe the tree state. Also verifies ``coder_submitted`` is NOT
    emitted on the failure path."""
    from src.agents.coder import ImplementationError

    harness.coder.implement = AsyncMock(side_effect=ImplementationError("budget exhausted"))

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        await _run_orch(harness)
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kinds = [r["kind"] for r in records]

    assert "coder_failed" in kinds
    assert "coder_submitted" not in kinds
    assert "branch_dead_end" not in kinds  # no tree node died
    # iter_end fires with outcome=skipped.
    end_recs = [r for r in records if r["kind"] == "iter_end"]
    assert end_recs, kinds
    assert end_recs[-1]["outcome"] == "skipped"
    # coder_failed reason carries the exception string.
    failed = next(r for r in records if r["kind"] == "coder_failed")
    assert "budget exhausted" in failed["reason"]


@pytest.mark.asyncio
async def test_planner_failure_emits_skipped_not_dead_end(tmp_path, harness):
    """PlanningError → planner_failed + iter_end(skipped). Mirrors the
    Coder skip-iter pattern: a Planner hiccup is branch-local — no tree
    mutation, no planner_selected/coder_submitted on the failure path,
    and the next iteration picks a different parent."""
    from src.agents.planner import PlanningError

    harness.planner.plan = AsyncMock(side_effect=PlanningError("submit_plan missing"))

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        await _run_orch(harness)
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kinds = [r["kind"] for r in records]

    assert "planner_failed" in kinds
    assert "planner_selected" not in kinds  # never reached
    assert "coder_submitted" not in kinds   # never reached
    assert "branch_dead_end" not in kinds   # no tree node died
    end_recs = [r for r in records if r["kind"] == "iter_end"]
    assert end_recs, kinds
    assert end_recs[-1]["outcome"] == "skipped"
    failed = next(r for r in records if r["kind"] == "planner_failed")
    assert "submit_plan missing" in failed["reason"]
    # Order: planner_failed must precede the iter_end(skipped).
    assert kinds.index("planner_failed") < kinds.index("iter_end")


@pytest.mark.asyncio
async def test_repeated_planner_failure_quarantines_parent_after_threshold(
    tmp_path, harness,
):
    """Two consecutive PlanningErrors on the same parent must bump
    ``parent.consecutive_agent_failures`` to ``QUARANTINE_THRESHOLD``,
    pulling the parent out of ``frontier()`` so subsequent iterations
    can't re-pick it forever and silently consume the entire
    ``max_depth`` budget. Without this, a deterministic Planner failure
    on the highest-scoring node spends 20 iterations doing nothing."""
    from src.agents.planner import PlanningError
    from src.search.tree import QUARANTINE_THRESHOLD

    harness.planner.plan = AsyncMock(side_effect=PlanningError("submit_plan missing"))
    # Bump max_depth so we can run more iterations than QUARANTINE_THRESHOLD.
    harness.config.max_depth = QUARANTINE_THRESHOLD + 2

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        result = await _run_orch(harness)
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kinds = [r["kind"] for r in records]

    # Two iterations should fail-and-skip; after that the only frontier
    # node (root) is quarantined, so the next select_next has no work →
    # the loop terminates via ALL_DEAD_END (frontier empty after pruning).
    planner_failed_count = kinds.count("planner_failed")
    assert planner_failed_count == QUARANTINE_THRESHOLD, (
        f"expected {QUARANTINE_THRESHOLD} planner_failed events before "
        f"the parent is quarantined, got {planner_failed_count}: {kinds!r}"
    )

    # Root node's failure counter should be at the threshold post-run.
    root = result.tree.get_node(0)
    assert root.consecutive_agent_failures >= QUARANTINE_THRESHOLD
    # And the frontier must be empty (root is the only node, and it's
    # quarantined).
    assert root not in result.tree.frontier()


@pytest.mark.asyncio
async def test_dead_end_iteration_event_sequence(tmp_path, harness):
    """Partial-workload bench failure → bench_done(is_fully_successful=False)
    → branch_dead_end → iter_end(dead_end). No score_computed or reviewer
    on the dead path.
    """
    # Placeholder mode (workloads=None) feeds the bench result through a
    # minted-once path; rebuild the bench with workload_errors so
    # ``is_fully_successful`` flips to False and the orchestrator trips
    # the dead_end gauntlet for the child.
    partial_bench = BenchmarkResult(
        median_latency_us=100.0,
        timed_runs=1,
        per_workload_latency_us={"wl-0": 100.0, "wl-1": float("inf")},
        workload_errors={"wl-1": "launch failed"},
    )

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        # The baseline bench uses the default (fully-successful) ``h.bench``
        # via the patch below; the child bench reuses the same patch so
        # every benchmark_kernel call returns the partial-failure result.
        # Baseline's is_fully_successful path is computed *before* we check,
        # so we need a branch to differ between baseline and child. Using
        # patch side_effect = [baseline_ok, child_partial] gives us that.
        call_seq = [harness.bench, partial_bench]

        def next_bench(*_args, **_kwargs):
            return call_seq.pop(0) if call_seq else partial_bench

        with (
            patch("src.eval.benchmark.benchmark_kernel", side_effect=next_bench),
            patch("src.eval.profiler.profile_kernel", MagicMock(return_value=_make_profile())),
        ):
            from src.search.orchestrator import Orchestrator
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)
    finally:
        events.unbind()
        fh.close()

    # Use strict JSON parsing to catch any Infinity/NaN tokens that
    # would slip into events.jsonl — these break RFC-8259 consumers.
    # Python's default ``json.loads`` allows Infinity; we force-reject
    # it here to act as a regression guard.
    def _strict_loads(s: str):
        return json.loads(
            s,
            parse_constant=lambda c: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {c}")
            ),
        )

    raw_lines = [line for line in (tmp_path / "events.jsonl").read_text().splitlines() if line.strip()]
    records = [_strict_loads(line) for line in raw_lines]
    kinds = [r["kind"] for r in records]

    # bench_done fires with is_fully_successful=False
    bench_recs = [r for r in records if r["kind"] == "bench_done"]
    assert any(r["is_fully_successful"] is False for r in bench_recs)
    # Failed workload latencies are serialized as null, not ``Infinity``.
    partial = next(r for r in bench_recs if r["is_fully_successful"] is False)
    assert None in partial["per_workload_us"], partial["per_workload_us"]
    # No score_computed or reviewer_feedback on the dead path
    assert "score_computed" not in kinds
    assert "reviewer_feedback" not in kinds
    # branch_dead_end comes before iter_end(dead_end)
    dead_idx = kinds.index("branch_dead_end")
    end_idx = kinds.index("iter_end")
    assert dead_idx < end_idx
    assert records[end_idx]["outcome"] == "dead_end"
    assert "reason" in records[dead_idx]


@pytest.mark.asyncio
async def test_runner_run_wrapped_in_trace_with_iter_metadata(harness, monkeypatch):
    """Each agent invocation in the iteration body is wrapped in
    ``with _iter_trace(iter_no, agent_name)``. We spy on the helper to
    confirm planner / coder / reviewer are all wrapped on iter 1."""
    from src.search import orchestrator as orch

    calls: list[dict] = []
    real = orch._iter_trace

    def spy(iter_no, agent_name):
        calls.append({"iter": iter_no, "agent": agent_name})
        return real(iter_no, agent_name)

    monkeypatch.setattr(orch, "_iter_trace", spy)
    await _run_orch(harness)
    assert {"iter": 1, "agent": "planner"} in calls
    assert {"iter": 1, "agent": "coder"} in calls
    assert {"iter": 1, "agent": "reviewer"} in calls


@pytest.mark.asyncio
async def test_committed_node_carries_iter_no(tmp_path, harness):
    """The child committed on iter N has node.iter_no == N. The happy
    path commits one child off the root on iter 1, so node 1 must carry
    iter_no=1 (root remains at the default -1)."""
    result = await _run_orch(harness)
    child = result.tree.get_node(1)
    assert child.iter_no == 1
    # Root is added before the loop body; its iter_no stays at the default.
    assert result.tree.get_node(0).iter_no == -1


@pytest.mark.asyncio
async def test_dump_node_called_on_committed_node(harness, monkeypatch):
    """On the ITER_ADVANCED path, ``tree_dump.dump_node`` is invoked for
    the committed child with the right iter_no. ``ncu_rep_src`` is None
    here because the test profile has no ``.ncu-rep`` artifact."""
    from src.runtime import tree_dump

    calls: list[dict] = []

    def spy(node, *, iter_no, ncu_rep_src):
        calls.append({"id": node.id, "iter_no": iter_no, "ncu_rep_src": ncu_rep_src})

    monkeypatch.setattr(tree_dump, "dump_node", spy)
    await _run_orch(harness)
    # Filter to non-root: the orchestrator also dumps the baseline root
    # (id=0, iter_no=-1); this test asserts on the child-side advance call.
    child_calls = [c for c in calls if c["id"] != 0]
    assert len(child_calls) == 1
    assert child_calls[0]["id"] == 1
    assert child_calls[0]["iter_no"] == 1
    assert child_calls[0]["ncu_rep_src"] is None


@pytest.mark.asyncio
async def test_dump_node_called_on_dead_end_with_dead_reason(harness, monkeypatch):
    """Dead-end iterations now DO call ``tree_dump.dump_node`` so operators
    inspecting "why did node N die" find a meta.json with the kill reason.

    The original Task-13 invariant (dead-ends do NOT dump) flipped when
    Codex caught that ``finalize_tree`` indexes the dead-end node in
    ``index.json`` while the per-node directory was missing — same node,
    two truths. Dead-end dumps carry the categorical cause via
    ``node.dead_reason`` (single source of truth across all DEAD_END
    paths) and the kill-site prose via ``failure_detail``; on the
    partial-bench path the reason is ``DeadReason.BENCH_FAILURE`` and
    the detail describes the workload errors.
    """
    from src.runtime import tree_dump
    from src.runtime.events import DeadReason
    from src.search.orchestrator import Orchestrator

    captured: list[dict] = []

    def spy(node, *, iter_no, ncu_rep_src, failure_detail=None):
        captured.append({
            "id": node.id, "iter_no": iter_no,
            "ncu_rep_src": ncu_rep_src,
            "dead_reason": node.dead_reason,
            "failure_detail": failure_detail,
        })

    monkeypatch.setattr(tree_dump, "dump_node", spy)

    partial_bench = BenchmarkResult(
        median_latency_us=100.0,
        timed_runs=1,
        per_workload_latency_us={"wl-0": 100.0, "wl-1": float("inf")},
        workload_errors={"wl-1": "launch failed"},
    )
    call_seq = [harness.bench, partial_bench]

    def next_bench(*_args, **_kwargs):
        return call_seq.pop(0) if call_seq else partial_bench

    with (
        patch("src.eval.benchmark.benchmark_kernel", side_effect=next_bench),
        patch("src.eval.profiler.profile_kernel", MagicMock(return_value=_make_profile())),
    ):
        orch = Orchestrator(
            harness.config, harness.planner, harness.coder,
            harness.reviewer, harness.retriever,
        )
        await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)

    # Filter to non-root: the orchestrator also dumps the baseline root
    # (id=0); this test asserts on the dead-end child dump.
    child_recs = [c for c in captured if c["id"] != 0]
    # Exactly one dump_node call captured the dead-end node and carried
    # the kill reason from _kill_branch through to dump_node.
    assert len(child_recs) == 1, child_recs
    rec = child_recs[0]
    assert rec["id"] == 1
    assert rec["iter_no"] == 1
    # No profiling on the dead-end path → ncu_rep_src is None.
    assert rec["ncu_rep_src"] is None
    # dead_reason is set on the node by _kill_branch (replaces the old
    # failure_reason kwarg, which duplicated dead_reason.value).
    assert rec["dead_reason"] == DeadReason.BENCH_FAILURE
    # detail is non-None (the orchestrator built a workload-errors string).
    assert rec["failure_detail"] is not None
    assert "wl-1" in rec["failure_detail"]


@pytest.mark.asyncio
async def test_dump_node_called_after_beam_prune(harness, monkeypatch):
    """``dump_node`` runs after ``beam_prune`` so meta.json reflects the
    post-prune ``branch_quality``. Pre-fix: dump_node ran first and the
    streamed meta.json said ``promising`` while the final index.json
    (post-prune) said ``dead_end``. Same node, two truths.

    Test pattern: replace ``beam_prune`` with a stub that marks every
    just-committed (non-root) node DEAD_END, then assert dump_node saw
    DEAD_END (not the pre-prune PROMISING that the reviewer assigned).
    """
    from src.runtime import tree_dump

    captured: list[dict] = []

    def spy(node, *, iter_no, ncu_rep_src, failure_detail=None):
        captured.append({
            "id": node.id,
            "branch_quality": node.branch_quality,
        })

    monkeypatch.setattr(tree_dump, "dump_node", spy)

    def evicting_prune(tree, beam_width, **kwargs):
        # Mark every just-committed (non-root) node DEAD_END.
        for n in list(tree._nodes.values()):
            if n.id != 0:
                n.branch_quality = BranchQuality.DEAD_END
        return []

    # Patch ``beam_prune`` at its source module. The orchestrator
    # imports it lazily inside ``run()`` via ``from src.search.beam
    # import beam_prune``, so by the time the loop body calls it, the
    # name resolves to the source-module attribute (which we replace).
    # Same applies to the ``_kill_branch`` helper's local re-import.
    from src.search import beam as beam_mod
    monkeypatch.setattr(beam_mod, "beam_prune", evicting_prune)

    await _run_orch(harness)

    # Filter to non-root: the orchestrator also dumps the baseline root
    # (id=0) before the search loop. Exactly one advance-path dump for
    # the committed child, and it observed the post-prune DEAD_END
    # branch_quality (not the PROMISING the reviewer assigned).
    child_recs = [c for c in captured if c["id"] != 0]
    assert len(child_recs) == 1, child_recs
    assert child_recs[0]["id"] == 1
    assert child_recs[0]["branch_quality"] == BranchQuality.DEAD_END


@pytest.mark.asyncio
async def test_dump_node_receives_real_ncu_rep_src_on_advance(harness, monkeypatch):
    """The orchestrator's profile_kernel call now produces a real
    ``ncu_rep_path``, and dump_node receives it (not None) on the advance
    path. Regression guard against a future change that drops the
    ``ncu_rep_src`` derivation at line ~929."""
    from pathlib import Path

    from src.runtime import tree_dump

    captured: list[dict] = []

    def spy(node, *, iter_no, ncu_rep_src, failure_detail=None):
        # Skip the baseline root dump (id=0); this test verifies the
        # advance-path child dump receives the populated ncu_rep_path.
        if node.id == 0:
            return
        captured.append({"ncu_rep_src": ncu_rep_src})

    monkeypatch.setattr(tree_dump, "dump_node", spy)

    # Build a profile with a populated ncu_rep_path. The profile is
    # what the orchestrator threads through to ncu_rep_src derivation.
    profile_with_rep = ProfilingResult(
        analytical=AnalyticalMetrics(
            achieved_tflops=1.0,
            achieved_bandwidth_gb_s=100.0,
            pct_peak_compute=0.1,
            pct_peak_bandwidth=0.5,
        ),
        ncu=NCUMetrics(
            sm_occupancy_pct=72.5,
            l2_hit_rate_pct=45.0,
            tensor_core_util_pct=0.0,
            warp_stall_dominant="long_scoreboard",
            warp_stall_dominant_pct=33.0,
            warp_stall_runner_up="wait",
            warp_stall_runner_up_pct=18.0,
        ),
        raw_metrics={},
        degraded_reason=None,
        ncu_rep_path=Path("/tmp/fake_rep.ncu-rep"),
    )

    await _run_orch(
        harness,
        profile_fake=MagicMock(return_value=profile_with_rep),
    )

    # The advance-path dump_node call received the populated
    # ncu_rep_path (not None).
    assert len(captured) == 1, captured
    assert captured[0]["ncu_rep_src"] is not None
    assert isinstance(captured[0]["ncu_rep_src"], Path)
    assert captured[0]["ncu_rep_src"] == Path("/tmp/fake_rep.ncu-rep")


@pytest.mark.asyncio
async def test_dump_node_not_called_on_skipped(harness, monkeypatch):
    """Skipped iterations (Coder failure → no tree mutation) must NOT
    call ``tree_dump.dump_node`` — there's no committed node to dump.

    The baseline root (id=0) dump fires before the loop body and is
    unrelated to the skipped-iteration invariant; filter it out.
    """
    from src.agents.coder import ImplementationError
    from src.runtime import tree_dump

    calls: list = []

    def spy(node, *args, **kwargs):
        calls.append(node.id)

    monkeypatch.setattr(tree_dump, "dump_node", spy)

    harness.coder.implement = AsyncMock(side_effect=ImplementationError("budget exhausted"))
    await _run_orch(harness)
    # Only the baseline root dump should have fired — no per-iter dump on
    # the skipped path.
    assert [c for c in calls if c != 0] == []


@pytest.mark.asyncio
async def test_dump_node_called_on_baseline_root(tmp_path, harness):
    """Regression: the baseline root node must be persisted to disk.

    Pre-fix, ``Orchestrator.run`` only invoked ``tree_dump.dump_node`` from
    the per-iter advance path and ``_kill_branch`` — never for the root.
    ``finalize_tree`` then indexed the root in ``index.json`` while
    ``tree/node_0/`` was missing on disk (no kernel.py / meta.json), the
    same shape of half-truth that motivated the dead-end dump fix.
    """
    import json
    from src.runtime import tree_dump

    tree_dump.bind(tmp_path / "tree")
    try:
        await _run_orch(harness)
    finally:
        tree_dump.unbind()

    node_dir = tmp_path / "tree" / "node_0"
    assert (node_dir / "kernel.py").exists(), "baseline kernel.py not written"
    assert (node_dir / "meta.json").exists(), "baseline meta.json not written"
    meta = json.loads((node_dir / "meta.json").read_text())
    assert meta["id"] == 0
    # Root carries the default sentinel iter_no=-1 — same value
    # ``finalize_tree``'s index entry uses for the baseline.
    assert meta["iter_no"] == -1


# ── sibling-aware contracts (2026-05-13) ────────────────────────────────


def test_regressed_sibling_actions_returns_only_regressed():
    """``regressed_sibling_actions`` filters siblings by Δ-SOL > 0.02 vs parent."""
    from dataclasses import dataclass
    from src.agents.reviewer import BranchQuality
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    from src.search.tree import SearchTree

    @dataclass
    class _StubScore:
        sol_score: float

    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=spec, source_code=""))
    root.score = _StubScore(sol_score=0.5)

    # Regressed by 0.07 (>= threshold)
    c1 = tree.add_child(root.id, Kernel(spec=spec, source_code=""),
                        action_applied="t1_block_size_tuning", iter_no=1)
    c1.score = _StubScore(sol_score=0.43)
    c1.branch_quality = BranchQuality.BLOCKED_POTENTIAL

    # Marginal (-0.01, below threshold)
    c2 = tree.add_child(root.id, Kernel(spec=spec, source_code=""),
                        action_applied="t3_tf32", iter_no=2)
    c2.score = _StubScore(sol_score=0.49)

    # No score yet (still scoring)
    tree.add_child(root.id, Kernel(spec=spec, source_code=""),
                   action_applied="t2_prefetching", iter_no=3)

    out = tree.regressed_sibling_actions(root.id)
    assert out == [("t1_block_size_tuning", 1)]


def test_regressed_sibling_actions_respects_exclude_id():
    from dataclasses import dataclass
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    from src.search.tree import SearchTree

    @dataclass
    class _StubScore:
        sol_score: float

    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=spec, source_code=""))
    root.score = _StubScore(sol_score=0.5)
    c1 = tree.add_child(root.id, Kernel(spec=spec, source_code=""),
                        action_applied="t1_block_size_tuning", iter_no=1)
    c1.score = _StubScore(sol_score=0.43)
    c2 = tree.add_child(root.id, Kernel(spec=spec, source_code=""),
                        action_applied="t3_loop_unroll", iter_no=2)
    c2.score = _StubScore(sol_score=0.40)

    out = tree.regressed_sibling_actions(root.id, exclude_id=c1.id)
    assert out == [("t3_loop_unroll", 2)]


def test_regressed_sibling_actions_excludes_neutral_boundary_sibling():
    """A sibling at exactly Δ = -0.02 from parent is in the neutral band
    ([-0.02, +0.02]) per reviewer/system.md branch-quality table, so the
    helper must NOT flag it as regressed. Strict inequality is load-bearing."""
    from dataclasses import dataclass
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    from src.search.tree import SearchTree

    @dataclass
    class _StubScore:
        sol_score: float

    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=spec, source_code=""))
    root.score = _StubScore(sol_score=0.50)
    boundary = tree.add_child(root.id, Kernel(spec=spec, source_code=""),
                              action_applied="t1_block_size_tuning", iter_no=1)
    boundary.score = _StubScore(sol_score=0.48)  # Δ = -0.02, neutral
    just_past = tree.add_child(root.id, Kernel(spec=spec, source_code=""),
                               action_applied="t3_loop_unroll", iter_no=2)
    just_past.score = _StubScore(sol_score=0.479)  # Δ = -0.021, regressed

    out = tree.regressed_sibling_actions(root.id)
    assert out == [("t3_loop_unroll", 2)]


def test_regressed_sibling_actions_empty_when_parent_unscored():
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    from src.search.tree import SearchTree

    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    tree = SearchTree()
    root = tree.add_root(Kernel(spec=spec, source_code=""))
    # parent has no score → helper returns []
    tree.add_child(root.id, Kernel(spec=spec, source_code=""),
                   action_applied="t1_block_size_tuning", iter_no=1)
    assert tree.regressed_sibling_actions(root.id) == []


@pytest.mark.asyncio
async def test_sibling_context_rendered_fires_for_planner_and_reviewer(
    tmp_path, harness,
):
    """A 2-iter run where iter 2 reuses iter 1's parent emits
    ``sibling_context_rendered`` once for the planner call and once for
    the reviewer call in iter 2 — and NEVER in iter 1 (no sibling exists)."""
    from src.search import beam as _beam

    # max_depth=2 so a second iteration runs. Force select_next to always
    # return root so both iters spawn siblings off the same parent.
    harness.config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=2,
        beam_width=3,
        sol_plateau_window=99,
    )

    # Make iter 1's child regress against root so it qualifies as a
    # "regressed sibling" for the iter-2 emit. baseline 100us, iter-1
    # child 400us, iter-2 child 400us; t_sol=50us. SOL drops to ~0.143,
    # comfortably below root's 0.5 by more than the 0.02 threshold.
    bench_seq = [
        BenchmarkResult(median_latency_us=100.0, timed_runs=1),  # baseline
        BenchmarkResult(median_latency_us=400.0, timed_runs=1),  # iter 1 child
        BenchmarkResult(median_latency_us=400.0, timed_runs=1),  # iter 2 child
    ]

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", side_effect=bench_seq),
            patch("src.eval.profiler.profile_kernel",
                  MagicMock(return_value=_make_profile())),
            patch.object(_beam, "select_next",
                         side_effect=lambda tree, eps: tree.get_node(0)),
        ):
            from src.search.orchestrator import Orchestrator
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            await orch.run(
                harness.baseline, workloads=None, roofline=harness.roofline,
            )
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    sibling_events = [r for r in records if r["kind"] == "sibling_context_rendered"]

    # Exactly two emits: planner side and reviewer side, both in iter 2.
    assert len(sibling_events) == 2, [r for r in sibling_events]
    consumers = sorted(r["consumer"] for r in sibling_events)
    assert consumers == ["planner", "reviewer"]
    for ev in sibling_events:
        assert ev["iter"] == 2
        assert ev["parent_node_id"] == "0"
        assert "tiling" in ev["regressed_actions"]

    # Planner-side fires BEFORE the iter-2 planner_selected event.
    planner_ev = next(r for r in sibling_events if r["consumer"] == "planner")
    planner_selected_iter2 = next(
        r for r in records
        if r["kind"] == "planner_selected" and r["iter"] == 2
    )
    assert records.index(planner_ev) < records.index(planner_selected_iter2)


@pytest.mark.asyncio
async def test_repeated_pathway_dead_end_fires_when_reviewer_judges_dead(
    tmp_path, harness,
):
    """When the iter-2 child's action matches a regressed sibling and the
    Reviewer verdict is DEAD_END, ``repeated_pathway_dead_end`` fires."""
    from src.search import beam as _beam

    harness.config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=2,
        beam_width=3,
        sol_plateau_window=99,
    )
    # Iter 2 reviewer judges DEAD_END.
    review_seq = [
        ReviewerFeedback(
            outcome="regressed",
            bottleneck_classification="memory_bound",
            branch_quality=BranchQuality.BLOCKED_POTENTIAL,
        ),
        ReviewerFeedback(
            outcome="regressed",
            bottleneck_classification="memory_bound",
            branch_quality=BranchQuality.DEAD_END,
        ),
    ]
    harness.reviewer.review = AsyncMock(side_effect=review_seq)

    bench_seq = [
        BenchmarkResult(median_latency_us=100.0, timed_runs=1),
        BenchmarkResult(median_latency_us=400.0, timed_runs=1),
        BenchmarkResult(median_latency_us=400.0, timed_runs=1),
    ]

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", side_effect=bench_seq),
            patch("src.eval.profiler.profile_kernel",
                  MagicMock(return_value=_make_profile())),
            patch.object(_beam, "select_next",
                         side_effect=lambda tree, eps: tree.get_node(0)),
        ):
            from src.search.orchestrator import Orchestrator
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            await orch.run(
                harness.baseline, workloads=None, roofline=harness.roofline,
            )
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    repeated = [r for r in records if r["kind"] == "repeated_pathway_dead_end"]
    assert len(repeated) == 1
    assert repeated[0]["iter"] == 2
    assert repeated[0]["action"] == "tiling"
    assert repeated[0]["sibling_iter"] == 1


@pytest.mark.asyncio
async def test_repeated_pathway_dead_end_does_not_fire_without_sibling_match(
    tmp_path, harness,
):
    """No ``repeated_pathway_dead_end`` when the iter-2 child's action
    differs from any regressed sibling (even when verdict == DEAD_END)."""
    from src.search import beam as _beam

    harness.config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=2,
        beam_width=3,
        sol_plateau_window=99,
    )
    # Different action per iter — iter 1 = "tiling", iter 2 = "fusion".
    plan_seq = [
        OptimizationPlan(tier=3, technique="tiling", params={},
                         target_region="", rationale="r1"),
        OptimizationPlan(tier=3, technique="fusion", params={},
                         target_region="", rationale="r2"),
    ]
    harness.planner.plan = AsyncMock(side_effect=plan_seq)
    harness.reviewer.review = AsyncMock(side_effect=[
        ReviewerFeedback(outcome="regressed",
                         bottleneck_classification="memory_bound",
                         branch_quality=BranchQuality.BLOCKED_POTENTIAL),
        ReviewerFeedback(outcome="regressed",
                         bottleneck_classification="memory_bound",
                         branch_quality=BranchQuality.DEAD_END),
    ])
    bench_seq = [
        BenchmarkResult(median_latency_us=100.0, timed_runs=1),
        BenchmarkResult(median_latency_us=400.0, timed_runs=1),
        BenchmarkResult(median_latency_us=400.0, timed_runs=1),
    ]

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", side_effect=bench_seq),
            patch("src.eval.profiler.profile_kernel",
                  MagicMock(return_value=_make_profile())),
            patch.object(_beam, "select_next",
                         side_effect=lambda tree, eps: tree.get_node(0)),
        ):
            from src.search.orchestrator import Orchestrator
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            await orch.run(
                harness.baseline, workloads=None, roofline=harness.roofline,
            )
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    repeated = [r for r in records if r["kind"] == "repeated_pathway_dead_end"]
    assert repeated == []
