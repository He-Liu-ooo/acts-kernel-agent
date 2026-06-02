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
    retriever.sample = MagicMock(return_value=[])

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
async def test_autotune_burn_in_done_payload_uses_winner_count(tmp_path, harness):
    """Autotune observability reports how many workload winners were captured."""
    from sol_execbench.core.data import Workload
    from src.kernels.compiler import CompilationResult
    from src.search.orchestrator import Orchestrator

    harness.config.max_depth = 0
    bench = BenchmarkResult(
        median_latency_us=100.0,
        timed_runs=1,
        autotune_winner_per_workload={
            "wl-a": {"kwargs": {"BLOCK": 64}, "num_warps": 4, "num_stages": 2},
        },
    )
    workload = Workload.model_validate({
        "uuid": "wl-a",
        "axes": {"M": 128},
        "inputs": {},
    })
    compile_result = CompilationResult(
        success=True,
        compiled_fn=lambda: None,
        triton_autotuner=object(),
    )

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch("src.kernels.compiler.compile_kernel", return_value=compile_result),
            patch("src.eval.benchmark.benchmark_kernel", return_value=bench),
        ):
            orch = Orchestrator(
                harness.config,
                harness.planner,
                harness.coder,
                harness.reviewer,
                harness.retriever,
            )
            await orch.run(
                harness.baseline,
                workloads=[workload],
                input_generators=[lambda seed: ()],
                roofline=harness.roofline,
            )
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    event = next(r for r in records if r["kind"] == "autotune_burn_in_done")
    assert event["workload_count"] == 1
    assert event["winner_count"] == 1
    assert "winner_recorded" not in event


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
    """A2 K-way contract: candidate-level partial-workload bench failures
    are per-candidate failures that emit ``coder_failed`` and skip the
    candidate. When every K candidate fails, the iter ends as SKIPPED —
    no tree node is added for failed candidates and no ``score_computed``
    or ``reviewer_feedback`` runs on the dead path.

    (Before A2, partial-bench failure of the lone Coder output created a
    tree node, marked it DEAD_END, and emitted ``bench_done`` +
    ``branch_dead_end`` + ``iter_end(dead_end)``. K-way moved the gate
    upstream so losers never enter the tree.)
    """
    # K=1 keeps the test focused on the single-candidate partial-bench
    # path; the K-way fan-out is exercised by the test_k_way_* group.
    harness.config.coder_n_candidates = 1

    partial_bench = BenchmarkResult(
        median_latency_us=100.0,
        timed_runs=1,
        per_workload_latency_us={"wl-0": 100.0, "wl-1": float("inf")},
        workload_errors={"wl-1": "launch failed"},
    )

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
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

    # Strict JSON parser: Infinity/NaN tokens would break RFC-8259
    # consumers; Python's default ``json.loads`` permits them, so we
    # force-reject here as a regression guard.
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

    # Partial-bench failure surfaces as a per-candidate coder_failed event.
    coder_failed = [r for r in records if r["kind"] == "coder_failed"]
    assert len(coder_failed) == 1
    assert coder_failed[0]["candidate_idx"] == 0
    assert "partial bench failure" in coder_failed[0]["reason"]
    assert "wl-1" in coder_failed[0]["reason"]
    # No tree node was created, so no profile / score / reviewer.
    assert "score_computed" not in kinds
    assert "reviewer_feedback" not in kinds
    assert "branch_dead_end" not in kinds
    # iter_end carries skipped (no winner emerged from the K candidates).
    end_idx = kinds.index("iter_end")
    assert records[end_idx]["outcome"] == "skipped"


@pytest.mark.asyncio
async def test_runner_run_wrapped_in_trace_with_iter_metadata(harness, monkeypatch):
    """Each agent invocation in the iteration body is wrapped in
    ``with trace_span("acts_iter", iter_no=..., agent=...)``. We spy on
    the helper (imported into the orchestrator module) to confirm
    planner / coder / reviewer are all wrapped on iter 1."""
    from src.search import orchestrator as orch

    calls: list[dict] = []
    real = orch.trace_span

    def spy(workflow_name, *, iter_no, agent, **extra):
        # AgentLabel is a `str`-subclass enum: comparing the recorded
        # member against the bare value works directly (no manual
        # coercion needed). The orchestrator passes AgentLabel.* — the
        # equality assertions below use the bare strings.
        calls.append({"workflow_name": workflow_name, "iter": iter_no, "agent": agent})
        return real(workflow_name, iter_no=iter_no, agent=agent, **extra)

    monkeypatch.setattr(orch, "trace_span", spy)
    await _run_orch(harness)
    assert {"workflow_name": "acts_iter", "iter": 1, "agent": "planner"} in calls
    assert {"workflow_name": "acts_iter", "iter": 1, "agent": "coder"} in calls
    assert {"workflow_name": "acts_iter", "iter": 1, "agent": "reviewer"} in calls


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
    """Dead-end iterations call ``tree_dump.dump_node`` so operators
    inspecting "why did node N die" find a meta.json with the kill reason.

    Under A2's K-way contract, candidate-level failures (bench, profile,
    repr-latency-unavailable) are gated out before ``tree.add_child``,
    so dead-end dumps now flow only from *winner-side* kills — i.e. the
    channel-B reward-hack-confirmed path. This test exercises that
    surviving ``_kill_branch`` site: the winner bench-succeeds and
    profiles cleanly, but its SOL score trips ``reward_hack_suspect``
    and the re-eval confirms the hack → ``_kill_branch(REWARD_HACK_CONFIRMED)``
    → ``dump_node`` carrying ``dead_reason``.
    """
    from unittest.mock import AsyncMock

    from src.eval.scorer import ScoreResult
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
        patch("src.eval.benchmark.benchmark_kernel", return_value=harness.bench),
        patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
        patch("src.eval.scorer.compute_sol_score", return_value=suspect_score),
        patch.object(
            Orchestrator,
            "_reward_hack_re_eval",
            AsyncMock(return_value=False),  # confirmed hack
        ),
    ):
        orch = Orchestrator(
            harness.config, harness.planner, harness.coder,
            harness.reviewer, harness.retriever,
        )
        await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)

    # Filter to non-root: the orchestrator also dumps the baseline root
    # (id=0); this test asserts on the dead-end child dump.
    child_recs = [c for c in captured if c["id"] != 0]
    assert len(child_recs) == 1, child_recs
    rec = child_recs[0]
    assert rec["id"] == 1
    assert rec["iter_no"] == 1
    # dead_reason flows from _kill_branch → node.mark_dead.
    assert rec["dead_reason"] == DeadReason.REWARD_HACK_CONFIRMED


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
async def test_dump_failure_summary_called_for_failure_nodes_on_skipped(
    harness, monkeypatch
):
    """Skipped iterations from K-way Coder/bench-layer failures call
    ``tree_dump.dump_failure_summary_node`` once per iter (not K times) —
    failure-node collapse attaches one summary node carrying all K
    failure_details, not K per-candidate nodes.

    Per failure-node collapse (2026-05-18): K failed candidates collapse
    into a single failure-summary node per iter; one
    ``dump_failure_summary_node`` call writes ``meta.json`` plus
    ``cand_<i>/{kernel.py, meta.json}`` for each candidate. Per-candidate
    ``coder_failed`` events still fire K times (unchanged).
    """
    from src.agents.coder import ImplementationError
    from src.runtime import tree_dump

    summary_calls: list = []

    def spy(node, *args, **kwargs):
        summary_calls.append(node.id)

    monkeypatch.setattr(tree_dump, "dump_failure_summary_node", spy)

    harness.coder.implement = AsyncMock(side_effect=ImplementationError("budget exhausted"))
    await _run_orch(harness)
    # Each all-K-fail iter dumps exactly one summary node, regardless of K.
    assert len(summary_calls) >= 1, (
        f"Expected ≥1 failure-summary dump; got {summary_calls}"
    )


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
    # coder_n_candidates=1 keeps the legacy single-Coder per-iter cardinality
    # so the bench_seq below stays aligned (A2 K-way fan-out is tested
    # separately in the test_k_way_* group).
    harness.config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=2,
        beam_width=3,
        sol_plateau_window=99,
        coder_n_candidates=1,
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

    # coder_n_candidates=1: A2 K-way fan-out is tested separately; this
    # test targets the sibling-pathway gating logic and uses a fixed-length
    # bench sequence.
    harness.config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=2,
        beam_width=3,
        sol_plateau_window=99,
        coder_n_candidates=1,
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

    # coder_n_candidates=1: A2 K-way fan-out is tested separately.
    harness.config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=2,
        beam_width=3,
        sol_plateau_window=99,
        coder_n_candidates=1,
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


# ── A1 PR 1/B: _record_autotune_winner ─────────────────────────────────


def test_record_autotune_winner_populates_per_workload():
    """The orchestrator copies benchmark-captured winners onto the kernel."""
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    from src.search.orchestrator import _record_autotune_winner

    spec = KernelSpec(name="t", kernel_type=KernelType.MATMUL)
    kernel = Kernel(spec=spec, source_code="# placeholder")
    bench = BenchmarkResult(autotune_winner_per_workload={
        "wl-1": {"kwargs": {"BLOCK_N": 1024}, "num_warps": 4, "num_stages": 3}
    })

    _record_autotune_winner(kernel, bench)

    assert kernel.autotune_winner == {
        "wl-1": {"kwargs": {"BLOCK_N": 1024}, "num_warps": 4, "num_stages": 3}
    }


def test_record_autotune_winner_skips_empty_benchmark_winners():
    """No benchmark winners means no mutation to the kernel."""
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    from src.search.orchestrator import _record_autotune_winner

    spec = KernelSpec(name="t", kernel_type=KernelType.CUSTOM)
    kernel = Kernel(
        spec=spec, source_code="# placeholder",
        autotune_winner={"existing": {"kwargs": {}, "num_warps": 1, "num_stages": 1}},
    )
    bench = BenchmarkResult()

    _record_autotune_winner(kernel, bench)
    assert kernel.autotune_winner == {
        "existing": {"kwargs": {}, "num_warps": 1, "num_stages": 1}
    }


# ── A3: orchestrator passes condensed source to Planner / Reviewer ─────


@pytest.mark.asyncio
async def test_planner_receives_render_condensed_source_call(tmp_path, harness):
    """Per-iter Planner call site at orchestrator.py:773 should invoke
    parent.kernel.render_condensed_source(representative_workload_uuid=...)
    and pass the result to planner.plan(kernel_source=...). Interface
    test — content correctness is in test_kernel.py."""
    from sol_execbench.core.data import Workload

    sentinel = "# condensed parent source — from spy\n"
    harness.baseline.render_condensed_source = MagicMock(return_value=sentinel)

    wl_a = Workload.model_validate({
        "uuid": "wl-a", "axes": {"M": 4096}, "inputs": {},
    })
    workloads = [wl_a]

    # Provide a bench with per_workload_latency_us populated so the
    # orchestrator's representative-latency gate doesn't skip the iter.
    bench_with_wl = BenchmarkResult(
        median_latency_us=100.0,
        timed_runs=1,
        per_workload_latency_us={"wl-a": 100.0},
    )

    with (
        patch("src.eval.benchmark.benchmark_kernel", return_value=bench_with_wl),
        patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
    ):
        from src.search.orchestrator import Orchestrator
        orch = Orchestrator(
            harness.config, harness.planner, harness.coder,
            harness.reviewer, harness.retriever,
        )
        await orch.run(
            harness.baseline, workloads=workloads, roofline=harness.roofline,
        )

    assert harness.planner.plan.await_count >= 1
    planner_call = harness.planner.plan.await_args_list[0]
    assert planner_call.kwargs["kernel_source"] == sentinel
    # render_condensed_source on the baseline was invoked with the
    # representative uuid (workloads[0].uuid).
    harness.baseline.render_condensed_source.assert_any_call(
        representative_workload_uuid="wl-a",
    )


@pytest.mark.asyncio
async def test_reviewer_receives_render_condensed_source_call(tmp_path, harness):
    """Both Reviewer call sites (baseline review at orchestrator.py:694 +
    per-iter Reviewer at :1191) should invoke
    <kernel>.render_condensed_source(representative_workload_uuid=...)
    and pass the result to reviewer.review(kernel_source=...)."""
    from sol_execbench.core.data import Workload

    sentinel = "# condensed child source — from class-level spy\n"

    bench_with_wl = BenchmarkResult(
        median_latency_us=100.0,
        timed_runs=1,
        per_workload_latency_us={"wl-a": 100.0},
    )

    with (
        patch("src.eval.benchmark.benchmark_kernel", return_value=bench_with_wl),
        patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
        patch(
            "src.kernels.kernel.Kernel.render_condensed_source",
            return_value=sentinel,
        ),
    ):
        from src.search.orchestrator import Orchestrator
        orch = Orchestrator(
            harness.config, harness.planner, harness.coder,
            harness.reviewer, harness.retriever,
        )
        wl_a = Workload.model_validate({
            "uuid": "wl-a", "axes": {"M": 4096}, "inputs": {},
        })
        await orch.run(
            harness.baseline, workloads=[wl_a], roofline=harness.roofline,
        )

    assert harness.reviewer.review.await_count >= 1
    for call in harness.reviewer.review.await_args_list:
        assert call.kwargs["kernel_source"] == sentinel


# ── A2: _select_best_candidate (best-of-survivors) ────────────────────


def test_select_best_candidate_picks_highest_sol_score():
    """A2: best-of-survivors ranks by SOL Score, which (with fixed T_b
    + T_SOL) maps monotonically to lower latency wins."""
    from src.search.orchestrator import _select_best_candidate

    spec = KernelSpec(name="t", kernel_type=KernelType.MATMUL)

    # T_b = 100us, T_SOL = 50us. Lower T_k → higher SOL Score.
    cand_a = (0, MagicMock(), Kernel(spec=spec, source_code="# a"),
              BenchmarkResult(median_latency_us=90.0, timed_runs=1), None)
    cand_b = (1, MagicMock(), Kernel(spec=spec, source_code="# b"),
              BenchmarkResult(median_latency_us=70.0, timed_runs=1), None)
    cand_c = (2, MagicMock(), Kernel(spec=spec, source_code="# c"),
              BenchmarkResult(median_latency_us=80.0, timed_runs=1), None)

    winner = _select_best_candidate(
        [cand_a, cand_b, cand_c],
        t_sol_us=50.0,
        baseline_latency_us=100.0,
    )
    # Candidate B has the lowest latency → highest SOL Score → wins.
    assert winner[0] == 1  # candidate_idx
    assert winner[3].median_latency_us == 70.0  # bench result


def test_select_best_candidate_tie_break_lowest_candidate_idx():
    """A2: ties on SOL Score resolve to the lowest candidate_idx —
    deterministic first-survivor rule independent of list order."""
    from src.search.orchestrator import _select_best_candidate

    spec = KernelSpec(name="t", kernel_type=KernelType.MATMUL)

    cand_a = (0, MagicMock(), Kernel(spec=spec, source_code="# a"),
              BenchmarkResult(median_latency_us=80.0, timed_runs=1), None)
    cand_b = (1, MagicMock(), Kernel(spec=spec, source_code="# b"),
              BenchmarkResult(median_latency_us=80.0, timed_runs=1), None)

    # Iteration order shouldn't matter for the tie-break.
    winner = _select_best_candidate(
        [cand_b, cand_a],
        t_sol_us=50.0,
        baseline_latency_us=100.0,
    )
    assert winner[0] == 0


def test_select_best_candidate_single_survivor_wins_trivially():
    """A2 Q5a: single survivor (K-1 failed) becomes the winner even if
    its SOL Score is below the baseline anchor."""
    from src.search.orchestrator import _select_best_candidate

    spec = KernelSpec(name="t", kernel_type=KernelType.MATMUL)
    cand = (3, MagicMock(), Kernel(spec=spec, source_code="# only"),
            BenchmarkResult(median_latency_us=200.0, timed_runs=1), None)

    # T_k=200 > T_b=100 — score < 0.5; still wins as the only survivor.
    winner = _select_best_candidate(
        [cand],
        t_sol_us=50.0,
        baseline_latency_us=100.0,
    )
    assert winner[0] == 3


def test_select_best_candidate_below_sol_ranks_by_speed():
    """Codex review P2-A: when multiple candidates run below T_SOL,
    the actual scorer returns distinct values > 1.0 for each (monotonic
    in t_k). ``_select_best_candidate`` must rank by those true scores,
    not clamp everyone to 1.0 — otherwise the lowest-idx tie-break
    silently picks the slower below-SOL candidate.
    """
    from src.search.orchestrator import _select_best_candidate

    spec = KernelSpec(name="t", kernel_type=KernelType.MATMUL)

    # T_b = 100us, T_SOL = 50us. Both candidates are below T_SOL.
    # cand_a (idx=0): T_k = 40us — score ≈ 50 / (40-50 + 50) = 1.25
    # cand_b (idx=1): T_k = 20us — score ≈ 50 / (20-50 + 50) = 2.50
    # cand_b is genuinely faster (lower latency, higher score).
    # Under the buggy "below-SOL → 1.0" clamp, both tied at 1.0 and
    # cand_a (lower idx) would have won.
    cand_a = (0, MagicMock(), Kernel(spec=spec, source_code="# 40us"),
              BenchmarkResult(median_latency_us=40.0, timed_runs=1), None)
    cand_b = (1, MagicMock(), Kernel(spec=spec, source_code="# 20us"),
              BenchmarkResult(median_latency_us=20.0, timed_runs=1), None)

    winner = _select_best_candidate(
        [cand_a, cand_b],
        t_sol_us=50.0,
        baseline_latency_us=100.0,
    )
    assert winner[0] == 1  # cand_b (20us) wins, not cand_a (40us)


def test_select_best_candidate_calibration_warning_mirrors_scorer():
    """Codex review #2: in the calibration-warning regime (T_b <= T_SOL,
    baseline already at hardware bound), ``compute_sol_score`` returns
    1.0 only for ``t_k <= t_sol`` and 0.0 otherwise — NOT 1.0 for all
    survivors. ``_select_best_candidate`` must mirror this exactly so a
    slow above-SOL candidate cannot tie with a below-SOL one and steal
    the win via the lowest-idx tie-break.
    """
    from src.search.orchestrator import _select_best_candidate

    spec = KernelSpec(name="t", kernel_type=KernelType.MATMUL)

    # Calibration warning: baseline (50us) already at T_SOL (50us).
    # cand_a (idx=0): T_k = 200us > T_SOL → scorer says 0.0
    # cand_b (idx=1): T_k = 40us  <= T_SOL → scorer says 1.0
    # Under the buggy "1.0 for everyone in calibration regime" branch,
    # cand_a would win on lowest-idx tie-break. With the fix, cand_b
    # wins because its score is genuinely higher.
    cand_a = (0, MagicMock(), Kernel(spec=spec, source_code="# slow"),
              BenchmarkResult(median_latency_us=200.0, timed_runs=1), None)
    cand_b = (1, MagicMock(), Kernel(spec=spec, source_code="# below-sol"),
              BenchmarkResult(median_latency_us=40.0, timed_runs=1), None)

    winner = _select_best_candidate(
        [cand_a, cand_b],
        t_sol_us=50.0,
        baseline_latency_us=50.0,  # T_b == T_SOL → calibration warning
    )
    assert winner[0] == 1  # cand_b wins (below-SOL, score=1.0)


# ── A2: K-way Coder fan-out integration tests ─────────────────────────


@pytest.mark.asyncio
async def test_k_way_gather_dispatches_k_coder_calls(tmp_path, harness):
    """A2: orchestrator awaits coder.implement exactly K times per iter."""
    harness.config.coder_n_candidates = 3
    # K distinct outputs so survivors are distinguishable downstream.
    outputs = [
        KernelCodeOutput.model_construct(
            source_code=f"# candidate {i}", triton_kernel_name="",
        )
        for i in range(3)
    ]
    harness.coder.implement = AsyncMock(side_effect=outputs)

    await _run_orch(harness)

    # max_depth=1 → exactly one iter → exactly K=3 implement awaits.
    assert harness.coder.implement.await_count == 3


@pytest.mark.asyncio
async def test_k_way_best_of_survivors_selects_highest_sol_score(tmp_path, harness):
    """A2: among K successful candidates, the lowest-latency one (highest
    SOL Score) becomes the tree node."""
    harness.config.coder_n_candidates = 3
    outputs = [
        KernelCodeOutput.model_construct(
            source_code=f"# candidate {i}", triton_kernel_name="",
        )
        for i in range(3)
    ]
    harness.coder.implement = AsyncMock(side_effect=outputs)

    # First bench is the baseline (consumed by orchestrator before the
    # iter loop); the next three are the K candidates. Candidate 1 has
    # the lowest latency → highest SOL Score → wins.
    benches = [
        BenchmarkResult(median_latency_us=200.0, timed_runs=1),  # baseline
        BenchmarkResult(median_latency_us=90.0, timed_runs=1),
        BenchmarkResult(median_latency_us=70.0, timed_runs=1),
        BenchmarkResult(median_latency_us=85.0, timed_runs=1),
    ]

    from src.search.orchestrator import Orchestrator
    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", side_effect=benches),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
        ):
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

    # Winner's source_code is candidate 1's.
    best = result.tree.best_node()
    assert best.kernel.source_code == "# candidate 1"

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    coder_rec = next(r for r in records if r["kind"] == "coder_submitted")
    assert coder_rec["winner_candidate_idx"] == 1
    assert coder_rec["n_candidates"] == 3
    assert coder_rec["n_survivors"] == 3
    # Happy path: fastest candidate profiled cleanly on the first try.
    assert coder_rec["n_profile_attempts"] == 1


@pytest.mark.asyncio
async def test_k_way_all_fail_marks_iter_skipped(tmp_path, harness):
    """A2: all K Coder calls raising ImplementationError → iter SKIPPED;
    parent's consecutive_agent_failures increments; K coder_failed events
    fire with distinct candidate_idx values."""
    from src.agents.coder import ImplementationError
    from src.search.orchestrator import Orchestrator

    harness.config.coder_n_candidates = 3
    harness.coder.implement = AsyncMock(side_effect=[
        ImplementationError("compile failed: candidate 0"),
        ImplementationError("correctness failed: candidate 1"),
        ImplementationError("budget exhausted: candidate 2"),
    ])

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", return_value=harness.bench),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
        ):
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
    # K coder_failed events with candidate_idx in {0, 1, 2}.
    failed = [r for r in records if r["kind"] == "coder_failed"]
    assert len(failed) == 3
    assert sorted(r["candidate_idx"] for r in failed) == [0, 1, 2]
    # iter_end carries skipped outcome.
    iter_end = next(r for r in records if r["kind"] == "iter_end")
    assert iter_end["outcome"] == "skipped"


@pytest.mark.asyncio
async def test_k_way_partial_failure_picks_surviving_winner(tmp_path, harness):
    """A2: K-1 raise, 1 survives → survivor becomes the tree node and
    K-1 coder_failed events fire with the failing indices."""
    from src.agents.coder import ImplementationError
    from src.search.orchestrator import Orchestrator

    harness.config.coder_n_candidates = 3
    survivor_output = KernelCodeOutput.model_construct(
        source_code="# the only survivor", triton_kernel_name="",
    )
    harness.coder.implement = AsyncMock(side_effect=[
        ImplementationError("compile failed: candidate 0"),
        survivor_output,
        ImplementationError("budget exhausted: candidate 2"),
    ])

    # Survivor's bench is faster than baseline so best_node prefers it
    # over the root (root.score = 0.5 at baseline=baseline tie).
    baseline_bench = BenchmarkResult(median_latency_us=200.0, timed_runs=1)
    survivor_bench = BenchmarkResult(median_latency_us=80.0, timed_runs=1)

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch(
                "src.eval.benchmark.benchmark_kernel",
                side_effect=[baseline_bench, survivor_bench],
            ),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
        ):
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

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    failed = [r for r in records if r["kind"] == "coder_failed"]
    assert len(failed) == 2
    assert sorted(r["candidate_idx"] for r in failed) == [0, 2]
    best = result.tree.best_node()
    assert best.kernel.source_code == "# the only survivor"
    # The surviving candidate (idx=1) is recorded as winner.
    coder_rec = next(r for r in records if r["kind"] == "coder_submitted")
    assert coder_rec["winner_candidate_idx"] == 1
    assert coder_rec["n_survivors"] == 1


@pytest.mark.asyncio
async def test_k_way_profile_failure_falls_back_to_next_ranked(tmp_path, harness):
    """Codex review #1: when the fastest candidate is unprofileable
    (ProfilerError on the rank-1 candidate), the orchestrator falls back
    to the next-ranked instead of killing the iter. The slower-but-
    profileable candidate becomes the iter's winner and the iter
    advances. The failed candidate emits ``coder_failed`` with the
    profile-error reason for postmortem.
    """
    from src.eval.profiler import ProfilerError
    from src.search.orchestrator import Orchestrator

    harness.config.coder_n_candidates = 2
    outputs = [
        KernelCodeOutput.model_construct(
            source_code=f"# candidate {i}", triton_kernel_name="",
        )
        for i in range(2)
    ]
    harness.coder.implement = AsyncMock(side_effect=outputs)

    # Candidate 0 is fastest (would normally win the SOL-Score rank);
    # candidate 1 is slower but profileable.
    benches = [
        BenchmarkResult(median_latency_us=200.0, timed_runs=1),  # baseline
        BenchmarkResult(median_latency_us=60.0, timed_runs=1),   # cand 0
        BenchmarkResult(median_latency_us=80.0, timed_runs=1),   # cand 1
    ]

    # profile_kernel raises on candidate 0 (the rank-1 pick) and returns
    # a valid result for candidate 1 (the fall-back winner).
    profile_calls = {"n": 0}

    def profile_side_effect(*args, **kwargs):
        profile_calls["n"] += 1
        if profile_calls["n"] == 1:
            raise ProfilerError("zero analytical latency")
        return _make_profile()

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", side_effect=benches),
            patch("src.eval.profiler.profile_kernel", side_effect=profile_side_effect),
        ):
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

    # Profile attempted twice — once on cand 0 (fastest, fails), once
    # on cand 1 (slower, succeeds).
    assert profile_calls["n"] == 2

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]

    # Fall-back path emits per-candidate coder_failed for the rank-1
    # candidate that profile-killed.
    coder_failed = [r for r in records if r["kind"] == "coder_failed"]
    assert len(coder_failed) == 1
    assert coder_failed[0]["candidate_idx"] == 0
    assert "profile error" in coder_failed[0]["reason"]

    # Winner is candidate 1 (the slower but profileable survivor).
    coder_rec = next(r for r in records if r["kind"] == "coder_submitted")
    assert coder_rec["winner_candidate_idx"] == 1
    # n_profile_attempts == 2 — cand 0 profile-failed and we fell back
    # to cand 1, which succeeded.
    assert coder_rec["n_profile_attempts"] == 2
    best = result.tree.best_node()
    assert best.kernel.source_code == "# candidate 1"


@pytest.mark.asyncio
async def test_k_way_all_profile_fail_marks_iter_skipped(tmp_path, harness):
    """Codex review #1: when every K candidate is unprofileable, the
    iter SKIPS without bumping ``consecutive_agent_failures`` (profile
    errors are infra, not agent fault — matches the legacy single-Coder
    ``_kill_branch(PROFILER_ERROR, bumps_agent_failures=False)``).
    """
    from src.eval.profiler import ProfilerError
    from src.search.orchestrator import Orchestrator
    from src.search.tree import QUARANTINE_THRESHOLD

    harness.config.coder_n_candidates = 2
    outputs = [
        KernelCodeOutput.model_construct(
            source_code=f"# candidate {i}", triton_kernel_name="",
        )
        for i in range(2)
    ]
    harness.coder.implement = AsyncMock(side_effect=outputs)

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", return_value=harness.bench),
            patch(
                "src.eval.profiler.profile_kernel",
                side_effect=ProfilerError("zero analytical latency"),
            ),
        ):
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

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # Both candidates emit coder_failed for profile error.
    coder_failed = [r for r in records if r["kind"] == "coder_failed"]
    assert len(coder_failed) == 2
    assert sorted(r["candidate_idx"] for r in coder_failed) == [0, 1]
    # iter_end SKIPPED.
    end = next(r for r in records if r["kind"] == "iter_end")
    assert end["outcome"] == "skipped"
    # Root's quarantine counter is NOT bumped (infra failures don't
    # quarantine the parent).
    assert result.tree.get_node(0).consecutive_agent_failures < QUARANTINE_THRESHOLD


@pytest.mark.asyncio
async def test_k_way_channel_a_reward_hack_aborts_iter_no_sibling_fallback(
    tmp_path, harness,
):
    """Codex review P1: when one of the K candidates raises
    ``RewardHackDetected`` during its bench, the iter aborts
    immediately — *no* earlier-benched sibling gets profiled or
    committed. ``per_iter_anti_cheat`` detects monkey-patching but does
    not restore the patched primitives, so any further work in the iter
    runs against a tainted process. The legacy K-way design (continue
    to next sibling) was unsafe; this test pins the corrected semantic.
    """
    from src.search.orchestrator import Orchestrator
    from sol_execbench.core.bench.reward_hack import RewardHackDetected

    harness.config.coder_n_candidates = 3
    outputs = [
        KernelCodeOutput.model_construct(
            source_code=f"# candidate {i}", triton_kernel_name="",
        )
        for i in range(3)
    ]
    harness.coder.implement = AsyncMock(side_effect=outputs)

    # Candidate 0 + 1 bench cleanly; candidate 2 raises RewardHackDetected.
    # Without the abort, candidates 0/1 are still in bench_results and
    # one of them would be profiled + committed.
    baseline_bench = BenchmarkResult(median_latency_us=200.0, timed_runs=1)
    clean_bench_a = BenchmarkResult(median_latency_us=80.0, timed_runs=1)
    clean_bench_b = BenchmarkResult(median_latency_us=90.0, timed_runs=1)

    def bench_side_effect(*args, **kwargs):
        return bench_seq.pop(0)

    bench_seq = [baseline_bench, clean_bench_a, clean_bench_b]

    def bench_then_hack(*args, **kwargs):
        if bench_seq:
            return bench_seq.pop(0)
        raise RewardHackDetected(
            "candidate 2 monkey-patched torch.cuda.Event.elapsed_time",
        )

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", side_effect=bench_then_hack),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
        ):
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

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # Dedicated channel-A event fires for the cheating candidate.
    rh_detected = [r for r in records if r["kind"] == "reward_hack_detected"]
    assert len(rh_detected) == 1
    assert rh_detected[0]["candidate_idx"] == 2

    # CRITICAL: no tree node is added for the iter. The earlier clean-
    # benched siblings (cand 0, 1) are NOT promoted under a tainted
    # process. ``best_node()`` falls back to root.
    children = [n for n in result.tree._nodes.values() if n.parent_id is not None]
    assert children == []
    assert result.best_node.id == 0  # root

    # iter_end SKIPPED. RewardHackDetected is agent-fault so the
    # parent's quarantine counter bumps.
    end = next(r for r in records if r["kind"] == "iter_end")
    assert end["outcome"] == "skipped"
    assert result.tree.get_node(0).consecutive_agent_failures >= 1


@pytest.mark.asyncio
async def test_k_way_partial_coder_success_resets_quarantine_counter(tmp_path, harness):
    """Partial Coder success (some K candidates fail agent-side, at least
    one succeeds) is "any Coder success" and clears the parent's
    quarantine counter even when downstream bench fails infra-only.
    The pre-K-way semantic was "Coder success → reset"; under K-way at
    T=1.0, partial success is the *expected* case (LLM decoder variance),
    not the exception. Mirror it.

    Sequence: iter 1 planner fails → counter=1.
    iter 2: K=2 Coder calls — one ImplementationError, one succeeds; the
    survivor reaches bench then bench infra-fails → counter must reset
    to 0 (not stay at 1 or bump to 2).
    iter 3: planner fails → counter=1 (not 2), parent stays eligible.
    """
    from src.agents.coder import ImplementationError
    from src.agents.planner import PlanningError
    from src.eval.benchmark import BenchmarkError, BenchmarkResult
    from src.search.orchestrator import Orchestrator
    from src.search.tree import QUARANTINE_THRESHOLD

    harness.config.coder_n_candidates = 2
    harness.config.max_depth = 3
    # K=2 per iter: 1 agent-fail, 1 succeed. Repeat across iters that
    # reach the Coder phase.
    success_output = KernelCodeOutput.model_construct(
        source_code="# child", triton_kernel_name="",
    )
    harness.coder.implement = AsyncMock(side_effect=[
        # iter 2 (iter 1 fails at planner):
        ImplementationError("turn budget"), success_output,
        # iter 3 (planner fails again, no Coder calls)
    ])
    plan_seq = [
        PlanningError("turn budget exhausted"),
        OptimizationPlan(
            tier=1, technique="tiling", params={}, target_region="",
            rationale="r",
        ),
        PlanningError("turn budget exhausted"),
    ]
    harness.planner.plan = AsyncMock(side_effect=plan_seq)

    baseline_bench = BenchmarkResult(median_latency_us=100.0, timed_runs=1)

    def bench_side_effect(*args, **kwargs):
        if bench_seq:
            return bench_seq.pop(0)
        raise BenchmarkError("0/3 workloads survived")

    bench_seq = [baseline_bench]  # baseline only

    with (
        patch("src.eval.benchmark.benchmark_kernel", side_effect=bench_side_effect),
        patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
    ):
        orch = Orchestrator(
            harness.config, harness.planner, harness.coder,
            harness.reviewer, harness.retriever,
        )
        result = await orch.run(
            harness.baseline, workloads=None, roofline=harness.roofline,
        )

    # Counter shouldn't reach QUARANTINE_THRESHOLD: iter 2's partial
    # Coder success resets to 0 even though one of the 2 K calls failed
    # agent-side; iter 3's planner failure bumps back to 1, not 2.
    root_counter = result.tree.get_node(0).consecutive_agent_failures
    assert root_counter < QUARANTINE_THRESHOLD, (
        f"partial Coder success must reset quarantine counter even when "
        f"K-1 candidates failed agent-side; counter={root_counter}"
    )


@pytest.mark.asyncio
async def test_k_way_infra_only_skip_resets_quarantine_counter(tmp_path, harness):
    """Codex review P2-B: when an iter SKIPS because every K candidate
    failed for *infra* reasons (bench/profile), the parent's
    ``consecutive_agent_failures`` counter resets to 0 — mirroring the
    pre-K-way single-Coder semantic of clearing the counter the moment
    valid Coder output is committed to the tree.

    Multi-iter sequence pinned here:
      iter 1: planner raises → counter = 1 (agent fault)
      iter 2: Coder produces K valid candidates → all fail bench (infra)
              → counter resets to 0 (not stuck at 1)
      iter 3: planner raises again → counter = 1 (NOT 2, so the parent
              is NOT pre-emptively quarantined)
    """
    from src.agents.planner import PlanningError
    from src.eval.benchmark import BenchmarkError, BenchmarkResult
    from src.search.orchestrator import Orchestrator
    from src.search.tree import QUARANTINE_THRESHOLD

    harness.config.coder_n_candidates = 2
    harness.config.max_depth = 3
    harness.coder.implement = AsyncMock(
        return_value=KernelCodeOutput.model_construct(
            source_code="# child", triton_kernel_name="",
        )
    )
    # iter 1: planner fault. iter 2: planner success. iter 3: planner fault.
    plan_seq = [
        PlanningError("turn budget exhausted"),
        OptimizationPlan(
            tier=1, technique="tiling", params={}, target_region="",
            rationale="r",
        ),
        PlanningError("turn budget exhausted"),
    ]
    harness.planner.plan = AsyncMock(side_effect=plan_seq)

    baseline_bench = BenchmarkResult(median_latency_us=100.0, timed_runs=1)

    def bench_side_effect(*args, **kwargs):
        if bench_seq:
            return bench_seq.pop(0)
        raise BenchmarkError("0/3 workloads survived")

    bench_seq = [baseline_bench]  # baseline only; child benches raise

    with (
        patch("src.eval.benchmark.benchmark_kernel", side_effect=bench_side_effect),
        patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
    ):
        orch = Orchestrator(
            harness.config, harness.planner, harness.coder,
            harness.reviewer, harness.retriever,
        )
        result = await orch.run(
            harness.baseline, workloads=None, roofline=harness.roofline,
        )

    # If P2-B is buggy: iter 1 bumps to 1, iter 2 leaves at 1, iter 3
    # bumps to 2 = QUARANTINE_THRESHOLD → root quarantined → ALL_DEAD_END
    # and counter ends at 2. With the fix: iter 2's infra-only skip
    # resets to 0, iter 3 bumps back to 1 → no quarantine.
    root_counter = result.tree.get_node(0).consecutive_agent_failures
    assert root_counter < QUARANTINE_THRESHOLD, (
        f"root quarantine counter must not reach QUARANTINE_THRESHOLD "
        f"when only one agent failure precedes the final iter; got {root_counter}"
    )


@pytest.mark.asyncio
async def test_k_way_per_candidate_trace_span_per_call(tmp_path, harness, monkeypatch):
    """Codex review P3: under K-way fan-out, each ``coder.implement`` call
    runs inside its own ``trace_span`` so the agents SDK trace closes K
    times per iter and ``UsageAccumulator.invocations`` ticks K times.
    A single outer ``trace_span`` would close once and underreport the
    coder call count by K× in ``usage.json`` and the cost report.
    """
    from src.runtime.usage import AgentLabel
    from src.search import orchestrator as orch

    harness.config.coder_n_candidates = 3
    outputs = [
        KernelCodeOutput.model_construct(
            source_code=f"# candidate {i}", triton_kernel_name="",
        )
        for i in range(3)
    ]
    harness.coder.implement = AsyncMock(side_effect=outputs)

    coder_trace_calls: list[dict] = []
    real = orch.trace_span

    def spy(workflow_name, *, iter_no, agent, **extra):
        # Capture all coder-agent trace_span invocations so we can count
        # them. The agents-SDK contextvar plumbing is exercised by the
        # real `trace_span` underneath.
        if agent == AgentLabel.CODER:
            coder_trace_calls.append({
                "workflow": workflow_name, "iter": iter_no,
                "candidate_idx": extra.get("candidate_idx"),
            })
        return real(workflow_name, iter_no=iter_no, agent=agent, **extra)

    monkeypatch.setattr(orch, "trace_span", spy)
    await _run_orch(harness)

    # K=3 coder.implement calls in iter 1 → 3 trace_span entries, each
    # carrying a distinct candidate_idx in {0, 1, 2}.
    assert len(coder_trace_calls) == 3
    assert sorted(c["candidate_idx"] for c in coder_trace_calls) == [0, 1, 2]
    assert all(c["iter"] == 1 for c in coder_trace_calls)


@pytest.mark.asyncio
async def test_k_way_per_candidate_anti_cheat_scopes_isolated(tmp_path, harness):
    """A2 + Codex finding #1: each surviving candidate enters its own
    per_iter_anti_cheat context (not one shared snapshot for all K)."""
    from contextlib import contextmanager
    from src.search.orchestrator import Orchestrator

    harness.config.coder_n_candidates = 3
    outputs = [
        KernelCodeOutput.model_construct(
            source_code=f"# candidate {i}", triton_kernel_name="",
        )
        for i in range(3)
    ]
    harness.coder.implement = AsyncMock(side_effect=outputs)

    enter_count = {"n": 0}

    @contextmanager
    def _spy_anti_cheat(critical_names):
        enter_count["n"] += 1
        yield MagicMock(snapshot={}, namespace={}, threads_before=0)

    with (
        patch("src.eval.benchmark.benchmark_kernel", return_value=harness.bench),
        patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
        patch("src.eval.anti_cheat.per_iter_anti_cheat", _spy_anti_cheat),
    ):
        orch = Orchestrator(
            harness.config, harness.planner, harness.coder,
            harness.reviewer, harness.retriever,
        )
        await orch.run(
            harness.baseline, workloads=None, roofline=harness.roofline,
        )

    # K=3 candidates → 3 separate per_iter_anti_cheat entries (one per
    # candidate's per_iter bench, not one shared for all K).
    assert enter_count["n"] == 3
