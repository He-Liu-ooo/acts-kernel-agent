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
            arithmetic_intensity=10.0,
            ridge_point=20.0,
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
        arithmetic_intensity=1.0,
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
    # would slip into events.jsonl — these break RFC-8259 consumers
    # (see Codex review 2026-04-23 Finding 2). Python's default
    # ``json.loads`` allows Infinity; we force-reject it here to act
    # as a regression guard.
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
