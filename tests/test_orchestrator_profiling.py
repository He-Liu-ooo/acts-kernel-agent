"""Orchestrator + profiler wiring tests.

Tier 1 (no GPU): ``profile_kernel`` is monkeypatched to a scripted fake
so the orchestrator's interaction with the real profiler surface is
covered without requiring torch / ncu on the host. The Tier 2 file
``tests/test_profiler_gpu.py`` exercises the real subprocess path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import rtx6000_ada_hardware as _rtx6000_ada
from src.agents.planner import OptimizationPlan
from src.agents.reviewer import BranchQuality, ReviewerFeedback
from src.config import ACTSConfig, HardwareSpec
from src.eval.benchmark import BenchmarkResult
from src.eval.profiler import (
    AnalyticalMetrics,
    NCUMetrics,
    ProfilerError,
    ProfilingResult,
)
from src.eval.roofline import RooflineResult
from src.eval.types import BottleneckType
from src.kernels.kernel import Kernel, KernelSpec, KernelType


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


def _make_analytical() -> AnalyticalMetrics:
    return AnalyticalMetrics(
        arithmetic_intensity=10.0,
        ridge_point=20.0,
        achieved_tflops=1.0,
        achieved_bandwidth_gb_s=100.0,
        pct_peak_compute=0.1,
        pct_peak_bandwidth=0.5,
    )


def _make_ncu() -> NCUMetrics:
    return NCUMetrics(
        sm_occupancy_pct=72.5,
        l2_hit_rate_pct=45.0,
        tensor_core_util_pct=0.0,
        warp_stall_dominant="long_scoreboard",
        warp_stall_dominant_pct=33.0,
        warp_stall_runner_up="wait",
        warp_stall_runner_up_pct=18.0,
    )


def _make_profile(
    *,
    degraded_reason: str | None = None,
    has_ncu: bool = True,
) -> ProfilingResult:
    return ProfilingResult(
        analytical=_make_analytical(),
        ncu=_make_ncu() if has_ncu and degraded_reason is None else None,
        raw_metrics={},
        degraded_reason=degraded_reason,
    )


@pytest.fixture
def harness():
    """Orchestrator harness with profile_kernel patched out by default.

    ``patch_profile`` is a callable the test uses to inject the fake's
    behavior (``side_effect`` or ``return_value``) before calling
    ``run_orch``.
    """
    config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=1,
        beam_width=3,
        sol_plateau_window=99,
    )
    planner = MagicMock()
    planner.plan = AsyncMock(return_value=OptimizationPlan(
        tier=1, technique="tiling", params={}, target_region="", rationale="",
    ))
    coder = MagicMock()
    coder.implement = AsyncMock(return_value="# child source")
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


async def _run_orch(h, *, profile_fake=None):
    from src.search.orchestrator import Orchestrator

    if profile_fake is None:
        profile_fake = MagicMock(return_value=_make_profile())
    with (
        patch("src.eval.benchmark.benchmark_kernel", return_value=h.bench),
        patch("src.eval.profiler.profile_kernel", profile_fake),
    ):
        orch = Orchestrator(h.config, h.planner, h.coder, h.reviewer, h.retriever)
        return await orch.run(h.baseline, workloads=None, roofline=h.roofline), profile_fake


# ── tests ──────────────────────────────────────────────────────────────


class TestProfilingAttachedToTree:
    @pytest.mark.asyncio
    async def test_child_node_gets_profiling_on_success(self, harness):
        """After a successful iteration, the child TreeNode carries the
        ProfilingResult returned by profile_kernel."""
        result, profile_fake = await _run_orch(harness)

        assert profile_fake.call_count == 1, (
            "profile_kernel should run exactly once per iteration (one child here)"
        )
        children = [n for n in result.tree._nodes.values() if n.parent_id is not None]
        assert len(children) == 1
        assert children[0].profiling is not None
        assert children[0].profiling.ncu is not None

    @pytest.mark.asyncio
    async def test_root_profiling_stays_none(self, harness):
        """Baseline never profiles — the root node should keep profiling=None."""
        result, _ = await _run_orch(harness)
        assert result.tree.get_node(0).profiling is None


class TestProfilingSkippedOnDeadBranch:
    @pytest.mark.asyncio
    async def test_partial_workload_failure_skips_profile(self, harness):
        """Benchmark partial-failure marks the child DEAD_END and profile_kernel
        is NOT called — there's no valid latency to pass it."""
        from src.search.orchestrator import Orchestrator

        baseline_bench = BenchmarkResult(median_latency_us=100.0, timed_runs=1)
        partial_child = BenchmarkResult(
            median_latency_us=50.0,
            timed_runs=1,
            per_workload_latency_us={"wl1": 50.0, "wl2": float("inf")},
            workload_errors={"wl2": "RuntimeError: CUDA launch failed"},
        )
        profile_fake = MagicMock(return_value=_make_profile())

        with (
            patch(
                "src.eval.benchmark.benchmark_kernel",
                side_effect=[baseline_bench, partial_child],
            ),
            patch("src.eval.profiler.profile_kernel", profile_fake),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            result = await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)

        assert profile_fake.call_count == 0, (
            "profile_kernel must not run when the child benchmark had partial failures"
        )
        child = [n for n in result.tree._nodes.values() if n.parent_id is not None][0]
        assert child.branch_quality is BranchQuality.DEAD_END
        assert child.profiling is None


class TestProfilerErrorMarksDeadEnd:
    @pytest.mark.asyncio
    async def test_analytical_failure_marks_dead_end(self, harness):
        """ProfilerError raised by profile_kernel (e.g. zero-latency analytical
        failure) marks the child DEAD_END and the orchestrator continues."""
        profile_fake = MagicMock(
            side_effect=ProfilerError("latency_s must be positive, got 0.0")
        )
        result, _ = await _run_orch(harness, profile_fake=profile_fake)

        child = [n for n in result.tree._nodes.values() if n.parent_id is not None][0]
        assert child.branch_quality is BranchQuality.DEAD_END
        assert child.profiling is None
        # Reviewer must be skipped when profile fails — no profile to hand it.
        assert harness.reviewer.review.await_count == 0


class TestDegradedProfileKeepsBranchAlive:
    @pytest.mark.asyncio
    async def test_degraded_ncu_still_attaches_profile(self, harness):
        """When NCU fails (subprocess non-zero, binary missing, parse error),
        profile_kernel returns degraded=True with ncu=None. The analytical
        metrics are still valid — the branch must stay alive and the
        degraded profile must still be attached to the node."""
        profile_fake = MagicMock(
            return_value=_make_profile(
                degraded_reason="ncu_binary_not_found",
                has_ncu=False,
            )
        )
        result, _ = await _run_orch(harness, profile_fake=profile_fake)

        child = [n for n in result.tree._nodes.values() if n.parent_id is not None][0]
        assert child.branch_quality is BranchQuality.PROMISING  # from reviewer mock
        assert child.profiling is not None
        assert child.profiling.degraded is True
        assert child.profiling.ncu is None


class TestRetrieverReceivesRunBottleneck:
    @pytest.mark.asyncio
    async def test_every_iteration_uses_run_level_bottleneck(self, harness):
        """Bottleneck classification is invariant per
        ``(problem, representative_workload, hardware)`` — see
        ``classify_run``. Every retrieve call across the run must receive
        the same run-level bottleneck, not a per-iter profiling-derived one.
        """
        from src.search.orchestrator import Orchestrator

        harness.config.max_depth = 2  # force two iterations
        profile_fake = MagicMock(side_effect=[_make_profile(), _make_profile()])
        with (
            patch("src.eval.benchmark.benchmark_kernel", return_value=harness.bench),
            patch("src.eval.profiler.profile_kernel", profile_fake),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)

        retrieve_calls = harness.retriever.retrieve.call_args_list
        assert len(retrieve_calls) >= 1
        # Roofline.bottleneck was MEMORY_BOUND — every retrieve call sees
        # exactly that, independent of iteration.
        for call in retrieve_calls:
            assert call.args[1] is BottleneckType.MEMORY_BOUND, (
                f"retrieve must always use run-level bottleneck; got {call.args[1]}"
            )


class TestRunBottleneckInReport:
    @pytest.mark.asyncio
    async def test_report_carries_run_level_bottleneck(self, harness):
        """``generate_report`` takes its ``bottleneck`` from
        ``result.run_bottleneck`` — the once-per-run classification that
        drove retriever / planner / reviewer."""
        from src.pipeline.report import generate_report
        from src.search.orchestrator import Orchestrator

        harness.config.max_depth = 2
        baseline_bench = BenchmarkResult(median_latency_us=100.0, timed_runs=1)
        child_bench = BenchmarkResult(median_latency_us=70.0, timed_runs=1)
        profile_fake = MagicMock(side_effect=[_make_profile(), _make_profile()])
        with (
            patch(
                "src.eval.benchmark.benchmark_kernel",
                side_effect=[baseline_bench, child_bench, child_bench],
            ),
            patch("src.eval.profiler.profile_kernel", profile_fake),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            result = await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)

        report = generate_report(result)
        # Roofline.bottleneck was MEMORY_BOUND — that's the run-level value.
        assert report.bottleneck is BottleneckType.MEMORY_BOUND
        # Placeholder path (no workloads, no problem): per-workload
        # bottlenecks stays empty — classify_workload requires Problem.
        assert report.winner_per_workload_bottlenecks == {}


class TestReviewerReceivesProfiling:
    @pytest.mark.asyncio
    async def test_reviewer_gets_profiling_kwarg(self, harness):
        """Each live iteration's reviewer.review call must receive the
        child's ProfilingResult via the profiling= kwarg — not just a
        stringified summary."""
        result, _ = await _run_orch(harness)

        assert harness.reviewer.review.await_count == 1
        kwargs = harness.reviewer.review.await_args.kwargs
        assert "profiling" in kwargs, "Reviewer must receive profiling= kwarg"
        assert isinstance(kwargs["profiling"], ProfilingResult)
        # Run-level bottleneck (invariant across iterations) lands on the
        # reviewer via the ``bottleneck=`` kwarg, not via the profile.
        assert kwargs["bottleneck"] is BottleneckType.MEMORY_BOUND


class TestReportWinnerReprofile:
    """Phase C re-profile on all selected workloads (spec §3.4)."""

    def test_generate_report_reprofiles_winner_per_workload(self, harness):
        """When ``workloads`` + ``input_generators`` + ``hardware_spec`` are
        supplied, ``generate_report`` calls ``profile_kernel`` once per
        workload and stores the result keyed by workload UUID."""
        from src.benchmark.problem import Workload
        from src.eval.scorer import ScoreResult
        from src.pipeline.report import generate_report
        from src.search.orchestrator import SearchResult, TerminationReason
        from src.search.tree import SearchTree

        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = ScoreResult(
            sol_score=0.3,
            baseline_latency_us=100.0,
            candidate_latency_us=100.0,
            t_sol_us=50.0,
            speedup=1.0,
        )
        child = tree.add_child(root.id, _make_kernel("child"), "tiling")
        child.score = ScoreResult(
            sol_score=0.8,
            baseline_latency_us=100.0,
            candidate_latency_us=60.0,
            t_sol_us=50.0,
            speedup=1.67,
        )
        child.profiling = _make_profile()
        search_result = SearchResult(
            best_node=child,
            total_iterations=1,
            termination_reason=TerminationReason.BUDGET,
            tree=tree,
        )

        workloads = [
            Workload(uuid="wl0", axes={"N": 128}),
            Workload(uuid="wl1", axes={"N": 256}),
            Workload(uuid="wl2", axes={"N": 512}),
        ]
        gens = [lambda s, i=i: (i, s) for i in range(3)]

        profile_fake = MagicMock(side_effect=[
            _make_profile(),
            _make_profile(),
            _make_profile(),
        ])
        with patch("src.eval.profiler.profile_kernel", profile_fake):
            report = generate_report(
                search_result,
                workloads=workloads,
                input_generators=gens,
                hardware_spec=harness.config.hardware,
            )

        assert profile_fake.call_count == 3
        assert set(report.winner_profiling_per_workload.keys()) == {"wl0", "wl1", "wl2"}

    def test_generate_report_no_reprofile_when_workloads_missing(self, harness):
        """Placeholder path: no workloads → empty per-workload dict, no
        profile_kernel calls."""
        from src.eval.scorer import ScoreResult
        from src.pipeline.report import generate_report
        from src.search.orchestrator import SearchResult, TerminationReason
        from src.search.tree import SearchTree

        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = ScoreResult(
            sol_score=0.3,
            baseline_latency_us=100.0,
            candidate_latency_us=100.0,
            t_sol_us=50.0,
            speedup=1.0,
        )
        search_result = SearchResult(
            best_node=root,
            total_iterations=0,
            termination_reason=TerminationReason.BUDGET,
            tree=tree,
        )

        profile_fake = MagicMock(return_value=_make_profile())
        with patch("src.eval.profiler.profile_kernel", profile_fake):
            report = generate_report(search_result)

        assert profile_fake.call_count == 0
        assert report.winner_profiling_per_workload == {}


class TestReportRenderingIncludesProfiling:
    def test_render_report_emits_analytical_and_ncu_blocks(self):
        """Rendered report should surface the run-level bottleneck,
        per-workload bottlenecks, and the analytical + NCU blocks."""
        from src.pipeline.report import OptimizationReport, render_report

        report = OptimizationReport(
            baseline_latency_us=100.0,
            best_latency_us=60.0,
            sol_score=0.8,
            speedup=1.67,
            bottleneck=BottleneckType.COMPUTE_BOUND,
            winner_per_workload_bottlenecks={
                "wl0": BottleneckType.COMPUTE_BOUND,
                "wl1": BottleneckType.BALANCED,
            },
            winner_profiling_per_workload={"wl0": _make_profile()},
            total_iterations=2,
            termination_reason="budget",
        )
        text = render_report(report)

        assert "Bottleneck (run)" in text
        assert "compute_bound" in text
        assert "Bottleneck (per workload)" in text
        assert "balanced" in text
        assert "TFLOPS" in text
        assert "NCU" in text  # NCU block header
        assert "long_scoreboard" in text  # top stall surfaced

    def test_render_report_suppresses_ncu_block_when_all_missing(self):
        """When every per-workload profile is degraded with ncu_binary_not_found
        (common on machines without the NCU CLI), skip the NCU lines to keep
        the output tidy."""
        from src.pipeline.report import OptimizationReport, render_report

        degraded = _make_profile(
            degraded_reason="ncu_binary_not_found",
            has_ncu=False,
        )
        report = OptimizationReport(
            baseline_latency_us=100.0,
            best_latency_us=60.0,
            sol_score=0.8,
            speedup=1.67,
            winner_profiling_per_workload={"wl0": degraded, "wl1": degraded},
            total_iterations=2,
            termination_reason="budget",
        )
        text = render_report(report)

        # Analytical block still present; NCU / degraded notice suppressed.
        assert "TFLOPS" in text
        assert "NCU" not in text
        assert "DEGRADED" not in text


class TestPerWorkloadLatencyPersistedOnNode:
    """``BenchmarkResult.per_workload_latency_us`` is the only per-workload
    signal we have — the aggregate median hides cross-workload variance.
    Phase C re-profiles the winner per workload and needs each workload's
    own latency to produce the right TFLOPs/bandwidth metrics. The
    orchestrator must persist that dict on the child TreeNode."""

    @pytest.mark.asyncio
    async def test_child_gets_per_workload_latency_from_bench(self, harness):
        """After a successful child iteration, the TreeNode should carry
        the benchmark's ``per_workload_latency_us`` so later re-profiling
        can use each workload's real latency."""
        from src.benchmark.problem import Workload
        from src.search.orchestrator import Orchestrator

        workloads = [
            Workload(uuid="wl0", axes={}),
            Workload(uuid="wl1", axes={}),
        ]
        baseline_bench = BenchmarkResult(
            median_latency_us=100.0,
            timed_runs=1,
            per_workload_latency_us={"wl0": 100.0, "wl1": 110.0},
        )
        child_bench = BenchmarkResult(
            median_latency_us=60.0,
            timed_runs=1,
            per_workload_latency_us={"wl0": 55.0, "wl1": 80.0},
        )
        profile_fake = MagicMock(return_value=_make_profile())
        input_generators = [lambda seed: (), lambda seed: ()]

        with (
            patch(
                "src.eval.benchmark.benchmark_kernel",
                side_effect=[baseline_bench, child_bench],
            ),
            patch("src.eval.profiler.profile_kernel", profile_fake),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            result = await orch.run(
                harness.baseline,
                workloads=workloads,
                roofline=harness.roofline,
                input_generators=input_generators,
            )

        children = [n for n in result.tree._nodes.values() if n.parent_id is not None]
        assert len(children) == 1
        assert children[0].per_workload_latency_us == {"wl0": 55.0, "wl1": 80.0}
        # Root never benchmarked a child via the iteration loop — its field
        # stays ``None`` (baseline bench is a separate call; we don't
        # persist it on the root to match the existing profiling=None
        # convention for the root).
        assert result.tree.get_node(0).per_workload_latency_us is None


class TestReportUsesPerWorkloadLatency:
    """``generate_report`` re-profiles the winner per workload. Each call
    must use the workload's *own* latency from
    ``best.per_workload_latency_us``, not the aggregate
    ``candidate_latency_us`` — otherwise analytical TFLOPs / bandwidth
    are wrong for any workload whose latency diverges from the aggregate."""

    def test_reprofile_uses_per_workload_latency_when_available(self, harness):
        from src.benchmark.problem import Workload
        from src.eval.scorer import ScoreResult
        from src.pipeline.report import generate_report
        from src.search.orchestrator import SearchResult, TerminationReason
        from src.search.tree import SearchTree

        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = ScoreResult(
            sol_score=0.3, baseline_latency_us=100.0, candidate_latency_us=100.0,
            t_sol_us=50.0, speedup=1.0,
        )
        child = tree.add_child(root.id, _make_kernel("child"), "tiling")
        child.score = ScoreResult(
            sol_score=0.8, baseline_latency_us=100.0, candidate_latency_us=60.0,
            t_sol_us=50.0, speedup=1.67,
        )
        child.profiling = _make_profile()
        child.per_workload_latency_us = {
            "wl0": 40.0,   # faster than aggregate
            "wl1": 60.0,   # matches aggregate
            "wl2": 100.0,  # slower than aggregate
        }
        search_result = SearchResult(
            best_node=child,
            total_iterations=1,
            termination_reason=TerminationReason.BUDGET,
            tree=tree,
        )
        workloads = [Workload(uuid=u, axes={}) for u in ("wl0", "wl1", "wl2")]
        gens = [lambda s: () for _ in range(3)]

        profile_fake = MagicMock(return_value=_make_profile())
        with patch("src.eval.profiler.profile_kernel", profile_fake):
            generate_report(
                search_result,
                workloads=workloads,
                input_generators=gens,
                hardware_spec=harness.config.hardware,
            )

        assert profile_fake.call_count == 3
        latencies = [call.kwargs["latency_s"] for call in profile_fake.call_args_list]
        assert latencies == [40.0 / 1e6, 60.0 / 1e6, 100.0 / 1e6]

    def test_reprofile_falls_back_to_aggregate_when_no_per_workload_dict(self, harness):
        """Legacy checkpoint (or placeholder path): no per_workload_latency_us
        on the node. The re-profile loop must fall back to the aggregate so
        old trees still render."""
        from src.benchmark.problem import Workload
        from src.eval.scorer import ScoreResult
        from src.pipeline.report import generate_report
        from src.search.orchestrator import SearchResult, TerminationReason
        from src.search.tree import SearchTree

        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = ScoreResult(
            sol_score=0.3, baseline_latency_us=100.0, candidate_latency_us=100.0,
            t_sol_us=50.0, speedup=1.0,
        )
        child = tree.add_child(root.id, _make_kernel("child"), "tiling")
        child.score = ScoreResult(
            sol_score=0.8, baseline_latency_us=100.0, candidate_latency_us=60.0,
            t_sol_us=50.0, speedup=1.67,
        )
        child.profiling = _make_profile()
        # No per_workload_latency_us set — legacy node.
        search_result = SearchResult(
            best_node=child,
            total_iterations=1,
            termination_reason=TerminationReason.BUDGET,
            tree=tree,
        )
        workloads = [Workload(uuid=u, axes={}) for u in ("wl0", "wl1")]
        gens = [lambda s: () for _ in range(2)]

        profile_fake = MagicMock(return_value=_make_profile())
        with patch("src.eval.profiler.profile_kernel", profile_fake):
            generate_report(
                search_result,
                workloads=workloads,
                input_generators=gens,
                hardware_spec=harness.config.hardware,
            )

        assert profile_fake.call_count == 2
        latencies = [call.kwargs["latency_s"] for call in profile_fake.call_args_list]
        # Both calls fall back to the aggregate candidate_latency_us (60us).
        assert latencies == [60.0 / 1e6, 60.0 / 1e6]

    def test_reprofile_forwards_problem_definition_path_when_problem_given(
        self, harness, tmp_path
    ):
        """When ``problem`` is passed, the re-profile must forward
        ``problem.definition_path`` to ``profile_kernel`` so the NCU
        subprocess can rebuild inputs. Without this, the driver ignores
        the in-process input_generator and NCU silently degrades."""
        from src.benchmark.problem import (
            AxisDef,
            Problem,
            TensorDef,
            Workload,
        )
        from src.eval.scorer import ScoreResult
        from src.pipeline.report import generate_report
        from src.search.orchestrator import SearchResult, TerminationReason
        from src.search.tree import SearchTree

        definition_path = tmp_path / "definition.json"
        definition_path.write_text("{}")  # never read by the fake

        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = ScoreResult(
            sol_score=0.3, baseline_latency_us=100.0, candidate_latency_us=100.0,
            t_sol_us=50.0, speedup=1.0,
        )
        child = tree.add_child(root.id, _make_kernel("child"), "tiling")
        child.score = ScoreResult(
            sol_score=0.8, baseline_latency_us=100.0, candidate_latency_us=60.0,
            t_sol_us=50.0, speedup=1.67,
        )
        child.profiling = _make_profile()
        child.per_workload_latency_us = {"wl0": 60.0}
        search_result = SearchResult(
            best_node=child,
            total_iterations=1,
            termination_reason=TerminationReason.BUDGET,
            tree=tree,
        )
        workloads = [Workload(uuid="wl0", axes={"M": 128, "N": 128, "K": 64})]
        problem = Problem(
            name="sol-matmul",
            axes={
                "M": AxisDef(type="var"),
                "N": AxisDef(type="var"),
                "K": AxisDef(type="var"),
            },
            inputs={
                "a": TensorDef(shape=["M", "K"], dtype="float32"),
                "b": TensorDef(shape=["K", "N"], dtype="float32"),
            },
            outputs={"c": TensorDef(shape=["M", "N"], dtype="float32")},
            reference_source="def run(a, b): return a @ b\n",
            op_type="matmul",
            workloads=workloads,
            definition_path=definition_path,
        )
        gens = [lambda s: ()]

        profile_fake = MagicMock(return_value=_make_profile())
        with patch("src.eval.profiler.profile_kernel", profile_fake):
            generate_report(
                search_result,
                workloads=workloads,
                input_generators=gens,
                hardware_spec=harness.config.hardware,
                problem=problem,
            )

        assert profile_fake.call_count == 1
        kwargs = profile_fake.call_args.kwargs
        assert kwargs["problem_definition_path"] == definition_path

    def test_reprofile_passes_none_definition_path_when_no_problem(self, harness):
        """Placeholder / legacy path: no ``problem`` kwarg. The re-profile
        must still call profile_kernel but with
        ``problem_definition_path=None`` — the driver then falls back to
        ``make_inputs`` / ``args`` (the pre-existing behavior)."""
        from src.benchmark.problem import Workload
        from src.eval.scorer import ScoreResult
        from src.pipeline.report import generate_report
        from src.search.orchestrator import SearchResult, TerminationReason
        from src.search.tree import SearchTree

        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = ScoreResult(
            sol_score=0.3, baseline_latency_us=100.0, candidate_latency_us=100.0,
            t_sol_us=50.0, speedup=1.0,
        )
        child = tree.add_child(root.id, _make_kernel("child"), "tiling")
        child.score = ScoreResult(
            sol_score=0.8, baseline_latency_us=100.0, candidate_latency_us=60.0,
            t_sol_us=50.0, speedup=1.67,
        )
        child.profiling = _make_profile()
        child.per_workload_latency_us = {"wl0": 60.0}
        search_result = SearchResult(
            best_node=child,
            total_iterations=1,
            termination_reason=TerminationReason.BUDGET,
            tree=tree,
        )
        workloads = [Workload(uuid="wl0", axes={})]
        gens = [lambda s: ()]

        profile_fake = MagicMock(return_value=_make_profile())
        with patch("src.eval.profiler.profile_kernel", profile_fake):
            generate_report(
                search_result,
                workloads=workloads,
                input_generators=gens,
                hardware_spec=harness.config.hardware,
            )

        assert profile_fake.call_count == 1
        assert profile_fake.call_args.kwargs["problem_definition_path"] is None


class TestRooflineInputsDerivedPerIteration:
    """SOL-shaped problems arrive at the orchestrator with
    ``KernelSpec.flop_count == 0`` and ``KernelSpec.memory_bytes == 0`` —
    ``problem_to_kernel_spec`` leaves them at zero because SOLAR supplies
    ``T_SOL`` directly. The orchestrator must rederive per-iteration
    ``(flops, nbytes)`` from the Problem + representative Workload rather
    than feed the zeroed spec into ``profile_kernel`` (which would raise
    ``ProfilerError`` and DEAD_END every branch)."""

    @pytest.mark.asyncio
    async def test_sol_problem_with_zero_spec_counts_still_profiles(self, harness):
        """A SOL-shaped Kernel has ``flop_count=0`` and ``memory_bytes=0``.
        When we pass a Problem + Workload, the orchestrator must thread
        helper-computed counts into profile_kernel, not the spec zeros."""
        from src.benchmark.problem import AxisDef, Problem, TensorDef, Workload
        from src.search.orchestrator import Orchestrator

        # Rebuild the baseline with zeroed counts (mirrors what
        # problem_to_kernel_spec produces for a SOL problem).
        harness.baseline = Kernel(
            spec=KernelSpec(
                name="sol-matmul",
                kernel_type=KernelType.MATMUL,
                flop_count=0,
                memory_bytes=0,
            ),
            source_code="# placeholder",
        )
        workloads = [Workload(uuid="wl0", axes={"M": 128, "N": 128, "K": 64})]
        problem = Problem(
            name="sol-matmul",
            axes={
                "M": AxisDef(type="var"),
                "N": AxisDef(type="var"),
                "K": AxisDef(type="var"),
            },
            inputs={
                "a": TensorDef(shape=["M", "K"], dtype="float32"),
                "b": TensorDef(shape=["K", "N"], dtype="float32"),
            },
            outputs={"c": TensorDef(shape=["M", "N"], dtype="float32")},
            reference_source="def run(a, b): return a @ b\n",
            op_type="matmul",
            workloads=workloads,
        )
        input_generators = [lambda seed: ()]

        child_bench = BenchmarkResult(
            median_latency_us=60.0,
            timed_runs=1,
            per_workload_latency_us={"wl0": 60.0},
        )
        baseline_bench = BenchmarkResult(
            median_latency_us=100.0,
            timed_runs=1,
            per_workload_latency_us={"wl0": 100.0},
        )
        profile_fake = MagicMock(return_value=_make_profile())

        with (
            patch(
                "src.eval.benchmark.benchmark_kernel",
                side_effect=[baseline_bench, child_bench],
            ),
            patch("src.eval.profiler.profile_kernel", profile_fake),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            await orch.run(
                harness.baseline,
                workloads=workloads,
                roofline=harness.roofline,
                input_generators=input_generators,
                problem=problem,
            )

        # profile_kernel must have been called with nonzero flops / nbytes
        # derived from the helper — NOT the spec's zeros.
        assert profile_fake.call_count == 1
        kwargs = profile_fake.call_args.kwargs
        expected_flops = 2 * 128 * 128 * 64  # 2·M·N·K
        expected_nbytes = (128 * 64 + 64 * 128 + 128 * 128) * 4  # fp32 I/O
        assert kwargs["flops"] == expected_flops
        assert kwargs["nbytes"] == expected_nbytes

    @pytest.mark.asyncio
    async def test_unknown_op_type_skips_profile_without_dead_end(self, harness):
        """When the helper returns (0, 0) — e.g. an op_type we haven't modelled
        — the orchestrator must skip the profile_kernel call, leave
        ``child.profiling = None``, and keep the branch alive (PROMISING).
        A zero-spec op must not DEAD_END the whole search."""
        from src.agents.reviewer import BranchQuality
        from src.benchmark.problem import AxisDef, Problem, TensorDef, Workload
        from src.search.orchestrator import Orchestrator

        harness.baseline = Kernel(
            spec=KernelSpec(
                name="sol-unknown",
                kernel_type=KernelType.CUSTOM,
                flop_count=0,
                memory_bytes=0,
            ),
            source_code="# placeholder",
        )
        workloads = [Workload(uuid="wl0", axes={"N": 256})]
        problem = Problem(
            name="sol-unknown",
            axes={"N": AxisDef(type="var")},
            inputs={"x": TensorDef(shape=["N"], dtype="float32")},
            outputs={"y": TensorDef(shape=["N"], dtype="float32")},
            reference_source="def run(x): return x\n",
            op_type="some_op_without_a_formula",
            workloads=workloads,
        )
        input_generators = [lambda seed: ()]

        baseline_bench = BenchmarkResult(
            median_latency_us=100.0,
            timed_runs=1,
            per_workload_latency_us={"wl0": 100.0},
        )
        child_bench = BenchmarkResult(
            median_latency_us=70.0,
            timed_runs=1,
            per_workload_latency_us={"wl0": 70.0},
        )
        profile_fake = MagicMock(return_value=_make_profile())

        with (
            patch(
                "src.eval.benchmark.benchmark_kernel",
                side_effect=[baseline_bench, child_bench],
            ),
            patch("src.eval.profiler.profile_kernel", profile_fake),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            result = await orch.run(
                harness.baseline,
                workloads=workloads,
                roofline=harness.roofline,
                input_generators=input_generators,
                problem=problem,
            )

        assert profile_fake.call_count == 0, (
            "profile_kernel must be skipped when the helper returns (0, 0)"
        )
        children = [n for n in result.tree._nodes.values() if n.parent_id is not None]
        assert len(children) == 1
        assert children[0].profiling is None
        # Branch must stay alive — the profile is an enrichment, not a gate.
        assert children[0].branch_quality is BranchQuality.PROMISING


class TestFailFastOnZeroHardware:
    @pytest.mark.asyncio
    async def test_run_raises_value_error_when_hardware_peaks_zero(self, harness):
        """A zeroed HardwareSpec (the ``detect_hardware()`` placeholder) makes
        every analytical profile raise ProfilerError and silently DEAD_END every
        branch. Orchestrator.run must fail-fast at the top with a ValueError
        mentioning peaks — this is a global config error, not a branch event."""
        from src.search.orchestrator import Orchestrator

        harness.config = ACTSConfig(
            hardware=HardwareSpec(),  # zeroed
            max_depth=1,
            beam_width=3,
            sol_plateau_window=99,
        )
        orch = Orchestrator(
            harness.config, harness.planner, harness.coder,
            harness.reviewer, harness.retriever,
        )
        with pytest.raises(ValueError, match="peak"):
            await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)
