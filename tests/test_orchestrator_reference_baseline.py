"""External reference baseline (Option C) — orchestrator T_b threading.

Tier 1 (no GPU): mirrors the mocked-orchestrator harness in
``tests/test_orchestrator_profiling.py`` (patched ``benchmark_kernel`` +
``profile_kernel``, mocked agents). Covers the two paths that matter:

  1. DEFAULT — no ``reference_baseline_latency_us``: the root's score uses
     the Triton baseline's own median as T_b (today's behavior — regression
     guard).
  2. OVERRIDE — ``run(reference_baseline_latency_us=REF)``: the root's
     score uses REF as T_b, and ``SearchResult.reference_baseline_latency_us``
     is REF.
"""

from __future__ import annotations

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

BASELINE_MEDIAN_US = 25.0  # Triton root's own measured median
REF = 10.0  # external reference's measured median (distinct from baseline)


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
    config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=1,
        beam_width=3,
        sol_plateau_window=99,
        coder_n_candidates=1,
    )
    planner = MagicMock()
    planner.plan = AsyncMock(return_value=OptimizationPlan(
        tier=1, technique="tiling", params={}, target_region="", rationale="",
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

    bench = BenchmarkResult(median_latency_us=BASELINE_MEDIAN_US, timed_runs=1)
    baseline = _make_kernel("root")
    # T_SOL well below both medians so calibration_warning stays off and
    # the score formula is in its normal regime.
    roofline = RooflineResult(
        t_sol_us=5.0,
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


async def _run_orch(h, **run_kwargs):
    from src.search.orchestrator import Orchestrator

    profile_fake = MagicMock(return_value=_make_profile())
    with (
        patch("src.eval.benchmark.benchmark_kernel", return_value=h.bench),
        patch("src.eval.profiler.profile_kernel", profile_fake),
    ):
        orch = Orchestrator(h.config, h.planner, h.coder, h.reviewer, h.retriever)
        return await orch.run(
            h.baseline, workloads=None, roofline=h.roofline, **run_kwargs
        )


@pytest.mark.asyncio
async def test_default_tb_is_baseline_median(harness):
    """REGRESSION GUARD: with no override, the root's score T_b equals the
    injected Triton baseline median (today's behavior)."""
    result = await _run_orch(harness)
    root_node = result.tree.get_node(0)
    assert root_node.score.baseline_latency_us == BASELINE_MEDIAN_US
    assert result.reference_baseline_latency_us is None
    # The Triton root's own median is carried explicitly, regardless of T_b.
    assert result.baseline_root_latency_us == BASELINE_MEDIAN_US


@pytest.mark.asyncio
async def test_reference_latency_overrides_tb_for_root(harness):
    """OVERRIDE: run(reference_baseline_latency_us=REF) makes the root's
    score use REF as T_b, not the Triton root's own median."""
    result = await _run_orch(harness, reference_baseline_latency_us=REF)
    root_node = result.tree.get_node(0)
    assert root_node.score.baseline_latency_us == REF
    assert result.reference_baseline_latency_us == REF
    # T_b is REF (10), but the Triton root's own median (25) is preserved
    # distinctly so the "report both" design survives the override.
    assert result.baseline_root_latency_us == BASELINE_MEDIAN_US
