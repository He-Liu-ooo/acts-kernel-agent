"""Tests for pipeline/report.py — Phase C report generation."""

from __future__ import annotations

import pytest

from src.eval.scorer import ScoreResult
from src.kernels.kernel import Kernel, KernelSpec, KernelType
from src.pipeline.report import OptimizationReport, generate_report, render_report
from src.search.orchestrator import SearchResult, TerminationReason
from src.search.tree import SearchTree


def _make_kernel(name: str = "test") -> Kernel:
    return Kernel(
        spec=KernelSpec(name=name, kernel_type=KernelType.MATMUL),
        source_code="# placeholder",
    )


def _make_score(
    sol: float,
    baseline: float = 100.0,
    t_sol: float = 50.0,
    *,
    reward_hack_suspect: bool = False,
    calibration_warning: bool = False,
) -> ScoreResult:
    # Invertible: candidate = baseline - sol * (baseline - t_sol).
    candidate = baseline - sol * (baseline - t_sol)
    return ScoreResult(
        sol_score=sol,
        baseline_latency_us=baseline,
        candidate_latency_us=candidate,
        t_sol_us=t_sol,
        speedup=baseline / candidate,
        reward_hack_suspect=reward_hack_suspect,
        calibration_warning=calibration_warning,
    )


def _build_result(
    *,
    best_id: int = 2,
    termination: TerminationReason = TerminationReason.BUDGET,
    iterations: int = 3,
) -> SearchResult:
    """Three-node chain: root(baseline, 0.3) → child(tiling, 0.6) → grand(vectorize, 0.8)."""
    tree = SearchTree()
    root = tree.add_root(_make_kernel("root"))
    root.score = _make_score(0.3)

    child = tree.add_child(root.id, _make_kernel("child"), "tiling")
    child.score = _make_score(0.6)

    grand = tree.add_child(child.id, _make_kernel("grand"), "vectorize")
    grand.score = _make_score(0.8)

    return SearchResult(
        best_node=tree.get_node(best_id),
        total_iterations=iterations,
        termination_reason=termination,
        tree=tree,
    )


class TestGenerateReport:
    def test_populates_scoring_fields_from_best_node_score(self):
        report = generate_report(_build_result(best_id=2))

        assert report.sol_score == pytest.approx(0.8)
        assert report.baseline_latency_us == pytest.approx(100.0)
        assert report.best_latency_us == pytest.approx(60.0)
        assert report.speedup == pytest.approx(100.0 / 60.0)
        assert report.remaining_headroom_pct == pytest.approx(20.0)

    def test_builds_technique_trace_from_root_to_best(self):
        report = generate_report(_build_result(best_id=2))
        # Root's action_applied is "" (baseline); trace is the applied actions
        # along root→best in order.
        assert report.technique_trace == ["tiling", "vectorize"]

    def test_trace_stops_at_best_node(self):
        """Sibling branches must not leak into the trace."""
        report = generate_report(_build_result(best_id=1))
        assert report.technique_trace == ["tiling"]

    def test_trace_excludes_root_baseline_placeholder(self):
        report = generate_report(_build_result(best_id=2))
        assert "" not in report.technique_trace

    def test_trace_empty_when_best_is_root(self):
        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = _make_score(0.3)
        result = SearchResult(
            best_node=root,
            total_iterations=0,
            termination_reason=TerminationReason.BUDGET,
            tree=tree,
        )
        assert generate_report(result).technique_trace == []

    def test_bottleneck_defaults_empty_when_no_run_bottleneck(self):
        """Placeholder path: SearchResult carries no run_bottleneck, no
        workloads/problem passed — both bottleneck surfaces stay empty.
        classify_run only fires inside the orchestrator; generate_report
        is a pure renderer."""
        report = generate_report(_build_result())
        assert report.bottleneck is None
        assert report.winner_per_workload_bottlenecks == {}

    def test_total_iterations_and_termination_reason_passthrough(self):
        report = generate_report(
            _build_result(iterations=7, termination=TerminationReason.PLATEAU)
        )
        assert report.total_iterations == 7
        assert report.termination_reason == "plateau"

    def test_termination_reason_is_plain_string(self):
        """Field type is `str`, so the enum must be unwrapped to its value."""
        report = generate_report(
            _build_result(termination=TerminationReason.SOL_TARGET)
        )
        assert isinstance(report.termination_reason, str)
        assert report.termination_reason == "sol_target"

    def test_handles_best_node_without_score(self):
        """Defensive path: if scoring failed, surface termination + iterations
        without crashing."""
        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        # root.score deliberately left None
        result = SearchResult(
            best_node=root,
            total_iterations=0,
            termination_reason=TerminationReason.ALL_DEAD_END,
            tree=tree,
        )
        report = generate_report(result)
        assert report.sol_score == 0.0
        assert report.baseline_latency_us == 0.0
        assert report.termination_reason == "all_dead_end"

    def test_propagates_reward_hack_suspect_flag(self):
        """SOL-ExecBench audit signal: candidate beats T_SOL. The report is
        the operator's first stop; dropping this flag hides a physics-violating
        result."""
        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = _make_score(0.3)
        child = tree.add_child(root.id, _make_kernel("child"), "tiling")
        child.score = _make_score(0.8, reward_hack_suspect=True)
        result = SearchResult(
            best_node=child,
            total_iterations=1,
            termination_reason=TerminationReason.BUDGET,
            tree=tree,
        )
        assert generate_report(result).reward_hack_suspect is True

    def test_propagates_calibration_warning_flag(self):
        """SOL-ExecBench audit signal: baseline already at/below T_SOL —
        speedups are not meaningful."""
        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = _make_score(0.3)
        child = tree.add_child(root.id, _make_kernel("child"), "tiling")
        child.score = _make_score(0.8, calibration_warning=True)
        result = SearchResult(
            best_node=child,
            total_iterations=1,
            termination_reason=TerminationReason.BUDGET,
            tree=tree,
        )
        assert generate_report(result).calibration_warning is True

    def test_audit_flags_default_false_when_no_score(self):
        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        result = SearchResult(
            best_node=root,
            total_iterations=0,
            termination_reason=TerminationReason.ALL_DEAD_END,
            tree=tree,
        )
        report = generate_report(result)
        assert report.reward_hack_suspect is False
        assert report.calibration_warning is False


class TestRenderReport:
    def test_includes_termination_reason_and_iterations(self):
        text = render_report(OptimizationReport(
            termination_reason="budget", total_iterations=20,
        ))
        assert "budget" in text
        assert "20" in text

    def test_includes_scoring_fields_when_present(self):
        text = render_report(OptimizationReport(
            baseline_latency_us=100.0,
            best_latency_us=60.0,
            sol_score=0.8,
            speedup=1.6667,
            remaining_headroom_pct=20.0,
            total_iterations=3,
            termination_reason="sol_target",
        ))
        assert "0.8" in text
        assert "1.67" in text
        assert "20" in text
        assert "100" in text
        assert "60" in text

    def test_includes_technique_trace(self):
        text = render_report(OptimizationReport(
            baseline_latency_us=100.0,
            best_latency_us=60.0,
            sol_score=0.8,
            speedup=1.67,
            technique_trace=["tiling", "vectorize"],
            total_iterations=2,
            termination_reason="budget",
        ))
        assert "tiling" in text
        assert "vectorize" in text

    def test_omits_scoring_block_when_no_data(self):
        """Degenerate report: skip the SOL/speedup lines rather than printing
        '0.00x speedup' which would mislead the reader."""
        text = render_report(OptimizationReport(
            termination_reason="all_dead_end", total_iterations=0,
        ))
        assert "all_dead_end" in text
        assert "speedup" not in text.lower()
        assert "SOL score" not in text

    def test_surfaces_reward_hack_suspect_prominently(self):
        """If the audit flag is set, the rendered summary must say so in a
        way an operator scanning the output cannot miss."""
        text = render_report(OptimizationReport(
            baseline_latency_us=100.0,
            best_latency_us=30.0,
            sol_score=1.2,
            speedup=3.33,
            remaining_headroom_pct=-20.0,
            total_iterations=3,
            termination_reason="sol_target",
            reward_hack_suspect=True,
        ))
        assert "reward_hack_suspect" in text.lower() or "reward-hack" in text.lower()

    def test_surfaces_calibration_warning_prominently(self):
        text = render_report(OptimizationReport(
            baseline_latency_us=40.0,
            best_latency_us=40.0,
            sol_score=0.0,
            speedup=1.0,
            remaining_headroom_pct=100.0,
            total_iterations=3,
            termination_reason="budget",
            calibration_warning=True,
        ))
        assert "calibration" in text.lower()

    def test_omits_audit_line_when_flags_clean(self):
        text = render_report(OptimizationReport(
            baseline_latency_us=100.0,
            best_latency_us=60.0,
            sol_score=0.8,
            speedup=1.67,
            total_iterations=3,
            termination_reason="budget",
        ))
        assert "reward" not in text.lower()
        assert "calibration" not in text.lower()


class TestRenderProfilingBlock:
    """Phase C ``Winner profile (per workload)`` rendering.

    Regression tests for the rendered units in the per-workload profile
    block. Two surfaces are easy to mis-render:

    * ``pct_peak_compute`` / ``pct_peak_bandwidth`` are stored as
      fractions in ``[0, 1]`` and the formatter multiplies by 100. The
      formatter must not double-multiply, and must not silently emit
      values way outside ``[0, ~150]`` (we allow some headroom for
      L2-cache-amplified bandwidth on tiny working sets).
    * NCU's stall metric ``smsp__average_warp_latency_issue_stalled_*.pct``
      is **not** a bounded percentage despite the name. Values in the
      thousands are normal. The formatter must not append ``%`` to it
      (which would lie to the operator) — the unit suffix lives in the
      format string explicitly.
    """

    def _make_profiling(
        self,
        *,
        achieved_tflops: float = 50.0,
        achieved_bandwidth_gb_s: float = 800.0,
        pct_peak_compute: float = 0.6,
        pct_peak_bandwidth: float = 0.85,
        sm_occupancy_pct: float = 70.0,
        l2_hit_rate_pct: float = 50.0,
        tensor_core_util_pct: float = 0.0,
        dominant_pct: float = 5343.86,
        runner_up_pct: float = 2929.48,
    ):
        from src.eval.profiler import (
            AnalyticalMetrics,
            NCUMetrics,
            ProfilingResult,
        )
        return ProfilingResult(
            analytical=AnalyticalMetrics(
                achieved_tflops=achieved_tflops,
                achieved_bandwidth_gb_s=achieved_bandwidth_gb_s,
                pct_peak_compute=pct_peak_compute,
                pct_peak_bandwidth=pct_peak_bandwidth,
            ),
            ncu=NCUMetrics(
                sm_occupancy_pct=sm_occupancy_pct,
                l2_hit_rate_pct=l2_hit_rate_pct,
                tensor_core_util_pct=tensor_core_util_pct,
                warp_stall_dominant="long_scoreboard",
                warp_stall_dominant_pct=dominant_pct,
                warp_stall_runner_up="short_scoreboard",
                warp_stall_runner_up_pct=runner_up_pct,
            ),
        )

    def test_stalls_render_without_percent_sign(self):
        """The NCU stall metric is unbounded (not a [0, 100] percentage).
        Appending ``%`` would mislead the operator into reading
        ``5343.86 cyc/inst×100`` as ``5343%`` of total stalls."""
        text = render_report(OptimizationReport(
            termination_reason="budget",
            total_iterations=1,
            winner_profiling_per_workload={
                "wl-uuid": self._make_profiling(
                    dominant_pct=534386.0,
                    runner_up_pct=292948.1,
                ),
            },
        ))
        # The rendered stall pair must NOT carry a trailing ``%``
        # character — it would imply the value is bounded.
        assert "534386.0%" not in text
        assert "292948.1%" not in text
        # The unit suffix must make the metric's nature explicit.
        assert "cyc/inst×100" in text
        # Stall reasons themselves must still be rendered for the
        # operator's bottleneck-attribution use case.
        assert "long_scoreboard" in text
        assert "short_scoreboard" in text

    def _make_degraded_for_missing_latency(self):
        """Sentinel ProfilingResult mirroring what generate_report
        produces for a workload whose per-workload latency is missing.
        Constructed via the renderer's own helper so the test cannot
        drift from the production marker / shape."""
        from src.pipeline.report import _degraded_for_missing_latency
        return _degraded_for_missing_latency()

    def _make_roofline(
        self,
        *,
        ai: float = 1.5,
        ridge: float = 32.0,
        bottleneck=None,
    ):
        from src.eval.roofline import RooflineResult
        from src.eval.types import BottleneckType
        return RooflineResult(
            t_sol_us=1.0,
            bottleneck=bottleneck or BottleneckType.MEMORY_BOUND,
            source="solar",
            arithmetic_intensity=ai,
            ridge_point=ridge,
        )

    def test_render_profiling_block_degrades_when_per_workload_latency_missing(self):
        """When generate_report can't pair a workload with its measured
        latency, it inserts the latency-missing sentinel. The renderer
        must surface a DEGRADED marker, keep the roofline summary
        visible, and emit no fabricated TFLOPS / GB/s / pct_peak line.
        """
        text = render_report(OptimizationReport(
            termination_reason="budget",
            total_iterations=1,
            winner_profiling_per_workload={
                "wl-uuid-missing": self._make_degraded_for_missing_latency(),
            },
            winner_roofline_per_workload={
                "wl-uuid-missing": self._make_roofline(ai=2.5, ridge=64.0),
            },
        ))
        assert "[DEGRADED:" in text
        assert "missing per-workload latency" in text
        # The fabricated-throughput line is the bug we are fixing — it
        # must not be present for a degraded workload.
        assert "TFLOPS" not in text
        assert "GB/s" not in text
        assert "pct_peak" not in text
        # Roofline data is independent of latency and must still render.
        assert "AI 2.50" in text
        assert "ridge 64.00" in text
        assert "memory_bound" in text

    def test_render_profiling_block_degrades_on_non_finite_latency(self):
        """Variant: non-finite per-workload latency goes through the
        same degraded path as missing. This test exercises the rendered
        output once the sentinel is in place — the upstream gating in
        ``generate_report`` is what produces the sentinel for `inf` /
        `nan`, and is exercised separately via the generate_report
        path; here we lock the renderer behavior."""
        text = render_report(OptimizationReport(
            termination_reason="budget",
            total_iterations=1,
            winner_profiling_per_workload={
                "wl-uuid-inf": self._make_degraded_for_missing_latency(),
            },
            winner_roofline_per_workload={
                "wl-uuid-inf": self._make_roofline(),
            },
        ))
        assert "[DEGRADED:" in text
        assert "TFLOPS" not in text

    def test_render_profiling_block_degrades_on_zero_latency(self):
        """Same renderer behavior for the zero-latency case — caller
        feeds the sentinel and the renderer suppresses analytical
        output."""
        text = render_report(OptimizationReport(
            termination_reason="budget",
            total_iterations=1,
            winner_profiling_per_workload={
                "wl-uuid-zero": self._make_degraded_for_missing_latency(),
            },
            winner_roofline_per_workload={
                "wl-uuid-zero": self._make_roofline(),
            },
        ))
        assert "[DEGRADED:" in text
        assert "TFLOPS" not in text

    def _make_profiling_with_dtype(
        self,
        *,
        compute_peak_dtype: str = "fp32",
        compute_peak_calibration_warning: bool = False,
    ):
        from src.eval.profiler import (
            AnalyticalMetrics,
            NCUMetrics,
            ProfilingResult,
        )
        return ProfilingResult(
            analytical=AnalyticalMetrics(
                achieved_tflops=36.42,
                achieved_bandwidth_gb_s=12.0,
                pct_peak_compute=0.10,
                pct_peak_bandwidth=0.012,
                compute_peak_dtype=compute_peak_dtype,
                compute_peak_calibration_warning=compute_peak_calibration_warning,
            ),
            ncu=NCUMetrics(
                sm_occupancy_pct=70.0, l2_hit_rate_pct=50.0,
                tensor_core_util_pct=0.0,
                warp_stall_dominant="x", warp_stall_dominant_pct=0.0,
                warp_stall_runner_up="y", warp_stall_runner_up_pct=0.0,
            ),
        )

    def test_render_dtype_label_next_to_pct_peak_compute(self):
        text = render_report(OptimizationReport(
            termination_reason="budget",
            total_iterations=1,
            winner_profiling_per_workload={
                "wl-uuid": self._make_profiling_with_dtype(compute_peak_dtype="bf16"),
            },
        ))
        assert "compute 10.0% [bf16]" in text

    def test_render_fp32_fallback_marker_visible(self):
        text = render_report(OptimizationReport(
            termination_reason="budget",
            total_iterations=1,
            winner_profiling_per_workload={
                "wl-uuid": self._make_profiling_with_dtype(
                    compute_peak_dtype="fp32_fallback",
                    compute_peak_calibration_warning=True,
                ),
            },
        ))
        assert "compute 10.0% [fp32_fallback]" in text

    def test_render_profiling_block_happy_path_unchanged(self):
        """Sanity: the analytical TFLOPS / GB/s / pct_peak line still
        renders for a workload with valid analytical metrics. Guards
        against the degraded branch accidentally swallowing all entries.
        """
        text = render_report(OptimizationReport(
            termination_reason="budget",
            total_iterations=1,
            winner_profiling_per_workload={
                "wl-uuid-ok": self._make_profiling(
                    achieved_tflops=12.5,
                    achieved_bandwidth_gb_s=400.0,
                    pct_peak_compute=0.5,
                    pct_peak_bandwidth=0.4,
                ),
            },
        ))
        assert "12.50 TFLOPS" in text
        assert "400.00 GB/s" in text
        assert "pct_peak" in text
        assert "[DEGRADED:" not in text

    def test_pct_peak_compute_and_bw_are_in_sane_range(self):
        """pct_peak fields are fractions in [0, 1] in the analytical
        contract; the formatter multiplies by 100 once. With sane
        analytical inputs the rendered values must stay in a plausible
        range — drifting outside [0, 150] indicates a double-multiply
        regression (e.g., the formatter being changed to read a value
        already in [0, 100])."""
        text = render_report(OptimizationReport(
            termination_reason="budget",
            total_iterations=1,
            winner_profiling_per_workload={
                "wl-uuid": self._make_profiling(
                    pct_peak_compute=0.637,
                    pct_peak_bandwidth=0.85,
                ),
            },
        ))
        # Spot-check the rendered values — 63.7% and 85.0% are the
        # expected single-multiply outputs.
        assert "63.7%" in text
        assert "85.0%" in text
        # Defensive: the rendered text must not contain a four-digit
        # pct_peak (which is the smoking gun for either a measurement
        # bug upstream OR a double-multiply in the formatter).
        import re
        for match in re.finditer(r"(\d+(?:\.\d+)?)%", text):
            value = float(match.group(1))
            # Allow occupancy / L2 / headroom up to 100; allow a small
            # cushion above for L2-amplified bandwidth on tiny working
            # sets (well-known measurement effect — DRAM traffic <
            # nbytes when L2 hits dominate).
            assert value <= 200.0, (
                f"Rendered percentage {match.group(0)!r} exceeds 200%; "
                "either the formatter double-multiplied a fraction or "
                "the upstream measurement is corrupt (latency / nbytes "
                "mismatch — see report.py:184 fallback to "
                "aggregate_latency_s)."
            )


# ── Hardware-spec block in report (2026-05-10) ─────────────────────────


class TestHardwareSpecInReport:
    def test_generate_report_populates_hardware_spec_when_provided(self):
        from src.config import HardwareSpec

        hw = HardwareSpec(
            name="NVIDIA RTX 6000 Ada Generation",
            freq_GHz=2.505,
            SRAM_capacity=100663296,
            DRAM_capacity=50876841984,
        )
        report = generate_report(_build_result(best_id=2), hardware_spec=hw)
        assert report.hardware_spec is hw

    def test_generate_report_hardware_spec_defaults_to_none(self):
        report = generate_report(_build_result(best_id=2))
        assert report.hardware_spec is None

    def test_render_includes_hardware_spec_block_when_populated(self):
        from src.config import HardwareSpec

        hw = HardwareSpec(
            name="NVIDIA RTX 6000 Ada Generation",
            freq_GHz=2.505,
            SRAM_capacity=100663296,
            DRAM_capacity=50876841984,
            SRAM_byte_per_cycle=2200,
            DRAM_byte_per_cycle=383,
            MAC_per_cycle_fp32_sm=18185,
            MAC_per_cycle_bf16_tc=72695,
        )
        report = generate_report(_build_result(best_id=2), hardware_spec=hw)
        text = render_report(report)
        assert "Hardware spec" in text
        assert "NVIDIA RTX 6000 Ada Generation" in text
        assert "2.505" in text
        assert "Peak FP32:" in text
        assert "Peak DRAM BW:" in text

    def test_render_omits_hardware_spec_block_when_none(self):
        report = generate_report(_build_result(best_id=2))
        text = render_report(report)
        assert "Hardware spec" not in text

    def test_render_includes_block_with_zeros_for_degraded_spec(self):
        """User chose 'always print full block': a default HardwareSpec()
        (all zeros, empty name) still emits the block."""
        from src.config import HardwareSpec

        hw = HardwareSpec()  # all defaults — degraded detection
        report = generate_report(_build_result(best_id=2), hardware_spec=hw)
        text = render_report(report)
        assert "Hardware spec" in text
        assert "Peak FP32:          0.00 TFLOPS" in text

    def test_render_separates_fp16_and_bf16_peaks(self):
        """Codex review [P2]: BF16 and FP16 throughput must render as
        distinct lines. Hardware where MAC_per_cycle_fp16_tc !=
        MAC_per_cycle_bf16_tc would otherwise see a misleading combined
        ``BF16/FP16`` label."""
        from src.config import HardwareSpec

        hw = HardwareSpec(
            name="hypothetical-divergent",
            freq_GHz=2.0,
            MAC_per_cycle_fp16_tc=20000,  # 80 TFLOPS @ 2 GHz
            MAC_per_cycle_bf16_tc=10000,  # 40 TFLOPS @ 2 GHz — half
        )
        report = generate_report(_build_result(best_id=2), hardware_spec=hw)
        text = render_report(report)

        # Distinct lines, distinct values:
        assert "Peak FP16:" in text
        assert "Peak BF16:" in text
        # The merged "BF16/FP16" label must be gone — it was the bug.
        assert "BF16/FP16" not in text
        # Values reflect their own MAC/cycle, not a shared one:
        assert "Peak FP16:          80.00 TFLOPS" in text
        assert "Peak BF16:          40.00 TFLOPS" in text


from src.runtime.usage import UsageBucket, UsageSnapshot


def _snap_three_iters() -> UsageSnapshot:
    """Build a representative 3-iter snapshot for render assertions."""
    p1 = UsageBucket(invocations=1, turns=2, input_tokens=1200, output_tokens=42)
    c0 = UsageBucket(invocations=1, turns=3, input_tokens=1800, output_tokens=28)  # baseline-translate
    c1 = UsageBucket(invocations=1, turns=5, input_tokens=3400, output_tokens=512)
    r1 = UsageBucket(invocations=1, turns=4, input_tokens=2100, output_tokens=98)
    by_iter_agent = {
        (0, "coder-translate"): c0,
        (1, "planner"): p1,
        (1, "coder"): c1,
        (1, "reviewer"): r1,
    }
    by_iter = {
        0: c0,
        1: p1 + c1 + r1,
    }
    by_agent = {"planner": p1, "coder": c1, "coder-translate": c0, "reviewer": r1}
    total = c0 + p1 + c1 + r1
    return UsageSnapshot(
        by_iter_agent=by_iter_agent,
        by_iter=by_iter,
        by_agent=by_agent,
        total=total,
        columns=("planner", "coder", "coder-translate", "reviewer"),
    )


class TestUsageBlockRender:
    def test_populated_snapshot_renders_table(self):
        from src.pipeline.report import OptimizationReport, render_report
        rep = OptimizationReport(
            baseline_latency_us=10.0, best_latency_us=5.0, sol_score=0.5,
            speedup=2.0, total_iterations=1, termination_reason="sol_target_reached",
            usage_stats=_snap_three_iters(),
        )
        text = render_report(rep)
        assert "Resource usage (LLM)" in text
        # Header row mentions every column.
        assert "planner" in text and "coder" in text
        assert "coder-translate" in text and "reviewer" in text
        # Iter rows present.
        assert "0 |" in text
        assert "1 |" in text
        # Em-dash for empty cells (iter 0 has no planner / reviewer).
        assert "—" in text
        # Cell format includes the arrow.
        assert "→" in text
        # k abbreviation kicks in for 1200, 1800, 2100, 3400.
        assert "1.2k" in text
        # Run-total row present.
        assert "total" in text.lower()

    def test_empty_snapshot_renders_fallback_line(self):
        from src.pipeline.report import OptimizationReport, render_report
        empty = UsageSnapshot(
            by_iter_agent={}, by_iter={}, by_agent={},
            total=UsageBucket(), columns=(),
        )
        rep = OptimizationReport(
            baseline_latency_us=10.0, best_latency_us=5.0, sol_score=0.5,
            speedup=2.0, total_iterations=1, termination_reason="budget_exhausted",
            usage_stats=empty,
        )
        text = render_report(rep)
        assert "Resource usage (LLM)" in text
        assert "(no LLM usage captured)" in text
        # No table → no em-dash, no arrow.
        assert "→" not in text

    def test_none_usage_stats_renders_fallback_line(self):
        from src.pipeline.report import OptimizationReport, render_report
        rep = OptimizationReport(
            baseline_latency_us=10.0, best_latency_us=5.0, sol_score=0.5,
            speedup=2.0, total_iterations=1, termination_reason="budget_exhausted",
            usage_stats=None,
        )
        text = render_report(rep)
        # None and empty-snapshot render identically.
        assert "Resource usage (LLM)" in text
        assert "(no LLM usage captured)" in text

    def test_cached_and_reasoning_footer_emitted_when_nonzero(self):
        from src.pipeline.report import OptimizationReport, render_report
        bucket = UsageBucket(
            invocations=1, turns=1,
            input_tokens=1000, output_tokens=500,
            cached_input_tokens=100, reasoning_output_tokens=300,
        )
        snap = UsageSnapshot(
            by_iter_agent={(1, "coder"): bucket},
            by_iter={1: bucket},
            by_agent={"coder": bucket},
            total=bucket,
            columns=("coder",),
        )
        rep = OptimizationReport(
            baseline_latency_us=10.0, best_latency_us=5.0, sol_score=0.5,
            speedup=2.0, total_iterations=1, termination_reason="sol_target_reached",
            usage_stats=snap,
        )
        text = render_report(rep)
        assert "of which cached input" in text
        assert "10.0%" in text  # 100 / 1000
        assert "of which reasoning output" in text
        assert "60.0%" in text  # 300 / 500

    def test_zero_sub_buckets_suppress_footer_lines(self):
        from src.pipeline.report import OptimizationReport, render_report
        bucket = UsageBucket(
            invocations=1, turns=1,
            input_tokens=1000, output_tokens=500,
            # cached and reasoning stay zero
        )
        snap = UsageSnapshot(
            by_iter_agent={(1, "coder"): bucket},
            by_iter={1: bucket},
            by_agent={"coder": bucket},
            total=bucket,
            columns=("coder",),
        )
        rep = OptimizationReport(
            baseline_latency_us=10.0, best_latency_us=5.0, sol_score=0.5,
            speedup=2.0, total_iterations=1, termination_reason="sol_target_reached",
            usage_stats=snap,
        )
        text = render_report(rep)
        assert "of which cached input" not in text
        assert "of which reasoning output" not in text


# ── swallowed-exception logging (Task 5) ────────────────────────────────


def test_report_dtype_gather_logs_on_generator_failure(caplog):
    """When the per-workload input generator raises (e.g. a poisoned CUDA
    context), the dtype-gather ``except`` must emit a WARNING before
    falling back to () — turning silent fp32_fallback into a diagnosable
    log line."""
    import logging
    from types import SimpleNamespace
    from unittest.mock import patch

    from src.config import HardwareSpec

    tree = SearchTree()
    root = tree.add_root(_make_kernel("root"))
    root.score = _make_score(0.3)
    best = tree.add_child(root.id, _make_kernel("winner"), "tiling")
    best.score = _make_score(0.8)
    best.per_workload_latency_us = {"w0": 42.0}

    result = SearchResult(
        best_node=best,
        total_iterations=1,
        termination_reason=TerminationReason.BUDGET,
        tree=tree,
    )

    workload = SimpleNamespace(uuid="w0", model_dump=lambda mode="json": {"uuid": "w0"})

    def _boom(_seed):
        raise RuntimeError("CUDA error: device-side assert triggered")

    caplog.set_level(logging.WARNING, logger="src.pipeline.report")
    with (
        patch("src.eval.profiler.profile_kernel", return_value=object()),
        patch("src.eval.profiler._collect_input_dtypes", return_value=[]),
    ):
        generate_report(
            result,
            workloads=[workload],
            input_generators=[_boom],
            hardware_spec=HardwareSpec(),
            definition=None,
        )

    assert any(
        "device-side assert" in r.message or "input generator raised" in r.message
        for r in caplog.records
    )
