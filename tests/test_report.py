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
