"""Report generation — Phase C."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.search.orchestrator import SearchResult


@dataclass
class OptimizationReport:
    """Final report of an ACTS optimization run."""

    baseline_latency_us: float = 0.0
    best_latency_us: float = 0.0
    sol_score: float = 0.0
    speedup: float = 0.0
    technique_trace: list[str] = field(default_factory=list)
    # pending profiler — per-iteration bottleneck classification needs
    # eval/profiler.py (GPU-blocked). Until then the list stays empty.
    bottleneck_transitions: list[str] = field(default_factory=list)
    remaining_headroom_pct: float = 0.0
    total_iterations: int = 0
    termination_reason: str = ""
    reward_hack_suspect: bool = False
    calibration_warning: bool = False


def generate_report(result: SearchResult) -> OptimizationReport:
    """Generate an optimization report from a completed search result.

    Includes baseline/best latency, SOL score, speedup, the root-to-best
    technique trace, and remaining headroom to the hardware limit.
    ``bottleneck_transitions`` is left empty until the profiler lands.
    """
    best = result.best_node
    path = result.tree.path_to_node(best.id)
    trace = [n.action_applied for n in path if n.action_applied]
    termination = result.termination_reason.value

    score = best.score
    if score is None:
        return OptimizationReport(
            technique_trace=trace,
            total_iterations=result.total_iterations,
            termination_reason=termination,
        )
    return OptimizationReport(
        baseline_latency_us=score.baseline_latency_us,
        best_latency_us=score.candidate_latency_us,
        sol_score=score.sol_score,
        speedup=score.speedup,
        technique_trace=trace,
        remaining_headroom_pct=(1.0 - score.sol_score) * 100,
        total_iterations=result.total_iterations,
        termination_reason=termination,
        reward_hack_suspect=score.reward_hack_suspect,
        calibration_warning=score.calibration_warning,
    )


def render_report(report: OptimizationReport) -> str:
    """Render a multi-line text summary of the optimization report.

    Skips the scoring block when ``baseline_latency_us == 0`` so a
    degenerate run (no scored best node) doesn't print misleading
    "0.00us / 0.00x" lines.
    """
    lines = [
        f"Search completed: {report.termination_reason}",
        f"  Iterations: {report.total_iterations}",
    ]
    if report.baseline_latency_us > 0:
        lines.extend([
            f"  Baseline:  {report.baseline_latency_us:.2f} us",
            f"  Best:      {report.best_latency_us:.2f} us",
            f"  SOL score: {report.sol_score:.4f}  (headroom {report.remaining_headroom_pct:.1f}%)",
            f"  Speedup:   {report.speedup:.2f}x",
        ])
    if report.technique_trace:
        lines.append(f"  Trace: {' → '.join(report.technique_trace)}")
    if report.reward_hack_suspect:
        lines.append("  [AUDIT] reward_hack_suspect — candidate beats T_SOL (physics violation)")
    if report.calibration_warning:
        lines.append("  [AUDIT] calibration_warning — baseline already at/below T_SOL")
    return "\n".join(lines)
