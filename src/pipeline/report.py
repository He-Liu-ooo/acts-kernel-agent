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
    bottleneck_transitions: list[str] = field(default_factory=list)
    remaining_headroom_pct: float = 0.0
    total_iterations: int = 0
    termination_reason: str = ""


def generate_report(result: SearchResult) -> OptimizationReport:
    """Generate a human-readable optimization report from search results.

    Includes: baseline vs best, SOL score progression, bottleneck transitions,
    technique trace, and remaining headroom to hardware limit.
    """
    best = result.best_node
    score = best.score
    if score is None:
        return OptimizationReport(
            total_iterations=result.total_iterations,
            termination_reason=result.termination_reason,
        )
    return OptimizationReport(
        baseline_latency_us=score.baseline_latency_us,
        best_latency_us=score.candidate_latency_us,
        sol_score=score.sol_score,
        speedup=score.speedup,
        remaining_headroom_pct=(1.0 - score.sol_score) * 100,
        total_iterations=result.total_iterations,
        termination_reason=result.termination_reason,
    )
