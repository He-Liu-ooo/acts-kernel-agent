"""SOL Score computation.

Called by the orchestrator after benchmarking and profiling.
Not part of the Coder's tool loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import sol_execbench.sol_score


@dataclass
class ScoreResult:
    """SOL score for a candidate kernel."""

    sol_score: float  # Range [0, 1] under normal conditions
    baseline_latency_us: float
    candidate_latency_us: float
    t_sol_us: float
    speedup: float  # baseline / candidate
    # Audit flags (SOL-ExecBench §4.3): surface SOLAR-bound / reward-hack
    # violations for review rather than silently passing them through.
    reward_hack_suspect: bool = False  # T_k < T_SOL — candidate beats physics
    calibration_warning: bool = False  # T_b <= T_SOL — baseline already at limit


def compute_sol_score(
    baseline_latency_us: float,
    candidate_latency_us: float,
    t_sol_us: float,
) -> ScoreResult:
    """Compute SOL score by delegating the formula to sol_execbench.sol_score.

    The wrapper layers ACTS's audit flags (reward_hack_suspect / calibration_warning)
    on top of SOL's pure-formula primitive.
    """
    raw = sol_execbench.sol_score.sol_score(
        t_k=candidate_latency_us / 1000,  # µs → ms
        t_p=baseline_latency_us / 1000,
        t_sol=t_sol_us / 1000,
    )
    return ScoreResult(
        sol_score=raw,
        baseline_latency_us=baseline_latency_us,
        candidate_latency_us=candidate_latency_us,
        t_sol_us=t_sol_us,
        speedup=baseline_latency_us / candidate_latency_us if candidate_latency_us > 0 else 0.0,
        reward_hack_suspect=candidate_latency_us < t_sol_us,
        calibration_warning=baseline_latency_us <= t_sol_us,
    )
