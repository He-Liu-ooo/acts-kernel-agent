"""SOL Score computation.

Called by the orchestrator after benchmarking and profiling.
Not part of the Coder's tool loop.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreResult:
    """SOL score for a candidate kernel."""

    sol_score: float  # Range [0, 1] under normal conditions
    baseline_latency_us: float
    candidate_latency_us: float
    t_sol_us: float
    speedup: float  # baseline / candidate
    # Audit flags (SOL-ExecBench paper, Section 4.3):
    # "If either assumption is violated in practice, we treat the case as an
    #  audit signal and report it for SOLAR bound review and reward-hacking
    #  inspection."
    reward_hack_suspect: bool = False  # T_k < T_SOL — candidate beats physics
    calibration_warning: bool = False  # T_b <= T_SOL — baseline already at limit


def compute_sol_score(
    baseline_latency_us: float,
    candidate_latency_us: float,
    t_sol_us: float,
) -> ScoreResult:
    """Compute the SOL score for a candidate kernel.

    S(T_k) = (T_b - T_SOL) / ((T_k - T_SOL) + (T_b - T_SOL))

    Where T_b = baseline, T_SOL = hardware limit, T_k = candidate.
    """
    calibration_warning = baseline_latency_us <= t_sol_us
    reward_hack_suspect = candidate_latency_us < t_sol_us

    gap = baseline_latency_us - t_sol_us
    if gap <= 0:
        # Baseline is already at or below hardware limit
        sol_score = 1.0
    else:
        denom = (candidate_latency_us - t_sol_us) + gap
        sol_score = gap / denom if denom > 0 else 1.0
    speedup = baseline_latency_us / candidate_latency_us if candidate_latency_us > 0 else 0.0
    return ScoreResult(
        sol_score=sol_score,
        baseline_latency_us=baseline_latency_us,
        candidate_latency_us=candidate_latency_us,
        t_sol_us=t_sol_us,
        speedup=speedup,
        reward_hack_suspect=reward_hack_suspect,
        calibration_warning=calibration_warning,
    )
