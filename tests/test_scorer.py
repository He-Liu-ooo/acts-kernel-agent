"""Tests for eval/scorer.py — SOL score computation."""

from src.eval.scorer import compute_sol_score


def test_baseline_equals_candidate():
    """T_k = T_b => S = 0.5 (no improvement)."""
    result = compute_sol_score(100.0, 100.0, 50.0)
    assert result.sol_score == 0.5
    assert result.speedup == 1.0
    assert not result.reward_hack_suspect
    assert not result.calibration_warning


def test_candidate_reaches_sol():
    """T_k = T_SOL => S = 1.0 (hardware limit)."""
    result = compute_sol_score(100.0, 50.0, 50.0)
    assert result.sol_score == 1.0
    assert not result.reward_hack_suspect
    assert not result.calibration_warning


def test_candidate_regressed():
    """T_k > T_b => S < 0.5 (regression)."""
    result = compute_sol_score(100.0, 200.0, 50.0)
    assert result.sol_score < 0.5
    assert not result.reward_hack_suspect


def test_reward_hack_flag():
    """T_k < T_SOL => reward_hack_suspect (candidate beats physics)."""
    result = compute_sol_score(100.0, 5.0, 10.0)
    assert result.reward_hack_suspect is True
    assert result.sol_score > 1.0  # Raw score, not clamped


def test_calibration_warning_flag():
    """T_b <= T_SOL => calibration_warning (baseline already at limit)."""
    result = compute_sol_score(8.0, 8.0, 10.0)
    assert result.calibration_warning is True
    assert result.sol_score == 1.0


def test_no_flags_normal_case():
    """Normal case: T_b > T_SOL and T_k >= T_SOL => no flags."""
    result = compute_sol_score(100.0, 70.0, 50.0)
    assert not result.reward_hack_suspect
    assert not result.calibration_warning
    assert 0.5 < result.sol_score < 1.0
