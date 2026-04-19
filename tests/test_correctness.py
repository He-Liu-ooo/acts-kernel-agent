"""Tests for eval/correctness.py — 5-stage verification gate.

Tests use a scalar-valued ComparisonPolicy so the module can be exercised
without torch in the test venv. Real torch wiring is covered by the
production policy (used when no policy is injected).
"""

from __future__ import annotations

from src.eval.correctness import CorrectnessStage, verify_correctness
from tests.conftest import ScalarPolicy, scalar_gen as _gen, scalar_ref as _ref


def _good_candidate(x: float) -> float:
    return x * 2.0


# ── Happy path ─────────────────────────────────────────────────────────────


def test_verify_passes_when_candidate_matches_reference():
    r = verify_correctness(_good_candidate, _ref, _gen, policy=ScalarPolicy())
    assert r.passed is True
    assert r.failed_stage is None
    assert r.error_message == ""


# ── Stage 1: smoke ─────────────────────────────────────────────────────────


def test_verify_fails_smoke_when_outputs_differ():
    def bad(x):
        return x * 3.0

    r = verify_correctness(bad, _ref, _gen, policy=ScalarPolicy())
    assert r.passed is False
    assert r.failed_stage is CorrectnessStage.SMOKE_TEST
    assert "smoke_test" in r.error_message


def test_verify_smoke_failure_short_circuits_later_stages():
    """If smoke fails, downstream stages are not run."""
    calls = {"candidate": 0, "reference": 0}

    def bad(x):
        calls["candidate"] += 1
        return x * 3.0

    def ref(x):
        calls["reference"] += 1
        return x * 2.0

    verify_correctness(bad, ref, _gen, policy=ScalarPolicy())
    # Smoke runs 1 trial → 1 call each. No further stages ran.
    assert calls["candidate"] == 1
    assert calls["reference"] == 1


# ── Stage 2: shape sweep ───────────────────────────────────────────────────


def test_verify_fails_shape_sweep_when_only_some_seeds_match():
    """Candidate passes smoke seed but fails on a later seed."""
    def sometimes_bad(x):
        # smoke uses seed=42 → x=43; shape sweep uses seeds 0..4
        return x * 2.0 if x == 43.0 else x * 3.0

    r = verify_correctness(
        sometimes_bad, _ref, _gen, policy=ScalarPolicy(), n_sweep_trials=5
    )
    assert r.passed is False
    assert r.failed_stage is CorrectnessStage.SHAPE_SWEEP


# ── Stage 3: numerical stability ───────────────────────────────────────────


def test_verify_fails_numerical_stability_on_nan_output():
    """Candidate that matches reference numerically but produces NaN fails stage 3."""
    # Craft: reference returns NaN, candidate returns NaN. Both match under
    # ScalarPolicy because NaN == NaN via abs() returns NaN which is not <=,
    # so compare would fail. Instead, use a candidate that matches ref for
    # trials but returns NaN at the stability seed (7).
    def cand(x):
        if x == 8.0:  # seed 7 → x=8
            return float("nan")
        return x * 2.0

    r = verify_correctness(cand, _ref, _gen, policy=ScalarPolicy())
    # Smoke (42→43) and sweep (0..4→1..5) pass; stage 3 (seed 7→8) triggers NaN.
    assert r.passed is False
    assert r.failed_stage is CorrectnessStage.NUMERICAL_STABILITY
    assert "nan" in r.error_message.lower() or "inf" in r.error_message.lower()


def test_verify_fails_numerical_stability_on_inf_output():
    def cand(x):
        if x == 8.0:
            return float("inf")
        return x * 2.0

    r = verify_correctness(cand, _ref, _gen, policy=ScalarPolicy())
    assert r.failed_stage is CorrectnessStage.NUMERICAL_STABILITY


def test_verify_fails_numerical_stability_when_finite_but_wrong():
    """A candidate whose stability-seed output is finite yet disagrees with
    the oracle must still fail stage 3 — the gate cannot certify seed-7
    correctness just by checking for NaN/Inf."""
    def cand(x):
        if x == 8.0:  # seed 7 → x=8
            return 999.0  # finite, but very wrong
        return x * 2.0

    r = verify_correctness(cand, _ref, _gen, policy=ScalarPolicy())
    assert r.passed is False
    assert r.failed_stage is CorrectnessStage.NUMERICAL_STABILITY


# ── Stage 4: determinism ───────────────────────────────────────────────────


def test_verify_fails_determinism_when_repeated_runs_differ():
    """Candidate whose successive determinism calls differ → stage 4 fails."""
    state = {"n": 0}

    def flaky(x):
        state["n"] += 1
        # Calls 1..7 (smoke + sweep + stability) are exact.
        # Determinism calls are 8 and 9 → return distinct values so the
        # bitwise check sees a mismatch.
        if state["n"] <= 7:
            return x * 2.0
        return x * 2.0 + state["n"] * 1e-12

    r = verify_correctness(flaky, _ref, _gen, policy=ScalarPolicy())
    assert r.passed is False
    assert r.failed_stage is CorrectnessStage.DETERMINISM


def test_verify_fails_determinism_when_repeatable_but_wrong():
    """A candidate that is repeatable on the determinism seed but disagrees
    with the oracle must fail stage 4 — bitwise self-equality alone can't
    certify seed-11 correctness."""
    def cand(x):
        if x == 12.0:  # seed 11 → x=12
            return 999.0  # repeatable wrong output
        return x * 2.0

    r = verify_correctness(cand, _ref, _gen, policy=ScalarPolicy())
    assert r.passed is False
    assert r.failed_stage is CorrectnessStage.DETERMINISM


# ── Stage 5: anti-cheat (strict tolerance) ────────────────────────────────


def test_verify_fails_anti_cheat_when_precision_exceeds_strict_tolerance():
    """Candidate within normal tolerance but outside strict tolerance fails stage 5."""
    def close_enough(x):
        # 5e-4 drift: within atol=1e-3 but outside strict_atol=1e-9 given the
        # small expected magnitude at anti-cheat seeds.
        return x * 2.0 + 5e-4

    r = verify_correctness(
        close_enough, _ref, _gen, policy=ScalarPolicy(),
        atol=1e-3, rtol=1e-3, strict_atol=1e-9, strict_rtol=1e-9,
    )
    assert r.passed is False
    assert r.failed_stage is CorrectnessStage.ANTI_CHEAT


def test_verify_passes_anti_cheat_when_precision_within_strict_tolerance():
    """Candidate that exactly matches reference passes stage 5."""
    r = verify_correctness(
        _good_candidate, _ref, _gen, policy=ScalarPolicy(),
        strict_atol=1e-8, strict_rtol=1e-8,
    )
    assert r.passed is True


# ── Error handling ─────────────────────────────────────────────────────────


def test_verify_fails_when_candidate_raises():
    """Candidate raising mid-trial is reported against the current stage."""
    def crashy(x):
        raise RuntimeError("kernel launch failed")

    r = verify_correctness(crashy, _ref, _gen, policy=ScalarPolicy())
    assert r.passed is False
    assert r.failed_stage is CorrectnessStage.SMOKE_TEST
    assert "RuntimeError" in r.error_message
    assert "kernel launch failed" in r.error_message


def test_verify_fails_when_candidate_raises_during_stability():
    """Crash during later stage attributes failure to that stage."""
    state = {"n": 0}

    def crashy_later(x):
        state["n"] += 1
        if state["n"] > 6:  # passes smoke (1) + sweep (5); crashes at stability
            raise ValueError("bad input")
        return x * 2.0

    r = verify_correctness(crashy_later, _ref, _gen, policy=ScalarPolicy())
    assert r.passed is False
    assert r.failed_stage is CorrectnessStage.NUMERICAL_STABILITY


# ── Max-error reporting ────────────────────────────────────────────────────


def test_verify_reports_worst_error_across_passing_stages():
    """On full pass, max_abs_error reflects the worst error observed."""
    def slightly_off(x):
        # Tiny error, within tolerance everywhere.
        return x * 2.0 + 1e-7

    r = verify_correctness(
        slightly_off, _ref, _gen, policy=ScalarPolicy(),
        atol=1e-3, rtol=1e-3, strict_atol=1e-3, strict_rtol=1e-3,
    )
    assert r.passed is True
    assert r.max_abs_error > 0
    assert r.max_abs_error < 2e-7


# ── Config: trial counts ───────────────────────────────────────────────────


def test_verify_respects_n_sweep_trials():
    calls = {"n": 0}

    def track(x):
        calls["n"] += 1
        return x * 2.0

    verify_correctness(
        track, _ref, _gen, policy=ScalarPolicy(),
        n_sweep_trials=7, n_anti_cheat_trials=2,
    )
    # 1 (smoke) + 7 (sweep) + 1 (stability) + 2 (determinism) + 2 (anti-cheat) = 13
    assert calls["n"] == 13
