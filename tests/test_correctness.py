"""Tests for eval/correctness.py — 5-stage verification gate.

Tests use a scalar-valued ComparisonPolicy so the module can be exercised
without torch in the test venv. Real torch wiring is covered by the
production policy (used when no policy is injected).
"""

from __future__ import annotations

import pytest

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
    # Stage 3 runs a fixed stability seed (7 → x=8). Candidate matches the
    # reference on smoke (42) and sweep (0..4) inputs but returns NaN at the
    # stability seed; verification must catch the NaN even though earlier
    # stages saw clean numerics.
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


# ── Defaults pinned to SOL-ExecBench's ToleranceSpec ───────────────────────


def test_verify_correctness_atol_rtol_defaults_match_sol_execbench():
    """Drift sentinel: ``verify_correctness`` atol/rtol defaults track
    SOL-ExecBench's ``ToleranceSpec`` defaults. Tightening below bf16's
    ULP floor (~7.8e-3 at unit magnitude) false-flags every bf16 problem.
    Skipped when SOL isn't importable (then the literals are unverifiable).
    """
    import inspect

    try:
        from sol_execbench.core.data.workload import ToleranceSpec
    except ImportError:
        pytest.skip("sol_execbench not importable")

    sol_defaults = ToleranceSpec()
    sig = inspect.signature(verify_correctness)
    assert sig.parameters["atol"].default == sol_defaults.max_atol
    assert sig.parameters["rtol"].default == sol_defaults.max_rtol


# ── Workload tolerance override ────────────────────────────────────────────


class _FakeTol:
    def __init__(self, max_atol: float, max_rtol: float) -> None:
        self.max_atol = max_atol
        self.max_rtol = max_rtol


class _FakeWorkload:
    """Duck-typed Workload — verify_correctness only reads
    ``workload.tolerance.max_atol`` and ``workload.tolerance.max_rtol`` for
    the override logic. Real ``sol_execbench.Workload`` is heavier
    (pydantic, axes, inputs, uuid); not needed for this unit test."""

    def __init__(self, max_atol: float, max_rtol: float) -> None:
        self.tolerance = _FakeTol(max_atol, max_rtol)


def test_workload_tolerance_overrides_loose_default_to_tight():
    """A candidate that passes both the loose (1e-2) and strict (1e-5)
    defaults fails when the workload's tolerance is 1e-3 — the workload's
    value wins. Absolute-error 5e-3 satisfies ScalarPolicy at every default
    threshold (smallest loose seed-0 ≈ 3e-2; smallest strict seed-1000
    ≈ 0.2) but exceeds the workload's 1e-3 atol at stage 1."""
    def slightly_off(x: float) -> float:
        return x * 2.0 + 5e-3

    r_default = verify_correctness(slightly_off, _ref, _gen, policy=ScalarPolicy())
    assert r_default.passed is True

    r_tight = verify_correctness(
        slightly_off, _ref, _gen, policy=ScalarPolicy(),
        workload=_FakeWorkload(max_atol=1e-3, max_rtol=0.0),
    )
    assert r_tight.passed is False
    assert r_tight.failed_stage is CorrectnessStage.SMOKE_TEST


def test_workload_tolerance_overrides_strict_anti_cheat_to_loose():
    """A candidate with 0.1% relative error passes the loose defaults at
    stages 1–4 (rtol*|ref| dominates) but fails the strict anti-cheat
    threshold at stage 5 (rtol drops 100×). With a workload at
    atol=rtol=1e-2 the override loosens anti-cheat too and the candidate
    passes — confirming the override applies to stage 5, not just 1–4."""
    def slightly_off(x: float) -> float:
        return x * 2.0 * 1.001   # 0.1% relative error

    r_default = verify_correctness(slightly_off, _ref, _gen, policy=ScalarPolicy())
    assert r_default.passed is False
    assert r_default.failed_stage is CorrectnessStage.ANTI_CHEAT

    r_loose = verify_correctness(
        slightly_off, _ref, _gen, policy=ScalarPolicy(),
        workload=_FakeWorkload(max_atol=1e-2, max_rtol=1e-2),
    )
    assert r_loose.passed is True


def test_workload_tolerance_override_is_opt_in():
    """When no workload is passed, the existing defaults still apply."""
    def good(x: float) -> float:
        return x * 2.0

    r = verify_correctness(good, _ref, _gen, policy=ScalarPolicy())
    assert r.passed is True


# ── tol_bound@max_abs telemetry ────────────────────────────────────────────


@pytest.mark.gpu
def test_torch_policy_reports_tol_bound_at_max_abs_position():
    """When ``TorchComparisonPolicy.compare`` rejects on tolerance, the
    failure reason must include ``tol_bound@max_abs`` computed as
    ``atol + rtol * |reference_at_max_abs_idx|``. This is the per-element
    bound SOL actually compared against at the worst-error position — it
    makes ``max_abs`` interpretable (over by 2× vs over by 500× look the
    same in ``max_abs`` alone, but differ wildly in the ratio to
    ``tol_bound``).
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("sol_execbench")

    from src.eval.correctness import TorchComparisonPolicy

    # Worst-error element is at index 1 with |reference|=100, error=1.0.
    # Other elements satisfy the bound. Expected:
    #   tol_bound@max_abs = 1e-5 + 1e-4 * 100 = 1.001e-2
    expected = torch.tensor([1.0, 100.0, 0.5], dtype=torch.float32)
    output = torch.tensor([1.00005, 99.0, 0.5], dtype=torch.float32)

    result = TorchComparisonPolicy().compare(
        output, expected, atol=1e-5, rtol=1e-4,
    )
    assert result.match is False
    assert "tol_bound@max_abs=" in result.reason
    assert "tol_bound@max_abs=1.001e-02" in result.reason, result.reason


@pytest.mark.gpu
def test_tol_bound_at_max_abs_helper_matches_sol_per_element_formula():
    """Direct test of the helper. The per-element bound at the worst-abs
    position must equal SOL's ``atol + rtol * |reference|`` evaluated at
    that index — same formula SOL applies element-wise inside
    ``compute_error_stats``.
    """
    torch = pytest.importorskip("torch")

    from src.eval.correctness import _tol_bound_at_max_abs

    # Worst-error position is index 2 (error 0.5, |ref|=10.0).
    expected = torch.tensor([1.0, 5.0, 10.0, 2.0], dtype=torch.float32)
    output = torch.tensor([1.0, 5.0, 10.5, 2.0], dtype=torch.float32)

    bound = _tol_bound_at_max_abs(output, expected, atol=1e-3, rtol=0.05)
    # 1e-3 + 0.05 * 10.0 = 0.501
    assert bound == pytest.approx(0.501, rel=1e-6)


# ── Multi-output normalization (SOL ``normalize_outputs`` integration) ───


@pytest.mark.gpu
def test_verify_correctness_handles_multi_output_reference():
    """Reference returns a tuple of two tensors; verify_correctness routes
    both candidate and reference outputs through ``normalize_outputs``
    (using the workload definition's ``outputs`` order) before per-name
    tolerance comparison. A correct multi-output candidate must pass."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for multi-output integration test")

    from sol_execbench.core.data import Definition

    definition = Definition.model_validate({
        "name": "two_out",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {
            "out1": {"shape": ["N"], "dtype": "float32"},
            "out2": {"shape": ["N"], "dtype": "float32"},
        },
        "reference": "def run(x):\n    return x.relu(), x.tanh()\n",
        "op_type": "elementwise",
    })

    def input_generator(seed: int) -> tuple:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return (torch.randn(64, generator=gen, device="cuda"),)

    def reference_fn(x):
        return x.relu(), x.tanh()

    def candidate_fn(x):
        return x.relu(), x.tanh()

    result = verify_correctness(
        candidate_fn=candidate_fn,
        reference_fn=reference_fn,
        input_generator=input_generator,
        definition=definition,
    )
    assert result.passed, (
        f"multi-output verify failed at stage "
        f"{result.failed_stage}: {result.error_message}"
    )


@pytest.mark.gpu
def test_verify_correctness_catches_wrong_second_output():
    """Multi-output check must compare every named output. A candidate that
    matches on output 1 but fakes output 2 must fail — otherwise the
    normalization wrapper would only be cosmetic."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for multi-output integration test")

    from sol_execbench.core.data import Definition

    definition = Definition.model_validate({
        "name": "two_out_bad",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {
            "out1": {"shape": ["N"], "dtype": "float32"},
            "out2": {"shape": ["N"], "dtype": "float32"},
        },
        "reference": "def run(x):\n    return x.relu(), x.tanh()\n",
        "op_type": "elementwise",
    })

    def input_generator(seed: int) -> tuple:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return (torch.randn(64, generator=gen, device="cuda"),)

    def reference_fn(x):
        return x.relu(), x.tanh()

    def candidate_fn(x):
        # First output correct; second output is sin(x), not tanh(x).
        return x.relu(), x.sin()

    result = verify_correctness(
        candidate_fn=candidate_fn,
        reference_fn=reference_fn,
        input_generator=input_generator,
        definition=definition,
    )
    assert not result.passed
    assert result.failed_stage is CorrectnessStage.SMOKE_TEST


# ── DPS (destination-passing-style) candidates ─────────────────────────────


@pytest.mark.gpu
def test_verify_correctness_handles_dps_multi_output_candidate():
    """A DPS candidate (``kernel_fn(x, out_a, out_b)``) must verify against a
    PyTorch reference that returns its outputs by value. The gate allocates
    output buffers per call via ``allocate_outputs`` and threads them in
    after the inputs; the filled buffers serve as the candidate's outputs
    for the comparison."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for DPS verification test")

    from sol_execbench.core.data import Definition, Workload

    from src.kernels.kernel import Kernel, KernelSpec, KernelType

    definition = Definition.model_validate({
        "name": "dps_two_out",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {
            "out1": {"shape": ["N"], "dtype": "float32"},
            "out2": {"shape": ["N"], "dtype": "float32"},
        },
        "reference": "def run(x):\n    return x.relu(), x.tanh()\n",
        "op_type": "elementwise",
    })
    workload = Workload.model_validate({
        "uuid": "wl-dps-1", "axes": {"N": 64}, "inputs": {},
    })

    def input_generator(seed: int) -> tuple:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return (torch.randn(64, generator=gen, device="cuda"),)

    def reference_fn(x):
        return x.relu(), x.tanh()

    def candidate_fn(x, out_a, out_b):
        out_a.copy_(x.relu())
        out_b.copy_(x.tanh())

    kernel = Kernel(
        spec=KernelSpec(name="dps_two_out", kernel_type=KernelType.ELEMENTWISE),
        source_code="",
        dps=True,
    )

    result = verify_correctness(
        candidate_fn=candidate_fn,
        reference_fn=reference_fn,
        input_generator=input_generator,
        definition=definition,
        kernel=kernel,
        workload=workload,
    )
    assert result.passed, (
        f"DPS verify failed at stage {result.failed_stage}: {result.error_message}"
    )


@pytest.mark.gpu
def test_verify_correctness_catches_wrong_dps_second_output():
    """A DPS candidate that writes the right buffer for output 1 but the
    wrong tensor for output 2 must fail the gate — every named output is
    compared, not just the first."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for DPS verification test")

    from sol_execbench.core.data import Definition, Workload

    from src.kernels.kernel import Kernel, KernelSpec, KernelType

    definition = Definition.model_validate({
        "name": "dps_two_out_bad",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {
            "out1": {"shape": ["N"], "dtype": "float32"},
            "out2": {"shape": ["N"], "dtype": "float32"},
        },
        "reference": "def run(x):\n    return x.relu(), x.tanh()\n",
        "op_type": "elementwise",
    })
    workload = Workload.model_validate({
        "uuid": "wl-dps-2", "axes": {"N": 64}, "inputs": {},
    })

    def input_generator(seed: int) -> tuple:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return (torch.randn(64, generator=gen, device="cuda"),)

    def reference_fn(x):
        return x.relu(), x.tanh()

    def candidate_fn(x, out_a, out_b):
        out_a.copy_(x.relu())
        out_b.copy_(x.sin())  # wrong: should be tanh

    kernel = Kernel(
        spec=KernelSpec(name="dps_two_out_bad", kernel_type=KernelType.ELEMENTWISE),
        source_code="",
        dps=True,
    )

    result = verify_correctness(
        candidate_fn=candidate_fn,
        reference_fn=reference_fn,
        input_generator=input_generator,
        definition=definition,
        kernel=kernel,
        workload=workload,
    )
    assert not result.passed
    assert result.failed_stage is CorrectnessStage.SMOKE_TEST
