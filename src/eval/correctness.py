"""5-stage correctness verification gate.

Called by Coder's ``check_correctness_tool`` during its turn. By the time
the Coder returns, correctness is guaranteed (or the branch fails).

Correctness is checked against a caller-supplied ``reference_fn`` — the
PyTorch reference from ``definition.json`` in production. Using the Triton
baseline as oracle would propagate its translation bugs through the run.

Stages (short-circuit on first failure):
    1. Smoke test          — single input, output matches reference
    2. Shape sweep         — N trials, varying seeds / input shapes
    3. Numerical stability — no NaN / Inf on normal inputs
    4. Determinism         — repeated runs on identical input match bitwise
    5. Anti-cheat          — randomized inputs under strict tolerance

Tensor comparison is delegated to a ``ComparisonPolicy`` — the production
torch policy is lazy-built so this module stays import-clean when torch
is absent (unit tests inject a scalar policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Protocol


class CorrectnessStage(Enum):
    SMOKE_TEST = "smoke_test"
    SHAPE_SWEEP = "shape_sweep"
    NUMERICAL_STABILITY = "numerical_stability"
    DETERMINISM = "determinism"
    ANTI_CHEAT = "anti_cheat"


@dataclass
class ComparisonResult:
    match: bool
    max_abs_error: float = 0.0
    reason: str = ""


@dataclass
class CorrectnessResult:
    passed: bool
    failed_stage: CorrectnessStage | None = None
    error_message: str = ""
    max_abs_error: float = 0.0


class ComparisonPolicy(Protocol):
    """Tensor-comparison abstraction.

    Production: torch-backed (``TorchComparisonPolicy``).
    Tests: scalar-backed, no torch dependency.
    """

    def compare(
        self, output: Any, expected: Any, *, atol: float, rtol: float
    ) -> ComparisonResult: ...

    def contains_non_finite(self, output: Any) -> bool: ...

    def bitwise_equal(self, a: Any, b: Any) -> bool: ...


@lru_cache(maxsize=1)
def _try_import_sol():
    """Resolve sol_execbench symbols once per process. Returns None if absent."""
    try:
        from sol_execbench.core.bench.correctness import compute_error_stats
        from sol_execbench.core.data.workload import ToleranceSpec
    except ImportError:
        return None
    return compute_error_stats, ToleranceSpec


class TorchComparisonPolicy:
    """Default production policy. Torch is imported lazily inside methods.

    When ``sol_execbench`` is importable, delegates element-wise comparison
    to its ``compute_error_stats`` — gives us matched-ratio tolerance,
    separate NaN/Inf flags, and a hard max-error cap for free. Falls
    back to a local ``torch.allclose``-based check when SOL-ExecBench
    isn't installed (keeps the module usable for non-SOL benchmarks).
    """

    def compare(
        self, output: Any, expected: Any, *, atol: float, rtol: float
    ) -> ComparisonResult:
        if output.shape != expected.shape:
            return ComparisonResult(
                match=False,
                reason=f"shape mismatch: {tuple(output.shape)} vs {tuple(expected.shape)}",
            )
        sol = _try_import_sol()
        if sol is None:
            return self._compare_fallback(output, expected, atol=atol, rtol=rtol)
        compute_error_stats, ToleranceSpec = sol

        tolerance = ToleranceSpec(
            max_atol=atol,
            max_rtol=rtol,
            required_matched_ratio=1.0,  # strict — every element must pass
        )
        correctness, exceeds = compute_error_stats(output, expected, tolerance)
        max_err = float(correctness.max_absolute_error or 0.0)
        if not exceeds:
            return ComparisonResult(match=True, max_abs_error=max_err)
        if correctness.has_nan:
            reason = "NaN in output or reference"
        elif correctness.has_inf:
            reason = "Inf in output or reference"
        else:
            reason = (
                f"tolerance exceeded: max_abs={max_err:.3e}, "
                f"max_rel={float(correctness.max_relative_error or 0.0):.3e} "
                f"(atol={atol}, rtol={rtol})"
            )
        return ComparisonResult(match=False, max_abs_error=max_err, reason=reason)

    @staticmethod
    def _compare_fallback(
        output: Any, expected: Any, *, atol: float, rtol: float
    ) -> ComparisonResult:
        import torch

        out_f = output.detach().float()
        exp_f = expected.detach().float()
        out_nan = torch.isnan(out_f)
        exp_nan = torch.isnan(exp_f)
        if out_nan.any() or exp_nan.any():
            if not torch.equal(out_nan, exp_nan):
                return ComparisonResult(match=False, reason="NaN position mismatch")
            mask = ~out_nan
            if not mask.any():
                return ComparisonResult(match=True, max_abs_error=0.0, reason="all-NaN")
            out_f = out_f[mask]
            exp_f = exp_f[mask]
        abs_err = (out_f - exp_f).abs()
        max_err = float(abs_err.max())
        if torch.allclose(out_f, exp_f, atol=atol, rtol=rtol):
            return ComparisonResult(match=True, max_abs_error=max_err)
        mean_err = float(abs_err.mean())
        return ComparisonResult(
            match=False,
            max_abs_error=max_err,
            reason=(
                f"tolerance exceeded: max_abs={max_err:.3e}, "
                f"mean_abs={mean_err:.3e} (atol={atol}, rtol={rtol})"
            ),
        )

    def contains_non_finite(self, output: Any) -> bool:
        import torch

        return bool(torch.isnan(output).any() or torch.isinf(output).any())

    def bitwise_equal(self, a: Any, b: Any) -> bool:
        import torch

        return bool(torch.equal(a, b))


@dataclass
class _StageOutcome:
    match: bool
    max_abs_error: float = 0.0
    reason: str = ""


def _run_compare_trial(
    candidate_fn: Callable[..., Any],
    reference_fn: Callable[..., Any],
    input_generator: Callable[[int], tuple],
    *,
    seed: int,
    policy: ComparisonPolicy,
    atol: float,
    rtol: float,
) -> _StageOutcome:
    try:
        args = input_generator(seed)
        expected = reference_fn(*args)
        output = candidate_fn(*args)
    except Exception as exc:
        return _StageOutcome(match=False, reason=f"{type(exc).__name__}: {exc}")
    cmp = policy.compare(output, expected, atol=atol, rtol=rtol)
    return _StageOutcome(
        match=cmp.match, max_abs_error=cmp.max_abs_error, reason=cmp.reason
    )


def _fail(
    stage: CorrectnessStage,
    reason: str,
    max_abs_error: float,
    *,
    trial: int | None = None,
) -> CorrectnessResult:
    prefix = f"[{stage.value}]"
    if trial is not None:
        prefix = f"{prefix} trial {trial}:"
    return CorrectnessResult(
        passed=False,
        failed_stage=stage,
        error_message=f"{prefix} {reason}",
        max_abs_error=max_abs_error,
    )


def verify_correctness(
    candidate_fn: Callable[..., Any],
    reference_fn: Callable[..., Any],
    input_generator: Callable[[int], tuple],
    *,
    policy: ComparisonPolicy | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    strict_atol: float = 1e-5,
    strict_rtol: float = 1e-4,
    n_sweep_trials: int = 5,
    n_anti_cheat_trials: int = 3,
) -> CorrectnessResult:
    """Run the 5-stage correctness gate.

    ``input_generator(seed)`` returns the args tuple for a trial. Seeds
    used: 42 (smoke), 0..n_sweep_trials-1 (sweep), 7 (stability), 11
    (determinism), 1000..1000+n_anti_cheat_trials-1 (anti-cheat).
    """
    policy = policy or TorchComparisonPolicy()
    worst_error = 0.0

    # Stage 1: Smoke test
    stage = CorrectnessStage.SMOKE_TEST
    r = _run_compare_trial(
        candidate_fn, reference_fn, input_generator,
        seed=42, policy=policy, atol=atol, rtol=rtol,
    )
    if not r.match:
        return _fail(stage, r.reason, r.max_abs_error)
    worst_error = max(worst_error, r.max_abs_error)

    # Stage 2: Shape sweep
    stage = CorrectnessStage.SHAPE_SWEEP
    for i in range(n_sweep_trials):
        r = _run_compare_trial(
            candidate_fn, reference_fn, input_generator,
            seed=i, policy=policy, atol=atol, rtol=rtol,
        )
        if not r.match:
            return _fail(stage, r.reason, r.max_abs_error, trial=i)
        worst_error = max(worst_error, r.max_abs_error)

    # Stage 3: Numerical stability — candidate must match oracle AND be finite.
    # The oracle compare guards against seed-7-specific wrong answers that
    # would otherwise slip past a pure NaN/Inf check.
    stage = CorrectnessStage.NUMERICAL_STABILITY
    try:
        args = input_generator(7)
        expected = reference_fn(*args)
        output = candidate_fn(*args)
    except Exception as exc:
        return _fail(stage, f"{type(exc).__name__}: {exc}", worst_error)
    cmp = policy.compare(output, expected, atol=atol, rtol=rtol)
    if not cmp.match:
        return _fail(stage, cmp.reason, cmp.max_abs_error)
    worst_error = max(worst_error, cmp.max_abs_error)
    if policy.contains_non_finite(output):
        return _fail(stage, "Output contains NaN or Inf on normal input.", worst_error)

    # Stage 4: Determinism — candidate must match oracle AND reproduce bitwise.
    # The oracle compare guards against seed-11-specific wrong answers that
    # would otherwise slip past a pure self-equality check.
    stage = CorrectnessStage.DETERMINISM
    try:
        args1 = input_generator(11)
        expected = reference_fn(*args1)
        out1 = candidate_fn(*args1)
        args2 = input_generator(11)
        out2 = candidate_fn(*args2)
    except Exception as exc:
        return _fail(stage, f"{type(exc).__name__}: {exc}", worst_error)
    cmp = policy.compare(out1, expected, atol=atol, rtol=rtol)
    if not cmp.match:
        return _fail(stage, cmp.reason, cmp.max_abs_error)
    worst_error = max(worst_error, cmp.max_abs_error)
    if not policy.bitwise_equal(out1, out2):
        return _fail(
            stage,
            "Repeated runs on identical input produced different outputs.",
            worst_error,
        )

    # Stage 5: Anti-cheat (strict tolerance, fresh seeds)
    stage = CorrectnessStage.ANTI_CHEAT
    for i in range(n_anti_cheat_trials):
        r = _run_compare_trial(
            candidate_fn, reference_fn, input_generator,
            seed=1000 + i, policy=policy, atol=strict_atol, rtol=strict_rtol,
        )
        if not r.match:
            return _fail(stage, r.reason, r.max_abs_error, trial=i)
        worst_error = max(worst_error, r.max_abs_error)

    return CorrectnessResult(passed=True, max_abs_error=worst_error)
