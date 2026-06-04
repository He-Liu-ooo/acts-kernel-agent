"""Tests for ``run_correctness_gate`` — the shared per-workload correctness
loop extracted from baseline_generator / reference_baseline.

``verify_correctness`` is patched out (``src.eval.correctness.verify_correctness``)
so these stay Tier-1 / torch-free: the gate is pure control flow over the
verify call, and these tests pin its enumerate / short-circuit / failure-report
semantics without exercising the real 5-stage verifier.
"""

from __future__ import annotations

from unittest import mock

import pytest

from src.eval.correctness import (
    CorrectnessGateFailure,
    CorrectnessResult,
    run_correctness_gate,
)


def _fn(*args):  # placeholder candidate / reference callables
    return args


def _gen(seed: int) -> tuple:
    return (seed,)


def _passed() -> CorrectnessResult:
    return CorrectnessResult(passed=True)


def _failed() -> CorrectnessResult:
    return CorrectnessResult(passed=False, error_message="boom")


def test_all_pass_returns_none_and_calls_verify_per_workload():
    wl0, wl1 = object(), object()
    with mock.patch(
        "src.eval.correctness.verify_correctness",
        return_value=_passed(),
    ) as verify:
        out = run_correctness_gate(
            candidate_fn=_fn,
            reference_fn=_fn,
            input_generators=[_gen, _gen],
            workloads=[wl0, wl1],
        )
    assert out is None
    assert verify.call_count == 2


def test_first_failure_reports_index_and_result_and_short_circuits():
    wl0, wl1, wl2 = object(), object(), object()
    results = [_passed(), _failed(), _passed()]
    with mock.patch(
        "src.eval.correctness.verify_correctness",
        side_effect=results,
    ) as verify:
        out = run_correctness_gate(
            candidate_fn=_fn,
            reference_fn=_fn,
            input_generators=[_gen, _gen, _gen],
            workloads=[wl0, wl1, wl2],
        )
    assert isinstance(out, CorrectnessGateFailure)
    assert out.index == 1
    assert out.workload is wl1
    assert out.result is results[1]
    assert out.exception is None
    # Stops at the first failure: verify ran twice (wl0 pass, wl1 fail).
    assert verify.call_count == 2


def test_verify_raising_is_reported_not_reraised_and_stops():
    wl0, wl1 = object(), object()
    boom = RuntimeError("kernel launch failed")
    with mock.patch(
        "src.eval.correctness.verify_correctness",
        side_effect=boom,
    ) as verify:
        out = run_correctness_gate(
            candidate_fn=_fn,
            reference_fn=_fn,
            input_generators=[_gen, _gen],
            workloads=[wl0, wl1],
        )
    assert isinstance(out, CorrectnessGateFailure)
    assert out.index == 0
    assert out.workload is wl0
    assert out.exception is boom
    assert out.result is None
    # Raised on the first call; iteration stopped.
    assert verify.call_count == 1


def test_length_mismatch_raises_value_error_without_calling_verify():
    with mock.patch(
        "src.eval.correctness.verify_correctness",
        return_value=_passed(),
    ) as verify:
        with pytest.raises(ValueError):
            run_correctness_gate(
                candidate_fn=_fn,
                reference_fn=_fn,
                input_generators=[_gen],
                workloads=[object(), object()],
            )
    assert verify.call_count == 0


def test_empty_lists_return_none():
    with mock.patch(
        "src.eval.correctness.verify_correctness",
        return_value=_passed(),
    ) as verify:
        out = run_correctness_gate(
            candidate_fn=_fn,
            reference_fn=_fn,
            input_generators=[],
            workloads=[],
        )
    assert out is None
    assert verify.call_count == 0
