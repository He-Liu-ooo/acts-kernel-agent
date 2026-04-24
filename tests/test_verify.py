"""Tests for pipeline/verify.py — post-search re-verification of the best kernel."""

from __future__ import annotations

from src.kernels.kernel import Kernel
from src.pipeline.verify import verify_optimized_kernel
from tests.conftest import (
    ScalarPolicy,
    make_kernel_spec,
    scalar_gen as _gen,
    scalar_ref as _ref,
)


def _spec():
    return make_kernel_spec(name="verify_test")


def test_verify_passes_on_matching_kernel(tmp_path):
    optimized = Kernel(spec=_spec(), source_code="def kernel_fn(x):\n    return x * 2.0\n")
    result = verify_optimized_kernel(
        optimized,
        reference_fn=_ref,
        input_generator=_gen,
        policy=ScalarPolicy(),
        cache_dir=tmp_path,
    )
    assert result.passed
    assert "passed" in result.details.lower()


def test_verify_fails_on_mismatched_kernel(tmp_path):
    optimized = Kernel(spec=_spec(), source_code="def kernel_fn(x):\n    return x * 3.0\n")
    result = verify_optimized_kernel(
        optimized,
        reference_fn=_ref,
        input_generator=_gen,
        policy=ScalarPolicy(),
        cache_dir=tmp_path,
    )
    assert not result.passed
    assert "smoke_test" in result.details


def test_verify_surfaces_compile_error(tmp_path):
    """A candidate that doesn't compile must fail verification with a compile-phrased reason."""
    optimized = Kernel(spec=_spec(), source_code="def kernel_fn(: broken\n")
    result = verify_optimized_kernel(
        optimized,
        reference_fn=_ref,
        input_generator=_gen,
        policy=ScalarPolicy(),
        cache_dir=tmp_path,
    )
    assert not result.passed
    assert "compil" in result.details.lower()


def test_verify_emits_start_and_done_events_on_pass(tmp_path):
    """verify_start fires at entry; verify_done at exit with passed=True
    and a non-empty detail_short on the happy path."""
    import json
    from src.runtime import events

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        optimized = Kernel(spec=_spec(), source_code="def kernel_fn(x):\n    return x * 2.0\n")
        verify_optimized_kernel(
            optimized,
            reference_fn=_ref,
            input_generator=_gen,
            policy=ScalarPolicy(),
            cache_dir=tmp_path,
        )
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kinds = [r["kind"] for r in records]
    assert kinds == ["verify_start", "verify_done"]
    done = records[-1]
    assert done["passed"] is True
    assert "detail_short" in done


def test_verify_emits_done_on_compile_failure(tmp_path):
    """Compile-failure path also emits a verify_done event so post-mortem
    replay sees every exit path."""
    import json
    from src.runtime import events

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        optimized = Kernel(spec=_spec(), source_code="def kernel_fn(: broken\n")
        verify_optimized_kernel(
            optimized,
            reference_fn=_ref,
            input_generator=_gen,
            policy=ScalarPolicy(),
            cache_dir=tmp_path,
        )
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kinds = [r["kind"] for r in records]
    assert kinds == ["verify_start", "verify_done"]
    done = records[-1]
    assert done["passed"] is False
    assert "compil" in done["detail_short"].lower()
