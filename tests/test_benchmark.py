"""Tests for eval/benchmark.py — CUDA-event latency measurement.

Tests inject a ``BenchmarkTimer`` (call-order recorder returning a fixed
elapsed-ms sequence) and a ``kernel_fn`` callable, so the dispatch /
aggregation / failure logic can be exercised without torch or a GPU.

GPU-backed behaviour (real CUDA events, real L2 flush) is covered by the
production timer path and not unit-tested here — test venv is torch-free
per the project's conftest convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import pytest

from src.benchmark.problem import Workload
from src.config import ACTSConfig
from src.eval.benchmark import BenchmarkError, BenchmarkResult, benchmark_kernel
from src.kernels.kernel import Kernel, KernelSpec, KernelType


# ── Fakes ──────────────────────────────────────────────────────────────────


@dataclass
class RecordingTimer:
    """Test double for BenchmarkTimer.

    Records every method call in order + returns a scripted sequence of
    elapsed-ms values from ``finalize_ms``.
    """

    elapsed_ms_sequence: list[float]
    calls: list[str] = field(default_factory=list)
    _idx: int = 0

    def prepare(self) -> None:
        self.calls.append("prepare")

    def flush_l2(self) -> None:
        self.calls.append("flush_l2")

    def record_start(self) -> None:
        self.calls.append("record_start")

    def record_end(self) -> None:
        self.calls.append("record_end")

    def finalize_ms(self) -> float:
        self.calls.append("finalize_ms")
        v = self.elapsed_ms_sequence[self._idx]
        self._idx += 1
        return v


def _make_kernel() -> Kernel:
    spec = KernelSpec(name="k", kernel_type=KernelType.ELEMENTWISE)
    return Kernel(spec=spec, source_code="")


def _wl(uuid: str) -> Workload:
    return Workload(uuid=uuid, axes={}, inputs={})


def _gen(seed: int) -> tuple:
    return ()


def _noop_kernel(*args) -> None:
    pass


def _run(
    *,
    workloads: list[Workload],
    generators: list[Callable[[int], tuple]],
    timer_sequence: list[float],
    kernel_fn: Callable = _noop_kernel,
    warmup: int = 2,
    timed: int = 3,
    discard_first: int = 1,
) -> tuple[BenchmarkResult, RecordingTimer]:
    """Helper: runs benchmark_kernel with injected timer + kernel_fn."""
    config = ACTSConfig()
    config.warmup_runs = warmup
    config.timed_runs = timed
    timer = RecordingTimer(elapsed_ms_sequence=timer_sequence)
    result = benchmark_kernel(
        _make_kernel(),
        config,
        workloads=workloads,
        input_generators=generators,
        timer_factory=lambda: timer,
        kernel_fn=kernel_fn,
        discard_first=discard_first,
    )
    return result, timer


# ── Dataclass / placeholder path ───────────────────────────────────────────


def test_benchmark_result_defaults_are_zero():
    r = BenchmarkResult()
    assert r.median_latency_us == 0.0
    assert r.min_latency_us == 0.0
    assert r.max_latency_us == 0.0
    assert r.warmup_runs == 0
    assert r.timed_runs == 0
    assert r.per_workload_latency_us == {}
    assert r.workload_errors == {}


def test_empty_workloads_returns_sentinel_latency_without_calling_timer():
    """Placeholder path (pre-SOL wiring): no workloads → non-zero sentinel so
    downstream SOL scoring doesn't silently collapse to 1.0. Preserves the
    prior synthetic-100us behavior."""
    config = ACTSConfig()
    timer_factory_called = {"n": 0}

    def factory():
        timer_factory_called["n"] += 1
        return RecordingTimer([])

    result = benchmark_kernel(
        _make_kernel(),
        config,
        workloads=[],
        input_generators=[],
        timer_factory=factory,
        kernel_fn=_noop_kernel,
    )
    assert result.median_latency_us == 100.0
    assert result.min_latency_us == 100.0
    assert result.max_latency_us == 100.0
    assert result.per_workload_latency_us == {}
    assert timer_factory_called["n"] == 0


# ── Input validation ───────────────────────────────────────────────────────


def test_mismatched_workloads_and_generators_raises():
    config = ACTSConfig()
    with pytest.raises(ValueError, match="workloads.*input_generators"):
        benchmark_kernel(
            _make_kernel(),
            config,
            workloads=[_wl("a"), _wl("b")],
            input_generators=[_gen],  # length 1 vs 2 workloads
            timer_factory=lambda: RecordingTimer([]),
            kernel_fn=_noop_kernel,
        )


# ── Happy path ─────────────────────────────────────────────────────────────


def test_single_workload_median_in_microseconds():
    """Timer emits 3 timed iters → median_ms * 1000 = microseconds."""
    result, _timer = _run(
        workloads=[_wl("wl1")],
        generators=[_gen],
        timer_sequence=[999.0, 0.010, 0.020, 0.030],
        warmup=2,
        timed=3,
        discard_first=1,
    )
    assert result.per_workload_latency_us["wl1"] == pytest.approx(20.0)
    assert result.median_latency_us == pytest.approx(20.0)
    assert result.workload_errors == {}


def test_multi_workload_median_of_medians():
    """3 workloads with per-workload medians 10us / 20us / 30us → overall median 20us."""
    sequence = [
        999.0, 0.010, 0.010, 0.010,
        999.0, 0.020, 0.020, 0.020,
        999.0, 0.030, 0.030, 0.030,
    ]
    result, _ = _run(
        workloads=[_wl("wl1"), _wl("wl2"), _wl("wl3")],
        generators=[_gen, _gen, _gen],
        timer_sequence=sequence,
        warmup=2,
        timed=3,
        discard_first=1,
    )
    assert result.per_workload_latency_us["wl1"] == pytest.approx(10.0)
    assert result.per_workload_latency_us["wl2"] == pytest.approx(20.0)
    assert result.per_workload_latency_us["wl3"] == pytest.approx(30.0)
    assert result.median_latency_us == pytest.approx(20.0)
    assert result.min_latency_us == pytest.approx(10.0)
    assert result.max_latency_us == pytest.approx(30.0)


def test_discard_first_drops_first_timed_iteration():
    """Outlier first timed iter (e.g., allocator pause) must not pollute median."""
    result, _ = _run(
        workloads=[_wl("wl1")],
        generators=[_gen],
        timer_sequence=[999.0, 0.010, 0.011, 0.012],
        warmup=1,
        timed=3,
        discard_first=1,
    )
    assert result.per_workload_latency_us["wl1"] == pytest.approx(11.0)


def test_config_warmup_and_timed_runs_echoed_in_result():
    """BenchmarkResult reports the counts actually used."""
    result, _ = _run(
        workloads=[_wl("wl1")],
        generators=[_gen],
        timer_sequence=[0.0, 0.010, 0.020, 0.030],
        warmup=5,
        timed=3,
        discard_first=1,
    )
    assert result.warmup_runs == 5
    assert result.timed_runs == 3


# ── Call-order: flush L2 before start.record ──────────────────────────────


def test_flush_l2_called_before_record_start_every_iter():
    """Option A: cold-cache timing → flush must precede start.record per iter."""
    _result, timer = _run(
        workloads=[_wl("wl1")],
        generators=[_gen],
        timer_sequence=[0.0, 0.010, 0.010, 0.010],
        warmup=0,
        timed=3,
        discard_first=1,
    )
    # Find every (flush_l2, record_start) pairing per iter; flush must precede.
    flush_positions = [i for i, c in enumerate(timer.calls) if c == "flush_l2"]
    start_positions = [i for i, c in enumerate(timer.calls) if c == "record_start"]
    assert len(flush_positions) == len(start_positions) >= 4  # 0 warmup + 4 iters (3+1)
    for flush_i, start_i in zip(flush_positions, start_positions):
        assert flush_i < start_i, f"flush at {flush_i} must precede record_start at {start_i}"


def test_iter_call_sequence_is_prepare_flush_start_end_finalize():
    """Each iter must emit: prepare, flush_l2, record_start, record_end, finalize_ms."""
    _result, timer = _run(
        workloads=[_wl("wl1")],
        generators=[_gen],
        timer_sequence=[0.010, 0.010],
        warmup=0,
        timed=1,
        discard_first=1,
    )
    expected = ["prepare", "flush_l2", "record_start", "record_end", "finalize_ms"] * 2
    assert timer.calls == expected


# ── Failure handling ──────────────────────────────────────────────────────


def test_workload_launch_failure_marks_inf_and_continues():
    """One of three workloads raises mid-timed → its latency is inf, others succeed."""

    def kernel(*args):
        if args and args[0] == "RAISE":
            raise RuntimeError("CUDA launch failed")

    def gen_ok(seed: int) -> tuple:
        return ("OK",)

    def gen_raise(seed: int) -> tuple:
        return ("RAISE",)

    config = ACTSConfig()
    config.warmup_runs = 1
    config.timed_runs = 2
    timer = RecordingTimer(elapsed_ms_sequence=[0.010] * 100)
    result = benchmark_kernel(
        _make_kernel(),
        config,
        workloads=[_wl("wl1"), _wl("wl2"), _wl("wl3")],
        input_generators=[gen_ok, gen_raise, gen_ok],
        timer_factory=lambda: timer,
        kernel_fn=kernel,
        discard_first=1,
    )
    assert math.isfinite(result.per_workload_latency_us["wl1"])
    assert math.isinf(result.per_workload_latency_us["wl2"])
    assert math.isfinite(result.per_workload_latency_us["wl3"])
    assert "wl2" in result.workload_errors
    assert "CUDA launch failed" in result.workload_errors["wl2"]


def test_warmup_failure_marks_workload_dead_without_timing():
    """Warmup raises → skip timed iters for that workload; mark inf.

    Uses 2 workloads so the survivor keeps us above the majority floor and
    the failing workload is recorded without tripping ``BenchmarkError``.
    """

    def kernel(*args):
        if args and args[0] == "RAISE":
            raise RuntimeError("OOM during warmup")

    def gen_raise(seed: int) -> tuple:
        return ("RAISE",)

    def gen_ok(seed: int) -> tuple:
        return ("OK",)

    config = ACTSConfig()
    config.warmup_runs = 1
    config.timed_runs = 5
    timer = RecordingTimer(elapsed_ms_sequence=[0.010] * 100)
    result = benchmark_kernel(
        _make_kernel(),
        config,
        workloads=[_wl("dies"), _wl("ok")],
        input_generators=[gen_raise, gen_ok],
        timer_factory=lambda: timer,
        kernel_fn=kernel,
        discard_first=1,
    )
    assert math.isinf(result.per_workload_latency_us["dies"])
    assert "OOM during warmup" in result.workload_errors["dies"]
    assert math.isfinite(result.per_workload_latency_us["ok"])
    # Timer.finalize_ms called only for the surviving workload's timed iters
    # (timed=5 + discard=1 = 6); no finalize for warmup or for dead workload.
    assert timer.calls.count("finalize_ms") == 6


def test_majority_workload_failure_raises_benchmark_error():
    """3 of 4 workloads fail → raise BenchmarkError (no viable result to return)."""

    def kernel(*args):
        if args and args[0] == "RAISE":
            raise RuntimeError("launch failed")

    def gen_ok(seed: int) -> tuple:
        return ("OK",)

    def gen_raise(seed: int) -> tuple:
        return ("RAISE",)

    config = ACTSConfig()
    config.warmup_runs = 1
    config.timed_runs = 2
    timer = RecordingTimer(elapsed_ms_sequence=[0.010] * 100)
    with pytest.raises(BenchmarkError, match="1/4"):
        benchmark_kernel(
            _make_kernel(),
            config,
            workloads=[_wl("a"), _wl("b"), _wl("c"), _wl("d")],
            input_generators=[gen_raise, gen_raise, gen_raise, gen_ok],
            timer_factory=lambda: timer,
            kernel_fn=kernel,
            discard_first=1,
        )


def test_timer_factory_called_per_workload():
    """Each workload must get a fresh timer — a CUDA fault on one workload
    can leave the stream in a sticky error state, so reusing the timer
    would let a local fault poison subsequent workloads."""
    created: list[RecordingTimer] = []

    def factory() -> RecordingTimer:
        t = RecordingTimer(elapsed_ms_sequence=[0.010] * 100)
        created.append(t)
        return t

    config = ACTSConfig()
    config.warmup_runs = 1
    config.timed_runs = 2
    benchmark_kernel(
        _make_kernel(),
        config,
        workloads=[_wl("a"), _wl("b"), _wl("c")],
        input_generators=[_gen, _gen, _gen],
        timer_factory=factory,
        kernel_fn=_noop_kernel,
        discard_first=1,
    )
    assert len(created) == 3, (
        f"Expected one timer per workload (3), got {len(created)}"
    )


def test_exactly_half_failure_does_not_raise():
    """2 of 4 survive (exactly 50%) → returns result with inf markers."""

    def kernel(*args):
        if args and args[0] == "RAISE":
            raise RuntimeError("boom")

    def gen_ok(seed: int) -> tuple:
        return ("OK",)

    def gen_raise(seed: int) -> tuple:
        return ("RAISE",)

    config = ACTSConfig()
    config.warmup_runs = 1
    config.timed_runs = 2
    timer = RecordingTimer(elapsed_ms_sequence=[0.010] * 100)
    result = benchmark_kernel(
        _make_kernel(),
        config,
        workloads=[_wl("a"), _wl("b"), _wl("c"), _wl("d")],
        input_generators=[gen_ok, gen_raise, gen_raise, gen_ok],
        timer_factory=lambda: timer,
        kernel_fn=kernel,
        discard_first=1,
    )
    finite = [v for v in result.per_workload_latency_us.values() if math.isfinite(v)]
    assert len(finite) == 2
    # Overall median computed from survivors only.
    assert math.isfinite(result.median_latency_us)
