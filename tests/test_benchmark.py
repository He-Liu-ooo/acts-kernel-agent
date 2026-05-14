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

from sol_execbench.core.data import Workload

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
    return Workload.model_validate({"uuid": uuid, "axes": {}, "inputs": {}})


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
    """Each timed iter must emit: prepare, flush_l2, record_start, record_end, finalize_ms.

    A1 PR 1: a single ``prepare`` call from the autotune burn-in step
    fires before any iter, so the call sequence prefix is one extra
    ``prepare`` followed by the standard per-iter sequence ``* N``.
    """
    _result, timer = _run(
        workloads=[_wl("wl1")],
        generators=[_gen],
        timer_sequence=[0.010, 0.010],
        warmup=0,
        timed=1,
        discard_first=1,
    )
    expected = ["prepare"] + (
        ["prepare", "flush_l2", "record_start", "record_end", "finalize_ms"] * 2
    )
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


# ── DPS path: pre-allocated output buffers ────────────────────────────────


def _dps_definition() -> "Definition":  # noqa: F821
    """Single-output definition matching the fake DPS kernel below."""
    from sol_execbench.core.data import Definition

    return Definition.model_validate({
        "name": "dps_unary",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x * 2.0\n",
        "op_type": "elementwise",
    })


def _make_dps_kernel() -> Kernel:
    spec = KernelSpec(name="dps_k", kernel_type=KernelType.ELEMENTWISE)
    return Kernel(spec=spec, source_code="", dps=True)


@pytest.mark.gpu
def test_benchmark_kernel_dps_allocates_outputs():
    """``kernel.dps=True`` → benchmark_kernel calls ``allocate_outputs`` per
    iter, hands the resulting buffers to the kernel as positional args
    after the inputs, and the kernel writes into them in place."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for DPS allocate_outputs path")

    definition = _dps_definition()
    workload = Workload.model_validate({
        "uuid": "wl-dps", "axes": {"N": 1024}, "inputs": {},
    })
    captured: dict = {"calls": 0}

    def kernel_fn(x, out):
        # DPS contract: outputs are positional args after inputs.
        captured["calls"] += 1
        captured["last_out_shape"] = tuple(out.shape)
        captured["last_out_dtype"] = out.dtype
        captured["last_out_device"] = out.device
        out.copy_(x * 2.0)

    def gen(seed: int):
        return (torch.randn(1024, dtype=torch.float32, device="cuda"),)

    config = ACTSConfig()
    config.warmup_runs = 2
    config.timed_runs = 3
    result = benchmark_kernel(
        _make_dps_kernel(),
        config,
        workloads=[workload],
        input_generators=[gen],
        kernel_fn=kernel_fn,
        definition=definition,
        discard_first=1,
    )
    assert result.is_fully_successful
    # A1 PR 1: burn-in (1) + warmup (2) + timed+discard (3+1) = 7 invocations.
    assert captured["calls"] == 7
    assert captured["last_out_shape"] == (1024,)
    assert captured["last_out_dtype"] is torch.float32
    assert captured["last_out_device"].type == "cuda"


@pytest.mark.gpu
def test_benchmark_kernel_dps_requires_definition():
    """Asking for the DPS path without a definition is a contract violation —
    we can't pre-allocate output buffers without the output spec."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for DPS allocate_outputs path")

    workload = Workload.model_validate({
        "uuid": "wl-dps", "axes": {"N": 1024}, "inputs": {},
    })
    config = ACTSConfig()
    config.warmup_runs = 1
    config.timed_runs = 1

    with pytest.raises(ValueError, match="definition"):
        benchmark_kernel(
            _make_dps_kernel(),
            config,
            workloads=[workload],
            input_generators=[lambda s: (torch.randn(1024, device="cuda"),)],
            kernel_fn=lambda x, out: None,
            # definition omitted on purpose
            discard_first=1,
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


# ── last_outputs flattening (dict return → tensor list) ──────────────────


@pytest.mark.gpu
def test_non_dps_dict_return_flattens_to_tensor_list():
    """Non-DPS kernels that return a ``dict`` of named outputs (e.g. LayerNorm
    returning ``{"y": ..., "mean": ..., "rstd": ...}``) must have their values
    flattened into ``last_outputs`` so ``check_lazy_outputs_after_bench`` sees
    real ``torch.Tensor`` instances — not the dict object itself.

    Regression: a naive ``last_outputs = ret if isinstance(ret, (list,
    tuple)) else [ret]`` wraps a dict in a list, so the lazy-output check
    sees a ``dict`` (not a ``Tensor``) and raises ``RewardHackDetected``
    after an otherwise successful benchmark — silently pruning every
    dict-return branch.
    """
    torch = pytest.importorskip("torch")
    from src.eval.anti_cheat import check_lazy_outputs_after_bench

    def kernel_fn(x):
        return {"y": x.relu(), "stats": x.mean()}

    def gen(seed: int) -> tuple:
        return (torch.randn(8, dtype=torch.float32),)

    timer = RecordingTimer(elapsed_ms_sequence=[0.010] * 100)
    config = ACTSConfig()
    config.warmup_runs = 1
    config.timed_runs = 2
    result = benchmark_kernel(
        _make_kernel(),
        config,
        workloads=[_wl("wl-dict")],
        input_generators=[gen],
        timer_factory=lambda: timer,
        kernel_fn=kernel_fn,
        discard_first=1,
    )
    assert result.is_fully_successful, (
        f"benchmark errored: {result.workload_errors}"
    )
    # Bug repro assertion: the dict's values must be flattened, not the
    # dict object packed into [ret].
    assert all(isinstance(t, torch.Tensor) for t in result.last_outputs), (
        f"expected list[Tensor], got types: "
        f"{[type(t).__name__ for t in result.last_outputs]}"
    )
    assert len(result.last_outputs) == 2
    # And the canonical post-bench check must accept these outputs.
    check_lazy_outputs_after_bench(result.last_outputs)


@pytest.mark.gpu
def test_non_dps_tuple_return_flattens_to_tensor_list():
    """Tuple returns must unpack into ``last_outputs`` (regression guard for
    the same flatten path that handles dicts)."""
    torch = pytest.importorskip("torch")
    from src.eval.anti_cheat import check_lazy_outputs_after_bench

    def kernel_fn(x):
        return (x.relu(), x.abs())

    def gen(seed: int) -> tuple:
        return (torch.randn(4, dtype=torch.float32),)

    timer = RecordingTimer(elapsed_ms_sequence=[0.010] * 100)
    config = ACTSConfig()
    config.warmup_runs = 1
    config.timed_runs = 2
    result = benchmark_kernel(
        _make_kernel(),
        config,
        workloads=[_wl("wl-tuple")],
        input_generators=[gen],
        timer_factory=lambda: timer,
        kernel_fn=kernel_fn,
        discard_first=1,
    )
    assert result.is_fully_successful
    assert all(isinstance(t, torch.Tensor) for t in result.last_outputs)
    assert len(result.last_outputs) == 2
    check_lazy_outputs_after_bench(result.last_outputs)


@pytest.mark.gpu
def test_non_dps_single_tensor_return_wraps_in_list():
    """Single-tensor return remains wrapped in a 1-elem list (existing
    contract — guard against over-flattening when we add the dict branch)."""
    torch = pytest.importorskip("torch")
    from src.eval.anti_cheat import check_lazy_outputs_after_bench

    def kernel_fn(x):
        return x.relu()

    def gen(seed: int) -> tuple:
        return (torch.randn(4, dtype=torch.float32),)

    timer = RecordingTimer(elapsed_ms_sequence=[0.010] * 100)
    config = ACTSConfig()
    config.warmup_runs = 1
    config.timed_runs = 2
    result = benchmark_kernel(
        _make_kernel(),
        config,
        workloads=[_wl("wl-single")],
        input_generators=[gen],
        timer_factory=lambda: timer,
        kernel_fn=kernel_fn,
        discard_first=1,
    )
    assert result.is_fully_successful
    assert len(result.last_outputs) == 1
    assert isinstance(result.last_outputs[0], torch.Tensor)
    check_lazy_outputs_after_bench(result.last_outputs)


# ── A1 PR 1: autotune burn-in + _key_tuple_for helper ──────────────────


def test_burn_in_fires_before_warmup_with_reserved_seed():
    """One extra fn invocation lands BEFORE the warmup loop (seed=-1)
    so @triton.autotune's compile+pick cost lands outside the timed
    window. Total fn calls = 1 burn-in + warmup + timed."""
    calls: list[tuple] = []

    def fn(*args):
        calls.append(args)

    def gen(seed: int) -> tuple:
        return (seed,)

    _run(
        workloads=[_wl("wl-burn")],
        generators=[gen],
        timer_sequence=[0.001, 0.002, 0.003, 0.004, 0.005],
        kernel_fn=fn,
        warmup=2,
        timed=3,
        discard_first=0,
    )
    # 1 burn-in + 2 warmup + 3 timed = 6 fn calls.
    assert len(calls) == 6
    # First call carries the reserved burn-in seed.
    assert calls[0] == (-1,)


def test_burn_in_failure_surfaces_as_workload_error():
    """fn raising during burn-in marks the workload latency as inf and
    records the error — same path as a warmup failure. With only one
    workload, the half-survivors-or-die gate fires and raises
    BenchmarkError."""

    def fn(*args):
        raise RuntimeError("autotune compile blew up")

    def gen(seed: int) -> tuple:
        return (seed,)

    with pytest.raises(BenchmarkError):
        _run(
            workloads=[_wl("wl-burn-fail")],
            generators=[gen],
            timer_sequence=[0.001],
            kernel_fn=fn,
            warmup=2,
            timed=2,
            discard_first=0,
        )


def test_key_tuple_for_resolves_axes():
    from src.eval.benchmark import _key_tuple_for

    wl = Workload.model_validate({
        "uuid": "wl-1",
        "axes": {"M": 4096, "N": 4096, "K": 4096},
        "inputs": {},
    })
    assert _key_tuple_for(wl, ["M", "N", "K"]) == (4096, 4096, 4096)


def test_key_tuple_for_unresolved_key_returns_none():
    """When a key isn't in workload.axes, the helper degrades to None.
    The orchestrator treats that as 'autotune_winner unavailable for this
    workload' and continues."""
    from src.eval.benchmark import _key_tuple_for

    wl = Workload.model_validate({
        "uuid": "wl-1",
        "axes": {"M": 4096},
        "inputs": {},
    })
    assert _key_tuple_for(wl, ["M", "X"]) is None


def test_key_tuple_for_empty_keys_returns_empty_tuple():
    """Edge case: kernel with no autotune_keys (e.g. legacy single-config
    starter) → empty tuple, not None. Cache lookup with () is a valid
    Triton autotune key when key=[] is supplied (which we forbid for
    Coder kernels, but starters can have it)."""
    from src.eval.benchmark import _key_tuple_for

    wl = Workload.model_validate({
        "uuid": "wl-1",
        "axes": {},
        "inputs": {},
    })
    assert _key_tuple_for(wl, []) == ()


def test_key_tuple_for_resolves_against_definition_const_axes():
    """Codex review 2026-05-14 finding #2: SOL problems split axes into
    const_axes (carried on Definition, invariant across workloads) and
    var_axes (carried on each Workload). The Coder's autotune key= list
    legitimately spans both — e.g. ``key=["B","M","N"]`` where B is var
    and M/N are const. Without consulting Definition, _key_tuple_for
    returns None for M/N and the winner never records.
    """
    from sol_execbench.core.data import Definition
    from src.eval.benchmark import _key_tuple_for

    definition = Definition.model_validate({
        "name": "matmul-fixed",
        "op_type": "matmul",
        "axes": {
            "M": {"type": "const", "value": 4096},
            "N": {"type": "const", "value": 4096},
        },
        "inputs": {},
        "outputs": {},
        "reference": "def run(): return 0",
    })
    # Workload only carries the var axis (B).
    wl = Workload.model_validate({
        "uuid": "wl-batch-2",
        "axes": {"B": 2},
        "inputs": {},
    })
    # Key list spans var + const. Should resolve all three using
    # workload.axes first then Definition.axes consts.
    assert _key_tuple_for(wl, ["B", "M", "N"], definition=definition) == (2, 4096, 4096)


def test_key_tuple_for_workload_axes_take_precedence_over_definition_consts():
    """If a workload re-binds an axis declared as const on the Definition,
    the workload value wins (workload is the immediate axis-resolution
    context per SOL's runtime contract)."""
    from sol_execbench.core.data import Definition
    from src.eval.benchmark import _key_tuple_for

    definition = Definition.model_validate({
        "name": "fake",
        "op_type": "matmul",
        "axes": {"M": {"type": "const", "value": 4096}},
        "inputs": {},
        "outputs": {},
        "reference": "def run(): return 0",
    })
    wl = Workload.model_validate({
        "uuid": "wl-override",
        "axes": {"M": 8192},
        "inputs": {},
    })
    assert _key_tuple_for(wl, ["M"], definition=definition) == (8192,)


def test_key_tuple_for_unresolved_after_definition_lookup_returns_none():
    """If a key is in neither workload.axes nor Definition.const_axes
    (e.g. it's a var axis the workload forgot to bind, or an AxisExpr),
    return None to degrade autotune_winner cleanly."""
    from sol_execbench.core.data import Definition
    from src.eval.benchmark import _key_tuple_for

    definition = Definition.model_validate({
        "name": "fake",
        "op_type": "matmul",
        "axes": {"M": {"type": "const", "value": 4096}},
        "inputs": {},
        "outputs": {},
        "reference": "def run(): return 0",
    })
    wl = Workload.model_validate({
        "uuid": "wl-missing-N",
        "axes": {},
        "inputs": {},
    })
    # N is neither in workload nor in definition.axes — degrade.
    assert _key_tuple_for(wl, ["M", "N"], definition=definition) is None
