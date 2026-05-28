"""Tier 1 mocked tests for src/eval/smem_check.py.

Covers helper logic without GPU: violation detection, ordering, fail-open
on warmup failures, duck-typed metadata reads. The Tier 2 GPU counterpart
lives in tests/test_smem_check_gpu.py.
"""

from src.eval.smem_check import (
    SMEMViolation,
    _latest_cache_entry,
    _read_compiled_smem,
    check_autotune_smem_budget,
)


class _FakeCompiledKernel:
    """Stand-in for triton.compiler.CompiledKernel — exposes .metadata.shared."""

    def __init__(self, shared_bytes):
        class _Meta:
            shared = shared_bytes

        self.metadata = _Meta()


class _FakeConfig:
    def __init__(self, kwargs, num_warps=4, num_stages=3):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages


class _FakeJITFunction:
    """Stand-in for triton.JITFunction. cache[device][hash] = CompiledKernel."""

    def __init__(self):
        self.cache = {0: {}}


class _FakeAutotuner:
    """Stand-in for Triton's Autotuner with a ``run`` class method.

    The recorder in ``_capture_jit_args_via_host_wrapper`` swaps
    ``self.run`` via ``autotuner.run = ...`` (instance dict) and restores
    it via ``del autotuner.run``, which uncovers the class method. The
    body returns ``None`` — production host wrappers tolerate that since
    the recorder runs a no-launch capture, not a real autotune.
    """

    def __init__(self, configs, fn):
        self.configs = configs
        self.fn = fn

    def run(self, *args, **kwargs):  # noqa: D401 — class method swapped by recorder
        return None


def _make_autotuner(config_smem_bytes):
    """``config_smem_bytes`` is list of (kwargs, smem_value) tuples.

    Builds a fake autotuner where ``fn.warmup`` pushes a fresh
    ``_FakeCompiledKernel`` with the matching smem into the cache.
    """
    fn = _FakeJITFunction()
    configs = [_FakeConfig(kw) for kw, _ in config_smem_bytes]
    smems = iter([smem for _, smem in config_smem_bytes])

    def _warmup(*args, **kwargs):
        h = len(fn.cache[0])
        fn.cache[0][h] = _FakeCompiledKernel(next(smems))

    fn.warmup = _warmup
    return _FakeAutotuner(configs, fn)


def _make_stub_host_wrapper(autotuner):
    """Minimal host wrapper that drives ``autotuner.run`` once with a single
    positional arg, mirroring the production recorder-path shape. Tests
    that don't care about the captured args use this to satisfy the new
    required ``host_wrapper_fn`` contract on ``check_autotune_smem_budget``.
    """

    def host_wrapper(*sample_args):
        autotuner.run(*sample_args, grid=(1,))

    return host_wrapper


def test_no_violations_returns_empty():
    autotuner = _make_autotuner([
        ({"BLOCK_M": 64}, 50000),
        ({"BLOCK_M": 128}, 80000),
    ])
    violations = check_autotune_smem_budget(
        autotuner, _make_stub_host_wrapper(autotuner),
        sample_args=(0,), cap_bytes=101376,
    )
    assert violations == []


def test_single_violation_detected():
    autotuner = _make_autotuner([
        ({"BLOCK_M": 64}, 50000),
        ({"BLOCK_M": 256}, 262144),  # overflow
        ({"BLOCK_M": 128}, 80000),
    ])
    violations = check_autotune_smem_budget(
        autotuner, _make_stub_host_wrapper(autotuner),
        sample_args=(0,), cap_bytes=101376,
    )
    assert len(violations) == 1
    assert violations[0].config_idx == 1
    assert violations[0].footprint == 262144


def test_multi_violation_detected_in_order():
    autotuner = _make_autotuner([
        ({"BLOCK_M": 256}, 262144),
        ({"BLOCK_M": 64}, 50000),
        ({"BLOCK_M": 192}, 196608),
        ({"BLOCK_M": 128}, 80000),
        ({"BLOCK_M": 384}, 393216),
    ])
    violations = check_autotune_smem_budget(
        autotuner, _make_stub_host_wrapper(autotuner),
        sample_args=(0,), cap_bytes=101376,
    )
    assert [v.config_idx for v in violations] == [0, 2, 4]


def test_warmup_failure_skips_config_and_continues():
    """Per-Config warmup failure is caught — that Config is skipped (no false
    positive), other Configs continue to be inspected."""
    fn = _FakeJITFunction()
    configs = [
        _FakeConfig({"BLOCK_M": 64}),
        _FakeConfig({"BLOCK_M": 128}),
        _FakeConfig({"BLOCK_M": 256}),
    ]
    call_count = {"n": 0}

    def _warmup(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated ptxas error")
        h = len(fn.cache[0])
        fn.cache[0][h] = _FakeCompiledKernel(50000)  # under cap

    fn.warmup = _warmup
    autotuner = _FakeAutotuner(configs, fn)
    violations = check_autotune_smem_budget(
        autotuner, _make_stub_host_wrapper(autotuner),
        sample_args=(0,), cap_bytes=101376,
    )
    assert violations == []  # all surviving configs under cap; config 1 skipped


def test_smem_metadata_duck_typed_legacy_shared_attr():
    """Older Triton exposes ``.shared`` instead of ``.metadata.shared``."""

    class _LegacyCompiled:
        shared = 50000

    assert _read_compiled_smem(_LegacyCompiled()) == 50000


def test_smem_metadata_missing_returns_none():
    """Neither attribute → ``_read_compiled_smem`` returns None (validator skips)."""

    class _NoSmem:
        pass

    assert _read_compiled_smem(_NoSmem()) is None


def test_latest_cache_entry_empty_returns_none():
    class _EmptyJit:
        cache = {}

    assert _latest_cache_entry(_EmptyJit()) is None


# ── Recorder-patch tests (Phase B redesign 2026-05-25) ────────────────


def test_capture_jit_args_records_first_kernel_call():
    """Recorder captures the args the host wrapper passes to autotuner.run."""
    from src.eval.smem_check import _capture_jit_args_via_host_wrapper

    fn = _FakeJITFunction()
    autotuner = _FakeAutotuner(configs=[_FakeConfig({"BLOCK_M": 32})], fn=fn)

    def host_wrapper(a, b):
        # Simulates a real matmul host wrapper computing extra args inline.
        c, M, N, K = "c-tensor", 32, 32, 16
        autotuner.run(a, b, c, M, N, K, grid=(1,))

    captured = _capture_jit_args_via_host_wrapper(
        autotuner, host_wrapper, sample_args=("a-tensor", "b-tensor")
    )
    args, kwargs = captured
    assert args == ("a-tensor", "b-tensor", "c-tensor", 32, 32, 16)
    # ``grid`` is stripped at replay time by the SMEM check (kwarg shape
    # depends on how the host wrapper passed it); we still capture it
    # honestly so the caller can decide what to drop.
    assert kwargs == {"grid": (1,)}


def test_capture_jit_args_records_only_first_of_multi_call():
    """Multi-call host wrappers (split-K): only first invocation captured."""
    from src.eval.smem_check import _capture_jit_args_via_host_wrapper

    fn = _FakeJITFunction()
    autotuner = _FakeAutotuner(configs=[_FakeConfig({"BLOCK_M": 32})], fn=fn)

    def host_wrapper():
        autotuner.run("first", grid=(1,))
        autotuner.run("second", grid=(1,))

    captured = _capture_jit_args_via_host_wrapper(autotuner, host_wrapper, sample_args=())
    args, kwargs = captured
    assert args == ("first",)
    assert kwargs == {"grid": (1,)}


def test_capture_jit_args_restores_autotuner_run_on_success():
    """After a clean drive, autotuner.__dict__['run'] is gone (class method restored)."""
    from src.eval.smem_check import _capture_jit_args_via_host_wrapper

    fn = _FakeJITFunction()
    autotuner = _FakeAutotuner(configs=[_FakeConfig({"BLOCK_M": 32})], fn=fn)

    def host_wrapper():
        autotuner.run("x", grid=(1,))

    _capture_jit_args_via_host_wrapper(autotuner, host_wrapper, sample_args=())
    assert "run" not in autotuner.__dict__


def test_capture_jit_args_restores_autotuner_run_on_exception(monkeypatch):
    """Host wrapper raises mid-call → restore still happens; host_wrapper_failed emitted."""
    from src.eval.smem_check import _capture_jit_args_via_host_wrapper

    fn = _FakeJITFunction()
    autotuner = _FakeAutotuner(configs=[_FakeConfig({"BLOCK_M": 32})], fn=fn)
    captured_events: list[tuple] = []
    monkeypatch.setattr(
        "src.eval.smem_check.events.emit",
        lambda kind, **kw: captured_events.append((kind, kw)),
    )

    def host_wrapper():
        raise RuntimeError("simulated wrapper bug")

    result = _capture_jit_args_via_host_wrapper(autotuner, host_wrapper, sample_args=())
    assert result is None
    assert "run" not in autotuner.__dict__
    assert any(
        k == "smem_check_skipped" and kw.get("reason") == "host_wrapper_failed"
        for k, kw in captured_events
    )


def test_capture_jit_args_returns_captured_on_post_capture_crash(monkeypatch):
    """Multi-call host wrapper that succeeds on call 1 then crashes on call 2:
    per the docstring contract that "only the first recorded invocation is
    used", the first capture is honored and a softer
    ``host_wrapper_crashed_after_capture`` event is emitted. The earlier
    behavior of returning ``None`` and emitting ``host_wrapper_failed`` would
    have discarded a perfectly good capture.
    """
    from src.eval.smem_check import _capture_jit_args_via_host_wrapper

    fn = _FakeJITFunction()
    autotuner = _FakeAutotuner(configs=[_FakeConfig({"BLOCK_M": 32})], fn=fn)
    captured_events: list[tuple] = []
    monkeypatch.setattr(
        "src.eval.smem_check.events.emit",
        lambda kind, **kw: captured_events.append((kind, kw)),
    )

    def host_wrapper():
        autotuner.run("first", "second-pos", k1="v1", grid=(1,))
        raise RuntimeError("simulated crash on step 2")

    result = _capture_jit_args_via_host_wrapper(autotuner, host_wrapper, sample_args=())
    assert result is not None
    args, kwargs = result
    assert args == ("first", "second-pos")
    assert kwargs == {"k1": "v1", "grid": (1,)}
    # Soft telemetry, not the hard host_wrapper_failed.
    assert any(
        k == "smem_check_skipped"
        and kw.get("reason") == "host_wrapper_crashed_after_capture"
        for k, kw in captured_events
    )
    assert not any(
        kw.get("reason") == "host_wrapper_failed" for _, kw in captured_events
    )
    # Run still restored.
    assert "run" not in autotuner.__dict__


def test_capture_jit_args_returns_none_on_recorder_no_capture(monkeypatch):
    """Host wrapper completes without invoking the autotuner → recorder_no_capture."""
    from src.eval.smem_check import _capture_jit_args_via_host_wrapper

    fn = _FakeJITFunction()
    autotuner = _FakeAutotuner(configs=[_FakeConfig({"BLOCK_M": 32})], fn=fn)
    captured_events: list[tuple] = []
    monkeypatch.setattr(
        "src.eval.smem_check.events.emit",
        lambda kind, **kw: captured_events.append((kind, kw)),
    )

    def host_wrapper():
        pass  # never calls autotuner.run

    result = _capture_jit_args_via_host_wrapper(autotuner, host_wrapper, sample_args=())
    assert result is None
    assert any(
        k == "smem_check_skipped" and kw.get("reason") == "recorder_no_capture"
        for k, kw in captured_events
    )


def test_check_autotune_smem_budget_with_host_wrapper_drives_then_warmups():
    """End-to-end recorder path: host wrapper captures args, then warmup per Config
    populates the cache, then SMEM is read per Config."""
    fn = _FakeJITFunction()
    configs = [
        _FakeConfig({"BLOCK_M": 32}),
        _FakeConfig({"BLOCK_M": 256}),  # overcommit
        _FakeConfig({"BLOCK_M": 64}),
    ]
    smems = iter([50000, 262144, 80000])

    def _warmup(*args, **kwargs):
        h = len(fn.cache[0])
        fn.cache[0][h] = _FakeCompiledKernel(next(smems))

    fn.warmup = _warmup
    autotuner = _FakeAutotuner(configs=configs, fn=fn)

    def host_wrapper(a, b):
        autotuner.run(a, b, "c", 32, 32, 16, grid=(1,))

    violations = check_autotune_smem_budget(
        autotuner, host_wrapper, sample_args=("a", "b"), cap_bytes=101376
    )
    assert len(violations) == 1
    assert violations[0].config_idx == 1
    assert violations[0].footprint == 262144


def test_check_autotune_smem_budget_emits_skip_when_cfg_overrides_recorded_kwarg(monkeypatch):
    """The host wrapper passes ``BLOCK_M=64`` explicitly (shape-dependent
    constexpr derived from runtime input dims) AND the autotuner has a
    Config with ``BLOCK_M=128``. Blindly preferring cfg.kwargs would let
    the measured SMEM diverge from production. Fix: detect the conflict,
    emit ``cfg_overrides_recorded_kwarg`` telemetry, skip the Config.
    """
    fn = _FakeJITFunction()
    warmup_calls: list[tuple] = []

    def _warmup(*args, **kwargs):
        warmup_calls.append((args, dict(kwargs)))
        h = len(fn.cache[0])
        fn.cache[0][h] = _FakeCompiledKernel(50000)

    fn.warmup = _warmup
    configs = [_FakeConfig({"BLOCK_M": 128})]  # autotune says 128
    autotuner = _FakeAutotuner(configs=configs, fn=fn)
    captured_events: list[tuple] = []
    monkeypatch.setattr(
        "src.eval.smem_check.events.emit",
        lambda kind, **kw: captured_events.append((kind, kw)),
    )

    def host_wrapper(a):
        # Host wrapper explicitly passes BLOCK_M=64 (runtime-derived).
        autotuner.run(a, BLOCK_M=64, grid=(1,))

    violations = check_autotune_smem_budget(
        autotuner, host_wrapper, sample_args=("a",), cap_bytes=101376,
    )
    # Config skipped — never measured.
    assert violations == []
    assert warmup_calls == []
    # Telemetry emitted with the right shape.
    assert any(
        k == "smem_check_skipped"
        and kw.get("reason") == "cfg_overrides_recorded_kwarg"
        and kw.get("config_idx") == 0
        and kw.get("key") == "BLOCK_M"
        for k, kw in captured_events
    )


def test_check_autotune_smem_budget_threads_iter_no_into_events(monkeypatch):
    """``iter_no=42`` (or any int) is threaded into every ``events.emit``
    call as the ``iter=`` kwarg so per-iter event correlation in
    ``events.jsonl`` works. Without it, the ``iter`` field on every
    skip event would be ``null``.
    """
    fn = _FakeJITFunction()
    configs = [_FakeConfig({"BLOCK_M": 64})]
    autotuner = _FakeAutotuner(configs=configs, fn=fn)

    def _warmup(*args, **kwargs):
        raise RuntimeError("force warmup_failed path")

    fn.warmup = _warmup
    captured_events: list[tuple] = []
    monkeypatch.setattr(
        "src.eval.smem_check.events.emit",
        lambda kind, *, iter=None, **kw: captured_events.append((kind, iter, kw)),
    )

    def host_wrapper(a):
        autotuner.run(a, grid=(1,))

    violations = check_autotune_smem_budget(
        autotuner, host_wrapper, sample_args=("a",), cap_bytes=101376, iter_no=42,
    )
    assert violations == []
    # Every emit observed must carry iter=42.
    assert captured_events, "expected at least one events.emit call"
    for kind, it, _kw in captured_events:
        assert it == 42, f"event {kind} missing iter=42 (got {it!r})"


def test_warmup_failed_event_carries_exception_traceback(monkeypatch):
    """``smem_check_skipped(reason='warmup_failed')`` must include
    ``exc_class`` and ``exc_traceback`` (formatted via ``traceback.format_exception``)
    so postmortems can distinguish the failure modes the bare
    ``except Exception:`` would otherwise mask: signature mismatch
    (TypeError), ptxas crash, OOM, etc. ``exc_msg`` (the legacy
    160-char head-slice) is replaced by the full-traceback tail-slice
    because Triton's CompilationError annotates the source line at the
    END of the chain — head-slicing cut it off (run_20260525T080053).
    """
    fn = _FakeJITFunction()
    configs = [_FakeConfig({"BLOCK_M": 64})]
    autotuner = _FakeAutotuner(configs=configs, fn=fn)

    def _warmup(*args, **kwargs):
        raise TypeError("missing required argument 'NUM_PRODUCER_WARPS'")

    fn.warmup = _warmup
    captured: list[dict] = []
    monkeypatch.setattr(
        "src.eval.smem_check.events.emit",
        lambda kind, *, iter=None, **kw: captured.append(
            {"kind": kind, "iter": iter, **kw}
        ),
    )

    def host_wrapper(a):
        autotuner.run(a, grid=(1,))

    violations = check_autotune_smem_budget(
        autotuner, host_wrapper, sample_args=("a",), cap_bytes=101376,
    )
    assert violations == []
    warmup_events = [e for e in captured if e.get("reason") == "warmup_failed"]
    assert len(warmup_events) == 1, captured
    ev = warmup_events[0]
    assert ev["exc_class"] == "TypeError"
    assert "exc_traceback" in ev
    assert "exc_msg" not in ev  # negative: legacy key is gone
    assert "TypeError" in ev["exc_traceback"]
    assert "NUM_PRODUCER_WARPS" in ev["exc_traceback"]
    assert ev["config_idx"] == 0


def test_warmup_failed_exc_traceback_capped_at_2048_chars():
    """Long Triton tracebacks can run into KB. ``exc_traceback`` caps at
    2048 chars (tail-sliced — see test_warmup_failed_exc_traceback_keeps_tail)
    so events.jsonl doesn't bloat under the worst case where every Config
    in K-way fan-out fails."""
    fn = _FakeJITFunction()
    configs = [_FakeConfig({"BLOCK_M": 64})]
    autotuner = _FakeAutotuner(configs=configs, fn=fn)
    long_msg = "x" * 5000

    def _warmup(*args, **kwargs):
        raise RuntimeError(long_msg)

    fn.warmup = _warmup
    captured: list[dict] = []
    import src.eval.smem_check as _smem
    orig_emit = _smem.events.emit
    _smem.events.emit = lambda kind, *, iter=None, **kw: captured.append(
        {"kind": kind, "iter": iter, **kw}
    )
    try:
        def host_wrapper(a):
            autotuner.run(a, grid=(1,))

        check_autotune_smem_budget(
            autotuner, host_wrapper, sample_args=("a",), cap_bytes=101376,
        )
    finally:
        _smem.events.emit = orig_emit
    warmup_events = [e for e in captured if e.get("reason") == "warmup_failed"]
    assert len(warmup_events) == 1
    assert len(warmup_events[0]["exc_traceback"]) <= 2048


def test_warmup_failed_exc_traceback_keeps_tail(monkeypatch):
    """Tail-slice preservation — Triton's CompilationError annotates the
    failing source line at the END of the exception chain (the
    ``at <line>:<col>:`` fragment). Head-slicing the 160-char cap was
    cutting that annotation; the new 2KB tail-slice must keep it.
    """
    fn = _FakeJITFunction()
    configs = [_FakeConfig({"BLOCK_M": 64})]
    autotuner = _FakeAutotuner(configs=configs, fn=fn)
    tail_marker = "AT_LINE_57_COL_45_TRITON_ANNOTATION"
    # Pad with leading X's so the head of the rendered traceback is
    # noise; the tail-slice must still surface the marker.
    long_msg = ("X" * 5000) + "\n" + tail_marker

    def _warmup(*args, **kwargs):
        raise RuntimeError(long_msg)

    fn.warmup = _warmup
    captured: list[dict] = []
    monkeypatch.setattr(
        "src.eval.smem_check.events.emit",
        lambda kind, *, iter=None, **kw: captured.append(
            {"kind": kind, "iter": iter, **kw}
        ),
    )

    def host_wrapper(a):
        autotuner.run(a, grid=(1,))

    check_autotune_smem_budget(
        autotuner, host_wrapper, sample_args=("a",), cap_bytes=101376,
    )
    warmup_events = [e for e in captured if e.get("reason") == "warmup_failed"]
    assert len(warmup_events) == 1
    tb = warmup_events[0]["exc_traceback"]
    assert tail_marker in tb, (
        f"tail marker missing from exc_traceback — head-slice regression? "
        f"tb head: {tb[:120]!r} tb tail: {tb[-120:]!r}"
    )


def test_check_autotune_smem_budget_replays_recorded_kwargs():
    """Host wrappers that pass shape/stride args via keyword: recorder MUST
    capture kwargs AND replay them in warmup. Without this, warmup raises
    on signature mismatch, helper emits warmup_failed for every Config,
    violations come back empty, and the SMEM check silently fails open
    on the production matmul shape.

    Regression test for the Codex finding (2026-05-25): the recorder
    captured only positional args, dropping any kwarg launches.
    """
    fn = _FakeJITFunction()
    received_warmup_calls: list[tuple[tuple, dict]] = []

    def _warmup(*args, **kwargs):
        received_warmup_calls.append((args, dict(kwargs)))
        h = len(fn.cache[0])
        fn.cache[0][h] = _FakeCompiledKernel(262144)  # overflow → violation

    fn.warmup = _warmup
    configs = [_FakeConfig({"BLOCK_M": 64, "BLOCK_N": 64})]
    autotuner = _FakeAutotuner(configs=configs, fn=fn)

    def host_wrapper(a, b):
        # Real-world matmul pattern: positional ptrs + keyword shape/stride.
        autotuner.run(
            a, b, "c-tensor",
            M=32, N=32, K=16,
            stride_am=1, stride_ak=32,
            grid=(1,),
        )

    violations = check_autotune_smem_budget(
        autotuner, host_wrapper,
        sample_args=("a-tensor", "b-tensor"),
        cap_bytes=101376,
    )
    # Violation detected (helper didn't fail open on the kwarg signature).
    assert len(violations) == 1
    assert violations[0].footprint == 262144

    # Exactly one warmup call fired (we have one Config).
    assert len(received_warmup_calls) == 1
    args, kwargs = received_warmup_calls[0]

    # Positional args from the host wrapper preserved.
    assert args == ("a-tensor", "b-tensor", "c-tensor")

    # Keyword args from the host wrapper preserved.
    assert kwargs.get("M") == 32
    assert kwargs.get("N") == 32
    assert kwargs.get("K") == 16
    assert kwargs.get("stride_am") == 1
    assert kwargs.get("stride_ak") == 32

    # cfg.kwargs (autotune constexprs) merged in.
    assert kwargs.get("BLOCK_M") == 64
    assert kwargs.get("BLOCK_N") == 64

    # Framework kwargs from cfg added by the SMEM check.
    assert kwargs.get("num_warps") == 4
    assert kwargs.get("num_stages") == 3
    # ``grid`` is overridden to (1,) for compile-only warmup, NOT the
    # host wrapper's grid (which is wrapper-shape-specific).
    assert kwargs.get("grid") == (1,)
    # ``warmup`` framework kwarg (if the host wrapper passed it) is stripped.
    assert "warmup" not in kwargs
