from __future__ import annotations

import threading

import pytest
import torch

from src.eval.anti_cheat import (
    AntiCheatContext,
    check_lazy_outputs_after_bench,
    per_iter_anti_cheat,
    strict_tolerance_check,
)
from sol_execbench.core.bench.reward_hack import RewardHackDetected


# Names that exist on torch.cuda.Event so snapshot picks up real ids.
CRITICAL_NAMES = ["elapsed_time", "synchronize", "wait", "record"]


def test_per_iter_anti_cheat_clean_run():
    with per_iter_anti_cheat(CRITICAL_NAMES) as ctx:
        assert isinstance(ctx, AntiCheatContext)
        assert isinstance(ctx.snapshot, dict)
        assert ctx.threads_before >= 1


def test_per_iter_anti_cheat_detects_namespace_tampering():
    namespace = vars(torch.cuda.Event)
    saved = namespace.get(CRITICAL_NAMES[0])
    try:
        with pytest.raises(RewardHackDetected):
            with per_iter_anti_cheat(CRITICAL_NAMES):
                # Direct dict assignment on a class' __dict__ may be blocked
                # for builtin types — fall back to setattr if vars() is read-only.
                try:
                    namespace[CRITICAL_NAMES[0]] = lambda *a, **k: None
                except TypeError:
                    setattr(torch.cuda.Event, CRITICAL_NAMES[0], lambda *a, **k: None)
    finally:
        if saved is not None:
            try:
                setattr(torch.cuda.Event, CRITICAL_NAMES[0], saved)
            except Exception:
                pass


def test_per_iter_anti_cheat_detects_thread_injection():
    stop = threading.Event()
    spawned: list[threading.Thread] = []

    def _spinner() -> None:
        # Block until the test asks the thread to exit so the thread is still
        # alive when the context manager runs its post-yield
        # ``check_thread_injection`` call.
        stop.wait(timeout=10.0)

    try:
        with pytest.raises(RewardHackDetected):
            with per_iter_anti_cheat(CRITICAL_NAMES):
                t = threading.Thread(target=_spinner, daemon=True)
                t.start()
                spawned.append(t)
    finally:
        stop.set()
        for t in spawned:
            t.join(timeout=2.0)


def test_check_lazy_outputs_after_bench_clean():
    outputs = [torch.zeros(4)]
    check_lazy_outputs_after_bench(outputs)  # no raise


def test_check_lazy_outputs_after_bench_detects_non_tensor():
    class FakeProxy:
        pass

    with pytest.raises(RewardHackDetected):
        check_lazy_outputs_after_bench([FakeProxy()])


def test_strict_tolerance_check_passes_on_close_outputs():
    cand = torch.zeros(8)
    ref = torch.zeros(8)
    assert strict_tolerance_check(cand, ref, atol=1e-5, rtol=1e-4) is True


def test_strict_tolerance_check_fails_on_distant_outputs():
    cand = torch.zeros(8)
    ref = torch.ones(8) * 100.0
    assert strict_tolerance_check(cand, ref, atol=1e-5, rtol=1e-4) is False
