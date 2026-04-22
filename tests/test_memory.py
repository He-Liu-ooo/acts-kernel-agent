"""Tests for memory/ — experience storage, retrieval, and ranking."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.eval.types import BottleneckType
from src.memory.experience import ActionRecord, Experience
from src.memory.store import MemoryStore
from src.memory.retriever import MemoryRetriever


def _exp(
    kernel_type: str = "matmul",
    bottleneck_before: BottleneckType = BottleneckType.MEMORY_BOUND,
    success: bool = True,
    speedup: float = 1.5,
    hardware: str = "H100",
    action_id: str = "tile_sizes",
) -> Experience:
    """Shorthand factory for test experiences."""
    return Experience(
        kernel_type=kernel_type,
        action_applied=ActionRecord(action_id=action_id, tier=1, name=action_id),
        speedup=speedup,
        bottleneck_before=bottleneck_before,
        success=success,
        hardware=hardware,
    )


def _retriever(*experiences: Experience, top_k: int = 5) -> MemoryRetriever:
    """Build a retriever with pre-loaded experiences."""
    with TemporaryDirectory() as d:
        store = MemoryStore(Path(d) / "mem.json")
        for e in experiences:
            store.add(e)
        return MemoryRetriever(store, top_k=top_k)


# ── kernel-type filtering ────────────────────────────────────────────────


def test_filters_by_kernel_type():
    """Only experiences matching the requested kernel type are returned."""
    r = _retriever(
        _exp(kernel_type="matmul"),
        _exp(kernel_type="softmax"),
        _exp(kernel_type="matmul"),
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    assert len(results) == 2
    assert all(e.kernel_type == "matmul" for e in results)


# ── bottleneck ranking ───────────────────────────────────────────────────


def test_bottleneck_match_ranks_higher():
    """Experiences matching current bottleneck rank above non-matching."""
    r = _retriever(
        _exp(bottleneck_before=BottleneckType.COMPUTE_BOUND, speedup=3.0),
        _exp(bottleneck_before=BottleneckType.MEMORY_BOUND, speedup=1.1),
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    assert results[0].bottleneck_before == BottleneckType.MEMORY_BOUND


# ── success/failure ranking ──────────────────────────────────────────────


def test_success_ranks_above_failure_same_bottleneck():
    """Among same-bottleneck experiences, successes rank above failures."""
    r = _retriever(
        _exp(success=False, speedup=0.8),
        _exp(success=True, speedup=1.5),
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    assert results[0].success is True


# ── speedup tiebreaking ──────────────────────────────────────────────────


def test_higher_speedup_ranks_first_among_equals():
    """Among experiences with same bottleneck and success, higher speedup wins."""
    r = _retriever(
        _exp(speedup=1.2),
        _exp(speedup=2.5),
        _exp(speedup=1.8),
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    speedups = [e.speedup for e in results]
    assert speedups == sorted(speedups, reverse=True)


# ── reserved failure slots ───────────────────────────────────────────────


def test_failures_included_via_reserved_slots():
    """Failures are returned even when enough successes exist to fill top_k."""
    r = _retriever(
        _exp(success=True, speedup=2.0, action_id="a1"),
        _exp(success=True, speedup=1.8, action_id="a2"),
        _exp(success=True, speedup=1.5, action_id="a3"),
        _exp(success=True, speedup=1.3, action_id="a4"),
        _exp(success=False, speedup=0.5, action_id="f1"),
        _exp(success=False, speedup=0.3, action_id="f2"),
        top_k=5,
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    failures = [e for e in results if not e.success]
    assert len(failures) >= 1


def test_failure_slots_scale_with_top_k():
    """Failure slots = max(1, top_k // 3). For top_k=9, 3 failure slots."""
    successes = [_exp(success=True, speedup=2.0 + i * 0.1, action_id=f"s{i}") for i in range(7)]
    failures = [_exp(success=False, speedup=0.5 + i * 0.1, action_id=f"f{i}") for i in range(5)]
    r = _retriever(*successes, *failures, top_k=9)
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    fail_count = sum(1 for e in results if not e.success)
    assert fail_count >= 3


def test_top_k_1_returns_best_success_not_failure():
    """top_k=1 must not collapse to only failures."""
    r = _retriever(
        _exp(success=True, speedup=2.0, action_id="s1"),
        _exp(success=False, speedup=0.5, action_id="f1"),
        top_k=1,
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    assert len(results) == 1
    assert results[0].success is True


def test_top_k_2_includes_both_pools():
    """top_k=2 with both pools should return one success and one failure."""
    r = _retriever(
        _exp(success=True, speedup=2.0, action_id="s1"),
        _exp(success=True, speedup=1.5, action_id="s2"),
        _exp(success=False, speedup=0.5, action_id="f1"),
        top_k=2,
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    assert len(results) == 2
    successes = [e for e in results if e.success]
    failures = [e for e in results if not e.success]
    assert len(successes) >= 1
    assert len(failures) >= 1


# ── pool fallback ────────────────────────────────────────────────────────


def test_all_successes_when_no_failures_exist():
    """If no failures exist, all slots go to successes."""
    r = _retriever(
        _exp(success=True, speedup=2.0, action_id="a1"),
        _exp(success=True, speedup=1.5, action_id="a2"),
        _exp(success=True, speedup=1.2, action_id="a3"),
        top_k=5,
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    assert len(results) == 3
    assert all(e.success for e in results)


def test_all_failures_when_no_successes_exist():
    """If no successes exist, all slots go to failures."""
    r = _retriever(
        _exp(success=False, speedup=0.8, action_id="f1"),
        _exp(success=False, speedup=0.5, action_id="f2"),
        top_k=5,
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    assert len(results) == 2
    assert all(not e.success for e in results)


# ── top_k cap ────────────────────────────────────────────────────────────


def test_never_exceeds_top_k():
    """Result count is capped at top_k regardless of store size."""
    exps = [_exp(action_id=f"a{i}", speedup=1.0 + i * 0.1) for i in range(20)]
    r = _retriever(*exps, top_k=5)
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    assert len(results) <= 5


# ── empty store ──────────────────────────────────────────────────────────


def test_empty_store_returns_empty():
    """No experiences => empty result."""
    r = _retriever(top_k=5)
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    assert results == []


# ── hardware filtering ───────────────────────────────────────────────────


def test_prefers_same_hardware():
    """Same-hardware experiences are preferred when enough exist."""
    r = _retriever(
        _exp(hardware="H100", speedup=1.5, action_id="a1"),
        _exp(hardware="H100", speedup=1.3, action_id="a2"),
        _exp(hardware="A100", speedup=2.0, action_id="a3"),
        top_k=2,
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND, hardware="H100")
    assert all(e.hardware == "H100" for e in results)


def test_hardware_fallback_when_too_few():
    """Cross-hardware experiences fill remaining slots if same-hardware insufficient."""
    r = _retriever(
        _exp(hardware="H100", speedup=1.5, action_id="a1"),
        _exp(hardware="A100", speedup=2.0, action_id="a2"),
        top_k=3,
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND, hardware="H100")
    assert len(results) == 2
    hw = {e.hardware for e in results}
    assert hw == {"H100", "A100"}


def test_no_hardware_filter_returns_all():
    """Without hardware arg, experiences from all hardware are considered."""
    r = _retriever(
        _exp(hardware="H100", action_id="a1"),
        _exp(hardware="A100", action_id="a2"),
        top_k=5,
    )
    results = r.retrieve("matmul", BottleneckType.MEMORY_BOUND)
    assert len(results) == 2


# ── MemoryStore round-trip ───────────────────────────────────────────────


def test_store_save_load_round_trip():
    """Experiences survive save/load cycle."""
    with TemporaryDirectory() as d:
        path = Path(d) / "mem.json"
        store = MemoryStore(path)
        exp = _exp(kernel_type="softmax", speedup=1.7)
        store.add(exp)

        store2 = MemoryStore(path)
        store2.load()
        loaded = store2.all()
        assert len(loaded) == 1
        assert loaded[0].kernel_type == "softmax"
        assert loaded[0].speedup == 1.7


def test_store_empty_load():
    """Loading from a nonexistent file gives empty list."""
    with TemporaryDirectory() as d:
        store = MemoryStore(Path(d) / "nope.json")
        store.load()
        assert store.all() == []


# ── legacy / malformed bottleneck tolerance ──────────────────────────────


def _write_legacy_record(path: Path, bottleneck_before: str, bottleneck_after: str) -> None:
    """Dump one experience record with the given bottleneck strings.

    The schema matches what older ``MemoryStore.save`` versions produced:
    ``bottleneck_before`` is still live, while ``bottleneck_after`` is a
    dead key preserved on-disk from pre-classify-once records. The load
    path must tolerate both the empty-string default (pre-profiler-PR)
    and the stale ``bottleneck_after`` key.
    """
    path.write_text(json.dumps([{
        "kernel_type": "matmul",
        "action_applied": {
            "action_id": "tile_sizes", "tier": 1, "name": "tile_sizes",
        },
        "metrics": {},
        "speedup": 1.0,
        "reviewer_summary": "",
        "bottleneck_before": bottleneck_before,
        "bottleneck_after": bottleneck_after,
        "hardware": "H100",
        "success": True,
    }]))


def test_load_tolerates_legacy_empty_bottleneck_strings():
    """Older MemoryStore files persisted ``bottleneck_before`` as ``""``.
    Loading must not crash the store — fall back to ``BALANCED``."""
    with TemporaryDirectory() as d:
        path = Path(d) / "mem.json"
        _write_legacy_record(path, "", "")

        store = MemoryStore(path)
        store.load()
        loaded = store.all()

        assert len(loaded) == 1
        assert loaded[0].bottleneck_before is BottleneckType.BALANCED


def test_load_tolerates_unknown_bottleneck_value():
    """Values not in the enum (e.g. a schema change, a hand-edited file)
    must fall back to ``BALANCED`` per record rather than aborting load."""
    with TemporaryDirectory() as d:
        path = Path(d) / "mem.json"
        _write_legacy_record(path, "latency_bound", "compute_bound")

        store = MemoryStore(path)
        store.load()
        loaded = store.all()

        assert len(loaded) == 1
        assert loaded[0].bottleneck_before is BottleneckType.BALANCED


def test_load_ignores_legacy_bottleneck_after_key():
    """``bottleneck_after`` is a dead field removed from ``Experience``.
    Legacy JSON records that still carry the key must load cleanly: the
    extra key is silently dropped, and the surviving fields parse as usual.
    """
    with TemporaryDirectory() as d:
        path = Path(d) / "mem.json"
        _write_legacy_record(path, "compute_bound", "memory_bound")

        store = MemoryStore(path)
        store.load()
        loaded = store.all()

        assert len(loaded) == 1
        assert loaded[0].bottleneck_before is BottleneckType.COMPUTE_BOUND
        # Dead field must not leak back onto the Experience instance.
        assert not hasattr(loaded[0], "bottleneck_after")
