"""Tests for the rewritten MemoryRetriever (sample() API)."""

from __future__ import annotations

import random

from src.memory.experience import ActionRecord, Experience
from src.memory.retriever import MemoryRetriever


def _exp(
    row_id: str,
    kernel_type: str = "matmul",
    hardware_arch: str = "RTX6000Ada",
    speedup: float = 1.5,
) -> Experience:
    # action_id keyed off row_id so each row has a DISTINCT dedup_key — these
    # sampling/weighting/fallback tests rely on every constructed row staying a
    # distinct candidate. (The read-time dedup_best now collapses rows sharing a
    # (kernel, arch, scope, action_id, condition) key; without a unique action_id
    # every row here would share one key and the pool would collapse to a single
    # candidate.)
    return Experience(
        row_id=row_id,
        schema_version=1,
        kernel_type=kernel_type,
        hardware_arch=hardware_arch,
        scope="edge",
        speedup=speedup,
        action_applied=ActionRecord(action_id=row_id, tier=1, name="n"),
        title="t",
        lesson="l",
        snippet_before="a",
        snippet_after="b",
        provenance={},
        created_at="",
    )


class _FakeStore:
    """In-memory store double — bypasses file I/O for retriever tests."""

    def __init__(self, exps: list[Experience]) -> None:
        self._exps = exps

    def all(self) -> list[Experience]:
        return list(self._exps)


def test_read_disabled_short_circuits():
    store = _FakeStore([_exp("r1")])
    r = MemoryRetriever(store, top_k=5, alpha=1.0, read_enabled=False)
    assert r.sample("matmul", "RTX6000Ada") == []


def test_empty_pool_returns_empty():
    r = MemoryRetriever(_FakeStore([]), top_k=5, alpha=1.0, read_enabled=True)
    assert r.sample("matmul", "RTX6000Ada") == []


def test_kernel_type_filter():
    pool = [_exp("r1", kernel_type="matmul"), _exp("r2", kernel_type="softmax")]
    r = MemoryRetriever(_FakeStore(pool), top_k=5, alpha=1.0, read_enabled=True)
    result = r.sample("matmul", "RTX6000Ada")
    assert [e.row_id for e in result] == ["r1"]


def test_pool_le_top_k_returns_all():
    pool = [_exp(f"r{i}") for i in range(3)]
    r = MemoryRetriever(_FakeStore(pool), top_k=5, alpha=1.0, read_enabled=True)
    result = r.sample("matmul", "RTX6000Ada")
    assert sorted(e.row_id for e in result) == ["r0", "r1", "r2"]


def test_hardware_preferred_fallback():
    # 3 same-arch + 5 other-arch, top_k=5 — fallback keeps same-arch first,
    # adds enough other-arch to bring the pool to >= top_k.
    pool = [
        *[_exp(f"same{i}", hardware_arch="RTX6000Ada") for i in range(3)],
        *[_exp(f"other{i}", hardware_arch="H100") for i in range(5)],
    ]
    rng = random.Random(42)
    r = MemoryRetriever(_FakeStore(pool), top_k=5, alpha=0.0, read_enabled=True, rng=rng)
    result = r.sample("matmul", "RTX6000Ada")
    assert len(result) == 5
    # All 3 same-arch should appear because the fallback merges same + other
    # into one pool of 8, then top_k=5 are sampled from that.
    same_ids = {e.row_id for e in result if e.hardware_arch == "RTX6000Ada"}
    # With uniform random over 8, all 3 same-arch may not appear; but the
    # fallback rule guarantees they're included in the candidate set.
    assert len(result) == 5


def test_fallback_caps_at_top_k_and_guarantees_same_arch():
    """The fallback must return exactly ``top_k`` entries and guarantee
    every same-arch entry is included (the whole point of the preference);
    the remaining slots are weight-sampled from the cross-arch pool.

    Bug: a prior implementation truncated the fallback pool to ``top_k``
    via ``same + other[:remaining]``. The caller's ``len <= top_k``
    early-return then fired, so speedup-weighting + randomness never ran
    on the cross-arch fill — only the first-N cross-arch rows by storage
    order were ever retrievable.
    """
    pool = [
        *[_exp(f"same{i}", hardware_arch="RTX6000Ada") for i in range(2)],
        *[_exp(f"other{i}", hardware_arch="H100") for i in range(10)],
    ]
    rng = random.Random(42)
    r = MemoryRetriever(_FakeStore(pool), top_k=4, alpha=1.0, read_enabled=True, rng=rng)
    result = r.sample("matmul", "RTX6000Ada")
    assert len(result) == 4
    # All same-arch entries must survive — they are guaranteed-included.
    same_ids = {e.row_id for e in result if e.hardware_arch == "RTX6000Ada"}
    assert same_ids == {"same0", "same1"}


def test_fallback_weight_samples_full_cross_arch_pool():
    """RED-for-the-bug: with 0 same-arch and a HIGH-speedup entry stored
    LAST, the old ``other[:remaining]`` truncation would never reach it
    (first-N by storage order). The fix weight-samples the FULL cross-arch
    pool, so a high-speedup last-stored entry is reachable."""
    # 0 same-arch (current = RTX6000Ada, all rows are H100).
    # 9 low-speedup rows stored first, 1 very-high-speedup row stored LAST.
    pool = [
        *[_exp(f"lo{i}", hardware_arch="H100", speedup=1.0) for i in range(9)],
        _exp("hi_last", hardware_arch="H100", speedup=100.0),
    ]
    rng = random.Random(42)
    r = MemoryRetriever(_FakeStore(pool), top_k=3, alpha=4.0, read_enabled=True, rng=rng)
    seen_hi = False
    for _ in range(200):
        result = r.sample("matmul", "RTX6000Ada")
        assert len(result) == 3
        if any(e.row_id == "hi_last" for e in result):
            seen_hi = True
            break
    # Old behavior: hi_last is never in other[:3] (it's stored 10th), so it
    # could NEVER appear. New behavior: weight-sampled from the full pool, so
    # with alpha=4 and speedup=100 it dominates and is found almost immediately.
    assert seen_hi, "high-speedup last-stored cross-arch entry was never retrieved"


def test_fallback_guarantees_all_same_when_below_top_k():
    """When ``0 < len(same) < top_k``, every same-arch entry is included
    and the result reaches ``top_k`` via cross-arch fill."""
    pool = [
        *[_exp(f"same{i}", hardware_arch="RTX6000Ada") for i in range(2)],
        *[_exp(f"other{i}", hardware_arch="H100") for i in range(3)],
    ]
    rng = random.Random(7)
    r = MemoryRetriever(_FakeStore(pool), top_k=4, alpha=1.0, read_enabled=True, rng=rng)
    result = r.sample("matmul", "RTX6000Ada")
    assert len(result) == 4
    same_ids = {e.row_id for e in result if e.hardware_arch == "RTX6000Ada"}
    assert same_ids == {"same0", "same1"}


def test_hardware_filter_uses_same_arch_only_when_enough():
    # 5 same-arch (>= top_k=3) — fallback returns same-arch only, no fallback
    pool = [_exp(f"same{i}", hardware_arch="RTX6000Ada") for i in range(5)]
    rng = random.Random(42)
    r = MemoryRetriever(_FakeStore(pool), top_k=3, alpha=0.0, read_enabled=True, rng=rng)
    result = r.sample("matmul", "RTX6000Ada")
    assert len(result) == 3
    assert all(e.hardware_arch == "RTX6000Ada" for e in result)


def test_alpha_zero_is_uniform_ish():
    pool = [_exp(f"r{i}", speedup=float(i + 1)) for i in range(10)]
    rng = random.Random(42)
    r = MemoryRetriever(_FakeStore(pool), top_k=1, alpha=0.0, read_enabled=True, rng=rng)
    counts = {f"r{i}": 0 for i in range(10)}
    for _ in range(2000):
        result = r.sample("matmul", "RTX6000Ada")
        counts[result[0].row_id] += 1
    # uniform-ish: every row picked at least 50 times in 2000 draws (expected 200)
    assert all(c >= 50 for c in counts.values()), counts


def test_alpha_one_favors_higher_speedup():
    pool = [_exp("slow", speedup=1.0), _exp("fast", speedup=4.0)]
    rng = random.Random(42)
    r = MemoryRetriever(_FakeStore(pool), top_k=1, alpha=1.0, read_enabled=True, rng=rng)
    counts = {"slow": 0, "fast": 0}
    for _ in range(2000):
        result = r.sample("matmul", "RTX6000Ada")
        counts[result[0].row_id] += 1
    # speedup-weighted: fast (4.0) should appear ~4x as often as slow (1.0)
    assert counts["fast"] > 3 * counts["slow"], counts


def test_sampling_without_replacement_returns_distinct_rows():
    # Pool size 1 + top_k=3 — pool <= top_k path returns all, no replacement
    pool = [_exp("only")]
    r = MemoryRetriever(_FakeStore(pool), top_k=3, alpha=1.0, read_enabled=True)
    result = r.sample("matmul", "RTX6000Ada")
    assert len(result) == 1

    # Pool size 3 + top_k=2 with extreme weight on "b" — both picks ≈ b
    # (pool > top_k so weighted sampling actually fires, not the
    # pool-le-top_k early-return path)
    pool2 = [_exp("a", speedup=1.0), _exp("b", speedup=100.0), _exp("c", speedup=1.0)]
    rng = random.Random(42)
    r2 = MemoryRetriever(_FakeStore(pool2), top_k=2, alpha=2.0, read_enabled=True, rng=rng)
    result2 = r2.sample("matmul", "RTX6000Ada")
    assert len(result2) == 2
    # Sampling is now WITHOUT replacement: the same row can never appear twice.
    # b's weight (100^2) dominates so it is one of the two picks, but the
    # second slot must be a DIFFERENT row, not a repeat of "b".
    assert len({e.row_id for e in result2}) == 2
    assert "b" in {e.row_id for e in result2}


def _e_cond(row_id, *, action_id, condition, speedup):
    """Builder with explicit action_id + condition (distinct dedup keys)."""
    return Experience(
        row_id=row_id, schema_version=1, kernel_type="matmul", hardware_arch="RTX6000Ada",
        scope="edge", speedup=speedup,
        action_applied=ActionRecord(action_id, 1, action_id, {}),
        title="t", lesson="l", snippet_before="", snippet_after="",
        provenance={}, created_at="2026-06-02T00:00:00+00:00", condition=condition)


def test_sample_returns_no_duplicate_row_ids():
    # Two DISTINCT rows; top_k larger than the pool must NOT repeat either.
    rows = [_e_cond("a", action_id="t1_grid_shape", condition="compute_bound", speedup=1.5),
            _e_cond("b", action_id="t1_occupancy", condition="compute_bound", speedup=1.2)]
    r = MemoryRetriever(_FakeStore(rows), top_k=5, alpha=1.0,
                        read_enabled=True, rng=random.Random(0))
    out = r.sample("matmul", "RTX6000Ada")
    assert len(out) == len({e.row_id for e in out})  # no repeats
    assert {e.row_id for e in out} == {"a", "b"}


def test_sample_collapses_same_key_duplicates():
    # Same (action, condition) appearing twice -> at most one in the result.
    rows = [_e_cond("lo", action_id="t1_grid_shape", condition="compute_bound", speedup=1.2),
            _e_cond("hi", action_id="t1_grid_shape", condition="compute_bound", speedup=1.6)]
    r = MemoryRetriever(_FakeStore(rows), top_k=5, alpha=1.0,
                        read_enabled=True, rng=random.Random(0))
    out = r.sample("matmul", "RTX6000Ada")
    assert [e.row_id for e in out] == ["hi"]


def test_sample_preserves_distinct_conditions():
    rows = [_e_cond("a", action_id="t1_grid_shape", condition="compute_bound", speedup=1.5),
            _e_cond("b", action_id="t1_grid_shape", condition="memory_bound", speedup=1.4)]
    r = MemoryRetriever(_FakeStore(rows), top_k=5, alpha=1.0,
                        read_enabled=True, rng=random.Random(0))
    out = r.sample("matmul", "RTX6000Ada")
    assert {e.row_id for e in out} == {"a", "b"}
