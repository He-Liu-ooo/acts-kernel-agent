"""Tests for Producer — G1 gating, cap eviction, G3 finalize, flush."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.experience import ActionRecord, Experience, _format_condition
from src.memory.producer import Producer
from src.memory.summarizer import SummarizerResult


@dataclass
class _FakeKernel:
    # Matches the real ``src/kernels/kernel.py:Kernel.source_code`` attribute
    # name. The opt-mem Producer reads ``node.kernel.source_code`` — using a
    # different attribute name here would mask the real bug Codex Adversarial
    # Round 3 caught (producer.py using ``.source`` against the wrong field).
    source_code: str = "def kernel(): pass"


@dataclass
class _FakeNode:
    id: str
    runtime_ms: float | None
    compiled: bool = True
    correct: bool = True
    kernel: _FakeKernel = field(default_factory=_FakeKernel)


class _FakeStore:
    def __init__(self) -> None:
        self.added: list[Experience] = []

    def add_many(self, exps) -> None:
        self.added.extend(list(exps))


class _FakeConfig:
    def __init__(self) -> None:
        self.opt_mem_write_enabled = True
        self.opt_mem_writes_per_session_cap = 20
        self.opt_mem_min_improvement_ratio = 1.05
        self.hardware = MagicMock()
        self.hardware.name = "RTX6000Ada"


def _ok_summary() -> SummarizerResult:
    return SummarizerResult(title="t", lesson="l", snippet_before="a", snippet_after="b")


def _producer(config, summarizer, store=None, run_id: str = "run-1") -> Producer:
    return Producer(
        store=store or _FakeStore(),
        summarizer=summarizer,
        config=config,
        run_id=run_id,
        kernel_type="matmul",
    )


def _action() -> ActionRecord:
    return ActionRecord(action_id="a", tier=1, name="n")


@pytest.mark.asyncio
async def test_write_disabled_short_circuits():
    cfg = _FakeConfig()
    cfg.opt_mem_write_enabled = False
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock()
    p = _producer(cfg, summarizer)
    await p.consider(_FakeNode("p", 10.0), _FakeNode("c", 5.0), _action())
    summarizer.summarize.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_not_compiled_skipped():
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock()
    p = _producer(cfg, summarizer)
    await p.consider(
        _FakeNode("p", 10.0),
        _FakeNode("c", 5.0, compiled=False),
        _action(),
    )
    summarizer.summarize.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_not_correct_skipped():
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock()
    p = _producer(cfg, summarizer)
    await p.consider(
        _FakeNode("p", 10.0),
        _FakeNode("c", 5.0, correct=False),
        _action(),
    )
    summarizer.summarize.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_runtime_skipped():
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock()
    p = _producer(cfg, summarizer)
    await p.consider(_FakeNode("p", None), _FakeNode("c", 5.0), _action())
    summarizer.summarize.assert_not_awaited()


@pytest.mark.asyncio
async def test_below_delta_skipped():
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock()
    p = _producer(cfg, summarizer)
    # ratio = 10.0 / 9.8 = 1.02 < 1.05
    await p.consider(_FakeNode("p", 10.0), _FakeNode("c", 9.8), _action())
    summarizer.summarize.assert_not_awaited()


@pytest.mark.asyncio
async def test_passing_gates_calls_summarizer_and_buffers():
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)
    await p.consider(_FakeNode("p", 10.0), _FakeNode("c", 5.0), _action())
    summarizer.summarize.assert_awaited_once()
    n = await p.flush()
    assert n == 1
    assert len(store.added) == 1
    assert store.added[0].scope == "edge"
    assert store.added[0].speedup == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_consider_threads_iter_no_to_summarizer():
    """``consider`` must forward the caller's ``iter_no`` to ``summarize`` so
    the summarizer's trace_span carries the right iter tag (Codex P2 — usage
    accounting drops untagged summarizer traces)."""
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    p = _producer(cfg, summarizer)
    await p.consider(
        _FakeNode("p", 10.0), _FakeNode("c", 5.0), _action(), iter_no=4,
    )
    assert summarizer.summarize.await_args.kwargs["iter_no"] == 4


@pytest.mark.asyncio
async def test_finalize_threads_iter_zero_to_summarize_run():
    """G3 finalize has no live iter; it passes ``iter_no=0`` (the
    baseline/translate out-of-loop convention) to ``summarize_run``."""
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize_run = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    p = _producer(cfg, summarizer)
    await p.finalize(_FakeNode("base", 10.0), _FakeNode("best", 5.0))
    assert summarizer.summarize_run.await_args.kwargs["iter_no"] == 0


@pytest.mark.asyncio
async def test_summarizer_returns_none_no_buffer():
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=None)
    summarizer.model_name = "mock-model"
    p = _producer(cfg, summarizer)
    await p.consider(_FakeNode("p", 10.0), _FakeNode("c", 5.0), _action())
    assert await p.flush() == 0


@pytest.mark.asyncio
async def test_skip_summarizer_when_ratio_would_be_evicted():
    """Regression for ultra-review finding: if the edge buffer is at cap
    and the new edge's ratio is below the worst buffered ratio, the row
    would be evicted by ``_buffer_append`` immediately after the LLM
    call. Pre-check eviction and skip the LLM call entirely."""
    cfg = _FakeConfig()
    cfg.opt_mem_writes_per_session_cap = 3  # 2 edge slots + 1 G3
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)

    # Fill the buffer with two high-ratio edges.
    await p.consider(_FakeNode("p1", 10.0), _FakeNode("c1", 5.0), _action())   # 2.0
    await p.consider(_FakeNode("p2", 10.0), _FakeNode("c2", 4.0), _action())   # 2.5
    assert summarizer.summarize.await_count == 2

    # Now a below-worst-buffered edge: ratio = 1.5 < min(2.0, 2.5) = 2.0.
    # Pre-check should reject it without calling the summarizer.
    await p.consider(_FakeNode("p3", 10.0), _FakeNode("c3", 10.0 / 1.5), _action())
    assert summarizer.summarize.await_count == 2, (
        "summarizer was called for an edge that would have been evicted"
    )


@pytest.mark.asyncio
async def test_cap_eviction_keeps_top_n_by_ratio():
    cfg = _FakeConfig()
    cfg.opt_mem_writes_per_session_cap = 3  # 2 edge slots + 1 G3 slot
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)
    ratios = [1.5, 1.1, 2.0, 1.3, 1.8]
    for i, ratio in enumerate(ratios):
        await p.consider(
            _FakeNode(f"p{i}", 10.0),
            _FakeNode(f"c{i}", 10.0 / ratio),
            _action(),
        )
    n = await p.flush()
    assert n == 2  # 2 edge slots filled, no G3 produced
    speedups = sorted(e.speedup for e in store.added)
    assert speedups == pytest.approx([1.8, 2.0])


@pytest.mark.asyncio
async def test_finalize_produces_g3_row():
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=_ok_summary())
    summarizer.summarize_run = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)
    await p.finalize(_FakeNode("baseline", 10.0), _FakeNode("best", 4.0))
    summarizer.summarize_run.assert_awaited_once()
    n = await p.flush()
    assert n == 1
    assert store.added[0].scope == "run"
    # Schema invariant: ``scope == "run"`` rows carry no applied action.
    assert store.added[0].action_applied is None


@pytest.mark.asyncio
async def test_finalize_short_circuits_when_cap_is_zero():
    """Regression for Codex finding 1: with ``cap == 0`` neither G1 nor G3
    should be allowed to flush. ``consider()`` already trips on the
    ``_edge_cap() == 0 and buffer empty`` gate; ``finalize()`` needs its
    own cap check or the G3 row escapes the contract.
    """
    cfg = _FakeConfig()
    cfg.opt_mem_writes_per_session_cap = 0
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=_ok_summary())
    summarizer.summarize_run = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)

    # G1 attempt: should be rejected (already guarded; double-check).
    await p.consider(_FakeNode("p", 10.0), _FakeNode("c", 5.0), _action())
    summarizer.summarize.assert_not_awaited()

    # G3 attempt: this is the buggy path Codex flagged.
    await p.finalize(_FakeNode("baseline", 10.0), _FakeNode("best", 4.0))
    summarizer.summarize_run.assert_not_awaited()

    # Flush must write nothing — no G1 buffer, no G3 row.
    assert await p.flush() == 0
    assert store.added == []


@pytest.mark.asyncio
async def test_finalize_below_delta_no_g3():
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize_run = AsyncMock()
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)
    await p.finalize(_FakeNode("baseline", 10.0), _FakeNode("best", 9.8))  # 1.02x
    summarizer.summarize_run.assert_not_awaited()
    assert await p.flush() == 0


@pytest.mark.asyncio
async def test_cap_reservation_for_g3():
    cfg = _FakeConfig()
    cfg.opt_mem_writes_per_session_cap = 3  # 2 edges + 1 G3
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=_ok_summary())
    summarizer.summarize_run = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)
    for i, ratio in enumerate([1.5, 2.0, 1.8]):
        await p.consider(
            _FakeNode(f"p{i}", 10.0),
            _FakeNode(f"c{i}", 10.0 / ratio),
            _action(),
        )
    await p.finalize(_FakeNode("baseline", 10.0), _FakeNode("best", 4.0))
    n = await p.flush()
    assert n == 3
    scopes = sorted(e.scope for e in store.added)
    assert scopes == ["edge", "edge", "run"]
    edge_speedups = sorted(e.speedup for e in store.added if e.scope == "edge")
    # cap=3, 1 reserved for G3 → 2 edge slots; best 2 of [1.5, 2.0, 1.8] = [1.8, 2.0]
    assert edge_speedups == pytest.approx([1.8, 2.0])


@pytest.mark.asyncio
async def test_flush_empty_returns_zero():
    cfg = _FakeConfig()
    summarizer = MagicMock()
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)
    assert await p.flush() == 0
    assert store.added == []


@pytest.mark.asyncio
async def test_row_id_is_deterministic():
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store, run_id="run-X")
    await p.consider(_FakeNode("parent-A", 10.0), _FakeNode("child-B", 5.0), _action())
    await p.flush()
    # Digest includes scope so G1 and G3 don't collide when (parent, child)
    # match. See test_g1_and_g3_row_ids_diverge_on_same_parent_child below.
    expected = hashlib.sha256(b"run-X||parent-A||child-B||edge").hexdigest()[:16]
    assert store.added[0].row_id == f"r_{expected}"


def test_g1_and_g3_row_ids_diverge_on_same_parent_child():
    """Regression for Codex finding 2: a G1 (edge) row and a G3 (run) row built
    from identical (run_id, parent_id, child_id) inputs represent distinct
    lessons, so their row_ids must differ — ``scope`` is folded into the digest.
    (Through the live flush a single-edge win can no longer emit both rows —
    finding #4's buffer-presence suppression drops the G3 when the edge was
    captured — so this asserts the digest divergence at ``_build_experience``,
    where the row-identity contract actually lives.)
    """
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.model_name = "mock-model"
    p = _producer(cfg, summarizer, run_id="run-Y")

    summary = _ok_summary()
    edge = p._build_experience(
        parent_node_id="root", child_node_id="winner", scope="edge",
        speedup=2.5, action=_action(), summary=summary, condition="",
    )
    run_ = p._build_experience(
        parent_node_id="root", child_node_id="winner", scope="run",
        speedup=2.5, action=None, summary=summary, condition="",
    )

    assert edge.scope == "edge" and run_.scope == "run"
    assert edge.row_id != run_.row_id, (
        "G1 and G3 collided on row_id when (parent, child) matched: "
        f"{edge.row_id} == {run_.row_id}"
    )


# --- condition formatting -------------------------------------------------


def test_format_condition_bottleneck_and_params():
    a = ActionRecord("t1_grid_shape", 1, "t1_grid_shape", {"BLOCK_N": "32"})
    assert _format_condition("compute_bound", a) == "compute_bound | BLOCK_N=32"


def test_format_condition_bottleneck_only_for_no_action():
    assert _format_condition("compute_bound", None) == "compute_bound"


def test_format_condition_sorts_params():
    a = ActionRecord("t", 1, "t", {"b": "2", "a": "1"})
    assert _format_condition("memory_bound", a) == "memory_bound | a=1, b=2"


# --- single-edge G3 suppression ------------------------------------------


def _seed_edge(p, *, run_id="run-1", parent_id="baseline", child_id="best",
               ratio=2.5) -> Experience:
    """Append a baseline→best edge Experience to the producer's buffer so the
    buffer-presence G3 suppression check finds a captured edge. Provenance
    carries string node ids (the producer always stringifies them)."""
    exp = Experience(
        row_id="r_seed", schema_version=1, kernel_type="matmul",
        hardware_arch="RTX6000Ada", scope="edge", speedup=ratio,
        action_applied=ActionRecord(action_id="a", tier=1, name="n"),
        title="t", lesson="l", snippet_before="a", snippet_after="b",
        provenance={
            "run_id": run_id,
            "parent_node_id": str(parent_id),
            "child_node_id": str(child_id),
            "summarizer_model": "mock-model",
        },
        created_at="",
    )
    p._edge_buffer.append((ratio, exp))
    return exp


@pytest.mark.asyncio
async def test_finalize_suppresses_single_edge_g3():
    """Single-edge run already captured: the baseline → best edge is present
    in ``_edge_buffer`` (provenance parent==baseline.id, child==best.id), so
    the run-scope G3 would duplicate it and is suppressed."""
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize_run = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)
    baseline = _FakeNode("baseline", 10.0)
    best = _FakeNode("best", 4.0)
    _seed_edge(p, parent_id="baseline", child_id="best")
    await p.finalize(baseline, best, bottleneck="compute_bound")
    assert p._g3_row is None
    summarizer.summarize_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_writes_g3_for_multi_edge():
    """Multi-edge run: no baseline→best edge in the buffer (the best node was
    reached through intermediates), so nothing duplicates the run-scope row →
    G3 written, carrying the bottleneck-only condition."""
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize_run = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)
    baseline = _FakeNode("baseline", 10.0)
    best = _FakeNode("best", 4.0)
    # Buffer holds only an unrelated edge (baseline→intermediate), not
    # baseline→best, so suppression must not fire.
    _seed_edge(p, parent_id="baseline", child_id="intermediate")
    await p.finalize(baseline, best, bottleneck="compute_bound")
    assert p._g3_row is not None
    assert p._g3_row.condition == "compute_bound"  # bottleneck-only on run rows


@pytest.mark.asyncio
async def test_finalize_writes_g3_when_cap_one_single_edge():
    """Regression for Codex finding 4: with ``cap == 1`` ``_edge_cap() == 0``
    so ``consider`` buffers nothing. A single-edge win (best is a direct child
    of baseline) must still WRITE a G3 row — the old ``parent_id`` heuristic
    suppressed it, dropping the only lesson of the run."""
    cfg = _FakeConfig()
    cfg.opt_mem_writes_per_session_cap = 1  # _edge_cap() == 0
    summarizer = MagicMock()
    summarizer.summarize_run = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)
    baseline = _FakeNode("baseline", 10.0)
    best = _FakeNode("best", 4.0)
    await p.finalize(baseline, best, bottleneck="compute_bound")
    assert p._g3_row is not None
    summarizer.summarize_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_writes_g3_when_edge_summary_was_none():
    """Same single-edge shape, but the baseline → best edge's summarize
    returned ``None`` so it never landed in ``_edge_buffer``. With nothing to
    duplicate, ``finalize`` writes the G3 row."""
    cfg = _FakeConfig()
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=None)  # edge summary fails
    summarizer.summarize_run = AsyncMock(return_value=_ok_summary())
    summarizer.model_name = "mock-model"
    store = _FakeStore()
    p = _producer(cfg, summarizer, store=store)
    baseline = _FakeNode("baseline", 10.0)
    best = _FakeNode("best", 4.0)
    # Edge summarize returns None → nothing buffered.
    await p.consider(baseline, best, _action())
    assert p._edge_buffer == []
    await p.finalize(baseline, best, bottleneck="compute_bound")
    assert p._g3_row is not None
