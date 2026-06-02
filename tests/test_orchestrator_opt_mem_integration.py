"""End-to-end opt-mem flow: Producer → MemoryStore on disk, fully mocked.

Exercises the read + write paths through real ``MemoryStore`` /
``MemoryRetriever`` / ``Producer`` instances. The summarizer is mocked
(no LLM calls); the orchestrator is bypassed — these tests drive the
producer directly with fake nodes to verify the producer ↔ store
contract end-to-end. See doc/specs/2026-05-24-optimization-memory-
design.md §13 (integration row).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.reviewer import BranchQuality
from src.memory.experience import ActionRecord
from src.memory.producer import Producer
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore
from src.memory.summarizer import SummarizerResult
from src.runtime.events import DeadReason
from src.search.orchestrator import _is_reviewer_rejected


@dataclass
class _Kernel:
    # Matches the real ``src/kernels/kernel.py:Kernel.source_code`` attribute
    # name; the Producer reads ``node.kernel.source_code``.
    source_code: str = "def kernel(): pass"


@dataclass
class _Node:
    id: str
    runtime_ms: float | None
    compiled: bool = True
    correct: bool = True
    kernel: _Kernel = field(default_factory=_Kernel)


def _mock_summarizer():
    s = MagicMock()
    s.model_name = "mock-model"
    s.summarize = AsyncMock(
        return_value=SummarizerResult(
            title="edge lesson",
            lesson="something changed",
            snippet_before="a",
            snippet_after="b",
        )
    )
    s.summarize_run = AsyncMock(
        return_value=SummarizerResult(
            title="cumulative",
            lesson="overall strategy",
            snippet_before="x",
            snippet_after="y",
        )
    )
    return s


def _config(tmp_path: Path, *, write_enabled: bool = True, read_enabled: bool = True):
    cfg = MagicMock()
    cfg.opt_mem_write_enabled = write_enabled
    cfg.opt_mem_read_enabled = read_enabled
    cfg.opt_mem_writes_per_session_cap = 20
    cfg.opt_mem_min_improvement_ratio = 1.05
    cfg.opt_mem_speedup_weight_alpha = 1.0
    cfg.opt_mem_store_path = tmp_path / "opt_mem" / "store.jsonl"
    cfg.hardware = MagicMock()
    cfg.hardware.name = "RTX6000Ada"
    return cfg


def _action() -> ActionRecord:
    return ActionRecord(action_id="a", tier=1, name="n")


@pytest.mark.asyncio
async def test_full_loop_writes_expected_rows(tmp_path: Path):
    """3 iters with improvements on iters 1 + 3 + cumulative G3.

    Iter 2 is below δ and should not produce a row. Final flush writes
    2 edge rows + 1 G3 row = 3 lines in the JSONL file.
    """
    cfg = _config(tmp_path)
    store = MemoryStore(cfg.opt_mem_store_path)
    summarizer = _mock_summarizer()
    producer = Producer(
        store=store,
        summarizer=summarizer,
        config=cfg,
        run_id="test-run",
        kernel_type="matmul",
    )

    parent = _Node("p", 10.0)
    child1 = _Node("c1", 5.0)   # ratio 2.0  → row
    child2 = _Node("c2", 9.8)   # ratio ~1.02 → below δ, no row
    child3 = _Node("c3", 3.0)   # ratio ~3.27 → row

    await producer.consider(parent, child1, _action())
    await producer.consider(child1, child2, _action())
    await producer.consider(child2, child3, _action())

    baseline = _Node("baseline", 10.0)
    best = _Node("best", 3.0)
    await producer.finalize(baseline, best)

    n = await producer.flush()
    assert n == 3

    assert cfg.opt_mem_store_path.exists()
    lines = cfg.opt_mem_store_path.read_text().splitlines()
    assert len(lines) == 3
    scopes = sorted(json.loads(line)["scope"] for line in lines)
    assert scopes == ["edge", "edge", "run"]


@pytest.mark.asyncio
async def test_write_disabled_writes_no_file(tmp_path: Path):
    cfg = _config(tmp_path, write_enabled=False)
    store = MemoryStore(cfg.opt_mem_store_path)
    summarizer = _mock_summarizer()
    producer = Producer(
        store=store,
        summarizer=summarizer,
        config=cfg,
        run_id="test-run",
        kernel_type="matmul",
    )
    await producer.consider(_Node("p", 10.0), _Node("c", 5.0), _action())
    await producer.finalize(_Node("baseline", 10.0), _Node("best", 3.0))
    n = await producer.flush()
    assert n == 0
    assert not cfg.opt_mem_store_path.exists()


@pytest.mark.asyncio
async def test_read_disabled_returns_empty_despite_seeded_store(tmp_path: Path):
    """``read_enabled=False`` short-circuits even when the store has rows."""
    cfg = _config(tmp_path, read_enabled=False)
    # Seed a row that would otherwise match.
    cfg.opt_mem_store_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.opt_mem_store_path.write_text(
        json.dumps({
            "row_id": "r1", "schema_version": 1, "kernel_type": "matmul",
            "hardware_arch": "RTX6000Ada", "scope": "edge", "speedup": 2.0,
            "action_applied": {
                "action_id": "a", "tier": 1, "name": "n", "parameters": {},
            },
            "title": "t", "lesson": "l",
            "snippet_before": "a", "snippet_after": "b",
            "provenance": {}, "created_at": "",
        }) + "\n"
    )
    store = MemoryStore(cfg.opt_mem_store_path)
    store.load()
    retriever = MemoryRetriever(
        store,
        top_k=cfg.opt_mem_writes_per_session_cap,
        alpha=cfg.opt_mem_speedup_weight_alpha,
        read_enabled=cfg.opt_mem_read_enabled,
    )
    assert retriever.sample("matmul", "RTX6000Ada") == []


# --------------------------------------------------------------------------
# Opt-mem write-gate predicate (Codex review P2 fix).
#
# The orchestrator's opt-mem write gate must skip ONLY Reviewer-rejected
# DEAD_END children, NOT children that ``beam_prune`` marked DEAD_END with
# ``DeadReason.BEAM_PRUNED``. Those edges may still have improved their
# parent and the Producer's δ-gate is the proper arbiter of whether they
# land in the store. These tests exercise the extracted ``_is_reviewer_
# rejected`` predicate directly with fake nodes (the predicate is the
# load-bearing part of the gate; the inline ``consider()`` body is unchanged).
# --------------------------------------------------------------------------


@dataclass
class _GateNode:
    """Minimal stand-in for the search-tree node the gate inspects."""

    branch_quality: BranchQuality | None
    dead_reason: DeadReason | None = None


def test_beam_pruned_dead_end_is_not_reviewer_rejected():
    """A BEAM_PRUNED DEAD_END child must pass the gate (reach consider)."""
    node = _GateNode(
        branch_quality=BranchQuality.DEAD_END,
        dead_reason=DeadReason.BEAM_PRUNED,
    )
    assert _is_reviewer_rejected(node) is False


def test_reviewer_judged_dead_end_is_reviewer_rejected():
    """A REVIEWER_JUDGED DEAD_END child must be skipped (not reach consider)."""
    node = _GateNode(
        branch_quality=BranchQuality.DEAD_END,
        dead_reason=DeadReason.REVIEWER_JUDGED,
    )
    assert _is_reviewer_rejected(node) is True


def test_non_dead_end_is_not_reviewer_rejected():
    """A live (PROMISING) child is never reviewer-rejected."""
    node = _GateNode(branch_quality=BranchQuality.PROMISING, dead_reason=None)
    assert _is_reviewer_rejected(node) is False
