"""TDD tests for the correctness fixes called out in CLEANUP.md.

Each test maps to one bug fix:
- ``test_save_load_preserves_kernel_dps`` → ``src/search/tree.py``
  ``_serialize_kernel`` / ``_deserialize_node`` round-trip the ``Kernel.dps``
  flag.
- ``test_best_node_excludes_dead_end_branches`` → ``src/search/tree.py``
  ``best_node()`` excludes ``BranchQuality.DEAD_END`` nodes, even when they
  carry a score.
- ``test_cache_key_invariant_under_dict_ordering`` → ``src/eval/profiler.py``
  ``_cache_key`` uses canonical JSON, so the same workload (different dict
  insertion order) hashes identically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.reviewer import BranchQuality
from src.eval.scorer import ScoreResult
from src.kernels.kernel import Kernel, KernelSpec, KernelType
from src.search.tree import SearchTree


def _make_kernel(name: str = "test", *, dps: bool = False) -> Kernel:
    return Kernel(
        spec=KernelSpec(name=name, kernel_type=KernelType.MATMUL),
        source_code="# placeholder",
        dps=dps,
    )


def _make_score(sol: float) -> ScoreResult:
    return ScoreResult(
        sol_score=sol,
        baseline_latency_us=100.0,
        candidate_latency_us=100.0 - sol * 50.0,
        t_sol_us=50.0,
        speedup=1.0 + sol,
    )


# ── tree: dps round-trip ─────────────────────────────────────────────────────

class TestKernelDpsRoundTrip:
    def test_save_load_preserves_kernel_dps_true(self, tmp_path: Path):
        """``Kernel.dps`` must round-trip through checkpoint save/load.
        Losing it on reload would silently change correctness/profiling
        semantics on a resumed run (DPS kernels write into output buffers
        passed as args, non-DPS return outputs)."""
        tree = SearchTree()
        root = tree.add_root(_make_kernel("root", dps=True))
        root.score = _make_score(0.3)

        save_path = tmp_path / "tree.json"
        tree.save(save_path)
        loaded = SearchTree.load(save_path)

        assert loaded.get_node(root.id).kernel.dps is True

    def test_save_load_preserves_kernel_dps_false(self, tmp_path: Path):
        """Default-False ``dps`` must round-trip too."""
        tree = SearchTree()
        root = tree.add_root(_make_kernel("root", dps=False))
        root.score = _make_score(0.3)

        save_path = tmp_path / "tree.json"
        tree.save(save_path)
        loaded = SearchTree.load(save_path)

        assert loaded.get_node(root.id).kernel.dps is False

    def test_load_legacy_checkpoint_without_dps_field(self, tmp_path: Path):
        """A checkpoint written before the dps field existed must still load,
        defaulting dps to False — matches the existing back-compat strategy
        for ``triton_kernel_name`` and ``consecutive_agent_failures``."""
        import json as _json

        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = _make_score(0.3)

        save_path = tmp_path / "tree.json"
        tree.save(save_path)

        # Strip ``dps`` from the on-disk JSON to simulate an older format.
        data = _json.loads(save_path.read_text())
        data["nodes"]["0"]["kernel"].pop("dps", None)
        save_path.write_text(_json.dumps(data))

        loaded = SearchTree.load(save_path)
        assert loaded.get_node(root.id).kernel.dps is False


# ── tree: best_node excludes DEAD_END ────────────────────────────────────────

class TestBestNodeExcludesDeadEnd:
    def test_dead_end_with_score_cannot_win(self):
        """A DEAD_END branch (e.g. reward_hack_confirmed) may still carry a
        scored result from before it was killed. ``best_node()`` must NOT
        return it, even when its score is the highest in the tree."""
        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = _make_score(0.3)

        # Higher-scoring child, but killed.
        dead = tree.add_child(root.id, _make_kernel("hack"), "tiling")
        dead.score = _make_score(0.99)
        dead.branch_quality = BranchQuality.DEAD_END

        # A live medium-score sibling.
        live = tree.add_child(root.id, _make_kernel("ok"), "unroll")
        live.score = _make_score(0.5)
        live.branch_quality = BranchQuality.PROMISING

        winner = tree.best_node()
        assert winner.id == live.id, (
            "best_node returned a DEAD_END child despite a live alternative"
        )

    def test_only_dead_end_falls_back_to_root(self):
        """When every non-root scored node is DEAD_END, fall back to root —
        same as the no-scored-nodes case."""
        tree = SearchTree()
        root = tree.add_root(_make_kernel("root"))
        root.score = _make_score(0.3)

        dead = tree.add_child(root.id, _make_kernel("hack"), "tiling")
        dead.score = _make_score(0.99)
        dead.branch_quality = BranchQuality.DEAD_END

        winner = tree.best_node()
        assert winner.id == root.id


# ── profiler: cache key canonicalization ─────────────────────────────────────

class TestCacheKeyCanonicalization:
    def test_cache_key_invariant_under_dict_ordering(self):
        """Workload dicts may be re-loaded from JSON in a different key
        order than they were originally constructed. The cache key must
        depend on the *value* of the workload, not its in-memory dict
        ordering — otherwise a fresh process can miss a valid cache entry
        written by an earlier run."""
        from src.eval.profiler import _cache_key

        wl_a = {"M": 256, "N": 512, "K": 128, "dtype": "float16"}
        wl_b = {"dtype": "float16", "K": 128, "N": 512, "M": 256}

        key_a = _cache_key(
            kernel_source="def k(): pass",
            workload=wl_a,
            mode="basic",
            kernel_name="k",
        )
        key_b = _cache_key(
            kernel_source="def k(): pass",
            workload=wl_b,
            mode="basic",
            kernel_name="k",
        )
        assert key_a == key_b

    def test_cache_key_changes_when_value_changes(self):
        """Sanity check: a real workload-value change must still produce a
        different key. (Otherwise the canonicalization could over-collapse.)"""
        from src.eval.profiler import _cache_key

        wl1 = {"M": 256, "N": 512}
        wl2 = {"M": 256, "N": 1024}
        key1 = _cache_key(
            kernel_source="def k(): pass", workload=wl1,
            mode="basic", kernel_name="k",
        )
        key2 = _cache_key(
            kernel_source="def k(): pass", workload=wl2,
            mode="basic", kernel_name="k",
        )
        assert key1 != key2

    def test_cache_key_handles_nested_workload(self):
        """Some workloads carry nested dicts (e.g. per-tensor metadata).
        Canonical JSON sorts at every level."""
        from src.eval.profiler import _cache_key

        wl_a = {"shape": {"M": 256, "N": 512}, "dtype": "fp16"}
        wl_b = {"dtype": "fp16", "shape": {"N": 512, "M": 256}}

        key_a = _cache_key(
            kernel_source="def k(): pass", workload=wl_a,
            mode="basic", kernel_name="k",
        )
        key_b = _cache_key(
            kernel_source="def k(): pass", workload=wl_b,
            mode="basic", kernel_name="k",
        )
        assert key_a == key_b
