"""Tier-1 integration tests for orchestrator-side failure-node
persistence at the K-way fan-out failure sites.

Borrows the harness from ``tests/test_orchestrator_events.py`` (mocked
Planner / Coder / Reviewer / eval stack). Each test exercises one of
the 5 emit sites that produce ``coder_failed`` and verifies whether a
failure node + ``failure_node_added`` event was added (Coder-layer +
bench-layer = yes; profile-layer = no, per
doc/specs/2026-05-17-failure-node-retention-design.md).

Spec: doc/specs/2026-05-17-failure-node-retention-design.md
Plan: doc/plans/2026-05-17-failure-node-retention-plan.md
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.runtime import events
# Re-export the harness fixture so this test file has access.
# pytest auto-discovers fixtures in conftest.py; the harness fixture
# lives in test_orchestrator_events.py and isn't conftest-scoped, so we
# import it directly and re-decorate.
from tests.test_orchestrator_events import (  # noqa: F401
    harness,
    _make_profile,
)


pytestmark = pytest.mark.asyncio


def _events_records(jsonl_path):
    return [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]


async def _run_and_collect(harness, tmp_path, *, bench_override=None, profile_fake=None):
    """Run orchestrator with events binding; return (result, records)."""
    from src.search.orchestrator import Orchestrator

    if profile_fake is None:
        profile_fake = MagicMock(return_value=_make_profile())
    bench_to_use = bench_override or harness.bench

    if callable(bench_to_use):
        bench_patch = patch("src.eval.benchmark.benchmark_kernel", side_effect=bench_to_use)
    else:
        bench_patch = patch("src.eval.benchmark.benchmark_kernel", return_value=bench_to_use)
    profile_patch = patch("src.eval.profiler.profile_kernel", profile_fake)

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with bench_patch, profile_patch:
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder,
                harness.reviewer, harness.retriever,
            )
            result = await orch.run(
                harness.baseline, workloads=None, roofline=harness.roofline,
            )
    finally:
        events.unbind()
        fh.close()

    records = _events_records(tmp_path / "events.jsonl")
    return result, records


# ── Coder-layer failures ──────────────────────────────────────────────────────


async def test_implementation_error_persists_failure_node(tmp_path, harness):
    """Coder-layer ImplementationError → failure node added + event emitted.

    Single-K iter for clarity; K-way semantics covered in
    ``test_kway_all_fail_persists_k_failure_nodes``.
    """
    from src.agents.coder import ImplementationError

    harness.config.coder_n_candidates = 1
    harness.coder.implement = AsyncMock(side_effect=ImplementationError(
        "Coder exhausted turn budget (8) without calling submit_kernel."
    ))

    result, records = await _run_and_collect(harness, tmp_path)
    kinds = [r["kind"] for r in records]

    # Both old and new event fire.
    assert "coder_failed" in kinds
    assert "failure_node_added" in kinds

    # Failure node added under root.
    failure_nodes = [
        n for n in result.tree.nodes()
        if n.failure_reason is not None
    ]
    assert len(failure_nodes) == 1
    failure = failure_nodes[0]
    assert failure.parent_id == 0  # root id
    assert "Coder exhausted turn budget" in failure.failure_reason
    assert failure.kernel is None  # turn-exhaust = no submitted kernel


async def test_entrypoint_binding_failure_persists_failure_node(tmp_path, harness):
    """Coder-layer EntrypointBinding failure → failure node + event."""
    harness.config.coder_n_candidates = 1
    # ``find_jit_name_in_entrypoint`` is imported lazily inside
    # Orchestrator.run() — patch at its source module, not at the
    # orchestrator module (which doesn't yet hold the name when patch
    # tries to read the attribute).
    with patch(
        "src.eval.profiler.find_jit_name_in_entrypoint",
        return_value=(False, "kernel name not found in entrypoint"),
    ):
        result, records = await _run_and_collect(harness, tmp_path)

    kinds = [r["kind"] for r in records]
    assert "coder_failed" in kinds
    assert "failure_node_added" in kinds

    failure_nodes = [n for n in result.tree.nodes() if n.failure_reason is not None]
    assert len(failure_nodes) == 1
    failure = failure_nodes[0]
    assert "EntrypointBinding" in failure.failure_reason
    # Kernel was submitted (just mis-bound), so kernel is not None.
    assert failure.kernel is not None


# ── Bench-layer failures ──────────────────────────────────────────────────────


async def test_benchmark_error_persists_failure_node(tmp_path, harness):
    """Bench-layer BenchmarkError → failure node added + event emitted."""
    from src.eval.benchmark import BenchmarkError

    harness.config.coder_n_candidates = 1

    # First bench call is baseline (succeeds); second is the candidate
    # (raises). Mirrors the call-sequencing pattern from
    # test_partial_bench_failure_surfaces_as_coder_failed.
    call_seq = [harness.bench]

    def next_bench(*_args, **_kwargs):
        if call_seq:
            return call_seq.pop(0)
        raise BenchmarkError(
            "autotune burn-in failed: AcceleratorError: CUDA error: "
            "operation not supported on global/shared address space"
        )

    result, records = await _run_and_collect(
        harness, tmp_path, bench_override=next_bench,
    )
    assert "failure_node_added" in [r["kind"] for r in records]
    failure_nodes = [n for n in result.tree.nodes() if n.failure_reason is not None]
    assert len(failure_nodes) == 1
    assert "operation not supported" in failure_nodes[0].failure_reason
    # Bench-layer = kernel was submitted, so kernel is not None.
    assert failure_nodes[0].kernel is not None


async def test_partial_bench_failure_persists_failure_node(tmp_path, harness):
    """``not cand_bench.is_fully_successful`` → failure node added."""
    from src.eval.benchmark import BenchmarkResult

    harness.config.coder_n_candidates = 1
    partial = BenchmarkResult(
        median_latency_us=100.0,
        timed_runs=1,
        per_workload_latency_us={"wl-0": 100.0, "wl-1": float("inf")},
        workload_errors={"wl-1": "launch failed"},
    )
    # First bench is baseline (happy), second is candidate (partial).
    call_seq = [harness.bench, partial]

    def next_bench(*_args, **_kwargs):
        return call_seq.pop(0) if call_seq else partial

    result, records = await _run_and_collect(
        harness, tmp_path, bench_override=next_bench,
    )
    assert "failure_node_added" in [r["kind"] for r in records]
    failure_nodes = [n for n in result.tree.nodes() if n.failure_reason is not None]
    assert len(failure_nodes) == 1
    assert "partial bench failure" in failure_nodes[0].failure_reason


# ── K-way fan-out semantics ───────────────────────────────────────────────────


async def test_kway_all_fail_persists_k_failure_nodes(tmp_path, harness):
    """K=3 iter with all 3 ImplementationError → 3 failure nodes, iter SKIPPED."""
    from src.agents.coder import ImplementationError

    harness.config.coder_n_candidates = 3
    harness.coder.implement = AsyncMock(side_effect=ImplementationError(
        "budget exhausted"
    ))

    result, records = await _run_and_collect(harness, tmp_path)
    failure_nodes = [n for n in result.tree.nodes() if n.failure_reason is not None]
    assert len(failure_nodes) == 3
    # All three parent → root.
    assert all(n.parent_id == 0 for n in failure_nodes)
    # Distinct candidate_idx in event payloads (0..2).
    events_added = [r for r in records if r["kind"] == "failure_node_added"]
    assert {e["candidate_idx"] for e in events_added} == {0, 1, 2}
    # Iter ended SKIPPED.
    iter_ends = [r for r in records if r["kind"] == "iter_end"]
    assert iter_ends[-1]["outcome"] == "skipped"


# ── Profile-layer (negative test) ─────────────────────────────────────────────


async def test_failure_node_writes_meta_and_kernel_to_disk(tmp_path, harness):
    """Failure-node persistence must stream artifacts to <run>/tree/node_<id>/
    so postmortems can inspect what crashed.

    Codex adversarial review (2026-05-17) [high] flagged that the in-
    memory failure node was added but never streamed — meta.json and
    kernel.py were missing for bench-layer failures, undermining the
    feature's stated postmortem goal. Regression guard verifies both
    files appear on disk with the raw failure_reason in meta.json.
    """
    from src.eval.benchmark import BenchmarkError
    from src.runtime import tree_dump

    harness.config.coder_n_candidates = 1
    call_seq = [harness.bench]

    def next_bench(*_args, **_kwargs):
        if call_seq:
            return call_seq.pop(0)
        raise BenchmarkError("autotune burn-in failed: cudaErrorInvalidAddressSpace")

    # Bind tree_dump to a per-test root so dump_node has somewhere to write.
    tree_root = tmp_path / "tree"
    tree_dump.bind(tree_root)
    try:
        result, _ = await _run_and_collect(
            harness, tmp_path, bench_override=next_bench,
        )
    finally:
        tree_dump.unbind()

    # One failure node added; verify its disk artifacts exist.
    failure_nodes = [n for n in result.tree.nodes() if n.failure_reason is not None]
    assert len(failure_nodes) == 1
    failure = failure_nodes[0]
    node_dir = tree_root / f"node_{failure.id}"
    assert node_dir.exists(), f"missing failure-node directory {node_dir}"
    meta_path = node_dir / "meta.json"
    assert meta_path.exists(), f"missing meta.json for failure node"
    # Bench-layer failure → kernel was submitted → kernel.py present.
    kernel_path = node_dir / "kernel.py"
    assert kernel_path.exists(), f"missing kernel.py for bench-layer failure"
    # meta.json carries the raw failure_reason via the failure_detail field.
    import json
    meta = json.loads(meta_path.read_text())
    assert "failure_detail" in meta
    assert "cudaErrorInvalidAddressSpace" in meta["failure_detail"]
    # dead_reason propagates as the categorical cause.
    assert meta.get("dead_reason") == "coder_failed"


async def test_turn_exhaust_failure_writes_meta_but_no_kernel(tmp_path, harness):
    """Turn-exhaust failure nodes have ``kernel=None`` so kernel.py is
    skipped by dump_node; meta.json must still appear."""
    from src.agents.coder import ImplementationError
    from src.runtime import tree_dump

    harness.config.coder_n_candidates = 1
    harness.coder.implement = AsyncMock(side_effect=ImplementationError(
        "Coder exhausted turn budget (8) without calling submit_kernel."
    ))

    tree_root = tmp_path / "tree"
    tree_dump.bind(tree_root)
    try:
        result, _ = await _run_and_collect(harness, tmp_path)
    finally:
        tree_dump.unbind()

    failure_nodes = [n for n in result.tree.nodes() if n.failure_reason is not None]
    assert len(failure_nodes) == 1
    failure = failure_nodes[0]
    node_dir = tree_root / f"node_{failure.id}"
    assert (node_dir / "meta.json").exists()
    # No kernel was submitted; kernel.py is intentionally absent.
    assert not (node_dir / "kernel.py").exists()


async def test_profile_layer_failure_does_not_persist_failure_node(tmp_path, harness):
    """Per A2 + spec: NCU-crash-on-benched-candidate stays drop-on-the-floor.

    No failure node, no failure_node_added event — even though
    ``coder_failed`` may fire as part of A2's existing bookkeeping. The
    bench succeeded; profile-layer crashes are infra-noise, not search
    signal.
    """
    from src.eval.profiler import ProfilerError

    harness.config.coder_n_candidates = 1
    # Bench succeeds (default harness.bench); profile raises
    # ProfilerError (the only exception class the orchestrator catches
    # at the profile-layer site — bare RuntimeError would propagate).
    profile_raises = MagicMock(side_effect=ProfilerError("NCU crash"))
    result, records = await _run_and_collect(
        harness, tmp_path, profile_fake=profile_raises,
    )

    failure_nodes = [n for n in result.tree.nodes() if n.failure_reason is not None]
    assert len(failure_nodes) == 0
    assert "failure_node_added" not in [r["kind"] for r in records]


# ── autotune_exclude end-to-end plumbing ──────────────────────────────────────


async def test_planner_autotune_exclude_reaches_coder_submit_validator(tmp_path, harness):
    """End-to-end: Planner returns a plan with ``autotune_exclude``;
    orchestrator forwards it through ``coder.implement(plan=...)``;
    Coder exhausts turns; failure node added with the existing
    ``Coder exhausted turn budget`` reason (no new failure class needed
    per doc/specs/2026-05-18-autotune-exclude-structured-bounds-design.md).

    The validator behavior is unit-tested in ``tests/test_coder.py``;
    this test verifies the new field reaches the Coder closure unchanged
    without an orchestrator-side modification.
    """
    from src.agents.coder import ImplementationError
    from src.agents.planner import OptimizationPlan

    harness.config.coder_n_candidates = 1
    harness.planner.plan = AsyncMock(return_value=OptimizationPlan(
        tier=1,
        technique="t1_block_size_tuning",
        params={"BLOCK_K": "32"},
        rationale="exclude the overcommitted config that crashed prior siblings",
        autotune_exclude=[{"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 4}],
    ))
    # Simulate Coder exhausting its turn budget — the natural outcome of
    # repeated validator rejection in a real run.
    harness.coder.implement = AsyncMock(side_effect=ImplementationError(
        "Coder exhausted turn budget (8) without calling submit_kernel."
    ))

    result, records = await _run_and_collect(harness, tmp_path)

    # Orchestrator forwarded the plan with autotune_exclude populated.
    call_args = harness.coder.implement.call_args
    assert call_args is not None
    plan_arg = call_args.kwargs.get("plan")
    assert plan_arg is not None
    assert plan_arg.autotune_exclude == [
        {"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 4}
    ]
    # Failure node added via the existing K-way Coder-failure flow.
    failure_nodes = [n for n in result.tree.nodes() if n.failure_reason is not None]
    assert len(failure_nodes) == 1
    assert "turn budget" in failure_nodes[0].failure_reason
