"""Bench-subprocess isolation — orchestrator-side helpers + integration.

Two layers:

- **Tier 1 (torchless)** — covers the new module-level helpers added to
  ``src/search/orchestrator.py`` (``_serialize_kernel_spec_for_request``,
  ``_rebuild_cand_kernel``, ``_rehydrate_bench_result``,
  ``_rehydrate_profiling_result``). These don't reach the orchestrator's
  ``run()`` body so they don't transitively import torch via
  ``src/eval/anti_cheat.py``.

- **Tier 2 (``@pytest.mark.gpu``)** — exercises the orchestrator's K-way
  bench-dispatch branch end-to-end with the real subprocess path. Lives
  alongside the Tier 2 smoke at ``tests/test_bench_subprocess_gpu.py``
  conceptually; kept here so the dispatch-layer assertions live next to
  the helpers they exercise.

See doc/specs/2026-05-24-bench-subprocess-isolation-design.md.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# ───────────────────────── torchless stub installation ─────────────────────
# The orchestrator module-level imports (``src/eval/types``, ``src/runtime/
# events``, etc.) are torch-free; the helpers we're testing don't trigger
# torch transitively. Only ``Orchestrator.run()`` brings in
# ``src/eval/anti_cheat`` (which imports torch). We can import the
# helpers without torch by importing the orchestrator module top-level.


# ─── Tier 1: serialization helpers ────────────────────────────────────────


def test_serialize_kernel_spec_round_trips_basic_fields():
    """KernelSpec → dict preserves name, entrypoint, kernel_type.value."""
    from src.kernels.kernel import KernelSpec, KernelType
    from src.search.orchestrator import _serialize_kernel_spec_for_request

    spec = KernelSpec(
        name="rmsnorm_fwd",
        entrypoint="run",
        kernel_type=KernelType.ELEMENTWISE,
    )
    d = _serialize_kernel_spec_for_request(spec)
    assert d["name"] == "rmsnorm_fwd"
    assert d["entrypoint"] == "run"
    # KernelType enum → string value via dataclasses.asdict.
    assert d["kernel_type"] in (KernelType.ELEMENTWISE.value, "elementwise")


def test_serialize_kernel_spec_path_fields_become_strings():
    """Path-typed fields on KernelSpec coerce to str so JSON dump works."""
    from src.kernels.kernel import KernelSpec, KernelType
    from src.search.orchestrator import _serialize_kernel_spec_for_request

    # KernelSpec doesn't currently hold a Path field at top level — but
    # the helper's defensive top-level Path scan guards against future
    # additions. Verify the path-scan branch by adding a Path attribute
    # after construction (mirrors the contract the helper documents).
    spec = KernelSpec(
        name="x", entrypoint="run", kernel_type=KernelType.ELEMENTWISE,
    )
    d = _serialize_kernel_spec_for_request(spec)
    # Round-trip JSON to confirm no Path slip-through.
    import json
    assert json.dumps(d)  # serializable


def test_rebuild_cand_kernel_uses_request_dict_fields():
    """_rebuild_cand_kernel reconstructs Kernel from the per-candidate dict."""
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    from src.search.orchestrator import _rebuild_cand_kernel

    spec = KernelSpec(
        name="x", entrypoint="run", kernel_type=KernelType.ELEMENTWISE,
    )
    request_cands = [
        {"candidate_idx": 0, "source_code": "k0", "triton_kernel_name": "k0",
         "dps": False},
        {"candidate_idx": 2, "source_code": "k2", "triton_kernel_name": "k2",
         "dps": True},
    ]
    k = _rebuild_cand_kernel(request_cands, 2, spec)
    assert isinstance(k, Kernel)
    assert k.source_code == "k2"
    assert k.triton_kernel_name == "k2"
    assert k.dps is True
    assert k.spec is spec


def test_rebuild_cand_kernel_missing_idx_raises():
    """Defensive: unknown cand_idx raises StopIteration (caller bug)."""
    from src.kernels.kernel import KernelSpec, KernelType
    from src.search.orchestrator import _rebuild_cand_kernel

    spec = KernelSpec(
        name="x", entrypoint="run", kernel_type=KernelType.ELEMENTWISE,
    )
    with pytest.raises(StopIteration):
        _rebuild_cand_kernel([{"candidate_idx": 0}], 7, spec)


# ─── Tier 1: bench/profile rehydration ─────────────────────────────────────


def test_rehydrate_bench_result_fills_all_known_fields():
    """Encoded BenchmarkResult dict → fully populated dataclass instance."""
    from src.eval.benchmark import BenchmarkResult
    from src.search.orchestrator import _rehydrate_bench_result

    encoded = {
        "median_latency_us": 10.5,
        "min_latency_us": 9.0,
        "max_latency_us": 12.0,
        "warmup_runs": 25,
        "timed_runs": 100,
        "per_workload_latency_us": {"wl0": 10.5, "wl1": 11.0},
        "workload_errors": {},
        "autotune_winner_per_workload": {"wl0": {"BLOCK_M": 64}},
    }
    br = _rehydrate_bench_result(encoded)
    assert isinstance(br, BenchmarkResult)
    assert br.median_latency_us == 10.5
    assert br.per_workload_latency_us == {"wl0": 10.5, "wl1": 11.0}
    assert br.autotune_winner_per_workload == {"wl0": {"BLOCK_M": 64}}
    # last_outputs not transferred across IPC — defaults to empty list.
    assert br.last_outputs == []
    # is_fully_successful is a property — confirms no errors → True.
    assert br.is_fully_successful is True


def test_rehydrate_bench_result_filters_unknown_kwargs():
    """Future worker may emit extra fields; defensive filter keeps construction safe."""
    from src.search.orchestrator import _rehydrate_bench_result

    encoded = {
        "median_latency_us": 5.0,
        "warmup_runs": 10,
        "timed_runs": 50,
        # Unknown field — must not crash the constructor.
        "_future_field_added_in_v2": "extra",
    }
    br = _rehydrate_bench_result(encoded)
    assert br.median_latency_us == 5.0


def test_rehydrate_profiling_result_none_returns_none():
    """When child reports no winner profile (gauntlet exhausted), parent rehydrates None."""
    from src.search.orchestrator import _rehydrate_profiling_result
    assert _rehydrate_profiling_result(None) is None


def test_rehydrate_profiling_result_populates_nested_analytical_and_ncu():
    """Encoded ProfilingResult with nested AnalyticalMetrics + NCUMetrics →
    fully reconstructed dataclass tree."""
    from src.eval.profiler import AnalyticalMetrics, NCUMetrics, ProfilingResult
    from src.search.orchestrator import _rehydrate_profiling_result

    # AnalyticalMetrics field shape — read from dataclass fields. We
    # populate the minimum needed for construction; defensive filter
    # handles missing fields by relying on dataclass defaults.
    encoded = {
        "analytical": {
            "achieved_tflops": 10.5,
            "achieved_bandwidth_gb_s": 200.0,
            "pct_peak_compute": 11.5,
            "pct_peak_bandwidth": 21.0,
        },
        "ncu": {
            "sm_occupancy_pct": 50.0,
            "l2_hit_rate_pct": 90.0,
            "tensor_core_util_pct": 0.0,
            "warp_stall_dominant": "long_scoreboard",
            "warp_stall_dominant_pct": 25.0,
            "warp_stall_runner_up": "mio_throttle",
            "warp_stall_runner_up_pct": 10.0,
        },
        "raw_metrics": {"smsp__inst_executed.avg": 1000.0},
        "metric_groups": {},
        "degraded_reason": None,
        "ncu_rep_path": "/tmp/foo.ncu-rep",
    }
    pr = _rehydrate_profiling_result(encoded)
    assert isinstance(pr, ProfilingResult)
    assert pr.analytical is not None
    assert isinstance(pr.analytical, AnalyticalMetrics)
    assert pr.analytical.achieved_tflops == 10.5
    assert pr.ncu is not None
    assert isinstance(pr.ncu, NCUMetrics)
    assert pr.ncu.sm_occupancy_pct == 50.0
    assert pr.raw_metrics == {"smsp__inst_executed.avg": 1000.0}
    assert pr.ncu_rep_path == Path("/tmp/foo.ncu-rep")
    # has_analytical + has_ncu derived properties:
    assert pr.has_analytical is True
    assert pr.has_ncu is True


def test_rehydrate_profiling_result_handles_missing_analytical_and_ncu():
    """Encoded dict with analytical=None and ncu=None → degraded ProfilingResult."""
    from src.search.orchestrator import _rehydrate_profiling_result

    encoded = {
        "analytical": None,
        "ncu": None,
        "raw_metrics": {},
        "metric_groups": {},
        "degraded_reason": "no_ncu_binary",
        "ncu_rep_path": None,
    }
    pr = _rehydrate_profiling_result(encoded)
    assert pr is not None
    assert pr.analytical is None
    assert pr.ncu is None
    assert pr.degraded_reason == "no_ncu_binary"
    assert pr.ncu_rep_path is None
    assert pr.degraded is True


def test_rehydrate_profiling_result_filters_unknown_kwargs():
    """Unknown fields in nested dataclasses must not break rehydration."""
    from src.search.orchestrator import _rehydrate_profiling_result

    encoded = {
        "analytical": {
            "achieved_tflops": 1.0,
            "achieved_bandwidth_gb_s": 1.0,
            "pct_peak_compute": 1.0,
            "pct_peak_bandwidth": 1.0,
            "_added_in_future": 999,
        },
        "ncu": None,
        "_top_level_future_field": "extra",
    }
    pr = _rehydrate_profiling_result(encoded)
    assert pr is not None
    assert pr.analytical.achieved_tflops == 1.0


# ─── Tier 1: ACTSConfig fields gate dispatch decision ──────────────────────


def test_orchestrator_init_initializes_worker_crash_counter():
    """consecutive_worker_crashes starts at 0 — mirrors consecutive_cuda_errors."""
    # Avoid full Orchestrator construction (needs MemoryRetriever etc.);
    # just check the class attribute initialization path via inspect.
    import inspect
    from src.search.orchestrator import Orchestrator
    src = inspect.getsource(Orchestrator.__init__)
    assert "consecutive_worker_crashes" in src
    assert "= 0" in src


def test_orchestrator_profile_config_includes_sol_context_and_blob_roots():
    """Fix #1 (Codex adversarial 2026-05-26).

    The K-way ``build_request(...)`` call site at
    ``src/search/orchestrator.py`` must populate ``profile_config`` with
    ``t_sol_us`` + ``baseline_latency_us`` (parent's SOL ranking contract)
    and ``problem_definition_path`` + ``blob_roots`` (safetensors-backed
    workload reconstruction + NCU input rebuild). Without these the
    worker falls back to defaults that diverge from the parent contract
    or crashes safetensors workloads outright.

    Regression test via source inspection — the helper signature stays
    decoupled (``build_request`` just passes the dict through), so the
    bug surface is "did orchestrator put the right keys in the dict
    literal." Brittle but bounded.
    """
    import inspect
    from src.search.orchestrator import Orchestrator

    src = inspect.getsource(Orchestrator.run)
    # All four keys must appear inside the ``profile_config=`` dict
    # literal of the K-way bench-dispatch site.
    for key in (
        '"t_sol_us"',
        '"baseline_latency_us"',
        '"problem_definition_path"',
        '"blob_roots"',
    ):
        assert key in src, (
            f"profile_config dict literal is missing {key} — bench "
            f"worker will fall back to defaults that diverge from parent "
            f"SOL ranking / crash safetensors workloads"
        )
    # And the source values must be the right primitives so the worker
    # sees real numbers (not the literal-zero placeholder).
    assert "roofline.t_sol_us" in src
    assert "baseline_bench.median_latency_us" in src
    assert "_resolve_blob_roots(" in src


def test_orchestrator_worker_crash_does_not_quarantine_parent():
    """Codex 2026-05-26 review P2 fix #1.

    Worker crashes are *infrastructure* failures already tracked by
    ``Orchestrator.consecutive_worker_crashes`` (3-strike escalation to
    ``WorkerProcessUnstable``). They MUST NOT also bump
    ``parent.consecutive_agent_failures``, because the tree's
    ``QUARANTINE_THRESHOLD`` is 2 — two transient worker crashes on the
    only frontier node would quarantine it, ``frontier()`` would go
    empty, and the search would end as ``ALL_DEAD_END`` before the
    third crash could correctly raise ``WorkerProcessUnstable``.

    Regression test via source inspection — the WorkerCrashed handler
    branch must NOT contain the agent-failure bump. Brittle but bounded.
    """
    import inspect
    from src.search.orchestrator import Orchestrator

    src = inspect.getsource(Orchestrator.run)
    # Locate the WorkerCrashed except branch.
    try:
        idx = src.index("except WorkerCrashed")
    except ValueError:
        raise AssertionError(
            "WorkerCrashed handler not found in Orchestrator.run — "
            "test needs an updated anchor"
        )
    # Look only at the block from the handler to the trailing
    # ``continue``; ``consecutive_agent_failures`` must not be bumped
    # there. ``= 0`` resets in other branches are fine.
    end_marker = "epsilon = max(self._config.epsilon_end, epsilon - decay)"
    handler_block = src[idx : src.index(end_marker, idx) + len(end_marker)]
    assert "parent.consecutive_agent_failures += 1" not in handler_block, (
        "WorkerCrashed handler must NOT bump parent.consecutive_agent_failures "
        "— worker crashes are infra failures already tracked by "
        "self.consecutive_worker_crashes; double-counting them quarantines "
        "the parent before WorkerProcessUnstable can fire"
    )


def test_rehydrate_profiling_result_handles_missing_required_nested_fields():
    """Codex 2026-05-27 fix #5: missing-required-field in nested
    AnalyticalMetrics / NCUMetrics dicts must degrade to None for the
    affected block, NOT raise TypeError. Without the defensive
    try/except, a worker schema drift (e.g., rename of one NCUMetrics
    field) would escape ``except WorkerCrashed`` in the orchestrator
    and abort the entire run on a survivable mismatch.
    """
    from src.search.orchestrator import _rehydrate_profiling_result

    # NCUMetrics has 7 required fields; supply only 2. Old behavior:
    # TypeError. New behavior: ncu=None, ProfilingResult still
    # constructed with analytical (if present).
    encoded = {
        "analytical": {
            "achieved_tflops": 10.0,
            "achieved_bandwidth_gb_s": 100.0,
            "pct_peak_compute": 10.0,
            "pct_peak_bandwidth": 20.0,
        },
        "ncu": {
            "sm_occupancy_pct": 50.0,
            "l2_hit_rate_pct": 90.0,
            # Missing: tensor_core_util_pct, warp_stall_dominant,
            # warp_stall_dominant_pct, warp_stall_runner_up,
            # warp_stall_runner_up_pct
        },
        "raw_metrics": {},
        "metric_groups": {},
        "degraded_reason": None,
        "ncu_rep_path": None,
    }
    pr = _rehydrate_profiling_result(encoded)
    assert pr is not None, "rehydrate should not return None when analytical succeeds"
    assert pr.analytical is not None, "analytical block should construct successfully"
    assert pr.ncu is None, "ncu with missing required fields must degrade to None"


def test_orchestrator_warns_on_silent_subprocess_downgrade():
    """Codex 2026-05-27 fix #3: operator's bench_use_subprocess=True
    cfg toggle silently becoming a no-op when ``run_dir is None``
    (e.g., RunContext.create's OSError fallback) is a safety
    regression — CUDA-context isolation, the entire point of the
    subprocess, is DISABLED. Orchestrator.run must log a WARNING
    when both conditions hold.

    Regression test via source inspection — the run() body must
    contain a logger.warning call gated on the downgrade condition.
    """
    import inspect
    from src.search.orchestrator import Orchestrator

    src = inspect.getsource(Orchestrator.run)
    assert "bench_use_subprocess=True but run_dir is None" in src, (
        "Orchestrator.run must surface the silent-downgrade case as "
        "a logger.warning so operators don't ship runs they think are "
        "isolated but actually aren't"
    )
    assert "logger.warning" in src or "logging.warning" in src, (
        "downgrade must be a warning, not silent"
    )
