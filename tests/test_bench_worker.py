"""Tier 1 tests for src/eval/bench_worker.py.

The bench_worker module lazy-imports torch / sol_execbench inside
``run_iter`` so the encoder + ``build_request`` tests are torchless. The
``run_iter`` tests monkeypatch the lazy-imported names at
``src.eval.bench_worker.<name>`` after the module first binds them.
"""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# SOL stubs so the worker can import on the torchless venv.
#
# ``RewardHackDetected`` is imported inside ``run_iter``; tests monkeypatch
# the symbol on the worker module after the lazy import binds it. The stub
# below covers the case where the worker's lazy import itself runs — Tier
# 1 venv has no sol_execbench installed, so we provide a minimal module
# tree before any test imports the worker.
# ---------------------------------------------------------------------------


def _install_sol_stubs() -> None:
    if "sol_execbench" in sys.modules:
        return

    pkg = types.ModuleType("sol_execbench")
    core = types.ModuleType("sol_execbench.core")
    bench = types.ModuleType("sol_execbench.core.bench")
    reward_hack = types.ModuleType("sol_execbench.core.bench.reward_hack")
    sol_score_mod = types.ModuleType("sol_execbench.sol_score")

    class RewardHackDetected(Exception):
        """Stub mirroring sol_execbench's reward-hack signal."""

    reward_hack.RewardHackDetected = RewardHackDetected

    def _stub_sol_score(*, t_k, t_p, t_sol):
        # Lower latency → higher score. Tests only need a monotonic ranking.
        if t_k <= 0:
            return 0.0
        return float(t_sol) / float(t_k)

    sol_score_mod.sol_score = _stub_sol_score

    pkg.core = core
    core.bench = bench
    bench.reward_hack = reward_hack

    sys.modules["sol_execbench"] = pkg
    sys.modules["sol_execbench.core"] = core
    sys.modules["sol_execbench.core.bench"] = bench
    sys.modules["sol_execbench.core.bench.reward_hack"] = reward_hack
    sys.modules["sol_execbench.sol_score"] = sol_score_mod


_install_sol_stubs()


# ---------------------------------------------------------------------------
# Encoder tests (Task 2.1)
# ---------------------------------------------------------------------------


def test_encode_path_becomes_str():
    from src.eval.bench_worker import _encode

    assert _encode(Path("/tmp/foo")) == "/tmp/foo"


def test_encode_inf_nan_sanitized_to_none():
    from src.eval.bench_worker import _encode

    assert _encode(math.inf) is None
    assert _encode(-math.inf) is None
    assert _encode(float("nan")) is None
    assert _encode(1.5) == 1.5


def test_encode_enum_value():
    from src.eval.bench_worker import _encode
    from src.eval.types import BottleneckType

    assert _encode(BottleneckType.MEMORY_BOUND) == BottleneckType.MEMORY_BOUND.value


def test_encode_benchmark_result_round_trip():
    """BenchmarkResult → dict preserves the load-bearing timing + workload fields."""
    from src.eval.bench_worker import _encode
    from src.eval.benchmark import BenchmarkResult

    br = BenchmarkResult(
        median_latency_us=10.5,
        min_latency_us=10.0,
        max_latency_us=11.2,
        warmup_runs=25,
        timed_runs=100,
        per_workload_latency_us={"wl_uuid_0": 10.5},
        workload_errors={},
        autotune_winner_per_workload={"wl_uuid_0": {"BLOCK_M": 64}},
    )
    encoded = _encode(br)
    assert isinstance(encoded, dict)
    assert encoded["median_latency_us"] == 10.5
    assert encoded["min_latency_us"] == 10.0
    assert encoded["max_latency_us"] == 11.2
    assert encoded["per_workload_latency_us"] == {"wl_uuid_0": 10.5}
    assert encoded["autotune_winner_per_workload"] == {"wl_uuid_0": {"BLOCK_M": 64}}


# ---------------------------------------------------------------------------
# build_request test (Task 2.5)
# ---------------------------------------------------------------------------


def test_build_request_assembles_schema_v1(tmp_path):
    from src.eval.bench_worker import build_request
    from src.config import HardwareSpec

    hw = HardwareSpec(name="TestGPU", freq_GHz=1.0, compute_capability=8.9)
    req = build_request(
        run_dir=tmp_path,
        iter_no=7,
        worker_dir=tmp_path / "iter_7" / "worker",
        ncu_cache_dir=tmp_path / "ncu_cache",
        candidates=[
            {
                "candidate_idx": 0,
                "source_code": "k0",
                "triton_kernel_name": "k0",
                "entrypoint": "run",
                "dps": False,
            },
            {
                "candidate_idx": 2,
                "source_code": "k2",
                "triton_kernel_name": "k2",
                "entrypoint": "run",
                "dps": False,
            },
        ],
        kernel_spec={
            "name": "test_problem",
            "kernel_type": "elementwise",
            "flop_count": 0,
            "memory_bytes": 0,
            "input_shapes": [],
            "definition_path": str(tmp_path / "problem.yaml"),
            "pytorch_reference": "",
            "t_sol_us": None,
            "entrypoint": "run",
        },
        workloads=[{"uuid": "wl0", "seed": 12345, "axes": {"N": 1024}}],
        definition_path=tmp_path / "problem.yaml",
        hardware_spec=hw,
        anti_cheat_critical_names=["torch.cuda.synchronize"],
        bench_config={"warmup_iters": 25, "repeat_iters": 100, "burn_in_seed": -1},
        profile_config={
            "ncu_enabled": True,
            "analytical_enabled": True,
            "iter_flops": 0,
            "iter_nbytes": 0,
            "repr_workload_idx": 0,
            "problem_definition_path": str(tmp_path / "problem.yaml"),
            "blob_roots": [],
        },
    )
    assert req["schema_version"] == 2  # bumped 2026-05-28 for input_dtypes plumbing
    assert req["iter_no"] == 7
    assert len(req["candidates"]) == 2
    assert req["candidates"][0]["candidate_idx"] == 0
    # cand_idx=1 was dropped in the parent's entrypoint-binding pre-filter.
    assert req["candidates"][1]["candidate_idx"] == 2
    assert req["hardware_spec"]["name"] == "TestGPU"
    assert req["worker_dir"] == str(tmp_path / "iter_7" / "worker")


# ---------------------------------------------------------------------------
# run_iter test helpers
# ---------------------------------------------------------------------------


def _make_successful_bench_result():
    from src.eval.benchmark import BenchmarkResult

    return BenchmarkResult(
        median_latency_us=10.0,
        min_latency_us=9.5,
        max_latency_us=11.0,
        warmup_runs=25,
        timed_runs=100,
        per_workload_latency_us={"wl0": 10.0},
        workload_errors={},
        autotune_winner_per_workload={},
    )


def _build_minimal_request(tmp_path, *, K: int, worker_dir: Path | None = None):
    """Assemble a minimally-valid request.json for K candidates."""
    from src.eval.bench_worker import build_request
    from src.config import HardwareSpec

    if worker_dir is None:
        worker_dir = tmp_path / "iter_1" / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)

    hw = HardwareSpec(name="TestGPU", freq_GHz=1.0, compute_capability=8.9)
    candidates = [
        {
            "candidate_idx": i,
            "source_code": f"# kernel cand {i}",
            "triton_kernel_name": f"k{i}",
            "entrypoint": "run",
            "dps": False,
        }
        for i in range(K)
    ]
    kernel_spec = {
        "name": "test_problem",
        "kernel_type": "elementwise",
        "flop_count": 1000,
        "memory_bytes": 4096,
        "input_shapes": [],
        "definition_path": str(tmp_path / "p.yaml"),
        "pytorch_reference": "",
        "t_sol_us": 5.0,
        "entrypoint": "run",
    }
    workloads = [{"uuid": "wl0", "seed": 12345, "axes": {"N": 1024}}]
    return build_request(
        run_dir=tmp_path,
        iter_no=1,
        worker_dir=worker_dir,
        ncu_cache_dir=tmp_path / "ncu_cache",
        candidates=candidates,
        kernel_spec=kernel_spec,
        workloads=workloads,
        definition_path=tmp_path / "p.yaml",
        hardware_spec=hw,
        anti_cheat_critical_names=[],
        bench_config={"warmup_iters": 25, "repeat_iters": 100, "burn_in_seed": -1},
        profile_config={
            "ncu_enabled": True,
            "analytical_enabled": True,
            "iter_flops": 1000,
            "iter_nbytes": 4096,
            "repr_workload_idx": 0,
            "problem_definition_path": str(tmp_path / "p.yaml"),
            "blob_roots": [],
            "t_sol_us": 5.0,
            "baseline_latency_us": 20.0,
        },
    )


def _patch_worker_internals(monkeypatch, *, bench_side_effect=None, profile_side_effect=None):
    """Patch the worker's lazy-loaded symbols with mocks.

    ``bench_side_effect`` / ``profile_side_effect`` are either a callable
    (invoked per call) or a list (popped from front). When a callable, it
    receives no args and is responsible for raising or returning. When a
    list, each entry is either the return value or an exception class
    instance to raise.
    """
    from src.eval import bench_worker as bw

    def _make_caller(side_effect):
        if side_effect is None:
            return lambda *a, **k: None
        if callable(side_effect):
            return lambda *a, **k: side_effect()
        # list-based: pop front, raise if exception, else return.
        def _call(*a, **k):
            value = side_effect.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        return _call

    monkeypatch.setattr(bw, "benchmark_kernel", _make_caller(bench_side_effect))
    monkeypatch.setattr(bw, "profile_kernel", _make_caller(profile_side_effect))
    # Precompile + rehydration + IO helpers — stubbed so the worker does
    # not need torch / sol_execbench / a real problem dir.
    monkeypatch.setattr(bw, "_safe_precompile", lambda kernel, role: (lambda *a, **k: None, None))
    monkeypatch.setattr(bw, "_rehydrate_workloads", lambda raw: [_StubWorkload(d) for d in raw])
    monkeypatch.setattr(bw, "_build_input_generators", lambda req, workloads: [lambda seed: () for _ in workloads])
    monkeypatch.setattr(bw, "_load_definition", lambda path: object())
    monkeypatch.setattr(bw, "_run_per_iter_anti_cheat", _noop_anti_cheat)
    monkeypatch.setattr(bw, "check_lazy_outputs_after_bench", lambda outputs: None)


class _StubWorkload:
    """Minimal workload object — just exposes ``.uuid``, ``.axes``,
    ``.seed`` so the worker can rank candidates and build the
    representative-workload dict for the profiler."""

    def __init__(self, data: dict):
        self.uuid = data.get("uuid", "wl0")
        self.axes = data.get("axes", {})
        self.seed = data.get("seed", 0)

    def model_dump(self, mode="json"):
        return {"uuid": self.uuid, "axes": self.axes, "seed": self.seed}


from contextlib import contextmanager


@contextmanager
def _noop_anti_cheat(critical_names):
    yield None


# ---------------------------------------------------------------------------
# run_iter tests
# ---------------------------------------------------------------------------


def test_run_iter_happy_path_all_K_succeed(tmp_path, monkeypatch):
    """K=2 candidates both succeed; winner_idx + winner_profile populated."""
    _patch_worker_internals(
        monkeypatch,
        bench_side_effect=lambda: _make_successful_bench_result(),
        profile_side_effect=lambda: _FakeProfile(),
    )

    from src.eval.bench_worker import run_iter

    req = _build_minimal_request(tmp_path, K=2)
    resp = run_iter(req)

    assert resp["schema_version"] == 2  # bumped 2026-05-28 for input_dtypes plumbing
    assert resp["iter_no"] == 1
    assert len(resp["candidates"]) == 2
    assert all(c["status"] == "success" for c in resp["candidates"])
    assert resp["winner_idx"] in (0, 1)
    assert resp["winner_profile"] is not None
    assert resp["aborted_by_channel_A"] is False
    assert resp["child_walltime_s"] >= 0.0


from dataclasses import dataclass, field as _dc_field


@dataclass
class _FakeProfile:
    """Dataclass stand-in for ``ProfilingResult`` used in mocked tests.

    Returned by the patched ``profile_kernel``. The worker calls
    ``_encode`` on it; ``_encode`` requires dataclass-ness to JSON-flatten
    via ``dataclasses.asdict``, so we use ``@dataclass`` here.
    """

    analytical: object | None = None
    ncu: object | None = None
    raw_metrics: dict = _dc_field(default_factory=dict)
    degraded_reason: str | None = None
    ncu_rep_path: object | None = None


def test_run_iter_all_K_fail_returns_no_winner(tmp_path, monkeypatch):
    from src.eval.benchmark import BenchmarkError

    def _raise_bench_err():
        raise BenchmarkError("0/1 workloads survived")

    _patch_worker_internals(monkeypatch, bench_side_effect=_raise_bench_err)

    from src.eval.bench_worker import run_iter

    resp = run_iter(_build_minimal_request(tmp_path, K=2))
    assert resp["winner_idx"] is None
    assert resp["winner_profile"] is None
    assert all(c["status"] == "bench_failed" for c in resp["candidates"])
    assert resp["aborted_by_channel_A"] is False


def test_run_iter_channel_a_mid_loop_marks_remaining_not_run(tmp_path, monkeypatch):
    """K=3, cand 1 trips reward-hack; cand 0 success, cand 2 not_run."""
    from sol_execbench.core.bench.reward_hack import RewardHackDetected

    call_state = {"n": 0}

    def _bench():
        call_state["n"] += 1
        if call_state["n"] == 2:
            raise RewardHackDetected("test taint")
        return _make_successful_bench_result()

    _patch_worker_internals(
        monkeypatch,
        bench_side_effect=_bench,
        profile_side_effect=lambda: _FakeProfile(),
    )

    from src.eval.bench_worker import run_iter

    resp = run_iter(_build_minimal_request(tmp_path, K=3))
    assert resp["candidates"][0]["status"] == "success"
    assert resp["candidates"][1]["status"] == "channel_a_tripped"
    assert resp["candidates"][1]["channel_A_flag"] is True
    assert resp["candidates"][2]["status"] == "not_run"
    assert resp["aborted_by_channel_A"] is True
    # Channel-A trips abort the iter without picking a winner — matches the
    # parent-side `break` semantics today.
    assert resp["winner_idx"] is None


def test_run_iter_profile_failure_keeps_winner_attached_with_null_profile(tmp_path, monkeypatch):
    """K=1 succeeds at bench but profile_kernel raises ProfilerError.

    The profile gauntlet (mirrors orchestrator.py:1230-1293) drops a
    profile-failed candidate and tries next-ranked. With K=1 there is no
    next-ranked, so the gauntlet exhausts and ``winner_idx is None``.
    """
    from src.eval.profiler import ProfilerError

    def _raise_profile_err():
        raise ProfilerError("ncu died")

    _patch_worker_internals(
        monkeypatch,
        bench_side_effect=lambda: _make_successful_bench_result(),
        profile_side_effect=_raise_profile_err,
    )

    from src.eval.bench_worker import run_iter

    resp = run_iter(_build_minimal_request(tmp_path, K=1))
    assert resp["candidates"][0]["status"] == "success"
    assert resp["winner_idx"] is None
    assert resp["winner_profile"] is None


def test_run_iter_happy_path_writes_no_worker_events(tmp_path, monkeypatch):
    """Codex 2026-05-27 fix #1+#2+#4: worker is NOT the per-candidate
    event emitter on the happy path. Parent owns ``bench_done`` /
    ``profile_done`` / ``coder_failed`` / ``reward_hack_detected``
    emits from its response-handling loop — worker emitting the same
    kinds was duplicating events in canonical events.jsonl after
    merge. Worker-side ``_emit`` is reserved for signals only the
    worker can see (profile-gauntlet drops on non-winners). A happy
    path with no gauntlet drops writes nothing to worker/events.jsonl.
    """
    _patch_worker_internals(
        monkeypatch,
        bench_side_effect=lambda: _make_successful_bench_result(),
        profile_side_effect=lambda: _FakeProfile(),
    )

    from src.eval.bench_worker import run_iter

    worker_dir = tmp_path / "iter_1" / "worker"
    req = _build_minimal_request(tmp_path, K=2, worker_dir=worker_dir)
    run_iter(req)

    events_path = worker_dir / "events.jsonl"
    # File may exist as a zero-byte side effect of prior _emit calls (it
    # doesn't today), or not exist at all. Either way the worker writes
    # no per-candidate event content on a clean happy path.
    if events_path.exists():
        assert events_path.read_text() == "", (
            "worker should NOT emit per-candidate events on happy path "
            "— parent re-emit + merge would double-count"
        )


def test_emit_includes_ts_field(tmp_path):
    """Codex 2026-05-27 fix #2: worker ``_emit`` must include a ``ts``
    field matching the parent's ``runtime/events.py::emit`` shape so
    the merged canonical events.jsonl is schema-homogeneous (every
    line has ``ts``, ``kind``, ``iter`` ...). Downstream consumers
    that key on ``ts`` for chronological ordering or windowing must
    not get KeyError on worker-origin records.
    """
    from src.eval.bench_worker import _emit

    worker_dir = tmp_path / "iter_0" / "worker"
    worker_dir.mkdir(parents=True)
    _emit(worker_dir, "coder_failed", iter=0, candidate_idx=1, reason="test")

    line = (worker_dir / "events.jsonl").read_text().strip()
    record = json.loads(line)
    assert "ts" in record, "worker _emit must include ts field"
    assert isinstance(record["ts"], str), "ts must be an ISO timestamp string"
    assert record["kind"] == "coder_failed"
    assert record["iter"] == 0
    assert record["candidate_idx"] == 1


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------


def test_main_cli_reads_request_writes_response(tmp_path, monkeypatch):
    _patch_worker_internals(
        monkeypatch,
        bench_side_effect=lambda: _make_successful_bench_result(),
        profile_side_effect=lambda: _FakeProfile(),
    )

    from src.eval.bench_worker import _main_cli

    worker_dir = tmp_path / "iter_0" / "worker"
    worker_dir.mkdir(parents=True)
    req = _build_minimal_request(tmp_path, K=1, worker_dir=worker_dir)
    request_path = worker_dir / "request.json"
    response_path = worker_dir / "response.json"
    request_path.write_text(json.dumps(req))

    rc = _main_cli(["--request", str(request_path), "--response", str(response_path)])
    assert rc == 0
    assert response_path.exists()
    resp = json.loads(response_path.read_text())
    assert resp["schema_version"] == 2  # bumped 2026-05-28 for input_dtypes plumbing
    assert resp["iter_no"] == 1


def test_run_iter_winner_profile_carries_renamed_ncu_rep_path(tmp_path, monkeypatch):
    """Fix #3 (Codex adversarial 2026-05-26).

    The worker renames the raw .ncu-rep produced by ``profile_kernel``
    into ``worker_dir/cand_<idx>.ncu-rep`` so the parent can find it
    without scanning. ``profile.ncu_rep_path`` MUST reflect that rename
    in the encoded response; otherwise ``tree_dump.dump_node`` reads a
    stale hashed path and silently fails to copy the artifact into the
    committed tree node.
    """
    src_rep = tmp_path / "abc123_sha.ncu-rep"
    src_rep.write_bytes(b"\x00\x01NCU")

    def _profile():
        return _FakeProfile(ncu_rep_path=src_rep)

    _patch_worker_internals(
        monkeypatch,
        bench_side_effect=lambda: _make_successful_bench_result(),
        profile_side_effect=_profile,
    )

    from src.eval.bench_worker import run_iter

    req = _build_minimal_request(tmp_path, K=1)
    resp = run_iter(req)

    assert resp["winner_idx"] == 0
    winner_profile = resp["winner_profile"]
    assert winner_profile is not None
    # The renamed path must be cand_<winner_idx>.ncu-rep under worker_dir;
    # the original hashed path is stale because rename moved the file.
    expected_rep = Path(req["worker_dir"]) / "cand_0.ncu-rep"
    assert expected_rep.exists(), "rename target file should exist on disk"
    assert winner_profile["ncu_rep_path"] == str(expected_rep), (
        f"encoded ncu_rep_path should reflect rename target; "
        f"got {winner_profile['ncu_rep_path']!r}"
    )


def test_run_iter_partial_bench_failure_marks_candidate_bench_failed(tmp_path, monkeypatch):
    """Codex 2026-05-26 review P1 fix.

    ``benchmark_kernel`` returns a ``BenchmarkResult`` with
    ``workload_errors`` populated when *some* workloads fail but enough
    survived to avoid ``BenchmarkError``. The OLD orchestrator gated
    such candidates with ``if not cand_bench.is_fully_successful:
    _accumulate_iter_failure(...); continue`` so they could never
    become winners. The worker must do the same — otherwise a kernel
    that crashes on a non-representative workload can win + commit +
    score in multi-workload runs, producing silently-wrong scores.
    """
    from src.eval.benchmark import BenchmarkResult

    def _partial_bench():
        return BenchmarkResult(
            median_latency_us=10.0,
            min_latency_us=9.5, max_latency_us=11.0,
            warmup_runs=25, timed_runs=100,
            per_workload_latency_us={"wl0": 10.0},  # only wl0 has timing
            workload_errors={"wl1": "CUDA OOM on 4096x4096 shape"},
            autotune_winner_per_workload={},
        )

    _patch_worker_internals(
        monkeypatch,
        bench_side_effect=_partial_bench,
        profile_side_effect=lambda: _FakeProfile(),
    )

    from src.eval.bench_worker import run_iter

    resp = run_iter(_build_minimal_request(tmp_path, K=1))
    # Partial-failure candidates MUST NOT reach the success path or the
    # profile gauntlet — they would become silently-wrong winners.
    assert resp["candidates"][0]["status"] == "bench_failed", (
        f"partial bench failure (workload_errors non-empty) must surface "
        f"as bench_failed; got {resp['candidates'][0]['status']!r}"
    )
    # Reason should mention the failing workload for postmortem.
    assert "wl1" in resp["candidates"][0]["reason"] or "partial" in resp["candidates"][0]["reason"].lower()
    # No winner since the single candidate failed.
    assert resp["winner_idx"] is None
    assert resp["winner_profile"] is None


def test_run_iter_empty_workloads_runs_placeholder_sentinel_path(tmp_path, monkeypatch):
    """Codex 2026-05-26 review P2 fix.

    The placeholder/no-workload run path uses ``benchmark_kernel``'s
    100us sentinel return (workloads + input_generators both empty). My
    refactor's worker unconditionally rebuilt input generators and
    loaded ``definition_path`` before bench, so the placeholder path
    crashed at load before ``benchmark_kernel`` could return the
    sentinel — three iters of that trip ``WorkerProcessUnstable``.

    The worker must skip the load + generator rebuild when workloads
    is empty, then call ``benchmark_kernel`` with empty lists and let
    it return the sentinel.
    """
    from src.eval.benchmark import BenchmarkResult
    from src.eval import bench_worker as bw

    # Real ``_load_definition`` would explode on Path("") / Path("."); we
    # patch it to raise to confirm the worker doesn't call it on the
    # empty-workload path.
    def _explode(*_args, **_kwargs):
        raise FileNotFoundError("worker should NOT call _load_definition with no workloads")

    def _explode_generators(*_args, **_kwargs):
        raise FileNotFoundError("worker should NOT call _build_input_generators with no workloads")

    # Stub the per-candidate eval path so the test only exercises the
    # rehydration guards. Bench returns the sentinel BR (what
    # benchmark_kernel actually does for empty workloads).
    monkeypatch.setattr(bw, "benchmark_kernel", lambda *a, **k: BenchmarkResult(
        median_latency_us=100.0, min_latency_us=100.0, max_latency_us=100.0,
        warmup_runs=25, timed_runs=100,
        per_workload_latency_us={}, workload_errors={},
        autotune_winner_per_workload={},
    ))
    monkeypatch.setattr(bw, "_safe_precompile", lambda kernel, role: (lambda *a, **k: None, None))
    monkeypatch.setattr(bw, "_load_definition", _explode)
    monkeypatch.setattr(bw, "_build_input_generators", _explode_generators)
    monkeypatch.setattr(bw, "_rehydrate_workloads", lambda raw: [])
    monkeypatch.setattr(bw, "_run_per_iter_anti_cheat", _noop_anti_cheat)
    monkeypatch.setattr(bw, "check_lazy_outputs_after_bench", lambda outputs: None)
    monkeypatch.setattr(bw, "profile_kernel", lambda *a, **k: _FakeProfile())

    from src.eval.bench_worker import run_iter

    # Request with empty workloads + placeholder-ish definition_path.
    req = _build_minimal_request(tmp_path, K=1)
    req["workloads"] = []
    req["definition_path"] = ""

    # Must not crash at load; placeholder smoke path stays runnable.
    resp = run_iter(req)
    assert resp["candidates"][0]["status"] == "success"
