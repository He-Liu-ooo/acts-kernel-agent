"""Orchestrator tests for anti-cheat wrap, reward-hack re-evaluation,
and CUDA sticky-state recovery.

Reuses the mock-agents harness pattern from ``test_orchestrator_events.py``
(self-contained copy here; importing across test modules couples their
lifecycles). Tier 1 (no GPU).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import rtx6000_ada_hardware as _rtx6000_ada
from src.agents.coder import KernelCodeOutput
from src.agents.planner import OptimizationPlan
from src.agents.reviewer import BranchQuality, ReviewerFeedback
from src.config import ACTSConfig
from src.eval.benchmark import BenchmarkResult
from src.eval.profiler import (
    AnalyticalMetrics,
    NCUMetrics,
    ProfilingResult,
)
from src.eval.roofline import RooflineResult
from src.eval.scorer import ScoreResult
from src.eval.types import BottleneckType
from src.kernels.kernel import Kernel, KernelSpec, KernelType
from src.runtime import events


def _make_kernel(name: str = "root") -> Kernel:
    return Kernel(
        spec=KernelSpec(
            name=name,
            kernel_type=KernelType.MATMUL,
            flop_count=1_000_000,
            memory_bytes=100_000,
        ),
        source_code="# placeholder",
    )


def _make_profile() -> ProfilingResult:
    return ProfilingResult(
        analytical=AnalyticalMetrics(
            achieved_tflops=1.0,
            achieved_bandwidth_gb_s=100.0,
            pct_peak_compute=0.1,
            pct_peak_bandwidth=0.5,
        ),
        ncu=NCUMetrics(
            sm_occupancy_pct=72.5,
            l2_hit_rate_pct=45.0,
            tensor_core_util_pct=0.0,
            warp_stall_dominant="long_scoreboard",
            warp_stall_dominant_pct=33.0,
            warp_stall_runner_up="wait",
            warp_stall_runner_up_pct=18.0,
        ),
        raw_metrics={},
        degraded_reason=None,
    )


@pytest.fixture
def harness():
    """Orchestrator harness with mocked agents returning happy-path values."""
    # coder_n_candidates=1: A2 K-way fan-out is covered by test_k_way_*
    # in test_orchestrator_events.py; these tests exercise orthogonal
    # iter-flow paths (reward-hack, CUDA recovery, trace_emitted, etc.)
    # that benefit from legacy single-Coder mock cardinality.
    config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=1,
        beam_width=3,
        sol_plateau_window=99,
        coder_n_candidates=1,
    )
    planner = MagicMock()
    planner.plan = AsyncMock(return_value=OptimizationPlan(
        tier=3, technique="tiling", params={}, target_region="",
        rationale="reshape loop tiling for better cache reuse",
    ))
    coder = MagicMock()
    coder.implement = AsyncMock(
        return_value=KernelCodeOutput.model_construct(
            source_code="# child source",
            triton_kernel_name="",
            dps=False,
        )
    )
    reviewer = MagicMock()
    reviewer.review = AsyncMock(return_value=ReviewerFeedback(
        outcome="improved",
        bottleneck_classification="memory_bound",
        branch_quality=BranchQuality.PROMISING,
    ))
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=[])

    bench = BenchmarkResult(median_latency_us=100.0, timed_runs=1)
    baseline = _make_kernel("root")
    roofline = RooflineResult(
        t_sol_us=50.0,
        bottleneck=BottleneckType.MEMORY_BOUND,
    )

    return SimpleNamespace(
        config=config,
        planner=planner,
        coder=coder,
        reviewer=reviewer,
        retriever=retriever,
        bench=bench,
        baseline=baseline,
        roofline=roofline,
    )


def _capture_events(tmp_path):
    """Returns ``(events_path, bind_callable, unbind_callable)``."""
    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)

    def _close():
        events.unbind()
        fh.close()

    return tmp_path / "events.jsonl", _close


def _read_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── reward_hack_detected (channel A) ───────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_marks_branch_dead_on_reward_hack_detected(tmp_path, harness):
    """A2: when ``per_iter_anti_cheat`` raises ``RewardHackDetected`` during
    a candidate's benchmark, the orchestrator emits ``coder_failed`` with
    a reward-hack reason (one per failing candidate) and the iter ends
    SKIPPED. The legacy single-Coder path that emitted ``reward_hack_detected``
    + ``branch_dead_end`` on a tree node is gated out under K-way — losers
    don't enter the tree. Channel-B reward-hack (the SOL-scorer's
    ``reward_hack_suspect`` re-eval) still fires for winning candidates;
    those tests live in ``test_orchestrator_re_evals_suspect_score_*``.
    """
    from src.search.orchestrator import Orchestrator
    from sol_execbench.core.bench.reward_hack import RewardHackDetected

    events_path, close = _capture_events(tmp_path)

    try:
        calls = {"n": 0}

        def side_effect(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return harness.bench  # baseline
            raise RewardHackDetected("monkey-patched torch.cuda.Event.elapsed_time")

        with (
            patch("src.eval.benchmark.benchmark_kernel", side_effect=side_effect),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder, harness.reviewer,
                harness.retriever,
            )
            result = await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)
    finally:
        close()

    records = _read_events(events_path)
    kinds = [r["kind"] for r in records]
    # Codex review #2: per-candidate channel-A reward-hack emits a
    # dedicated ``reward_hack_detected`` event (with ``candidate_idx``)
    # so telemetry consumers watching trust-boundary violations don't
    # have to substring-match a generic ``coder_failed.reason``.
    rh_detected = [r for r in records if r["kind"] == "reward_hack_detected"]
    assert len(rh_detected) >= 1
    assert any("monkey-patched" in r["reason"] for r in rh_detected)
    assert all("candidate_idx" in r for r in rh_detected)
    # The matching ``coder_failed`` also fires for the all-failures
    # bookkeeping (n_candidates / n_survivors aggregation in the iter).
    coder_failed = [r for r in records if r["kind"] == "coder_failed"]
    assert len(coder_failed) >= 1
    assert any("reward-hack" in r["reason"] for r in coder_failed)
    # No tree child for the failed candidate.
    children = [n for n in result.tree._nodes.values() if n.parent_id is not None]
    assert children == []
    # iter_end SKIPPED — RewardHackDetected is agent-fault so it bumps
    # ``consecutive_agent_failures``; the run still completes cleanly.
    end = next(r for r in records if r["kind"] == "iter_end")
    assert end["outcome"] == "skipped"


# ── reward_hack_suspect (channel B): re-eval cleared / confirmed ──────


@pytest.mark.asyncio
async def test_orchestrator_re_evals_suspect_score_and_accepts_cleared(
    tmp_path, harness,
):
    """When ``score.reward_hack_suspect`` is True and the re-eval clears
    the suspicion, the orchestrator emits ``reward_hack_cleared`` and
    accepts the original score (advanced iteration)."""
    from src.search.orchestrator import Orchestrator

    events_path, close = _capture_events(tmp_path)

    suspect_score = ScoreResult(
        sol_score=0.99,
        baseline_latency_us=100.0,
        candidate_latency_us=40.0,
        t_sol_us=50.0,
        speedup=2.5,
        reward_hack_suspect=True,
        calibration_warning=False,
    )

    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", return_value=harness.bench),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
            patch("src.eval.scorer.compute_sol_score", return_value=suspect_score),
            patch.object(
                Orchestrator,
                "_reward_hack_re_eval",
                AsyncMock(return_value=True),
            ),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder, harness.reviewer,
                harness.retriever,
            )
            await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)
    finally:
        close()

    records = _read_events(events_path)
    kinds = [r["kind"] for r in records]
    assert "reward_hack_cleared" in kinds, kinds
    assert "reward_hack_confirmed" not in kinds
    # iter_end with outcome=advanced (the score is accepted).
    end = next(r for r in records if r["kind"] == "iter_end")
    assert end["outcome"] == "advanced"


@pytest.mark.asyncio
async def test_orchestrator_re_evals_suspect_score_and_marks_dead_on_confirm(
    tmp_path, harness,
):
    """When the re-eval still flags the candidate, ``reward_hack_confirmed``
    fires and the child is DEAD_END."""
    from src.search.orchestrator import Orchestrator

    events_path, close = _capture_events(tmp_path)

    suspect_score = ScoreResult(
        sol_score=0.99,
        baseline_latency_us=100.0,
        candidate_latency_us=40.0,
        t_sol_us=50.0,
        speedup=2.5,
        reward_hack_suspect=True,
        calibration_warning=False,
    )

    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", return_value=harness.bench),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
            patch("src.eval.scorer.compute_sol_score", return_value=suspect_score),
            patch.object(
                Orchestrator,
                "_reward_hack_re_eval",
                AsyncMock(return_value=False),
            ),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder, harness.reviewer,
                harness.retriever,
            )
            await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)
    finally:
        close()

    records = _read_events(events_path)
    kinds = [r["kind"] for r in records]
    assert "reward_hack_confirmed" in kinds, kinds
    assert "reward_hack_cleared" not in kinds
    end = next(r for r in records if r["kind"] == "branch_dead_end")
    assert "reward_hack_confirmed" in end["reason"]


# ── calibration_warning ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_emits_calibration_warning_on_score_flag(tmp_path, harness):
    """``score.calibration_warning`` is plumbed through to the event
    stream so post-run analysis sees the T_k vs T_SOL margin issue."""
    from src.search.orchestrator import Orchestrator

    events_path, close = _capture_events(tmp_path)

    warned_score = ScoreResult(
        sol_score=0.5,
        baseline_latency_us=100.0,
        candidate_latency_us=200.0,
        t_sol_us=50.0,
        speedup=0.5,
        reward_hack_suspect=False,
        calibration_warning=True,
    )

    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", return_value=harness.bench),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
            patch("src.eval.scorer.compute_sol_score", return_value=warned_score),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder, harness.reviewer,
                harness.retriever,
            )
            await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)
    finally:
        close()

    records = _read_events(events_path)
    kinds = [r["kind"] for r in records]
    assert "calibration_warning" in kinds, kinds


# ── CUDA sticky-state recovery ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_recovers_from_transient_cuda_error(tmp_path, harness):
    """A2: a transient CUDA sticky-state error during a candidate's bench
    plus a successful ``torch.cuda.synchronize()`` → per-candidate
    ``coder_failed`` event (with the CUDA-sticky-state reason), iter
    SKIPPED, run continues without ``CUDAContextPoisoned``. The legacy
    ``branch_dead_end`` event used to fire from ``_kill_branch(CUDA_ERROR)``;
    K-way replaces that with the per-candidate ``coder_failed`` path
    since failed candidates no longer enter the tree.
    """
    import torch

    from src.search.orchestrator import Orchestrator

    events_path, close = _capture_events(tmp_path)

    calls = {"n": 0}

    def side_effect(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return harness.bench  # baseline
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", side_effect=side_effect),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
            patch.object(torch.cuda, "synchronize", MagicMock()),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder, harness.reviewer,
                harness.retriever,
            )
            # Should not raise CUDAContextPoisoned.
            await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)
    finally:
        close()

    records = _read_events(events_path)
    coder_failed = [r for r in records if r["kind"] == "coder_failed"]
    assert coder_failed, [r["kind"] for r in records]
    assert any("CUDA sticky-state" in r["reason"] for r in coder_failed)
    end = next(r for r in records if r["kind"] == "iter_end")
    assert end["outcome"] == "skipped"


@pytest.mark.asyncio
async def test_orchestrator_run_fatal_after_three_consecutive_cuda_errors(
    tmp_path, harness,
):
    """3 consecutive synchronize() failures → CUDAContextPoisoned."""
    import torch

    from src.search.orchestrator import CUDAContextPoisoned, Orchestrator

    events_path, close = _capture_events(tmp_path)

    # Use a config with enough depth so we can rack up 3 consecutive
    # CUDA errors before BUDGET kicks in. coder_n_candidates=1 isolates
    # this test from A2's K-way fan-out (otherwise K=4 sticky-CUDA errors
    # within iter 1 would hit the 3-strike CUDAContextPoisoned mid-iter,
    # which would also raise — but the legacy semantic is cross-iter).
    config = ACTSConfig(
        hardware=_rtx6000_ada(),
        max_depth=5,
        beam_width=3,
        sol_plateau_window=99,
        coder_n_candidates=1,
    )

    calls = {"n": 0}

    def side_effect(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return harness.bench  # baseline
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    sync_fail = MagicMock(side_effect=RuntimeError("CUDA error: sync failed"))

    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", side_effect=side_effect),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
            patch.object(torch.cuda, "synchronize", sync_fail),
        ):
            orch = Orchestrator(
                config, harness.planner, harness.coder, harness.reviewer,
                harness.retriever,
            )
            with pytest.raises(CUDAContextPoisoned):
                await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)
    finally:
        close()


# ── trace_emitted ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_emits_trace_emitted_per_evaluation(tmp_path, harness):
    """One ``trace_emitted`` event per advanced iteration."""
    from src.search.orchestrator import Orchestrator

    events_path, close = _capture_events(tmp_path)

    try:
        with (
            patch("src.eval.benchmark.benchmark_kernel", return_value=harness.bench),
            patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
        ):
            orch = Orchestrator(
                harness.config, harness.planner, harness.coder, harness.reviewer,
                harness.retriever,
            )
            await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)
    finally:
        close()

    records = _read_events(events_path)
    kinds = [r["kind"] for r in records]
    assert kinds.count("trace_emitted") == 1, kinds


# ── reward_hack re-eval: multi-output (tuple/dict) shape handling ──────
#
# Invariant: ``_reward_hack_re_eval`` must run the strict tolerance check
# for multi-output kernels (tuple/dict), not just single-tensor outputs.
# Tuple/dict outputs route through the same normalized comparator that
# ``verify_correctness`` uses (``_compare_outputs`` +
# ``_build_normalize_context``); each named output is compared
# name-by-name under the strict tolerance. A bypass that returned True
# unconditionally for multi-output candidates would fail-OPEN: a suspect
# kernel cleared without any output comparison.


def _two_out_definition():
    """Synthetic ``Definition`` with two named outputs.

    Mirrors the multi-output schema used by the correctness integration
    tests — two float32 outputs, one shape axis. The reference
    body is irrelevant to the re-eval (it doesn't exec the source); we
    inject candidate / reference callables directly.
    """
    from sol_execbench.core.data import Definition

    return Definition.model_validate({
        "name": "rh_two_out",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {
            "out1": {"shape": ["N"], "dtype": "float32"},
            "out2": {"shape": ["N"], "dtype": "float32"},
        },
        "reference": "def run(x):\n    return x.relu(), x.tanh()\n",
        "op_type": "elementwise",
    })


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_reward_hack_re_eval_clears_matching_tuple_output(harness):
    """A candidate returning a ``(tensor, tensor)`` tuple matching the
    reference's ``(tensor, tensor)`` tuple must be CLEARED (return True).

    This pins the Option-A fix: instead of falling through on multi-output
    shapes, the re-eval normalizes via SOL's ``normalize_outputs`` and
    compares each named output. A correct multi-output kernel still gets
    cleared.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for tuple-output re-eval")

    from src.kernels.compiler import CompilationResult
    from src.search.orchestrator import Orchestrator

    def reference_fn(x):
        return x.relu(), x.tanh()

    def candidate_fn(x):
        return x.relu(), x.tanh()

    def input_generator(seed: int) -> tuple:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return (torch.randn(64, generator=gen, device="cuda"),)

    orch = Orchestrator(
        harness.config, harness.planner, harness.coder, harness.reviewer,
        harness.retriever,
    )
    child = SimpleNamespace(id="c1")
    workloads = [SimpleNamespace(name="w0")]

    fake_compiled = CompilationResult(success=True, compiled_fn=candidate_fn)

    with patch(
        "src.kernels.compiler.compile_kernel",
        return_value=fake_compiled,
    ):
        cleared = await orch._reward_hack_re_eval(
            child,
            harness.baseline,
            workloads,
            [input_generator],
            reference_fn=reference_fn,
            definition=_two_out_definition(),
        )

    assert cleared is True, (
        "matching multi-output tuple should be cleared by the re-eval; "
        "got fail-closed instead — the normalized comparator path is broken"
    )


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_reward_hack_re_eval_rejects_mismatched_tuple_output(harness):
    """A candidate returning a ``(tensor, tensor)`` tuple where ONE element
    diverges from the reference must NOT be cleared (return False).

    Pre-fix bug: the function returned True for any non-single-tensor
    output without comparing — the suspect kernel was fail-OPEN-cleared.
    Post-fix: the normalized per-name comparator catches the mismatched
    output and fail-CLOSES.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for tuple-output re-eval")

    from src.kernels.compiler import CompilationResult
    from src.search.orchestrator import Orchestrator

    def reference_fn(x):
        return x.relu(), x.tanh()

    def candidate_fn(x):
        # First output correct, second output diverges (sin instead of tanh).
        return x.relu(), x.sin()

    def input_generator(seed: int) -> tuple:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return (torch.randn(64, generator=gen, device="cuda"),)

    orch = Orchestrator(
        harness.config, harness.planner, harness.coder, harness.reviewer,
        harness.retriever,
    )
    child = SimpleNamespace(id="c2")
    workloads = [SimpleNamespace(name="w0")]

    fake_compiled = CompilationResult(success=True, compiled_fn=candidate_fn)

    with patch(
        "src.kernels.compiler.compile_kernel",
        return_value=fake_compiled,
    ):
        cleared = await orch._reward_hack_re_eval(
            child,
            harness.baseline,
            workloads,
            [input_generator],
            reference_fn=reference_fn,
            definition=_two_out_definition(),
        )

    assert cleared is False, (
        "mismatched multi-output tuple must be fail-closed (not cleared) "
        "by the re-eval — guards against the fail-open hole where any "
        "non-single-tensor output was auto-cleared without comparison"
    )


# ── reward_hack re-eval: DPS kernel call shape ────────────────────────
#
# Invariant: ``_reward_hack_re_eval`` must respect ``kernel.dps`` when
# calling the candidate. DPS kernels expose the signature
# ``kernel_fn(*inputs, *outputs)`` — calling them with only ``*inputs``
# raises TypeError, the catch-all ``except Exception`` returns False, and
# any DPS branch that trips ``reward_hack_suspect`` is auto-confirmed as
# a hack even when its outputs match the reference. The candidate must
# be wrapped per-workload via ``_maybe_wrap_dps_candidate`` so
# ``allocate_outputs`` provisions fresh buffers and the candidate is
# invoked as ``candidate_fn(*inputs, *outputs)``.


def _dps_one_out_definition():
    """Synthetic ``Definition`` with one DPS output for the DPS re-eval tests.

    Single float32 output of shape ``[N]``, matching the relu/tanh kernels
    we use as test candidates (single-input → single-output).
    """
    from sol_execbench.core.data import Definition

    return Definition.model_validate({
        "name": "rh_dps_one_out",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x.relu()\n",
        "op_type": "elementwise",
    })


def _dps_workload(n: int = 64):
    """Return a real ``Workload`` resolving ``N=n`` so
    ``get_resolved_axes_values`` produces a concrete shape for
    ``allocate_outputs``."""
    from sol_execbench.core.data import Workload

    return Workload.model_validate({
        "uuid": "w-dps-0",
        "axes": {"N": n},
        "inputs": {"x": {"type": "random"}},
    })


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_reward_hack_re_eval_dps_clears_matching_kernel(harness):
    """DPS candidate writing into pre-allocated output buffer must be CLEARED.

    Regression: an unwrapped ``cand_fn(*inputs)`` call would raise
    TypeError (the DPS host wrapper expects ``kernel_fn(x, out)``); the
    catch-all ``except Exception`` would fail-closed and return False
    even though the kernel was correct, incorrectly confirming the
    branch as a hack.

    Contract: ``_maybe_wrap_dps_candidate`` allocates the output buffer,
    calls ``kernel_fn(x, out)``, and the normalized comparator sees
    ``out`` matching ``reference_fn(x)`` — cleared.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for DPS re-eval")

    from src.kernels.compiler import CompilationResult
    from src.search.orchestrator import Orchestrator

    def reference_fn(x):
        return x.relu()

    def candidate_fn(x, out):
        # DPS host wrapper: write result into pre-allocated `out`. No return.
        out.copy_(x.relu())

    def input_generator(seed: int) -> tuple:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return (torch.randn(64, generator=gen, device="cuda"),)

    dps_kernel = Kernel(
        spec=KernelSpec(
            name="dps_relu",
            kernel_type=KernelType.ELEMENTWISE,
            flop_count=64,
            memory_bytes=64 * 4 * 2,
        ),
        source_code="# dps relu",
        dps=True,
    )

    orch = Orchestrator(
        harness.config, harness.planner, harness.coder, harness.reviewer,
        harness.retriever,
    )
    child = SimpleNamespace(id="dps-clear")
    workloads = [_dps_workload(n=64)]

    fake_compiled = CompilationResult(success=True, compiled_fn=candidate_fn)

    with patch(
        "src.kernels.compiler.compile_kernel",
        return_value=fake_compiled,
    ):
        cleared = await orch._reward_hack_re_eval(
            child,
            dps_kernel,
            workloads,
            [input_generator],
            reference_fn=reference_fn,
            definition=_dps_one_out_definition(),
        )

    assert cleared is True, (
        "DPS candidate matching the reference must be cleared. Without "
        "the DPS wrap, ``cand_fn(*inputs)`` raises TypeError and the "
        "fail-closed branch returns False — auto-confirming a valid "
        "kernel as a reward hack."
    )


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_reward_hack_re_eval_dps_rejects_mismatched_kernel(harness):
    """DPS candidate computing a different op (tanh vs reference relu) must
    NOT be cleared — the fix must not accidentally let real hacks through.

    With the wrapper in place, the kernel runs successfully (no TypeError)
    and the strict-tolerance comparator catches the divergence; fail-closed.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for DPS re-eval")

    from src.kernels.compiler import CompilationResult
    from src.search.orchestrator import Orchestrator

    def reference_fn(x):
        return x.relu()

    def candidate_fn(x, out):
        # Wrong op: writes tanh into the DPS output buffer.
        out.copy_(x.tanh())

    def input_generator(seed: int) -> tuple:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return (torch.randn(64, generator=gen, device="cuda"),)

    dps_kernel = Kernel(
        spec=KernelSpec(
            name="dps_tanh_pretending_to_be_relu",
            kernel_type=KernelType.ELEMENTWISE,
            flop_count=64,
            memory_bytes=64 * 4 * 2,
        ),
        source_code="# dps tanh masquerading",
        dps=True,
    )

    orch = Orchestrator(
        harness.config, harness.planner, harness.coder, harness.reviewer,
        harness.retriever,
    )
    child = SimpleNamespace(id="dps-reject")
    workloads = [_dps_workload(n=64)]

    fake_compiled = CompilationResult(success=True, compiled_fn=candidate_fn)

    with patch(
        "src.kernels.compiler.compile_kernel",
        return_value=fake_compiled,
    ):
        cleared = await orch._reward_hack_re_eval(
            child,
            dps_kernel,
            workloads,
            [input_generator],
            reference_fn=reference_fn,
            definition=_dps_one_out_definition(),
        )

    assert cleared is False, (
        "DPS candidate diverging from the reference must remain not-cleared "
        "— the wrapper fix must not silence real correctness mismatches."
    )


@pytest.mark.asyncio
async def test_orchestrator_threads_dps_onto_child_kernel(tmp_path, harness):
    """When the Coder's output declares ``dps=True``, the child Kernel
    constructed for the search tree must carry that flag through so the
    benchmark + correctness layers know to allocate output buffers."""
    from src.search.orchestrator import Orchestrator

    harness.coder.implement = AsyncMock(
        return_value=KernelCodeOutput.model_construct(
            source_code="# child source",
            triton_kernel_name="",
            dps=True,
        )
    )

    captured: dict = {}

    def fake_bench(kernel, config, **kwargs):
        captured["dps"] = kernel.dps
        captured["definition"] = kwargs.get("definition")
        return harness.bench

    with (
        patch("src.eval.benchmark.benchmark_kernel", side_effect=fake_bench),
        patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
    ):
        orch = Orchestrator(
            harness.config, harness.planner, harness.coder, harness.reviewer,
            harness.retriever,
        )
        await orch.run(harness.baseline, workloads=None, roofline=harness.roofline)

    assert captured["dps"] is True


@pytest.mark.asyncio
async def test_orchestrator_threads_workloads_and_definition_into_coder(
    tmp_path, harness,
):
    """The orchestrator's per-iteration Coder dispatch must forward
    ``workloads`` and ``definition`` so the correctness tool's
    ``verify_correctness`` call sees a non-None workload — that's what
    activates the workload-tolerance override (atol/rtol from
    ``Workload.tolerance``) for stages 1-4 and anti-cheat. Without these
    kwargs, anti-cheat falls back to its hardcoded strict defaults
    (1e-5 / 1e-4) and rejects mathematically correct kernels whose
    workload spec carries looser bounds (e.g. bf16 matmul, atol≈9e-5).
    """
    from sol_execbench.core.data import Definition, Workload
    from src.search.orchestrator import Orchestrator

    definition = Definition.model_validate({
        "name": "noop",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x): return x\n",
    })
    workload = Workload.model_validate(
        {"uuid": "wl0", "axes": {"N": 256}, "inputs": {}}
    )
    # ``total_flops`` / ``total_fused_bytes`` populated so
    # ``compute_roofline_inputs`` early-returns those values without
    # touching shape-formula resolution.
    roofline_with_counts = RooflineResult(
        t_sol_us=harness.roofline.t_sol_us,
        bottleneck=harness.roofline.bottleneck,
        source="solar",
        total_flops=1_000_000,
        total_fused_bytes=100_000,
    )

    with (
        patch("src.eval.benchmark.benchmark_kernel", return_value=harness.bench),
        patch("src.eval.profiler.profile_kernel", return_value=_make_profile()),
    ):
        orch = Orchestrator(
            harness.config, harness.planner, harness.coder, harness.reviewer,
            harness.retriever,
        )
        await orch.run(
            harness.baseline,
            workloads=[workload],
            roofline=roofline_with_counts,
            definition=definition,
        )

    harness.coder.implement.assert_called()
    kwargs = harness.coder.implement.call_args.kwargs
    assert kwargs.get("workloads") == [workload]
    assert kwargs.get("definition") is definition
