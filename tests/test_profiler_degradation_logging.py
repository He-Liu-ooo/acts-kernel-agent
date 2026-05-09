"""Tests for profiler-side diagnostic logging on NCU degradation.

Regression guard against the silent-degradation case: NCU returned a
degraded ``ProfilingResult`` (empty top_stalls, tensor-core util 0.0)
but emitted no log line, so the run could not be triaged from
``run.log`` / ``events.jsonl`` alone — the ``degraded_reason`` slug
existed on the result but never reached a log sink.

Tier 1 (GPU-free). Asserts ``profile_kernel`` writes a WARNING line
carrying the ``degraded_reason`` slug on every degraded return path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from conftest import rtx6000_ada_hardware as _rtx6000_ada
from src.eval import profiler as profiler_mod
from src.eval.profiler import ProfilingResult, profile_kernel
from src.kernels.kernel import Kernel, KernelSpec, KernelType


def _identity_input_generator(seed: int = 0) -> tuple:
    return ()


@pytest.fixture
def sample_kernel() -> Kernel:
    return Kernel(
        spec=KernelSpec(
            name="my_elementwise",
            kernel_type=KernelType.ELEMENTWISE,
            entrypoint="elementwise_add_kernel",
        ),
        source_code=(
            "def elementwise_add_kernel(*args, **kwargs):\n"
            "    return None\n"
        ),
    )


@pytest.fixture
def sample_workload() -> dict:
    return {"uuid": "workload-0", "axes": {"N": 1024}, "inputs": {}}


@pytest.fixture(autouse=True)
def _reset_module_caches(monkeypatch):
    """Clear the module-global NCU caches between tests so prior failure
    state doesn't leak."""
    monkeypatch.setattr(
        profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False
    )
    profiler_mod._reset_ncu_permission_cache()
    monkeypatch.delenv(profiler_mod._NCU_DISABLE_ENV, raising=False)


def _stub_compile(monkeypatch, tmp_path):
    """Make ``compile_kernel`` succeed without triton — degraded paths
    after the compile step still need to reach the log assertion."""
    from src.kernels.compiler import CompilationResult

    fake_source = tmp_path / "fake_compiled.py"
    fake_source.write_text("# stub\n")

    def fake_compile_kernel(kernel, cache_dir=None):
        return CompilationResult(
            success=True,
            compiled_fn=lambda *a, **kw: None,
            source_path=fake_source,
        )

    monkeypatch.setattr(profiler_mod, "compile_kernel", fake_compile_kernel)


def _profile(kernel, workload):
    return profile_kernel(
        kernel,
        workload,
        _identity_input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=1_000_000,
        nbytes=4_000_000,
        latency_s=1e-3,
        mode="curated",
        timeout_s=10.0,
        cache_dir=None,
    )


def test_run_ncu_degraded_logs_reason(
    monkeypatch, tmp_path, sample_kernel, sample_workload, caplog
):
    """When ``_run_ncu`` returns degraded, ``profile_kernel`` must emit a
    WARNING-level log line carrying the ``degraded_reason`` slug. This is
    the primary triage signal for silent-NCU cases — without it,
    ``run.log`` shows only the orchestrator's bare ``profile_done`` event
    (top_stalls=[], tc_util=0.0) which is indistinguishable from a real
    no-stall kernel.
    """
    _stub_compile(monkeypatch, tmp_path)
    monkeypatch.setattr(profiler_mod, "_discover_ncu_binary", lambda: "/usr/bin/ncu")

    reason = "ncu_skipped:permanently_unavailable:nvgpuctrperm"

    def fake_run_ncu(*args, **kwargs):
        return "", -1, True, reason

    monkeypatch.setattr(profiler_mod, "_run_ncu", fake_run_ncu)

    with caplog.at_level(logging.WARNING, logger="src.eval.profiler"):
        result = _profile(sample_kernel, sample_workload)

    assert isinstance(result, ProfilingResult)
    assert result.degraded is True
    assert result.degraded_reason == reason

    # The slug must appear in at least one WARNING-level record so the
    # run.log line is greppable post-mortem.
    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "src.eval.profiler"
    ]
    assert warnings, (
        "profile_kernel returned degraded but emitted no WARNING log; "
        "silent-NCU runs cannot be triaged from run.log alone"
    )
    assert any(reason in r.getMessage() for r in warnings), (
        f"no WARNING record carried the slug {reason!r}; "
        f"got messages: {[r.getMessage() for r in warnings]}"
    )


def test_parser_degraded_logs_reason(
    monkeypatch, tmp_path, sample_kernel, sample_workload, caplog
):
    """When ``_run_ncu`` succeeds but ``_parse_ncu_csv`` degrades (e.g.
    ``no_matching_kernel``, ``stalls_incomplete``), the parser-side
    ``degraded_reason`` must also surface as a WARNING. Same triage
    signal as the subprocess path."""
    _stub_compile(monkeypatch, tmp_path)
    monkeypatch.setattr(profiler_mod, "_discover_ncu_binary", lambda: "/usr/bin/ncu")

    # Subprocess succeeds with stdout that has no matching kernel rows.
    def fake_run_ncu(*args, **kwargs):
        return (
            '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"0","other_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","%","50"\n',
            0,
            False,
            None,
        )

    monkeypatch.setattr(profiler_mod, "_run_ncu", fake_run_ncu)

    with caplog.at_level(logging.WARNING, logger="src.eval.profiler"):
        result = _profile(sample_kernel, sample_workload)

    assert result.degraded is True
    assert result.degraded_reason == "no_matching_kernel"

    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "src.eval.profiler"
    ]
    assert any("no_matching_kernel" in r.getMessage() for r in warnings), (
        f"parser-degraded path emitted no WARNING with slug; "
        f"got messages: {[r.getMessage() for r in warnings]}"
    )


def test_disabled_via_env_logs_reason(
    monkeypatch, tmp_path, sample_kernel, sample_workload, caplog
):
    """``ACTS_DISABLE_NCU=1`` short-circuits before subprocess fork; the
    ``ncu_disabled_via_env`` slug must still surface as a WARNING so
    operators see "I asked for NCU off" reflected in run.log."""
    monkeypatch.setenv(profiler_mod._NCU_DISABLE_ENV, "1")

    with caplog.at_level(logging.WARNING, logger="src.eval.profiler"):
        result = _profile(sample_kernel, sample_workload)

    assert result.degraded is True
    assert result.degraded_reason == "ncu_disabled_via_env"

    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "src.eval.profiler"
    ]
    assert any("ncu_disabled_via_env" in r.getMessage() for r in warnings)


def test_happy_path_does_not_log_warning(
    monkeypatch, tmp_path, sample_kernel, sample_workload, caplog
):
    """No-false-positive guard: a clean NCU run must NOT emit a WARNING.
    The diagnostic line is for degraded cases only."""
    _stub_compile(monkeypatch, tmp_path)
    monkeypatch.setattr(profiler_mod, "_discover_ncu_binary", lambda: "/usr/bin/ncu")

    canned_csv = (
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","elementwise_add_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","%","55.0"\n'
        '"0","elementwise_add_kernel","lts__t_sector_hit_rate.pct","%","72.5"\n'
    )
    # Add the 18 stall rows so the parser doesn't degrade with stalls_incomplete.
    for i, reason in enumerate((
        "barrier", "branch_resolving", "dispatch_stall", "drain", "imc_miss",
        "lg_throttle", "long_scoreboard", "math_pipe_throttle", "membar",
        "mio_throttle", "misc", "no_instruction", "not_selected", "selected",
        "short_scoreboard", "sleeping", "tex_throttle", "wait",
    )):
        canned_csv += (
            f'"0","elementwise_add_kernel","smsp__average_warp_latency_issue_stalled_{reason}.pct","%","{i + 1}"\n'
        )

    def fake_run_ncu(*args, **kwargs):
        return canned_csv, 0, False, None

    monkeypatch.setattr(profiler_mod, "_run_ncu", fake_run_ncu)

    with caplog.at_level(logging.WARNING, logger="src.eval.profiler"):
        result = _profile(sample_kernel, sample_workload)

    assert result.degraded is False
    assert result.ncu is not None

    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "src.eval.profiler"
    ]
    assert not warnings, (
        f"happy path emitted unexpected WARNING(s): "
        f"{[r.getMessage() for r in warnings]}"
    )
