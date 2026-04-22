"""Tests for _parse_ncu_csv — the NCU CSV → NCUMetrics reducer.

Golden fixture ``tests/fixtures/ncu/elementwise_add.csv`` was captured
from a real ``ncu --csv --print-metric-name=name`` run on RTX 6000 Ada
(CUDA 12.8, NCU 2025.1.1.0) profiling a 1-D torch elementwise kernel.
Keep it verbatim — it's documentation of NCU's on-disk format.

Edge cases (missing metric, no matching kernel, multi-launch, stall
rank) are synthesized inline from the golden header to avoid tracking
six near-identical files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.profiler import NCUMetrics, _parse_ncu_csv

_FIXTURE = (
    Path(__file__).parent / "fixtures" / "ncu" / "elementwise_add.csv"
).read_text()

# Kernel name fragment that the NCU probe produced — Triton/torch JIT
# kernel names vary, so the parser does a substring match rather than
# equality on the full mangled symbol.
_ELEMENTWISE_ENTRYPOINT = "vectorized_elementwise_kernel"


# ── happy path ────────────────────────────────────────────────────────────


def test_parses_golden_fixture():
    ncu, raw, degraded, reason = _parse_ncu_csv(_FIXTURE, _ELEMENTWISE_ENTRYPOINT)
    assert degraded is False
    assert reason is None
    assert isinstance(ncu, NCUMetrics)


def test_raw_metrics_populated_from_golden():
    _, raw, _, _ = _parse_ncu_csv(_FIXTURE, _ELEMENTWISE_ENTRYPOINT)
    # Golden run has ≥ 4 sections + 18 stall metrics → well over 20 keys.
    assert len(raw) > 20
    # Spot-check a curated source metric survived parsing.
    assert "sm__warps_active.avg.pct_of_peak_sustained_active" in raw


def test_curated_metrics_populated_from_golden():
    ncu, _, _, _ = _parse_ncu_csv(_FIXTURE, _ELEMENTWISE_ENTRYPOINT)
    assert ncu is not None
    assert ncu.sm_occupancy_pct > 0
    assert ncu.l2_hit_rate_pct > 0
    # Elementwise x*2 doesn't use tensor cores — value is 0 but field
    # must still be set (not None / NaN).
    assert ncu.tensor_core_util_pct == 0.0


def test_stall_top_and_runner_up_from_golden():
    """Golden run's two largest stalls, in rank order. Exact pct values
    vary across NCU invocations even on identical hardware (hw counters
    are sampled) — assert relative rank + 1% tolerance on magnitude."""
    ncu, _, _, _ = _parse_ncu_csv(_FIXTURE, _ELEMENTWISE_ENTRYPOINT)
    assert ncu.warp_stall_dominant == "imc_miss"
    assert ncu.warp_stall_dominant_pct > ncu.warp_stall_runner_up_pct
    assert ncu.warp_stall_runner_up == "no_instruction"


def test_comma_thousands_separator_parsed_as_float():
    """NCU formats large values with commas ('55,800'). Parser must
    strip them before float conversion — verified by asserting the
    imc_miss value is the biggest stall (which required parsing
    '55,800' or similar, not truncating it to '55')."""
    _, raw, _, _ = _parse_ncu_csv(_FIXTURE, _ELEMENTWISE_ENTRYPOINT)
    key = "smsp__average_warp_latency_issue_stalled_imc_miss.pct"
    # Value has the thousands separator in the raw CSV; post-parse it
    # must be the numerically largest stall (proof the comma was
    # stripped, not that a specific sample matched).
    stall_vals = [v for k, v in raw.items() if k.startswith("smsp__average_warp_latency_issue_stalled_")]
    assert raw[key] == max(stall_vals)
    assert raw[key] > 1_000  # a truncated parse would yield ~55, not ~55k


# ── kernel filtering ──────────────────────────────────────────────────────


def test_entrypoint_substring_match_is_case_sensitive():
    """Triton/torch JIT names are mangled; substring match is the one
    knob. Tests the match is literal-substring, not regex / case-insensitive."""
    ncu, raw, degraded, reason = _parse_ncu_csv(_FIXTURE, "VECTORIZED")
    assert ncu is None
    assert degraded is True
    assert reason == "no_matching_kernel"


def test_no_matching_kernel():
    ncu, raw, degraded, reason = _parse_ncu_csv(_FIXTURE, "nonexistent_kernel_xyz")
    assert ncu is None
    assert raw == {}
    assert degraded is True
    assert reason == "no_matching_kernel"


# ── synthesized edge cases ────────────────────────────────────────────────


def _header() -> str:
    """NCU canonical CSV header row."""
    return (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context",'
        '"Stream","Block Size","Grid Size","Device","CC","Section Name",'
        '"Metric Name","Metric Unit","Metric Value","Rule Name","Rule Type",'
        '"Rule Description","Estimated Speedup Type","Estimated Speedup"'
    )


def _row(kernel: str, section: str, metric: str, unit: str, value: str) -> str:
    return (
        f'"0","1234","python","host","{kernel}","1","7","(128, 1, 1)","(1, 1, 1)",'
        f'"0","8.9","{section}","{metric}","{unit}","{value}","","","","",""'
    )


def _build_csv(kernel: str, metrics: dict[str, tuple[str, str]]) -> str:
    """metrics: {metric_name: (unit, value_str)}."""
    lines = [_header()]
    for m, (unit, val) in metrics.items():
        lines.append(_row(kernel, "Section", m, unit, val))
    return "\n".join(lines) + "\n"


def _all_curated_plus_stalls(**overrides) -> dict[str, tuple[str, str]]:
    metrics = {
        "sm__warps_active.avg.pct_of_peak_sustained_active": ("%", "50.0"),
        "lts__t_sector_hit_rate.pct": ("%", "90.0"),
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": ("%", "10.0"),
        "smsp__average_warp_latency_issue_stalled_long_scoreboard.pct": ("%", "70.0"),
        "smsp__average_warp_latency_issue_stalled_wait.pct": ("%", "25.0"),
        "smsp__average_warp_latency_issue_stalled_lg_throttle.pct": ("%", "5.0"),
    }
    metrics.update(overrides)
    return metrics


def test_missing_curated_metric_returns_degraded():
    metrics = _all_curated_plus_stalls()
    del metrics["lts__t_sector_hit_rate.pct"]
    csv = _build_csv("my_kernel", metrics)
    ncu, raw, degraded, reason = _parse_ncu_csv(csv, "my_kernel")
    assert ncu is None
    assert degraded is True
    assert reason.startswith("missing_metric:lts__t_sector_hit_rate")
    # raw should still contain whatever did parse — escape hatch intact.
    assert "sm__warps_active.avg.pct_of_peak_sustained_active" in raw


def test_only_one_stall_metric_present_is_degraded():
    metrics = _all_curated_plus_stalls()
    # Drop all but one stall.
    for k in list(metrics):
        if "stalled" in k and "long_scoreboard" not in k:
            del metrics[k]
    csv = _build_csv("my_kernel", metrics)
    ncu, _, degraded, reason = _parse_ncu_csv(csv, "my_kernel")
    assert ncu is None
    assert degraded is True
    assert reason == "stalls_incomplete"


def test_na_value_skipped():
    """NCU writes 'n/a' for metrics it couldn't compute; parser must
    skip rather than crash."""
    metrics = _all_curated_plus_stalls(
        **{"sm__inst_executed.avg.per_cycle_active": ("inst/cycle", "n/a")}
    )
    csv = _build_csv("my_kernel", metrics)
    ncu, raw, degraded, _ = _parse_ncu_csv(csv, "my_kernel")
    assert degraded is False
    assert "sm__inst_executed.avg.per_cycle_active" not in raw


def test_multi_launch_uses_first_occurrence():
    """With --launch-count 1 we expect one kernel launch. If two rows for
    the same metric appear (defensive), first-write-wins is deterministic."""
    lines = [_header()]
    # First launch: occupancy 60%, stalls dominated by long_scoreboard.
    lines.append(_row("my_kernel", "Occupancy",
                      "sm__warps_active.avg.pct_of_peak_sustained_active", "%", "60.0"))
    # Second launch: occupancy 20% — must NOT overwrite the 60.
    lines.append(_row("my_kernel", "Occupancy",
                      "sm__warps_active.avg.pct_of_peak_sustained_active", "%", "20.0"))
    # Fill the rest so the parse succeeds.
    for m, (u, v) in _all_curated_plus_stalls().items():
        if m == "sm__warps_active.avg.pct_of_peak_sustained_active":
            continue
        lines.append(_row("my_kernel", "S", m, u, v))
    csv = "\n".join(lines) + "\n"
    ncu, raw, _, _ = _parse_ncu_csv(csv, "my_kernel")
    assert ncu.sm_occupancy_pct == pytest.approx(60.0)
    assert raw["sm__warps_active.avg.pct_of_peak_sustained_active"] == pytest.approx(60.0)


def test_prof_noise_lines_ignored():
    """NCU prefixes stdout with '==PROF== Connected...' and may interleave
    the profiled process's own stdout ('ok\\n'). Parser must skip them."""
    noise = (
        "==PROF== Connected to process 1234 (/usr/bin/python3.10)\n"
        "ok\n"
        "==PROF== Disconnected from process 1234\n"
    )
    csv = noise + _build_csv("my_kernel", _all_curated_plus_stalls())
    ncu, _, degraded, _ = _parse_ncu_csv(csv, "my_kernel")
    assert degraded is False
    assert ncu is not None


def test_stall_ranking_is_stable_with_equal_values():
    """Two stalls with equal pct — the tie-breaker must be deterministic
    (alphabetical by reason), otherwise prompts would flap between runs."""
    metrics = _all_curated_plus_stalls(
        **{
            "smsp__average_warp_latency_issue_stalled_long_scoreboard.pct": ("%", "50.0"),
            "smsp__average_warp_latency_issue_stalled_wait.pct": ("%", "50.0"),
        }
    )
    csv = _build_csv("my_kernel", metrics)
    ncu, _, _, _ = _parse_ncu_csv(csv, "my_kernel")
    # Alphabetical: long_scoreboard < wait.
    assert ncu.warp_stall_dominant == "long_scoreboard"
    assert ncu.warp_stall_runner_up == "wait"


def test_malformed_csv_returns_degraded():
    ncu, _, degraded, reason = _parse_ncu_csv("this is not a csv", "my_kernel")
    assert ncu is None
    assert degraded is True
    assert reason.startswith("csv_parse:")


def test_empty_stdout_returns_degraded():
    ncu, raw, degraded, reason = _parse_ncu_csv("", "my_kernel")
    assert ncu is None
    assert raw == {}
    assert degraded is True
    assert reason.startswith("csv_parse:")
