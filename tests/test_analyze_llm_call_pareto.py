"""Tests for scripts/analyze_llm_call_pareto.py — parsing logic only.

The analysis script imports from `scripts.analyze_llm_call_pareto`; we
test the pure-Python parsing helpers against synthetic fixtures so that
the schema-coupling failure mode (a future change to report.txt or
usage.json silently desynchronizing the analyzer) gets caught here
instead of in the experiment's analysis pipeline.

Plot rendering is NOT tested — it's matplotlib glue and the structure
emerges from the dataframe; we test the dataframe instead.
"""

import json
import sys
from pathlib import Path

import pytest


# Make `scripts/` importable as a package so `from scripts.analyze_llm_call_pareto
# import …` resolves. This mirrors the repo's bare-script layout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_USAGE_JSON_FIXTURE = {
    "schema_version": 1,
    "columns": ["planner", "coder", "coder-translate", "reviewer"],
    "by_iter": [],
    "by_agent": {
        "planner": {
            "invocations": 5, "turns": 12,
            "input_tokens": 10000, "output_tokens": 2000,
            "cached_input_tokens": 0, "reasoning_output_tokens": 0,
        },
        "coder": {
            "invocations": 20, "turns": 80,
            "input_tokens": 200000, "output_tokens": 40000,
            "cached_input_tokens": 0, "reasoning_output_tokens": 0,
        },
        "coder-translate": {
            "invocations": 0, "turns": 0,
            "input_tokens": 0, "output_tokens": 0,
            "cached_input_tokens": 0, "reasoning_output_tokens": 0,
        },
        "reviewer": {
            "invocations": 5, "turns": 15,
            "input_tokens": 30000, "output_tokens": 5000,
            "cached_input_tokens": 0, "reasoning_output_tokens": 0,
        },
    },
    "total": {
        "invocations": 30, "turns": 107,
        "input_tokens": 240000, "output_tokens": 47000,
        "cached_input_tokens": 0, "reasoning_output_tokens": 0,
    },
}

_REPORT_TXT_FIXTURE = """Search completed: plateau
  Iterations: 5
  Baseline:  3958.27 us
  Best:      2500.00 us
  SOL score: 0.7321  (headroom 26.8%)
  Speedup:   1.58x
  Trace: t1_occupancy → t2_shared_memory_tiling
  Bottleneck (run): compute_bound
"""


def _write_rep_fixture(rep_dir: Path, *, sol_score: float, runtime_us: float,
                       planner_turns: int = 12, coder_turns: int = 80,
                       reviewer_turns: int = 15) -> None:
    """Write a synthetic (usage.json, report.txt) pair into rep_dir."""
    rep_dir.mkdir(parents=True, exist_ok=True)
    usage = json.loads(json.dumps(_USAGE_JSON_FIXTURE))  # deep copy
    usage["by_agent"]["planner"]["turns"] = planner_turns
    usage["by_agent"]["coder"]["turns"] = coder_turns
    usage["by_agent"]["reviewer"]["turns"] = reviewer_turns
    usage["total"]["turns"] = planner_turns + coder_turns + reviewer_turns
    (rep_dir / "usage.json").write_text(json.dumps(usage))
    report = _REPORT_TXT_FIXTURE.replace(
        "SOL score: 0.7321", f"SOL score: {sol_score:.4f}",
    ).replace(
        "Best:      2500.00 us", f"Best:      {runtime_us:.2f} us",
    )
    (rep_dir / "report.txt").write_text(report)


# ── parse_report_txt ───────────────────────────────────────────────────


def test_parse_report_txt_extracts_sol_and_runtime(tmp_path):
    from scripts.analyze_llm_call_pareto import parse_report_txt

    (tmp_path / "report.txt").write_text(_REPORT_TXT_FIXTURE)
    parsed = parse_report_txt(tmp_path / "report.txt")
    assert parsed["sol_score_best"] == pytest.approx(0.7321)
    assert parsed["runtime_us_best"] == pytest.approx(2500.0)
    assert parsed["iterations"] == 5
    assert parsed["termination_reason"] == "plateau"


def test_parse_report_txt_missing_sol_returns_none(tmp_path):
    """Malformed/incomplete report.txt — sol/runtime are None, run is
    marked not-completed."""
    from scripts.analyze_llm_call_pareto import parse_report_txt

    (tmp_path / "report.txt").write_text("Run died before scoring\n")
    parsed = parse_report_txt(tmp_path / "report.txt")
    assert parsed["sol_score_best"] is None
    assert parsed["runtime_us_best"] is None


# ── parse_usage_json ───────────────────────────────────────────────────


def test_parse_usage_json_extracts_per_agent_turns(tmp_path):
    from scripts.analyze_llm_call_pareto import parse_usage_json

    (tmp_path / "usage.json").write_text(json.dumps(_USAGE_JSON_FIXTURE))
    parsed = parse_usage_json(tmp_path / "usage.json")
    # "Realized LLM calls" is operationalized as turns (each turn is one
    # generation span). Invocations would undercount multi-turn agents.
    assert parsed["realized_calls_planner"] == 12
    assert parsed["realized_calls_coder"] == 80
    assert parsed["realized_calls_reviewer"] == 15
    assert parsed["realized_calls_total"] == 107


def test_parse_usage_json_missing_file_returns_zeros(tmp_path):
    from scripts.analyze_llm_call_pareto import parse_usage_json

    parsed = parse_usage_json(tmp_path / "nonexistent.json")
    assert parsed["realized_calls_total"] == 0


# ── build_dataframe end-to-end ─────────────────────────────────────────


def test_build_dataframe_walks_sweep_dir(tmp_path):
    from scripts.analyze_llm_call_pareto import build_dataframe

    # Two regimes, two reps each. One rep is incomplete (no report.txt).
    _write_rep_fixture(
        tmp_path / "regime_03_default" / "rep_0",
        sol_score=0.51, runtime_us=3871.74,
    )
    _write_rep_fixture(
        tmp_path / "regime_03_default" / "rep_1",
        sol_score=0.55, runtime_us=3700.00,
    )
    _write_rep_fixture(
        tmp_path / "regime_07_maxed" / "rep_0",
        sol_score=0.62, runtime_us=3200.00,
        coder_turns=160, planner_turns=24, reviewer_turns=40,
    )
    # Incomplete rep — directory exists with only usage.json.
    incomplete = tmp_path / "regime_07_maxed" / "rep_1"
    incomplete.mkdir(parents=True)
    (incomplete / "usage.json").write_text(json.dumps(_USAGE_JSON_FIXTURE))

    df = build_dataframe(tmp_path)
    assert len(df) == 4
    assert set(df["regime_name"]) == {"regime_03_default", "regime_07_maxed"}
    completed = df[df["completed"]]
    assert len(completed) == 3
    incomplete_rows = df[~df["completed"]]
    assert len(incomplete_rows) == 1
    assert incomplete_rows.iloc[0]["regime_name"] == "regime_07_maxed"
    assert incomplete_rows.iloc[0]["rep_idx"] == 1


def test_build_dataframe_empty_sweep_dir_returns_empty_df(tmp_path):
    """No regime subdirs → empty dataframe, no crash."""
    from scripts.analyze_llm_call_pareto import build_dataframe

    df = build_dataframe(tmp_path)
    assert len(df) == 0
    # Schema columns still present so downstream plotting doesn't crash.
    assert "regime_name" in df.columns
    assert "realized_calls_total" in df.columns
    assert "sol_score_best" in df.columns
