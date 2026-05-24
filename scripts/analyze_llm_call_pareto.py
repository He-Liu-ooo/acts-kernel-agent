#!/usr/bin/env python
"""LLM-call Pareto experiment — analysis pipeline.

Walks ``runs/sweep_l1_048/`` (or a directory passed via --sweep-dir),
parses per-rep ``usage.json`` + ``report.txt``, builds a pandas
dataframe matching the schema in
``doc/specs/2026-05-19-llm-call-pareto-experiment-design.md`` § 7,
and emits three plots + a summary table to
``<sweep_dir>/analysis/``.

Usage:
    python scripts/analyze_llm_call_pareto.py
    python scripts/analyze_llm_call_pareto.py --sweep-dir runs/sweep_l1_048

Plot output requires matplotlib; the parser helpers are pure-Python
and tested in tests/test_analyze_llm_call_pareto.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ── schema ──────────────────────────────────────────────────────────────

# Column order matches doc/specs/…-design.md § 7 dataframe schema.
_DATAFRAME_COLUMNS = [
    "regime_name", "rep_idx",
    "realized_calls_total",
    "realized_calls_planner",
    "realized_calls_coder",
    "realized_calls_reviewer",
    "sol_score_best", "runtime_us_best",
    "iterations", "termination_reason",
    "completed",
]


# ── single-rep parsers ─────────────────────────────────────────────────

_SOL_RE = re.compile(r"SOL score:\s+([0-9.]+)")
_BEST_RE = re.compile(r"Best:\s+([0-9.]+)\s*us")
_ITER_RE = re.compile(r"Iterations:\s+(\d+)")
_TERM_RE = re.compile(r"Search completed:\s+(\S+)")


def parse_report_txt(path: Path) -> dict[str, Any]:
    """Extract end-of-run metrics from report.txt.

    Returns a dict with sol_score_best / runtime_us_best / iterations /
    termination_reason. Missing fields surface as None so the caller's
    ``completed`` gate can flip cleanly on partial data.
    """
    if not path.exists():
        return {
            "sol_score_best": None, "runtime_us_best": None,
            "iterations": None, "termination_reason": None,
        }
    text = path.read_text()
    sol_match = _SOL_RE.search(text)
    best_match = _BEST_RE.search(text)
    iter_match = _ITER_RE.search(text)
    term_match = _TERM_RE.search(text)
    return {
        "sol_score_best": float(sol_match.group(1)) if sol_match else None,
        "runtime_us_best": float(best_match.group(1)) if best_match else None,
        "iterations": int(iter_match.group(1)) if iter_match else None,
        "termination_reason": term_match.group(1) if term_match else None,
    }


def parse_usage_json(path: Path) -> dict[str, int]:
    """Extract per-agent and total realized call counts from usage.json.

    "Realized LLM calls" is operationalized as **turns** (each turn is
    one generation span) — `invocations` undercounts multi-turn agents
    because the orchestrator invokes the Planner once per iter but a
    Pydantic-slip retry makes that one invocation into two LLM calls.
    """
    if not path.exists():
        return {
            "realized_calls_total": 0,
            "realized_calls_planner": 0,
            "realized_calls_coder": 0,
            "realized_calls_reviewer": 0,
        }
    data = json.loads(path.read_text())
    by_agent = data.get("by_agent", {})
    total = data.get("total", {})
    return {
        "realized_calls_total": int(total.get("turns", 0)),
        "realized_calls_planner": int(by_agent.get("planner", {}).get("turns", 0)),
        # `coder` + `coder-translate` are tracked separately in usage.json;
        # for end-of-run Pareto we want the per-iter Coder allocation,
        # which is `coder`. The translate-side calls are baseline-gen and
        # don't tradeoff against the per-iter allocation surface.
        "realized_calls_coder": int(by_agent.get("coder", {}).get("turns", 0)),
        "realized_calls_reviewer": int(by_agent.get("reviewer", {}).get("turns", 0)),
    }


# ── dataframe construction ─────────────────────────────────────────────


def _iter_rep_dirs(sweep_dir: Path):
    """Yield (regime_name, rep_idx, rep_dir) for every rep_<N>/ under
    every regime_<NN>_*/ subdir of sweep_dir."""
    for regime_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        if not regime_dir.name.startswith("regime_"):
            continue
        for rep_dir in sorted(p for p in regime_dir.iterdir() if p.is_dir()):
            if not rep_dir.name.startswith("rep_"):
                continue
            try:
                rep_idx = int(rep_dir.name.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            yield regime_dir.name, rep_idx, rep_dir


def build_dataframe(sweep_dir: Path) -> pd.DataFrame:
    """Walk sweep_dir and return one row per rep matching the spec schema."""
    rows: list[dict[str, Any]] = []
    for regime_name, rep_idx, rep_dir in _iter_rep_dirs(sweep_dir):
        report = parse_report_txt(rep_dir / "report.txt")
        usage = parse_usage_json(rep_dir / "usage.json")
        completed = report["sol_score_best"] is not None
        rows.append({
            "regime_name": regime_name,
            "rep_idx": rep_idx,
            **usage,
            **report,
            "completed": completed,
        })
    if not rows:
        return pd.DataFrame(columns=_DATAFRAME_COLUMNS)
    return pd.DataFrame(rows, columns=_DATAFRAME_COLUMNS)


# ── plotting ───────────────────────────────────────────────────────────


def _plot_pareto(df: pd.DataFrame, y_col: str, title: str, out_path: Path,
                 y_label: str, invert_y: bool = False) -> None:
    import matplotlib.pyplot as plt

    completed = df[df["completed"]]
    if completed.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    regimes = sorted(completed["regime_name"].unique())
    cmap = plt.get_cmap("tab10")
    for i, regime in enumerate(regimes):
        sub = completed[completed["regime_name"] == regime]
        ax.scatter(
            sub["realized_calls_total"], sub[y_col],
            label=regime, color=cmap(i % 10), s=60, alpha=0.75,
        )
    ax.set_xlabel("Realized total LLM calls (turns)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if invert_y:
        ax.invert_yaxis()
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_allocation_shares(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    completed = df[df["completed"]]
    if completed.empty:
        return

    agg = completed.groupby("regime_name")[[
        "realized_calls_planner", "realized_calls_coder", "realized_calls_reviewer",
    ]].median()
    totals = agg.sum(axis=1).replace(0, 1)  # avoid div-by-zero on bad data
    shares = agg.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    shares.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Share of realized calls (%)")
    ax.set_title("Per-regime median allocation shares")
    ax.legend(loc="best", fontsize="small")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    completed = df[df["completed"]]
    counts = df.groupby("regime_name").size().rename("reps_attempted")
    completes = completed.groupby("regime_name").size().rename("reps_completed")

    if completed.empty:
        out_path.write_text("No completed runs.\n")
        return

    agg = completed.groupby("regime_name").agg(
        sol_med=("sol_score_best", "median"),
        sol_min=("sol_score_best", "min"),
        sol_max=("sol_score_best", "max"),
        runtime_med=("runtime_us_best", "median"),
        runtime_min=("runtime_us_best", "min"),
        runtime_max=("runtime_us_best", "max"),
        calls_med=("realized_calls_total", "median"),
    )
    agg = agg.join(counts).join(completes)
    agg["completion_rate"] = agg["reps_completed"] / agg["reps_attempted"]

    lines = [
        "# LLM-call Pareto sweep — summary",
        "",
        "| Regime | SOL (med [min, max]) | Runtime µs (med [min, max]) | Realized calls (med) | Completion |",
        "|---|---|---|---:|---:|",
    ]
    for regime, row in agg.sort_index().iterrows():
        lines.append(
            f"| {regime} "
            f"| {row['sol_med']:.4f} [{row['sol_min']:.4f}, {row['sol_max']:.4f}] "
            f"| {row['runtime_med']:.0f} [{row['runtime_min']:.0f}, {row['runtime_max']:.0f}] "
            f"| {row['calls_med']:.0f} "
            f"| {int(row['reps_completed'])}/{int(row['reps_attempted'])} "
            f"({row['completion_rate']:.0%}) |"
        )
    out_path.write_text("\n".join(lines) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=Path,
        default=Path("runs/sweep_l1_048"),
        help="Sweep run directory (default: runs/sweep_l1_048)",
    )
    args = parser.parse_args()

    sweep_dir: Path = args.sweep_dir
    if not sweep_dir.exists():
        raise SystemExit(f"sweep_dir not found: {sweep_dir}")

    out_dir = sweep_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    df = build_dataframe(sweep_dir)
    df.to_csv(out_dir / "df.csv", index=False)

    _plot_pareto(
        df, "sol_score_best",
        title="SOL score vs realized LLM calls (best-of-tree, end of run)",
        out_path=out_dir / "plot_1_sol_pareto.png",
        y_label="SOL score (best-of-tree)",
    )
    _plot_pareto(
        df, "runtime_us_best",
        title="Runtime vs realized LLM calls (best-of-tree, end of run)",
        out_path=out_dir / "plot_2_runtime_pareto.png",
        y_label="Runtime (µs, best-of-tree, lower=better)",
        invert_y=True,
    )
    _plot_allocation_shares(df, out_dir / "plot_3_allocation_shares.png")
    _write_summary_table(df, out_dir / "summary_table.md")

    print(f"Wrote {len(df)} rows to {out_dir / 'df.csv'}")
    print(f"  completed: {int(df['completed'].sum())}")
    print(f"  failed:    {int((~df['completed']).sum())}")
    print(f"  plots:     {out_dir}")


if __name__ == "__main__":
    main()
