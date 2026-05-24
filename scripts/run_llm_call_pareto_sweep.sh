#!/usr/bin/env bash
# LLM-call Pareto experiment — sweep driver.
# Iterates 10 regime cfgs × 3 reps and runs each via the standard
# `python -m src.pipeline.optimize` entry point. Resumable: skips any
# (regime, rep) whose run_dir already contains a completed report.txt.
#
# Usage:
#   scripts/run_llm_call_pareto_sweep.sh
#
# Outputs to runs/sweep_l1_048/ — see
# doc/specs/2026-05-19-llm-call-pareto-experiment-design.md § 6.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${ACTS_VENV:-$HOME/.venvs/acts_run_venv}"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "venv not found at $VENV — rebuild from configs/venvs/3.12.md" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

CFG_DIR="configs/experiments/llm_call_pareto_l1_048"
SWEEP_DIR="runs/sweep_l1_048"
MANIFEST="$SWEEP_DIR/sweep_manifest.txt"
FAILURES="$SWEEP_DIR/failures.tsv"
# Rep indices to run: [REP_START, REP_START + REPS).
# Defaults run rep_1 and rep_2; override via env to widen or shift.
REP_START="${REP_START:-1}"
REPS="${REPS:-2}"

mkdir -p "$SWEEP_DIR"

# Sweep manifest: record git SHA + GPU snapshot once per sweep invocation
# so reruns under different repo state or different GPU state stay
# distinguishable in analysis.
{
    echo "=== sweep_manifest ==="
    echo "started_at_utc:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_sha:         $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status:      $(git -C "$REPO_ROOT" status --short 2>/dev/null | tr '\n' ';')"
    echo "venv:            $VENV"
    echo "rep_start:       $REP_START"
    echo "reps_per_cell:   $REPS"
    echo
    echo "=== cfg list ==="
    ls -1 "$CFG_DIR"/*.cfg
    echo
    echo "=== nvidia-smi -q (truncated) ==="
    nvidia-smi -q 2>/dev/null | head -40 || echo "nvidia-smi unavailable"
} >> "$MANIFEST"

# Ensure failures.tsv has a header (once)
if [[ ! -s "$FAILURES" ]]; then
    printf "regime\trep\treason\twallclock_s\n" > "$FAILURES"
fi

total_ok=0
total_skip=0
total_fail=0

# Explicit regime allowlist: two-digit ids the sweep should run.
# Override via env: REGIME_ALLOWLIST="03 04 05".
REGIME_ALLOWLIST="${REGIME_ALLOWLIST:-03 04 05 06 07 08 09 10}"

for cfg in "$CFG_DIR"/*.cfg; do
    regime="$(basename "$cfg" .cfg)"   # e.g. regime_03_default
    regime_num="${regime#regime_}"
    regime_num="${regime_num%%_*}"
    if [[ " $REGIME_ALLOWLIST " != *" $regime_num "* ]]; then
        echo "[filter] $regime — not in allowlist ($REGIME_ALLOWLIST)"
        continue
    fi
    for ((rep=REP_START; rep<REP_START+REPS; rep++)); do
        run_dir="$SWEEP_DIR/$regime/rep_$rep"
        report="$run_dir/report.txt"

        if [[ -f "$report" ]]; then
            echo "[skip] $regime rep_$rep — report.txt present"
            total_skip=$((total_skip + 1))
            continue
        fi

        mkdir -p "$run_dir"
        echo "[run]  $regime rep_$rep → $run_dir"
        start_s=$SECONDS
        # `set +e` for this region so a failed run doesn't abort the sweep.
        set +e
        python -m src.pipeline.optimize \
            --config "$cfg" \
            --run-dir "$run_dir" \
            > "$run_dir/sweep_stdout.log" 2> "$run_dir/sweep_stderr.log"
        rc=$?
        set -e
        elapsed=$((SECONDS - start_s))

        if [[ -f "$report" ]]; then
            echo "[ok]   $regime rep_$rep — ${elapsed}s (exit ${rc})"
            total_ok=$((total_ok + 1))
        else
            reason="exit_${rc}_no_report"
            echo "[fail] $regime rep_$rep — ${reason} after ${elapsed}s"
            printf "%s\t%d\t%s\t%d\n" "$regime" "$rep" "$reason" "$elapsed" >> "$FAILURES"
            total_fail=$((total_fail + 1))
        fi
    done
done

echo
echo "=== sweep complete ==="
echo "ok:    $total_ok"
echo "skip:  $total_skip"
echo "fail:  $total_fail"
echo "manifest: $MANIFEST"
echo "failures: $FAILURES"
