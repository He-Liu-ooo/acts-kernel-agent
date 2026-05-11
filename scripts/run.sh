#!/usr/bin/env bash
# Thin wrapper: activate the Tier 2 venv (acts_run_venv) and run the
# ACTS pipeline. All flags are forwarded to `python -m src.pipeline.optimize`.
#
# Usage:
#   scripts/run.sh                                   # placeholder matmul smoke
#   scripts/run.sh --config configs/example.cfg      # cfg-driven run
#   scripts/run.sh --config my.cfg --run-dir ./out   # custom artifact dir
#
# Problem path, GPU index, and reset-clocks now live in the libconfig
# `.cfg` (parsed by `libconf`) — see configs/example.cfg.

set -euo pipefail

VENV="${ACTS_VENV:-$HOME/.venvs/acts_run_venv}"

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "venv not found at $VENV — rebuild from configs/venvs/3.12.md" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

exec python -m src.pipeline.optimize "$@"
