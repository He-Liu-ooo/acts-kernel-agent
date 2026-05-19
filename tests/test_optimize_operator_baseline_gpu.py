"""Tier 2 GPU smoke test for operator-supplied Triton baseline.

Runs the full ``python -m src.pipeline.optimize`` flow against the
configs/smoke_operator_baseline.cfg setup. Asserts:
  (a) run completes without raising,
  (b) the root-node kernel source matches the operator-supplied fixture
      verbatim (proves the operator path was taken, not Coder),
  (c) Phase C report renders to <run_dir>/report.txt.

Requires ~/.venvs/acts_run_venv (Python 3.12 + cu128 torch + SOLAR +
openai-agents). Marked @pytest.mark.gpu so Tier 1 sweeps skip it.
See doc/specs/2026-05-16-operator-supplied-triton-baseline-design.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
CFG = REPO_ROOT / "configs" / "smoke_operator_baseline.cfg"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "operator_baselines" / "smoke_rmsnorm.py"


@pytest.mark.gpu
def test_e2e_operator_baseline_smoke(tmp_path):
    """End-to-end smoke: operator-supplied baseline path completes."""
    assert CFG.exists(), f"smoke cfg missing: {CFG}"
    assert FIXTURE.exists(), f"smoke fixture missing: {FIXTURE}"

    run_root = tmp_path / "runs"
    cmd = [
        sys.executable, "-m", "src.pipeline.optimize",
        "--config", str(CFG),
        "--run-dir", str(run_root),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"optimize.py exited {result.returncode}\n"
        f"STDOUT (last 2000):\n{result.stdout[-2000:]}\n"
        f"STDERR (last 2000):\n{result.stderr[-2000:]}"
    )

    # --run-dir is a root; the actual run lives in <root>/run_<UTC>/.
    run_dirs = sorted(run_root.glob("run_*"))
    assert run_dirs, f"no run_* subdir under {run_root}"
    run_dir = run_dirs[-1]

    # (b) root-node kernel source matches the operator fixture verbatim.
    tree_dir = run_dir / "tree"
    assert tree_dir.exists(), f"tree/ missing under {run_dir}"
    node_dirs = sorted(tree_dir.glob("node_*"))
    assert node_dirs, f"no tree/node_* under {tree_dir}"
    # node_0 is the root (baseline).
    root_kernel = node_dirs[0] / "kernel.py"
    assert root_kernel.exists(), f"kernel.py missing in root node {node_dirs[0]}"
    assert root_kernel.read_text() == FIXTURE.read_text(), (
        "baseline source diverged from operator fixture — operator path "
        "may not have been taken"
    )

    # (c) Phase C report rendered.
    report = run_dir / "report.txt"
    assert report.exists(), f"report.txt missing under {run_dir}"
