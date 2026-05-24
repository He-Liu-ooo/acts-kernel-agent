#!/usr/bin/env python3
"""Re-render tree.dot + tree.png for existing run directories with the
latency-aware label format.

Walks <root> for any directory matching ``*/tree/index.json`` and
regenerates ``tree.dot`` + ``tree.png`` in that ``tree/`` dir, reading
per-node fields from ``tree/node_<id>/meta.json``. Label format matches
``src/runtime/tree_dump._render_dot`` after the 2026-05-20 latency edit.

Usage:
    scripts/rerender_trees.py <root>           # walk + re-render
    scripts/rerender_trees.py <root> --dry-run # list trees, don't write

Idempotent. Skips tree dirs missing index.json or with no node_*/meta.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Color classes copied verbatim from src/runtime/tree_dump.py to keep this
# script self-contained (no torch/sol_execbench import-time tax).
_BQ_COLORS = {
    "promising": "#c8f7c5",
    "plateau": "#fff4c2",
    "blocked_potential": "#d8d8f0",
    "dead_end": "#f7c8c8",
    "dead_beam_pruned": "#fae3e3",
    "dead_reviewer_judged": "#f7c8c8",
    "dead_infra_error": "#d88a8a",
    "neutral": "#e0e0e0",
    "best": "#88e088",
}

_NON_INFRA_DEAD = {
    "beam_pruned": "dead_beam_pruned",
    "reviewer_judged": "dead_reviewer_judged",
}


def _dead_reason_class(dr: str | None) -> str:
    if dr is None:
        return "dead_end"
    return _NON_INFRA_DEAD.get(dr, "dead_infra_error")


def _node_color_class(meta: dict, is_best: bool) -> str:
    if is_best:
        return "best"
    bq = meta.get("branch_quality")
    if bq == "dead_end":
        return _dead_reason_class(meta.get("dead_reason"))
    return bq or "neutral"


def _representative_latency_us(meta: dict) -> float | None:
    """Mirror tree_dump._node_summary's latency-pick logic."""
    score = meta.get("score")
    if isinstance(score, dict) and score.get("candidate_latency_us") is not None:
        return float(score["candidate_latency_us"])
    per_workload = meta.get("per_workload_latency_us") or {}
    if not per_workload:
        return None
    sorted_lats = sorted(float(v) for v in per_workload.values())
    return sorted_lats[len(sorted_lats) // 2]


def _label_for(meta: dict, is_best: bool) -> list[str]:
    iter_no = meta.get("iter_no", -1)
    action = meta.get("action_applied") or "baseline"
    star = " ★" if is_best else ""
    lines = [f"iter={iter_no} · {action}{star}"]

    score = meta.get("score")
    if isinstance(score, dict) and score.get("sol_score") is not None:
        lines.append(f"SOL={score['sol_score']:.2f}")
    if isinstance(score, dict) and score.get("speedup") is not None:
        lines.append(f"speedup={score['speedup']:.2f}x")

    lat_us = _representative_latency_us(meta)
    if lat_us is not None:
        lines.append(f"latency={lat_us:.0f} us")

    bq = meta.get("branch_quality")
    if bq:
        lines.append(bq.upper())
    if meta.get("dead_reason"):
        lines.append(meta["dead_reason"].upper())

    # Failure-summary nodes (carry failure_details, no scoring fields).
    fd = meta.get("failure_details")
    if isinstance(fd, list) and fd:
        lines.append(f"failed × {len(fd)}")

    return lines


def render_dot(tree_dir: Path, index: dict) -> str:
    best_id = index.get("best_node_id")
    out = [
        "digraph search_tree {",
        "  rankdir=TB;",
        '  node [shape=box, style=filled, fontname="monospace"];',
        "",
    ]
    # Use the node ids from index.json's edges + nodes list as the
    # authoritative set (some failure-summary nodes have meta.json in
    # ``node_<id>/`` even though the index excludes them — defensively
    # skip those we can't find on disk, mirroring tree_dump's behavior).
    node_ids: list[int] = [n["id"] for n in index.get("nodes", [])]
    for nid in node_ids:
        meta_path = tree_dir / f"node_{nid}" / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        is_best = (nid == best_id)
        label_lines = _label_for(meta, is_best=is_best)
        label = "\\n".join(label_lines)
        color = _BQ_COLORS[_node_color_class(meta, is_best=is_best)]
        out.append(f'  n{nid} [label="{label}", fillcolor="{color}"];')
    out.append("")
    for parent_id, child_id in index.get("edges", []):
        if parent_id is None:
            continue
        # Pull action from the child's meta for the edge label.
        meta_path = tree_dir / f"node_{child_id}" / "meta.json"
        action = "baseline"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                action = meta.get("action_applied") or "baseline"
            except (json.JSONDecodeError, OSError):
                pass
        out.append(f'  n{parent_id} -> n{child_id} [label="{action}"];')
    out.append("}")
    return "\n".join(out) + "\n"


def render_one(tree_dir: Path, dry_run: bool) -> tuple[str, str]:
    """Re-render one tree dir. Returns (status, message)."""
    index_path = tree_dir / "index.json"
    if not index_path.exists():
        return ("skip", "no index.json")
    try:
        index = json.loads(index_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return ("error", f"index.json unreadable: {exc}")

    if not any(tree_dir.glob("node_*/meta.json")):
        return ("skip", "no per-node meta.json files")

    dot_src = render_dot(tree_dir, index)
    dot_path = tree_dir / "tree.dot"
    png_path = tree_dir / "tree.png"

    if dry_run:
        return ("would-render", f"{len(index.get('nodes', []))} nodes")

    try:
        dot_path.write_text(dot_src)
    except OSError as exc:
        return ("error", f"tree.dot write failed: {exc}")

    dot_bin = shutil.which("dot")
    if dot_bin is None:
        return ("partial", "tree.dot updated; Graphviz `dot` not on PATH")

    try:
        result = subprocess.run(
            [dot_bin, "-Tpng", str(dot_path), "-o", str(png_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ("error", f"dot subprocess failed: {exc}")
    if result.returncode != 0:
        return ("error", f"dot exited {result.returncode}: {(result.stderr or '').strip()[:200]}")
    return ("ok", f"{len(index.get('nodes', []))} nodes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path, help="Directory to walk (e.g. runs/sweep_l1_048).")
    ap.add_argument("--dry-run", action="store_true", help="List tree dirs that would be re-rendered.")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"error: {args.root} does not exist", file=sys.stderr)
        return 2

    tree_dirs = sorted({p.parent for p in args.root.rglob("tree/index.json")})
    if not tree_dirs:
        print(f"no tree/index.json found under {args.root}", file=sys.stderr)
        return 1

    totals = {"ok": 0, "skip": 0, "partial": 0, "error": 0, "would-render": 0}
    for tree_dir in tree_dirs:
        status, msg = render_one(tree_dir, dry_run=args.dry_run)
        totals[status] = totals.get(status, 0) + 1
        rel = tree_dir.relative_to(args.root)
        print(f"[{status:>13}] {rel} — {msg}")

    print()
    summary = ", ".join(f"{k}={v}" for k, v in totals.items() if v)
    print(f"summary: {summary}")
    return 0 if totals.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
