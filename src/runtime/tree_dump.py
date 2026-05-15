"""Per-node search-tree dump under <run_dir>/tree/.

Module-level state mirrors src/runtime/events.py: a single bound root
directory per process. dump_node and finalize_tree are no-ops when
unbound; they never raise (OSError logged + swallowed) so a tree-dump
hiccup cannot kill a running search.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.runtime.events import DeadReason

if TYPE_CHECKING:
    from src.search.tree import TreeNode

logger = logging.getLogger(__name__)

_root: Path | None = None


def bind(tree_root: Path) -> None:
    """Set the tree-dump root directory. Idempotent. Creates the dir
    (and any missing parents) if absent."""
    global _root
    tree_root.mkdir(parents=True, exist_ok=True)
    _root = tree_root


def unbind() -> None:
    """Clear the bound root. Idempotent."""
    global _root
    _root = None


def is_bound() -> bool:
    """Test helper — returns True iff bind() set a root."""
    return _root is not None


def dump_node(node: "TreeNode", *, iter_no: int,
              ncu_rep_src: Path | None,
              failure_detail: str | None = None) -> None:
    """Stream-write tree/node_<id>/ for one committed node.

    Writes kernel.py + meta.json unconditionally; ncu.json when
    profiling.raw_metrics is non-empty; ncu.ncu-rep when ncu_rep_src
    points to an existing file. No-op when unbound; never raises.

    When ``failure_detail`` is non-None, meta.json carries a top-level
    ``failure_detail`` field — the prose exception text from the kill
    site (CUDA OOM message, missing UUID, etc.). The categorical
    DEAD_END cause lives on ``node.dead_reason`` and surfaces in
    meta.json as the top-level ``dead_reason`` field via
    ``_late_bound_fields``; the two together replace the old
    ``failure: {reason, detail}`` block, which duplicated ``dead_reason``
    on the reason axis.
    """
    if _root is None:
        return
    try:
        node_dir = _root / f"node_{node.id}"
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "kernel.py").write_text(node.kernel.source_code)
        meta = _build_meta(
            node,
            iter_no=iter_no,
            failure_detail=failure_detail,
        )
        (node_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if node.profiling is not None and node.profiling.raw_metrics:
            groups = node.profiling.metric_groups
            if not groups:
                from src.eval.profiler import _build_metric_groups

                groups = _build_metric_groups(node.profiling.raw_metrics)
            (node_dir / "ncu.json").write_text(
                json.dumps(
                    {
                        "raw": node.profiling.raw_metrics,
                        "groups": groups,
                    },
                    indent=2,
                )
            )
        if ncu_rep_src is not None and ncu_rep_src.exists():
            dst = node_dir / "ncu.ncu-rep"
            try:
                # Hardlink for dedup across nodes sharing a source-hash.
                dst.unlink(missing_ok=True)
                dst.hardlink_to(ncu_rep_src)
            except (OSError, NotImplementedError):
                # Cross-device or filesystem without hardlinks — fall
                # back to a plain copy.
                shutil.copy(ncu_rep_src, dst)
    except OSError as exc:
        logger.warning("tree_dump.dump_node failed for node %s: %s",
                       node.id, exc)


def _branch_quality_str(node: "TreeNode") -> str | None:
    """Return ``branch_quality.value`` or None when unset. Single source
    of truth for the lowercased-enum-or-null shape used by both the
    full meta.json builder and the UI summary."""
    from src.agents.reviewer import BranchQuality
    return (
        node.branch_quality.value
        if isinstance(node.branch_quality, BranchQuality)
        else None
    )


def _late_bound_fields(node: "TreeNode") -> dict[str, Any]:
    """Single source of truth for the meta.json fields that mutate
    after the streamed dump (beam eviction, post-dump scoring, late-attached
    children, post-review feedback attach). Both the initial ``_build_meta``
    write and the ``finalize_tree`` rewrite read from here so the on-disk
    shape stays consistent if a future late-bound field joins.

    ``dead_reason`` is late-bound for the same reason as ``branch_quality``:
    beam pruning and the Reviewer-judged path can flip a node's status
    after its initial dump. It's the single source of truth for the
    DEAD_END cause across all three paths (infra-error kills,
    beam-pruning, Reviewer-judged); the kill-site prose lives separately
    as ``failure_detail`` and is only present when the kill site carried
    a dynamic message (exception text, workload-errors string).
    """
    from src.search.tree import (
        _serialize_per_workload_latency,
        _serialize_review_feedback,
        _serialize_score,
    )

    return {
        "branch_quality": _branch_quality_str(node),
        "dead_reason": (
            node.dead_reason.value
            if isinstance(node.dead_reason, DeadReason)
            else None
        ),
        "score": _serialize_score(node.score),
        # Spec §4.4 contract: empty dict (not null) on unmeasured nodes —
        # the shared serializer returns None, so coerce.
        "per_workload_latency_us": _serialize_per_workload_latency(
            node.per_workload_latency_us
        ) or {},
        "children_ids": list(node.children_ids),
        "last_review": _serialize_review_feedback(node.last_review),
    }


def _build_meta(node: "TreeNode", *, iter_no: int,
                failure_detail: str | None = None) -> dict:
    """Compose meta.json shape from a TreeNode. See spec §4.4.

    When ``failure_detail`` is non-None, a top-level ``failure_detail``
    field is appended — the kill-site prose. The categorical DEAD_END
    cause comes in via ``_late_bound_fields(node)["dead_reason"]``.
    """
    # ``asdict(None)`` raises; dump_node only catches OSError.
    analytical = None
    if node.profiling is not None and node.profiling.has_analytical:
        analytical = asdict(node.profiling.analytical)
    result: dict = {
        "id": node.id,
        "parent_id": node.parent_id,
        "depth": node.depth,
        "iter_no": iter_no,
        "action_applied": node.action_applied,
        "analytical": analytical,
        "consecutive_agent_failures": node.consecutive_agent_failures,
        "trace_workflow": "acts_iter",
        **_late_bound_fields(node),
    }
    if failure_detail is not None:
        result["failure_detail"] = failure_detail
    return result


def _node_summary(node: "TreeNode", *, is_best: bool) -> dict:
    """Reduce a TreeNode to a label-dict consumed by index.json + the
    three visualization formatters. Returned shape is stable; callers
    treat ``sol_score`` / ``speedup`` / ``branch_quality`` as optional."""
    from src.agents.reviewer import BranchQuality

    bq = _branch_quality_str(node)
    sol_score = node.score.sol_score if node.score is not None else None
    speedup = node.score.speedup if node.score is not None else None
    is_dead = node.branch_quality == BranchQuality.DEAD_END
    dr = node.dead_reason.value if isinstance(node.dead_reason, DeadReason) else None
    return {
        "id": node.id,
        "iter_no": node.iter_no,
        "action": node.action_applied or "baseline",
        "branch_quality": bq,
        "dead_reason": dr,
        "sol_score": sol_score,
        "speedup": speedup,
        "is_best": is_best,
        "is_dead": is_dead,
    }


def finalize_tree(tree) -> None:
    """End-of-run write: tree/{index.json, tree.txt, tree.dot, tree.mmd,
    tree.preview.md, tree.png (best-effort if Graphviz is installed)}.

    Also rewrites each per-node meta.json's late-bound fields
    (``branch_quality``, ``dead_reason``, ``score``,
    ``per_workload_latency_us``, ``children_ids``, ``last_review``)
    from the final tree state, so nodes whose state
    mutated after their streamed dump (beam eviction, root dumped
    pre-scoring, or root dumped before iters attached children) reflect
    the truth on disk. The rewrite preserves every other key (including
    ``failure_detail``) and skips nodes that never streamed a meta.json.

    No-op when unbound. Never raises."""
    if _root is None:
        return
    try:
        for node in tree.nodes():
            meta_path = _root / f"node_{node.id}" / "meta.json"
            if not meta_path.exists():
                # Node was added but never streamed (e.g., root, or a
                # crash before _kill_branch ran). Skip — finalize_tree
                # is a consistency layer, not a recovery layer.
                continue
            try:
                existing = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                # Corrupted or unreadable. Skip — never raise.
                continue
            fresh = _late_bound_fields(node)
            if all(existing.get(k) == v for k, v in fresh.items()):
                continue  # no rewrite needed
            existing.update(fresh)
            try:
                meta_path.write_text(json.dumps(existing, indent=2))
            except OSError as exc:
                logger.warning(
                    "tree_dump.finalize_tree meta rewrite failed for "
                    "node %s: %s", node.id, exc,
                )

        best_id = tree.best_node().id
        nodes = [
            _node_summary(n, is_best=(n.id == best_id))
            for n in tree.nodes()
        ]
        edges = [
            [n.parent_id, n.id]
            for n in tree.nodes()
            if n.parent_id is not None
        ]
        index = {
            "best_node_id": best_id,
            "total_nodes": len(tree),
            "edges": edges,
            "nodes": nodes,
        }
        (_root / "index.json").write_text(json.dumps(index, indent=2))
        (_root / "tree.txt").write_text(_render_ascii(tree, best_id))
        dot_path = _root / "tree.dot"
        dot_path.write_text(_render_dot(tree, best_id))
        mermaid_src = _render_mermaid(tree, best_id)
        (_root / "tree.mmd").write_text(mermaid_src)
        (_root / "tree.preview.md").write_text(
            f"```mermaid\n{mermaid_src}```\n"
        )
        _render_png_best_effort(dot_path, _root / "tree.png")
    except (OSError, KeyError) as exc:
        logger.warning("tree_dump.finalize_tree failed: %s", exc)


def _render_png_best_effort(dot_path: Path, png_path: Path) -> None:
    """Run ``dot -Tpng <dot_path> -o <png_path>`` when Graphviz is on PATH.

    Best-effort — Graphviz absence is logged at DEBUG (operator may just
    not have it installed on this host) and any subprocess failure is
    logged at WARNING but never raises. The .dot source already shipped;
    the PNG is a convenience render.
    """
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        logger.debug(
            "tree_dump.finalize_tree: dot binary not on PATH — skipping "
            "tree.png render (run scripts/visualize_tree.sh manually if "
            "you want the rendered image).",
        )
        return
    try:
        result = subprocess.run(
            [dot_bin, "-Tpng", str(dot_path), "-o", str(png_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("tree_dump.finalize_tree: dot subprocess failed: %s", exc)
        return
    if result.returncode != 0:
        logger.warning(
            "tree_dump.finalize_tree: dot exited %d (%s)",
            result.returncode,
            (result.stderr or "").strip()[:200],
        )


def _render_ascii(tree, best_id: int) -> str:
    """Depth-first traversal with Unicode box-drawing connectors. See
    spec §7.1 for the format."""
    lines: list[str] = []

    def _label(n_id: int) -> str:
        n = tree.get_node(n_id)
        s = _node_summary(n, is_best=(n_id == best_id))
        score_part = f"SOL={s['sol_score']:.2f}" if s["sol_score"] is not None else "SOL=n/a"
        status_part = ""
        if s["branch_quality"] is not None:
            status_part = f" {s['branch_quality'].upper()}"
        action_part = s["action"] if s["action"] else "baseline"
        iter_part = f"iter={s['iter_no']} " if s["iter_no"] >= 0 else ""
        star = " ★ best" if s["is_best"] else ""
        return f"[{n_id}] {iter_part}{action_part}{status_part}  {score_part}{star}"

    def _walk(n_id: int, prefix: str, is_last: bool, is_root: bool) -> None:
        connector = "" if is_root else ("└── " if is_last else "├── ")
        lines.append(prefix + connector + _label(n_id))
        children = tree.get_node(n_id).children_ids
        new_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")
        for i, child_id in enumerate(children):
            _walk(child_id, new_prefix, i == len(children) - 1, is_root=False)

    if tree.has_node(0):
        _walk(0, "", is_last=True, is_root=True)
    return "\n".join(lines) + "\n"


# Single source of truth for node colors, shared by DOT and Mermaid
# renderers. Keys are ``BranchQuality.value`` strings plus pseudo-classes:
# ``"neutral"`` for null branch_quality (root / not-yet-reviewed),
# ``"best"`` for the run's winning node (overrides any other color),
# and dead-reason-derived classes that refine the generic ``"dead_end"``
# shade into three semantic groups:
#   - ``"dead_beam_pruned"``: node ran fine, lost the beam competition
#     (lightest — score still valid as a final answer).
#   - ``"dead_reviewer_judged"``: node ran fine, Reviewer judged the
#     branch over (medium — score exists but Reviewer says don't promote).
#   - ``"dead_infra_error"``: node never produced a trustworthy score
#     (darkest — CUDA / profiler / bench / reward-hack failures).
# Generic ``"dead_end"`` remains as the legacy-checkpoint fallback when
# ``dead_reason`` is None.
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

# Map ``DeadReason.value`` strings to one of the three dead-* color
# classes above. The only non-infra-error reasons are listed explicitly;
# every other ``DeadReason`` member is bucketed as ``dead_infra_error``.
# A new non-infra category must add itself to ``_NON_INFRA_DEAD`` —
# defaulting to infra-error matches the historical shape of additions
# (every reason added after BEAM_PRUNED / REVIEWER_JUDGED has been an
# infra-error variant).
_NON_INFRA_DEAD: dict[DeadReason, str] = {
    DeadReason.BEAM_PRUNED: "dead_beam_pruned",
    DeadReason.REVIEWER_JUDGED: "dead_reviewer_judged",
}
_DEAD_REASON_CLASS: dict[str, str] = {
    r.value: _NON_INFRA_DEAD.get(r, "dead_infra_error") for r in DeadReason
}


def _node_color_class(summary: dict) -> str:
    """Pick the color class for one node summary.

    Precedence: best > dead-reason refinement > branch_quality >
    neutral. Legacy DEAD_END nodes (no ``dead_reason``) fall back to
    the generic ``dead_end`` shade.
    """
    if summary["is_best"]:
        return "best"
    if summary["is_dead"]:
        dr = summary.get("dead_reason")
        if dr is not None:
            return _DEAD_REASON_CLASS.get(dr, "dead_end")
        return "dead_end"
    return summary["branch_quality"] or "neutral"


def _render_dot(tree, best_id: int) -> str:
    """Graphviz source. See spec §7.2."""
    lines = [
        "digraph search_tree {",
        "  rankdir=TB;",
        '  node [shape=box, style=filled, fontname="monospace"];',
        "",
    ]
    for n in tree.nodes():
        s = _node_summary(n, is_best=(n.id == best_id))
        cls = _node_color_class(s)
        color = _BQ_COLORS[cls]
        score_part = f"SOL={s['sol_score']:.2f}" if s["sol_score"] is not None else ""
        bq_part = s["branch_quality"].upper() if s["branch_quality"] else ""
        star = " ★" if s["is_best"] else ""
        action = s["action"] if s["action"] else "baseline"
        label_lines = [f"iter={s['iter_no']} · {action}{star}"]
        if score_part:
            label_lines.append(score_part)
        if s["speedup"] is not None:
            label_lines.append(f"speedup={s['speedup']:.2f}x")
        if bq_part:
            label_lines.append(bq_part)
        if s.get("dead_reason"):
            label_lines.append(s["dead_reason"].upper())
        label = "\\n".join(label_lines)
        lines.append(f'  n{n.id} [label="{label}", fillcolor="{color}"];')
    lines.append("")
    for n in tree.nodes():
        if n.parent_id is None:
            continue
        action = n.action_applied or "baseline"
        lines.append(f'  n{n.parent_id} -> n{n.id} [label="{action}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _render_mermaid(tree, best_id: int) -> str:
    """Mermaid graph TD. See spec §7.3."""
    lines = ["graph TD"]
    for n in tree.nodes():
        s = _node_summary(n, is_best=(n.id == best_id))
        cls = _node_color_class(s)
        action = s["action"] if s["action"] else "baseline"
        score_part = f"SOL={s['sol_score']:.2f}" if s["sol_score"] is not None else "SOL=n/a"
        star = " ★" if s["is_best"] else ""
        bq_part = f"<br/>{s['branch_quality'].upper()}" if s["branch_quality"] else ""
        dr_part = f"<br/>{s['dead_reason'].upper()}" if s.get("dead_reason") else ""
        label = f"iter={s['iter_no']} · {action}{star}<br/>{score_part}{bq_part}{dr_part}"
        lines.append(f'  n{n.id}["{label}"]:::{cls}')
    for n in tree.nodes():
        if n.parent_id is None:
            continue
        lines.append(f"  n{n.parent_id} --> n{n.id}")
    for cls, color in _BQ_COLORS.items():
        lines.append(f"  classDef {cls} fill:{color};")
    return "\n".join(lines) + "\n"
