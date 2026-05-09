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
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
              failure_reason: str | None = None,
              failure_detail: str | None = None) -> None:
    """Stream-write tree/node_<id>/ for one committed node.

    Writes kernel.py + meta.json unconditionally; ncu.json when
    profiling.raw_metrics is non-empty; ncu.ncu-rep when ncu_rep_src
    points to an existing file. No-op when unbound; never raises.

    When ``failure_reason`` or ``failure_detail`` is non-None, meta.json
    additionally carries a top-level ``failure`` object — used by the
    dead-end branch in the orchestrator. Validation of ``failure_reason``
    against ``DEAD_REASONS`` is the caller's responsibility; this layer
    is pure transport.
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
            failure_reason=failure_reason,
            failure_detail=failure_detail,
        )
        (node_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if node.profiling is not None and node.profiling.raw_metrics:
            (node_dir / "ncu.json").write_text(
                json.dumps(node.profiling.raw_metrics, indent=2)
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
    """Single source of truth for the four meta.json fields that mutate
    after the streamed dump (beam eviction, post-dump scoring, late-attached
    children). Both the initial ``_build_meta`` write and the
    ``finalize_tree`` rewrite read from here so the on-disk shape stays
    consistent if a future late-bound field joins."""
    from src.search.tree import _serialize_per_workload_latency, _serialize_score

    return {
        "branch_quality": _branch_quality_str(node),
        "score": _serialize_score(node.score),
        # Spec §4.4 contract: empty dict (not null) on unmeasured nodes —
        # the shared serializer returns None, so coerce.
        "per_workload_latency_us": _serialize_per_workload_latency(
            node.per_workload_latency_us
        ) or {},
        "children_ids": list(node.children_ids),
    }


def _build_meta(node: "TreeNode", *, iter_no: int,
                failure_reason: str | None = None,
                failure_detail: str | None = None) -> dict:
    """Compose meta.json shape from a TreeNode. See spec §4.4.

    When either ``failure_reason`` or ``failure_detail`` is non-None,
    a top-level ``failure`` object is appended to the result. When
    both are None (the dominant advance-path case), the key is absent.
    """
    analytical = (
        asdict(node.profiling.analytical)
        if node.profiling is not None
        else None
    )
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
    if failure_reason is not None or failure_detail is not None:
        result["failure"] = {"reason": failure_reason, "detail": failure_detail}
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
    return {
        "id": node.id,
        "iter_no": node.iter_no,
        "action": node.action_applied or "baseline",
        "branch_quality": bq,
        "sol_score": sol_score,
        "speedup": speedup,
        "is_best": is_best,
        "is_dead": is_dead,
    }


def finalize_tree(tree) -> None:
    """End-of-run write: tree/{index.json, tree.txt, tree.dot, tree.mmd}.

    Also rewrites each per-node meta.json's late-bound fields
    (``branch_quality``, ``score``, ``per_workload_latency_us``,
    ``children_ids``) from the final tree state, so nodes whose state
    mutated after their streamed dump (beam eviction, root dumped
    pre-scoring, or root dumped before iters attached children) reflect
    the truth on disk. The rewrite preserves every other key (including
    ``failure``) and skips nodes that never streamed a meta.json.

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
        (_root / "tree.dot").write_text(_render_dot(tree, best_id))
        (_root / "tree.mmd").write_text(_render_mermaid(tree, best_id))
    except (OSError, KeyError) as exc:
        logger.warning("tree_dump.finalize_tree failed: %s", exc)


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


# Single source of truth for branch-quality colors, shared by DOT and
# Mermaid renderers. Keys are ``BranchQuality.value`` strings plus two
# pseudo-classes: ``"neutral"`` for null branch_quality (root / not-yet-
# reviewed), ``"best"`` for the run's winning node (overrides any
# branch_quality color).
_BQ_COLORS = {
    "promising": "#c8f7c5",
    "plateau": "#fff4c2",
    "blocked_potential": "#d8d8f0",
    "dead_end": "#f7c8c8",
    "neutral": "#e0e0e0",
    "best": "#88e088",
}


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
        cls = "best" if s["is_best"] else (s["branch_quality"] or "neutral")
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
        cls = "best" if s["is_best"] else (s["branch_quality"] or "neutral")
        action = s["action"] if s["action"] else "baseline"
        score_part = f"SOL={s['sol_score']:.2f}" if s["sol_score"] is not None else "SOL=n/a"
        star = " ★" if s["is_best"] else ""
        bq_part = f"<br/>{s['branch_quality'].upper()}" if s["branch_quality"] else ""
        label = f"iter={s['iter_no']} · {action}{star}<br/>{score_part}{bq_part}"
        lines.append(f'  n{n.id}["{label}"]:::{cls}')
    for n in tree.nodes():
        if n.parent_id is None:
            continue
        lines.append(f"  n{n.parent_id} --> n{n.id}")
    for cls, color in _BQ_COLORS.items():
        lines.append(f"  classDef {cls} fill:{color};")
    return "\n".join(lines) + "\n"
