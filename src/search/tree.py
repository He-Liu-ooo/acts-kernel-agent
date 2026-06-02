"""Tree search state management."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.runtime.events import DeadReason

if TYPE_CHECKING:
    from src.agents.reviewer import BranchQuality, ReviewerFeedback
    from src.eval.profiler import ProfilingResult
    from src.eval.scorer import ScoreResult
    from src.kernels.kernel import Kernel


@dataclass
class FailureDetail:
    """One failed K-way Coder candidate collapsed into a failure-summary node.

    Carried by ``TreeNode.failure_details`` (a list) on summary nodes;
    the list has one entry per failed candidate from that iter.
    ``reason`` is truncated upstream (~200 chars) and read verbatim by
    ``render_siblings``' failure-flatten path. ``has_kernel_source``
    mirrors on-disk presence of
    ``tree/node_<id>/cand_<candidate_idx>/kernel.py`` — False on
    turn-exhaust paths where the Coder never submitted a kernel.
    """

    candidate_idx: int
    reason: str
    has_kernel_source: bool


@dataclass
class TreeNode:
    """A node in the search tree representing one kernel version."""

    id: int
    # ``None`` only on turn-exhaust failure nodes (Coder never reached
    # submit_kernel); all other nodes carry the submitted source.
    kernel: "Kernel | None"
    parent_id: int | None = None
    children_ids: list[int] = field(default_factory=list)
    score: ScoreResult | None = None
    branch_quality: BranchQuality | None = None
    action_applied: str = ""
    # Plan parameters that produced this node (e.g., ``{"BLOCK_N": 32}``).
    # Populated at add_child time from ``OptimizationPlan.params``. Read by
    # ``SearchTree.render_siblings`` so the sibling line carries the param
    # set alongside the technique ID — a sibling that tried ``BLOCK_N=32``
    # and one that tried ``BLOCK_N=16`` are not interchangeable evidence.
    # ``None`` on the root and on legacy checkpoints predating this field.
    action_params: dict | None = None
    depth: int = 0
    # Populated after each iteration's profile_kernel call. None for the
    # root (no profile run at baseline construction) and for children
    # whose benchmark failed (see orchestrator dead_end path).
    profiling: ProfilingResult | None = None
    # Per-workload latency (µs) carried from the child's BenchmarkResult.
    # Phase C's winner re-profile reads this to drive each workload's
    # analytical metrics off its *own* latency rather than the aggregate
    # median. ``None`` on the root and on legacy checkpoints predating
    # the field; report.py falls back to the aggregate in that case.
    per_workload_latency_us: dict[str, float] | None = None
    # Aggregate runtime (ms) — synthetic field set by the orchestrator
    # right after the node is scored, derived from
    # ``bench.median_latency_us / 1000``. Read by the opt-mem ``Producer``
    # to compute parent → child speedup ratios. ``None`` on legacy
    # checkpoints, on dead-end nodes that never benched, and on root
    # before baseline benchmarking completes.
    runtime_ms: float | None = None
    # Counts back-to-back Coder/Planner failures on this parent. Reset
    # to zero whenever an iteration successfully spawns a child off this
    # node. ``frontier()`` excludes nodes at or above
    # ``QUARANTINE_THRESHOLD`` so a deterministically-failing parent
    # doesn't burn the whole search budget by being re-selected forever.
    consecutive_agent_failures: int = 0
    # Iteration index (1-based) that produced this node. ``-1`` for the
    # root and for legacy checkpoints predating the field.
    iter_no: int = -1
    # Latest Reviewer feedback for *this* kernel. Populated:
    #   - For root: by the baseline review pass during Phase A setup.
    #   - For children: at the iteration that scored them (after review()).
    # Read by the Planner when this node becomes a parent in some future
    # iteration via _render_review_for_planner. None on legacy checkpoints
    # predating this field.
    last_review: "ReviewerFeedback | None" = None
    # Distinguishes the DEAD_END causes that ``branch_quality`` collapses
    # (see ``SearchTree.best_node`` for the eligibility taxonomy). ``None``
    # on live nodes and legacy checkpoints; paired with
    # ``branch_quality = DEAD_END`` at every kill site via ``mark_dead``.
    dead_reason: "DeadReason | None" = None
    # Per-candidate failure list on failure-summary nodes attached via
    # ``add_failure_summary``. ``None`` on the root, on every live /
    # success node, and on every legacy DEAD_END node that predates
    # failure-node collapse. The list has one ``FailureDetail`` per
    # failed K-way candidate from the iter that produced this node;
    # set together with ``branch_quality=DEAD_END`` and
    # ``dead_reason=CODER_FAILED`` at end-of-iter persistence. Read by
    # ``render_siblings`` (flattened into the FAILED-block accumulator)
    # and by ``tree_dump.dump_failure_summary_node`` (per-candidate
    # ``cand_<idx>/`` disk layout).
    failure_details: list[FailureDetail] | None = None

    def mark_dead(self, reason: "DeadReason") -> None:
        """Mark this node DEAD_END and record the cause atomically.

        Every DEAD_END node must carry a ``dead_reason``; otherwise
        ``_eligible_for_best`` falls back to the legacy-unknown branch
        and excludes the node from ``best_node``. Use this method instead
        of writing ``branch_quality`` + ``dead_reason`` separately.
        """
        from src.agents.reviewer import BranchQuality

        self.branch_quality = BranchQuality.DEAD_END
        self.dead_reason = reason


# A parent that has caused this many consecutive Planner/Coder failures
# is quarantined from ``frontier()``. Two tolerates one transient API
# blip; raising it loosens the guarantee against deterministic failures.
QUARANTINE_THRESHOLD: int = 2


def _format_action_label(action: str, params: dict | None) -> str:
    """Shared formatter for success + failure sibling render lines."""
    if params:
        params_str = "{" + ", ".join(f"{k}:{v}" for k, v in params.items()) + "}"
        return f"{action} {params_str}"
    return action or "(no action)"


class SearchTree:
    """Manages the tree search state: nodes, frontier, and expansion."""

    def __init__(self) -> None:
        self._nodes: dict[int, TreeNode] = {}
        self._next_id: int = 0

    def add_root(self, kernel: Kernel) -> TreeNode:
        """Add the root node (baseline kernel) to the tree."""
        node = TreeNode(id=self._next_id, kernel=kernel, depth=0)
        self._nodes[node.id] = node
        self._next_id += 1
        return node

    def add_child(
        self,
        parent_id: int,
        kernel: Kernel,
        action_applied: str,
        *,
        iter_no: int = -1,
        action_params: dict | None = None,
    ) -> TreeNode:
        """Add a child node resulting from an optimization action."""
        parent = self._nodes[parent_id]
        node = TreeNode(
            id=self._next_id,
            kernel=kernel,
            parent_id=parent_id,
            action_applied=action_applied,
            action_params=action_params,
            depth=parent.depth + 1,
            iter_no=iter_no,
        )
        parent.children_ids.append(node.id)
        self._nodes[node.id] = node
        self._next_id += 1
        return node

    def add_failure_summary(
        self,
        parent_id: int,
        *,
        action_applied: str,
        action_params: dict | None,
        iter_no: int,
        failure_details: list[FailureDetail],
    ) -> TreeNode:
        """Attach one failure-summary node under ``parent_id`` collapsing
        all failed K-way Coder candidates from a single iter into one
        tree entry.

        Node properties: ``kernel=None`` (per-candidate sources live on
        disk under ``tree/node_<id>/cand_<idx>/kernel.py``, not on the
        TreeNode), ``score=None``, ``branch_quality=DEAD_END``,
        ``dead_reason=CODER_FAILED`` (set atomically via ``mark_dead``
        so ``_eligible_for_best`` excludes it). Excluded from
        ``frontier()`` and ``best_node()`` like the legacy per-candidate
        failure nodes were.

        ``failure_details`` MUST be non-empty — callers don't attach an
        empty summary; if the iter had zero failures, no summary node
        is added. ``consecutive_agent_failures`` stays an iter-level
        orchestrator counter — not bumped here.
        """
        if not failure_details:
            raise ValueError("failure_details must be non-empty")
        parent = self._nodes[parent_id]
        node = TreeNode(
            id=self._next_id,
            kernel=None,
            parent_id=parent_id,
            action_applied=action_applied,
            action_params=action_params,
            depth=parent.depth + 1,
            iter_no=iter_no,
            failure_details=list(failure_details),
        )
        node.mark_dead(DeadReason.CODER_FAILED)
        parent.children_ids.append(node.id)
        self._nodes[node.id] = node
        self._next_id += 1
        return node

    def get_node(self, node_id: int) -> TreeNode:
        """Retrieve a node by ID."""
        return self._nodes[node_id]

    def nodes(self):
        """Iterate over every node in the tree (insertion order)."""
        return self._nodes.values()

    def has_node(self, node_id: int) -> bool:
        """Return True iff a node with this ID exists."""
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def frontier(self) -> list[TreeNode]:
        """Return all expandable frontier nodes — neither marked dead_end
        nor quarantined for repeated Planner/Coder failures.
        """
        from src.agents.reviewer import BranchQuality

        return [
            n for n in self._nodes.values()
            if n.branch_quality != BranchQuality.DEAD_END
            and n.consecutive_agent_failures < QUARANTINE_THRESHOLD
        ]

    def best_node(self) -> TreeNode:
        """Return the node with the highest SOL score.

        DEAD_END nodes are filtered by ``dead_reason``, not the flag
        alone. ``branch_quality == DEAD_END`` collapses three distinct
        causes (infra error, Reviewer verdict, beam-pruning eviction),
        but only two of them invalidate the node's measured score:

        - **Infra-error reasons** (CUDA_ERROR, BENCH_FAILURE,
          PROFILER_ERROR, REWARD_HACK*, REPR_LATENCY_UNAVAILABLE,
          AGENT_FAILURE) — the kernel never produced a trustworthy
          measurement. Excluded.
        - **REVIEWER_JUDGED** — the kernel ran fine but the Reviewer
          classified the branch as regressed/over. Excluded so the run's
          winner aligns with the Reviewer's verdict, not a
          Reviewer-rejected score that happens to be numerically best.
        - **BEAM_PRUNED** — the kernel ran fine, the bench measurement
          is valid, the node simply lost the beam competition. Its score
          is just as trustworthy as any live node's, so it stays eligible
          here. Without this carve-out, a high-scoring node evicted at
          iter K is invisible to the winner pick when later iterations
          regress — exactly the silent-slow-ship hazard Codex flagged.

        Legacy DEAD_END nodes (no ``dead_reason`` recorded) are excluded
        as a safe default — we don't know which class they belong to.

        Quarantined nodes (``consecutive_agent_failures >=
        QUARANTINE_THRESHOLD``) are intentionally still candidates here
        — quarantine prevents a deterministically-failing parent from
        being re-selected for further expansion, but its own measured
        score remains a valid final answer if it happens to be the run's
        best.
        """
        scored = [
            n for n in self._nodes.values()
            if n.score is not None and _eligible_for_best(n)
        ]
        if not scored:
            # Fall back to root
            return self._nodes[0]
        return max(scored, key=lambda n: n.score.sol_score)

    def path_to_node(self, node_id: int) -> list[TreeNode]:
        """Return the path from root to the given node."""
        node = self._nodes[node_id]  # KeyError if not found
        path = []
        while True:
            path.append(node)
            if node.parent_id is None:
                break
            node = self._nodes[node.parent_id]
        path.reverse()
        return path

    # ── rendering ────────────────────────────────────────────────────────

    def render_path(self, node_id: int) -> str:
        """Render the root-to-node path as a human-readable trajectory.

        Consumed by the Planner (path-to-parent) and Reviewer
        (path-to-child) so they can reason about which actions have
        already been tried on this branch, rather than seeing only the
        immediate parent's SOL.

        Shape::

            Path (depth D):
              [0] baseline — SOL 0.300
              [1] tiling (PROMISING) — SOL 0.600
              [2] vectorize — SOL 0.800  ← current
        """
        path = self.path_to_node(node_id)
        lines = [f"Path (depth {len(path) - 1}):"]
        for i, node in enumerate(path):
            action = node.action_applied or "baseline"
            sol = f"{node.score.sol_score:.3f}" if node.score is not None else "n/a"
            quality = f" ({node.branch_quality.value.upper()})" if node.branch_quality else ""
            cursor = "  ← current" if i == len(path) - 1 else ""
            lines.append(f"  [{i}] {action}{quality} — SOL {sol}{cursor}")
        return "\n".join(lines)

    def render_siblings(
        self,
        parent_id: int,
        exclude_id: int | None = None,
        *,
        consumer: str = "planner",
        failure_cap: int = 8,
    ) -> str:
        """Render children of *parent_id* (other than *exclude_id*) as one-liners.

        Returns ``""`` when the parent has no qualifying children — callers
        gate the prompt section on truthiness, mirroring the existing
        ``tree_context`` / ``reviewer_feedback`` omission pattern.

        Two consumer modes:

        - ``consumer="planner"`` (default) — success siblings first, then
          failure siblings (FAILED format), deduped on
          ``(action, params, failure_reason)`` with ``×N`` for N≥2.
        - ``consumer="reviewer"`` — success siblings only.

        Success-line format::

            - <action> {<params>}: SOL <score> (Δ <delta>), <outcome>, <branch_quality>

        Failure-line format::

            - <action> {<params>}: FAILED [×N — ]<raw failure_reason>

        ``failure_cap`` (>0) caps the rendered failure lines to the
        most-recent ``cap`` entries by ``(iter_no, candidate_idx)``
        ascending (tail-N). Over-cap renders prepend ``... (M earlier
        failures omitted)``. ``cap = 0`` means uncapped. Success
        siblings are never capped.

        A still-scoring success sibling appears with sentinels rather
        than being skipped — the Planner benefits from knowing the
        action was attempted, even before scoring lands.
        """
        parent = self._nodes[parent_id]
        parent_score = parent.score.sol_score if parent.score is not None else None
        success_lines: list[str] = []
        # Entry: (iter_no, child_id, action, params, reason). child_id is
        # the within-iter tie-breaker since candidate_idx isn't a node field.
        failure_entries: list[tuple[int, int, str, dict | None, str]] = []

        for child_id in parent.children_ids:
            if exclude_id is not None and child_id == exclude_id:
                continue
            child = self._nodes[child_id]
            if child.failure_details is not None:
                # Flatten summary's details so dedup downstream sees
                # one entry per failed candidate.
                for fd in child.failure_details:
                    failure_entries.append((
                        child.iter_no,
                        child.id,
                        child.action_applied,
                        child.action_params,
                        fd.reason,
                    ))
                continue
            action_label = _format_action_label(
                child.action_applied, child.action_params,
            )
            if child.score is not None:
                sol = f"SOL {child.score.sol_score:.3f}"
                if parent_score is not None:
                    delta_str = f"Δ {child.score.sol_score - parent_score:+.3f}"
                else:
                    delta_str = "Δ n/a"
            else:
                sol = "SOL n/a"
                delta_str = "Δ n/a"
            outcome = (
                child.last_review.outcome
                if child.last_review is not None
                else "(no review yet)"
            )
            bq = (
                child.branch_quality.value
                if child.branch_quality is not None
                else "(unscored)"
            )
            success_lines.append(
                f"- {action_label}: {sol} ({delta_str}), {outcome}, {bq}"
            )

        # Reviewer consumer: success only, early return.
        if consumer == "reviewer":
            return "\n".join(success_lines)

        # Planner consumer: dedup failures, apply cap, render.
        def _params_key(p: dict | None) -> tuple:
            # Canonicalize so dicts with different insertion order dedup.
            return () if p is None else tuple(sorted(p.items()))

        # Dedup on (action, params, reason). Each group's ``latest``
        # tracks its newest occurrence so cap-tail keeps recurring
        # signatures the Planner most needs to see; first-occurrence
        # ordering would silently evict them.
        failure_entries.sort(key=lambda e: (e[0], e[1]))
        groups: dict[tuple, dict] = {}
        for iter_no, child_id, action, params, reason in failure_entries:
            key = (action, _params_key(params), reason)
            if key not in groups:
                groups[key] = {
                    "count": 0, "action": action, "params": params,
                    "reason": reason, "latest": (iter_no, child_id),
                }
            groups[key]["count"] += 1
            groups[key]["latest"] = (iter_no, child_id)

        order: list[tuple] = sorted(
            groups.keys(), key=lambda k: groups[k]["latest"],
        )

        # Apply cap (counts rendered lines = unique dedup groups).
        omitted = 0
        if failure_cap and len(order) > failure_cap:
            omitted = len(order) - failure_cap
            order = order[-failure_cap:]

        failure_lines: list[str] = []
        if omitted:
            failure_lines.append(f"... ({omitted} earlier failures omitted)")
        for key in order:
            g = groups[key]
            action_label = _format_action_label(g["action"], g["params"])
            count_suffix = f" ×{g['count']}" if g["count"] >= 2 else ""
            failure_lines.append(
                f"- {action_label}: FAILED{count_suffix} — {g['reason']}"
            )

        return "\n".join(success_lines + failure_lines)

    def regressed_sibling_actions(
        self,
        parent_id: int,
        exclude_id: int | None = None,
    ) -> list[tuple[str, int]]:
        """Return ``[(action_applied, iter_no), ...]`` for siblings of
        *parent_id* whose score is strictly more than ``_SOL_DELTA_EPSILON``
        below the parent's score.

        The strict-inequality boundary matches the Reviewer contract
        (reviewer/system.md branch-quality table: neutral band
        ``[−eps, +eps]`` is closed; regression is ``Δ < −eps``), so a
        sibling at exactly Δ = −eps is neutral here, not regressed. The
        ``+ 1e-9`` tolerance matches the project's float-boundary
        convention (see ``detect_plateau``). Siblings without scores
        (still-running / errored) are not counted.

        Shares the ``_SOL_DELTA_EPSILON`` constant with the Reviewer's
        rule-based fallback so the two consumers of the −0.02 threshold
        cannot drift.
        """
        from src.agents.reviewer import _SOL_DELTA_EPSILON

        parent = self._nodes[parent_id]
        if parent.score is None:
            return []
        out: list[tuple[str, int]] = []
        for child_id in parent.children_ids:
            if exclude_id is not None and child_id == exclude_id:
                continue
            child = self._nodes[child_id]
            if child.score is None:
                continue
            if (parent.score.sol_score - child.score.sol_score) > _SOL_DELTA_EPSILON + 1e-9:
                out.append((child.action_applied, child.iter_no))
        return out

    # ── checkpointing ────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialize tree state to JSON for mid-search recovery.

        Writes to a temp file first, then atomically replaces the target
        so a crash mid-write can't corrupt the checkpoint.
        """
        data = {
            "next_id": self._next_id,
            "nodes": {
                str(nid): _serialize_node(node)
                for nid, node in self._nodes.items()
            },
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: Path) -> SearchTree:
        """Deserialize tree state from JSON."""
        data = json.loads(path.read_text())
        tree = cls()
        tree._next_id = data["next_id"]
        for nid_str, node_data in data["nodes"].items():
            tree._nodes[int(nid_str)] = _deserialize_node(node_data)
        return tree


# ── best-node eligibility ────────────────────────────────────────────────────

def _eligible_for_best(node: TreeNode) -> bool:
    """Decide whether a node's score is trustworthy enough to win the run.

    See ``SearchTree.best_node`` for the rationale. A node is eligible iff
    its branch_quality is not DEAD_END, OR it was killed by beam pruning
    (in which case the measurement is still valid — only frontier
    eligibility was revoked).
    """
    from src.agents.reviewer import BranchQuality

    if node.branch_quality != BranchQuality.DEAD_END:
        return True
    return node.dead_reason == DeadReason.BEAM_PRUNED


# ── serialization helpers ────────────────────────────────────────────────────

def _serialize_node(node: TreeNode) -> dict:
    from src.agents.reviewer import BranchQuality

    return {
        "id": node.id,
        "parent_id": node.parent_id,
        "children_ids": node.children_ids,
        "action_applied": node.action_applied,
        "action_params": node.action_params,
        "depth": node.depth,
        "branch_quality": node.branch_quality.value if isinstance(node.branch_quality, BranchQuality) else None,
        "dead_reason": node.dead_reason.value if isinstance(node.dead_reason, DeadReason) else None,
        "score": _serialize_score(node.score),
        "kernel": _serialize_kernel(node.kernel),
        "profiling": _serialize_profiling(node.profiling),
        "per_workload_latency_us": _serialize_per_workload_latency(node.per_workload_latency_us),
        "runtime_ms": node.runtime_ms,
        "consecutive_agent_failures": node.consecutive_agent_failures,
        "iter_no": node.iter_no,
        "last_review": _serialize_review_feedback(node.last_review),
        "failure_details": (
            [asdict(fd) for fd in node.failure_details]
            if node.failure_details is not None
            else None
        ),
    }


def _serialize_per_workload_latency(
    per_workload_latency_us: dict[str, float] | None,
) -> dict[str, float] | None:
    """``math.inf`` is a legitimate sentinel for "workload crashed" but JSON
    rejects it — round-trip via the sentinel string ``"inf"``. ``None``
    passes through unchanged so legacy checkpoints (and the root node) stay
    distinguishable from "measured, empty"."""
    if per_workload_latency_us is None:
        return None
    return {
        uuid: ("inf" if math.isinf(v) else v)
        for uuid, v in per_workload_latency_us.items()
    }


def _deserialize_per_workload_latency(
    data: dict | None,
) -> dict[str, float] | None:
    if data is None:
        return None
    return {
        uuid: (float("inf") if v == "inf" else float(v))
        for uuid, v in data.items()
    }


def _serialize_profiling(profiling):
    """Serialize a ProfilingResult for checkpoint. Returns ``None`` when
    the node has no profile (root, or a branch that died before the
    profiler ran).
    """
    if profiling is None:
        return None
    return {
        # analytical=None for nbytes=0 profiles; preserved round-trip.
        "analytical": (
            asdict(profiling.analytical)
            if profiling.has_analytical
            else None
        ),
        "ncu": asdict(profiling.ncu) if profiling.ncu is not None else None,
        "raw_metrics": dict(profiling.raw_metrics),
        "metric_groups": dict(profiling.metric_groups),
        "degraded_reason": profiling.degraded_reason,
    }


def _serialize_review_feedback(fb) -> dict | None:
    """Serialize a ReviewerFeedback for checkpoint / meta.json. None → None;
    enums coerce to ``.value`` strings; everything else is JSON-native.
    Shared by ``_serialize_node`` (checkpoint) and ``tree_dump._late_bound_fields``
    (meta.json) so the two on-disk shapes stay consistent.
    """
    if fb is None:
        return None
    bc = fb.bottleneck_classification
    return {
        "outcome": fb.outcome,
        "metric_deltas": dict(fb.metric_deltas),
        "bottleneck_classification": bc.value if hasattr(bc, "value") else bc,
        "bottleneck_diagnosis": fb.bottleneck_diagnosis,
        "suggestions": list(fb.suggestions),
        "branch_quality": fb.branch_quality.value,
        "conditional_assessment": fb.conditional_assessment,
        "degraded": fb.degraded,
        "error_reason": fb.error_reason,
    }


def _deserialize_review_feedback(data: dict | None):
    """Reconstruct a ReviewerFeedback from its serialized dict. None → None.
    ``branch_quality`` is rebuilt as the BranchQuality enum;
    ``bottleneck_classification`` stays as the stored string (the
    dataclass field type is ``str`` — see ``ReviewerFeedback`` definition).
    Missing keys fall back to dataclass defaults so legacy checkpoints
    that wrote a partial dict still load.
    """
    if data is None:
        return None
    from src.agents.reviewer import BranchQuality, ReviewerFeedback

    return ReviewerFeedback(
        outcome=data.get("outcome", ""),
        metric_deltas=dict(data.get("metric_deltas", {})),
        bottleneck_classification=data.get("bottleneck_classification", ""),
        bottleneck_diagnosis=data.get("bottleneck_diagnosis", ""),
        suggestions=list(data.get("suggestions", [])),
        branch_quality=BranchQuality(
            data.get("branch_quality", BranchQuality.PROMISING.value)
        ),
        conditional_assessment=data.get("conditional_assessment", ""),
        degraded=data.get("degraded", False),
        error_reason=data.get("error_reason", ""),
    )


def _serialize_score(score: ScoreResult | None) -> dict | None:
    if score is None:
        return None
    return {
        "sol_score": score.sol_score,
        "baseline_latency_us": score.baseline_latency_us,
        "candidate_latency_us": score.candidate_latency_us,
        "t_sol_us": score.t_sol_us,
        "speedup": score.speedup,
        "reward_hack_suspect": score.reward_hack_suspect,
        "calibration_warning": score.calibration_warning,
    }


def _serialize_kernel(kernel: "Kernel | None") -> dict | None:
    # ``kernel=None`` on turn-exhaust failure nodes; round-trip the sentinel.
    if kernel is None:
        return None
    return {
        "spec": {
            "name": kernel.spec.name,
            "kernel_type": kernel.spec.kernel_type.value,
            "flop_count": kernel.spec.flop_count,
            "memory_bytes": kernel.spec.memory_bytes,
            "input_shapes": kernel.spec.input_shapes,
            "definition_path": str(kernel.spec.definition_path) if kernel.spec.definition_path else None,
            "pytorch_reference": kernel.spec.pytorch_reference,
            "t_sol_us": kernel.spec.t_sol_us,
        },
        "source_code": kernel.source_code,
        "autotune_configs": kernel.autotune_configs,
        "autotune_keys": kernel.autotune_keys,
        "autotune_winner": kernel.autotune_winner,
        "triton_kernel_name": kernel.triton_kernel_name,
        "dps": kernel.dps,
    }


def _deserialize_node(data: dict) -> TreeNode:
    from src.agents.reviewer import BranchQuality
    from src.eval.scorer import ScoreResult
    from src.kernels.kernel import Kernel, KernelSpec, KernelType

    score = None
    if data["score"] is not None:
        s = data["score"]
        score = ScoreResult(
            sol_score=s["sol_score"],
            baseline_latency_us=s["baseline_latency_us"],
            candidate_latency_us=s["candidate_latency_us"],
            t_sol_us=s["t_sol_us"],
            speedup=s["speedup"],
            reward_hack_suspect=s.get("reward_hack_suspect", False),
            calibration_warning=s.get("calibration_warning", False),
        )

    bq = None
    if data["branch_quality"] is not None:
        bq = BranchQuality(data["branch_quality"])

    # ``.get(..., None)`` keeps pre-dead_reason checkpoints loadable —
    # legacy DEAD_END nodes simply have no recorded cause; downstream
    # code treats that as "unknown" rather than raising.
    dr = None
    dr_raw = data.get("dead_reason")
    if dr_raw is not None:
        dr = DeadReason(dr_raw)

    k = data["kernel"]
    # Turn-exhaust failure nodes serialize with ``kernel=null``.
    if k is None:
        kernel = None
    # A1 PR 1: detect pre-autotune checkpoints by the absence of the new
    # ``autotune_configs`` field, route through Kernel.from_legacy_dict so
    # legacy num_warps/num_stages/block_size triples become single-entry
    # autotune_configs lists. New-format checkpoints take the explicit path.
    elif "autotune_configs" not in k:
        kernel = Kernel.from_legacy_dict(k)
    else:
        kernel = Kernel(
            spec=KernelSpec.from_dict(k["spec"]),
            source_code=k["source_code"],
            # ``.get`` with empty-string default keeps older checkpoints
            # loadable; the profiler's regex fallback handles them.
            triton_kernel_name=k.get("triton_kernel_name", ""),
            # ``.get`` with False default for older checkpoints written before
            # the DPS field existed. Losing dps on reload would silently
            # change correctness/profiling behavior on the resumed run.
            dps=k.get("dps", False),
            autotune_configs=k["autotune_configs"],
            autotune_keys=k.get("autotune_keys", []),
            autotune_winner=k.get("autotune_winner") or {},
        )

    return TreeNode(
        id=data["id"],
        kernel=kernel,
        parent_id=data["parent_id"],
        children_ids=data["children_ids"],
        score=score,
        branch_quality=bq,
        action_applied=data["action_applied"],
        # ``.get(..., None)`` keeps pre-action_params checkpoints loadable —
        # legacy nodes have no recorded plan params; sibling rendering will
        # fall back to bare action_applied.
        action_params=data.get("action_params"),
        depth=data["depth"],
        profiling=_deserialize_profiling(data.get("profiling")),
        per_workload_latency_us=_deserialize_per_workload_latency(
            data.get("per_workload_latency_us")
        ),
        # ``.get(..., None)`` keeps pre-opt-mem checkpoints loadable —
        # legacy nodes have no recorded runtime_ms; downstream
        # ``Producer.consider`` short-circuits cleanly on ``None``.
        runtime_ms=data.get("runtime_ms"),
        # ``.get(..., 0)`` keeps pre-quarantine checkpoints loadable —
        # legacy nodes default to "no failures recorded yet."
        consecutive_agent_failures=data.get("consecutive_agent_failures", 0),
        # ``.get(..., -1)`` keeps pre-iter_no checkpoints loadable —
        # legacy nodes default to the same sentinel as the root.
        iter_no=data.get("iter_no", -1),
        # ``.get(..., None)`` keeps pre-last_review checkpoints loadable —
        # legacy nodes have no Reviewer feedback recorded.
        last_review=_deserialize_review_feedback(data.get("last_review")),
        dead_reason=dr,
        failure_details=_deserialize_failure_details(data),
    )


def _deserialize_failure_details(data: dict) -> "list[FailureDetail] | None":
    """Read failure_details from checkpoint; synthesize from legacy
    failure_reason when loading pre-collapse checkpoints.

    has_kernel_source=False unconditionally for legacy synthesis: legacy
    on-disk layout has kernel.py at tree/node_<id>/kernel.py (flat), not
    at the new tree/node_<id>/cand_0/kernel.py path the flag points at.
    Setting True would mislead postmortem readers. See the design spec's
    "has_kernel_source semantics under legacy load" note for rationale.
    """
    new_fmt = data.get("failure_details")
    if new_fmt is not None:
        return [FailureDetail(**fd) for fd in new_fmt]
    legacy = data.get("failure_reason")
    if legacy is not None:
        return [FailureDetail(
            candidate_idx=0,
            reason=legacy,
            has_kernel_source=False,
        )]
    return None


def _deserialize_profiling(data):
    """Rehydrate a ``ProfilingResult`` from checkpoint JSON. Returns
    ``None`` when the node was saved without profile data (old-format
    checkpoints or nodes that never profiled)."""
    if data is None:
        return None
    from src.eval.profiler import AnalyticalMetrics, NCUMetrics, ProfilingResult

    a = data["analytical"]
    # Stale ``classification`` / ``arithmetic_intensity`` / ``ridge_point``
    # keys on legacy checkpoints are silently ignored — those fields live at
    # the run level now (see ``classify_run`` for classification,
    # ``RooflineResult`` for AI / ridge_point).
    #
    # ``a is None`` on checkpoints written post-2026-05-13 for nodes where
    # nbytes was 0 at profile time (a+b decoupling: analytical was skipped,
    # NCU may still be present). Mirror the serializer's None-passthrough.
    if a is None:
        analytical = None
    else:
        analytical = AnalyticalMetrics(
            achieved_tflops=a["achieved_tflops"],
            achieved_bandwidth_gb_s=a["achieved_bandwidth_gb_s"],
            pct_peak_compute=a["pct_peak_compute"],
            pct_peak_bandwidth=a["pct_peak_bandwidth"],
        )
    ncu = None
    if data.get("ncu") is not None:
        n = data["ncu"]
        ncu = NCUMetrics(
            sm_occupancy_pct=n["sm_occupancy_pct"],
            l2_hit_rate_pct=n["l2_hit_rate_pct"],
            tensor_core_util_pct=n["tensor_core_util_pct"],
            warp_stall_dominant=n["warp_stall_dominant"],
            warp_stall_dominant_pct=n["warp_stall_dominant_pct"],
            warp_stall_runner_up=n["warp_stall_runner_up"],
            warp_stall_runner_up_pct=n["warp_stall_runner_up_pct"],
        )
    raw_metrics = dict(data.get("raw_metrics") or {})
    metric_groups = data.get("metric_groups")
    if not isinstance(metric_groups, dict):
        from src.eval.profiler import _build_metric_groups

        metric_groups = _build_metric_groups(raw_metrics)
    return ProfilingResult(
        analytical=analytical,
        ncu=ncu,
        raw_metrics=raw_metrics,
        metric_groups=metric_groups,
        degraded_reason=data.get("degraded_reason"),
    )
