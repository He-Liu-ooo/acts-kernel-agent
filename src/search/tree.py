"""Tree search state management."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.reviewer import BranchQuality
    from src.eval.profiler import ProfilingResult
    from src.eval.scorer import ScoreResult
    from src.kernels.kernel import Kernel


@dataclass
class TreeNode:
    """A node in the search tree representing one kernel version."""

    id: int
    kernel: Kernel
    parent_id: int | None = None
    children_ids: list[int] = field(default_factory=list)
    score: ScoreResult | None = None
    branch_quality: BranchQuality | None = None
    action_applied: str = ""
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
    ) -> TreeNode:
        """Add a child node resulting from an optimization action."""
        parent = self._nodes[parent_id]
        node = TreeNode(
            id=self._next_id,
            kernel=kernel,
            parent_id=parent_id,
            action_applied=action_applied,
            depth=parent.depth + 1,
        )
        parent.children_ids.append(node.id)
        self._nodes[node.id] = node
        self._next_id += 1
        return node

    def get_node(self, node_id: int) -> TreeNode:
        """Retrieve a node by ID."""
        return self._nodes[node_id]

    def frontier(self) -> list[TreeNode]:
        """Return all expandable frontier nodes (not dead_end)."""
        from src.agents.reviewer import BranchQuality

        return [
            n for n in self._nodes.values()
            if n.branch_quality != BranchQuality.DEAD_END
        ]

    def best_node(self) -> TreeNode:
        """Return the node with the highest SOL score."""
        scored = [n for n in self._nodes.values() if n.score is not None]
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


# ── serialization helpers ────────────────────────────────────────────────────

def _serialize_node(node: TreeNode) -> dict:
    from src.agents.reviewer import BranchQuality

    return {
        "id": node.id,
        "parent_id": node.parent_id,
        "children_ids": node.children_ids,
        "action_applied": node.action_applied,
        "depth": node.depth,
        "branch_quality": node.branch_quality.value if isinstance(node.branch_quality, BranchQuality) else None,
        "score": _serialize_score(node.score),
        "kernel": _serialize_kernel(node.kernel),
        "profiling": _serialize_profiling(node.profiling),
        "per_workload_latency_us": _serialize_per_workload_latency(node.per_workload_latency_us),
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
        "analytical": asdict(profiling.analytical),
        "ncu": asdict(profiling.ncu) if profiling.ncu is not None else None,
        "raw_metrics": dict(profiling.raw_metrics),
        "degraded_reason": profiling.degraded_reason,
    }


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


def _serialize_kernel(kernel: Kernel) -> dict:
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
        "num_warps": kernel.num_warps,
        "num_stages": kernel.num_stages,
        "block_size": kernel.block_size,
        "triton_kernel_name": kernel.triton_kernel_name,
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

    k = data["kernel"]
    ks = k["spec"]
    def_path = Path(ks["definition_path"]) if ks["definition_path"] else None
    kernel = Kernel(
        spec=KernelSpec(
            name=ks["name"],
            kernel_type=KernelType(ks["kernel_type"]),
            flop_count=ks["flop_count"],
            memory_bytes=ks["memory_bytes"],
            input_shapes=ks["input_shapes"],
            definition_path=def_path,
            pytorch_reference=ks["pytorch_reference"],
            t_sol_us=ks["t_sol_us"],
        ),
        source_code=k["source_code"],
        num_warps=k["num_warps"],
        num_stages=k["num_stages"],
        block_size=k["block_size"],
        # ``.get`` with empty-string default keeps legacy checkpoints
        # (pre-T4) loadable; the profiler's regex fallback handles them.
        triton_kernel_name=k.get("triton_kernel_name", ""),
    )

    return TreeNode(
        id=data["id"],
        kernel=kernel,
        parent_id=data["parent_id"],
        children_ids=data["children_ids"],
        score=score,
        branch_quality=bq,
        action_applied=data["action_applied"],
        depth=data["depth"],
        profiling=_deserialize_profiling(data.get("profiling")),
        per_workload_latency_us=_deserialize_per_workload_latency(
            data.get("per_workload_latency_us")
        ),
    )


def _deserialize_profiling(data):
    """Rehydrate a ``ProfilingResult`` from checkpoint JSON. Returns
    ``None`` when the node was saved without profile data (old-format
    checkpoints or nodes that never profiled)."""
    if data is None:
        return None
    from src.eval.profiler import AnalyticalMetrics, NCUMetrics, ProfilingResult

    a = data["analytical"]
    # Stale ``classification`` keys on legacy checkpoints are silently
    # ignored — the field lives at the run level now (see ``classify_run``).
    analytical = AnalyticalMetrics(
        arithmetic_intensity=a["arithmetic_intensity"],
        ridge_point=a["ridge_point"],
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
    return ProfilingResult(
        analytical=analytical,
        ncu=ncu,
        raw_metrics=dict(data.get("raw_metrics") or {}),
        degraded_reason=data.get("degraded_reason"),
    )
