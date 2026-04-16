"""Tree search state management."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.evaluator import BranchQuality
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
        from src.agents.evaluator import BranchQuality

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
    from src.agents.evaluator import BranchQuality

    return {
        "id": node.id,
        "parent_id": node.parent_id,
        "children_ids": node.children_ids,
        "action_applied": node.action_applied,
        "depth": node.depth,
        "branch_quality": node.branch_quality.value if isinstance(node.branch_quality, BranchQuality) else None,
        "score": _serialize_score(node.score),
        "kernel": _serialize_kernel(node.kernel),
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
    }


def _deserialize_node(data: dict) -> TreeNode:
    from src.agents.evaluator import BranchQuality
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
    )
