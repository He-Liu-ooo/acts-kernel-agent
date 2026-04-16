"""Tree search state management."""

from __future__ import annotations

from dataclasses import dataclass, field
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
