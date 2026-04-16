"""Beam pruning logic for search tree."""

from __future__ import annotations

import random

from src.agents.evaluator import BranchQuality
from src.search.tree import SearchTree, TreeNode


def beam_prune(tree: SearchTree, beam_width: int) -> list[int]:
    """Prune the frontier to at most beam_width nodes.

    Returns the IDs of pruned nodes. Keeps the highest-scoring
    nodes in the frontier.
    """
    frontier = tree.frontier()
    if len(frontier) <= beam_width:
        return []

    # Sort by SOL score descending (unscored nodes sort last)
    frontier.sort(
        key=lambda n: n.score.sol_score if n.score else -1.0,
        reverse=True,
    )
    pruned_ids = []
    for node in frontier[beam_width:]:
        node.branch_quality = BranchQuality.DEAD_END
        pruned_ids.append(node.id)
    return pruned_ids


def select_next(
    tree: SearchTree,
    epsilon: float,
) -> TreeNode:
    """Epsilon-greedy selection of the next node to expand.

    With probability (1 - epsilon): expand highest-scoring frontier node.
    With probability epsilon: expand a random frontier node.
    """
    frontier = tree.frontier()
    if not frontier:
        return tree.best_node()

    if random.random() < epsilon:
        return random.choice(frontier)

    # Greedy: pick highest-scoring
    return max(
        frontier,
        key=lambda n: n.score.sol_score if n.score else -1.0,
    )
