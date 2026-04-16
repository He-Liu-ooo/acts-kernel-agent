"""Beam pruning logic for search tree."""

from __future__ import annotations

import random

from src.agents.evaluator import BranchQuality
from src.search.tree import SearchTree, TreeNode

# B3: branch-quality bonus added to raw SOL score for ranking.
# Small enough that large score gaps still dominate.
_QUALITY_BONUS = {
    BranchQuality.PROMISING: 0.05,
    BranchQuality.BLOCKED_POTENTIAL: 0.02,
    BranchQuality.PLATEAU: -0.02,
}

# B2: max score gap for a diversity swap. A minority-action node must be
# within this distance of the worst kept node to earn a rescue slot.
_DIVERSITY_GAP_LIMIT = 0.3


def _effective_score(node: TreeNode) -> float:
    raw = node.score.sol_score if node.score else -1.0
    return raw + _QUALITY_BONUS.get(node.branch_quality, 0.0)


def beam_prune(tree: SearchTree, beam_width: int, *, enable_diversity: bool = True) -> list[int]:
    """Prune the frontier to at most beam_width nodes.

    Returns the IDs of pruned nodes.

    Ranking uses effective score (raw SOL + branch-quality bonus).
    After score-based selection, a diversity pass rescues the best node
    of each missing action type — but only when it's within
    ``_DIVERSITY_GAP_LIMIT`` of the cutoff and a redundant action has
    a node to swap out.
    """
    frontier = tree.frontier()
    if len(frontier) <= beam_width:
        return []

    # B3: sort by quality-weighted effective score
    frontier.sort(key=_effective_score, reverse=True)

    kept = {n.id for n in frontier[:beam_width]}

    if not enable_diversity:
        pruned_ids = []
        for node in frontier[beam_width:]:
            node.branch_quality = BranchQuality.DEAD_END
            pruned_ids.append(node.id)
        return pruned_ids

    # B2: diversity rescue pass
    kept_actions = {n.action_applied for n in frontier if n.id in kept and n.action_applied}
    cutoff = _effective_score(frontier[beam_width - 1])

    for candidate in frontier[beam_width:]:
        if not candidate.action_applied or candidate.action_applied in kept_actions:
            continue
        if cutoff - _effective_score(candidate) > _DIVERSITY_GAP_LIMIT:
            continue
        # Find most-represented action with >1 kept nodes
        by_action: dict[str, list[TreeNode]] = {}
        for n in frontier:
            if n.id in kept:
                by_action.setdefault(n.action_applied, []).append(n)
        redundant = [(a, ns) for a, ns in by_action.items() if len(ns) > 1]
        if not redundant:
            break
        _, donor_nodes = max(redundant, key=lambda x: len(x[1]))
        victim = min(donor_nodes, key=_effective_score)
        kept.discard(victim.id)
        kept.add(candidate.id)
        kept_actions.add(candidate.action_applied)

    pruned_ids = []
    for node in frontier:
        if node.id not in kept:
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
