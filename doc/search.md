# Search — `src/search/`

Tree search with beam pruning. 3 LLM agents coordinated by a deterministic orchestrator.

## SearchTree — `tree.py`

Manages tree state: nodes, frontier, and expansion.

### TreeNode

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique node ID |
| `kernel` | Kernel | Kernel version at this node |
| `parent_id` | int \| None | None for root |
| `children_ids` | list[int] | Child node IDs |
| `score` | ScoreResult \| None | SOL score from eval |
| `branch_quality` | BranchQuality \| None | Reviewer's assessment |
| `action_applied` | str | Technique name that produced this node |
| `depth` | int | Distance from root |

### Methods

- `add_root(kernel) -> TreeNode`: Add baseline as root.
- `add_child(parent_id, kernel, action) -> TreeNode`: Add optimization result.
- `get_node(id) -> TreeNode`: Lookup.
- `frontier() -> list[TreeNode]`: All non-dead_end nodes.
- `best_node() -> TreeNode`: Highest SOL score.
- `path_to_node(id) -> list[TreeNode]`: Ordered path from root to given node. Raises `KeyError` for unknown IDs.
- `render_path(id) -> str`: Human-readable trajectory `"[i] action (QUALITY) — SOL s.sss"` from root to the given node, with the last step marked `← current`. Consumed by the Planner (path-to-parent) and Reviewer (path-to-child) so both agents reason about which actions have already been tried on this branch, not just the immediate parent.
- `save(path)`: Serialize tree to JSON checkpoint. Uses atomic write (temp file + `os.replace`) so a crash mid-write can't corrupt the file.
- `SearchTree.load(path) -> SearchTree`: Deserialize from JSON checkpoint. Raises `FileNotFoundError` for missing files. Preserves `_next_id` so new nodes don't collide.

## Beam Pruning — `beam.py`

### `beam_prune(tree, beam_width, *, enable_diversity=True) -> list[int]`

Prune frontier to `beam_width` nodes. Returns pruned node IDs.

Ranking uses **effective score** = raw SOL score + branch-quality bonus (B3):

| BranchQuality | Bonus |
|---------------|-------|
| PROMISING | +0.05 |
| BLOCKED_POTENTIAL | +0.02 |
| PLATEAU | -0.02 |
| None | 0 |

After score-based selection, a **diversity rescue pass** (B2) swaps in the best node of each missing action type — but only when:
1. The candidate's effective score is within 0.3 of the worst kept node (large score gaps still dominate).
2. There's a redundant action type with >1 kept nodes to swap out.
3. The candidate has a non-empty `action_applied` (root/baseline nodes are excluded).

Diversity can be disabled via config (`beam_diversity = false`) or `enable_diversity=False` parameter.

### `select_next(tree, epsilon) -> TreeNode`

Epsilon-greedy selection. With probability (1−ε) pick best, with probability ε pick random.

## Orchestrator — `orchestrator.py`

Deterministic orchestrator. Not an LLM — pure Python control flow.

### `detect_plateau(score_history, window, delta) -> bool`

Returns True if the best score hasn't improved beyond `delta` over the last `window` entries. Used for global search termination — distinct from per-branch `BranchQuality.PLATEAU`.

### Per-Iteration Flow

1. Select node (epsilon-greedy from frontier)
2. Retrieve past experiences from optimization memory
3. **Planner**: profiling + memory + feedback → `OptimizationPlan`
4. **Coder** (with tools): plan + kernel → optimized kernel (self-corrects via compile + correctness tools)
5. **Orchestrator-side eval**: benchmark → NCU → roofline → SOL score
6. **Reviewer**: eval results → `ReviewerFeedback` + `branch_quality`
7. Tree update: add node, score, beam prune
8. Memory update: store experience

### Termination

- `sol_target`: SOL score ≥ 0.95 (within 5% of hardware limit)
- `plateau`: Best score stalled for `sol_plateau_window` iterations (checked via `detect_plateau`)
- `budget`: `max_depth` iterations exhausted
- `all_dead_end`: no expandable frontier nodes

### SearchResult

Output: `{best_node, total_iterations, termination_reason}`.
