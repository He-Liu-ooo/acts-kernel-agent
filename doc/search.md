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

All real implemented logic.

## Beam Pruning — `beam.py`

- `beam_prune(tree, beam_width) -> list[int]`: Prune frontier to beam_width nodes. Keeps highest-scoring. Returns pruned node IDs.
- `select_next(tree, epsilon) -> TreeNode`: Epsilon-greedy selection. With probability (1-ε) pick best, with probability ε pick random.

All real implemented logic.

## Orchestrator — `orchestrator.py`

Deterministic orchestrator. Not an LLM — pure Python control flow.

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
- `plateau`: SOL score stalled for `sol_plateau_window` iterations
- `budget`: `max_depth` iterations exhausted
- `all_dead_end`: no expandable frontier nodes

### SearchResult

Output: `{best_node, total_iterations, termination_reason}`.
