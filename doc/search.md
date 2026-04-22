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

### `Orchestrator.run()` signature

```python
async def run(
    baseline: Kernel,
    workloads: list[Workload] | None = None,
    roofline: RooflineResult | None = None,
    *,
    reference_fn: Callable | None = None,
    input_generators: list[Callable[[int], tuple]] | None = None,
    problem_definition_path: Path | None = None,
    problem: Problem | None = None,
) -> SearchResult
```

| Argument | Purpose |
|----------|---------|
| `baseline` | Triton baseline kernel — root of the search tree |
| `workloads` | Representative subset for iterative benchmarking (SOL mode); `None` uses `kernel.spec.input_shapes` (legacy) |
| `roofline` | Pre-computed SOLAR result; `None` falls back to built-in `compute_roofline()` from `KernelSpec.flop_count` / `memory_bytes` |
| `reference_fn` | PyTorch oracle (from `definition.json`). Threaded into the Coder's correctness tool. Required when the Coder is LLM-backed |
| `input_generators` | One seed→args generator per selected workload. Threaded verbatim into the Coder's correctness tool so every iteration verifies on the full coverage set |
| `problem_definition_path` | SOL-ExecBench `definition.json` path. The profiler subprocess driver re-loads it to rebuild the (unpicklable) input generator. `None` falls back to `module.make_inputs` or `spec['args']` — only safe for Tier 2 self-contained kernels |
| `problem` | Parsed `Problem` used once per run to derive the hoisted `(flops, nbytes)` for the analytical profiler. `None` falls back to `baseline.spec.flop_count` / `memory_bytes` — correct for placeholder starter kernels |

### Fail-fast hardware guard

`run()` aborts immediately with `ValueError` when `config.hardware.peak_flops_fp32 <= 0` or `peak_memory_bandwidth_gb_s <= 0`. A zeroed `HardwareSpec` (the `detect_hardware()` fallback) would make every analytical profile raise `ProfilerError` and silently DEAD_END every branch — that's a global config error, not a branch event. `pipeline/optimize.py` substitutes `_PLACEHOLDER_HARDWARE_SPEC` (a populated RTX 6000 Ada stand-in) before calling `run()` so the CLI smoke path stays alive.

### Representative-workload hoist

`repr_idx = len(workloads) // 2` (middle of the selected-workload list so large/small-axis outliers don't dominate the profile; `0` when `workloads` is empty or length < 2). The analytical profiler's `(flops, nbytes)` are derived **once** from `(problem, workloads[repr_idx])` via `compute_roofline_inputs` and reused across all iterations — these are invariant per run, so recomputing per-iter would just repeat the same call. `repr_input_generator` and `repr_workload_axes` are captured the same way.

### Per-Iteration Flow

1. Check frontier — return `ALL_DEAD_END` if empty
2. Select node (epsilon-greedy from frontier)
3. Retrieve past experiences from optimization memory (filtered by `run_bottleneck`)
4. **Planner**: kernel source + profiling summary + memory + `tree_context=render_path(parent.id)` + `bottleneck=run_bottleneck` → `OptimizationPlan`
5. **Coder** (with tools): plan + kernel + `kernel_spec`/`reference_fn`/`input_generators` → optimized kernel (self-corrects via compile + correctness tools)
6. Add child node to tree — `child.score` and `per_workload_latency_us` are **not** committed yet
7. **Benchmark** child — `BenchmarkError` (majority-failure) OR `not is_fully_successful` (partial failure) → mark branch `DEAD_END`, `beam_prune`, next iteration
8. **Profile** child on representative workload — skip when `repr_workload_latency_s` is None; `ProfilerError` → mark `DEAD_END`, `beam_prune`, next iteration; `(flops, nbytes) == (0, 0)` (no formula for op_type) → keep branch alive but skip profile
9. Commit `child.profiling`, `child.score` (via `compute_sol_score`), `child.per_workload_latency_us` to the tree node
10. **Reviewer**: eval results + `run_bottleneck` + live `ProfilingResult` + `tree_context=render_path(child.id)` → `ReviewerFeedback` + `branch_quality`. When profiling was skipped, defaults `branch_quality` to `PROMISING` (keeps the branch alive so `beam_prune` treats it normally)
11. `beam_prune(tree, beam_width, enable_diversity=config.beam_diversity)`
12. Termination checks: `sol_target` (child.score ≥ threshold), `plateau` (via `detect_plateau` on `best_scores`), else decay epsilon and continue
13. Budget exhausted after `max_depth` iterations → `BUDGET`

Baseline benchmark partial failure is **not** caught — no baseline means no signal, and the orchestrator raises `BenchmarkError` so the caller can surface it.

### `TerminationReason`

`str`-subclass enum so legacy string comparisons in downstream consumers still work.

| Value | Meaning |
|-------|---------|
| `SOL_TARGET` | `child.score.sol_score ≥ config.sol_target` (default 0.95 — within 5% of hardware limit) |
| `PLATEAU` | Best score stalled across `sol_plateau_window` iterations, delta ≤ `sol_plateau_delta` |
| `BUDGET` | `max_depth` iterations exhausted without early termination |
| `ALL_DEAD_END` | Frontier empty at iteration start — no expandable nodes |

### SearchResult

Output: `{best_node, total_iterations, termination_reason, tree, run_bottleneck}`. `tree` is the full `SearchTree` carried forward so Phase C (`pipeline/report.py`) can reconstruct the root-to-best path for `technique_trace` without the orchestrator having to denormalize every path-derived view upfront. See PROCESS.md → Deferred Improvements (`SearchResult.tree` → lighter path snapshot) for when to swap this for a precomputed `best_path` / `technique_trace`.

`run_bottleneck` is the once-per-run `BottleneckType` produced by `eval/roofline.py::classify_run` immediately after roofline resolution. It is the single source of truth for retriever / planner / reviewer across every iteration (per-iter re-classification would only recompute the same answer because the problem + representative workload + hardware don't change within a run). Phase C reads it straight into `OptimizationReport.bottleneck`.

### Score + profile ordering (fail-closed on profile failure)

Within an iteration, the order is `benchmark → profile → commit score + per_workload_latency_us`. The child's `ScoreResult` is **not** written to the node until after the profile gauntlet clears, because `SearchTree.best_node()` filters only on `score is not None` — a `ProfilerError`-killed branch that had already committed a score could be promoted to the final winner. The deferred commit keeps the DEAD_END invariant aligned with promotability.

### Prompt-side helpers

- `_render_profiling_for_planner(profiling)` — compact comma-separated summary (`pct_peak_compute=..%, pct_peak_bandwidth=..%, ai=..`, plus `sm_occupancy`/`l2_hit_rate`/`dominant_stall` when NCU is present, or `[DEGRADED: <reason>]` otherwise). Feeds the Planner's `Profiling summary` section; the Reviewer builds a richer two-block analytical+NCU view from the `ProfilingResult` dataclass directly (see `reviewer.render_profiling_summary`).
- `_representative_latency_s(bench, workloads, repr_idx)` — returns the representative workload's latency in seconds, or `None` when that workload failed. Falls back to `bench.median_latency_us / 1e6` on the placeholder path (no SOL workloads).
- `_NO_PROFILE_SUMMARY` — sentinel string (`"[no profiling data available]"`) threaded into the Planner prompt when profiling is unavailable.
