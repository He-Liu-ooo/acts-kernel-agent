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
| `consecutive_agent_failures` | `int` | Count of consecutive Planner/Coder failures on this parent. Reset to 0 after a successful `add_child`. Drives quarantine via `frontier()`. |

### Methods

- `add_root(kernel) -> TreeNode`: Add baseline as root.
- `add_child(parent_id, kernel, action) -> TreeNode`: Add optimization result.
- `get_node(id) -> TreeNode`: Lookup.
- `frontier() -> list[TreeNode]`: All non-dead_end nodes whose `consecutive_agent_failures < QUARANTINE_THRESHOLD` (module constant, default 2). Quarantining repeat-failing parents prevents `select_next` from burning the search budget on a deterministically-failing node by re-picking it forever.
- `best_node() -> TreeNode`: Highest SOL score. Quarantined nodes are intentionally still considered here — quarantine blocks future expansion (a `frontier()` concern), not winner-as-final-answer; the node's already-measured score is still valid as the run's best.
- `path_to_node(id) -> list[TreeNode]`: Ordered path from root to given node. Raises `KeyError` for unknown IDs.
- `render_path(id) -> str`: Human-readable trajectory `"[i] action (QUALITY) — SOL s.sss"` from root to the given node, with the last step marked `← current`. Consumed by the Planner (path-to-parent) and Reviewer (path-to-child) so both agents reason about which actions have already been tried on this branch, not just the immediate parent.
- `save(path)`: Serialize tree to JSON checkpoint. Uses atomic write (temp file + `os.replace`) so a crash mid-write can't corrupt the file.
- `SearchTree.load(path) -> SearchTree`: Deserialize from JSON checkpoint. Raises `FileNotFoundError` for missing files. Preserves `_next_id` so new nodes don't collide. Legacy checkpoints (pre-quarantine) default `consecutive_agent_failures` to 0 — `data.get("consecutive_agent_failures", 0)` keeps them loadable.

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
    definition: Definition | None = None,
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
| `definition` | Parsed SOL `Definition` (the SOL-ExecBench replacement for the legacy ACTS `Problem` type, which has been removed). Used once per run to derive the hoisted `(flops, nbytes)` for the analytical profiler via `compute_roofline_inputs(definition, workloads[repr_idx])`, and threaded into `benchmark_kernel` for DPS-mode output allocation. `None` falls back to `baseline.spec.flop_count` / `memory_bytes` — correct for placeholder starter kernels |

### Fail-fast hardware guard

`run()` aborts immediately with `ValueError` when `config.hardware.peak_flops_fp32 <= 0` or `peak_memory_bandwidth_gb_s <= 0`. A zeroed `HardwareSpec` (the `detect_hardware()` fallback) would make every analytical profile raise `ProfilerError` and silently DEAD_END every branch — that's a global config error, not a branch event. `pipeline/optimize.py` substitutes `_PLACEHOLDER_HARDWARE_SPEC` (a populated RTX 6000 Ada stand-in) before calling `run()` so the CLI smoke path stays alive.

### Representative-workload hoist

`repr_idx = len(workloads) // 2` (middle of the selected-workload list so large/small-axis outliers don't dominate the profile; `0` when `workloads` is empty or length < 2). The analytical profiler's `(flops, nbytes)` are derived **once** from `(definition, workloads[repr_idx])` via `compute_roofline_inputs` and reused across all iterations — these are invariant per run, so recomputing per-iter would just repeat the same call. `repr_input_generator` and `repr_workload_axes` (`workloads[repr_idx].model_dump(mode="json")`) are captured the same way. Placeholder path (no SOL `definition`): `iter_flops` / `iter_nbytes` fall back to `baseline.spec.flop_count` / `memory_bytes` and `repr_workload_axes` is `{}`.

### Per-Iteration Flow

1. Check frontier — return `ALL_DEAD_END` if empty
2. Select node (epsilon-greedy from frontier)
3. Retrieve past experiences from optimization memory (filtered by `run_bottleneck`)
4. **Planner**: kernel source + profiling summary + memory + `tree_context=render_path(parent.id)` + `bottleneck=run_bottleneck` → `OptimizationPlan`
   - On `PlanningError`: increment `parent.consecutive_agent_failures += 1`, emit `planner_failed`, decay epsilon, skip the iteration (no tree mutation). The next `select_next` either picks a different parent, or — if this parent's failures hit `QUARANTINE_THRESHOLD` — picks any other frontier node.
5. **Coder** (with tools): plan + kernel + `kernel_spec`/`reference_fn`/`input_generators` → optimized kernel (self-corrects via compile + correctness tools)
   - On `ImplementationError`: increment `parent.consecutive_agent_failures += 1`, emit the coder-failure event, decay epsilon, skip the iteration (no tree mutation). Same quarantine accounting as the Planner path — repeated failures on the same parent push it past `QUARANTINE_THRESHOLD` and out of `frontier()`.
6. Add child node to tree — `child.score` and `per_workload_latency_us` are **not** committed yet
   - On successful `tree.add_child(parent.id, child_kernel, plan.technique)`: reset `parent.consecutive_agent_failures = 0`. A productive parent shouldn't be permanently quarantined for one earlier transient blip.
7. **Benchmark** child — call wraps in `per_iter_anti_cheat(self._config.anti_cheat_critical_names)` (channel A reward-hack detection); `definition` is threaded into `benchmark_kernel` so DPS kernels can pre-allocate per-iter outputs via `allocate_outputs(definition, workload)`. `BenchmarkError` (majority-failure) OR `not is_fully_successful` (partial failure) → mark branch `DEAD_END`, `beam_prune`, next iteration. After bench succeeds, `BenchmarkResult.last_outputs` (last workload's last-iter output tensors) is fed into `check_lazy_outputs_after_bench` to catch lazy/proxy returns; the list is then `.clear()`-ed so large GPU tensors don't pin through the LLM round-trip ahead
8. **Profile** child on representative workload — skip when `repr_workload_latency_s` is None; `ProfilerError` → mark `DEAD_END`, `beam_prune`, next iteration; `(flops, nbytes) == (0, 0)` (no formula for op_type) → keep branch alive but skip profile
9. Commit `child.profiling`, `child.score` (via `compute_sol_score`), `child.per_workload_latency_us` to the tree node. The emitted `score_computed` event carries `t_sol_source` (`"solar"` or `"builtin"`) so audit can distinguish SOLAR-grounded scores from `compute_roofline()` fallback-grounded ones.
10. **Reviewer**: eval results + `run_bottleneck` + live `ProfilingResult` + `tree_context=render_path(child.id)` → `ReviewerFeedback` + `branch_quality`. When profiling was skipped, defaults `branch_quality` to `PROMISING` (keeps the branch alive so `beam_prune` treats it normally)
11. `beam_prune(tree, beam_width, enable_diversity=config.beam_diversity)`
12. Termination checks: `sol_target` (child.score ≥ threshold), `plateau` (via `detect_plateau` on `best_scores`), else decay epsilon and continue
13. Budget exhausted after `max_depth` iterations → `BUDGET`

Baseline benchmark partial failure is **not** caught — no baseline means no signal, and the orchestrator raises `BenchmarkError` so the caller can surface it.

### Anti-cheat / reward-hack flow

Three independent detector channels feed the same DEAD_END routing pipeline, with one shared helper (`_kill_branch`) consolidating the per-site side-effects:

- **Channel A — in-band per-iter anti-cheat.** The eval block (benchmark) is executed inside `per_iter_anti_cheat(self._config.anti_cheat_critical_names)`. A candidate that monkey-patches `torch.cuda.Event`, spawns a background thread, or returns lazy/proxy outputs raises `RewardHackDetected` from inside the context manager. The orchestrator catches it, emits `reward_hack_detected{iter, reason, child_id}`, and routes through `_kill_branch(..., reason=DEAD_REWARD_HACK, bumps_agent_failures=True)` — agent-accountable, so the parent's `consecutive_agent_failures` is incremented and quarantine kicks in on repeat offenders.
- **Channel B — post-bench reward-hack re-eval.** When `score.reward_hack_suspect` is set (the SOL scorer's `T_k < ~T_SOL` margin signal), `_reward_hack_re_eval(child, kernel, workloads, input_generators, reference_fn=..., definition=...)` re-runs the candidate against the reference oracle with strict tolerance (`atol=1e-5`, `rtol=1e-4`) and a fresh `per_iter_anti_cheat` snapshot. The re-eval uses `maybe_wrap_dps_candidate` so DPS kernels are correctly invoked with pre-allocated outputs, and `compare_outputs` with `build_normalize_context(definition)` so multi-output (tuple/dict) returns compare name-by-name via SOL's `normalize_outputs`. Cleared → emit `reward_hack_cleared{iter, child_id}`, accept the original score and continue. Not cleared → emit `reward_hack_confirmed{iter, child_id}` and route through `_kill_branch(..., reason=DEAD_REWARD_HACK_CONFIRMED, bumps_agent_failures=True)`. Errors during the re-eval (compile failure, exception during oracle call) are treated as "not cleared" — fail-closed so a crash doesn't accidentally promote the suspect score. Skip path: when `reference_fn` / `input_generators` / `workloads` are absent (placeholder runs), the re-eval returns `True` (cleared) since there is no oracle to compare against.
- **Channel C — calibration warning.** When `score.calibration_warning` is set (less severe T_k vs. T_SOL margin), the orchestrator emits `calibration_warning{iter, child_id, t_k_us, t_sol_us}` for observability but does **not** kill the branch — calibration is a roofline-tightness signal, not a cheating signal.

**CUDA sticky-state recovery.** A subset of `RuntimeError` messages (substring match against `_CUDA_STICKY_PATTERNS = ("illegal memory access", "device-side assert", "unspecified launch failure", "misaligned address", "out of memory", "cublas", "cudnn")`) trigger a single `torch.cuda.synchronize()` retry. A loose `"cuda" in msg` check is intentionally **not** used — strings like "operation not implemented for CUDA" are real bugs and must propagate. Non-matching `RuntimeError`s re-raise. After the sync, the branch is killed via `_kill_branch(..., reason=DEAD_CUDA_ERROR)` (infra failure, no agent-failure bump). A run-level `consecutive_cuda_errors` counter is incremented when the sync itself fails; on the third consecutive failure the orchestrator raises `CUDAContextPoisoned` to abort the whole run rather than burn iterations producing meaningless results from a poisoned device. The counter resets to zero after any successful bench or successful sync.

**`_kill_branch(child, parent, iter_no, *, reason, detail, bumps_agent_failures)` helper.** Consolidates the six DEAD_END exit sites (channel A reward-hack, channel B reward-hack-confirmed, CUDA sticky-state, partial bench failure, repr-workload-latency-unavailable, profiler error). Each call: marks `child.branch_quality = DEAD_END`, optionally bumps `parent.consecutive_agent_failures` (True only for agent-output failures — Coder/Planner produced a buggy/cheating kernel; False for infra failures where the agent isn't accountable), runs `beam_prune`, and emits the `branch_dead_end{reason, detail}` + `iter_end{outcome=dead_end}` pair. The trailing `epsilon = max(epsilon_end, epsilon - decay)` decay still lives at each call site because `epsilon` and `decay` are local to `run()`'s frame.

**`DEAD_*` reason constants** (defined in `runtime/events.py`, frozen in `DEAD_REASONS`):

| Constant | When |
|----------|------|
| `DEAD_REWARD_HACK` | Channel A — in-band `RewardHackDetected` raised inside `per_iter_anti_cheat` |
| `DEAD_REWARD_HACK_CONFIRMED` | Channel B — post-bench re-eval failed to clear `reward_hack_suspect` |
| `DEAD_CUDA_ERROR` | Sticky-state CUDA error caught + recovered via `synchronize()` (run continues) |
| `DEAD_BENCH_FAILURE` | Child benchmark partial-workload failure (`not is_fully_successful`) or `BenchmarkError` |
| `DEAD_PROFILER_ERROR` | `ProfilerError` from `profile_kernel` (zero latency, missing peaks, etc.) |
| `DEAD_REPR_LATENCY_UNAVAILABLE` | Representative workload's measurement was `inf` (partial slice failure on the middle workload) |
| `DEAD_AGENT_FAILURE` | Reserved for agent-output failures routed through `_kill_branch` (Planner/Coder skip-iter cases currently emit `planner_failed` / `coder_failed` directly without DEAD_END routing) |

**`_emit_trace(iter_no, child, bench, roofline, definition, workloads, repr_idx)`.** Builds a SOL `Trace` per evaluation (`Definition` name + representative `Workload` + `Evaluation{status, environment, timestamp, correctness, performance}`) and fires `trace_emitted{iter, child_id, trace}`. The cached `Environment` is built lazily on first use via `env_snapshot(device="cuda:0")` (with a CPU-only fabricated stub fallback so `Trace`'s `NonEmptyString` validator doesn't reject it). Best-effort — all exceptions are swallowed so a trace serialization hiccup never interrupts the search loop. The placeholder path (no SOL workloads / no `definition`) emits a minimal `trace_emitted{iter, child_id, latency_us}` payload without a full `Trace` object. **Note:** the file-dump side of trace emission was deliberately removed in this PR — PR 3 will re-add it once `_run_dir` is properly threaded through the orchestrator.

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
