# Search — `src/search/`

Tree search with beam pruning. 3 LLM agents coordinated by a deterministic orchestrator.

## SearchTree — `tree.py`

Manages tree state: nodes, frontier, and expansion. The persisted form of the search tree lives under `<run_dir>/tree/` — see [`runtime.md`](runtime.md) for the layout and visualization formats.

### TreeNode

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique node ID |
| `kernel` | Kernel | Kernel version at this node |
| `parent_id` | int \| None | None for root |
| `children_ids` | list[int] | Child node IDs |
| `score` | ScoreResult \| None | SOL score from eval |
| `branch_quality` | BranchQuality \| None | Reviewer's assessment |
| `dead_reason` | `DeadReason \| None` | Why this node was marked `DEAD_END`. None on live nodes; set together with `branch_quality = DEAD_END` at every kill site (`_kill_branch`, `beam_prune`, the Reviewer-feedback application). Distinguishes the three causes `branch_quality` alone collapses into one flag: `BEAM_PRUNED` (lost beam competition — measurement valid), `REVIEWER_JUDGED` (kernel ran fine, Reviewer judged the branch over), and the infra-error members (CUDA / profiler / bench / reward-hack failures — measurement untrustworthy). Read by `best_node()` to decide which DEAD_END nodes are still promotable as the run's winner. None on legacy checkpoints. |
| `action_applied` | str | Technique name that produced this node |
| `depth` | int | Distance from root |
| `consecutive_agent_failures` | `int` | Count of consecutive Planner/Coder failures on this parent. Reset to 0 after a successful `add_child`. Drives quarantine via `frontier()`. |
| `iter_no` | `int` | Iteration index (1-based) that produced this node. Default `-1` for the root and for legacy checkpoints predating the field. Threaded in via `add_child`'s keyword-only argument so the on-disk per-node directory under `<run_dir>/tree/node_<id>/` can record which iteration committed it. |
| `last_review` | `ReviewerFeedback \| None` | Latest Reviewer feedback for this kernel — set at the iter that scored it (children) or by the Phase A baseline review pass (root); legacy checkpoints default to None via `_deserialize_review_feedback`. |

### Methods

- `add_root(kernel) -> TreeNode`: Add baseline as root.
- `add_child(parent_id, kernel, action_applied, *, iter_no=-1) -> TreeNode`: Add optimization result. `iter_no` is the iteration index that produced this child; defaults to `-1` (root / legacy checkpoint sentinel).
- `get_node(id) -> TreeNode`: Lookup.
- `nodes() -> Iterable[TreeNode]`: Iterate over every node in the tree, in insertion order. Used by the on-disk tree dumper to walk the full set when building `<run_dir>/tree/index.json`.
- `has_node(node_id) -> bool`: Membership test — True iff a node with this ID exists.
- `__len__() -> int`: Total node count (root + every committed child, including DEAD_END / quarantined nodes).
- `frontier() -> list[TreeNode]`: All non-dead_end nodes whose `consecutive_agent_failures < QUARANTINE_THRESHOLD` (module constant, default 2). Quarantining repeat-failing parents prevents `select_next` from burning the search budget on a deterministically-failing node by re-picking it forever.
- `best_node() -> TreeNode`: Highest SOL score, filtered by `dead_reason` rather than the `DEAD_END` flag alone. Beam-pruned nodes (`dead_reason == BEAM_PRUNED`) stay eligible — their measurement was clean, only their frontier eligibility was revoked, so a high-scoring node pruned at iter K is still promotable when later iterations regress. `REVIEWER_JUDGED` and all infra-error reasons (CUDA / profiler / bench / reward-hack) are excluded — those scores are either untrustworthy or the Reviewer explicitly told us not to promote. Legacy DEAD_END nodes with no recorded `dead_reason` are excluded as a safe default. Quarantined nodes are intentionally still considered — quarantine blocks future expansion (a `frontier()` concern), not winner-as-final-answer; the node's already-measured score is still valid as the run's best.
- `path_to_node(id) -> list[TreeNode]`: Ordered path from root to given node. Raises `KeyError` for unknown IDs.
- `render_path(id) -> str`: Human-readable trajectory `"[i] action (QUALITY) — SOL s.sss"` from root to the given node, with the last step marked `← current`. Consumed by the Planner (path-to-parent) and Reviewer (path-to-child) so both agents reason about which actions have already been tried on this branch, not just the immediate parent.
- `save(path)`: Serialize tree to JSON checkpoint. Uses atomic write (temp file + `os.replace`) so a crash mid-write can't corrupt the file.
- `SearchTree.load(path) -> SearchTree`: Deserialize from JSON checkpoint. Raises `FileNotFoundError` for missing files. Preserves `_next_id` so new nodes don't collide. Legacy checkpoints (pre-quarantine) default `consecutive_agent_failures` to 0 — `data.get("consecutive_agent_failures", 0)` keeps them loadable. `_deserialize_profiling` round-trips `analytical=None` (post-2026-05-13 checkpoints for nodes profiled with `nbytes=0`): the serializer writes `"analytical": null`, the deserializer mirrors the None pass-through and rebuilds `ProfilingResult(analytical=None, ncu=..., ...)`. Legacy checkpoints carrying a populated analytical dict still load via the same code path.

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
| `definition` | Parsed SOL `Definition` (the SOL-ExecBench replacement for the legacy ACTS `Problem` type, which has been removed). Used once per run to derive the hoisted `(flops, nbytes)` for the analytical profiler via `compute_roofline_inputs(definition, workloads[repr_idx], roofline=roofline)` (run-level `roofline` is threaded in so SOLAR's `total_flops` / `total_fused_bytes` outrank the shape-formula fallback — important on op_type=None problems where the formula bails), and threaded into `benchmark_kernel` for DPS-mode output allocation. `None` falls back to `baseline.spec.flop_count` / `memory_bytes` — correct for placeholder starter kernels |

### Fail-fast hardware guard

`run()` aborts immediately with `ValueError` when `config.hardware.peak_flops_fp32 <= 0` or `peak_memory_bandwidth_gb_s <= 0`. A zeroed `HardwareSpec` (the `detect_hardware()` fallback) would make every analytical profile raise `ProfilerError` and silently DEAD_END every branch — that's a global config error, not a branch event. `pipeline/optimize.py` substitutes `_PLACEHOLDER_HARDWARE_SPEC` (a populated RTX 6000 Ada stand-in) before calling `run()` so the CLI smoke path stays alive.

### Representative-workload hoist

`repr_idx = len(workloads) // 2` (middle of the selected-workload list so large/small-axis outliers don't dominate the profile; `0` when `workloads` is empty or length < 2). The analytical profiler's `(flops, nbytes)` are derived **once** from `(definition, workloads[repr_idx])` via `compute_roofline_inputs(..., roofline=roofline)` and reused across all iterations — these are invariant per run, so recomputing per-iter would just repeat the same call. The run-level `roofline` is threaded in so SOLAR's `total_flops` / `total_fused_bytes` outrank the shape-formula fallback (the shape formula bails on every `op_type=None` problem, including the L1 cases). `repr_input_generator` and `repr_workload_axes` (`workloads[repr_idx].model_dump(mode="json")`) are captured the same way. Placeholder path (no SOL `definition`): `iter_flops` / `iter_nbytes` fall back to `baseline.spec.flop_count` / `memory_bytes` and `repr_workload_axes` is `{}`.

### Baseline review pass

Phase A (iter=0, before the per-iter loop starts) runs `profile_kernel(root)` followed by `Reviewer.review(prev_sol_score=None)` against the baseline kernel and writes the result into `root.last_review`. This seeds the Planner's reviewer-feedback channel for iter=1 — without it, the first iteration's Planner would see `reviewer_feedback=None` and lose the baseline's bottleneck diagnosis / suggestions / conditional assessment. A `DEAD_END` verdict from the baseline review is **clamped** to `PROMISING` on `root.branch_quality` so the search can't be aborted before it starts; the underlying `feedback.branch_quality` carried in `root.last_review` is left untouched so downstream consumers still see the raw verdict. Failures (profile or review) are **swallowed** — `root.last_review` stays `None`, the iter=1 Planner sees `None`, and the search proceeds as if Phase A weren't there. A `reviewer_feedback` event is emitted with `iter=0` regardless of outcome (with `feedback=None` on failure) so audit can see the baseline pass ran. Implemented by `_apply_baseline_feedback_to_root` (invokes profiler + reviewer + clamping) and `_resolve_blob_roots` (dedup helper that consolidates the previously inline workload-blob root resolution shared between Phase A and per-iter profile calls).

### Per-Iteration Flow

1. Check frontier — return `ALL_DEAD_END` if empty
2. Select node (epsilon-greedy from frontier)
3. Retrieve past experiences from optimization memory (filtered by `run_bottleneck`)
4. **Planner**: kernel source + profiling summary + memory + `tree_context=render_path(parent.id)` + `bottleneck=run_bottleneck` (+ `reviewer_feedback=_render_review_for_planner(parent.last_review)`) → `OptimizationPlan`
   - On `PlanningError`: increment `parent.consecutive_agent_failures += 1`, emit `planner_failed`, decay epsilon, skip the iteration (no tree mutation). The next `select_next` either picks a different parent, or — if this parent's failures hit `QUARANTINE_THRESHOLD` — picks any other frontier node.
5. **Coder** (with tools): plan + kernel + `kernel_spec`/`reference_fn`/`input_generators` → optimized kernel (self-corrects via compile + correctness tools)
   - On `ImplementationError`: increment `parent.consecutive_agent_failures += 1`, emit the coder-failure event, decay epsilon, skip the iteration (no tree mutation). Same quarantine accounting as the Planner path — repeated failures on the same parent push it past `QUARANTINE_THRESHOLD` and out of `frontier()`.
6. Add child node to tree — `child.score` and `per_workload_latency_us` are **not** committed yet
   - On successful `tree.add_child(parent.id, child_kernel, plan.technique)`: reset `parent.consecutive_agent_failures = 0`. A productive parent shouldn't be permanently quarantined for one earlier transient blip.
7. **Benchmark** child — call wraps in `per_iter_anti_cheat(self._config.anti_cheat_critical_names)` (channel A reward-hack detection); `definition` is threaded into `benchmark_kernel` so DPS kernels can pre-allocate per-iter outputs via `allocate_outputs(definition, workload)`. `BenchmarkError` (majority-failure) OR `not is_fully_successful` (partial failure) → mark branch `DEAD_END`, `beam_prune`, next iteration. After bench succeeds, `BenchmarkResult.last_outputs` (last workload's last-iter output tensors) is fed into `check_lazy_outputs_after_bench` to catch lazy/proxy returns; the list is then `.clear()`-ed so large GPU tensors don't pin through the LLM round-trip ahead
8. **Profile** child on representative workload — skip when `repr_workload_latency_s` is None; `ProfilerError` → mark `DEAD_END`, `beam_prune`, next iteration. There is **no** `(flops, nbytes) > 0` gate at either profile site (baseline-pass or per-iter): `profile_kernel` is always called and handles `nbytes=0` internally (`analytical=None`, NCU still runs) so an op_type with no shape-formula doesn't lose its NCU telemetry.
9. Commit `child.profiling`, `child.score` (via `compute_sol_score`), `child.per_workload_latency_us` to the tree node. The emitted `score_computed` event carries `t_sol_source` (`"solar"` or `"builtin"`) so audit can distinguish SOLAR-grounded scores from `compute_roofline()` fallback-grounded ones.
10. **Reviewer**: eval results + `run_bottleneck` + live `ProfilingResult` + `tree_context=render_path(child.id)` → `ReviewerFeedback` + `branch_quality`. After review, `child.last_review = feedback` is set (mirrors the existing `child.branch_quality = feedback.branch_quality` assignment) so the next iter's Planner — when this child is selected as parent — sees the curated subset via `_render_review_for_planner`. When profiling was skipped, defaults `branch_quality` to `PROMISING` (keeps the branch alive so `beam_prune` treats it normally)
11. `beam_prune(tree, beam_width, enable_diversity=config.beam_diversity)` — then `tree_dump.dump_node(child, iter_no=iter_no, ncu_rep_src=...)` writes `<run_dir>/tree/node_<id>/{kernel.py, ncu.json, ncu.ncu-rep, meta.json}` so the on-disk `meta.json` reflects the post-prune `branch_quality` (an evicted child correctly serializes as `dead_end`).
12. Termination checks: `sol_target` (`best_node()` eligible winner's score ≥ threshold — filters out Reviewer-judged / infra-error DEAD_END children even if their raw score crosses the bar), `plateau` (via `detect_plateau` on `best_scores`), else decay epsilon and continue
13. Budget exhausted after `max_depth` iterations → `BUDGET`

Baseline benchmark partial failure is **not** caught — no baseline means no signal, and the orchestrator raises `BenchmarkError` so the caller can surface it.

### Anti-cheat / reward-hack flow

Three independent detector channels feed the same DEAD_END routing pipeline, with one shared helper (`_kill_branch`) consolidating the per-site side-effects:

- **Channel A — in-band per-iter anti-cheat.** The eval block (benchmark) is executed inside `per_iter_anti_cheat(self._config.anti_cheat_critical_names)`. A candidate that monkey-patches `torch.cuda.Event`, spawns a background thread, or returns lazy/proxy outputs raises `RewardHackDetected` from inside the context manager. The orchestrator catches it, emits `reward_hack_detected{iter, reason, child_id}`, and routes through `_kill_branch(..., reason=DeadReason.REWARD_HACK, bumps_agent_failures=True)` — agent-accountable, so the parent's `consecutive_agent_failures` is incremented and quarantine kicks in on repeat offenders.
- **Channel B — post-bench reward-hack re-eval.** When `score.reward_hack_suspect` is set (the SOL scorer's `T_k < ~T_SOL` margin signal), `_reward_hack_re_eval(child, kernel, workloads, input_generators, reference_fn=..., definition=...)` re-runs the candidate against the reference oracle with strict tolerance (`atol=1e-5`, `rtol=1e-4`) and a fresh `per_iter_anti_cheat` snapshot. The re-eval uses `maybe_wrap_dps_candidate` so DPS kernels are correctly invoked with pre-allocated outputs, and `compare_outputs` with `build_normalize_context(definition)` so multi-output (tuple/dict) returns compare name-by-name via SOL's `normalize_outputs`. Cleared → emit `reward_hack_cleared{iter, child_id}`, accept the original score and continue. Not cleared → emit `reward_hack_confirmed{iter, child_id}` and route through `_kill_branch(..., reason=DeadReason.REWARD_HACK_CONFIRMED, bumps_agent_failures=True)`. Errors during the re-eval (compile failure, exception during oracle call) are treated as "not cleared" — fail-closed so a crash doesn't accidentally promote the suspect score. Skip path: when `reference_fn` / `input_generators` / `workloads` are absent (placeholder runs), the re-eval returns `True` (cleared) since there is no oracle to compare against.
- **Channel C — calibration warning.** When `score.calibration_warning` is set (less severe T_k vs. T_SOL margin), the orchestrator emits `calibration_warning{iter, child_id, t_k_us, t_sol_us}` for observability but does **not** kill the branch — calibration is a roofline-tightness signal, not a cheating signal.

**CUDA sticky-state recovery.** A subset of `RuntimeError` messages (substring match against `_CUDA_STICKY_PATTERNS = ("illegal memory access", "device-side assert", "unspecified launch failure", "misaligned address", "out of memory", "cublas", "cudnn")`) trigger a single `torch.cuda.synchronize()` retry. A loose `"cuda" in msg` check is intentionally **not** used — strings like "operation not implemented for CUDA" are real bugs and must propagate. Non-matching `RuntimeError`s re-raise. After the sync, the branch is killed via `_kill_branch(..., reason=DeadReason.CUDA_ERROR)` (infra failure, no agent-failure bump). A run-level `consecutive_cuda_errors` counter is incremented when the sync itself fails; on the third consecutive failure the orchestrator raises `CUDAContextPoisoned` to abort the whole run rather than burn iterations producing meaningless results from a poisoned device. The counter resets to zero after any successful bench or successful sync.

**`_kill_branch(child, parent, iter_no, *, reason, detail, bumps_agent_failures)` helper.** Consolidates the six infra-error DEAD_END exit sites (channel A reward-hack, channel B reward-hack-confirmed, CUDA sticky-state, partial bench failure, repr-workload-latency-unavailable, profiler error). Each call: marks `child.branch_quality = DEAD_END` **and** `child.dead_reason = reason` so downstream readers (`best_node`, memory distillation, tree viz) can distinguish infra-error kills from the other DEAD_END causes (`BEAM_PRUNED`, `REVIEWER_JUDGED`) that route through different sites. Then: optionally bumps `parent.consecutive_agent_failures` (True only for agent-output failures — Coder/Planner produced a buggy/cheating kernel; False for infra failures where the agent isn't accountable), runs `beam_prune`, calls `tree_dump.dump_node(child, iter_no=iter_no, ncu_rep_src=None, failure_detail=detail)` so the dead-end node still gets a per-node directory under `<run_dir>/tree/node_<id>/` carrying the kill-site prose; the categorical cause flows via `child.dead_reason` (already set above) and surfaces in `meta.json` as the top-level `dead_reason` field through `_late_bound_fields`, and emits the `branch_dead_end{reason, detail}` + `iter_end{outcome=dead_end}` pair. The trailing `epsilon = max(epsilon_end, epsilon - decay)` decay still lives at each call site because `epsilon` and `decay` are local to `run()`'s frame.

**`DeadReason` enum** (defined in `runtime/events.py` as `class DeadReason(str, Enum)`, frozen in `DEAD_REASONS = frozenset(DeadReason)`; the `str` base lets members JSON-serialize as their string value directly). Three semantic groups:

| Member | Group | When |
|--------|-------|------|
| `REWARD_HACK` | infra error | Channel A — in-band `RewardHackDetected` raised inside `per_iter_anti_cheat` |
| `REWARD_HACK_CONFIRMED` | infra error | Channel B — post-bench re-eval failed to clear `reward_hack_suspect` |
| `CUDA_ERROR` | infra error | Sticky-state CUDA error caught + recovered via `synchronize()` (run continues) |
| `BENCH_FAILURE` | infra error | Child benchmark partial-workload failure (`not is_fully_successful`) or `BenchmarkError` |
| `PROFILER_ERROR` | infra error | `ProfilerError` from `profile_kernel` (zero latency, missing peaks, etc.) |
| `REPR_LATENCY_UNAVAILABLE` | infra error | Representative workload's measurement was `inf` (partial slice failure on the middle workload) |
| `AGENT_FAILURE` | infra error | Reserved for agent-output failures routed through `_kill_branch` (Planner/Coder skip-iter cases currently emit `planner_failed` / `coder_failed` directly without DEAD_END routing) |
| `BEAM_PRUNED` | promotable | Node ran fine but lost the beam competition. Set inside `beam_prune` (both diversity-enabled and diversity-disabled branches). Measurement is trustworthy, so `best_node()` keeps these eligible as the run's winner. |
| `REVIEWER_JUDGED` | excluded | Reviewer's `branch_quality` verdict for this iter was `DEAD_END` — kernel ran fine but the Reviewer classified the branch as regressed/over. Set in the reviewer-feedback application path alongside `child.branch_quality = feedback.branch_quality`. Excluded from `best_node()` so the run's winner aligns with the Reviewer's verdict. |

**`_emit_trace(iter_no, child, bench, roofline, definition, workloads, repr_idx)`.** Builds a SOL `Trace` per evaluation (`Definition` name + representative `Workload` + `Evaluation{status, environment, timestamp, correctness, performance}`) and fires `trace_emitted{iter, child_id, trace}` (the full SOL `Trace` payload rides inside the event, which `RunContext` writes through to `events.jsonl`). The cached `Environment` is built lazily on first use via `env_snapshot(device="cuda:0")` (with a CPU-only fabricated stub fallback so `Trace`'s `NonEmptyString` validator doesn't reject it). Best-effort — all exceptions are swallowed so a trace serialization hiccup never interrupts the search loop. The placeholder path (no SOL workloads / no `definition`) emits a minimal `trace_emitted{iter, child_id, latency_us}` payload without a full `Trace` object. **Note:** there is currently no separate per-evaluation trace file dump — the trace lives only inside the `trace_emitted` event in `events.jsonl`. A standalone `traces/eval_<iter>.json` side-channel is a deferred follow-up, gated on threading the run directory into the orchestrator (the orchestrator does not currently hold a `run_dir` handle; that lives on `RunContext` in `src/runtime/run_context.py`).

### `TerminationReason`

`str`-subclass enum so legacy string comparisons in downstream consumers still work.

| Value | Meaning |
|-------|---------|
| `SOL_TARGET` | `best_node().score.sol_score ≥ config.sol_target` (default 0.95 — within 5% of hardware limit; `best_node()` excludes REVIEWER_JUDGED + infra-error DEAD_ENDs via the `dead_reason` filter, so a sub-target eligible winner can't be shipped under a target-hit banner just because a disqualified child's raw score crossed the bar) |
| `PLATEAU` | Best score stalled across `sol_plateau_window` iterations, delta ≤ `sol_plateau_delta` |
| `BUDGET` | `max_depth` iterations exhausted without early termination |
| `ALL_DEAD_END` | Frontier empty at iteration start — no expandable nodes |

### SearchResult

Output: `{best_node, total_iterations, termination_reason, tree, run_bottleneck}`. `tree` is the full `SearchTree` carried forward so Phase C (`pipeline/report.py`) can reconstruct the root-to-best path for `technique_trace` without the orchestrator having to denormalize every path-derived view upfront. See PROCESS.md → Deferred Improvements (`SearchResult.tree` → lighter path snapshot) for when to swap this for a precomputed `best_path` / `technique_trace`.

`run_bottleneck` is the once-per-run `BottleneckType` produced by `eval/roofline.py::classify_run` immediately after roofline resolution. It is the single source of truth for retriever / planner / reviewer across every iteration (per-iter re-classification would only recompute the same answer because the problem + representative workload + hardware don't change within a run). Phase C reads it straight into `OptimizationReport.bottleneck`.

### Score + profile ordering (fail-closed on profile failure)

Within an iteration, the order is `benchmark → profile → commit score + per_workload_latency_us`. The child's `ScoreResult` is **not** written to the node until after the profile gauntlet clears, because `SearchTree.best_node()` filters on `score is not None` plus the promotability rule on `dead_reason` — a `ProfilerError`-killed branch that had already committed a score could be promoted to the final winner. The deferred commit keeps the DEAD_END invariant aligned with promotability: at the moment a node carries a score, the promotability check still has to hold.

### Prompt-side helpers

- `_render_profiling_for_planner(profiling, roofline=None)` — compact comma-separated summary. Guards each block on the dataclass's own presence flags: `pct_peak_compute` / `pct_peak_bandwidth` ride through only when `profiling.has_analytical` (omitted when `analytical=None`, e.g. nbytes couldn't be derived); `arithmetic_intensity` rides through only when `roofline` is non-None; `sm_occupancy` / `l2_hit_rate` / `dominant_stall` ride through only when `profiling.has_ncu`, otherwise `[DEGRADED: <reason>]`. Feeds the Planner's `Profiling summary` section; the Reviewer builds a richer two-block analytical+NCU view from the `ProfilingResult` dataclass directly (see `reviewer.render_profiling_summary`).
- `_representative_latency_s(bench, workloads, repr_idx)` — returns the representative workload's latency in seconds, or `None` when that workload failed. Falls back to `bench.median_latency_us / 1e6` on the placeholder path (no SOL workloads).
- `_NO_PROFILE_SUMMARY` — sentinel string (`"[no profiling data available]"`) threaded into the Planner prompt when profiling is unavailable.
