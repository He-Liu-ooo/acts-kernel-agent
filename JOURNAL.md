# ACTS — Design Journal

Records the coding process and rationales behind each design choice. Organized by component. Within each section, amendments are dated when a decision is revisited or reversed.

---

## Search

### Tree search with beam pruning (over iteration or full evolutionary)

**Rationale**: Simple iteration (Astra) can't backtrack — if branch A→B hits a dead end, it can only go forward or revert. Full evolutionary (EvoToolkit) is expensive and overkill for single-kernel optimization. Tree search can backtrack (iteration can't) and is cheaper than evolutionary. Best-first with beam constraint adapts to uneven branch depths (unlike level-synchronized beam search). Epsilon-greedy prevents getting stuck in local optima.

No evolutionary fallback — single strategy keeps the search layer simple and debuggable.

### Parent retention

When a node is expanded, the parent stays in the frontier. This is the key advantage over linear iteration — the search can return to A and try A→C after A→B fails.

### Child retention (keeping regressed children)

Some optimizations require passing through a performance valley (e.g., restructuring memory layout is temporarily slower but enables vectorized access for a net gain). AutoKernel's greedy revert-on-regression policy can never discover these paths. Regressed children are handled by three mechanisms: (1) score-based deprioritization, (2) beam constraint pruning, (3) Reviewer `branch_quality` override.

### Diversity-aware beam pruning (B2) + branch-quality weighting (B3) (2026-04-16)

**B3 — quality-weighted effective score**: Raw SOL score alone doesn't capture the Reviewer's assessment. A PROMISING node at 0.60 may be more valuable than a PLATEAU node at 0.62, because "promising" means the Reviewer sees visible underlying improvement. Small bonuses (+0.05 PROMISING, +0.02 BLOCKED_POTENTIAL, -0.02 PLATEAU) shift the ranking without overriding large score gaps.

**B2 — diversity rescue**: Pure score ranking can collapse the frontier to one action type (e.g., all "tiling" nodes). This starves exploration — if tiling is a local optimum, the search can't escape. The diversity pass rescues one node per missing action type, but only if it's close enough to the cutoff (within 0.3) and there's redundancy to swap out. This preserves the PRD's "simple and debuggable" principle: diversity is a single post-sort pass, not a complex multi-objective ranking.

**Root exclusion**: The orchestrator creates the root with `action_applied=""`. Without exclusion, diversity would rescue the root (unique empty action) over useful optimization nodes. Empty actions are excluded from diversity accounting.

**Configurable**: `beam_diversity` config flag (default `true`). Allows disabling diversity for ablation studies or problems where pure exploitation is preferred.

### Atomic checkpoint writes (2026-04-16)

**Rationale**: Checkpointing exists to survive crashes. Writing directly to the final path means a crash mid-write corrupts the only recovery point — defeating the purpose. Temp file + `os.replace` is atomic on POSIX: the checkpoint is either the old version or the new version, never partial.

### Global plateau detection (2026-04-16)

**Rationale**: Two distinct plateau concepts in the system:

- **Branch-level**: Reviewer marks individual nodes as `BranchQuality.PLATEAU`. These stay in the frontier but get deprioritized by score + quality weighting. This steers the search away from stale branches.
- **Global**: The best score across the entire tree hasn't improved in `sol_plateau_window` consecutive iterations. This terminates the search — no branch is making progress.

`detect_plateau` tracks the global best score per iteration (not the child's score, which could regress while the global best stays flat). The function lives in `orchestrator.py` (decision C2) because the tree is a pure data structure — tracking score history is a control-flow concern.

### Reviewer branch quality values

- `"promising"` — regression but underlying improvement visible (e.g., "memory traffic dropped 40%, one more fix should recover latency")
- `"blocked_potential"` — optimization is correct but benefit masked by a different bottleneck. E.g., memory optimization on compute-bound kernel shows no latency gain, but if compute bottleneck is resolved, the memory optimization would unlock 15-25% improvement. Must provide `conditional_assessment`: what follow-up action would unblock the potential.
- `"plateau"` — diminishing returns
- `"dead_end"` — fundamental mismatch, prune immediately

### Distinguishing DEAD_END causes via `dead_reason` (2026-05-11)

**Problem**: `branch_quality == DEAD_END` collapsed three semantically distinct causes onto the same flag. The same node could be DEAD_END because of (a) infrastructure error (CUDA crash, profiler failure, bench failure, reward-hack confirmation), (b) Reviewer's verdict that the branch is regressed/over, or (c) beam-pruning eviction. Reasons WERE distinguished in telemetry (`branch_dead_end` event's reason codes in `events.py`), but never on the node — so `best_node()` had to exclude every DEAD_END node uniformly to stay safe against case (a). That meant a beam-pruned node carrying the run's best valid score was invisible to the final winner pick, the plateau tracker, and the report — Codex's adversarial review flagged this as a silent-slow-ship hazard.

**Fix**: Lift the existing telemetry reason codes (`DEAD_REWARD_HACK`, `DEAD_CUDA_ERROR`, …) into a typed `DeadReason(str, Enum)` in `events.py` with two new members for the previously-unrecorded cases:

- `BEAM_PRUNED` — set by `beam.beam_prune` when a node loses the beam competition. Score is trustworthy.
- `REVIEWER_JUDGED` — set by the orchestrator when the Reviewer's `branch_quality` verdict is `DEAD_END`. Kernel ran fine; the Reviewer just classified the branch as over.

Persist on `TreeNode.dead_reason` (None on live nodes and legacy checkpoints). The `(str, Enum)` base keeps members JSON-serializable as their string value, so the same enum drives both telemetry payloads and checkpoint records — single source of truth for the wire format.

**Behavior change in `best_node()`**: filter on `dead_reason` rather than the DEAD_END flag alone. Beam-pruned nodes are eligible (their score is valid); Reviewer-judged and all infra-error reasons stay excluded; legacy DEAD_END nodes without a recorded reason are excluded as a safe default. The `frontier()` filter is unchanged — every DEAD_END cause should remove a node from future expansion.

**Tree-viz colors**: three semantic shade groups (`dead_beam_pruned` lightest → `dead_reviewer_judged` medium → `dead_infra_error` darkest) instead of a single dead-end shade. Generic `dead_end` color stays as fallback for legacy checkpoints.

**Why an enum, not string constants**: the user picked enum over string constants for type safety + grouping under one symbol. The codebase already uses the `BranchQuality(str, Enum)` pattern, so this is consistent. The module-level `DEAD_*` aliases that existed in `events.py` were removed (not kept as enum-member aliases) to prevent drift between two ways of referring to the same value.

**Why `DEAD_REASONS = frozenset(DeadReason)` survived**: kept as the membership set even though enum typing already enforces validity, because it's still the natural way to iterate over "all dead reasons" in tests and visualization code. Runtime validation of `_emit_dead_end(reason)` was removed — the typed parameter does the work.

### Serial beam expansion (2026-04-19, /simplify review)

**Rationale**: `Orchestrator.run()` expands one frontier node per iteration despite `beam_width ≥ 1`. Parallelizing via `asyncio.gather` across the top-k picks would amortize three sequential LLM calls (Planner → Coder → Reviewer) across k concurrent branches — the largest wallclock-latency win available. Deliberately deferred because three downstream components assume single-writer semantics on the tree:

- **`beam_prune`**: the diversity-aware pass (see B2 above) ranks the current frontier once per iteration. Concurrent expansion would either need a frontier-snapshot-per-worker (stale rankings) or a post-join re-prune (defeats the parallelism win for small k).
- **`MemoryStore.add()`**: today a single-file JSON rewrite per add. Concurrent writers would race on the file. The deferred "batched flush" improvement (see `PROCESS.md` → Deferred Improvements) is a prerequisite — not a blocker, but parallelism pulls it onto the critical path.
- **Checkpoint writes**: atomic temp-file + `os.replace` is correct for one writer; N writers racing on the same checkpoint path would corrupt recovery state even with atomic replace.

**Decision**: keep expansion serial until a real benchmark shows LLM latency is the dominant cost. At that point, design the change as a coordinated restructure — frontier snapshots + batched memory flush + per-worker checkpoint slots — rather than dropping `asyncio.gather` into the hot path. Recorded with its trigger in `PROCESS.md` → Deferred Improvements → "Parallel beam expansion via asyncio.gather".

### Search-tree recording — per-node dump under <run_dir>/tree/ (2026-05-02)

**Per-node directory, streamed per-iter**: each node gets `tree/node_<id>/{kernel.py, ncu.json, ncu.ncu-rep, meta.json}` over one inlined JSON blob. **Why**: postmortem is `cat node_5/kernel.py` and per-file diffs, not jq on a megabyte blob. Streaming after each `beam_prune` (not end-of-run) preserves partial state when a multi-minute live run crashes mid-search.

**Bind/unbind module-level state**: mirrors `events.py`. **Why**: Orchestrator stays unaware of `run_dir`; no parameter-threading through the search loop. RunContext owns lifecycle.

**Trace cross-reference via metadata, not duplication**: per-node `meta.json` carries only `trace_workflow="acts_iter"` + `iter_no`; Planner/Reviewer prose stays in `traces/*.jsonl`. **Why**: traces are the single source of truth for LLM-call detail — duplicating prose would drift.

**Post-prune dump + `finalize_tree` rewrite**: streamed dump runs after `beam_prune` so `branch_quality` reflects post-prune state; `finalize_tree` rewrites every `meta.json` from final tree state at end-of-run. **Why**: streamed-only would silently disagree with `index.json` once the beam fills and evicts already-dumped nodes.

**`failure_detail` on `dump_node`** (Codex adversarial round): `_kill_branch` now calls `dump_node` with the kill-site prose surfaced into the per-node directory. **Why**: "why did node 5 die" postmortem needs the DEAD_END node's directory, not just an index entry. *Amended 2026-05-11*: the originally-paired `failure_reason` parameter was retired and the nested `meta.json.failure` block flattened — `dead_reason` is now the categorical single source of truth and `failure_detail` carries only kill-site prose. See "Retire nested `failure: {reason, detail}` block" entry below for the merge rationale.

**`.ncu-rep` decoupled from `cache_dir`**: capture path is `(cache_dir or _ncu_tmpdir())`; NCU `-f` handles persistent-tmpdir collisions. **Why**: orchestrator path doesn't pass `cache_dir`, so tying `.ncu-rep` to it would silently drop the report on every real run.

**Non-goals**: not a checkpoint primitive (`SearchTree.save/load` left for future resume); no live rendering; `.ncu-rep` captured but no analysis tooling on top of it.

**Amendment 2026-05-08 — `finalize_tree` refreshes late-bound node fields.** The 2026-05-08 root-dump fix (item #1: call `dump_node(root, …)` after baseline `per_workload_latency_us` assignment) wrote `node_0/meta.json` *before* `root.score` was computed, leaving `score: null` in the persisted artifact even though `index.json` and the in-memory tree had the scored baseline. `finalize_tree` already rewrote `branch_quality` post-prune; extending it to also refresh `score`, `per_workload_latency_us`, and (post a second Codex review) `children_ids` for already-dumped nodes addresses this regression class structurally — any future late-bound node field gets the same treatment without needing every dump-call site to be moved or duplicated. Codex adversarial review caught the original `score` instance; a follow-up `/codex:review` caught `children_ids` (root dumps with `children_ids: []` and the field stayed null even after iters added children). The refresh-finalize approach was preferred over moving the root dump call (which would only fix the known fields, not the class). New regression tests assert `node_0/meta.json["score"]` and `["children_ids"]` match the in-memory root after `finalize_tree` runs. Lesson: when extending the refresh-field list, enumerate **all** dump-time-captured fields that can be late-bound, not just the ones the latest review surfaced.

### Baseline review pass at iter=0 (2026-05-10/11)

**Why baseline review**: prior to this change the Reviewer only ran on child nodes (iter≥1), so the Planner's first expansion of the root had no `reviewer_feedback` to ground its proposals. With the new Planner-consumes-parent-`last_review` flow ("Planner now consumes parent's last_review" in Agents below), the root needs a `last_review` populated by a profile + review pass before iter 1 starts. Orchestrator now runs `profile_kernel(root)` + `Reviewer.review(prev_sol_score=None)` at iter=0 via a new `_apply_baseline_feedback_to_root` helper, then writes `root.last_review = feedback` and emits the standard `reviewer_feedback` event with `iter=0`.

**Why `DEAD_END` is clamped at baseline**: the Reviewer's `branch_quality` verdict is grounded in the delta against a parent. At baseline there is no parent — `prev_sol_score=None` — so a `DEAD_END` verdict has no signal to support it. If the baseline review returned `DEAD_END` and we honored it, the frontier would empty as `ALL_DEAD_END` *before* the first iteration ran. Clamp policy: if the baseline review returns `DEAD_END`, downgrade to the next-worst quality before storing. Subsequent iterations restore the normal Reviewer contract (parent score available → all four verdicts are admissible).

**Error-swallow contract**: baseline failure (Reviewer call raises, profile fails, etc.) is non-fatal. The helper logs and swallows; `root.last_review` stays `None`. The iter-1 Planner then sees `reviewer_feedback=None` — exactly the same payload it received pre-feature when no baseline review existed. Safe failure mode: degrades to prior behavior rather than aborting the run.

**Reference**: `_apply_baseline_feedback_to_root` in `src/search/orchestrator.py`. The same commit also deduped a prior inline blob-roots resolution block into `_resolve_blob_roots`; that helper is a refactoring side-effect, not a search-semantics change.

### SOL_TARGET termination gate filters on eligible winner, not raw child score (2026-05-11)

**Bug**: pre-fix, the termination check at the advance path in `src/search/orchestrator.py` used `child.score.sol_score >= sol_target`. A Reviewer-judged DEAD_END child could clear `sol_target` while being correctly excluded from `best_node()` via the `dead_reason` filter landed earlier 2026-05-11 ("Distinguishing DEAD_END causes via `dead_reason`" above). The gate fired on the child's raw score and returned `TerminationReason.SOL_TARGET` with `best` pointing to a different (sub-target) eligible node — shipping a sub-target winner under a "target hit" banner.

**Fix**: termination now gates on `best.score is not None and best.score.sol_score >= self._config.sol_target`. The new check can't fire unless a *trustworthy promotable* node meets the bar.

**Invariant**: termination eligibility now matches `best_node()` promotability — single source of truth for "what counts as a winner."

**Source**: Codex adversarial review finding, 2026-05-11.

### Retire nested `failure: {reason, detail}` block — flat `failure_detail` + `dead_reason` only (2026-05-11)

`failure.reason` always duplicated `dead_reason.value` — both were set together at every `_kill_branch` site, so the nested block carried no information that wasn't already on the node.

`dead_reason` is set on **every** DEAD_END node (infra-error kills, beam-pruned, Reviewer-judged) via the node field landed earlier 2026-05-11 ("Distinguishing DEAD_END causes via `dead_reason`" above); the nested `failure` block only covered the infra-error subset, so the two surfaces disagreed on coverage as well as on duplication.

**Merge**: drop `failure_reason` from `dump_node`'s signature in `src/runtime/tree_dump.py`; `meta.json` carries `dead_reason` (categorical, all DEAD_END paths, sourced from `_late_bound_fields(node)["dead_reason"]` in `_build_meta`) plus an optional top-level `failure_detail` (kill-site prose, only when the kill site had a dynamic message — exception text, workload-errors string).

**Result**: `dead_reason` is the single source of truth for the DEAD_END cause; `failure_detail` carries only the prose that doesn't fit in the enum.

### A2 — K-way Coder fan-out per iter (2026-05-15)

**Symptom that drove this.** "Baseline always wins" persisted across trial runs even after A1 landed the autotune-decorator foundation: the median Coder draft at the existing forced T=1.0 regresses against a Triton baseline that's already decent. One draft per iter, scored against a strong reference, loses on variance alone. The fix is best-of-K, not better-of-one — convert the per-iter Coder call from a single `coder.implement(...)` into K parallel calls via `asyncio.gather`, then rank the survivors against each other (and the baseline) before promoting one to the tree.

**K=4 chosen.** Matches AccelOpt's `num_samples=2 × breadth=2 = 4`, the canonical best-of-N point in code-gen literature — wide enough to absorb decoder variance at T=1.0, narrow enough that the LLM-call budget per iter stays linear in a sane constant. Diversity comes from decoder stochasticity at the already-forced T=1.0; per-call temperature schedules and prompt-perturbation strategies were considered (Q2 in the brainstorm) and rejected — both increase the surface area for a regression with no evidence that T=1.0 sampling is the bottleneck on diversity.

**Best-of-K folds into ONE tree child (Q1).** The K candidates compete inside the iter, not on the tree. Promoting K children at once would break beam-pruning's frontier semantics — `beam_prune` was designed against the invariant "one expansion per iter produces ≤1 child" and the diversity/quality pass would have to re-rank K-at-a-time against the global frontier, which is the same change set that the deferred parallel-beam-expansion entry (2026-04-19) flagged as a coordinated restructure. K-way *tree* fan-out is deferred to the evotoolkit-inspired C1/C2 work; A2 is strictly within-iter.

**Best-of-any-survivor (Q5a).** A2 is a reliability layer: if even one of K candidates makes it past `coder.implement` to bench/profile, that iter produces a tree node. The tree-side beam prune already handles the "valid score but loses to baseline" case (regressed-child retention plus quality-weighted scoring), so the in-iter selector doesn't need to second-guess it — rank by SOL Score, promote the best, let beam pruning sort out whether the parent stays in the frontier.

**Rank-and-profile-fallback for winner selection.** Pre-A2, a ProfilerError on the candidate killed the iter — acceptable when there was only one candidate. Post-A2, killing the iter on a rank-1 ProfilerError would defeat the whole reliability benefit when K−1 valid siblings stand ready. New flow in `_select_best_candidate`: rank `bench_results` by SOL Score, profile in order, the first profile-success wins. `tree.add_child` happens only after profile succeeds. Preceding candidates that failed to profile emit `coder_failed` events and produce no tree node — they were rank-superior but unprofileable, which is an infra-error class, not a search signal worth persisting.

**Channel-A reward-hack aborts the iter, no sibling fallback.** `per_iter_anti_cheat` is a detector, not a restorer: it snapshots method IDs on entry and raises `RewardHackDetected` on drift, but does *not* restore patched primitives. After a Channel-A violation the process is tainted — any later candidate's bench/profile runs against a potentially-monkeypatched torch/triton. Mitigation: any `RewardHackDetected` during per-candidate bench aborts the iter immediately. The iter ends SKIPPED; `consecutive_agent_failures` bumps (it's an agent-fault — a candidate that patches the eval harness is misbehaving, regardless of which sibling it was). Channel-B reward-hack on the winner still kills the iter without sibling-promote fallback because the winner is post-rank — falling back to a rank-N loser when the winner is the only candidate that was profiled hard enough to surface the hack is the wrong move. Rare enough in practice (agent-fault, not infra) that the iter-kill is acceptable. Explicit state restoration on detection is filed as deferred tech-debt: subsequent iters currently keep running under the same potentially-patched process.

**Quarantine-counter reset semantic.** Pre-K-way, any successful Coder call cleared `consecutive_agent_failures`. The K-way analog: any *survivor* — one of K making it past `asyncio.gather` — clears the counter, regardless of how many siblings failed agent-side. Stochastic decoder variance at T=1.0 means K−1 failures out of K is not the parent's fault; the parent did its job by producing K live draws.

**SOL-score helper reuses the canonical scorer.** `_select_best_candidate` calls `sol_execbench.sol_score.sol_score` directly (lazy-imported at the call site). Earlier drafts duplicated the formula inline "to avoid a cycle"; a grep through `src/eval/scorer.py` confirmed no cycle exists — the orchestrator already imports the canonical scorer along other paths. Inline duplication would have created a silent drift surface against a numerator/denominator that's already non-trivial.

**Tech-debt notes.** The canonical `sol_score` has a denominator singularity at `t_k = 2·t_sol − t_b` (latent in the upstream scorer; the helper inherits it untouched — A2 doesn't fix what it didn't break, and a guard belongs in the canonical surface, not here). K-way assumes a concurrent LLM backend — serial backends degrade to K× serial calls per iter, paying the latency without the wall-clock win, but no API contract changes. From the `/simplify` pass, four deferred refactors are filed for follow-up: (1) `_make_kernel` / `_make_profile` fixture consolidation (pre-A2 duplication across ~3–4 test files, not regressed by A2 but newly visible against the K-way test surface); (2) the 5-tuple `(idx, coder_out, kernel, bench, autotuner)` flowing through `_select_best_candidate` wants to be a dataclass; (3) `coder_failed.reason` is currently a free-form string and should land as an enum on the next event-schema sweep; (4) `_bench_one_candidate` extraction + a CUDA sticky-state helper to factor out the per-candidate bench teardown that A2 inlined to ship.

### Failure-node retention — rationale from run_20260517T044132 postmortem (2026-05-17)

**Postmortem that drove this.** A 12-iter live run on `L1/048_fused_gate_up_projection_with_swiglu` produced one accepted child per iter for iter 1–5, then 100% Coder failure for iter 6–12 (27 `coder_failed` events vs 5 `coder_submitted`). The failure pattern fell into two modes that turned out to be the same bug viewed from two sides:

1. **`autotune burn-in failed: CUDA error: operation not supported on global/shared address space`** (cudaErrorInvalidAddressSpace) during the autotune burn-in launch. The LLM-generated `@triton.autotune` config list shipped an aggressive `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=4` entry that — combined with the dual FP32 accumulator pattern (`acc_gate` + `acc_up`, each 128×128) — overcommits the device's register + shared-memory budget. Triton compiles the config and the launch faults the CUDA context; once one config faults, the remaining workloads inherit the poisoned context and the iter reports `0/3 workloads survived`.

2. **"Coder exhausted turn budget (8) without calling `submit_kernel`"**. Trace `trace_4bce81c8` (iter 7 cand 0) shows the Coder in a tight `compile_kernel_tool → check_correctness_tool` loop, 4 round-trips deep — `check_correctness_tool` keeps returning the same `cudaErrorInvalidAddressSpace`, and the Coder edits the *kernel body* each round (pointer arithmetic, masks) without ever touching the autotune config list at the top of the file. The 4 successive `compile_kernel_tool` inputs in that trace show identical config blocks across all turns. The error message ("invalid address space" → naturally reads as a pointer bug) actively misleads the Coder away from the actual root cause; the 8-turn budget then expires.

**Why iter 1–5 succeeded with the same config list.** node_1's kernel ships the *byte-identical* config list as iter-6 cand-0. The difference is body-state: iter 1–5's bodies (single-accumulator GEMM, then iter 5's GMEM-spill warp-spec design) kept the borderline config just inside the budget, so Triton compile-pruned it and chose a smaller winner. Iter 5's reviewer feedback ("remove the GMEM spill, fuse single-pass") pushed iter 6+ Coders back to a clean dual-accumulator body — same shape as node_1 but apparently slightly different register-pressure profile, enough to push the 128×128×64×stages=4 entry across the compile-prune/launch-fault threshold.

**Why the search machinery couldn't escape.** Three properties of the current tree contract combined to trap the search inside the failing pathway:

- **Failed candidates leave no trace on the tree.** A `coder_failed` event is emitted to `events.jsonl`, the iter is marked SKIPPED, and the candidate is dropped. No `TreeNode` is created. From the search's perspective, iter 6–11 produced zero children at all four parents tried.
- **Sibling context only carries successful siblings.** The A2-era `render_siblings(parent_id)` method returns one-liners for prior *children* of the parent — children that didn't exist for the failed iters. From iter 6 onward, every new candidate's Planner saw the same 2–3 successful siblings as iter 5 did. The Planner had no signal that 4 prior candidates at this parent had all bench-failed with the same CUDA-address class.
- **Reviewer never fires on failed iters.** `reviewer_feedback` only fires after `bench_done` + `score_computed`. The 27 failed candidates produced no Reviewer trace at all, so the Reviewer→Planner channel had nothing to say beyond what iter 5's review had already established. The Planner kept proposing variations of "halve BLOCK_K," "reduce register pressure" — Tier-1 actions that the Coder couldn't realize because the autotune block, not the kernel body, was the binding constraint.

**Decision: failure nodes belong on the tree.** Failed candidates persist as tree nodes carrying (a) the action that was attempted, (b) a typed failure class — drawn from a small enum extracted from `coder_failed.reason` at orchestrator time, not at planner-render time — and (c) the originating planner params + Coder source. They are not expandable (no descendants are spawned from them), but they ARE sibling-eligible: `render_siblings` includes them in the parent's sibling list with a failure-format rendering (`action=<id>, params={…}, FAILURE: <class>, turns_used=<n>`) parallel to the success-format (`SOL, Δ, outcome, branch_quality`). This subsumes the originally-proposed "dead_branch signal after N consecutive child failures" entirely — N consecutive failure siblings at the same parent is a computable property of the sibling list, not a separate orchestrator-side counter, and the Planner can read the pattern directly instead of receiving a derived alarm.

**Why failure classes, not raw reason strings.** The two failure modes above are very different actionable signals — "autotune burn-in launch fault" tells the Planner "narrow the config grid," while "Coder exhausted turn budget without submit" tells the Planner "the technique you proposed is hard to realize from this parent." Collapsing both to `FAILURE` loses the routing information. A small fixed enum (candidates: `coder_turn_budget`, `autotune_compile_fault`, `autotune_launch_fault`, `correctness_mismatch`, `bench_timeout`) extracted at orchestrator time keeps the failed-sibling rendering symmetric with the successful-sibling rendering (one field per slot, no free-form prose for the LLM to misread).

**Relationship to `consecutive_agent_failures` + `branch_quality=DEAD_END`.** The existing `consecutive_agent_failures` counter triggers a Reviewer-side DEAD_END verdict after N agent-fault iters; that semantic survives unchanged for *whole-iter* failures (e.g., `RewardHackDetected`). What's new is that *per-candidate* failures within a K-way fan-out now leave a tree artifact. The two systems are complementary: `consecutive_agent_failures` measures iters, failure nodes measure children-at-a-parent.

**Relationship to A2's "rank-superior but unprofileable" pattern.** A2 already documents that pre-winner candidates whose profile call fails emit `coder_failed` and produce no tree node — that decision was made before this postmortem and treated infra-error failures as not-worth-persisting. Failure-node retention reverses that for a specific class of failures: not infra-error in the random-noise sense (a transient CUDA hiccup), but *systematic* failures where the Planner's next pick should account for what just didn't work. The two cases are distinguishable by the failure class enum — `bench_timeout` / random CUDA errors stay drop-on-the-floor; `autotune_*_fault` / `coder_turn_budget` / `correctness_mismatch` persist.

**What this enables (deferred to the brainstorm pass).** Once failure nodes carry their action+params, two follow-on capabilities become reachable: (1) the Planner can route around dead branches by reading "3 prior siblings here failed with autotune_launch_fault" and proposing an action class that doesn't trip the same constraint; (2) the Planner→Coder contract can include autotune-config bounds as plan params (rather than the Coder copy-pasting the parent's list verbatim), since the Planner is now the agent that sees the failure pattern. The exact rendering format, the failure-class taxonomy, the question of whether failure-node rendering counts toward `repeated_pathway_dead_end` event detection, and whether the Reviewer should also see failure siblings (today it only consumes the consumer="reviewer" sibling-context render) are all open and belong in the brainstorm.

**Pointer.** Postmortem run: `runs/run_20260517T044132_970459Z`. Surviving accepted nodes: `tree/node_{0..5}`. Identical autotune block visible at `tree/node_1/kernel.py:6-17` (success) and trace `trace_7c95c4feeef340e1b34e544382a4d392` (iter 6 cand 0 submission, fault). Coder loop pathology in trace `trace_4bce81c8dc5049fdaff5ee8e5529ee02` (iter 7 cand 0, 8 turns of compile→correctness with no `submit_kernel`). A2's tech-debt note (3) — "`coder_failed.reason` should land as an enum on the next event-schema sweep" — is the natural pickup point for the failure-class taxonomy.

### Autotune-exclude structured bounds — escalation from failure-node retention's soft prompt rule (2026-05-18)

**Predecessor and why it wasn't enough.** Failure-node retention (2026-05-17, above) landed as a *soft* contract: failure siblings render into the Planner's parent-context, the Planner's rationale text names the autotune configs to avoid, and the Coder is expected to read that prose and steer clear. Run `run_20260518T035408_454910Z` showed the soft contract failing in the obvious way. Four+ successive iters crashed with `cudaErrorInvalidAddressSpace` despite the Planner picking different techniques each time (`t2_register_caching`, `t3_tf32`, `t1_occupancy`, `t1_block_size_tuning`) — the Planner *was* learning from the failure siblings and varying its action. The Coder ignored the channel and copy-pasted the parent's overcommitted autotune block `{BLOCK_M:128, BLOCK_N:128, BLOCK_K:32, num_warps:8, num_stages:4}` into every draft. The Planner held up its end; the Coder dropped the rationale text on the floor.

**Escalation.** Surface the bound as a structured field — `autotune_exclude: list[dict[str, int]]` on the Planner's structured output — and have Coder's `submit_kernel` validator hard-reject any `@triton.autotune` Config whose flattened keys partial-match an exclude pattern. The validator emits `submit_kernel FAILED: autotune_exclude violation. ...` so the Coder's next turn sees a mechanically-enforced rejection rather than a prose hint it can ignore. The same field is also rendered into Coder's *user prompt* at generation time, not just on submit-rejection: the prompt is the information channel, the validator is the backstop. Prompt-only would leave the soft-contract failure mode in place; validator-only would burn turn budget teaching the Coder a constraint the prompt could have stated upfront.

**Partial-match semantics.** A pattern dict's listed keys must all equal the corresponding Config keys; missing keys are wildcards. Narrow patterns (the full flat dict of a known-bad config) target one entry; broad patterns (`{"num_stages": 4}`) sweep an entire axis. Planner defaults to narrow — exclude exactly the failing config — and only widens after 3+ failed siblings at the same parent share an axis. This keeps the bound informative without collapsing the autotune search prematurely.

**Hard-reject, not strip-and-accept.** Strip-and-accept (filter the bad Config out of the submitted list and keep the rest) would let the Coder keep producing the bad block iter after iter; the negative signal *is* the feature. A submit that violates the bound is a submit the Coder needs to re-draft.

**Implementation note.** `_flatten_autotune_config(cfg)` was extracted to `src/kernels/kernel.py` so both the validator and the existing `_render_autotune_winner` agree on what "flat" means (kwargs ∪ {`num_warps`, `num_stages`}); inline duplication across the two consumers would have been a silent drift surface.

**Cross-ref.** A2 K-way Coder fan-out (2026-05-15) — the K-way iteration is what generates the failure-sibling cohort the Planner now learns from; without K-way, a parent rarely accumulates enough failure siblings for the exclude list to be informative.

---

## Agents

### 3 LLM agents + deterministic orchestrator

**Rationale**: After analyzing AccelOpt (2-agent), STARK (3-agent), Astra (5-agent), we initially chose 4 agents (Planner, Coder, Reviewer, Debugger). Revised to 3 agents after deciding to give the Coder compile/correctness tools via the OpenAI Agents SDK — see "Debugger merged into Coder" below.

**SDK choice**: OpenAI Agents SDK. Provides `Agent`, `Runner.run`, `function_tool`, structured output parsing, and model-swapping via `OpenAIChatCompletionsModel` (any OpenAI-compatible API works). AccelOpt and Astra both use this SDK. AccelOpt uses it as a thin single-call wrapper; Astra uses it with `function_tool` for compile/benchmark/test tools. ACTS follows Astra's pattern for the Coder (tool-using) and AccelOpt's pattern for Planner/Reviewer (single-call, no tools).

### Why not 2 (merging Reviewer into Planner)

| Concern | 2-agent (merged) | 3-agent (separate Reviewer) |
|---------|------------------|-----------------------------|
| Planner prompt size | Large (profiling data + memory + action library + eval results) | Focused (memory + action library + Reviewer's distilled summary) |
| Auditability | Hard to tell if bad planning came from bad analysis or bad technique selection | Each agent's reasoning is isolated and inspectable |
| Model flexibility | Must use expensive model for both | Reviewer can use cheaper model |
| Extensibility | Adding future metrics requires changing Planner | Reviewer absorbs new metrics; Planner interface unchanged |

### Debugger merged into Coder (2026-04-13)

Originally had 4 agents — a separate Debugger that diagnosed compilation/correctness failures and produced fix plans for the Coder. Merged into Coder after deciding to use the OpenAI Agents SDK with `function_tool`.

**Why merge**: If the Coder has compile and correctness-check tools, it can self-correct within its own turn. A compilation typo that previously required Coder → eval (fail) → Debugger → Coder (3 LLM calls, 2 orchestrator round-trips) now resolves in one Coder call with an internal tool loop. The separate Debugger agent added complexity without adding capability.

**Why not keep Debugger as escalation**: If a fresh prompt helps break out of a rut, that's an argument for retrying the Coder with different context, not for a separate agent. The tree search also provides natural recovery — a failed branch is pruned, and the search explores other branches.

**Retry budget**: Coder gets `max_debug_retries` self-correction attempts per iteration. If exhausted, the branch is marked dead. *Implementation evolved across 2026-04-18 → 2026-04-22*: original wiring used a hardcoded `_MAX_TURNS = 7` (= 2×3 + 1 — three compile+correctness tries plus a final structured-output turn). `CoderAgent.__init__` later took `ACTSConfig` so the budget travels with run config. After the option-α refactor moved the Coder's final emission from SDK `output_type=` enforcement to a `submit_kernel` tool call, the formula bumped to `2 * max_debug_retries + 2` (= 8 by default — the +2 reserves the `submit_kernel` tool call plus a final plain-text confirmation that terminates the SDK loop). See "Coder Pydantic structured output" → Turn-budget entry below for the up-to-date formula.

### Why not 5 (Astra-style)

Astra's Orchestrator agent is unreliable (better as deterministic code). Astra's separate Tester and Benchmarker are wasteful — correctness checking and benchmarking are deterministic operations that don't need LLM agents. Our eval harness runs these as code; the Reviewer interprets the results.

### Agent model choices

- *Planner*: Strongest reasoning model (planning quality is the bottleneck).
- *Coder*: Strong code + reasoning model (implements plans and self-corrects via tools; called every iteration).
- *Reviewer*: Can be cheaper model (analysis is easier than planning).

### Future: context-adaptive agent specialization

From advisor discussion: agent specialization should be driven by LLM context window capacity as a finite resource. V1 uses 3 agents with large-context model. For smaller-context models, increase specialization:
- Large context (200K+): 3 agents
- Medium context (32-128K): 5-6 agents (Reviewer splits into Compute-Reviewer and Memory-Reviewer)
- Small context (8-32K): 7+ agents (further specialization, higher communication overhead)

**Hierarchical agent capabilities**: Upper-level agents (orchestrator, Planner) should be discriminative. Lower-level agents (Coder) should be more capable with more tools.

### Planner: Pydantic output_type over JSON-mode parsing (2026-04-17)

**Rationale**: Two approaches for structured LLM output: (1) Pydantic `output_type` on the SDK `Agent` — the SDK handles schema enforcement and parsing automatically, (2) JSON-mode with manual `json.loads()` + validation. Chose `output_type` because: the SDK generates the JSON schema from the Pydantic model and enforces it at the API level (constrained decoding), parsing errors are handled by the SDK retry logic, and the output model serves as the contract between agents. The Pydantic model (`OptimizationPlanOutput`) is converted to an internal dataclass (`OptimizationPlan`) via `_output_to_plan()` to keep Pydantic out of the rest of the codebase.

*Subsequently* (option α, 2026-04-26) the Planner moved off `output_type=OptimizationPlanOutput` onto a `submit_plan` tool call — the same SDK strict-schema rejection the Coder hit on 2026-04-22 surfaced on the Planner's first live call (Pydantic `dict[str, X]` fields trip `additionalProperties should not be set`, and DeepSeek-reasoner additionally rejects any `response_format=json_schema`). The Pydantic validator still runs (inside the tool body), so the contract and in-loop retry behavior are preserved verbatim — only the SDK-wire shape changed. See "Planner + Reviewer submit-tool migration (2026-04-26)" entry below.

### Planner system prompt design (2026-04-17)

**Rationale**: Analyzed prompt designs from 3 reference repos:
- AccelOpt: includes NKI API reference + experience feedback loop in system prompt
- Astra: terse "strategist" prompt, constraint co-location, all agents inline
- AutoKernel: 700+ line mega-prompt with tiered playbook, anti-patterns, gain ranges

Adopted a hybrid approach: bottleneck→technique mapping tables from AutoKernel's playbook pattern, anti-patterns section (7 rules), expected gains by tier (risk/reward table), experience interpretation guide, and 6 decision rules. Excluded Triton API reference (unlike AccelOpt's NKI reference) since Triton is well-represented in LLM pretraining data.

### LLM backend choice: DeepSeek V3 (2026-04-17)

**Rationale**: Evaluated Chinese model APIs for the LLM backend. Chose DeepSeek V3 as default for all agents. Key factors: strong Triton/CUDA knowledge in pretraining, reliable JSON mode for structured output, ~$0.27/1M input tokens (viable for 100+ iterations), native OpenAI-compatible API. GLM-5.1 (Zhipu) bookmarked for future evaluation — demonstrated strong kernel optimization capability (KernelBench L3: 3.6x, 14h CUDA optimization at 35.7x) but structured output reliability unverified and API not yet stabilized.

### Reviewer: Pydantic output_type, rule-based fallback, explicit degraded signal (2026-04-17)

**Rationale**: Mirrored the Planner's Pydantic structured-output pattern so both single-call agents have the same shape — the SDK enforces schema via constrained decoding, and the Pydantic model (`ReviewerFeedbackOutput`) is converted to an internal dataclass (`ReviewerFeedback`) via `_output_to_feedback()` to keep Pydantic out of the rest of the codebase. Strict `Literal` / enum typing on `bottleneck_classification` and `branch_quality` surfaces hallucinated values as retry-worthy errors inside `run_agent`, rather than silently propagating garbage strings that would break downstream beam weighting.

**Rule-based fallback** exists for two distinct paths: (1) no model configured — expected, quiet fallback; (2) LLM call exhausted retries — unexpected, must be visible. The `degraded` / `error_reason` fields on `ReviewerFeedback` distinguish these: the orchestrator logs a warning when a degraded reviewer drove a branch_quality decision, because a broken reviewer silently pushing PROMISING → PLATEAU would corrupt beam weighting and memory entries across the whole run.

**`prompt_dir` constructor parameter**: reserved for the future Compute-Reviewer / Memory-Reviewer split. A specialized reviewer is one constructor arg away — no subclassing or prompt-string plumbing required.

*Subsequently* (option α, 2026-04-26) the Reviewer moved off `output_type=ReviewerFeedbackOutput` onto a `submit_review` tool call for the same SDK reasons that drove the Coder (2026-04-22) and Planner (2026-04-26) migrations — strict-schema rejection on Pydantic `dict[str, X]` plus DeepSeek-reasoner's blanket rejection of `response_format=json_schema`. The Pydantic validator still runs inside the tool body; the rule-based degraded fallback still serves transient API blips and now also handles `max_turns_exceeded` / `missing_submit_review` failure modes via `error_reason` tags. See "Planner + Reviewer submit-tool migration (2026-04-26)" entry below.

### Multi-turn Reviewer deferred — kept single-call through the profiler PR (2026-04-21)

**Context**: with the profiler landing `ProfilingResult.raw_metrics` (the full NCU dump) alongside the curated `NCUMetrics` subset (occupancy, L2, tensor-core util, top-2 stalls), the obvious follow-up is a Reviewer that can query the raw dump when the curated signals don't match the kernel's real bottleneck signature, or request a re-profile with different `--section` / `--metrics`.

**Decision**: not in this PR. Defer to a follow-up with its own brainstorming + design pass.

**Rationale**:
1. **Agent-shape change, not a profiler change**. Going from single Pydantic call to tool-using agent (Coder-style) is a contract break, not an incremental tweak — new turn budget, new failure modes when NCU subprocess is mid-query, new prompt contract. Per CLAUDE.md step 2 it warrants `superpowers:brainstorming`, not an inline sketch at the end of a 30-file PR.
2. **No real-run data on curated-set failures**. We haven't yet seen a Reviewer diagnosis that was wrong because the curated set was too narrow. Building the escape hatch before the pain is visible risks optimizing for the wrong signature — e.g. picking tool variant A (raw-metrics lookup, ~0 cost) when the real need is variant B (on-demand re-profile with different sections, ~30s per query), or vice versa. The first real end-to-end run is the forcing function.
3. **PR discipline**. The profiler PR already crosses ~30 files (profiler + `BottleneckType` refactor + orchestrator/report/reviewer wiring + GPU tests + cleanup). Folding in a Reviewer agent-shape change dilutes review focus and inflates PR size past the "small PR" rule.
4. **Two variants with very different cost profiles worth designing separately**:
   - **Variant A**: tool exposes `raw_metrics` dict already on `ProfilingResult`. No new NCU subprocess. Effectively free.
   - **Variant B**: tool triggers a fresh `ncu` call with different `--section` / `--metrics` on the same kernel. Expensive (~30s on RTX 6000 Ada), requires cache-key expansion to include the metric set requested, and introduces partial-failure modes mid-review.
   Variant A is probably the first step — cheap, and its limits will reveal whether B is worth the subprocess latency.

**Trigger for revisiting**: first real end-to-end run where the Reviewer's curated-set-based diagnosis is visibly signal-starved — top-2 stalls + headline metrics don't explain the measured bottleneck, and the LLM or rule-based fallback produces generic / incorrect technique guidance. At that point the failure shape is concrete and we can design the tool around it.

> **Updated 2026-04-27**: deferral overridden by explicit user intent; see "Multi-turn Reviewer (Variant A): on-demand metric queries — preemptive capability behind a flag (2026-04-27)" below.

### Multi-turn Reviewer (Variant A): on-demand metric queries — preemptive capability behind a flag (2026-04-27)

**Context**: The 2026-04-21 deferral entry kept the Reviewer single-call until a real signal-starvation case appeared. Today's decision overrides that trigger: ship Variant A as a preemptive capability behind a default-off flag (`ACTSConfig.reviewer_metric_queries`), so the escape valve exists before the first signal-starvation case is observed. Variant B (re-profile subprocess) and beyond stay deferred with the same triggers; only Variant A — in-memory `raw_metrics` lookup — is implemented.

**Locked decisions** (all confirmed during 2026-04-27 brainstorming):
- Variant A only. One new tool: `query_metric(names: list[str]) -> dict[str, str]`.
- Menu of available metric keys pre-loaded into the user prompt; no separate discovery tool.
- `max_turns=6` (`2N+2` with `N=2`) when the flag is on; `max_turns=4` (existing) when off. Prompt heuristic caps fetches at **one per review** to preserve the in-band submit-retry headroom inside the 6-turn budget — querying twice technically fits without a retry but leaves no slack for a corrective resubmit on a Pydantic slip; the heuristic encodes that contract. No hard fetch-count cap in code; the budget is the only enforced limit. Codex adversarial review (2026-04-27) flagged the original "at most twice" wording as inconsistent with the budget — tightened to "at most once" before commit.
- Default off. Existing single-call submit-tool path is the verified default.
- Strict-mode opt-out reused for both the new tool's `list[str]` arg and `dict[str, str]` return value (same SDK trap as the planner submit-tool dict params, see "Strict-mode opt-out for submit-tool dict params, 2026-04-26").

**Operating procedure (LLM contract)**: Read curated → submit; fetch is the exception. Encoded in `prompts/reviewer/system.md` `## Tools` section.

**Failure contract is unchanged**: no new degraded `error_reason` tags, no new exception types. The `max_turns_exceeded` / `missing_submit_review` / `llm_retries_exhausted` paths cover the multi-turn case identically.

**Defensive input validation in `_make_query_metric_tool`** (added post-Codex-review, 2026-04-27): `strict_mode=False` opts out of the SDK's pre-validation, so the tool body itself checks `isinstance(names, list)` and falls back to a recoverable `{"_error": "..."}` dict on shape drift. Element-level `str()` coercion defends against integer/None elements. Without these guards, a bare-string `names` payload would iterate char-by-char and emit garbage events to `events.jsonl` — silent corruption rather than a crash, but observability-poisoning. Codex adversarial review flagged this asymmetry with `submit_review` (which IS Pydantic-wrapped); the fix restores symmetry without rejecting opt-α's strict-mode-off design.

**Trigger for revisiting Variant B**: a real run where the LLM consistently asks `query_metric` for keys *not* in `raw_metrics`, AND the curated NCU section is genuinely the wrong section for the bottleneck shape.

### Coder: Pydantic output_type, tool placeholders, explicit failure contract (2026-04-18)

**Rationale**: Mirrored the Planner/Reviewer Pydantic structured-output pattern — `KernelCodeOutput` is the typed contract for the Coder's final answer, and schema validation is what catches drift between the LLM's output and the rest-of-pipeline's expectations. *Originally* (2026-04-18) the model was sent to the SDK via `output_type=KernelCodeOutput`, which the SDK translated to `response_format=json_schema` on the chat-completions request; the T4 follow-up (2026-04-22) added a second field (`triton_kernel_name`) plus a cross-field `@model_validator`. *Subsequently* (option α, 2026-04-22) the submission flow moved to a `submit_kernel(source_code, triton_kernel_name)` tool call because reasoning-model providers (DeepSeek-reasoner) reject the SDK's `response_format=json_schema` field; the Pydantic validator still runs (inside the tool body), so the contract and the in-loop retry behavior are preserved verbatim — only the SDK-wire shape changed. See "Coder routes final answer through submit_kernel (option α, 2026-04-22)" entry below.

**Tool wiring — closure-capture factories (2026-04-18)**: `_make_compile_tool(kernel_spec)` and `_make_correctness_tool(kernel_spec, reference_fn, input_generator)` return plain callables closed over per-problem context. `implement()` wraps them with `function_tool` at call time and builds a fresh `Agent` per invocation. Alternatives considered: SDK `RunContextWrapper` (adds SDK-specific plumbing to tool signatures), module-level mutable state (racy, un-testable). Closure-capture keeps the factories unit-testable without the SDK installed, matches the pattern in Astra/autokernel, and the per-call Agent construction is cheap (no network, no model instantiation — only object wrapping).

**Turn budget — `_max_turns = 2 × config.max_debug_retries + 2` (current; was `+ 1` pre-α)**: each self-correction cycle is one `compile_kernel_tool` call + one `check_correctness_tool` call. The +2 over `2N` reserves one turn for the `submit_kernel` tool call and one for the brief plain-text confirmation that terminates the SDK loop. Default `ACTSConfig.max_debug_retries = 3` gives 8 (was 7 under the pre-α `output_type=` flow, which only needed one extra turn for the structured-output emission). User framing still holds: "3 tries means code can fail 2 times" — the third attempt must pass or the agent calls `submit_kernel` with its best compiling effort. `CoderAgent.__init__` accepts `ACTSConfig` so the budget travels with the run config.

**Failure contract — one sanctioned output in every case**:
- `run_agent()` returns `None` (transient retry exhaustion) → `implement()` raises `ImplementationError`.
- SDK tool loop hits `_max_turns` without ever calling `submit_kernel` → `_run_tool_agent` catches the SDK's `MaxTurnsExceeded` and converts to `ImplementationError` (option γ, 2026-04-22). If the model managed to call `submit_kernel` before the budget ran out, the captured submission is returned instead — the run merely went over budget after the answer landed.
- The prompt instructs the model to call `submit_kernel` with "the last version that compiled cleanly" when its tool retries don't converge. This is the *only* legal failure submission, aligned explicitly with the `KernelCodeOutput` schema (which has no rationale field) and the hard rule that forbids submitting sources that were never compiled. No rationale side-channel, no multi-field schema, no prose stuffed into `source_code`.
- Without a model configured → returns a `KernelCodeOutput.model_construct` stub (validation skipped) carrying the unchanged source and an empty `triton_kernel_name`; the profiler's regex fallback handles the empty-name case.

Orchestrator-side handling of `ImplementationError` is wired (option γ, 2026-04-22): `Orchestrator.run` catches it around the per-iteration `coder.implement` call, logs a warning, decays epsilon, and continues to the next iteration without adding a tree node. `baseline_generator.py`'s 3-attempt retry loop catches the same exception during Phase A. SDK `MaxTurnsExceeded` no longer leaks past the Coder boundary.

**No Reviewer feedback in the Coder's user prompt**: the Planner already consumes Reviewer feedback and distills its conclusions into the plan. Injecting feedback again at the Coder level would risk the Coder second-guessing the plan instead of implementing it. `build_user_prompt()` is plan-only (+ current kernel).

**Temperature split — Coder 0.0, Planner/Reviewer 0.3 (2026-04-18)**: determinism is load-bearing for code generation — variance in kernel code is almost always noise, not creativity — so the Coder runs at 0.0. Upstream agents benefit from a small amount of variance: Planner explores technique selection across tiers instead of deterministically picking the highest-ranked option every time, and the Reviewer's diagnosis wording varies slightly without drifting off-schema (strict Pydantic enums on `bottleneck_classification` and `branch_quality` still pin the structure). Pinning tests (`test_plan_uses_nonzero_temperature`, `test_review_uses_nonzero_temperature`) guard against regression to 0.0.

### LLM backend retry policy: narrow transient catch + jittered backoff + logging (2026-04-17)

**Rationale**: The original `run_agent` caught `Exception` broadly. That conflates two fundamentally different failure modes: **transient** (rate limit, timeout, 5xx — the right response is "wait, try again") and **permanent** (auth error, schema violation, programmer bug — the right response is "fail fast, surface the cause"). Retrying a 401 doesn't fix it; it just wastes wall-clock and hides the real problem in a retry-exhausted warning.

**Narrow catch**: retry only a fixed tuple of `openai` exceptions (`RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`). Every other exception propagates immediately. The `retriable` parameter is exposed so tests can inject a synthetic exception class without requiring the `openai` package installed.

**Exponential backoff with ±25% jitter**: `delay * 2^(attempt-1) * uniform(0.75, 1.25)`. Jitter prevents thundering-herd synchronization when multiple in-flight agents hit the same rate-limit wall at once — all waking up at exactly the same instant would just hit the limit again.

**Named-logger observability**: `logger.info` per retry, `logger.warning` on exhaustion — both include the exception class name. The Reviewer uses this to populate `error_reason` when it falls back, so a downstream operator reading the log can tell "rate-limited 3× then exhausted" from "unreachable endpoint" without reading the code.

### Planner now consumes parent's last_review (2026-05-10)

**Rationale**: the Planner's user prompt previously carried profiling deltas + run-bottleneck but not the prior iteration's Reviewer verdict. Adding `parent.last_review` to the prompt closes the loop — the Reviewer's diagnosis of *the parent kernel* directly informs *the next plan that extends it*. Implemented via a new `TreeNode.last_review: ReviewerFeedback | None` field (read at orchestrator line ~537, written at line ~939+ on each child after its review), plus the existing `parent_profiling_summary` it already consumed.

**Why a curated subset, not the full `ReviewerFeedback`**: a new module-level helper `_render_review_for_planner(fb)` in `src/agents/reviewer.py` returns only `outcome` / `bottleneck_diagnosis` / `suggestions` / `conditional_assessment`. The other Reviewer fields are deliberately omitted:

- *`metric_deltas` skipped* — the Planner already gets profiling deltas via `parent_profiling_summary` (per-iter, structured). Re-emitting them through the Reviewer's distilled narrative would duplicate data and risk drift between the two surfaces.
- *`bottleneck_classification` skipped* — the Planner already gets `run_bottleneck` (passed as a separate arg from the orchestrator). The Reviewer's per-node classification was added for completeness, but for planning purposes the run-level label is the authoritative one.
- *`branch_quality` skipped* — search-engine concern (beam pruning + quarantine gates). Surfacing it in the Planner's prompt would risk the model self-censoring on `PLATEAU`/`DEAD_END` parents rather than letting the search engine handle pruning. The Planner's job is to propose; the search engine decides what survives.

**Serialization**: `_serialize_review_feedback` / `_deserialize_review_feedback` were added so `TreeNode.last_review` round-trips through `tree/node_<id>/meta.json` and survives legacy-checkpoint restores (nodes saved before the field existed deserialize with `last_review=None`).

**Tracking note**: the helper's docstring currently references a spec document under `doc/specs/`. Per CLAUDE.md, specs/plans are not committed and get deleted post-merge, so that reference will become dangling. Recording the rationale here in JOURNAL is what makes the docstring's spec reference unnecessary going forward — readers looking for "why this subset" should land here, not on a missing file.

### Cross-attempt memory: thread prior baseline-attempt errors into next `translate()` (2026-05-13)

**Motivating evidence**: `runs/run_20260512T153106_544768Z` ran 3 baseline-generation attempts × 8 turns. All three failed on the same `AttributeError("module 'triton.language' has no attribute 'tanh'")`. Attempt 1 hit it on turn 2 and evolved past it; attempts 2 and 3 re-emitted `tl.tanh` on their *first* turn, because each `Runner.run` starts with an empty SDK message history. Within an attempt the SDK typed-item list feeds tool errors back; across attempts the feedback was zero. Fix: `baseline_generator` accumulates per-attempt tool errors into `AttemptFailure` records and renders them into the next `translate()` user prompt under "## Prior attempt failures" with `### Attempt N` sub-headers.

**Cumulative across attempts, not last-attempt-only**. Attempt 3 sees errors from attempts 1 AND 2 in chronological order. With `max_baseline_retries=3` the section is bounded at ~8 KB even pathologically, and recurring errors are themselves load-bearing signal — same `tl.tanh` in Attempts 1 AND 2 means the model is persistently wrong, not unlucky once. Truncating to the last attempt would destroy that signal.

**Verbatim tool-error strings, not LLM-curated summaries**. The model already saw these strings in-loop on the attempts where they happened; replaying as-is needs no interpretation. Curation would need either heuristic regex extraction (brittle on Triton tracebacks) or another LLM call (cost + new failure surface).

**`AttemptFailure` colocated with `ImplementationError` in `coder.py`** (consumer side), imported by `baseline_generator.py` (producer side). Tiny frozen dataclass — producer crosses the boundary in one direction, consumer already owns the matching `ImplementationError`.

**`_record_failure` helper collapses 4 FAILED-branch sites**. All three tool factories (`_make_compile_tool`, `_make_correctness_tool`, `_make_submit_tool`) had identical `if error_log is not None: error_log.append(msg); return msg` patterns. Helper makes the next factory automatically participate — alternative is a fourth copy-paste and a near-certain miss on the fifth.

**Submit-tool wired through `error_log` (Codex catch)**. Without this, an attempt that fails *only* at submit time (Pydantic reject) has empty `tool_errors` and the prompt-builder emits the "no tool errors recorded — likely a reasoning-content budget issue" placeholder — actively misleading. Threading the shared list through `_make_submit_tool` (~10 lines + 4 tests) fixes the submit-failure path, which is also the shape most likely to recur identically across attempts (reasoning lands where the schema rejects, cold-start retry lands in the same place).

**Trigger-gated tech debt — prompt-size cap deferred**. Codex flagged that pathological Triton tracebacks could push the cumulative section past the context window on tighter-context providers, and `BadRequestError` is not in `RETRIABLE_EXCEPTIONS` so failure is terminal. Tracked in `PROCESS.md` — DeepSeek-reasoner's 128 K is comfortably bounded; first non-DeepSeek provider trip motivates a concrete cap shape better than guessing now.

### Reviewer launch-bound guard + regression-row rewrite (2026-05-13)

**Two coordinated edits to `src/prompts/reviewer/system.md`**, both costing frontier nodes pre-fix: the Reviewer mis-attributed latency to occupancy on launch-bound workloads and stamped `dead_end` on regressions that still had a stated unblock pathway.

**Launch-bound guard — banded interpretation of `pct_peak: bw`.** Three bands: `< 5%` → launch-bound (~3 µs fixed overhead, occupancy not the lever); `[5%, 50%]` → partially amortized (occupancy *may* be a lever but isn't necessarily THE one); `> 70%` → near memory ceiling (fusion / precision / L2-reuse). Mechanical "raise occupancy / shrink block / add parallelism" suggestions based on a low occupancy number alone are forbidden — any occupancy-targeting suggestion must cite a specific stall metric. Matching anti-pattern added.

**Regression-row rewrite — `dead_end` requires absent OR repeated-pathway failure.** Pre-rewrite, any `SOL delta < −0.02` regression mapped to `dead_end`. New rule: regression with a non-empty `conditional_assessment` pathway not yet failed on an ancestor/sibling = `blocked_potential`; regression with no pathway OR pathway already regressed on parent/sibling = `dead_end`.

**Why it matters.** `frontier()` permanently excludes DEAD_END nodes (see "Distinguishing DEAD_END causes via `dead_reason`" above) — mis-stamping a recoverable regression removes a promotable node from search forever. `blocked_potential` already meant "branch isn't over, next action is the unblock"; the rewrite extends that semantic from bottleneck-masking to regression-with-stated-lever. The launch-bound guard is the upstream half — cleaner diagnoses produce cleaner branch-quality verdicts.

### Sibling-aware Planner/Reviewer contracts + regression-polarity rules (2026-05-13)

**Motivating evidence**: postmortem of `runs/run_20260513T090733_257562Z` (3 iters, plateau-out). Iter 2 spawned from node 1 (parent SOL 0.5051), picked `t1_block_size_tuning` with BLOCK_N=32, regressed to 0.4339 (Δ −0.071). Iter 3 spawned from the *same parent* node 1 and re-picked `t1_block_size_tuning` with a nearly identical rationale — because the iter-3 Planner saw `tree.render_path(parent.id)` (root→parent only) and `parent.last_review`, and had **no view of node 2's existence**. The iter-2 Reviewer compounded the bias: `conditional_assessment` read "the block-size tuning action space is still wide open — only one configuration has been tried" — a promote-the-class phrasing emitted *after* a regression, pulling the next Planner straight back into the failed action.

**Two contract drifts being closed**:

1. **Planner amnesia on siblings.** The Planner's user prompt had no sibling channel. Closed by adding `SearchTree.render_siblings(parent_id, exclude_id=None)` and threading the result through `PlannerAgent.build_user_prompt` as a new `## Siblings already tried from this parent` section (inserted between `## Search tree context` and `## Reviewer feedback`). Reviewer gets the same section (between `## Search tree context` and `## Knowledge base context`) with `exclude_id=child.id` so it doesn't see itself. Omit-when-empty pattern matches the existing `tree_context` / `reviewer_feedback` gating.

2. **Reviewer suggestion-polarity bias on regression.** A new "Regression-polarity rules" subsection in `prompts/reviewer/system.md` (after "Suggestion rules", before "Anti-patterns") encodes the **Moderate polarity rule**: on a regression iter, the Reviewer MAY prescribe a parameter fix for the same `action_applied` ID **only if** `bottleneck_diagnosis` cites a specific metric movement and ties it to the parameter changed (e.g. "BLOCK_N=32 drove L1 hit rate 62% → 0%, so a larger BLOCK_N restores reuse"). Without the metric chain, the Reviewer must stay diagnostic — no "tune this action's space further" / "wide open" / "only one configuration tried" language. The matching Planner rule: do NOT re-pick a sibling's regressed action unless the Reviewer's current diagnosis grounds a param change in a metric delta.

**Why Moderate, not Strict or Light**:

- **Strict** (never re-propose a failed action, ever) was rejected as too rigid — it rules out legitimate "tried BLOCK_N=32, the L1-hit-rate metric chain says try BLOCK_N=16" moves. The action class isn't the unit of failure; the (action, params) tuple is, and only when the metric story doesn't ground a different params choice.
- **Light** (only ban specific phrases) was rejected as insufficient — the Reviewer would learn to paraphrase. The rule needs to attach to *evidence* (metric → param tie), not vocabulary.
- **Moderate** ties the escape hatch to grounded reasoning: the Reviewer can re-propose the action class iff it shows its work.

**Why one-line per sibling, not the full Reviewer summary**:

Each sibling line is `<action_applied> {<params>}: SOL <score> (Δ <delta vs parent>), <reviewer outcome>, <branch_quality>` — ~80–120 chars. At beam_width=10 the section is ~1.2 KB; comfortably inside DeepSeek-reasoner's 128K window. Rendering each sibling's full Reviewer summary (multi-paragraph `bottleneck_diagnosis` + suggestions list) would inflate this 20–40× without adding decision-relevant signal — the Planner's question is "what action class regressed and by how much," not "what was the Reviewer's full prose on each sibling." Token economy wins; the orchestrator already has the full Reviewer feedback in the trace if a postmortem needs it. Sentinels (`"SOL n/a"`, `"(no review yet)"`, `"(unscored)"`) render still-scoring siblings rather than skipping them, so the Planner sees in-flight siblings too.

**Why prompt-only on both agent sides (no schema changes)**:

`PlannerOutput` and `ReviewerFeedbackOutput` stay as-is. Adding sibling-visibility fields to the schemas would change the SDK strict-schema surface (which already trips on `dict[str, X]` per the option-α migrations in this file) and propagate through every consumer of those types. Prompt-only edits ship as new kwargs on `build_user_prompt` / `plan` / `review`, with the section omitted when `sibling_context == ""` — zero blast radius outside the prompt assembly path. The `repeated_pathway_dead_end` event is added to `CORE_EVENT_KINDS` (alongside the new `sibling_context_rendered` event) for telemetry, but neither event is a verdict — `dead_reason` stays the single source of truth.

**Definition of "regressed sibling"**: SOL delta vs parent < −0.02. The −0.02 threshold matches the existing branch-quality cut in `reviewer/system.md` L61–62; using the numeric delta rather than the Reviewer's free-form `outcome` label avoids depending on prose that varies across runs. Siblings without scores (still-running / errored) are not counted as regressed but still render with sentinels.

**Out of scope (deliberately)**:

- **Plateau-rule changes** (`detect_plateau` window/delta or running-best-vs-per-iter basis) — separate concern from this postmortem; the iter-3 regression would have killed the branch correctly under either plateau policy if the Planner had seen its sibling.
- **`MemoryStore` / `Experience` changes** — within-run sibling memory comes from the live tree, not the cross-run experience store. `MemoryStore` remains a cross-run concern with its own design surface.
- **`_kill_branch` / `DeadReason` enum changes** — `repeated_pathway_dead_end` is an *event* (jq-able from `events.jsonl`), not a new dead reason. The kill path stays `REVIEWER_JUDGED` when the Reviewer's verdict is `dead_end`, including for sibling-grounded kills.
- **Lint events for Planner-picked-failed-sibling-action or Reviewer-suggestion-polarity-violations** — deferred until we observe the prompt rules failing. The risk acknowledgement: the Moderate polarity rule depends on LLM compliance. If postmortems show the model writing metric chains that don't actually tie metric to param, the escalation path is a post-hoc lint event, not tightening the prompt further.

---

## Action Library

### Structured actions over free-form prompts

**Rationale**: All successful frameworks independently discovered that free-form prompts fail — the LLM hallucinates intrinsics, applies incompatible techniques, or makes vague changes. CUDA-Agent (SKILL.md templates), STARK (grounded code-region markers), AutoKernel (6-tier playbook) all solved this the same way: shift the LLM from "figure out what to do" to "correctly apply this specific technique."

### High-level recipes, not code templates

Not as high-level as "optimize memory" (too vague for Coder) and not as low-level as full code templates (too rigid for diverse kernel shapes). The `guidance` recipe lets the Coder adapt to each kernel while staying grounded.

### Reliability over ceiling

Both extremes work — AutoKernel (structured) and AVO/AccelOpt (free-form) both achieve strong results. Structured approaches are more *reliable* (consistent across runs); free-form has higher *ceiling* (can discover novel techniques). We choose reliability as default.

### Spatial grounding via `target_region`

Inspired by STARK's grounded instruction technique (Meta AI/Duke). STARK's ablation showed +20pp success rate and +42% speedup when adding grounding on top of multi-agent coordination alone. Rather than STARK's exact marker format, the Planner includes a `target_region` field — a natural language pointer to the code region the action should apply to. Reviewer validates whether Coder modified the correct region.

### Objective-agnostic actions

Actions themselves don't change when power/ELP modes are added. Only the Planner's selection criteria and scorer change.

### Initial guidance authoring decisions (2026-04-27)

**Trigger**: First live GPU run (rmsnorm, 2026-04-26) showed the Planner picked `t1_block_size_tuning` for all 3 iterations even after the Reviewer flagged each as "regressed." Root cause: every applicable action carried `guidance="Placeholder guidance."` — the Planner had nothing to discriminate on. Authored real guidance for the ~18 actions across `tier{1..6}_*.py`.

**Five design choices (Q&A 2026-04-27)**:

- **Q1 — Format depth**: hybrid (tier1–2 terse ~50 words; tier3–6 structured ~120 words). Prompt-token cost was the constraint; mechanical knobs don't need decision scaffolding, architectural rewrites do.
- **Q2 — Anti-pattern provenance**: mix. Populated where upstream repos (AccelOpt / Astra / autokernel / cuda-optimized-skill / evotoolkit) gave explicit warnings; left empty where they didn't. Hand-fabricating anti-patterns from imagined failures risks anchoring the Planner on non-existent hazards. Sparse-but-grounded beats dense-and-speculative.
- **Q3 — `expected_impact` shape**: qualitative descriptors only ("typically modest", "high-variance, kernel-dependent", "potentially large on memory-bound kernels"). Dropped the rough numeric ranges that were already there as placeholders. Reason: with SOLAR not yet wired and `T_SOL=0.0` in every score event, no real ratio data exists to calibrate ranges; rough numbers actively mislead the Planner toward "expected 3-5x" actions over honest "high-variance" ones.
- **Q4 — Tier asymmetry**: tier-matched density (consistent with Q1c). Same reasoning — guidance density should match the decision surface the Planner faces at that tier.
- **Q5 — Source mining**: own knowledge of GPU kernel optimization for tier1–3 (well-trodden ground); upstream-repo grep for tier4–6 where novel patterns live. The "9-paper KB" referenced in earlier PROCESS.md notes does not exist as a directory; only the 5 upstream repos under `repo/` are mineable.

**Two known limitations recorded as a Deferred Improvement (PROCESS.md → "Action library KB refinement")**: (a) `expected_impact` is qualitative because no real T_SOL data exists yet to calibrate; (b) `anti_patterns` is sparse because no failed-kernel `Experience` corpus exists in `MemoryStore` yet. Trigger to revisit: after SOLAR adapter lands, OR after ≥10 live runs accumulate enough failed-kernel records.

**Why not skip to Q1=(b) structured-everywhere**: prompt-token cost compounds across iterations. The orchestrator filters the registry by `(kernel_type, bottleneck)` per iter, but a typical filter still surfaces 5–10 actions; structured-everywhere would put ~1200 words of action context into every Planner call. Hybrid keeps the typical iter under ~600 words while still arming the Planner with the high-decision-surface tier3–6 detail when those tiers are applicable.

### Action registry: hardware-only structured gating; `preconditions` strings demoted to LLM-visible documentation (2026-05-14)

**Bug behind the change**: the prior `list_applicable(kernel_type, bottleneck)` did `if bottleneck in action.preconditions` — comparing the orchestrator's bottleneck label (e.g. `"memory_bound"`) against the free-text list a Tier-5 action carried (e.g. `["compute_capability >= 9.0"]`). That string is never in that list, so every hardware-gated action quietly failed the filter on the wrong axis, and conversely a Tier-2 action whose `preconditions=["memory_bound"]` actually matched got through for the right reason by accident. The whole `preconditions: list[str]` surface was load-bearing for filtering yet had no enforced schema; "compute_capability >= 9.0" and "memory_bound" coexisted as untyped strings whose only consumer was a substring test.

**What we tried first, then reverted**: a full bottleneck-as-filter design (PROCESS.md's A4 investigation) — fold the bottleneck classification into a typed `applicable_bottlenecks` set per action so the filter actually means what it reads. We backed out because the per-action bottleneck classification is too coarse to be load-bearing: most actions plausibly help under more than one bottleneck, and a kernel's bottleneck label itself shifts iteration-to-iteration as the search moves. Hard-gating on it would cause silent under-recall on actions that would have helped, with no telemetry to notice. We chose to keep bottleneck as Planner-prompt context (which the LLM reads with all the other signal) rather than as a filter.

**What survives**: hardware-only structured gating via the new typed `Action.min_compute_capability: float | None`. `list_applicable(kernel_type, *, hardware=None)` filters on kernel-type match + structured hardware gate only. The `Action.preconditions: list[str]` strings stay on the dataclass but their job is now LLM-visible documentation — they render into the Planner's system prompt as advisory natural-language hints about when a technique fits. Two parallel surfaces with different audiences: the structured field is the enforcement layer (deterministic, default-deny on unknown hardware); the strings are read by the LLM (advisory, never machine-enforced). The dataclass docstring now states this split explicitly so future contributors don't re-conflate the two.

### `HardwareSpec.compute_capability` + Tier-5 `min_compute_capability` (2026-05-14)

**Symptom**: a real run on RTX 6000 Ada (sm_89) picked `t5_h100_wgmma` — a Hopper-only action — and crashed at compile time. The Planner had the catalog and an `available_actions=[]` from the orchestrator, which the Planner's `_validate_and_convert` guard interprets as "no constraint." Anything in the catalog was therefore fair game. The hand-wavy `preconditions=["compute_capability >= 9.0"]` string the Tier-5 action carried was advisory text nobody enforced.

**Two-side fix**: a typed gate (`Action.min_compute_capability`, see entry above) closes the registry side. On the hardware side, `HardwareSpec` grew a `compute_capability: float = 0.0` field that `load_hardware_spec` reads from the YAML and `detect_hardware` populates from `props.major + props.minor / 10` (defensive `getattr` because the test stubs don't always carry both attrs). 0.0 is the sentinel for "unknown" — `list_applicable` treats it as default-deny on any hardware-gated action, which is what we want when running on a host whose probe failed rather than silently letting Hopper actions through. The YAML-vs-detected merge prefers detected ground truth over what `configs/arch/RTX6000Ada.yaml` claims, since the detected value is closest to the actual driver.

Tier-5 populates the structured field: `t5_h100_tma`, `t5_h100_wgmma`, and `t5_hopper_cluster` declare `min_compute_capability=9.0`; `t5_a100_cp_async` declares `8.0`. The third half of the fix is on the orchestrator: `Orchestrator.__init__` now caches `self._action_registry = build_default_registry()`, builds `available_actions` once per run just after `classify_run` by calling `list_applicable(kernel_type, hardware=self._config.hardware)`, and threads the filtered ID list into every `planner.plan(available_actions=...)` call. The empty-list bypass is gone; the Planner's existing validation guard now hard-rejects any out-of-list technique. Hopper-only-on-Ada is closed at three layers (registry filter, orchestrator wiring, Planner validation) rather than relying on any one of them.

---

## Evaluation

### Correctness-first, then profiling

**Rationale**: A fast-but-wrong kernel is never benchmarked. robust-kbench showed that KernelBench can be exploited (output caching, precision degradation, tolerance gaming). The 5-stage gate catches all of these.

### Eval harness split: Coder-side vs orchestrator-side (2026-04-13)

After merging the Debugger into the Coder (giving Coder compile + correctness tools), the eval harness naturally splits into two call sites:

- **Coder-side** (via `function_tool`): `compiler.py`, `correctness.py`, `anti_cheat.py`. Run inside the Coder's turn. By the time the Coder returns, the kernel is compiled and correct.
- **Orchestrator-side**: `benchmark.py`, `profiler.py`, `roofline.py`, `scorer.py`. Run by the orchestrator after the Coder returns. The Coder never sees benchmark/profiling results directly — this prevents the LLM from gaming latency numbers.

**Why not give the Coder benchmark tools too**: The Coder should optimize for correctness, not for benchmark numbers. If the Coder could benchmark, it might overfit to specific input sizes or learn to game the measurement. Keeping benchmark/profiling orchestrator-only maintains the separation: the Coder writes correct code, the eval harness measures it, and the Reviewer interprets the results.

### SOL-ExecBench benchmarking integration — current protocol kept, `do_bench`-shape deferred (2026-04-20)

Surveyed `/home/hel19/workspace/projects/self-evolved-llm/repo/benchmark/SOL-ExecBench` in response to a Codex adversarial review that flagged our per-iteration timing shape as vulnerable to CUDA sticky-error contamination. Their canonical timer (`src/sol_execbench/core/bench/timing.py::do_bench`) pre-allocates `rep` start/end `torch.cuda.Event` pairs upfront, runs the warmup + timed loops with one `torch.cuda.synchronize()` before each `start.record()` and a single global sync after the timed loop, then computes `start.elapsed_time(end)` for each pair. Their isolation model (`src/sol_execbench/driver/templates/eval_driver.py`) is **per-solution subprocess**, not per-workload — inside the subprocess, between workloads they do only `gc.collect()` + `torch.cuda.empty_cache()` + explicit tensor-ref cleanup. Per-workload subprocesses are not their answer to sticky CUDA errors.

**Decision**: Keep the current `BenchmarkTimer` protocol (`prepare` / `flush_l2` / `record_start` / `record_end` / `finalize_ms` per iter) for now, and fix Codex's findings in place (fail-closed on baseline partial-workload failures; fresh timer instance per workload). Defer the `do_bench`-shape rewrite as its own phase item.

**Why defer**: adopting the `do_bench` shape requires redesigning the `BenchmarkTimer` Protocol, since the torch-free test venv injects a `RecordingTimer` that asserts the per-iter call order — 12 tests in `tests/test_benchmark.py` depend on it. The replacement seam (e.g. `BenchmarkTimer.time(fn, setup, warmup, rep) → list[float]`, or a `pre-allocate events + iterate + collect` trio) needs its own design discussion so tests keep a torch-free injection point while matching the upstream shape. The per-iter sync cost we'd save is not yet on-profile — production `rep` counts haven't run against live CUDA. Pay once when GPU runs prove the cost, not proactively.

**Why not subprocess-per-workload** (Codex's recommendation): SOL-ExecBench doesn't do this either. Their answer is subprocess-per-*solution* (belongs at `pipeline/optimize.py` level, already tracked as a deferred Tier 3 item) plus lightweight per-workload cleanup inside the subprocess. Subprocess-per-workload would add hundreds of ms per workload × iterations × candidates — a large architectural cost to solve a problem that currently degrades gracefully (a sticky CUDA fault drops survivors → `BenchmarkError` → child DEAD_END).

### Profiling feedback pipeline — full → Reviewer, distilled → Planner

Reference frameworks handle this differently:
- AccelOpt: filters aggressively via config file (Planner often sees only latency)
- Astra: passes ALL profiling data + pre-computed interpretation
- AutoKernel: writes results to disk, agent reads on-demand

We chose hybrid: Reviewer gets all raw profiling data (NCU metrics, latency, cache rates, stall reasons). Reviewer produces structured summary. Planner receives only the summary.

**Why not pass everything to Planner directly (Astra-style)**: AccelOpt found that filtering improves planning quality — LLMs get confused by too many metrics. The Reviewer acts as an intelligent filter: it can surface unexpected metrics when relevant (e.g., "spill rate spiked to 15%") while suppressing noise, which a static config file cannot do.

### Profiling tool choice

Since we target Triton on NVIDIA, we use CUDA Events for latency (lightweight, accurate) and NCU for deep hardware profiling (standard NVIDIA tool). AccelOpt uses `neuron-profile` (NKI-specific), Astra uses CUDA Events + NVML + PyTorch profiler, AutoKernel uses Triton's `do_bench()` + roofline, SwizzlePerf uses `rocprofv3` (AMD).

### Hardware specs — detect internally, don't expose to agents

**Rationale**: No reference framework passes raw hardware specs to the LLM agent. Profiling metrics are more actionable ("L2 cache hit rate = 40%" tells the agent what's wrong) than raw specs ("L2 cache = 50 MB" requires reasoning about working set sizes). LLMs also hallucinate hardware details.

Detection → internal roofline analysis → Reviewer sees profiling + roofline classification → Planner sees Reviewer's distilled summary. Fits the profiling feedback pipeline above.

### Profiler approach: analytical classification + curated NCU section (2026-04-20)

**Context**: `eval/profiler.py` is the next module. PRD §Evaluation Harness specifies NCU with `--set full` per iteration to produce dynamic bottleneck classification + rich metrics (occupancy, stall reasons, cache hit rates, throughputs). Survey of reference repos showed nobody actually runs NCU per candidate:

- **autokernel/bench.py:1072-1082** derives bottleneck analytically: `AI = flops/nbytes`, `ridge = peak_TFLOPS/peak_BW`, classify by `AI < ridge`. Zero-overhead because `flops`/`bytes` and kernel latency are already in hand.
- **AccelOpt/scripts/planner.py:45-55** runs a domain profiler once per candidate, dumps all metrics to a JSON blob, then applies a **`displayed_profiles` whitelist** when constructing the Planner prompt. Collection is broad, surface is narrow and curated.
- **Astra / SOL-ExecBench**: no NCU surface at all. SOL-ExecBench is CUDA-event timing only.

**Decision**: Hybrid — analytical classification every iteration (free), plus NCU `--section` with a curated 4-metric set (occupancy, warp stall reasons, L2 hit rate, tensor-core utilization). `--set full` becomes an opt-in debug mode.

**Why (b) curated over (c) full**:

1. **Cost.** NCU uses kernel replay; `--set full` is ~2-5 s per candidate on a Triton kernel. At beam 3 × depth 20 = 60 candidates → 2-5 min of pure NCU overhead per problem, on top of benchmark + 3 LLM calls per iter. Curated `--section` is ~5-10× cheaper for the same action-relevant signal.
2. **Signal-to-noise.** `--set full` produces 60+ metrics. The PRD itself already routes profiling through the Reviewer as an "intelligent filter" — collecting the full set is work we throw away at the prompt boundary. AccelOpt converged on the same whitelist pattern.
3. **Action-library alignment.** Every curated metric earns its keep by mapping to a tier: occupancy → Tier 1 (sizing); stall reasons → Tier 2/3 refinement; L2 hit rate → Tier 2 (tiling); tensor-core util → Tier 3 (mixed precision). Metrics without a Planner-visible action don't enter the curated set.
4. **Graceful degradation.** Analytical classification is computed independently of NCU. If NCU fails (missing `ncu` binary, permissions, subprocess crash, timeout), the Reviewer still gets a bottleneck classification and continues — same fail-closed pattern as `BenchmarkResult.is_fully_successful`.

**Escape hatches**: (a) `ProfilingResult.raw_metrics: dict[str, float]` stores whatever NCU actually returned, so a future Reviewer/prompt can reference a metric without a code change. (b) `ACTS_PROFILER_MODE=full` (or config flag) upgrades a single run to `--set full` — useful when prompt-engineering the Reviewer or investigating a puzzling candidate. (c) Per-source-hash caching (same hash key as `kernels/compiler.py`) — re-profiling the same kernel source is wasteful.

**NCU invocation mechanism — subprocess**. Three options were weighed: (i) `ncu --csv --section <list> --export <out>` subprocess + CSV parse; (ii) `nsight-compute` Python API (in-process); (iii) ship analytical-only, defer NCU. Chose (i) for three reasons:
1. **Portability + isolation match ACTS's fail-closed design** — an NCU crash, hang, or missing-binary can't take down the orchestrator. SOL-ExecBench's `eval_driver.py` uses the same subprocess-isolation pattern for correctness, so this is a proven shape in this ecosystem.
2. **CSV output is stable across CUDA versions** in a way the `nsight-compute` Python API isn't; the Python API is NVIDIA-proprietary and moves between CUDA releases.
3. **Subprocess launch overhead (~100-300 ms) is small compared to replay cost (500-2000 ms)** — the dominant cost is NCU itself, not how we invoke it. Optimizing the wrapper is premature.

(iii) was tempting as a "ship today" move but rejected because the 4 curated metrics (occupancy, stall reasons, L2 hit rate, tensor-core utilization) aren't derivable from anything else — deferring them means the Reviewer never gets the signals that disambiguate Tier 2 from Tier 3 actions.

**Curated metric set — 4 NCU-only signals**. Analytical overlay produces achieved TFLOPs, achieved GB/s, arithmetic intensity, ridge point, and the memory-vs-compute-bound classification (all from SOLAR's `flops`/`bytes` + measured latency, zero NCU cost). NCU is therefore reserved for signals analytical can't derive:

| Bucket | NCU section | Metric | Action tier informed |
|---|---|---|---|
| Occupancy | `Occupancy` | `sm__warps_active.avg.pct_of_peak_sustained_active` | Tier 1 (block/grid sizing) — low occupancy → oversized blocks or register pressure |
| Warp stall reason | `WarpStateStats` | Dominant stall class (`stall_long_sb`, `stall_short_scoreboard`, `stall_no_instruction`, …) | Tier 2 vs Tier 3 disambiguation — memory stall → Tier 2; exec-dependency stall → Tier 3 |
| L2 hit rate | `MemoryWorkloadAnalysis` | `lts__t_sector_hit_rate.pct` | Tier 2 (tiling, shared-memory caching) — low L2 hit → reuse opportunity |
| Tensor core utilization | `ComputeWorkloadAnalysis` | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | Tier 3 (mixed precision) — low TC util on compute-bound → TF32/BF16 headroom |

Three sections total (`Occupancy`, `WarpStateStats`, `MemoryWorkloadAnalysis` + `ComputeWorkloadAnalysis`). Dropped sections: `SchedulerStats`, `SourceCounters`, `InstructionStats` — no direct action-tier mapping. Dropped from the placeholder `ProfilingResult`: `memory_throughput_gb_s` and `compute_throughput_tflops` — moved to analytical, which frees NCU from running duplicative sections.

**Workload scope — representative-in-loop, all-at-terminal**. Benchmark runs multi-workload (median-of-medians across 2-3 SOL workloads) per PRD. Three options for profiler granularity were weighed: (a) profile all selected workloads (matches benchmark; 2-3× NCU cost per candidate); (b) profile one representative workload (cheap but hides shape-dependent bottleneck shifts); (c) representative-in-loop + full-suite at terminal nodes. Chose (c):
- **In-loop**: profile `workload[0]` only (the representative already chosen for `benchmark[0]`). Keeps per-iteration NCU cost ≤ ~500 ms and matches the fact that search iteration is where cost dominates.
- **Terminal/Phase C**: when the search terminates, profile the winner on all selected workloads so Phase C's `bottleneck_transitions` is computed from real multi-workload data — same philosophy as Phase C already re-running the full workload suite on the winner.

This matches the cost shape of the rest of the pipeline: cheap per-iteration signal, rich one-shot reporting.

**Failure taxonomy — NCU degrades the signal, analytical failures kill the branch**. Core principle: analytical classification is the floor (required for downstream retriever/reviewer/`bottleneck_transitions`); NCU metrics are the bonus. Failure modes:

| Failure | Cause | Outcome |
|---|---|---|
| NCU binary missing | `ncu` not on `$PATH` / CUDA toolkit not installed | Log once at startup, mark profiler NCU-disabled, every candidate gets analytical-only `ProfilingResult`. Orchestrator continues. |
| NCU subprocess crash / non-zero exit | Segfault, OOM, signal | Per-candidate log, fall back to analytical-only for that candidate. Branch NOT killed — analytical signal is still valid. |
| NCU subprocess timeout | Hang on malformed kernel | Kill subprocess, log, fall back to analytical-only. Branch NOT killed. |
| NCU CSV parse failure | Unexpected format (new CUDA version, partial output) | Log, fall back to analytical-only. Branch NOT killed. |
| Analytical computation failure | Missing `flops`/`bytes` (shouldn't happen post-SOLAR) or zero latency | Branch IS killed — classification is required downstream. Matches `BenchmarkResult.is_fully_successful`'s fail-closed contract. |

> *Superseded 2026-05-13 (analytical-missing row).* The "branch IS killed" outcome no longer holds for the missing-flops/bytes case. `profile_kernel(nbytes=0)` now sets `analytical=None` and continues with NCU alone; the orchestrator's `(flops > 0 and nbytes > 0)` gate was removed. Bottleneck classification was already decoupled from analytical at 2026-04-22 ("Bottleneck classify-once"), so the "required downstream" justification no longer applies. Zero-latency still kills the branch via the existing `BenchmarkResult.is_fully_successful` path. See "a+b decoupling — SOLAR counts surfaced, analytical roofline tolerates missing inputs (2026-05-13)" above.

Subprocess timeout default: **30 s per candidate** (covers `--section` replay for reasonable kernel sizes; malformed/hung kernels are killed fast enough not to stall the search). Configurable via `profiler_timeout_s`.

**Cache layout — source-hash keyed, no eviction**. Same pattern as `kernels/compiler.py`'s compile cache:
- Directory: `~/.cache/acts/profiler/` (override via `ACTS_PROFILER_CACHE_DIR`).
- Key: source hash (reused from the compiler cache) + metric-set version string. The version suffix invalidates stale entries when the curated metric list changes, so a metric table edit auto-busts the cache instead of silently serving old results.
- Value: JSON-serialized `ProfilingResult`.
- Eviction: none initially. Each result is ~1 KB; a 10k-candidate history is ~10 MB. Add LRU only if the cache shows up in a profile or on-disk footprint becomes a concern.

**Stall-class extraction — top-1 + runner-up, not top-3**. The `WarpStateStats` section emits ~10 stall classes; only the dominant one drives a concrete action, but borderline cases matter too. Surface:
- `warp_stall_dominant: str` (e.g., `"stall_long_sb"`) + `warp_stall_dominant_pct: float`
- `warp_stall_runner_up: str` + `warp_stall_runner_up_pct: float`

Rationale: top-1 tells the Reviewer which tier to target (stall-memory-throttle → Tier 2; stall-exec-dependency → Tier 3); runner-up catches mixed cases ("stall-memory 32%, stall-exec 29%" → don't commit to a single tier). Top-3 adds a metric the Reviewer rarely acts on, dilutes prompt signal, and duplicates information already preserved in `raw_metrics` for anyone who needs it.

**Real-GPU tests required when a GPU is available (process decision, 2026-04-20)**. Fake-`ncu` subprocess tests (shell script on `$PATH`) cover every failure path in the driver cheaply, but they cannot catch (a) NCU metric-name drift between CUDA versions, (b) whether curated sections are available on the target GPU architecture, (c) whether `--kernel-name regex:<entrypoint>` actually matches Triton's mangled kernel names, or (d) whether the subprocess driver imports and launches correctly. On a GPU-equipped dev machine, a "manual smoke script not in CI" is a dodge — if the machine can run the test, the test is required.

**Done gate for `eval/profiler.py`**:
1. Tier 1 (GPU-free, fake-`ncu`, 5 test files) passes in `~/.venvs/acts_test_venv`.
2. Tier 2 (`tests/test_profiler_gpu.py`, `@pytest.mark.gpu`, real `ncu` on the RTX 6000 Ada / CUDA 12.8 host) passes locally.
3. Codex + user review clean.

Tier 2's test list covers add-kernel + matmul correctness of classification, the Triton kernel-name regex (the single silent-failure risk), cache-hit-skips-ncu, full-mode raw-metrics population, and Phase C multi-workload re-profile.

**Broader principle — applies to all future modules touching GPU/CUDA/NCU**: if the dev machine can run the test, the test is required for "done." `@pytest.mark.gpu` skips cleanly on GPU-less CI but is expected to run locally before commit. This rule is recorded in auto-memory (`feedback_gpu_tests_required.md`) so future modules (`eval/anti_cheat.py`, `benchmark/solar_adapter.py`, first-live-GPU-run) don't get the same dodge attempted.

Updates the design intent referenced in "Dynamic bottleneck reclassification — deferred to profiler implementation (2026-04-15)" below — that entry described *what* to wire dynamically; this one describes *how* the profiler produces the signal. The original standalone design spec (`docs/superpowers/specs/2026-04-20-eval-profiler-design.md`) was deleted after it diverged from the implementation — see "NCU subprocess reality check" below for what was actually built, and "Bottleneck classify-once (2026-04-22)" for why the per-iter reclassification plan was reversed.

### Profiler implementation — NCU/driver divergences from the design spec (2026-04-21)

**Context**: `eval/profiler.py` implementation probed real `ncu` (2025.1.1.0) on RTX 6000 Ada / CUDA 12.8. Several spec assumptions didn't survive first contact. Recording here as first-class facts so the next person touching the profiler doesn't re-discover them, and so the spec's silences aren't re-inherited by future modules that shell out to NCU.

**NCU invocation — command-line shape the spec got wrong**:

1. **Raw metric names require `--print-metric-name=name`**. Default is `label` (human-readable "Achieved Occupancy"), which varies with locale + NCU version. The dotted raw form (`sm__warps_active.avg.pct_of_peak_sustained_active`) — the only form our parser keys off — is emitted only when this flag is passed. The spec's command shape omitted it; a curated-metric mismatch silently degraded every run.

2. **Stall metrics aren't in any `--section`; they must be requested via explicit `--metrics`**. Wildcards (`_*.pct`) do NOT expand — all 18 stall reasons must be enumerated. The correct family is `smsp__average_warp_latency_issue_stalled_<reason>.pct` (singular "warp", not "warps"; "latency" not "latencies"). The spec's `smsp__average_warps_issue_stalled_*` prefix would emit no metrics.

3. **Subprocess must use `sys.executable`, not bare `"python"`**. NCU forks with the caller's environment, but bare `python` PATH-resolves to whichever interpreter is first on `$PATH` — rarely the venv with torch/triton. Failure mode is silent: `ModuleNotFoundError: No module named 'torch'` lands in the driver subprocess stderr, which NCU doesn't capture — the operator sees only `==ERROR== The application returned an error code (1)`. Would have broken every non-system-Python install.

4. **NCU stdout isn't pure CSV**. It's CSV prefixed with `==PROF== Connected...` banner lines and interleaved with the profiled process's own stdout (e.g. `ok\n`). Parser must skip non-CSV lines rather than assume well-formed stdout.

5. **Numeric values can be comma-formatted** (`"5,000.00"`). Parser strips commas before `float()`.

6. **Tensor-core util metric isn't universal**. `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` is absent from `ComputeWorkloadAnalysis` for pure-memory kernels on NCU 2025.1.1.0. The spec marked it required — would have crashed every memory-bound candidate. Demoted to `_CURATED_OPTIONAL` (defaults to 0.0 when missing). The other three curated metrics remain required.

7. **Dev-only: `/tmp/nsight-compute-lock` can be owned by another user** on shared hosts with sticky bit on `/tmp`. Workaround: `mkdir -p /tmp/<user>_ncu && TMPDIR=/tmp/<user>_ncu ncu ...`. Not a CI concern (single-user runners); documented for Tier 2 fixture setup.

**Subprocess contract — two-name, self-contained kernel convention**:

8. **`spec.entrypoint` is the host-wrapper name, not the GPU kernel symbol**. Triton's `@triton.jit` function can't be called as `fn(*args)` — it requires `fn[grid](*args)` and raises "Cannot call @triton.jit'd outside of the scope of a kernel" otherwise. Convention: every kernel source exposes `def run(...)` as the host wrapper that builds the grid and launches the JIT'd function. The driver calls `module.run`. Meanwhile NCU's `--kernel-name regex:` targets the *GPU symbol*, extracted from source via `_extract_triton_kernel_name()` (regex against `@triton.jit\s*(?:\(...\))?\s*\n\s*def\s+(\w+)`), falling back to `spec.entrypoint`. Two roles, two names — the KernelSpec didn't need a schema change, but the driver has to keep them separate.

9. **Inputs must be rebuildable from pickle-safe state, not an in-process closure**. The parent's `input_generator` closure can't cross the subprocess boundary — arbitrary closures don't pickle. Driver input-resolution priority: (a) `problem_dir` → `load_problem(dir)` + `build_input_generator(problem, workload)(seed)`; (b) `module.make_inputs(seed)` if the source exposes it; (c) `spec["args"]` as a last-resort literal; (d) `()`. The in-process `input_generator` parameter to `profile_kernel()` is intentionally discarded at the subprocess boundary (documented with `_ = input_generator`) — it's retained in the signature only for API symmetry with the non-subprocess callers. Mirrors Astra/AccelOpt's self-contained-kernel pattern.

10. **`load_problem()` expects a directory, not a file**. A late Codex finding: the spec key `problem_json` implied a JSON file path, and the serializer passed `problem.definition_path` (which points at `definition.json`). But SOL-ExecBench's `load_problem(path)` expects the directory containing `definition.json` + the sibling `workload.jsonl` — it does `path / "definition.json"` internally. Passing the file path made the driver try to open `<definition.json>/definition.json` and every SOL NCU run silently degraded to analytical-only. Fix: renamed the spec key to `problem_dir`, serialize `Path(problem_definition_path).parent`, and added a Tier 2 `test_profile_with_problem_definition_path_is_not_degraded` that would have caught this end-to-end — Tier 1 fake-`ncu` can't, because the fake never execs the driver.

**Cross-cutting lesson**: NCU is a subprocess that forks another Python subprocess (our driver). Two hop points, each with its own environment/argv/stdout discipline; each of the above was a silent failure (degraded run, empty metric set, wrong kernel, wrong file open) rather than a loud crash. Cost-of-detection is exactly why the `feedback_gpu_tests_required.md` rule exists — Tier 1 fake-`ncu` tests green up clean through every one of these bugs. The `@pytest.mark.gpu` suite is the only layer that forces the two hops to actually run end-to-end.

**Spec supersession**: `docs/superpowers/specs/2026-04-20-eval-profiler-design.md` diverged too far from the implementation (items 1-3, 6, 9-10 contradict the spec; item 7 wasn't there). Canonical design rationale lives in this JOURNAL entry + the `Profiler approach` entry above. The spec file is deleted in the same commit — no SUPERSEDED marker, since `docs/superpowers/specs/` has no other residents to preserve a convention for.

### SOL-ExecBench upstream API shapes (2026-04-29)

**Rationale**. Three SOL upstream API shapes are load-bearing at our integration boundary and not obvious from the field names alone. Each was discovered the hard way (validator failures or test reds) and warrants a single canonical record so a future contributor doesn't re-derive them.

**What changed.** No code change — this is a documentation entry capturing the shapes that already drive `src/eval/anti_cheat.py`, `src/search/orchestrator.py::_emit_trace`, and the `Trace`/`Evaluation`/`Performance` plumbing.

The shapes:

1. **`compute_error_stats(candidate, reference, spec) -> (Correctness, exceeds: bool)`**, not a single object with a `.passed` attribute. Verified at `src/eval/anti_cheat.py::strict_tolerance_check` line 83: `_correctness, exceeds = compute_error_stats(...)`. The `exceeds` boolean is the negation of "passed" — `return not exceeds` to get a pass-style flag. Mistaking this for `.passed` produces an `AttributeError` at the first tolerance check, not a silent miscompute.
2. **`Trace.definition` is `NonEmptyString`**, not a sub-model. We pass `definition.name` (a string) — passing the `Definition` object itself trips Pydantic's `NonEmptyString` validator. Verified at `src/search/orchestrator.py::_emit_trace` (`def` at line 1103); construction at line 1165 with `definition=definition.name` at line 1166.
3. **`Trace.evaluation` nests `Performance(latency_ms=, reference_latency_ms=, speedup_factor=)`**, with the `_ms` suffix and `speedup_factor` (not `latency_us` or `speedup`). Our internal benchmark numbers carry microseconds — convert at the SOL boundary: `latency_ms=bench.median_latency_us / 1000.0`. Verified at `src/search/orchestrator.py::_emit_trace` lines 1159–1163 (Performance kwargs). The mismatch would be a Pydantic validation error at `Trace` construction (caught in the broad `except` and silently swallowed), so a contributor who renames our internal fields to match SOL's surface needs to update both the `_us → _ms` divide and the `speedup → speedup_factor` rename.

**What we explicitly did NOT do.** No abstraction layer over these shapes. They're not re-exported from a `src/sol_compat.py` shim — the field-rename + type-cast lives at the single call site that needs it. Adding a shim would just push the rediscovery cost into a new layer.

**Trigger for revisit.** SOL upstream renames any of these fields (e.g. `compute_error_stats` returns a single `CorrectnessResult` with a `.passed` attribute, or `Performance.speedup_factor` is renamed `speedup`). Fix at the integration boundary (`anti_cheat.py::strict_tolerance_check`, `orchestrator.py::_emit_trace`) — do not propagate the SOL renames into ACTS-internal field names.

### Subprocess driver integration tests required, not just unit tests (2026-04-29)

**Rationale**. The `_profiler_driver.py` subprocess (NCU → forked Python child) was the silent-failure capital of the profiler PR series. Tier 1 unit tests with fake `ncu` greened up clean through every single bug listed below — none of them surfaced until either an adversarial Codex review or a real-GPU integration run forced the two hops to actually execute. Parallel to the `feedback_gpu_tests_required.md` rule, but narrower: any subprocess driver added to ACTS needs end-to-end integration coverage, not just unit-level mocks of its inputs and outputs.

**What changed.** No code change — this is a lesson-capture entry pointing at the bug history so future subprocess drivers (e.g. clock-lock probe, isolated correctness runner) get an integration test from day one rather than after a real-GPU red.

**Bug history that motivates the rule.**

- **F1 — `sol_load.load(...)` shadowed by an inner import**. Adversarial Codex caught a name shadow that made the driver call a stub from a sibling import path instead of the real SOL loader. Tier 1 mocked at the wrong boundary; the import-shadow only fires when the real module is on `sys.path`.
- **G2 — `blob_roots` not forwarded to the subprocess spec**. Workloads that load `safetensors` blobs from a base directory degraded to empty inputs because the parent's `blob_roots` configuration didn't ride along in the JSON spec. Tier 1's synthetic inputs hid this — only a real workload with blob refs surfaced the empty-tensor case.
- **G4 — `kernel_fn(*inputs)` without DPS branch**. The driver called the kernel as a value-returning function and assumed the return value was the output tensor. For destination-passing-style kernels (where the output tensor is *passed in* and the function returns `None`), the driver captured `None` as the output and downstream tolerance checks compared `None` to the reference. Tier 1 only tested value-returning kernels; the DPS branch needed a real DPS Triton kernel to surface.

**What we explicitly did NOT do.** No "Tier 1.5" mock-the-subprocess-shape test layer. The lesson is not "add more mocks" — every one of the bugs above lives at a boundary the mocks specifically can't reach (real loader, real blob filesystem, real kernel return convention). Either Tier 2 (`@pytest.mark.gpu` end-to-end) or an explicit subprocess integration test that runs the actual driver script with a real Python interpreter.

**Trigger for revisit.** Any new subprocess driver added to ACTS — not just GPU-bound ones. The two-hop pattern (parent → subprocess → forked child) is the failure mode, regardless of whether the inner hop is `ncu`, `nvprof`, a Docker container, or a separate Python process for clock-lock. First commit of the new driver gets a Tier 2 integration test in the same PR; review will block on its absence.

### Python 3.10 → 3.12 bump for SOL-ExecBench (2026-04-29)

**Rationale**. SOL-ExecBench pins `requires-python = ">=3.12"`. The pre-SOL ACTS venv was Python 3.10. Three install paths were considered for getting 3.12 onto Ubuntu 20.04 (the dev host); only one worked end-to-end without `sudo` rights or an OS upgrade.

**What changed.** `pyproject.toml` `requires-python` bumped from `>=3.10` to `>=3.12`. Production venv path is `/tmp/acts_run_venv` built via `uv venv --python 3.12`; the older `/tmp/acts_test_venv` (3.10, no torch) stays for torch-less unit tests. Canonical install recipe lives in `configs/venvs/3.12.md`.

**Install path comparison.**

- **deadsnakes PPA (rejected — no packages)**. The natural first try on Ubuntu 20.04 — `add-apt-repository ppa:deadsnakes/ppa` historically backports modern CPython. apt resolution returned no Python 3.12 packages for 20.04 by the time of the bump; deadsnakes had moved on. The 2026-04-22 SOL integration tightening entry above still shows the deadsnakes-based recipe — that recipe was correct as of 04-22 but stopped working before the install actually happened.
- **System upgrade (rejected — out of scope)**. Bumping the OS to 22.04+ to get a `python3.12` apt package would have shipped 3.12, but a host upgrade is wildly out of proportion for a venv requirement. Vetoed.
- **`uv` (chosen — userspace, no `sudo`)**. `uv` ships its own CPython distributions and auto-fetches 3.12 on first `uv venv --python 3.12` invocation. No PPA, no `sudo`, no OS upgrade. Same recipe also handles torch (cu128 wheels via `--index-url`) and SOL-ExecBench (editable + `--no-deps` to skip the cu13-only deps that don't have cu128 wheels).

**What we explicitly did NOT do.** No conda. No pyenv. No system-level Python install. The dev-host invariant ("no `sudo` modifications") makes `uv` the only path that works, and the resulting venv is portable enough that a future host bump (20.04 → 22.04) won't require redoing the recipe.

**cu128 wheels + `pip install -e SOL --no-deps` rationale**. Already documented in `configs/venvs/3.12.md` ("Why `--no-deps`"): SOL's `pyproject.toml` lists `cuda-tile==1.1.0`, `nvidia-cutlass-dsl[cu13]==4.4.1`, and `nvidia-cudnn-frontend==1.18.0` (cu13-paired build) — all cu13-only with no cu128 wheels. Letting pip resolve them either fails outright or pulls cu13 wheels that crash on the first GPU op. ACTS doesn't use the SOL features that need any of these (CUTLASS / cuTe DSL / cuTile target Blackwell sm_100+, ACTS targets Ada sm_89), so the `--no-deps` install + hand-curated `pip install pydantic safetensors click rich pyyaml pytest pytest-asyncio` sidesteps the entire cu13 surface.

**Trigger for revisit.**
- SOL upstream relaxes `requires-python` back to `>=3.10` (unlikely — they're on the modern-Python track). Then the dual-venv split (`/tmp/acts_test_venv` for unit tests, `/tmp/acts_run_venv` for live runs) collapses back to a single 3.10 venv.
- A SOL feature ACTS wants pulls in cuda-tile / cutlass-dsl / cudnn-frontend for real. Then either the cu13 stack arrives on Ubuntu 22.04+ (host bump) or ACTS stays scoped to the cu128-compatible SOL surface.

### NCU `.ncu-rep` capture + CSV extraction — two-subprocess architecture (2026-05-08)

**Rationale.** Commit 166d697 added `-f -o <ncu-rep-path>` to NCU's argv to capture binary `.ncu-rep` files alongside the JSON cache. On NCU 2025.1.1.0 (the host), passing `-o` *suppresses* the CSV stream from stdout — it's not redirected, it's not delivered anywhere, it just stops. Stdout reduces to four `==PROF==` banner lines. `_parse_ncu_csv` strips banners, finds zero CSV rows, emits `csv_parse:no_header`. The 2026-05-03 live run silently degraded (no stalls reported, no clear signal why); the 2026-05-08 live run showed the same shape, but loud — the diagnostic emission added in profiler.py earlier today made `csv_parse:no_header` greppable from `run.log`. NCU's CLI provides no flag combination that delivers both binary `.ncu-rep` and CSV stdout from a single invocation: `--csv` + `-o` ≠ both outputs, `--log-file` redirects the (banner-only) stdout to a file rather than peeling CSV off, and there's no `--csv-file` flag. Verified by direct repro: `-o` alone → 3 banners + binary; `-o + --log-file <path>` → 3 banners in `<path>` + binary (no CSV anywhere); `ncu --import <rep> --csv` → full CSV with the parser-expected `Kernel Name` / `Metric Name` / `Metric Value` columns.

**What changed.** `_run_ncu` now performs two subprocess calls per profile:

1. **Profile** — existing argv with `-f -o <rep_path>`; produces the binary `.ncu-rep` and a banner-only stdout that we discard. The capture path from 166d697 stays unchanged.
2. **Extract** — `ncu --import <rep_path> --csv --page details`; stdout is a CSV with the columns `_parse_ncu_csv` already understands. Feed that into the existing parser.

`_parse_ncu_csv` signature unchanged. Mock NCU shell scripts in `tests/test_profiler_subprocess.py` and `tests/test_profiler_cache.py` now honor `--import <rep>` mode (when `--import` is in argv, emit the cached CSV; otherwise capture mode emits banners + writes a marker rep file).

**Settled design choices.**

- **Two subprocess calls, sequential.** Per-profile cost is one extra `ncu` fork (~hundreds of ms; no GPU work in the import call — just binary→CSV serialization). At ~3 profiles per iter × ~30 iters per run, the tax is ~20–45s on a 5–15 min run, <5% wallclock. Bearable, especially since the binary report itself is irrecoverable once thrown away.
- **`.ncu-rep` capture stays.** Useful for: `ncu-ui` postmortem (source-level attribution, all sections, raw counters), the deferred Reviewer Variant B feature (`reprofile(sections, metrics)` — re-extract sections via `ncu --import` without re-running the GPU profile), audit-trail completeness for the search-tree-recording feature.
- **`ncu --import` runs in the same `_run_ncu` call site.** Encapsulated as `_extract_ncu_csv(rep_path) -> str`, a pure shell-out helper. Failure mode if extract subprocess fails: distinct degradation slug `ncu_import_failed:<rc>` (separate from `csv_parse:no_header` which is parser-side; this one is "the binary is there but post-processing the binary failed").
- **Mock fixture: modify both existing mocks** (`tests/test_profiler_subprocess.py` + `tests/test_profiler_cache.py`) to honor `--import <rep>` mode rather than adding parallel fixtures. One source of truth for "what real NCU does."
- **Regression-guard test**: assert the parser would fail with `csv_parse:no_header` if the `--import` extract step were skipped. Pins the failure mode against a future "is this second call redundant?" cleanup.
- **No NCU-version probe**: two-subprocess path runs unconditionally. Single-host dev box; pre-2025 NCU isn't a target. Keeps the code one-branch.

**Why not Option A (`--log-file <path>` alongside `-o`).** The earlier entry that proposed this was wrong. Empirically, NCU 2025.x's `--log-file` redirects whatever stdout *would have been* — and when `-o` is set, that's just banners. The flag has no awareness of "CSV stream that didn't get printed because of `-o`." Verified directly: `ncu --csv -f -o caseB --log-file caseB.log <kernel>` produces a 651KB `.ncu-rep` plus a 3-line `caseB.log` containing `==PROF== Connected`, `==PROF== Disconnected`, `==PROF== Report: <rep>`. No CSV. The diagnosing subagent's earlier conclusion was inferred from `--log-file`'s help text ("Send all tool output to the specified file") rather than tested with `-o` and `--log-file` set together. The Option-A working-tree implementation passed unit tests because the mock honored the *desired* contract; against real NCU, all 6 GPU profiler tests failed with `csv_parse:no_header`. Cost of the wrong path: one JOURNAL amendment + one re-implementation pass. Lesson recorded in `feedback_repro_flag_combinations.md` (auto-memory).

**Why not Option C (drop `-o`, single subprocess, no `.ncu-rep`).** Loses the search-tree-recording feature's `tree/node_<id>/ncu.ncu-rep` capture entirely (the very motivation for 166d697) and forecloses the deferred Reviewer Variant B. Saves the second subprocess (~5% wallclock) but trades irrecoverable artifact loss for it. Considered briefly; rejected.

**Why not Option D (NCU Python/SDK API: read binary in-process).** Would deliver single-subprocess + both outputs, but adds a CUDA-toolkit-version-coupled Python dependency (`ncu-report` module ships with the toolkit and its API surface is not stable across major versions). Brittle for a long-lived dev tool. Considered; rejected.

**Trigger for revisit.**

- NCU 2026.x changes the `-o`-suppresses-stdout behavior or adds a `--csv-file` flag. Watch `ncu --help` on toolkit upgrades.
- The `_extract_ncu_csv` cost shows up in real run profiling — at that point the cu-toolkit Python SDK option (D) becomes worth its dependency cost.
- Reviewer Variant B lands. The `ncu --import` invocation logic ends up in two places (profile path + reprofile path); consolidate into a shared `_run_import_csv(rep_path, sections, metrics)` helper.

**Amendment 2026-05-08 (post-Codex review) — `_extract_ncu_csv` propagates the same `TMPDIR` env as `_run_ncu`.** The original two-subprocess implementation passed `env=...` to the capture subprocess (carrying the user-scoped `TMPDIR` workaround for the `nsight-compute-lock` ownership issue that hits on shared `/tmp`), but the new `_extract_ncu_csv` subprocess inherited the process default `/tmp`. On hosts that need the workaround, capture would succeed but import could fail after — undoing the gain. Symmetric fix: thread the same `TMPDIR` env into `_extract_ncu_csv`'s `subprocess.run`. Lesson: when adding a sibling subprocess to an established pattern, audit the existing call's `env`/`cwd`/`stdin` parameters and replicate; default-inherit is not safe for ones with operational workarounds.

### NCU binary discovery — `_discover_ncu_binary()` fallback for venvs that don't add cuda to PATH (2026-05-08)

**Rationale.** `_discover_ncu_binary()` (`src/eval/profiler.py:814`) currently relies on `shutil.which("ncu")` only. The 2026-05-08 venv relocation rebuilt `~/.venvs/acts_run_venv` from `configs/venvs/3.12.md`'s canonical recipe, which does not include a step to prepend `/usr/local/cuda-12.8/bin` to the venv's activate-script PATH. The OLD `/tmp/acts_gpu_venv` had this patched manually (per the now-superseded `reference_test_venv.md` auto-memory). When the new venv was built clean from the recipe, the PATH adjustment was lost; the live optimize run on 2026-05-08 then failed with `ncu_binary_not_found` × 6 (one per profile attempt). The GPU pytest sweep happened to pass because pytest's collection / fixture wiring inherits a different PATH than the orchestrator's subprocess invocations, but that's an implementation accident; depending on it is brittle.

Either way, the operator-side activate-script patching is brittle as a primary fix: any clean rebuild from the canonical recipe loses NCU discoverability silently. The load-bearing fix needs to be code-side so the system survives ad-hoc venv rebuilds.

**What changed.** Two-pronged:

- **Code**: `_discover_ncu_binary()` falls back to `/usr/local/cuda-12.8/bin/ncu` when `shutil.which("ncu")` returns None. Hardcoded path matches the host driver / cuda toolkit version per `configs/venvs/3.12.md`'s host invariant ("Host: Ubuntu 20.04, NVIDIA RTX 6000 Ada, driver 570.172.08, CUDA 12.8").
- **Recipe**: `configs/venvs/3.12.md` gets a "Step 7: prepend `/usr/local/cuda-12.8/bin` to activate" subsection so the operator surface is also clean — venvs built from the recipe now have ncu on PATH from activation alone, no fallback exercised.

**Settled design choices** (2026-05-08 inline question-list discussion):

- **Hardcoded fallback path, not version-glob**. The host has 4 cuda installs (11.0, 11.8, 12.4, 12.8); a glob would need version-sort logic and could pick a cuda the current driver doesn't support. Hardcoding to 12.8 is tighter coupling to the host config but more correct. If the host cuda gets bumped, the recipe needs updating anyway, so adding the path string to the same hardcoded set is a one-edit lockstep.
- **Both code-fallback AND activate-script patch**. Code is load-bearing (survives ad-hoc rebuilds, the failure mode today's session already hit twice); recipe is operator-facing (clean signal about what setup needs).
- **No env-var override** (e.g., `ACTS_NCU_BINARY=/path`). Considered; rejected — adds API surface without solving a real ergonomic; if the operator wants a non-default ncu, prepending PATH or symlinking `ncu` is the conventional Linux move and works without code support.

**Trigger for revisit.**

- Host cuda version bumps from 12.8. Both `_discover_ncu_binary()`'s fallback and the recipe's activate-step need updating in lockstep.
- ACTS deploys to a host without `/usr/local/cuda-*/bin/ncu` (e.g., Conda-managed CUDA, a containerized runtime, a Mac dev box). At that point the hardcoded fallback becomes wrong; switch to the version-glob path or env-var-driven discovery.
- Persistent ncu version mismatches between what activate-PATH resolves and what the fallback resolves (e.g., user's PATH points at cuda-12.4 but fallback finds cuda-12.8). Surface a startup warning if the two disagree.

**Amendment 2026-05-08 — thread the discovered binary into `_run_ncu`'s subprocess invocation.** The initial fix added the fallback to `_discover_ncu_binary()` but `_build_ncu_argv` still hardcoded argv[0] to the bare string `"ncu"` — so when PATH lacked ncu, `subprocess.run` would still raise `FileNotFoundError` and degrade as `ncu_binary_not_found`, even though the fallback discovery had succeeded. The fallback was effectively dead code in the exact clean-venv scenario it was designed to fix; today's live run only worked because the activate-script PATH patch made `shutil.which("ncu")` succeed, so the fallback path was never exercised. The unit test that "verified" the fallback only checked `_discover_ncu_binary()`'s return value, not whether `_run_ncu` actually launched the right binary.

`_run_ncu` now substitutes `_discover_ncu_binary()` for argv[0] before `subprocess.run`. Mirrors the pattern `_extract_ncu_csv` already uses for the `ncu --import` invocation (which is why that one worked). New regression test: PATH-clean (`shutil.which("ncu")` returns None) + fallback-present scenario, asserts `_run_ncu` succeeds end-to-end (not just `_discover_ncu_binary` returning a path). Codex adversarial review caught it.

### Clock-lock verify — query `clocks.applications.*` instead of `clocks.current.*` (2026-05-08)

**Rationale.** `_verify_gpu0_locked` (`src/pipeline/optimize.py:280-322`) was reading `clocks.current.{graphics,memory}` — the GPU's *live* clock frequency at the instant of query. On RTX 6000 Ada the GPU drops back to idle (graphics ~210 MHz, DRAM ~810 MHz) within microseconds of any kernel completing; the wake-op (32-element `torch.zeros + 1.0`) does fire the clocks up to lock target, but the next `sudo nvidia-smi` subprocess takes tens of ms to cold-start and loses the race against idle drop. Live optimize run on 2026-05-08 hit `WARNING GPU 0 graphics-clock mismatch — expected 2505 MHz, got 210 MHz` and rolled back the partial lock with `verify_failed`. The lock itself was in force throughout — `clocks.applications.*` consistently reported 2505/10001 (the lock target). ACTS was reading the wrong field for its lock-correctness predicate.

The 2026-05-07 first diagnosis attributed `verify_failed` to `nvidia-persistenced.service` being disabled. That hypothesis was wrong; persistence-mode is now enabled on all 4 GPUs (verified by `nvidia-smi --query-gpu=persistence_mode --format=csv` and `systemctl is-active nvidia-persistenced`). Persistence keeps the *applications* clock pinned (which we read after this fix), so it's load-bearing for the lock target itself, but it doesn't keep the GPU busy — `current.*` would still show idle drift even with persistence on. The persistence fix and this query-field fix are both required, in series.

**What changed.**

- `_verify_gpu0_locked`'s query-csv argument switches from `--query-gpu=clocks.current.graphics,clocks.current.memory` to `--query-gpu=clocks.applications.graphics,clocks.applications.memory`.
- The 32-element wake-op `torch.zeros(32, device='cuda:0') + 1.0` becomes dead code — `applications.*` reflects the lock target whether the GPU is busy or idle. Wake-op removed; `_verify_gpu0_locked` is purely a query-and-compare now.
- Tolerance unchanged (50 MHz). Field name unchanged in the warning message ("graphics-clock mismatch", "memory-clock mismatch") because the user-facing semantics are still "this clock didn't lock to the target."
- GPU-0-only scoping unchanged. All `nvidia-smi` calls in `src/pipeline/optimize.py` already route through `_nvidia_smi()` which always appends `-i 0`; audit confirmed no leaks. The fix is field-only, not scope-related.

**Settled design choices** (2026-05-08 inline question-list discussion):

- **`clocks.applications.*` over `clocks.current.*`**. The `applications` field is exactly what `nvidia-smi -lgc <gpu>,<gpu>` and `-lmc <dram>,<dram>` write — it's the lock target nvidia-smi pins, and persists at that value as long as the lock holds, whether the GPU is busy or idle. `current` is the live frequency, which depends on idle/busy state and is therefore load-state-coupled. The lock-correctness predicate we actually want is "did the lock subprocess pin the clock target?", not "is the GPU running at lock target *right this microsecond*?". The field name documents the answer.
- **Drop the wake-op entirely**, don't keep it as a defense-in-depth. It was load-bearing for the wrong reason (forcing `current.*` up); with `applications.*` it does nothing useful and adds startup latency, an exception surface (CUDA init can fail in degraded states), and a sequencing constraint (kernel must launch *before* verify reads). Dead code; remove.
- **Don't add a redundancy check** ("warn if `applications` and `current` disagree by more than X"). Considered briefly; rejected — `current` differs from `applications` *by design* whenever the GPU is idle, so the check would either fire constantly (false positives) or need a deferred-check window after compute that re-introduces the wake-op race we just eliminated.
- **Tolerance stays 50 MHz**. `applications.*` is set exactly by the lock subprocess; observed drift from target should be 0 in healthy state. The 50 MHz tolerance handles any future driver-side rounding (none observed today on RTX 6000 Ada). No change needed.

**Why not Option 2 (retry-on-current).** Keeps the strict "live frequency" check via 3× retry with sleep + repeated wake-op. Adds 100–500ms to startup; still races on heavily loaded hosts; the strict check it preserves isn't actually the lock-correctness predicate (idle drift is the GPU working as designed, not a lock failure). Option 1 fixes the regression with a smaller diff and a more correct semantic.

**Why not Option 3 (settle-delay).** Sleep before verify. Doesn't address the race — once any wake kernel completes, the GPU drops idle regardless of how long we slept beforehand. Insufficient alone; would need to combine with the busy-loop wake from Option 2 to actually close the race.

**Trigger for revisit.**

- Hardware-level lock failures (thermal override, driver bug pinning `applications.*` to target while the silicon ignores it). Today's check would miss these. If real-world runs show suspiciously bad-and-stable latencies, layer Option 2 on top: verify `applications.*` matches lock target *and* `current.*` is within ±200 MHz of target during a measured profiling kernel (which is a very different observation point than the post-lock instant).
- Multi-process GPU contention. With `clocks.applications.*` we read what *we* pinned; if another process pins different applications clocks, we'd read theirs not ours. ACTS is single-tenant on this host today; revisit if multi-tenant.

### Summary-only, contrastive injection

**Rationale**: AccelOpt's ablation shows memory improves **cost-efficiency** (16% fewer iterations) but not peak quality. Memory is an accelerant, not a capability unlock.

### Summary-only, not code snippets

Planner doesn't need 200 lines of old kernel code. Summaries are cheaper (fewer tokens), more generalizable (not tied to specific shapes), and capture the causal insight that matters. AccelOpt stores full slow-fast pairs but the LLM mostly uses the optimization summary, not the code.

### Both successes and failures stored

Following AccelOpt. Failures prevent repeating mistakes ("split-K on this matmul shape caused 2x regression because K dimension was too small").

### Contrastive format over absolute summaries

Simply stating "tiling gave 1.35x on a matmul" tells the Planner WHAT worked. The contrastive format tells WHY it worked (uncoalesced → coalesced) and HOW the current kernel matches the "before" case. Stronger signal for technique selection.

### JSON backend

Simple, git-friendly, human-readable. No embedding infrastructure needed. Sufficient for kernel-type filtering + bottleneck matching retrieval.

### Injection into Planner only

Not into Coder (has the structured plan), not into Reviewer (evaluates current results independently). Planner is where strategy decisions happen.

### Relationship to search tree

Search tree = intra-task working memory (full state per node, orchestrator uses for navigation). Optimization memory = inter-task long-term memory (distilled summaries, Planner uses for strategy). At task end, orchestrator distills tree's most informative paths into memory entries.

### Tree context for Planner

Planner doesn't read tree directly. Orchestrator extracts brief tree context (what actions tried at this state + outcomes). Prevents redundant exploration without exposing full tree. Combined with optimization memory, Planner sees: (1) what's been tried on THIS kernel, (2) what worked on SIMILAR past kernels, (3) what CAN be done, (4) what's happening NOW.

### Scored retrieval with reserved failure slots (2026-04-16)

**Rationale**: The skeleton retriever partitioned experiences by bottleneck match (exact first, then rest) but had no ranking within each partition and no guarantee that failures would surface. Three problems:

1. **No success/failure differentiation**: The Planner needs both — successes to know what works, failures to know what to avoid. Pure score ranking would push failures to the bottom since they have low speedup (< 1.0), potentially excluding them entirely at small top_k.

2. **No hardware awareness**: Experiences from different GPUs may be less relevant (e.g., an H100 tiling strategy may not transfer to A100). Same-hardware experiences should be preferred, with cross-hardware fallback when the same-hardware pool is too small.

3. **No secondary ranking**: Among experiences with the same bottleneck match status, there was no ordering — insertion order determined results.

**Scoring design**: Bottleneck match (+10) dominates, ensuring relevant experiences rank first. Success bonus (+3) separates successes from failures within the same bottleneck tier. Speedup (+min(speedup, 5.0), capped to prevent one outlier from dominating) provides fine-grained ordering. Tiebreaker is speedup.

**Reserved failure slots**: `max(1, top_k // 3)` slots reserved for failures (at top_k >= 3). This ensures the Planner always sees "don't do this" examples alongside "do this" examples. For top_k < 3, no reservation — the single or two slots are too scarce to split, so pure score ranking applies (successes naturally outscore failures due to the +3 bonus).

**Hardware filter is optional**: The retriever accepts `hardware=""` (default), which skips hardware filtering. The orchestrator is still a skeleton and doesn't pass hardware — this will be wired when the orchestrator gets its real implementation.

### Future: Reviewer Knowledge Base

Three-tier structure: Compute-Reviewer KB, Memory-Reviewer KB, Shared Interaction KB.

**Static vs evolved knowledge**: Static reference organized around diagnostic reasoning chain — not just "what is SM occupancy" but "low occupancy + high register usage + good throughput-per-SM = register-efficient but parallelism-starved → occupancy-limited compute-bound." Evolved knowledge accumulates from real runs.

**Two-dimensional retrieval**: Metric-triggered ("current profiling shows pattern X → retrieve entries about X") + Action-triggered ("action Y was just applied → retrieve entries about known side-effects of Y").

**Static KB construction**: LLM-assisted extraction from textbooks + human review. Each chapter yields one entry per diagnostic pattern (not per-chapter). Entry format: source, trigger, pattern, diagnosis, reasoning_chain, recommended_actions, anti_patterns.

### Future: full knowledge architecture

```
Search Tree (V1)          — intra-task, ephemeral → Orchestrator
Optimization Memory (V1)  — inter-task, persistent → Planner
Reviewer KB (Future)       — inter-task, persistent → Reviewer
Post-task Distillation     — tree → memory entries + KB entries
```

**Update timing**: During a task, experiences live only in search tree. Optimization memory entries come from previous tasks only. Distillation happens once at task end.

**Relationship between stores**: Optimization memory tells Planner *what to do*; Reviewer KB tells Reviewer *what's happening*. Mutually reinforcing — better diagnosis leads to more accurate memory, which leads to better decisions, which produce clearer signals.

### Persist `raw_metrics` across the NCU cache (2026-05-11)

`_save_ncu_cache` already wrote the full raw CSV-parsed metric dict to disk alongside the curated `NCUMetrics`; the loader threw the raw half away on rehydration, hard-coding `raw_metrics={}` on cache hits.

**Why surface it now**: the Reviewer's `query_metric` tool reads from `ProfilingResult.raw_metrics`. Today the Reviewer only runs in the iteration that produced the profile, so the cache-hit gap wasn't noticed. But three concrete future scenarios all need persisted raw:
- **Re-Review cached profiles** — A/B-testing Reviewer prompts against historical runs.
- **Bottleneck-shift / per-workload-speedup investigations** (existing trigger-gated entries in PROCESS.md) — cross-iteration metric trajectories where the smoking gun lives outside the curated set.
- **Metric-set-version drift** — adding a new curated field shouldn't require re-profiling every cached kernel; the data was already on disk.

**Why now and not later**: the change was a five-line loader fix plus one call-site update. Cost in disk is ~5 KB per profile (~0.5 MB per 30-iter run), trivial next to the ~80 KB `.ncu-rep` we already accept. No new on-disk format — `_save_ncu_cache` was already persisting raw, the loader just wasn't reading it back. Pre-2026-05-11 cache entries without a `raw` field load with `raw_metrics={}` for back-compat — same shape as a degraded re-profile would produce.

### a+b decoupling — SOLAR counts surfaced, analytical roofline tolerates missing inputs (2026-05-13)

**Motivating evidence**: `runs/run_20260513T030257_891519Z` ran 3 iterations against an L1 SOL-ExecBench problem. Every L1 `Definition` carries `op_type=None`, so `compute_roofline_inputs` (the shape-formula path) bailed to `(0, 0)`, and the orchestrator's `if iter_flops > 0 and iter_nbytes > 0` gate skipped profiling entirely with a `"skipping profile — no (flops, nbytes) for op_type=…"` warning. The Reviewer never fired, the Planner saw no diagnostic feedback, and the search picked `t1_block_size_tuning` three times in a row, all regressed, all stamped `branch_quality=promising` because the Reviewer was disabled. The final report showed `Bottleneck (run): compute_bound` with empty per-workload data.

**Decision — two-layer decoupling, "a + b"**:

1. **(a) Surface SOLAR's own `total_flops` / `total_fused_bytes`** through `SolarResult` → `RooflineResult`, and let `compute_roofline_inputs(definition, workload, *, roofline=None)` prefer them over shape formulas when both are positive. SOLAR already counts MACs and bytes during its einsum analysis (`perf["workload"]["total_flops"]`, `perf[roofline_model]["memory_bytes"]`); the adapter just wasn't carrying them across the boundary. Every L1 problem SOLAR can analyze now feeds analytical with real counts even when `op_type=None`.
2. **(b) Make analytical optional**. `ProfilingResult.analytical: AnalyticalMetrics | None` with a new `has_analytical` property; `profile_kernel(nbytes=0)` skips `_compute_analytical` and sets `analytical=None`; NCU still runs. Both orchestrator gates (`iter_flops > 0 and iter_nbytes > 0`) were dropped — the Reviewer is no longer gated on analytical availability. Per-iter call sites pass `roofline=roofline` so (a) takes effect; Phase C re-profile (`report.py::_resolve_workload_roofline`) reordered to derive SOLAR first, then call `compute_roofline_inputs` with `roofline=solar`.

**Why both, not just (a)**: SOLAR can also fail (bridge can't synthesize a `Model`, unresolved axes, optional dep absent). With (a) alone, those cases still kill the iteration. (b) is the catch-all — even when SOLAR fails *and* the shape formula bails, NCU survives and the Reviewer gets occupancy / stall / L2-hit signals. Analytical degrades to "[unavailable — no byte count]" in rendered prose; bottleneck classification stays SOLAR-sourced via `classify_run` (which doesn't read analytical anyway, per the 2026-04-22 entry).

**Renderer / serializer fan-out (Codex-review-driven fixes)**: making `analytical` Optional forced three call sites to grow guards. `reviewer.py::_render_profiling_for_reviewer` and `orchestrator.py::_render_profiling_for_planner` were reading `a.pct_peak_compute` unconditionally (would `AttributeError` on `None`); `tree_dump.py::_build_meta` and `tree.py::_serialize_profiling` / `_deserialize_profiling` called `asdict(node.profiling.analytical)` and rebuilt unconditionally (would `TypeError` on `None`). All four guard on `has_analytical` now. Checkpoints written before today deserialize with `analytical={...}` as before; post-fix checkpoints with `analytical=None` round-trip cleanly.

**Why not "just author shape formulas for L1 op_types"**: the L1 problem family has hundreds of distinct kernels — flux_rope, gelu_tanh, swiglu_quant, attention variants. Hand-coding a flop formula per kernel-type is exactly the work SOLAR exists to automate. (a) reuses that work; the shape-formula table stays as the SOLAR-less fallback for placeholder / custom-problem paths.

**Tradeoff carried forward**: SOLAR-derived counts inherit SOLAR's accuracy contract (exact for MAC-dominated ops, 0 for pure-elementwise — the `0.0` AI case the `SolarResult` docstring documents). When SOLAR returns `total_flops > 0, total_fused_bytes > 0`, the analytical profiler's `%peak_compute` / `%peak_bandwidth` are as physics-accurate as the SOL score itself. When SOLAR returns 0 for either, the shape formula path runs; when both fall through, analytical drops out entirely and the Reviewer leans on NCU.

**Partially resolves PROCESS Backlog item "SOLAR-vs-shape-formula reconciliation (2026-05-10)"**: the reconciliation that backlog item asked for — SOLAR counts winning over shape formulas where both are available — landed via (a). Per-op overrides (zero-copy views, mask-typed bools) remain deferred; that's where SOLAR's counts still beat shape formulas by *more* than the shape formulas can recover, and where Reviewer prose drifts on view-heavy kernels.

**/simplify pass folded in**: added `has_analytical` property mirroring the existing `has_ncu`; cut narrative comments down to load-bearing rationale; dropped defensive `getattr` lookups now that the field is explicitly typed `Optional`.

### Tensor-core NCU metric promoted to OPTIONAL (2026-05-14)

**Regression introduced earlier, surfaced now**: an earlier PR added the tensor-core counter (`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`) to `_CURATED_REQUIRED` alongside occupancy and L2 hit rate. The `_parse_ncu_csv` contract is "if any required metric is absent, the whole NCU block degrades to a parse failure," so on any kernel/GPU combination that doesn't emit the tensor-core counter, profiling silently lost occupancy + L2 + stall data along with it. Ada elementwise kernels are one such combination — NCU emits 0 rows for the tensor pipe section because nothing in the kernel touches the tensor pipeline. The "degrade gracefully" contract from the 2026-04-20 profiler design entry (analytical is the floor, NCU is the bonus) was inverted: one missing diagnostic was killing the rest.

**Fix**: split `_CURATED_REQUIRED` (occupancy + L2, the two metrics every profile-able Triton kernel emits on this hardware) from a new `_CURATED_OPTIONAL` dict containing the tensor-core counter. `NCUMetrics.tensor_core_util_pct: float | None` is the one optional field on the dataclass; everything else stays non-optional. `_parse_ncu_csv` uses `raw.get(name)` for optional metrics — absence yields `None` rather than triggering the parse-failure path. Only REQUIRED metric absence still degrades the full block. Renderers (`reviewer.py::_render_profiling_for_reviewer`, `pipeline/report.py`) handle the new `None` case by rendering `"n/a"` so the Reviewer prose stays grounded and the report stays readable.

**Why we got here**: the original 2026-04-20 curated-set design listed four NCU signals (occupancy, stall reasons, L2 hit rate, tensor-core utilization) and treated them as a single tier of "required for the curated section." The Ada-elementwise case proves the four aren't symmetric — three are universal, one is workload-conditional. The class split is the right resolution rather than relaxing the parse-failure contract globally, because occupancy + L2 + stalls really do need to all-or-nothing together (their meanings interlock; missing one means the parse went wrong).

---

## Benchmark & Scoring

### SOL-ExecBench as benchmark suite (over KernelBench) (2026-04-14)

**Rationale**: KernelBench (Ouyang et al., 2025) measures speedup over PyTorch eager — a mutable software baseline that tells nothing about proximity to hardware limits. A 10x speedup over PyTorch can still be 100x away from hardware SOL. SOL-ExecBench (NVIDIA, 2026) reframes evaluation around closing the gap to hardware Speed-of-Light, providing 235 problems from 124 production AI models across BF16/FP8/NVFP4 precisions with forward and backward passes.

### HardwareSpec uses SOLAR arch YAML schema directly (2026-04-15)

**Rationale**: SOLAR arch config YAMLs (e.g., `H100_PCIe.yaml`, `B200.yaml`) define hardware in roofline-oriented terms: per-cycle throughput by precision (MAC/cycle for FP32, BF16, FP8, NVFP4, etc.), SRAM/DRAM capacities and bandwidth, and clock frequency. Rather than maintaining a separate `HardwareSpec` schema and translating between the two, `HardwareSpec` uses SOLAR's schema directly. This means:

- `load_hardware_spec(path)` reads a SOLAR YAML into a `HardwareSpec`
- SOLAR's Python API and ACTS's built-in roofline both consume the same data
- Derived properties (`peak_flops_fp32`, `peak_memory_bandwidth_gb_s`) are computed from the raw per-cycle fields + frequency, matching the formulas in SOLAR's comments (e.g., `MAC_per_cycle_bf16_tc * freq_GHz * 2` = PFLOPS) _(superseded 2026-05-02: this formula reading was the source of an off-by-1000 unit bug; see `peak_flops_*` properties in `src/config.py` — actual divisor is `1e3` for TFLOPS, not the implicit `1e6` that PFLOPS framing suggested)_

The alternative — a GPU-metadata-oriented schema (SM count, compute capability, peak TFLOPS) — would require translating to/from SOLAR's schema at the boundary, and the "peak TFLOPS" values would need to know which precision to report for. SOLAR's schema is more precise: it distinguishes FP32 SM cores from BF16 Tensor Cores from FP8 Tensor Cores.

### SOLAR for T_SOL derivation (over hand-derived roofline) (2026-04-14)

**Rationale**: Hand-derived roofline (classical `max(FLOPs/throughput, bytes/bandwidth)`) is fragile — it requires manually counting FLOPs and memory traffic per kernel, and naive roofline overestimates achievable performance for kernels with complex data reuse. SOLAR automates this: it traces the PyTorch reference, converts to einsum notation, and derives hardware-grounded bounds that account for cache hierarchy and fusion opportunities.

SOLAR produces three roofline models: unfused, fused, fused_prefetched. We use **fused** (intermediate tensors excluded, per-op roofline) as T_SOL. The fused_prefetched model assumes perfect overlap which is often unreachable in Triton — using it would make SOL scores pessimistic and cause plateau detection to trigger too early.

### Triton baseline via LLM translation (2026-04-14)

**Rationale**: SOL-ExecBench provides only PyTorch references. ACTS optimizes Triton code, so each problem needs a Triton starting point. The Coder agent generates a PyTorch-to-Triton translation at problem load time. This is a well-scoped task: the PyTorch reference defines exact semantics, shapes, and dtypes — the LLM just writes a functionally equivalent Triton kernel.

Correctness is verified against the PyTorch reference before accepting the baseline. The Coder gets up to `max_baseline_retries` attempts since some L2 multi-op fused subgraphs are non-trivial to translate. If all attempts fail, the problem is skipped.

### PyTorch as correctness reference, Triton as scoring baseline (2026-04-14)

**Rationale**: Two distinct roles that must not be conflated:

- **Correctness reference** = PyTorch. Always. The PyTorch `run()` function is the ground-truth specification, validated by the SOL-ExecBench team with human review and execution-based checking. If the Triton baseline had subtle bugs, using it as correctness reference would propagate those bugs as "correct" throughout optimization.
- **Scoring baseline (T_b)** = Triton baseline latency. T_b defines S=0.5 in the SOL score — the "no improvement" midpoint. Since ACTS optimizes Triton code, the meaningful zero-progress point is the Triton starting point. If the Triton baseline is slower than PyTorch, using PyTorch as T_b would make early iterations look like regressions when they're actually just catching up. The SOL-ExecBench code explicitly allows T_b to be any fast implementation.

### T_b measured once, not recomputed (2026-04-14)

**Rationale**: T_b is a fixed anchor for scoring. Recomputing it each iteration introduces noise to the metric itself, making it hard to distinguish real improvements from measurement jitter. More critically, a non-stationary T_b breaks plateau detection — consecutive SOL score readings become incomparable.

T_b is measured once at startup with generous repetitions (warmup + 100 timed runs), using the mean (consistent with SOL-ExecBench's `do_bench` default). GPU clocks are locked during the entire ACTS run for reproducibility. A periodic "reference health check" (re-measure Triton baseline every N iterations) can flag hardware drift (>5% = abort), but does not update T_b for scoring.

### Workload selection for iterative benchmarking (2026-04-14)

**Rationale**: SOL-ExecBench problems have 7-48 workloads each (different batch sizes, sequence lengths). Running all workloads every iteration is expensive. During the search loop, ACTS benchmarks on 2-3 representative workloads. The full workload suite runs only at final evaluation (Phase C).

### SOL score invariant violations as audit signals (2026-04-15)

**Source**: SOL-ExecBench paper, Section 4.3: *"We assume T_b > T_SOL and T_k ≥ T_SOL... If either assumption is violated in practice, we treat the case as an audit signal and report it for SOLAR bound review and reward-hacking inspection."*

Two violation cases:

- **T_k < T_SOL** (candidate beats speed-of-light): Almost certainly reward hacking — the kernel is exploiting a measurement loophole (concurrency exploits, state caching, environment manipulation per paper Table 3 / Section 4.4.1). `ScoreResult.reward_hack_suspect = True`. The raw SOL score > 1.0 is intentionally not clamped — the anomalous value is itself the signal. Downstream consumers (orchestrator, anti_cheat) should inspect before accepting the node.

- **T_b ≤ T_SOL** (baseline already at or below hardware limit): Either SOLAR's bound is too loose for this problem, or the baseline is exceptionally well-optimized. `ScoreResult.calibration_warning = True`. Score is set to 1.0 (problem already solved). Not necessarily reward hacking — could be legitimate calibration issue.

**Why not clamp to [0, 1]**: Clamping hides the anomaly. The paper treats these as audit signals, not edge cases to suppress. Keeping the raw value lets the anti-cheat module make an informed decision. This also connects `scorer.py` (orchestrator-side eval) to `anti_cheat.py` (currently Coder-side only) — creating a second anti-cheat surface at the performance level, not just the correctness level.

### SOL-ExecBench integration — tiered adoption (2026-04-18)

**Context**: SOL-ExecBench (NVIDIA) is the declared benchmark for V1. Its `core/bench` package carries reusable machinery — error-stats computation, input generation, tolerance spec, reward-hack detection, subprocess-isolated eval driver. The framework must eventually support other benchmarks too (KernelBench, etc.), so integration depth is a design choice, not a one-shot.

**Decision**: tiered adoption, scoped by marginal value at each feature.

- **Tier 1 (landed this increment)**:
  - `TorchComparisonPolicy.compare` delegates to `sol_execbench.compute_error_stats` when importable. Gives us matched-ratio tolerance, separate NaN/Inf flags, and a hard max-error cap for free. Falls back to `torch.allclose` when SOL isn't installed — keeps the module usable for non-SOL benchmarks.
  - `eval/inputs.build_input_generator` wraps `sol_execbench.core.bench.io.gen_inputs` so real problems get heuristic-aware inputs (probability softmaxing, shape/dtype dispatch) without re-implementing them.
  - `eval/inputs.build_reference_fn` is pure-Python (exec source into namespace) — torch only loads when the reference actually runs, so the module imports cleanly in torch-less test venvs.

- **Tier 2 (deferred, recorded in PROCESS.md)**: adopting SOL's `Definition` / `Workload` / `Trace` pydantic models end-to-end. Today ACTS parses them into hand-written dataclasses and `eval/inputs.py` round-trips back to dict for SOL's consumers. Refactor trigger: when `benchmark/baseline_generator.py` starts passing definitions through the full pipeline and the duplicated schema has somewhere to spread.

- **Tier 3 (skipped for now, recorded in PROCESS.md)**: subprocess-isolated eval driver (`ProblemPackager` + `eval_driver.py`) and reward-hack detection (`core/bench/reward_hack.py`). Both target threat models our internal bounded search doesn't have — the search runs code ACTS generated itself, on a controlled env. Revisit only if we hit real crashing kernels or accept external code.

  **Correctness reframing (2026-04-18, Codex adversarial review)**: Skipping subprocess isolation is not only a safety trade-off — it's also a correctness trade-off. A candidate whose module-scope code mutates shared modules (e.g. rebinds `torch.matmul`) can silently alter subsequent `reference_fn` calls inside the same process, so later stages of `verify_correctness` compare wrong-against-wrong and return `passed=True`. Codex demonstrated this with a toy candidate. For our threat model (our own LLM, bounded search) the probability is low but the failure is silent. Acceptable for now, but the Tier 3a trigger now includes "silent oracle corruption observed," not only "crashes."

**Why tiered, not all-in**: SOL-ExecBench requires Python ≥3.12, torch ≥2.10, cuTile, CUTLASS DSL. Hard-adopting its models and subprocess harness would force those versions everywhere and couple ACTS to SOL's upgrade cadence. Keeping SOL at the edges (tolerance + input gen) lets ACTS stay benchmark-agnostic for future KernelBench support while getting the high-value pieces today.

**Why not KernelBench yet**: PRD already documents the SOL-over-KernelBench decision (below). Multi-benchmark support stays a V2+ concern — the `ComparisonPolicy` Protocol + callable-based `input_generator` already give us the seams, so the cost of adding KernelBench later is low.

### SOL integration tightening — CUDA 12.8 plan (2026-04-22)

**Refines the 2026-04-18 "tiered adoption" entry above.** That entry gated deeper
integration on "hard install coupling to cu13." Investigation 2026-04-22 showed
that coupling is packaging-only; the triggers on PROCESS.md's Deferred
entries for "Adopt SOL pydantic end-to-end" and "Adopt `do_bench` protocol"
are now more attractive to fire than when written.

**Context**: Dev host is Ubuntu 20.04 / CUDA 12.8 / driver 570 / Ada SM_89.
CUDA 13 is not installable bare-metal on 20.04 (glibc 2.31 floor blocks CUDA
13's 2.34+ requirement). User constraint: tighten ACTS↔SOL integration so
SOL-owned functionality is used directly rather than re-implemented, while
(A) avoiding any cu13-dependent surface and (B) preserving support for
non-SOL benchmarks (KernelBench, custom).

**Finding**: `src/sol_execbench/**` framework code has **zero runtime imports**
of `cutlass`, `cuda_tile`, `cudnn_frontend`, `apache_tvm_ffi`, or
`torch_c_dlpack_ext`. The cu13 coupling lives entirely in:

1. `pyproject.toml` install manifest — pip-resolved but never loaded.
2. `driver/templates/build_ext.py:44` — `CUTLASS_DIR` for user-solution C++
   compiles. Only activates when a submitted kernel's language is CUTLASS.
3. `core/data/solution.py:41-48` — language enum *strings*
   (`"cutlass"` / `"cute_dsl"` / `"cutile"` / `"cudnn_frontend"`). Labels,
   not imports.
4. `tests/docker/dependencies/` — Docker-image smoke tests; not library code.
5. `examples/cute_dsl/` — sample user solutions; data, not framework.

None of these are reachable from ACTS's use path (ACTS generates Triton,
consumes SOL as a library, never invokes `sol-execbench` CLI).

**Second finding (benchmark-agnostic posture)**: `Definition` is a
general-purpose kernel IR — named tensor inputs/outputs, symbolic axes,
a pure-Python `def run(...)` reference. Nothing in the schema references
SOL-ExecBench categories, leaderboard, HuggingFace dataset, or scoring
protocol. The 2026-04-18 entry's hedge ("Problem abstraction may need
to stay benchmark-agnostic") was over-cautious: `Definition` **is** the
benchmark-agnostic type. KernelBench plugs in via a converter, not via
a parallel Problem abstraction.

**Install strategy on cu12.8** (unblocks everything below):

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-venv
python3.12 -m venv /tmp/acts_run_venv && source /tmp/acts_run_venv/bin/activate
pip install "torch>=2.10" "torchvision>=0.24" \
  --index-url https://download.pytorch.org/whl/cu128
pip install -e /home/hel19/workspace/projects/self-evolved-llm/repo/benchmark/SOL-ExecBench --no-deps
pip install pydantic safetensors numpy click rich pyyaml \
  pytest pytest-asyncio triton "openai-agents>=0.1"
```

Smoke test:
`python -c "from sol_execbench.core.data import Definition, Workload; from sol_execbench.core.bench.io import gen_inputs; from sol_execbench.core.bench.timing import do_bench; print('ok')"`.
If this prints `ok`, cu13 packages were never loaded.

**Decision — five-tier integration plan**:

- **Tier 1 — Schema adoption** (biggest LOC reduction). Replace ACTS's
  hand-written `src/benchmark/problem.py` dataclasses, `problem_loader.py`,
  and `solution_formatter.py` with direct use of
  `sol_execbench.core.data.{Definition, Workload, Solution, Trace}`. Drop
  `_problem_to_sol_dict` / `_workload_to_sol_dict` shims in `eval/inputs.py`.
  Net: ~-240 LOC + ~+60 thin wrappers = ~-180 LOC. Trigger for
  PROCESS.md "Adopt SOL pydantic end-to-end" already fired
  (`baseline_generator.py` landed).

- **Tier 2 — Timing adoption**. Replace `_TorchCudaTimer` in
  `eval/benchmark.py` with `sol_execbench.core.bench.timing.time_runnable`
  / `do_bench`. Redesign test seam: swap the per-iter `BenchmarkTimer`
  Protocol for a `BenchmarkFn = Callable[[fn, setup], float]` type alias;
  tests inject a mock callable instead of asserting call-order on
  `prepare/flush/record/finalize`. Net: ~-60 LOC production + ~-120 LOC
  tests. Trigger for PROCESS.md "Adopt `do_bench` protocol" fires before
  first live multi-workload run. Do this as its own phase — design
  discussion required on the replacement test seam.

- **Tier 3 — `sol_score` delegation**. 5-line wrapper in
  `eval/scorer.py::compute_sol_score` calling
  `sol_execbench.sol_score.sol_score(t_k, t_p, t_sol)`, then adding
  `reward_hack_suspect` / `calibration_warning` flags and packing into
  `ScoreResult`. Keeps the formula canonical.

- **Tier 4 — Optional reward-hack + clock-lock**. Wire
  `sol_execbench.core.bench.reward_hack.{check_monkey_patch,
  check_thread_injection, check_result_caching}` into
  `eval/anti_cheat.py` when threat model justifies. Add
  `sol_execbench.core.bench.clock_lock.are_clocks_locked` as a startup
  warning in `pipeline/optimize.py`. Both stay deferred per existing
  triggers.

- **Tier 5 — Benchmark adapter pattern** (preserves Requirement B).
  Move SOL-specific loading into `src/benchmarks/sol_execbench/load.py`
  returning `tuple[Definition, list[Workload]]`. Downstream pipeline
  consumes those SOL types directly — benchmark-agnostic. Scaffold
  empty `src/benchmarks/kernelbench/` and `src/benchmarks/custom/`
  dirs so the contract is visible. KernelBench converter is a future
  phase when the need lands.

**Execution order**: Env setup → Tier 1 (schemas) as one phase → Tier 3 +
Tier 5 scaffold as one phase (low-risk, shakes out import paths) →
Tier 2 (timing) as its own phase with the test-seam design discussion.
Tier 4 stays deferred.

**What cu12.8 blocks** (bounded blast radius, all off ACTS's path):

| Blocked surface | Why | Impact on ACTS |
|---|---|---|
| User-solution compile for CUTLASS / cuTe DSL / cuTile / cuDNN-graph | `build_ext.py` expects cutlass-dsl[cu13] headers | None — ACTS generates Triton. Would matter only if Coder were extended to emit CUTLASS. |
| Full `sol-execbench` CLI in-process | CLI routes through build_ext → cu13 | None — ACTS's orchestrator owns the eval loop; we're integrating primitives. |
| `cuda-tile==1.1.0` runtime (Blackwell tile abstractions) | cu13 + SM_100 only | None — Ada is SM_89, blocked by hardware not toolkit. |
| `nvidia-cutlass-dsl[cu13]` runtime | No cu12 channel | None today. Matters only if ACTS ever imports cuTe DSL as a search target. |
| `nvidia-cudnn-frontend==1.18.0` (cuDNN 9.x graph API) | Targets cuDNN 9 on cu13 | None — ACTS doesn't use cuDNN graph API. Older cu12-compatible frontend exists if needed later. |
| SOL's Docker end-to-end harness | Image is Ubuntu 22.04 + cu13 | Runnable with host driver bump to 580+, but not needed for library-level integration. |
| cu130-only torch features | Any Blackwell-specific torch feature | None for Ada. cu128 wheels have parity for non-Blackwell features. |

**Why not just Docker-everything**: SOL's README recommends Docker for the
full CLI eval. For ACTS's pattern (SOL-as-library) that's overkill, adds a
container boundary around what should be a Python import, and doesn't
solve the host driver bump (580+ still needed even with Docker). Library
integration on 12.8 is the proportionate answer.

**Retrospective (2026-04-28, post-mega-PR)**: Plan landed as designed. cu12.8
stayed clean — no cu13 wheels were ever installed (verified by smoke test
in `configs/venvs/3.12.md`: `torch 2.11.0+cu128 cuda 12.8`). `--no-deps`
plus a hand-curated dep list correctly skipped `cuda-tile`,
`nvidia-cutlass-dsl[cu13]`, and `nvidia-cudnn-frontend` — none of those
primitives are reached from any ACTS code path on Ada. Tier 1 (schemas),
Tier 2 (timing via `do_bench`/`time_runnable`), Tier 3 (`sol_score`
delegation), and Tier 5 (benchmark adapter pattern under `src/benchmarks/`)
all shipped; Tier 4 (reward-hack + clock-lock) wired via `eval/anti_cheat.py`
ahead of the original "deferred" schedule once the live-run threat model
warranted it. The blocked surfaces in the table above remain blocked, as
expected, with zero observed impact on ACTS workflows.

### SOL integration scope expansion — adopt every applicable primitive (2026-04-27)

**Extends** the 2026-04-22 entry above. That plan staged five tiers and
treated reward-hack / clock-lock (Tier 4) as optional, while excluding
several SOL surfaces by silence — output handling helpers, the per-iter
memory pool, safetensors input loading, and the subprocess-isolated eval
driver. User direction this session: tighten to the maximum SOL surface
ACTS can use under the cu12.8 constraint. The "use as library" boundary
holds — we still don't adopt SOL's CLI — but everything below that
boundary is in scope.

**Decision**: integrate every SOL primitive that has a counterpart concept
in ACTS or that unlocks a SOL-problem shape ACTS would otherwise fail to
consume. Promote Tier 4 from optional to required. The architectural
change to subprocess-per-evaluation, previously a separate Deferred entry,
is now in scope.

**Newly in scope beyond the 2026-04-22 plan**:

- **Full reward-hack adoption** (Tier 4 promoted from optional). All five
  detectors: `check_monkey_patch`, `check_thread_injection`,
  `check_lazy_outputs`, `snapshot_critical_functions`,
  `check_eval_integrity`. The last two cover namespace tampering between
  snapshot and check — not in the 2026-04-22 plan.
- **Active clock locking**. Beyond `are_clocks_locked` warning: drive
  clocks via `lock_clocks` / `verify_clocks` / `unlock_clocks` +
  `BenchmarkConfig` + `device_config.get_clock_preset(device_name)`.
  Requires `sudo nvidia-smi --lock-gpu-clocks`. Removes boost-clock
  variance as a real source of timing noise on Ada / H100.
- **Output handling**. `core/bench/io.py::normalize_outputs` (tuple/dict/
  scalar return shapes) + `allocate_outputs` (DPS output buffers).
  Currently the 5-stage gate and benchmark loop assume single-tensor
  outputs; this unlocks SOL problems with multi-output or
  destination-passing-style kernels.
- **Per-iter memory pool**. `ShiftingMemoryPoolAllocator` advances
  `data_ptr` each timed iteration to defeat kernels that memoize results
  keyed on input tensor `id()`. Pairs with the Tier 2 `do_bench` adoption
  via the `setup` callback. Modest extra VRAM (~256 B × iters per tensor).
- **Safetensors input loading**. `core/bench/io.py::load_safetensors` for
  workloads whose `workload.jsonl` references safetensors blobs (some
  SOL problems carry frozen weight tensors via safetensors paths instead
  of random init). Without this, those problems error at input
  generation.
- **Subprocess-isolated evaluation**. Adopt `driver/problem_packager.py::
  ProblemPackager` + `driver/templates/eval_driver.py::_make_eval` +
  `core/bench/utils.py::make_eval`. Each candidate runs in a fresh
  subprocess. Architectural change: ACTS Coder runs `compile_kernel` +
  `verify_correctness` inline as tools, and benchmark + profile run
  inline in the orchestrator. Trades per-call subprocess latency
  (~hundreds of ms) for crash safety, OOM tolerance, and reward-hack
  robustness. The previously-separate "Subprocess-isolated correctness /
  benchmark (Tier 3)" Deferred entry folds into this scope.

**Transitive integrations** (come along with the above):

- `core/utils.py::env_snapshot` + `hardware_from_device` — needed to
  populate the `Environment` field of `Trace` (Tier 1).
- `core/utils.py::redirect_stdio_to_file` + `flush_stdio_streams` and
  `core/bench/utils.py::_read_log_file` — used by the subprocess driver
  to capture candidate-kernel stdout / stderr.
- All `core/data/workload.py` input variants — `RandomInput` /
  `ScalarInput` / `SafetensorsInput` / `CustomInput` / `ToleranceSpec`
  (Tier 1 + safetensors).
- All `core/data/solution.py` types — `SourceFile` / `BuildSpec` /
  `CompileOptions` / `SupportedLanguages` / `SupportedHardware` /
  `SupportedBindings`.
- All `core/data/trace.py` types — `Correctness` / `Performance` /
  `Environment` / `EvaluationStatus` / `Evaluation` / `Trace`.

**Stays out, even after this expansion**:

| Skipped | Why |
|---|---|
| `cli/main.py` (the `sol-execbench` shell command) | ACTS consumes SOL as a library — the orchestrator owns the eval loop. Adopting the CLI would invert the architecture. |
| `core/utils.py::is_cuda_available` / `list_cuda_devices` | Trivial wrappers over `torch.cuda.is_available()` / `get_device_name(i)`. ACTS already calls torch directly in `config.py::detect_hardware()`. |
| Compile + run path for CUTLASS / cuTe DSL / cuTile / cuDNN-frontend solutions | Tier 1 brings the enum *labels* (`SupportedLanguages.{cute_dsl, cutile, cutlass, cudnn_frontend}`) for free, but actually compiling and running kernels in those languages requires `driver/templates/build_ext.py` + `cuda-tile==1.1.0` + `nvidia-cutlass-dsl[cu13]` + cuDNN-9 headers — all hard-blocked by cu12.8. ACTS's Coder generates Triton anyway. |
| `driver/templates/build_ext.py` | Same reason — explicit `CUTLASS_DIR` env var, expects CUTLASS 3.x headers + cu13 toolchain. |
| Internal helpers: `io.py::_rand_tensor`, `_generate_heuristic_tensor`, `_cast_to_fp4x2`; `shapes.py::_BIN_OPS`, `_UNARY_OPS`; `reward_hack.py::_ELAPSED_TIME_ADDR` | Module-private; reachable through their public callers (`gen_inputs`, `resolve_shape_expression`, `check_monkey_patch`). No need to import directly. |
| SOL `tests/`, `examples/`, `data/`, `docker/`, `scripts/` | Repo infrastructure, not library code. |

**Already-integrated baseline** (unchanged by this expansion):
`set_seed` (in `eval/inputs.py::build_input_generator`),
`compute_error_stats` (in `eval/correctness.py::TorchComparisonPolicy`,
delegates when SOL importable, falls back to `torch.allclose`),
`gen_inputs` (in `build_input_generator`). These transition from
"optional with fallback" to load-bearing once SOL becomes the only
runtime path.

**Correction to 2026-04-22 entry**: Tier 4 named `check_result_caching` —
the actual SOL function is `check_lazy_outputs` (catches lazy / deferred
outputs that look like cached results). Same idea, correct name will
appear in the spec.

**Design implication carried forward**:
`reward_hack.py::_ELAPSED_TIME_ADDR` snapshots `torch.cuda.Event.
elapsed_time`'s `id()` at SOL module-load time. For `check_monkey_patch`
to be load-bearing, ACTS must `import sol_execbench` early enough that no
candidate kernel has touched torch first. Likely solution:
`pipeline/optimize.py::main` imports SOL before the Coder is constructed
or the LLM is invoked. The spec will pin the import-order contract.

**Why this is recordable now, before the spec**: scope changes shape the
options for *how* to land it (per-tier PRs vs themed mega-PR vs gated
rollout). Capturing the IN/OUT decision in JOURNAL means the option
analysis in the spec can refer to a stable scope rather than re-deriving
it. PROCESS still describes the 2026-04-22 5-tier plan as the
canonical-but-superseded baseline; PROCESS will be updated with the
expanded plan when the spec lands and we transition to writing-plans.

### SOL integration scope refinement — Tier 8 (subprocess) deferred (2026-04-27)

**Same-day revision** of the entry above. The "all in scope" decision
that promoted Tier 8 (subprocess-isolated evaluation via
`ProblemPackager` + SOL eval driver) into the active SOL integration
phase was reconsidered during design presentation. After a concrete
inline-vs-subprocess functional comparison, Tier 8 is dropped from
active scope and returned to deferred status with explicit triggers.
Tiers 1–7 stay in active scope unchanged.

**Functional parity for ACTS's actual use case**: for everything
ACTS does on the success path, inline and subprocess produce
identical results — Triton compile, 5-stage correctness gate, latency
measurement, SOL score, NCU profile (NCU is already a subprocess
from either parent), reports. Subprocess unlocks **operational**
benefits but no new capabilities:

- Crash recovery quality — kernel SEGV / illegal-memory / OOM puts
  the parent's CUDA context in sticky-error state when inline;
  subprocess fully isolates. ACTS today catches these inline and
  marks DEAD_END but recovery is coarser.
- Memory-leak isolation — child exit reclaims GPU memory; inline
  needs explicit `torch.cuda.empty_cache()`.
- Cross-iteration global-state isolation — Tier 4's
  `check_eval_integrity` catches named-function tampering but not
  arbitrary global state; subprocess catches all of it because state
  dies with the child.
- Tampering robustness against state-based reward hacks (weakref
  caches, import-time hooks) that escape Tier 4's named-function
  checks.

**Cost is real**:

- ~200–500ms per evaluation for fork + Python startup + `import
  torch / triton / sol_execbench`. At 5–10 candidates × 20–50
  iterations × dozens of problems per benchmarking sweep, this is
  10s of minutes to hours of wallclock that doesn't go to actual
  optimization.
- Doubles surface area in the PR — parent-side ProblemPackager driver
  + child-side eval_driver template + IPC serialize / deserialize
  (`Definition` + `Workload` + kernel source over JSON, `Trace` back).
- Complicates testing (two-process integration tests, IPC mocking)
  and debugging (pdb across the boundary, log aggregation across
  parent + per-eval log files).

**Current threat model doesn't justify**: ACTS's Coder is our own
LLM, generating Triton (constrained API), operating against a bounded
internal search. There's no adversary trying to game the scorer. The
5-stage correctness gate plus Tier 4's in-process `reward_hack`
detector set covers the realistic tampering surface. Triton-post-gate
crashes are rare enough that inline DEAD_END handling is functional,
even if not as crisp as subprocess recovery.

**Triggers for revisiting**:

- *Trigger A*: ACTS evaluates externally-sourced kernels — KernelBench
  external solutions, RL-discovered kernels from elsewhere, anything
  not generated by our own Coder. The threat-model assumption "our
  own LLM, well-prompted, narrow API" no longer holds, and isolation
  becomes load-bearing.
- *Trigger B*: ACTS runs on multi-tenant GPU hardware where another
  tenant could attempt cross-process tampering. Today the dev box
  is single-tenant.
- *Trigger C*: A live run shows real, frequent kernel crashes that
  disrupt the orchestrator (>1% of evaluations, or any case where
  inline DEAD_END recovery requires manual intervention). Triton +
  the 5-stage gate currently keep this rate near zero.

**SOL surfaces dropped from active scope along with Tier 8**:
`sol_execbench.driver.problem_packager.ProblemPackager`,
`sol_execbench.driver.templates.eval_driver._make_eval`,
`sol_execbench.core.bench.utils.make_eval`, plus the transitive
stdout/stderr-capture helpers `core/utils.py::redirect_stdio_to_file`
+ `flush_stdio_streams` and `core/bench/utils.py::_read_log_file`.
These remain available behind the trigger gate; adopting them later
is a clean addition (no schema changes, just a new code path parallel
to the inline one).

**Active scope after this refinement**:

- Group 1 (in-process library primitives): Tiers 1–7. Unchanged.
- Group 2 (architectural change): empty. Header removed from PROCESS.
- Delivery shape (option B): three PRs total — env bump → library
  primitives mega-PR (T1 + 3 + 4 + 5 + 6 + 7) → timing redesign (T2).
  One fewer PR than the prior 4-PR sequence.

**Why this refinement during design, not after spec**: the "all in
scope" decision was correct as a scope statement, but Tier 8 differs
from Tiers 1–7 in kind — it changes ACTS's process model, not just
which library functions we call. The functional-parity analysis
surfaced that distinction concretely. Catching it during section-1
architecture review avoids paying spec-writing and implementation
cost on a tier that was always more expensive than the others and
less needed today.

### Benchmark-agnosticism elevated to architectural commitment (2026-04-27)

**Context**: during design review of the SOL integration, the user
asked the load-bearing question — does the pipeline still work for
non-SOL kernels (custom problems, KernelBench, etc.) after Tier 1
makes SOL pydantic types canonical? Walking through the data flow
showed yes: `src/benchmarks/<name>/load.py` is the only place
benchmark-format knowledge lives, and once a `Definition` +
`list[Workload]` exist in memory, downstream stages don't know or
care where they came from.

**Decision**: elevate this from "current state" to an explicit
architectural contract recorded in PRD. The pipeline (`src/pipeline/`,
`src/search/`, `src/eval/`, `src/agents/`, `src/kernels/`,
`src/memory/`, `src/actions/`) operates exclusively on SOL pydantic
types and does not import or reference any benchmark-specific
on-disk format. The only place benchmark-format knowledge is allowed
to live is `src/benchmarks/<name>/load.py`. Adding SOL-specific code
paths outside the adapter is a violation of the contract — future
PRs that try to (e.g.) parse `definition.json` directly in the
orchestrator, or branch on benchmark category in `eval/`, should be
rejected at review.

**Why elevate now, not later**: the SOL integration mega-PR will
touch roughly half of `src/`, with several files reaching for SOL
types directly. Without an explicit architectural commitment, it's
easy for a future PR to slip in "convenient" SOL-specific code paths
in the orchestrator or eval modules — each individually small, but
cumulatively reintroducing benchmark coupling. Pinning the
benchmark-agnosticism guarantee in PRD before the mega-PR lands gives
review a clean rule to enforce against.

**Constraints recorded alongside** (apply to all benchmarks, inherent
to the SOL schema and ACTS's Triton-first approach): pure-function
reference, static output count, SOL dtype enum, GPU-only correctness
reference, Triton-translatable operations. These are real but
benchmark-agnostic — they constrain SOL-format problems and KernelBench
problems and one-off custom problems equally.

**Where pinned**: PRD → "Benchmark Source — SOL-ExecBench" section,
sub-paragraph "Benchmark-agnosticism guarantee" with the constraint
list. PROCESS doesn't need a separate entry; the existing SOL Tier 5
description and the Backlog adapter scaffold already reflect the
implementation side. Future per-benchmark adapters (KernelBench,
TritonBench, etc.) follow the same `tuple[Definition, list[Workload]]`
contract.

### Dynamic bottleneck reclassification — deferred to profiler implementation (2026-04-15)

**Context**: The orchestrator currently computes bottleneck classification once from the baseline roofline (via SOLAR) and reuses it for all iterations. This is correct for the skeleton phase — `profiler.py` is a placeholder returning zeros. However, the PRD specifies two bottleneck sources:

- **Static** (SOLAR, once at problem load): Is the *problem* fundamentally compute-bound or memory-bound?
- **Dynamic** (NCU profiling, each iteration): Is the *current candidate kernel* compute-bound or memory-bound?

Optimizations can shift a kernel's bottleneck (e.g., memory optimization moves it from memory-bound to compute-bound). When the real NCU profiler is implemented, the orchestrator loop should call `profile_kernel()` per candidate and pass the dynamic classification to memory retrieval, reviewer feedback, and planning. The static T_SOL remains constant — only the bottleneck classification updates.

**Decision**: Record and defer. No skeleton code change needed — would be routing placeholder data through a dynamic classification path. Wire when `profiler.py` gets real NCU integration.

**Superseded (2026-04-22)**: See "Bottleneck classify-once" below. The dynamic-per-iter path was not built, because the premise turned out to be wrong for this search shape: classification is invariant per `(problem, representative workload, hardware)` within a run, so a per-iter re-derivation would recompute the same answer every iteration.

### Bottleneck classify-once (2026-04-22)

**Context**: When the real analytical profiler landed (`eval/profiler.py`), the natural next step seemed to be plumbing its per-iter `AnalyticalMetrics.classification` into retriever / planner / reviewer so the search loop could react to a kernel drifting from memory-bound to compute-bound — the "dynamic reclassification" plan from 2026-04-15.

**Why the dynamic plan was wrong**: The profiler's analytical inputs are `(flops, nbytes, latency, hardware)`. For a given search run:
- `flops` and `nbytes` come from `compute_roofline_inputs(problem, representative_workload)` — invariant (we fix `repr_idx = len(workloads) // 2` once at run start).
- `hardware` is invariant.
- Only `latency` changes per iteration.

Bottleneck classification is a function of the ratio `arithmetic_intensity = flops / nbytes` against the hardware ridge point `peak_compute / peak_bw`. *Latency does not enter that ratio.* So per-iter reclassification would literally recompute the same label every iteration. The "dynamic" story was wrong about what varies.

A kernel can shift its effective bottleneck only by changing its data access pattern (shared memory tiling, coalescing, etc.) — but none of those change `flops` or `nbytes` against the representative workload the classifier sees. They change runtime / achieved bandwidth, which are diagnostic, not classificatory.

**Decision**: Classify once per run via a new `classify_run(hardware, roofline, baseline_spec)` helper in `eval/roofline.py`. Thread the `BottleneckType` result through the orchestrator as `SearchResult.run_bottleneck`, past the retriever (replaces a would-be per-iter signal), the Planner (as a dedicated `## Run context` prompt section), and the Reviewer (same). Drop the dead fields that the dynamic plan had added speculatively: `AnalyticalMetrics.classification`, `ProfilingResult.classification`, `Experience.bottleneck_after`.

**Per-workload diagnostics**: The operator can still ask "how do individual workloads land relative to the ridge" — a single representative-workload label can't answer that. Phase C populates `OptimizationReport.winner_per_workload_bottlenecks` via a second helper `classify_workload(problem, workload, hardware)` for every selected workload. This replaces the (also dropped) `OptimizationReport.bottleneck_transitions` field, which was built around the per-iter assumption.

**Superseded by 2026-04-28 ("SOLAR as sole bottleneck source")**: The per-workload `classify_workload` helper described above was deleted. Phase C's per-workload labels now come from `derive_t_sol_from_solar(...).bottleneck` directly — both bottleneck surfaces (run-level and per-workload) collapse onto SOLAR. The run-level `classify_run` decision documented above stands; only the per-workload helper was retired. See the 2026-04-28 entry below for the rationale (consistency, dtype-aware accuracy, single source of truth).

**Typing change bundled in**: The previously-deferred "Thread `BottleneckType` end-to-end" item (PROCESS → Deferred Improvements) rode along — `BottleneckType` moved from `eval/roofline.py` into a leaf `eval/types.py` module (preventing the circular-import headache that would otherwise arise once `memory/experience.py` and `eval/profiler.py` both type-check against it), and every call-site that used `.value` strings now takes the enum directly (Planner / Reviewer / `OptimizationReport.bottleneck` / `winner_per_workload_bottlenecks`).

**Follow-on Codex review fixes** (same PR):
- `src/pipeline/optimize.py` now applies the zero-peak placeholder hardware substitution to caller-supplied `ACTSConfig` as well, not just the `config is None` path. Without this, `optimize(problem_path, config=ACTSConfig())` would hit the orchestrator's new fail-fast guard and raise before the first iteration.
- `src/search/orchestrator.py` defers assigning `child.score` + `child.per_workload_latency_us` to the tree node until after the profile DEAD_END gauntlet clears. `SearchTree.best_node()` filters on `score is not None` and ignores `branch_quality`, so a ProfilerError-killed branch with a high benchmark score could otherwise be promoted as the final winner.

### Coder declares `triton_kernel_name` explicitly (T4, 2026-04-22)

**Context**: Pre-flight for the first live GPU run revealed the profiler's NCU `--kernel-name regex:` filter was sourced via `_extract_triton_kernel_name(source)` — a regex that returns the *first* `@triton.jit def` in the kernel source. For single-kernel modules this is correct. For fused kernels with helper jit functions (`@triton.jit def _epilogue` followed by `@triton.jit def main_kernel`), the regex would silently profile the helper instead of the dominant kernel — bad metrics flowing into Reviewer diagnosis without any visible failure.

**Three options considered**:

- **A — Prompt-only**: Update `prompts/coder/system.md` to mandate the `@triton.jit + host wrapper` convention. Cheap; doesn't address the multi-jit case (still picks first via regex). Failure mode is silent NCU degradation.
- **B — Prompt + tool-side validation**: Same as A plus a regex check inside `compile_kernel_tool` that asserts at least one `@triton.jit def` exists. Catches "no triton.jit at all" in-loop but does not address the multi-jit ambiguity — still picks first via regex.
- **C — Pydantic field + explicit declaration**: Add `triton_kernel_name: str` to `KernelCodeOutput` with a `@model_validator(mode="after")` that asserts the declared name appears in source as `@triton.jit def <name>`. Coder is responsible for naming the dominant kernel. Profiler reads `Kernel.triton_kernel_name` first; regex extraction stays as fallback for hand-written starters / test fixtures.

**Decision: C**, despite the wider blast radius. Three reasons:

1. **The worst failure mode of A and B is the worst kind of bug**: silently mis-profiled metrics flowing into the Reviewer's diagnosis. C surfaces the mismatch as a Pydantic validation failure the SDK can retry against, before a single subprocess runs.
2. **C aligns with the project's existing pattern**: every other agent-orchestrator boundary in the codebase (`PlanOutput`, `ReviewerFeedbackOutput`) already carries explicit Pydantic-validated metadata. The "Coder generates source, profiler regex-extracts the bit it needs" path was the only place where load-bearing metadata travelled via implicit string parsing.
3. **C moves the source of truth to the right place**. With regex-only extraction, the contract "what NCU profiles" lived in two places: the kernel source string AND the regex in `profiler.py`. If Triton evolves (`@triton.autotune` wrapping `@triton.jit`, future DSL syntax), the regex breaks silently. With C, the contract lives in the schema; regex demotes to defense-in-depth.

**Implementation**: `Kernel.triton_kernel_name` field added (default `""` for back-compat with hand-written kernels and pre-T4 checkpoints). `KernelCodeOutput.triton_kernel_name` is required, cross-validated against `triton_kernel_names_in(source_code)` (the multi-name-returning sibling of `_extract_triton_kernel_name`). `CoderAgent.implement` and `.translate` now return `KernelCodeOutput` (not bare `str`) so callers thread both fields through. `profile_kernel` resolution priority is `kernel.triton_kernel_name → regex fallback → entrypoint last-ditch`. Coder system + translate prompts both gain a Hard Rule documenting the schema.

**What's NOT in scope** (intentional YAGNI): no separate `KernelSpec.host_wrapper_name` field — `entrypoint` already plays that role at the per-problem level. No memory_store migration — the new field is on `Kernel`, not `Experience`. No regex deprecation — kept as fallback for hand-written / test kernels.

### SOLAR adapter design (2026-04-27)

**Reuse SOLAR, don't reimplement**. ACTS calls SOLAR's published Python API for all four pipeline stages (`PyTorchProcessor` → `PyTorchToEinsum` → `EinsumGraphAnalyzer` → `EinsumGraphPerfModel`) directly — no subprocess calls, no reimplementation of MAC counting / einsum conversion / roofline math. Bridge + arch YAML are the only ACTS-side code.

**Bridge: synthesize a SOLAR-shaped `Model` from `Problem` + representative `Workload`**. SOLAR expects a model file with `class Model(nn.Module)` + `def get_inputs()`; ACTS holds the reference as `def run(*tensors)` inside a `Problem` dataclass. Bridge folds const + var + expr axes (fixed-point eval for expressions like `half_head_dim = attention_head_dim // 2` from flux_rope) into concrete integer shapes. Soft-fall to `None` on bridge `ValueError` so the load path can fall back to the built-in analytical roofline rather than crashing.

**Arch resolution priority**: explicit `arch_yaml_path` (forwarded from `config.arch_config_path`) > SOLAR-bundled name (`H100_PCIe`, `B200`) > ACTS-supplied YAML (`_ACTS_ARCH_YAMLS` lookup, currently mapping `RTX6000Ada`, `NVIDIA RTX 6000 Ada Generation`, `placeholder-RTX6000Ada` all to `configs/arch/RTX6000Ada.yaml`) > fallback to `H100_PCIe` with WARNING. The placeholder alias was added after Codex flagged that the placeholder substitution path silently fell through to H100 even though the placeholder's peaks already mirror Ada — SOLAR was computing T_SOL against H100 while in-process roofline used Ada peaks.

**Read only `fused` model, ignore `unfused` and `fused_prefetched`**. `fused` matches what a well-fused Triton kernel achieves; `unfused` is too pessimistic (every tensor through DRAM) and `fused_prefetched` is too optimistic (perfect compute/memory overlap, unreachable in Triton). Multi-roofline expansion explicitly out of scope — per-op breakdown / headroom analysis is the profiler's job, not SOLAR's.

**`SolarResult.bottleneck` typed as `BottleneckType` enum, not string**. Original implementation had two parallel string-keyed mappings (SOLAR's `"memory"` / `"compute"` / `"balanced"` → ACTS's `"memory_bound"` / `"compute_bound"` / `"balanced"` → `BottleneckType` enum) that could drift. Mapped once at the SOLAR boundary; downstream passes the enum through.

**Backward-pass kernels deferred**. SOLAR ships `BackwardProcessor` for gradient-graph analysis but the bridge currently synthesizes only `Model` + `get_inputs()`. Filed in PROCESS Backlog → "Backward-kernel SOLAR support" with the schema decision (parse `Problem.op_type` suffix vs add explicit `Problem.kind` field) as an open question.

### Per-iteration analytical flops/nbytes — shape-based formulas, not SOLAR-derived (2026-05-10)

**Decision**: ACTS computes its own coarse `(flops, nbytes)` for the per-iteration analytical profiler via `src/benchmark/roofline_shapes.py::compute_roofline_inputs(definition, workload)` — a shape-based formula table — instead of reaching into SOLAR's `total_flops` / `total_fused_bytes`.

**Five reasons this is the right boundary**:

1. **The SOLAR adapter doesn't expose raw counts.** `solar_adapter.SolarResult` returns four fields (`t_sol_us`, `bottleneck`, `arithmetic_intensity`, `ridge_point`). SOLAR's internal `total_flops` and `total_fused_bytes` (`solar/perf/perf_model.py:134, 247`) are *consumed* inside SOLAR to produce that AI ratio, then dropped at the adapter boundary. Pulling the raw numbers out would require widening the adapter API and pinning ACTS to SOLAR's internal field names — the adapter was deliberately scoped to the four numbers SOLAR is authoritative for.

2. **The analytical profiler runs even when SOLAR doesn't.** `derive_t_sol_from_solar` returns `None` whenever SOLAR isn't installed; the placeholder pipeline never calls it at all. `_compute_analytical` (`src/eval/profiler.py:323`) is required, fail-closed every iteration regardless — and raises `ProfilerError` on `nbytes <= 0`. If `(flops, nbytes)` came from SOLAR, every non-SOL run would crash the analytical path. The shape-based formula is what makes the profiler torch-only, no-SOLAR-required.

3. **Per-iteration cost asymmetry.** SOLAR is a 4-stage Python pipeline (PyTorchProcessor → PyTorchToEinsum → EinsumGraphAnalyzer → EinsumGraphPerfModel) — heavy enough that ACTS runs it once at run start to derive `T_SOL` and never again. The analytical profiler runs every iteration (~30 iters × ~3 children = ~90 calls per problem). Shape-based formulas (`2·M·N·K`, `C·numel(out)`) are O(1) Python arithmetic, free at that frequency; calling SOLAR per-iter would dwarf the actual NCU subprocess in wallclock.

4. **Different fidelity needs for different consumers.** SOLAR's flops/nbytes feed `T_SOL` → SOL Score (the *primary scoring signal* — physics-accurate, fusion-aware, dtype-aware, view-elision-aware; worth its cost). The shape-based formulas feed achieved-TFLOPS / achieved-bandwidth / %peak in `AnalyticalMetrics` → *diagnostic prose for the Reviewer* (coarse is fine — the Reviewer uses these as bottleneck hints, not ground truth). SOLAR owns score correctness; the shape formulas own diagnostic signal.

5. **`KernelSpec.flop_count` / `memory_bytes` are static; workloads are parametric.** `_definition_to_kernel_spec` (`src/pipeline/optimize.py:677`) deliberately leaves both at 0 — a SOL `Definition` describes a problem family, but `(flops, nbytes)` depend on the concrete `Workload.var_axes` (M/N/K). They can't live on the spec; `compute_roofline_inputs(definition, workload)` is the per-iter, per-workload bridge between the static spec and the dynamic profile call.

**Tradeoff**: shape-based formulas overcount in the cases SOLAR's per-op overrides handle (zero-copy view ops, embedding gathered-rows, bool-typed masks). The Reviewer's analytical %peak signals will be biased downward for kernels heavy in those ops. Acceptable because (a) bottleneck classification is SOLAR-sourced, not analytical-sourced; (b) the Reviewer treats analytical %peak as a hint, not ground truth. Filed in PROCESS Backlog → "SOLAR-vs-shape-formula reconciliation" as a candidate for future discussion if Reviewer prose starts disagreeing with measured behavior on view-heavy / mask-heavy kernels.

> *Partially superseded 2026-05-13.* Reason #1 above — "the SOLAR adapter doesn't expose raw counts" — no longer holds. `SolarResult.total_flops` and `total_fused_bytes` are now surfaced through `RooflineResult` and outrank the shape-formula path inside `compute_roofline_inputs(..., roofline=...)`. The shape-formula table stays as the fallback when SOLAR isn't available *or* its counts are 0 (pure-elementwise / reduction kernels where SOLAR's MAC counter excludes non-MAC ops). Reasons #2–#5 (cost asymmetry, fidelity tiers, static-vs-parametric spec) still motivate keeping shape formulas as that fallback — the change is precedence-only, not removal. See "a+b decoupling — SOLAR counts surfaced, analytical roofline tolerates missing inputs (2026-05-13)" above.

### Hardware spec validation (2026-04-27)

**Problem**: two sources populate `HardwareSpec` for overlapping fields — `detect_hardware()` (runtime probe) for `name`, `freq_GHz`, `SRAM_capacity`, `DRAM_capacity`, and `load_hardware_spec(yaml_path)` for the same fields plus per-precision MAC tables. `load_config()` is mutually exclusive (YAML or detect, never merged), so the system can't catch a wrong-YAML-for-this-GPU misconfiguration. Same gap exists at the placeholder substitution path (Ada-shaped placeholder substituted on top of H100 detection silently routes SOLAR to the Ada YAML).

**Solution**: `validate_hardware_spec(spec, detected) -> list[str]` compares the three overlapping fields with 10% tolerance and per-field skip-if-zero. Returns mismatch messages; empty = no mismatch. Called from two sites: `load_config()` after YAML load, `optimize()` before placeholder substitution. Logs `WARNING` per mismatch (doesn't raise — sometimes you legitimately model GPU X while running on GPU Y for ablation).

**Why these three fields**: `DRAM_capacity` is the GPU-family fingerprint (Ada 48 GB ≠ H100 80 GB). `SRAM_capacity` (L2) is the within-family discriminator (Ada 96 MB vs H100 50 MB) — catches mismatches DRAM alone misses (e.g. Ada vs L40S, both 48 GB DRAM). `freq_GHz` — both sources report boost clock, so >10% delta is almost certainly wrong YAML rather than legitimate variance.

**Why not `name`**: aliases vary (`"RTX6000Ada"` vs `"NVIDIA RTX 6000 Ada Generation"`); fuzzy matching would either miss real mismatches or raise on the legitimate alias case.

### `hardware_spec` carried on `OptimizationReport` (2026-05-10/11)

**Rationale**: `OptimizationReport` previously rendered run metadata (winner score, per-workload latencies, bottleneck labels) but did not record the resolved `HardwareSpec` the run actually used. That left `report.txt` postmortems missing the single most important piece of provenance — *which hardware peaks did the SOL score / roofline classification get computed against?* — and forced the reader to cross-reference the `ACTSConfig` dump (which is pre-substitution and only carries the YAML path, not the resolved values).

**Implementation**: new `hardware_spec: HardwareSpec | None = None` field on `OptimizationReport`, threaded through `generate_report(...)` so callers populate it from the same `HardwareSpec` instance the orchestrator used. `render_report` calls a new `_render_hardware_spec_block` helper when populated; otherwise the block is omitted (back-compat with checkpoints / older callers that don't pass it).

**Why "always emit all fields, even zero"** (per `_render_hardware_spec_block`'s docstring): degraded-detection runs — no CUDA driver, unregistered GPU, placeholder substitution — populate `HardwareSpec` with zeros on the fields detection couldn't fill. Rendering "Tensor-core FP16 peak: 0.0 TFLOPS" makes the degradation visible at a glance. The alternative — skipping zero fields — would silently render a half-populated block that looks plausible until the reader notices a missing peak entry, which is the worst kind of observability bug.

**Distinction from the `ACTSConfig` dump**: the `=== ACTSConfig (resolved at run start) ===` JSON block records what the user / CLI asked for — YAML path, placeholder name, override flags. `hardware_spec` records what the system resolved that into. Both are needed: the config tells you the request, the spec tells you the answer.

### `report.txt` + ACTSConfig JSON dump persistence (2026-05-10/11)

**Rationale**: prior to this change, the rendered optimization report only existed on stdout — once the terminal scrolled past or the run was launched via a script, the report was lost. `<run_dir>/` already had `events.jsonl`, traces, the search tree dump, and `run.log`, but no human-readable summary of "what shipped." Adding `<run_dir>/report.txt` as a persisted artifact closes that gap.

**Why split stdout vs persisted file**: terminal output stays focused on the live results (the operator running interactively wants to *read* it); `<run_dir>/report.txt` carries the full rendered report plus an `=== ACTSConfig (resolved at run start) ===` JSON dump appended below it for offline postmortem and reproducibility. The two surfaces serve different consumers — keeping them separate avoids cluttering the live stdout with the config dump while still capturing it for reproducibility.

**Best-effort write contract**: `main()` writes `report.txt` after `optimize()` returns. The write is wrapped in a try/except on `OSError` (disk full, permissions, read-only mount) that logs a `WARNING` and continues — a failed persistence write does *not* abort the run or mask the successful optimization result. The run's primary artifacts (tree dump, events, traces) have already landed by this point; the report is a derived, secondary artifact, so a write failure here degrades observability rather than correctness.

### Operator-supplied Triton baseline — bypass `CoderAgent.translate()` (2026-05-16)

**Trigger**: the LLM-translation path ("Triton baseline via LLM translation" above) was the only way to seed `T_b`, but operators arriving with a hand-tuned single-config kernel, a `@triton.heuristics`-style kernel, or an external-library port (FlashAttention reference, cuBLAS Triton port) had no way to anchor the run on that vetted artifact. Live runs were re-translating known-good kernels on every invocation, paying LLM tokens + retries to reproduce something the operator already had on disk. The fix is a second baseline strategy — `load_operator_baseline` — that reads a path from cfg, runs the same correctness gate the translator path runs, and skips the Coder entirely.

**Q1 — Autotune validator flag-controlled, default skip.** The autotune contract enforced for Coder-emitted kernels (`KernelCodeOutput._autotune_decorator_well_formed`: ≥4 `triton.Config` entries, non-empty `key=[...]`) is opt-in for operator-supplied baselines, gated on a new `[runtime] triton_baseline_enforce_autotune: bool = False`. *Why*: operators with hand-tuned single-config kernels, heuristics-decorated kernels, or external-library ports should be accepted as-is. *Trade-off*: default-skip means a non-autotuned root competes against A1-mandated autotuning descendants — slightly asymmetric search, but acceptable since operators picking this feature are deliberately anchoring on a vetted artifact.

**Q2 — Auto-detect kernel name + cfg field for dps.** `triton_kernel_name` is auto-resolved via `triton_kernel_names_in()` when the source has exactly one `@triton.jit def`; multi-kernel files require explicit `[runtime] triton_baseline_kernel_name`. `dps` is unknowable from source (host-wrapper runtime contract) and is always cfg-supplied. *Why*: minimal boilerplate for the 95% single-kernel case; explicit when ambiguous.

**Q3 — Hard-fail, no retries, no fallback.** Any post-load gate failure raises `BaselineGenerationError` with the failing stage in the message; no LLM retry loop (there's no Coder to re-prompt); no fallback to `generate_triton_baseline` (would defeat the wallclock trigger and silently mask operator bugs). *Why*: the whole point of this feature is "operator already has a vetted kernel" — if the vetted kernel fails, the operator wants to know, not have the run quietly switch strategies.

**Q4 — Call-site dispatch in `pipeline/optimize.py` (`_dispatch_baseline`).** Each baseline strategy (`generate_triton_baseline`, `load_operator_baseline`) stays a separately-testable single-purpose function. The future curated per-op starter library (PROCESS.md Tier C, investigation item C4) plugs in as another branch at the same seam. *Why*: simpler than an umbrella function with a strategy registry; each baseline path's tests stay focused on one source-of-kernel.

**Q5 — Warn-not-raise on stray cfg keys.** When `triton_baseline_path` is unset but other `triton_baseline_*` keys are set, `ACTSConfig.__post_init__` logs a `WARNING` listing the stray keys. *Why*: operator can comment out `triton_baseline_path` for a quick A/B run without scrubbing the rest of the cfg. Aligns with `validate_hardware_spec`'s warn-not-raise pattern.

**Codex adversarial review finding (HIGH, fixed)**: the original `load_operator_baseline` gate sequence only checked that `triton_baseline_kernel_name` appeared as a `@triton.jit def` somewhere in source. But correctness later executes `spec.entrypoint` (the host wrapper). A source can pass correctness while declaring one JIT kernel and launching a different one — profiler/autotune attribution then silently targets the never-launched function. *Fix*: a new pure-AST helper `find_jit_name_in_entrypoint(source, entrypoint, jit_name) -> tuple[bool, str]` (`src/eval/profiler.py`) verifies the entrypoint actually launches `jit_name` via one of three patterns: subscript launch (`<jit_name>[grid](args)`), `.run` call, or a single-level alias. Direct-launch pass-through handles `jit_name == entrypoint`; missing-entrypoint and unparseable-source defer to the compile gate.

**Symmetric application at three call sites, with intentional failure-handling asymmetry**: every place a Triton kernel becomes either a baseline anchor or a tree node now runs the same launch-binding check:

- `generate_triton_baseline` (LLM baseline, `baseline_generator.py`): append to `prior_failures`, continue retry.
- `load_operator_baseline` (operator baseline, `baseline_generator.py`): raise `BaselineGenerationError`, hard-fail.
- Orchestrator per-iter K-way candidate (`src/search/orchestrator.py`): emit `coder_failed` event, drop candidate, K-way fan-out picks another survivor.

The baseline-side hard-fail vs candidate-side drop is intentional: a mis-bound baseline corrupts the run anchor for every subsequent iteration, so the operator needs to see it; a mis-bound candidate is one of K and the K-way reliability layer is designed to absorb single-candidate losses.

**Documented remaining gap**: string-keyed indirection (`getattr(module, "name")[grid](...)`), dynamic kernel selection from dicts/lists, and multi-hop alias chains. The realistic attack surface is covered; exotic cases get a clear error rather than silent mis-attribution.

**Sibling to investigation item C4** (PROCESS.md Framework Enhancement Investigation Tier C — curated per-op starter library): this entry is the operator-runtime version. C4 will reuse `load_operator_baseline` unchanged; the C4 work becomes an `op_type → path` registry sitting in front of the existing loader at the same `_dispatch_baseline` seam.

**Update (2026-05-17): explicit dispatch flag.** Mid-implementation user-review feedback flipped the dispatch shape and tightened Q5. The implicit "truthy `triton_baseline_path` ⇒ load operator baseline" toggle is retired in favor of an explicit `use_operator_baseline: bool = False` field on `ACTSConfig`. `_dispatch_baseline` in `src/pipeline/optimize.py` now branches on the flag alone; the path field becomes a pure data input.

*Why*: with an implicit toggle, operator intent is buried inside the path field — reading the cfg, you can't distinguish "operator wants the loader and forgot to set the path" from "operator wants the LLM translator and left a stale path lying around." The explicit flag makes the cfg self-documenting and gives `_dispatch_baseline` a single source of truth instead of inferring intent from a string's emptiness.

*Q5 revised — asymmetric consistency rules at cfg load*:
- **flag=True + `triton_baseline_path` empty → raise `ValueError`** in `ACTSConfig.__post_init__`. The operator declared intent to use the loader but didn't supply the file; this is a fail-fast operator error, not a config quirk, and surfacing it at cfg load beats letting the loader entry raise `BaselineGenerationError` mid-run.
- **flag=False + any stray `triton_baseline_*` field set → `WARNING`**, listing the stray keys. Same Q5 spirit (mid-iteration A/B-toggle tolerance), but the warn-set is wider now — it includes `triton_baseline_path` itself, since the field is no longer the dispatch toggle and a leftover value is just dead config.

*Trade-off*: the asymmetric strictness (raise where the operator clearly stated intent, warn elsewhere) catches typos and forgotten-toggles at cfg load instead of as a later `BaselineGenerationError` from the loader, while still tolerating the "comment out the flag for a quick A/B run without scrubbing every related key" workflow Q5 was designed to support.

---

## Backend

### Triton (V1)

**Rationale**: From domain researchers: **agents are not good at writing CUDA-level code** — too complicated, small differences cause huge performance variation.

Triton effectively gives us Tiers 1-3.5. CUDA gives all 6 tiers — but the agent can't reliably use Tiers 4-6. Having knobs the agent can't turn wastes search budget: a failed Tier 5 CUDA attempt costs a full iteration while a successful Tier 2 Triton attempt adds a real tree node.

**Agent success rate matters more than peak performance ceiling.** KernelEvolve (Meta) validates this: uses Triton, achieves 100% pass rate on KernelBench, works cross-hardware. Tiers 1-3 already yield 10-50%+ gains for most kernels — sufficient to prove the ACTS architecture.

**Known limitation**: V1 cannot compete with hand-tuned libraries (cuBLAS, cuDNN, FlashAttention) on kernels requiring warp specialization or architecture-specific intrinsics. Deliberate tradeoff — prove framework first, chase peak performance later.

---

## Memory

### Optimization-memory rewrite — distilled lessons, not raw experiences (2026-05-28)

**Rationale**: The v1 `Experience` schema (`metrics`, `reviewer_summary`, `bottleneck_before`, `success`, `hardware`) had a retriever wired into the orchestrator but **no producer code anywhere in `src/`** — the store was read-only by construction. The schema also blurred two questions the literature treats separately: what raw evidence was collected (profile dict, bottleneck label, correctness flags) and what reusable optimization advice the run produced. The first changes every iter; the second is what cross-run transfer learning actually needs.

**Decision**: Rewrite the schema as an AccelOpt-style distilled lesson — `(title, lesson, snippet_before, snippet_after)` produced by a summarizer LLM from a `(parent_kernel, child_kernel, speedup)` triple — and add the missing producer. The row carries the lesson + the changed-region snippets, not the raw profile or full source. Profile data stays live-only (consumed by the Planner via the current-iter profile dump, never persisted into opt-mem).

Concrete axes (see `doc/specs/2026-05-24-optimization-memory-design.md` for the full decision table):

| Axis | Choice | Why |
|------|--------|-----|
| Granularity | Per-improving-edge (G1) + one cumulative G3 row at run end | The orchestrator tree already exposes every (parent → child) edge with score and runtime; G1 is plumbing. G3 captures multi-step strategy the per-edge rows lose. |
| Row shape | Distilled summary + snippet (B-Lesson) | AccelOpt has shipped numbers showing this works; B-Raw (profile-delta rows) is what we'd be inventing without prior art. |
| Injection | Planner only (I1) | ACTS's Planner picks from a typed action vocabulary — cross-kernel lessons are highest-leverage at action selection. The Coder's decisions (tile sizes, vectorization) are kernel-specific; lessons are noise there. |
| Retrieval | Kernel-type filter + same-arch preference + `speedup ** α` weighted random with replacement (R3) | Pool size on day 1 is zero; on day N is a few hundred. Random sampling within filters keeps the Planner seeing diverse lessons instead of the same top-3 every iter; α weights toward higher-impact lessons without hard-pinning. |
| Storage | JSONL append-only, one global shared file (S2) | Append is O(1) per row (vs. v1's whole-file rewrite per `add`), crash-safe per row via `f.flush()`, greppable for debugging "why did the Planner choose X." |
| Payload | Summary + snippets embedded in the row; no full source, no sidecar | User confirmed run dirs get cleaned (so any pointer back into a run dir would dangle); single self-contained file. |
| Class name | Kept `Experience` (rewrite fields only) | Rename to `Lesson` was taxonomy-for-its-own-sake; the class still represents "something we learned from one optimization attempt." Less churn across `planner.py`, tests, and downstream docs. |
| Summarizer model | Reuse shared `self._model` (M1) | Per-role model dispatch is a separate, larger refactor that affects all four agents equally. Defer until there's empirical evidence the summarizer wants a different model. |

**Config** — five new knobs on `ACTSConfig`, all under `[memory]` in the `.cfg`:

- `opt_mem_read_enabled=True` — short-circuit retrieval to `[]` when False.
- `opt_mem_write_enabled=False` — **OFF by default**, opt-in per run. Ablation runs cannot pollute the shared store without explicit intent.
- `opt_mem_writes_per_session_cap=20` — top-N edge rows by speedup; cap reserves 1 slot for the G3 cumulative row.
- `opt_mem_min_improvement_ratio=1.05` (δ) — per-edge threshold; below this, kernel-timing noise dominates.
- `opt_mem_speedup_weight_alpha=1.0` (α) — exponent on speedup in the retriever's weighted sample.
- `opt_mem_store_path=Path("opt_mem/store.jsonl")` — single shared global file.

**Failure modes** are bounded: every summarizer error path (LLM raises, malformed JSON, `"No optimization found"` title, empty/identical snippets) returns `None` with a single warn log. The orchestrator's `_flush_opt_mem(root, tree)` helper wraps finalize + flush in a try/except so opt-mem hiccups never break a successful search return. Matches `coder.py`'s skip-iter-on-agent-hiccup pattern: opt-mem is best-effort by design.

**Migration**: zero call sites referenced v1 fields outside the memory module + the unwired orchestrator integration. No production data existed (no writer was ever wired). Wholesale rewrite in place, no deprecation cycle, no legacy file handling.

**Deferred** (out of scope for this round; spec §14): in-session reflexion (KernelAgent-style "what worked / avoid_patterns"), per-role model dispatch (Summarizer with its own model config), workload-shape similarity in retrieval, symptom-keyed retrieval (would require putting profile data back in the store, which the B-Lesson decision rejects), seeded pattern library, concurrent-writer safety (`flock()`).

---

## Development Process

### CLI → cfg consolidation (2026-05-11)

**Rationale**: The CLI carried five flags (`problem_path` positional, `--run-dir`, `--trace-dir`, `--reset-clocks`, `--gpu-index`) alongside an unused `load_config()` that read a `.cfg` file. Three of the five (`problem_path`, `--reset-clocks`, `--gpu-index`) were *invocation knobs* that belonged with the algorithmic config (`beam_width`, `sol_target`, …) but had been promoted to CLI for ease-of-tweak. Result: two ways to configure ACTS, and a stranded `load_config()` that nobody called from production.

**Decision**: collapse onto `.cfg` as the single configuration surface, in **libconfig format** (parsed by the pure-Python `libconf` package, added to `pyproject.toml`). CLI shrinks to three flags — `--config`, `--run-dir`, `--trace-dir` — which are the genuinely *per-invocation* knobs (where does this run's cfg/artifacts live). Everything else moves into `ACTSConfig` and `load_config()`:

- `runtime.problem_path` — replaces the positional argument.
- `runtime.reset_clocks` — replaces `--reset-clocks`. Toggle on, run once, toggle off.
- `hardware.gpu_index` — replaces `--gpu-index`. Module-top preparse opens the libconfig cfg (via `libconf`, pure-Python and CUDA-free) before any CUDA-aware import, extracts `gpu_index`, and sets `CUDA_VISIBLE_DEVICES`. The ordering constraint is preserved: cfg path is scanned out of `sys.argv` (`_preparse_config_path`), the cfg is opened, the value is read, and `os.environ["CUDA_VISIBLE_DEVICES"]` is set — all before `import sol_execbench`. The import-order contract test (`test_import_order_contract_sol_first`) gains a `libconf` allowlist entry since the package is pure Python and can't compromise SOL's reward-hack address snapshot.

**Why not flatten everything onto CLI instead**: 17 algorithmic knobs as CLI flags would balloon `--help` and force users to memorize or rediscover the surface for every run. `.cfg` already had the right shape (file-based, idempotent, diff-friendly). The argparse path was the orphaned one.

**Why libconfig (libconf) over INI (configparser)**: native types (`bool`, `int`, `float`, `str`, nested groups) without string-coercion plumbing, and proper nested-group syntax instead of flat `[section]` headers. INI's quirks (every value is a string until you call `getboolean`/`getint`/`getfloat`; nested data needs flattening into `[section.subsection]` conventions) accumulated friction that the rewrite removed. libconf is pure Python and adds only ~25 KB to the dep set.

**Why not delete CLI entirely**: `--run-dir` and `--trace-dir` are *truly per-invocation* — every run wants its own artifacts directory. `--config` is the bootstrap pointer; without it the loader doesn't know which cfg to read. Keeping these three preserves the "set everything once in `acts.cfg`, vary only artifacts per run" workflow without one-cfg-per-problem proliferation.

**Default behavior without `--config`**: `ACTSConfig()` with `detect_hardware()` — matches the prior all-defaults path so `python -m src.pipeline.optimize` still runs the placeholder matmul smoke test. The cfg-less path is the smoke test; the cfg path is real work.

**Tripwire**: `main()` asserts `acts_config.gpu_index == int(_GPU_INDEX)` after argparse so the module-top preparse and the in-`main()` load agree. A mismatch means the cfg file was edited between import time and `main()` — a deployment bug, not user error.

**Source**: user observation that "there are two ways to config" in a 2026-05-11 design discussion.

### Always-runnable framework

**Rationale**: Prevents the common failure mode of building a large codebase that doesn't run until everything is done. By keeping the framework complete-but-shallow, we test pipeline flow early and catch integration issues before investing in deep implementation.

### Logger system before first live GPU run (2026-04-23)

**Context**: The first multi-minute live run was about to kick off with zero progress signal — every `logger.info`/`logger.warning` was silently dropped (no `basicConfig`), reducing post-mortem to a single final exception line. Wrong forensic surface for a run spanning many LLM calls and GPU subprocesses.

**Three sinks, not one**: `run.log` (human tail-able), `events.jsonl` (structured snapshots for tooling/ablation), `traces/*.jsonl` (per-call SDK records, reusing `JSONLTraceProcessor` from `5281cdf`). Each answers a different question; collapsing them would force each consumer to reparse another's format.

**Coder event truthfulness** (Codex adversarial review catch): originally emitted `coder_compiled(passed=True)` + `coder_correctness(passed=True)` on `implement()` return, but the orchestrator cannot verify those gates from the return value — the SDK's `submit_kernel` validates the structured output, not the gates. Changed to `coder_submitted` (no pass claim) and `coder_failed(reason)`; per-tool-call detail lives in `traces/*.jsonl`.

**Microsecond-precision run-dir names**: second-precision collides when ablation scripts or CI jobs share `--run-dir`. Same format now used by `trace_processor.py`, consolidated via `src/runtime/timefmt.py::filename_ts`.

**RunContext owns trace wiring**: post-review refactor removed `main()`'s `explicit_trace_processor` + `_enable_traces_if_possible` helper. `RunContext.create(trace_dir=...)` now owns default and override paths.

**Deliberately out of scope (v1)**: no Rich/tqdm live terminal UI (plain stdlib + `jq` is enough), no log rotation / disk quota / size caps (one run ≈ a few MB), no remote log shipping (Loki / Datadog), no "resume a run into the same run-dir" (new `main()` always creates a fresh `run_<UTC>/` — resume is a checkpoint concern, not a logger one), no cross-run aggregation index, no per-agent sub-loggers beyond stdlib `getLogger(__name__)`. Revisit triggers: live UX pain during multi-hour batches (→ Rich), disk pressure on long CI (→ rotation), need to compare runs (→ index).

### `ACTS_OPENAI_DEBUG` opt-in for openai/httpx DEBUG logs (2026-05-13)

**Context.** The logger entry above clamps `openai`, `httpx`, and `agents` at WARNING uniformly — their DEBUG output (per-request bodies, full message histories) would bury the per-iter ACTS narrative in `run.log`. Right default until a thinking-model failure (DeepSeek-reasoner returning `finish_reason="length"` mid-tool-call, empty `choices[0]`, malformed reasoning block) has to be diagnosed from `run.log` alone — at which point the silenced lines are exactly the ones needed.

**`_silenced_loggers()` gates on `ACTS_OPENAI_DEBUG`** (`"1"` / `"true"` / `"yes"` truthy). When set, only `agents` stays at WARNING; `openai` and `httpx` inherit root-logger DEBUG, so request/response bodies, `finish_reason`, and raw `choices[0]` land in `<run_dir>/run.log`. `agents` stays silenced — its DEBUG noise is structural SDK trace plumbing already captured in `<run_dir>/traces/*.jsonl`.

**Why env-var, not a cfg flag.** Diagnostic verbosity is a per-invocation knob and shouldn't drift into the persisted `acts.cfg`. Also: cfg is read after `RunContext.create()` (per "CLI → cfg consolidation, 2026-05-11"), so cfg-threaded would be a layer-ordering change for a debug-only path.

**Why default off.** Request bodies — system prompts, tool definitions, full message histories — persist to disk. Verbose (tens of KB per Planner call) and may carry sensitive data (kernel sources, NCU dumps, fixture API tokens). Default-on would silently fatten `run.log` and create a leak surface in shared-checkpoint scenarios.

### Correctness tolerance — adopt SOL-ExecBench's defaults verbatim (2026-04-26)

**Context**: First successful logger run against `examples/triton/rmsnorm/` exposed that the Coder was producing structurally correct bf16 RMSNorm kernels — compile passing, math right — that all failed correctness with `max_abs ≈ 7.812e-3` on workload 2/3. That value is exactly `2^-7`, the bf16 ULP at unit magnitude. Our `verify_correctness` defaults (`atol=rtol=1e-3`, `required_matched_ratio=1.0` hardcoded in `TorchComparisonPolicy.compare`) sat *below* bf16's quantization noise floor, making the acceptance test mathematically unsatisfiable for the dtype. The Coder kept iterating until the turn budget ran out, producing the misleading symptom `MaxTurnsExceeded`.

**Decision**: align with SOL-ExecBench's `ToleranceSpec` defaults verbatim — `max_atol=max_rtol=1e-2`, `required_matched_ratio=0.99`. Three reasons:

1. **The bar SOL ships is the bar SOL expects to be tested at.** Our ACTS gate evaluates SOL problems; using a stricter bar than SOL itself rejects kernels SOL would accept and silently breaks the contract we're benchmarking against.
2. **bf16 ULP is a physical floor, not a tunable.** No amount of Coder iteration produces a bf16 output closer to the fp32 reference than ~7.8e-3 at magnitude 1. Tightening below this floor is asking for the impossible.
3. **The 1% slack absorbs outliers, not bugs.** SOL's `required_matched_ratio=0.99` lets ~1% of elements fail the per-element bound while still passing overall. The hard `max_error_cap` in `compute_error_stats` would still catch a kernel with rare catastrophic outliers, so the slack doesn't unsafe the gate — it just stops it from false-flagging fp32→bf16 round-trip noise.

**Implementation**: `TorchComparisonPolicy.compare` no longer overrides `required_matched_ratio` — passes `ToleranceSpec(max_atol=atol, max_rtol=rtol)` and lets SOL's default kick in. Single source of truth, zero literal `0.99` in our code. `verify_correctness` defaults `atol=rtol=1e-2` mirror SOL's `max_atol=max_rtol`. Anti-cheat (stage 5) keeps its independent `strict_atol=1e-5, strict_rtol=1e-4` — that gate is ours, not SOL's, and serves a different threat model (reward hacking under randomized inputs).

**Drift sentinel**: `tests/test_correctness.py::test_verify_correctness_atol_rtol_defaults_match_sol_execbench` reads `ToleranceSpec()` defaults at runtime and asserts the function signature defaults match. If SOL bumps to e.g. `1.5e-2`, the test fails and forces an update. Test skips gracefully when `sol_execbench` isn't importable (tier-1 venv).

**What's NOT in scope**: dtype-aware tolerance table (e.g., bf16→1e-2, fp16→5e-3, fp32→1e-4) — premature; SOL itself didn't bother and treats one set of defaults as universal. Per-problem `tolerance` overrides — schema-supported by SOL's `Workload.tolerance` field but never exercised in any shipped example, so plumbing it through buys nothing today. Loosening the anti-cheat strict tolerances — those are an independent gate and the previous strict values still match how the stage is documented in PROCESS / doc/eval.

### Workload `tolerance` overrides stage-5 anti-cheat (2026-05-13)

**Amendment to the 2026-04-26 "Correctness tolerance" entry above.** Two of its "NOT in scope" items now flip: per-problem `Workload.tolerance` overrides are wired through, *and* they override stage-5 strict tolerances (not just stages 1–4). `verify_correctness` reads `workload.tolerance` at the top — when present, its `max_atol` / `max_rtol` overwrite `atol` / `rtol` *and* `strict_atol` / `strict_rtol` for the rest of the call.

**Why override even the strict gate.** Stage 5's prior hardcoded `1e-5` / `1e-4` were tighter than SOL's defaults so a kernel couldn't pass on canned seeds and fail elsewhere. Once a workload ships an explicit `ToleranceSpec`, that spec *is* the acceptance contract — rejecting at a stricter bar rejects kernels the benchmark would accept (same failure mode the 2026-04-26 entry fixed for the loose defaults). The anti-cheat semantic ("fresh seeds, no overfit to canned inputs") is preserved by stage 5's seed-1000+ trials; only the tolerance changes.

**Opt-in via workload presence.** Callers passing `workload=None` keep the prior tight `1e-5` / `1e-4` behaviour verbatim. Override fires only when both a `Workload` is supplied AND its `tolerance` attribute is non-None.

**Drift sentinel.** `tests/test_correctness.py` pins three cases: tighter workload tol fails stage 1, looser workload tol passes stage 5 where defaults fail, and `workload=None` preserves defaults. The strict-override case is load-bearing.

### Planner + Reviewer submit-tool migration (2026-04-26)

**Context**: The first live GPU run with the logger system (2026-04-26) cleared baseline (~50 s) and died on the Planner's first call with `agents.exceptions.UserError: additionalProperties should not be set for object types. Strict JSON schema is enabled, but the output type is not valid.` — same SDK error family the Coder hit on 2026-04-22 (Pydantic `dict[str, X]` fields trip the SDK's strict-schema validator). The obvious quick-fix `strict_json_schema=False` does **not** solve it: reading `agents/models/chatcmpl_converter.py:104-111` confirms the SDK still wires `response_format=json_schema` on the chat-completions request regardless of the strict flag (the flag only toggles the `"strict"` sub-field), and DeepSeek-reasoner rejects **any** `response_format=json_schema` on its endpoint. `output_type=` is a dead path for both Planner and Reviewer on our chosen backend.

**Decision**: apply option α (Coder's 2026-04-22 fix) uniformly. Planner gets a `submit_plan` tool, Reviewer gets a `submit_review` tool. Each agent builds a fresh `Agent` per call with `[submit_*]` as the only tool, no `output_type`. Pydantic validation runs **inside the tool body**; on failure the tool returns a standard error string (`SUBMIT_PLAN_FAILED:` / `SUBMIT_REVIEW_FAILED:`) so the SDK hands it back to the LLM as an observation, preserving the in-loop validation-retry behavior of the old flow. Shared `SUBMIT_OK_SENTINEL` + `format_submit_validation_error()` helpers live in `llm_backend.py`; the Coder was refactored onto these helpers so all three agents share one error-shape contract.

**Turn budget — `max_turns=4` for Planner and Reviewer**: `2N + 2` with `N=1` in-band validation retry (turn 1 invalid submit, turn 2 corrected resubmission, turn 3 plain-text confirmation that terminates the SDK loop, plus 1 buffer). Coder's existing `2 × max_debug_retries + 2` formula is unchanged — it owns a multi-tool debug loop; Planner/Reviewer have only the one tool.

**Failure-handling decision matrix**:
- *Planner failure* → typed `PlanningError` from `Planner.plan()`. Orchestrator catches at the per-iteration boundary (mirrors the existing `ImplementationError` branch from the Coder), logs a warning, decays epsilon, skips the iteration without adding a tree node.
- *Reviewer failure* → falls through to the existing rule-based degraded fallback with new `error_reason` tags (`max_turns_exceeded`, `missing_submit_review`) added to the existing transient-API tags. Operator sees the degradation in the run log; orchestrator still gets a usable `ReviewerFeedback`.
- *`MaxTurnsExceeded` recovery* — both agents prefer the captured submission if `submit_*` did get called before the budget ran out (option-γ pattern from the Coder migration); otherwise raise (Planner) or degrade (Reviewer).

**Quarantine fix (Codex adversarial review catch)**: skip-iteration alone is insufficient for Planner failure. `select_next` is greedy; a parent that consistently makes the Planner fail keeps getting re-picked, burning the entire `max_depth` budget on a dead branch. Fix: `consecutive_agent_failures: int` counter on `TreeNode`, incremented in **both** the Coder and Planner orchestrator catches, reset on successful `tree.add_child(...)`. `SearchTree.frontier()` excludes any node at or above `QUARANTINE_THRESHOLD = 2`. `best_node()` intentionally still considers quarantined nodes — quarantine is a forward-expansion gate only; if the quarantined node is still the best result, it remains a valid final answer.

**Codex review fixes folded in**:
- `LLMBackend.has_model` now gates on `_SDK_AVAILABLE` in addition to model presence. Pre-migration, SDK-absent + model-stub silently fell into the rule-based fallback; post-migration the `submit_*` wrappers would raise `TypeError` because `function_tool` is unavailable. Gating `has_model` restores the silent-fallback behavior test fixtures rely on.
- `submit_plan` / `submit_review` signatures mirror Pydantic optionality (`params: dict | None = None`, etc.). The old `output_type=` path accepted omitted optional fields via Pydantic defaults during deserialization; the new tool path needs explicit Python defaults so the same omissions land at the validator with the same shape.
- `max_turns 2 → 4` to reserve the in-band validation-retry budget. The pre-migration `2` was sized for `output_type=` (one turn structured output + one confirmation) and would have left zero room for in-loop correction.

**What is NOT in scope** (intentional YAGNI):
- *No generic submit-tool factory.* Three agents on the pattern is below the threshold where DRY-ing pays; per-agent typed wrappers are the right granularity, and `function_tool` requires explicit signatures (a generic factory would defeat its parameter introspection).
- *No removal of the Reviewer's rule-based fallback.* It still serves transient-API blips that predate this migration; collapsing to "raise on any LLM failure" would regress run robustness.
- *No expansion of the Reviewer into a multi-tool agent.* The "Multi-turn Reviewer deferred" entry (2026-04-21) still stands — wire-format migration is unrelated to that trigger.
- *No per-agent quarantine threshold tuning.* `QUARANTINE_THRESHOLD = 2` applies uniformly until live-run data argues otherwise.

**Trigger for revisit**:
- First live run where quarantine fires on a non-pathological parent (suggests threshold too aggressive, bump to 3 or split per-agent).
- First live run where `max_turns=4` is too few for Planner/Reviewer validation retries (bump to `2N+2` with `N=2`, i.e. `max_turns=6`).
- First live run where DeepSeek consistently omits both required *and* optional `submit_*` fields (fix the prompt, not the budget).

### Strict-mode opt-out for submit-tool dict params (2026-04-26)

The first live GPU run after the Planner + Reviewer submit-tool migration (rmsnorm on RTX 6000 Ada, runs/run_20260426T152032_091547Z/) died on iter 1's first `function_tool(submit_plan)` call with `agents.exceptions.UserError: additionalProperties should not be set for object types`. Same error class that originally killed the `output_type=Pydantic` path — but raised against the *tool parameter* schema this time, not the output schema.

**Root cause**. The migration moved structured submission from `output_type=OptimizationPlanOutput` to a `submit_plan` function tool. The Pydantic model still has a `params: dict[str, str]` field, and the corresponding tool param has the same annotation. The SDK's strict-schema validator (`agents.strict_schema._ensure_strict_json_schema`) walks both sides — output schemas *and* tool parameter schemas — and rejects `dict[str, X]` because the JSON-schema translation produces `additionalProperties: {type: ...}`, which strict mode forbids. The Coder migration didn't hit this because `submit_kernel(source_code: str, triton_kernel_name: str)` has no dict params.

**Fix**: pass `strict_mode=False` to the `function_tool` call when registering the submit tools (`function_tool(_make_submit_plan_tool(captured), strict_mode=False)`, same for reviewer). The SDK's `function_tool` exposes this exact escape hatch in its docstring; setting it to False skips the pre-flight strict-schema validation. The tool schema is still sent to the API; only the SDK-side strictness check is disabled.

**Why this is safe**. End-to-end type safety is preserved because Pydantic validation runs *inside the tool body*: the tool calls `OptimizationPlanOutput(**kwargs)` and returns `format_submit_validation_error(...)` on `ValidationError`, which the SDK relays back to the LLM as the tool-call response. Malformed payloads bounce through the existing in-loop retry budget (`max_turns=4` reserves room for one corrective resubmit). The validator we lose is OpenAI-side coercion of arg types before the function call lands; we already validate downstream of that, so the loss is redundant.

**Why not restructure the dict to a JSON-string param** (the alternative considered). It would keep strict mode but require the LLM to nest-encode JSON inside a JSON tool call — an awkward UX that hurts model accuracy on the structured fields we care about. The strict-mode opt-out is the documented SDK affordance for exactly this case.

**Tests**. Two changes:
1. Widened all 20 existing `side_effect=lambda f: f` patches in `tests/test_planner.py` + `tests/test_reviewer.py` to `lambda f, **kw: f` so they don't `TypeError` on the new kwarg.
2. Added `test_submit_tool_registered_with_strict_mode_false` in each agent's test file — a regression guard that records `function_tool` kwargs and asserts `[{"strict_mode": False}]`. Catches future "tidy-up" refactors that delete the kwarg without realizing it's load-bearing.

**Validation**. Re-ran the live GPU run after the fix — Phase A → B → C completed cleanly in 406s, plateau termination after 3 iterations (no candidate beat baseline). Submit-tool migration now genuinely works end-to-end.

**Trigger for revisit**:
- A future SDK release tightens `strict_mode=False` semantics in a way that breaks our tool calls. Re-evaluate the JSON-string-param alternative if so.
- A future Pydantic model adds a *required* `dict[str, X]` field whose schema the LLM gets wrong frequently. Strict-mode would catch the wrong shape one round earlier; the in-loop retry catches it one round later. Acceptable trade today, revisit if retry budget pressure rises.

### SOLAR as sole bottleneck source (2026-04-28)

Both bottleneck classification surfaces — run-level and per-workload — now derive from SOLAR. Previously the per-workload surface in `OptimizationReport.winner_per_workload_bottlenecks` used the analytical band classifier (`flops/nbytes` ratio compared to the hardware ridge); now it calls `derive_t_sol_from_solar` once per selected workload and reads `RooflineResult.bottleneck`.

**What changed.** Three coordinated edits:

1. **`classify_workload(definition, workload, hardware) -> BottleneckType` deleted** from `src/eval/roofline.py`. It was unreachable in production — `report.py::generate_report` had been duplicating its logic inline (because `classify_workload`'s strict ValueError-on-no-formula didn't fit the placeholder-path fallback in `_resolve_workload_roofline`), and the helper itself was only exercised by 7 tests. Pure dead weight.
2. **`report.py::generate_report` per-workload classification path** swapped from `classify_bottleneck(flops/nbytes, ridge_point)` to `derive_t_sol_from_solar(definition, w, hardware_spec, arch_yaml_path=...).bottleneck`. Workloads where SOLAR returns `None` (not installed) or `definition is None` (placeholder mode) are omitted from the dict rather than fall back to the analytical classifier.
3. **`optimize.py` rich-args plumbing**: `optimize()` now returns `(SearchResult, OptimizationReport)` and builds the report inside its own scope where `definition` / `workloads` / `arch_yaml_path` / `blob_roots` are in scope. Previously the bare `generate_report(result)` call in `main()` couldn't access those locals, so the per-workload classification never fired in production runs. Net effect: post-2026-04-28 production runs actually surface per-workload SOLAR labels in their printed report.

**Why SOLAR everywhere instead of analytical.**

- *Consistency.* The SOLAR-vs-analytical split was a vestige of the bottleneck-classify-once refactor (2026-04-22), which rightly avoided running SOLAR per-iteration but left the per-workload surface analytical because Phase C was the only consumer and "cheap is fine." Q2's correction: Phase C runs once per run, so the cost is a one-shot `O(N_workloads)` SOLAR-pipeline invocation at report time. ~10 workloads × ~hundreds of milliseconds each = sub-second overhead at the very tail of a run that already took minutes.
- *Accuracy.* The analytical classifier uses `peak_flops_fp32` regardless of workload dtype — for tensor-core workloads (fp16/bf16) it mislabels because the real ridge is much higher (`peak_flops_fp16_tc`). The "Per-dtype peak in `_compute_analytical()` ridge" Deferred entry tracked this. SOLAR's graph analysis sees the actual operations and doesn't trip on dtype.
- *Single source of truth.* Today SOLAR is authoritative for run-level (via `classify_run`) and analytical was the per-workload odd one out. Collapsing both surfaces onto SOLAR removes the asymmetry and the need for two thresholds to stay calibrated against each other.

**`compute_roofline_inputs` retains exactly two callers** post-Q2, both feeding `_compute_analytical` for arithmetic-intensity / %peak math (which is *not* bottleneck classification — that surface is a Reviewer signal, not a search-routing signal):

1. `orchestrator.py::Orchestrator.run` — per-iter representative-workload profile.
2. `report.py::_resolve_workload_roofline` — Phase C re-profile, called once per selected workload.

A new caller almost always means either a third `profile_kernel` site (legitimate — extend the docstring's call-site list) or bottleneck classification accidentally re-routed away from SOLAR (regression — fix instead). The function's docstring spells this out so a future contributor doesn't drift back into the analytical-classifier pattern.

**What this means for the analytical formula.** It survives, but only on the cold/fallback path: `compute_roofline()` (in `roofline.py`) is the no-SOLAR fallback that `classify_run` uses when `RooflineResult` is `None`. That's the only remaining bottleneck-classifying caller of `classify_bottleneck`, and it's reachable only when SOLAR isn't installed at all. Production runs with SOLAR live never hit it.

**Side effects on Deferred Improvements.**

- "Per-dtype peak in `_compute_analytical()` ridge" entry's blast radius shrinks: it no longer affects `OptimizationReport.winner_per_workload_bottlenecks` (which is now SOLAR), only the `analytical.pct_peak_compute` / `pct_peak_bandwidth` numbers the Reviewer reads. The fix's value is now scoped to Reviewer signal accuracy, not search-routing or report labeling. Trigger remains the same; impact is narrower.

**Trigger for revisit.**
- A live run where Phase C report's per-workload SOLAR runs are too slow on a problem with many selected workloads (>10s aggregate). Then either cache by `(op_type, axes_signature)` or fall back to a coarse analytical first pass with SOLAR on disagreement only.
- A SOLAR upgrade that introduces a per-workload-classification fast path. Adopt directly instead of going through the full pipeline N times.

### 2026-04-29 — SOL migration retrospective: superseding notes

Pointer entry consolidating six earlier notes that the SOL-ExecBench integration (PR2, commits through `a305be1`) made factually stale. Past entries are not edited — read them in chronological order; this section is the terminal correction.

**SDK choice (2026-04-13 entry, "3 LLM agents + deterministic orchestrator")**. The line "ACTS follows Astra's pattern for the Coder (tool-using) and AccelOpt's pattern for Planner/Reviewer (single-call, no tools)" is no longer true. All three agents are tool-using: Planner emits via `submit_plan`, Reviewer via `submit_review` (plus optional `query_metric` for multi-turn metric queries, commit `6d6e62d`), Coder via `submit_kernel`. The single-call/no-tools shape was abandoned during the option-α migration (2026-04-26) for SDK strict-schema reasons already documented in the "Planner + Reviewer submit-tool migration" entry.

**Tier 1 `torch.allclose` fallback (2026-04-18 entry, "SOL-ExecBench tiered adoption")**. The clause "Falls back to `torch.allclose` when SOL isn't installed" no longer holds. `TorchComparisonPolicy.compare` now requires `sol_execbench` and fails closed when it is absent — non-SOL benchmarks are expected to provide their own `ComparisonPolicy`. Removed during the SOL-everywhere tightening so that a missing SOL install can never silently degrade the correctness gate.

**Install strategy on cu12.8 (2026-04-22 entry, "SOL integration tightening — CUDA 12.8 plan")**. The `sudo add-apt-repository ppa:deadsnakes/ppa` + `python3.12 -m venv` + manual `pip install` recipe is superseded. Canonical recipe is now `configs/venvs/3.12.md` — Python 3.12 + `uv` + editable `sol_execbench` install (`uv pip install -e ...`). The deadsnakes/sudo path still works on hosts without `uv`, but the project recipe and CI assume the `uv`-based flow.

**Tier 2 timing claim (2026-04-22 retrospective, "Plan landed as designed")**. The retrospective listed "Tier 2 (timing via `do_bench`/`time_runnable`)" as shipped. This is incorrect — Tier 2 GPU timing was *not* migrated to SOL's `do_bench`/`time_runnable`. Source still uses `_TorchCudaTimer` for end-to-end kernel timing; SOL's timing primitives are referenced only in tests/specs. Tier 2 timing is deferred (tracked in PROCESS Deferred Improvements). The other tiers in that retrospective (1, 3, 4, 5) shipped as described.

**SOLAR adapter bridge (2026-04-27 entry, "SOLAR adapter design")**. The bridge no longer synthesizes from `Problem` + representative `Workload`; it synthesizes from SOL's pydantic `Definition` + `Workload`. The `Problem` dataclass and `src/benchmark/problem.py` / `problem_loader.py` / `solution_formatter.py` were deleted in the schema migration — `Definition` is now the single benchmark-agnostic kernel IR, loaded via `src/benchmarks/sol_execbench.py::load`.

**Backward-pass schema question (2026-04-27 entry, "Backward-pass kernels deferred")**. The open question "parse `Problem.op_type` suffix vs add explicit `Problem.kind` field" is now closed by elimination: `Problem` no longer exists, and `Definition` carries no `op_type` / `kind` field. Backward kernels are identified by spec name suffix (e.g. `*_backward`). Filed-in-backlog wording for "Backward-kernel SOLAR support" should be read with this convention in mind.

### A1 PR 1 — `@triton.autotune` integration foundation (2026-05-14)

**Symptom that drove this.** Trial ACTS runs on rmsnorm / softmax / matmul consistently terminated with the Triton baseline as the search's best node — even after Phase B exhausted budget. Diagnosed in the PROCESS.md "Framework Enhancement Investigation — 'Baseline Always Wins' (2026-05-13)" section: the search machinery wasn't broken, the comparison was unfair.

**Three root causes from the 5-way recon.** (1) The baseline path runs the reference Triton kernel through `@triton.autotune`, which sweeps a config grid and bakes in the winner; ACTS-generated candidates compile as single-config kernels, so a "block_size_tuning" Planner action that nudges one knob lands against a Triton-autotuned wall it can't see. (2) One sample per plan: a candidate gets one realization, while the autotuned baseline effectively sampled `len(configs)` realizations and kept the best — so even when ACTS hits a structurally better recipe, run-to-run noise on the single sample loses. (3) Per-iter signal feeding the Planner is thin: NCU bottleneck label + timing delta, no autotune-winner trace, no "you've already tried this config family" history. Planner re-proposes near-duplicates because the observation surface doesn't tell it not to.

**PR 1 scope decision.** Foundation only: `@triton.autotune` everywhere (baseline and candidates), one burn-in call to pay the autotune cost before warmup, validator that enforces autotune presence in submitted source, and winner-recording so the Planner can see which config the search picked. Tier A items A2 (multi-sample per plan with config-space awareness) and A3 (autotune-winner-aware Planner prompt + history) and A4 (Reviewer integration with autotune-winner deltas) are deferred to their own brainstorm rounds. Tier B (search-machinery changes — config-space-aware MCTS expansion) and Tier C (eval-machinery overhaul — multi-realization timing budget) deferred entirely; revisit after Tier A data lands.

**Key design decisions.**

- **Sweep lives via `@triton.autotune` inside the Coder's source (option γ in the spec)**, not an outer harness sweep. The harness alternative (option β) would have required ACTS to own a config-grid runner and would have re-implemented what Triton already does correctly; option γ keeps the sweep where the kernel author expects it and where Triton's own caching kicks in. Cost: a single burn-in call (`seed=-1`, pre-warmup) absorbs the autotune compile + bench so the timed window measures the winner alone, not autotune's exploration. Burn-in is one extra launch per kernel, amortized across all timed launches.
- **`Kernel` dataclass replaces `num_warps` / `num_stages` / `block_size` with `autotune_configs` + `autotune_keys` + `autotune_winner`.** Breaking change to the dataclass shape — the legacy single-config fields don't compose with autotune. `from_legacy_dict` shim translates old checkpoint dicts forward (single-config → one-entry `autotune_configs` list, empty `autotune_keys`, `autotune_winner={}`) so prior runs' tree dumps still load. New runs write the new shape.
- **Coder validator enforces ≥4 configs + non-empty `key=`.** Single-config autotune defeats the whole point (no sweep happens), and an autotune block with `key=[]` would re-tune every launch (cache miss every call). The 4-config floor is heuristic — wide enough that the sweep is meaningful, narrow enough that Triton's compile budget stays bounded. Failure surfaces via the existing SDK tool-loop retry: the validator returns `SUBMIT_KERNEL_FAILED: …` and the Coder gets one more turn to fix. No new retry machinery.
- **Triton-API observation worth recording.** The autotune cache lives on the `Autotuner` object, which Triton exposes as `module.<triton_kernel_name>` — *not* on the host wrapper that `compile_kernel` resolves as `entrypoint`. The entrypoint is the user-defined Python function that calls the autotuned kernel; reading `entrypoint.cache` returns `AttributeError`. `CompilationResult.triton_autotuner` was added to surface the `Autotuner` directly so winner-extraction code reaches the right object. Cache keys are PREFIX-matched (Triton appends dtype / pointer-alignment tags after the user's `key=` values), not exact-matched — winner lookup uses `next(v for k, v in autotuner.cache.items() if k[:len(user_keys)] == user_keys)`, not `cache[user_keys]`.

**What's NOT in this PR.** Action library reshape — the tier-1 action IDs (`block_size_tuning`, `num_warps_tuning`, `num_stages_tuning`) survive as Planner-facing labels but their semantics flip from "pick one value" to "propose a config family for the autotune block." That rewrite is PR 2. Per-kernel-type recipe library (e.g., "for reductions, sweep BLOCK_M × num_warps × num_stages with this grid") is PR 3 — needs A2's multi-sample data to calibrate the grids. Live-run validation deferred per user direction; PR 1 ships with unit + integration tests but no live-GPU correctness sweep until PR 2 lands. The action library's text-references in `doc/agents.md` and the Planner prompt for `block_size_tuning` remain valid through PR 1 because tier-1 IDs are preserved; the semantic rephrase happens in PR 2.

**Pointer.** Design spec at `doc/specs/2026-05-14-a1-triton-autotune-design.md`; plan at `doc/plans/2026-05-14-a1-triton-autotune-pr1.md`. Both uncommitted per CLAUDE.md "specs and plans are never committed."


### A3 refinement — autotune-block condensation in Planner/Reviewer prompts (2026-05-14)

The original A3 framing (PROCESS.md "Framework Enhancement Investigation" Tier A) was "Planner sees parent kernel source." Brainstorm discovery: A3-as-stated is **already implemented** — the orchestrator passes `kernel_source=parent.kernel.source_code` to `planner.plan(...)`, and `PlannerAgent.build_user_prompt` renders it as the first prompt section under `## Current kernel`. The Astra recon agent that produced the original PROCESS.md row mis-read the PRD's "Planner sees the parent node's curated Reviewer prose" wording — that describes the Reviewer→Planner feedback channel, not the full prompt.

The brainstorm pivoted to a refinement that surfaced naturally: now that A1 mandates `@triton.autotune` with ≥4 configs, every parent source rendered to read-only consumers (Planner + Reviewer) carries ~7–15 lines of decorator boilerplate at the top. Signal-density refinement — not a fix for the "baseline always wins" symptom, which traces to A1 (parameter-axis gap, landed) and A2 (one-candidate-per-plan, separate).

**Key design decisions** (full spec: `doc/specs/2026-05-14-a3-autotune-condense-design.md`, uncommitted per CLAUDE.md):

- Scope = Planner + Reviewer only. Coder receives verbatim source — it must edit the decorator block (the reframed `t1_block_size_tuning` action from A1 widens the autotune sweep).
- Condensation happens **orchestrator-side, before agent calls** — the orchestrator calls `kernel.render_condensed_source(...)` at three sites: the baseline-review path, the per-iter Planner call, and the per-iter Reviewer call. The result is passed as `planner.plan(kernel_source=...)` / `reviewer.review(kernel_source=...)`. Agent signatures + `render_kernel_section` (in `llm_backend.py`) are unchanged. Cleanest seam — agents stay stateless consumers of whatever the orchestrator renders.
- Rendered shape = single-line `# autotune: BLOCK_M ∈ {64,128,256}, ...` comment replacing the decorator block, plus an optional second `# winner (representative wl): ...` line when `autotune_winner` carries the representative workload's uuid. Comment prefix is single-hash `#` (Python idiomatic; the prompt's source fence is `` ```python ``).
- Method lives on the `Kernel` dataclass — co-located with the parsed `autotune_configs / autotune_keys / autotune_winner` fields and the existing `_parse_autotune_from_source` AST helper from A1.
- All AST/parsing failures degrade silently to verbatim source.
- Plan: `doc/plans/2026-05-14-a3-autotune-condense.md`, uncommitted per CLAUDE.md.

### A1/A3 review-pass fixes (2026-05-14)

Five rounds of review fired against the A1/A3 working tree after the design-rationale entries above landed: Codex adversarial review (2 findings), Codex standard review (2 findings), `/simplify` (7 fixes from 3-agent dispatch), and two follow-up cleanups (Q9 + Q10) the user explicitly requested from the /simplify reviewer's skip-list.

**Codex adversarial review** (both fixed):
- **#1 [high] — Trust-boundary slippage in per-iter precompile.** `compile_kernel(child_kernel)` was running BEFORE the `per_iter_anti_cheat` snapshot opened, so a candidate's import-time side effects (`exec_module` runs top-level Python) became the new baseline rather than registering as drift. Fix: move the precompile inside the `with per_iter_anti_cheat(...)` block at `src/search/orchestrator.py`.
- **#2 [medium] — Target-blind autotune parser.** `_parse_autotune_from_source` walked the AST and returned the first `@triton.autotune` decorator anywhere in source. For fused kernels with an autotuned helper preceding the primary kernel, this attributed the helper's configs/keys to the named primary. Fix: thread `triton_kernel_name` through the parser (and into `Kernel.__post_init__`); regression test in `tests/test_kernel.py::test_parse_autotune_targets_named_kernel_with_autotuned_helper_first`.

**Codex standard review** (both fixed):
- **#1 [P2] — Precompile contract wasn't best-effort.** The comment said "non-fatal" but `compile_kernel(...)` could raise before returning a `CompilationResult` (read-only `.acts_cache`, disk full) and abort `Orchestrator.run()`. Fix: wrap in `try/except Exception` at both baseline + child sites; lazy-compile fallback in `benchmark_kernel` covers the degradation. Later cleaned up into the shared `_safe_precompile(kernel, *, role)` helper.
- **#2 [P2] — `_key_tuple_for` missed Definition const axes.** SOL splits axes into per-`Definition` const + per-`Workload` var. The Coder's autotune `key=` legitimately spans both (e.g. `key=["B","M","N"]` where B is var, M/N are const). The helper only resolved against `workload.axes`, so winners never recorded for the common SOL shape. Fix: add `definition=None` kwarg; fall back to `definition.axes[k].value` when the axis class-name is `AxisConst`. Three regression tests in `test_benchmark.py`.

**`/simplify` refactor pass** (7 small fixes, 250/0 sweep):
- Extracted shared `_find_autotune_decorator_call(source, triton_kernel_name)` AST helper; `_parse_autotune_from_source` and `_find_autotune_decorator_span` both delegate.
- Added cached `_autotune_span` field on `Kernel` populated in `__post_init__`, so `render_condensed_source` skips re-parse on every call.
- Extracted `_fmt_axis(name, values)` for the union-per-axis rendering in `_render_autotune_summary` (was 3-way duplicated).
- Replaced stringly-typed `getattr(def_axis, "type", "") == "const"` with `type(def_axis).__name__ == "AxisConst"` in `_key_tuple_for`.
- Extracted `_safe_precompile(kernel, *, role)` collapsing the baseline + child precompile try/except blocks into one helper.
- Trimmed verbose Codex-narration comments to one-line citations.
- Aligned `autotune_burn_in_done` event docstring in `runtime/events.py` to the actual emit payload (`workload_count`, `winner_recorded` — not the originally documented `workload_uuid` / `burn_in_us`).

**Q9 — `KernelSpec.from_dict`**: `Kernel.from_legacy_dict` (in `kernels/kernel.py`) and `_deserialize_node` (in `search/tree.py`) both rebuilt `KernelSpec` from a dict the same way. Extracted `KernelSpec.from_dict(data)` classmethod; both call sites delegate.

**Q10 — `autotune_winner` always a dict**: Field was `dict[str, dict] | None = None`, which compressed three states (never-benched, benched-no-winner, benched-with-winners) into two values — and in practice `_record_autotune_winner` only writes when winners is non-empty, so benched-no-winner collapsed to `None` anyway. Changed to `dict[str, dict] = field(default_factory=dict)`; updated all `is None` consumers (orchestrator's two `winner_recorded=` emits → `bool(...)`; `render_condensed_source` lost redundant truthiness guard; `tree._deserialize_node` legacy-checkpoint path uses `k.get("autotune_winner") or {}`; 5 test asserts flipped from `is None` to `== {}`).

**Final state**: 250 passed / 0 failed across 8 test files post-fix sweep. Working tree carries A1 PR 1 + A3 + 4 Codex-finding fixes + 7 /simplify refactors + Q9 + Q10, all uncommitted per CLAUDE.md "specs and plans live uncommitted."

**Pointer for evidence**: post-implementation, inspect one rendered Planner prompt in `events.jsonl` or `traces/*.jsonl` from the next live run and confirm `"# autotune:"` appears in place of `"@triton.autotune("`.

### Autotune cache clear at `benchmark_kernel` entry (2026-05-14)

**Bug**: per-workload winner attribution in the A1 PR uses `set(autotuner.cache) - before` to attribute a newly-baked autotune winner to the workload that triggered it. The delta math only works if `before` is empty at the start of each workload's bench loop. In practice it wasn't: the Coder's correctness pass (`check_correctness_tool`) compiles and *launches* the kernel before `benchmark_kernel` runs, because that's how it verifies the kernel produces the right answer. The first launch of a `@triton.autotune`-decorated kernel sweeps its config grid, picks a winner, and caches the winner under the call-site's key tuple. So by the time `benchmark_kernel` starts, the autotune cache may already contain an entry for the workload's shape — the per-workload `set(cache) - before` delta is empty, the winner is silently dropped, and the orchestrator-side `autotune_winner` dict stays empty even though autotune ran.

**Fix**: `benchmark_kernel` calls `autotuner.cache.clear()` once at entry, gated on `autotuner_usable` (the existing guard for "this kernel has an attached `Autotuner` we can introspect"). Defensive: an `AttributeError` on `.cache.clear()` — for any Triton version that doesn't expose the cache dict — flips `autotuner_usable = False` with a WARNING and the function continues with winner-recording disabled rather than aborting.

**Tradeoff**: the clear discards whatever autotune work the correctness pass already paid for — the kernel will re-sweep its config grid on the first warmup launch of each workload. The cost is real (one extra config-grid sweep per benched workload, ~hundreds of ms on the kernels we've measured) but it's amortized into the existing warmup loop, which was already there for thermal/clock stabilization. The warmup pass would have re-touched the autotune cache regardless; we're paying for one extra sweep, not introducing a new sweep that wouldn't have happened. The alternative — preserving the correctness-pass autotune work — would require either reading the existing cache entry and threading it forward (fragile across Triton versions, since the cache key format includes Triton-internal dtype/alignment tags) or attributing the winner to "whichever workload first compiled it" (which is correctness, not bench, and therefore the wrong attribution). Clearing is the simpler invariant: at bench entry, the cache is empty; whatever lands in it during benching belongs to the workload currently being benched.

### Resource accounting — write-side LLM usage tap + per-iter table in report (2026-05-15)

**Write-side tap over post-hoc parse over second JSONL stream.** The SDK's `generation` span carries `usage` natively at the model-call boundary. Parsing `traces/*.jsonl` post-hoc wastes O(N) reads on a file that grows multi-MB with reasoning models; a second JSONL stream duplicates the payload with new drift surface. Chosen path: extend `JSONLTraceProcessor` with an in-memory `UsageAccumulator` on the existing per-span callback path (one extra dict-write per span); snapshot via `ctx.usage_snapshot()` at end-of-run.

**Bucket at trace-close, not span-end.** The SDK can fire `on_span_end` before its enclosing `on_trace_end` — trace metadata isn't readable until the wrapping context exits. Spans buffer in `_pending[trace_id]`; `on_trace_close` resolves them against `metadata.{iter, agent}` and credits the bucket. Never-closing traces drop at snapshot — correct (don't attribute to a phantom).

**Late spans after trace_close are credited (Codex adversarial-medium).** Initial design dropped post-close spans via `_seen_trace_ids: set`; Codex flagged that SDK callbacks fire from arbitrary worker threads and a late `span_end` after `trace_end` is plausible. Replaced with `_closed_traces: dict[trace_id, (iter, agent)]`; late spans credit the stored bucket with no invocation bump (close already credited it).

**Prompt/completion fallback (Codex native-P2).** Usage surfaces under two key shapes: Responses-API (`input_tokens` / `output_tokens`, what the openai-agents SDK documents) and Chat-Completions (`prompt_tokens` / `completion_tokens`, what the fixture at `tests/test_trace_processor.py:53` uses). `_pick_token_count` checks `primary in usage` (not `usage.get(primary) is None`) so a present-but-zero new name still wins over a non-zero legacy name.

**Cell folds sub-buckets; footer surfaces them.** Each cell renders `<calls> (<turns>) / <input>→<output>`. `cached_input_tokens` and `reasoning_output_tokens` stay folded into parent totals so 80-col terminals can read the table, then surface as run-total footer lines (`of which cached input: X (Y%)`) only when non-zero. Non-thinking runs see a cleaner table.

**Tap-never-raises discipline (Codex adversarial-high).** Module discipline is "never let trace I/O kill a run." Initial tap sat outside the existing try/except; wrapped in `_safe_usage_tap(self, tap_callable)` matching the `_write` pattern. `_parse_span_usage` also hardened with `isinstance(usage, dict)` guards + a `_safe_int` helper — belt-and-suspenders if the tap is ever called outside the guard.

**`AgentLabel(str, Enum)` over `enum.StrEnum`.** Tier-1 venv is Python 3.10; `StrEnum` is 3.11+. The pre-3.11 idiom gives str-subclass semantics (`AgentLabel.CODER == "coder"`) on both venvs. Gotcha: on 3.10, `str(AgentLabel.CODER_TRANSLATE)` returns `"AgentLabel.CODER_TRANSLATE"`, not `"coder-translate"` — `_coerce_agent_label` in `sdk_trace.py` reads `.value` for enum members. Pinned by `test_sdk_trace.py::test_agent_label_coder_translate_stringifies_to_value`.

**`trace_span` is its own module, not in `agents/trace_processor.py`.** The shim constructs SDK traces (producer); the processor handles them (consumer). Splitting keeps `trace_processor.py` focused on persistence + accounting and lets `search/orchestrator.py` + `benchmark/baseline_generator.py` import a small shared helper without dragging in the processor's lifecycle.

### Failure-node collapse — bad-streak tree inflation (2026-05-18)

**Trigger.** `runs/run_20260518T035408_454910Z` produced `tree/index.json` with **116 nodes** on a 30-iter run despite `max_depth=30`. Iters 3–30 all-K-failed (every one of K=4 Coder candidates raised `ImplementationError`), and each failure was persisted as its own dead-end node per the 2026-05-17 failure-node retention design. Worst case under retention: `K × max_depth = 120` nodes. `tree.txt` becomes unreadable; per-node `tree/node_<id>/` directories proliferate; `SearchTree` carries K× the necessary `TreeNode` objects.

**The render side was already collapsed.** `render_siblings` dedups failures across separate child nodes on the `(action, params, reason)` triple with `×N` collapse and `failure_cap=8`. The planner-facing signal-quality issue was already handled — the bloat is purely in the on-disk tree representation, the data structure, and human-readable visualizations. The cost of inaction was visible-to-humans only.

**Decision.** Collapse each iter's K failed K-way Coder candidates into **one failure-summary tree node** attached to the same parent, with a new `failure_details: list[FailureDetail]` field replacing the legacy single-string `failure_reason`. The summary node carries `kernel=None` (per-candidate kernels live on disk under `tree/node_<id>/cand_<idx>/kernel.py`, not on the TreeNode). Mixed-outcome iters (winner + F failures) get 1 winner + 1 failure-summary sibling. All-K-fail iters get 1 failure-summary sibling. Worst case per iter: **2 nodes** (vs. previous K). For the trigger run, total tree drops from 116 → ~32 nodes.

**`add_failure_summary` replaces `add_failure_child`.** The legacy per-candidate API is removed entirely. `failure_reason: str | None` is removed from `TreeNode`; the deserializer synthesizes a one-element `failure_details` list on load from any legacy `failure_reason` field — `has_kernel_source=False` unconditionally on legacy synthesis because the legacy on-disk layout has `kernel.py` at the flat path `tree/node_<id>/kernel.py`, not at `cand_0/kernel.py` (setting `True` would mislead postmortem readers).

**Orchestrator: iter-level accumulator pattern.** `_persist_failure_node` (per-candidate immediate persistence) is replaced by `_accumulate_iter_failure` (per-candidate append to an iter-level list + per-candidate `coder_failed` event) plus `_persist_iter_failure_summary` (end-of-iter: attach one summary node + dump per-candidate artifacts + emit `failure_summary_added`). The 5 K-way failure sites (ImplementationError, EntrypointBinding, BenchmarkError, recoverable CUDA sticky-state, partial bench) all push into the accumulator instead of mutating the tree immediately; persistence fires once per iter at the four iter-exit points (all-K-fail skip, reward-hack abort, bench-all-fail skip, profile-gauntlet skip, winner-commit). Profile-layer failures (rank-and-profile-fallback) stay event-only — unchanged "downstream-of-truth, don't dilute the FAILED block" rationale.

**`consecutive_agent_failures` invariant preserved.** Still iter-level, bumped at most once per iter (not K times). The refactor moved the persistence call but not the bump call. Regression test `test_kway_all_fail_bumps_quarantine_counter_once` pins this.

**On-disk layout: `cand_<idx>/` subdirs for postmortem.** `tree/node_<id>/meta.json` (summary fields + `failure_details` list); `tree/node_<id>/cand_<i>/{kernel.py, meta.json}` per failed candidate. `kernel.py` is omitted when `failure_details[i].has_kernel_source=False` (turn-exhaust paths). `dump_failure_summary_node` is new; `dump_node` keeps its existing `failure_detail` parameter for non-K-way `_kill_branch` paths (CUDA_ERROR, REVIEWER_JUDGED, etc.) — the spec was too aggressive initially, only K-way migrates.

**Breaking event-log change.** `failure_node_added{iter, candidate_idx, node_id, parent_id, action, params, reason}` → `failure_summary_added{iter, node_id, parent_id, action, params, candidate_count}`. One event per iter (not K), no `candidate_idx` (multi-valued — available in per-candidate `cand_<idx>/meta.json`), no `reason` (multi-valued — same). The per-candidate `coder_failed{iter, candidate_idx, reason}` event is **unchanged** — K events per iter, same payload. Any external dashboard / scripts parsing `events.jsonl` for the old name break silently.

**Render contract: byte-identical for the common case.** `render_siblings`' for-child loop flattens each summary's `failure_details` into the existing `failure_entries` accumulator. The downstream dedup-on-`(action, params, reason)` + `×N` + `failure_cap` logic runs identically. K candidates failing with the same reason still collapse to a single `FAILED ×K` line; K candidates with distinct reasons render K lines. The output is identical to the pre-collapse shape modulo cosmetic ties on `(iter_no, child_id)` ordering inside one summary.

**Pointer.** Spec: `doc/specs/2026-05-18-failure-node-collapse-design.md` (uncommitted scratch). Plan: `doc/plans/2026-05-18-failure-node-collapse-plan.md`. Trigger run: `runs/run_20260518T035408_454910Z` (116 nodes, 30 iters; ~32 nodes under collapse).

### Bench-subprocess isolation refactor (Scope B) — design rationale (2026-05-20)

**Trigger.** `runs/sweep_l1_048/regime_03_default/rep_0/run_20260520T011532_929220Z` — every K-way candidate of iters 1–5 (and counting) failed with `AcceleratorError: CUDA error: operation not supported on global/shared address space` (`cudaErrorInvalidAddressSpace`) raised inside Triton's autotune burn-in. **Smoking gun.** `tree/node_1/cand_1/kernel.py`, `tree/node_1/cand_3/kernel.py`, `tree/node_2/cand_0/kernel.py` are sha256-identical to `tree/node_0/kernel.py` — the baseline that ran cleanly through `autotune_burn_in_done{workload_count:3, winner_count:3}` at iter 0. A kernel that succeeded at 09:22 cannot fail because of its source at 09:35 — the parent's CUDA context is poisoned and every subsequent launch in the process inherits the sticky error.

**Why current recovery doesn't engage.** `_CUDA_STICKY_PATTERNS` at `src/search/orchestrator.py:37-45` lists 7 substrings (`illegal memory access`, `device-side assert`, `unspecified launch failure`, `misaligned address`, `out of memory`, `cublas`, `cudnn`); `"operation not supported on global/shared address space"` matches none. The bench layer wraps the per-workload `AcceleratorError` into `BenchmarkError("0/N workloads survived ...")`; the orchestrator catches that at `orchestrator.py:1058` (the `except BenchmarkError` branch), records the failure, `continue`s. The `except RuntimeError` + `_CUDA_STICKY_PATTERNS` branch at line 1065 — where `torch.cuda.synchronize()` would fire and `CUDAContextPoisoned` could escalate — is never reached. The run burns through `max_depth` iters of LLM time on a dead context, with `consecutive_agent_failures` never tripping because each iter cleanly produces a failure-summary node and "advances."

**Why pattern-list expansion alone is the wrong fix.** Even with the substring added to `_CUDA_STICKY_PATTERNS`, `torch.cuda.synchronize()` does not clear context-level sticky errors — those persist until the context is destroyed (process exit). Pattern expansion only buys *faster abort* by escalating to `CUDAContextPoisoned`. Useful as belt-and-suspenders (saves ~80 min wasted iters → ~15 min single failed iter for this run), but it doesn't let the search continue past the error. That ceiling motivates the subprocess refactor.

**Three scopes considered.**

- **Scope A — per-candidate bench subprocess.** Spawns at every `benchmark_kernel` call inside Coder's tool loop. Max isolation; intra-iter poisoning leaves K-1 siblings alive. Cost: ~160 spawns × ~4 s ≈ **10 min/run** pure startup, plus per-spawn Triton JIT cache cold-start.
- **Scope B — per-iter bench subprocess.** Spawns once at the start of an iter's K-way bench phase after Coder has returned K kernel sources; subprocess does compile + autotune-burn-in + bench for all K serially (or via its own `asyncio.gather`), returns K results, exits. Cost: ~30 iters × ~4 s ≈ **2 min/run**. Blast radius drops from "whole run wasted" to "one iter wasted." Caveat: K candidates within one iter still share a context; if one poisons, the iter's other K-1 die together.
- **Scope C — per-run subprocess (status quo).** Zero startup; max blast radius. Today's design.

**Decision: Scope B.** Cost/benefit math: ~2 min overhead trades for ~80 min wasted iters per poisoned run; one IPC boundary (K kernel paths in, K results out) instead of K per iter; intra-iter shared-context poisoning is a corner case worth deferring to Scope A only if postmortem evidence shows it recurring. Adopt Scope B as default; promote to Scope A iff K-1 sibling survival inside a poisoned iter becomes a frequent win.

**Costs that survive scope choice (apply to A and B both).** Input-tensor IPC for ~150 MB bfloat16 workloads — pick "regenerate in child from RNG seeds" since L1/048-class workloads are `type=random` with deterministic seeds; mmap and CUDA IPC handles are heavier alternatives reserved for non-random inputs. `events.jsonl` plumbing across the process boundary (child writes its own chunk, parent concatenates — simpler than stdout-streaming-and-demux). Reward-hack channel-A detector relocates into the subprocess (taint-died-with-child changes the existing `iter aborts` rationale). Autotune-winner extraction from Triton's in-memory `Autotuner.cache` must serialize across the result channel since the in-memory cache vanishes on subprocess exit. `asyncio` wrapping at the Coder tool boundary — `asyncio.create_subprocess_exec` instead of synchronous `benchmark_kernel`. Debuggability degrades from full parent-side tracebacks to whatever the child wrote to stderr before dying.

**Open questions reserved for the brainstorm round.** IPC payload shape (JSON vs pickle for `BenchmarkResult` + autotune-winner dict + workload_errors + reward-hack channel-A flag + trace deltas). Whether the bench-isolation subprocess is spawned by the orchestrator post-Coder (Scope B as scoped above) or by Coder's tool layer mid-loop (changes Coder iteration semantics — currently it retries on its own benchmark errors). Failure-mode vocabulary for child exit codes (clean / child-raised / OOM-killed / segfaulted) vs the existing `BenchmarkError` / `CUDAContextPoisoned` taxonomy. Disposition of `consecutive_cuda_errors` — parent-side and counting `synchronize()` failures, which the parent no longer sees once bench moves to a child (dead code vs repurposed as child-died-N-times-in-a-row).

**Upstream prevention not addressed by this refactor.** The root cause of the *initial* context poisoning (between `baseline_ready` at 09:22:14 and iter-1 bench-error at 09:35:43) is not directly evidenced in the logs; the leading suspect is NCU's first profile call during Reviewer's iter-0 baseline review (NCU + cache-flushing on Ada has been the historical suspect for this exact error class). Subprocess isolation contains the blast radius but doesn't fix the source — NCU-isolation investigation is a separate workstream.

**Relation to the deferred SOL Tier 8.** This is the narrower, bench-only specialization of the broader "subprocess-isolated evaluation (SOL Tier 8)" item in PROCESS → Trigger-gated tech-debt (deferred 2026-04-27). That entry's Trigger C — *"live run with frequent kernel crashes (>1% of evals, or any case requiring manual intervention)"* — has fired: this sweep run is the manual-intervention case. Scope B promotes the bench-eval slice of Tier 8 to Active queue while leaving correctness / profiler eval-subprocess work in the broader tech-debt bucket.

**Pointer.** Spec + plan to be written under `doc/specs/` and `doc/plans/` per CLAUDE.md workflow step 2 (uncommitted scratch). Trigger run: `runs/sweep_l1_048/regime_03_default/rep_0/run_20260520T011532_929220Z`.

### `compile_kernel` import-cache anti-cheat residuals — investigation + deferral (2026-05-24)

**Trigger.** Question raised during Scope B+ spec review: does `compile_kernel`'s `sys.modules` cache (`src/kernels/compiler.py:73-82`) undermine `per_iter_anti_cheat`'s snapshot-vs-recheck pattern? Investigation traced the mechanism and the Scope B+ refactor's interaction with it.

**Mechanism (pre-Scope-B+).** `compile_kernel` keys its in-process cache on `acts_compiled_{spec.name}_{source_hash[:12]}`. Same source → same key → cache hit → `exec_module` is skipped. The Coder's `compile_kernel_tool` (`src/agents/coder.py:274-286`) and `check_correctness_tool` (`:336-338`) call `compile_kernel` *during the Coder turn*, before the orchestrator enters `per_iter_anti_cheat`. Sequence for a cheating candidate: (1) Coder tool fires `exec_module`, import-time monkey-patches land in the parent's `torch.cuda` namespace, module is cached; (2) orchestrator enters `per_iter_anti_cheat`, `snapshot_critical_functions(vars(torch.cuda.Event), critical_names)` captures the *already-patched* state; (3) `_safe_precompile` inside the with-block cache-hits → no re-`exec_module` → no new drift to detect; (4) `check_eval_integrity` at exit compares polluted-snapshot to still-polluted-namespace → passes tautologically. `check_monkey_patch` still catches patches on `Event.elapsed_time` (uses `_ELAPSED_TIME_ADDR` snapshotted at sol_execbench module load) — but the other four `critical_names` (`synchronize`, `wait`, `record`, `query`) rely entirely on `check_eval_integrity`, which is the layer the cache defeats.

**What Scope B+ fixes structurally.** `per_iter_anti_cheat` now runs in the *child* process. Child starts cold: empty `sys.modules`, fresh `torch.cuda` namespace. Snapshot is taken before any candidate code touches *that* process. `_safe_precompile` inside the with-block cache-misses (child's `sys.modules` is empty) → `exec_module` fires inside the with-block → patches land between snapshot and recheck → `check_eval_integrity` catches. The cache itself is unchanged; subprocess just relocates the snapshot to a process where the cache is empty.

**What Scope B+ does not fix.** Two residuals survive, recorded in Scope B+ decision #9 + §13 and in PROCESS → Trigger-gated tech-debt:

- **Within-iter K-sibling pollution in the child.** Cand 0's `exec_module` patches land inside its own `per_iter_anti_cheat` block. If the patch is on a name in `critical_names`, `check_eval_integrity` raises → channel-A immediate-abort (Scope B+ decision #5) kills the iter before cand 1..K-1 run. If the patch is on *anything else* (an Event attribute outside `critical_names`, a method on `torch.cuda` directly, `torch.compile`, etc.), cand 0 passes "cleanly" and the patch persists in the child's namespace into cand 1's snapshot. The §4 `sys.modules.pop` between candidates does not help here — the patch lives in the `torch.cuda` namespace, not in `sys.modules`. Real fix is explicit namespace restoration (Scope B+ §13 "Channel-A state restoration in child") or per-candidate subprocess (JOURNAL "Scope A").

- **Parent-side `compile_kernel` cache drift.** Coder tools (out of scope per Scope B+ §2) still run in the parent. Each Coder turn populates parent's `sys.modules["acts_compiled_*"]` and lands whatever patches the candidate's module body executes. Across a 20-iter run, the parent's namespace progressively drifts. The bench path is unaffected (child is fresh), but **Phase C winner re-profile** (parent-side per §6.3) eventually runs in this drifted parent. If the winner's source was previously compiled in a Coder turn, the cache holds the module and any patches are still live. Real fix is Scope B+ §13 "Option D" (Coder tools into a subprocess) or moving Phase C re-profile into a subprocess.

**Why defer.** Threat-model gap, not a mechanism gap. Every observed Coder output (Claude, GPT-5) is best-effort optimization, not adversarial — no run in `runs/sweep_l1_048/` has tripped `RewardHackDetected`. Scope B+ was justified by an observed CUDA-poisoning failure; widening its scope to "and also seal the cache" adds review surface to a refactor with its own load-bearing case. Within Scope B+, the cheap defensive `sys.modules.pop` between candidates (decision #10, §4 architecture line) is the only addition this investigation produced — it costs ~5 lines and pins the invariant that any future in-child re-entrant snapshot stays safe, but does not close either residual. Real fixes (within-iter namespace restoration, parent-side Coder-tool subprocess) wait for evidence that the residuals materialize.

**Decision.** Land Scope B+ as specified, including the §4 `sys.modules.pop` defensive guard. Defer within-iter restoration and parent-side cache fixes to triggered tech-debt items.

**Pointer.** Spec amendment: decision #9 (split cross-iter / within-iter / parent-side), decision #10 (defensive pop), §4 architecture (pop in pseudocode), §13 future scope (residuals tied to existing items). PROCESS.md tech-debt: amended "`per_iter_anti_cheat` state restoration" entry under A2 follow-ups.

### HW-spec injection — Phase A broadcast + Phase B ptxas-truth SMEM check (2026-05-24)

**Phase B pivoted from submit-time static formula to compile-time ptxas-truth.** PROCESS.md active-queue item #3 originally sketched a static SMEM pre-check at `submit_kernel`: parse `BLOCK_M / BLOCK_N / BLOCK_K / num_stages / dtype` out of the candidate source and reject any config where `2 × num_stages × (BLOCK_M + BLOCK_N) × BLOCK_K × dtype_bytes` exceeds `shared_mem_per_block_bytes`. Brainstorm tore the formula approach down on four grounds: (i) **parser brittleness** — multi-dtype kernels, dtype inferred from input pointers, `.to()` chains mid-kernel, accumulator dtypes diverging from input dtypes (fp16 in / fp32 acc) all mean the dtype the formula needs isn't a stable AST surface; (ii) **formula imprecision** — ptxas fuses loads, hoists allocations, shares buffers across pipeline stages, so a static `2 × stages × tile_bytes` formula simultaneously over-counts (no fusion model) and under-counts (no SASS-level scratch); (iii) **shape dispatch** — matmul / reduction / elementwise / attention each want a different formula, and a Triton kernel's shape isn't a contract anywhere; (iv) **`BLOCK_M/N/K` is convention, not enforcement** — a kernel that names its tile constants `TILE_M / TILE_N` defeats the parser silently. Better signal exists one layer down: `CompiledKernel.metadata.shared` is the actual ptxas-allocated SMEM in bytes, populated after Triton finishes compiling. For `@triton.autotune`-decorated kernels, we already call `kernel.warmup(*sample_args, grid=(1,))` to materialize each Config's `CompiledKernel`; reading `compiled.metadata.shared` per Config tells us truth instead of formula. No parser, no shape dispatch, catches every kernel correctly. Final design: SMEM check lives inside `compile_kernel_tool` after `compile_kernel()` succeeds, delegated to `check_autotune_smem_budget` in a new `src/eval/smem_check.py`; overflow returns a structured `Compile FAILED: shared-memory budget exceeded` string listing the offending Configs + ptxas footprint in bytes + the cap. Rides the existing Coder in-loop tool-error retry path; `error_log` captures the rejection across attempts so the Coder can learn from cross-attempt history.

**Pre-existing anti-cheat condition surfaced (NOT introduced here, but worth pinning).** PROCESS.md §35 documents "`_safe_precompile` runs *inside* `per_iter_anti_cheat` so candidate import-time side effects register as drift." But `compile_kernel` is hash-keyed cached (per `kernels/compiler.py` — file-backed importlib load + hash-keyed cache). The Coder's `compile_kernel_tool` is the **first** site that compiles a candidate kernel, and it fires *before* the orchestrator's `per_iter_anti_cheat` window opens. By the time `_safe_precompile` runs inside that window, the cache hits → no fresh `exec_module` → no import-time side-effect detection. Import-time side effects already escape today; that ship sailed when `compile_kernel_tool` was wired. Adding `kernel.warmup()` calls inside the Coder tool loop (for the ptxas-truth read) doesn't change the surface — warmup is Triton-internal (ptxas + SASS build + cache write), no torch monkey-patch hook. **Follow-up filed as trigger-gated tech-debt**: widen `per_iter_anti_cheat` to wrap the Coder tool loop, or detect import-time side effects at a different point. Revisit when a real run shows an import-time patch was missed.

**Routing chose broadcast over need-only or silent.** Upstream survey (KernelAgent Meta paper, AccelOpt, cuda-optimized-skill) shows three established patterns: **broadcast** (one HW-spec block in every agent's system prompt), **need-only** (Coder gets it, Planner/Reviewer don't), **silent** (Coder reads it programmatically, no agent sees text). Chose broadcast through the shared `render_run_context()` helper because (a) it matches the existing Pattern F once-per-run header — same seam, same lifecycle; (b) single source of truth — one render function, one HW-spec format string, three readers; (c) all three roles benefit from the cap visibility: Coder for autotune Config sizing (don't propose tiles that overflow), Planner for `plan.params` BLOCK suggestions (don't suggest `BLOCK_M=256 BLOCK_K=128` on a 48 KB cap card), Reviewer for `pct_peak` reasoning context (knowing SMEM cap is one factor in occupancy interpretation). The Coder is the K=4 cost amplifier — its prompt fires K times per iter — but the HW-spec block is only ~6 lines, which is well below the noise floor relative to autotune-condensed parent source + plan + error_log.

**Field additions constrained to two SMEM fields.** `HardwareSpec` gained `shared_mem_per_block_bytes` and `shared_mem_per_multiprocessor_bytes`. Considered and rejected: `sm_count` (no immediate consumer — autokernel uses it for `pct_peak` math, but we compute pct_peak via roofline already with bandwidth + FLOPs, not by counting SMs), `regs_per_block` and `regs_per_thread_max` (not statically predictable for Triton — ptxas decides register allocation per kernel post-compile, and no upstream repo we surveyed does proactive register-pressure checking; you can read it from `CompiledKernel.metadata.num_regs` post-compile but you can't propose a check from it without baking in a register-pressure model, which is its own design). Adding fields with no consumer is governance debt; deferred until a concrete check needs them.

**Phase B production-shape limitation surfaced during Tier-2 GPU validation: `kernel.warmup()` signature mismatch on host-wrapper kernels.** `check_autotune_smem_budget` calls `autotuner.fn.warmup(*sample_args, **cfg.kwargs, grid=(1,))` to compile each Config without launching, then reads `CompiledKernel.metadata.shared` for ptxas-truth SMEM. The Coder's `_run_tool_agent` synthesizes `sample_args = input_generators[0](0)` — i.e. the **user-facing inputs** the host wrapper accepts (typically `(a, b)` for matmul). But the JIT device function signature includes everything the host wrapper internally derives — output buffer `c`, dimensions `M, N, K`, strides — none of which are in `sample_args`. Result: `warmup()` raises for every Config, the helper catches per-Config and emits `smem_check_skipped(reason='warmup_failed', config_idx=i)`, the violations list comes back empty, and the compile tool returns its success string. The proactive in-loop rejection that motivated Phase B doesn't fire in the actual matmul case. Production-realistic Tier-2 test `test_check_autotune_smem_budget_skips_on_signature_mismatch` pins this fail-open behavior so the contract is at least honest. The `autotune_exclude` reactive path (landed 2026-05-18) remains the working safety net: a Config that overflows still crashes during the Coder's `check_correctness_tool` (which calls the host wrapper, which triggers Triton autotune burn-in, which actually launches each Config), the iter dies, the failed-sibling block surfaces the Config pattern to the Planner, and the next iter excludes it. **Follow-up filed as trigger-gated tech-debt: a real in-loop Phase B needs either** (a) `sample_args` derived from the host wrapper's signature (e.g., by introspecting the wrapper's source AST to find the JIT call site and synthesizing scratch outputs / dims), (b) a Coder-declared JIT signature (similar to how `triton_kernel_name` is declared today via `KernelCodeOutput`), or (c) moving Phase B to the orchestrator's `_safe_precompile` after `check_correctness_tool` has driven autotune burn-in and populated the cache for inspection. Each option has trade-offs (parse-fragility / Coder-prompt-surface-growth / no-in-loop-LLM-feedback). Revisit when a real run shows the autotune_exclude reactive path is insufficient. **Follow-up:** resolved 2026-05-25 — see "Phase B recorder-patch redesign" entry below; option (a)-flavored design (drive the host wrapper itself with a no-launch recorder patched onto `autotuner.run`) closes the limitation without parsing wrapper source.

**Also during Tier-2 validation: Triton 3.6 cache-attribute migration.** Initial test failure surfaced that `JITFunction.cache` no longer exists in Triton 3.6.0 — it was replaced by `JITFunction.device_caches`, a `defaultdict[int → tuple]` where `entry[0]` is the actual `dict[signature_str → CompiledKernel]` (entries 1–4 hold signature-canonicalization maps + GPUTarget + backend + binder). Fixed `_latest_cache_entry` to walk `device_caches[dev][0]` first, with fallback to the legacy `cache[dev]` shape for older Triton compatibility. The fix is duck-typed via `getattr`, so both shapes work without a version sniff. `CompiledKernel.metadata.shared` (and the legacy `.shared` peer) remained stable across the same migration; the Tier-1 mocked test for the legacy attribute continues to pass.

**Pointer.** Spec: `doc/specs/2026-05-24-coding-hw-spec-design.md`. Plan: `doc/plans/2026-05-24-coding-hw-spec-plan.md`. Both uncommitted per CLAUDE.md "specs and plans are never committed."

### Phase B recorder-patch redesign — fix for production warmup-signature limitation (2026-05-25)

**Fix in one sentence.** New helper `_capture_jit_args_via_host_wrapper` in `src/eval/smem_check.py` drives `host_wrapper_fn(*sample_args)` once with `autotuner.run` instance-patched to a no-launch recorder, captures the args the host wrapper passes to the autotuner on the first internal kernel invocation, and restores via `del autotuner.run`; top-level `check_autotune_smem_budget` gains a `host_wrapper_fn` positional and now warmups each Config with the captured args instead of with the user-facing `sample_args`, resolving the production-shape limitation flagged in the 2026-05-24 JOURNAL entry.

**Why it works.** Triton 3.6.0 ground truth: `Autotuner.__getitem__` is inherited from `KernelInterface` and returns a lambda that does late-bound `self.run(...)` lookup at call time, so assigning `autotuner.run = recording_fn` cleanly shadows the class-method descriptor via the instance dict — the lambda resolves `self.run` through normal attribute lookup, finds the instance attribute first, and dispatches to the recorder instead of the inherited method. The host wrapper tolerates `None` return from the recorder because it allocates the output buffer `c` BEFORE calling the kernel and returns `c` regardless; the recorder's job is to swallow the call (capture `(args, kwargs)`, return `None`, no launch) without breaking the wrapper's control flow. Restore is via `del autotuner.run` inside a `finally` block — deletes the instance attribute, revealing the class method again — so even an exception in the wrapper's pre-launch host code doesn't leave the autotuner monkey-patched into subsequent compiles. Verified end-to-end on a host-wrapper matmul: `test_check_autotune_smem_budget_rejects_host_wrapper_matmul_overcommit` rejects a `BLOCK_M=256 / BLOCK_N=256 / BLOCK_K=64 / num_stages=4` fp32 Config on Ada's 99 KB per-block cap, with the rejection surfacing through the same `Compile FAILED: shared-memory budget exceeded` string the 2026-05-24 design specified.

**Two new skip reasons added to `smem_check_skipped`.** `host_wrapper_failed` fires when driving the wrapper raises — acceptable degradation because the same wrapper will fail in `check_correctness_tool` immediately afterward with a clearer signal, so the SMEM check declining to act on a wrapper-broken candidate doesn't hide the bug, it just defers the surfacing by one tool step. `recorder_no_capture` fires when the wrapper completes without invoking the kernel (defensive; not expected in practice — a Triton host wrapper that doesn't call its autotuned kernel is a malformed wrapper, but the helper shouldn't crash on it). Total skip reasons now stand at six: the prior `no_autotuner`, `no_hardware_cap`, `sample_args_missing`, `warmup_failed` from the 2026-05-24 landing, plus these two new ones.

**Remaining limitation.** Multi-call host wrappers — split-K matmul that launches the kernel once per K-slice, multi-stage fusions that call the kernel separately for forward / backward passes within one wrapper invocation — the recorder captures only the first internal kernel invocation. Acceptable because the autotune Config set is shared across all calls (one `@triton.autotune` block, one Config list), and SMEM allocation is per-launch-invariant within that Config set: ptxas decides the per-Config SMEM footprint at compile time, not at launch time, so reading metadata after a single warmup is sufficient evidence regardless of how many times the kernel will eventually be launched. Revisit if a real run shows a kernel with per-call-site-divergent SMEM that the single-capture heuristic misses.

**Triton version pin.** `test_recorder_patch_takes_effect_on_real_autotuner` (Tier-2 GPU) verifies the instance-attribute monkey-patch contract on the live Triton install — constructs an actual `triton.autotune`-decorated kernel, patches `autotuner.run` with a sentinel recorder, drives the wrapper, asserts the recorder was hit (proving `__getitem__` resolved through the instance dict) and that `del autotuner.run` restores the class method. If a future Triton release changes `Autotuner.__getitem__` or refactors `KernelInterface` to bypass late-bound `self.run` lookup — e.g. caches the bound method at decorator time, or dispatches through a C-extension fast path that doesn't honor instance attributes — this test fails BEFORE the SMEM check silently no-ops in production. Cheap insurance against a Triton-internal refactor that would otherwise quietly regress the helper to the 2026-05-24 fail-open state.

**Codex round-3 fixes (2026-05-25, same-day amendment).** Adversarial review caught two real issues with the recorder-patch redesign on first landing — both fixed inline before the round closed. (1) The recorder captured only positional args from `autotuner.run(*args, **kwargs)` and dropped the kwargs entirely. Real host wrappers frequently pass shape/stride parameters by keyword (`matmul_kernel[grid](a, b, c, M=M, K=K, stride_am=a.stride(0), ...)`); without those kwargs, the per-Config `warmup()` raised on signature mismatch, the helper emitted `warmup_failed` for every Config, the violations list came back empty, and `compile_kernel_tool` reported success — regenerating the exact fail-open mode the redesign was supposed to close, just at the positional-vs-keyword boundary instead of the user-facing-vs-JIT-signature boundary. Fixed by extending `_capture_jit_args_via_host_wrapper` to return `(args, kwargs)`, stripping the framework-injected `grid` / `warmup` kwargs at replay, merging recorded kwargs with `cfg.kwargs` (cfg wins on overlapping constexpr keys), and threading the merged dict into `autotuner.fn.warmup`. Regression test `test_check_autotune_smem_budget_replays_recorded_kwargs` exercises a matmul-shaped host wrapper with keyword-only shape/stride launch args. (2) The baseline-generation path (`CoderAgent.translate`) used `src/prompts/coder/translate.md`, which Task 7's doc sweep never touched. That prompt's "Autotune (A1)" section still told the Coder to include oversized SMEM configs and let Triton's compile-failure pruning drop them at runtime — directly contradicting the new `compile_kernel_tool` rejection policy. Worse, `baseline_generator.generate_triton_baseline` calls `translate()` without a `bottleneck` (the once-per-run classification hasn't fired yet at baseline time), so `build_translate_prompt`'s `if bottleneck is not None:` gate skipped `render_run_context` entirely and the Coder drafting the baseline kernel saw neither the hw budget block nor the SMEM-cap hard rule on the very first attempt — with a 3-attempt retry budget that the stale-prompt path could easily burn out on. Fixed by (a) updating `translate.md` with the same `shared_mem_per_block_bytes` hard rule from `system.md` and explicitly retiring the "include oversized configs, let Triton prune" guidance as a stale pattern, (b) relaxing `render_run_context` to accept `bottleneck=None` and still emit the hw block (renders a "not yet classified (baseline generation)" sentinel line in place of the bottleneck value), and (c) removing `build_translate_prompt`'s `if bottleneck is not None:` gate so the run-context section now renders whenever either bottleneck or hardware is supplied. Regression tests `test_render_run_context_renders_hw_when_bottleneck_none` and `test_render_run_context_returns_empty_when_both_none` pin the new contract.

**Codex round-4 fixes (2026-05-25, second same-day amendment).** Standard review caught a fourth-deepest-layer fail-open and a telemetry gap. (1) **`triton_kernel_name` is empty at compile_kernel_tool time, so the autotuner never resolves.** The Coder's tool flow declares `triton_kernel_name` only at `submit_kernel` (the LLM passes it as a tool arg there), but `compile_kernel_tool` runs first. `_make_compile_tool` constructed `Kernel(spec=..., source_code=...)` without setting the name, so `compile_kernel`'s `_resolve_triton_autotuner` short-circuited on `kernel.triton_kernel_name == ""` and returned None — meaning every production iter took the `no_autotuner` skip branch and the SMEM check has never actually fired since landing. This is the fourth fail-open layer in three review rounds (first the static formula, then the warmup-signature mismatch, then the kwargs drop, now the missing kernel-name). The reason the previous rounds' tests didn't catch it: Tier-1 mocks stubbed `compile_kernel` to return a result with a pre-resolved `triton_autotuner=fake_object`, and the Tier-2 GPU test called `check_autotune_smem_budget` directly on the `@triton.autotune`-decorated kernel object — neither traversed the `compile_kernel → kernel.triton_kernel_name → resolve autotuner` chain that production code traverses. Fixed by auto-deriving the name in `_make_compile_tool` via the existing `triton_kernel_names_in(source)` helper: exactly-one `@triton.jit def` in source → use that name; multiple-or-zero → leave empty and let the `no_autotuner` skip path engage cleanly. The Coder's later `submit_kernel` validator still cross-checks the LLM-declared name against source independently, so auto-derivation here doesn't weaken that contract. (2) **Telemetry gap on `no_hardware_cap`.** When `hardware is None` or `shared_mem_per_block_bytes == 0`, the outer guard silently bypassed without emitting `smem_check_skipped(reason='no_hardware_cap')`, so `events.jsonl` couldn't distinguish "check ran and passed" from "check was bypassed because no cap configured" — making it hard to tell whether the new check actually ran in any given run. Fixed by emitting the documented event in the no-cap branch. (3) **New Tier-2 GPU integration test that traverses the full production path.** `test_compile_kernel_tool_rejects_smem_overflow_end_to_end` builds a real KernelSpec + host-wrapper matmul source with an overcommitted Config, invokes `_make_compile_tool(...)(source)`, and asserts the tool returns `Compile FAILED: shared-memory budget exceeded`. This is the test the previous three review rounds were missing — it exercises `Kernel(source) → compile_kernel → _resolve_triton_autotuner → check_autotune_smem_budget → rejection` end-to-end so a future regression in autotuner resolution surfaces here instead of via empty `events.jsonl` entries in a real run. The lesson is the integration boundary itself: each round's fix was correct, but the unit tests stubbed past the very layer where the next round's bug lived. **Meta-observation**: Phase B's contract surface between the Coder layer and the SMEM helper has been wrong at four successive depths. Each fix is small and correct; the underlying complexity is "Phase B needs the host wrapper, the autotuner, the captured args + kwargs, the kernel name, the cap, and the right Triton-version metadata accessor — and any one of them being absent at the wrong moment makes the check silently no-op." With the integration test in place, future regressions in any of these joints now surface at test time rather than in `events.jsonl`.

**Pointer.** Spec: `doc/specs/2026-05-24-coding-hw-spec-design.md` §6 (now contains the recorder-patch redesign). Plan: `doc/plans/2026-05-25-phase-b-recorder-patch-plan.md`. Both uncommitted per CLAUDE.md "specs and plans are never committed."

### `/code-review` 15-fix sweep over the recorder-patch redesign (2026-05-25)

**One-sentence summary.** A `/code-review` pass (five finder angles + one-vote verifier + sweep) on the recorder-patch Phase B redesign surfaced 15 confirmed/plausible findings; the fix wave landed in the same session across `src/eval/smem_check.py`, `src/agents/coder.py`, `src/agents/planner.py`, `src/agents/llm_backend.py`, `src/config.py`, `src/search/orchestrator.py`, `src/benchmark/baseline_generator.py`, four test files, and three doc/prompt files — closing the post-recorder fail-open chain, plumbing iter telemetry through every emit site, and tightening the hw block's rendering across Coder + Planner.

**Headline — DPS-kernel fail-open resolved (fix #1).** The post-recorder limitation from the 2026-05-24 entry (matmul host wrappers with `def f(a, b, c)` signature failing the recorder drive with TypeError → empty violations → fail-open) is now materially closed. `_maybe_synth_dps_outputs(host_wrapper_fn, sample_args, *, iter_no)` in `coder.py` uses `inspect.signature(host_wrapper_fn)` to detect missing positional args and synthesizes them via `torch.empty_like(sample_args[0])`. Emission of `smem_check_skipped(reason="dps_synth_failed")` only fires when the signature is unparseable or the heuristic can't determine the shape; in the common matmul case the recorder receives a properly-shaped sample-args tuple and the SMEM check actually fires. The 2026-05-24 "production-shape limitation" paragraph is now obsolete in the matmul case the limitation was filed for.

**Multi-call recorder preserves captured args across post-capture crashes (fix #2).** The recorder's outer `except Exception` was unconditionally returning `None`, discarding any first-call capture if the wrapper later raised on a second internal launch or post-launch host code. Now: if `captured["called"]` is True at the catch site, the captured args/kwargs are returned with a new telemetry slug `host_wrapper_crashed_after_capture` so post-run analysis can distinguish "no capture" from "captured but wrapper later raised". Honors the docstring's "only the first recorded invocation is used".

**kwarg-conflict detection (fix #3).** `cfg.kwargs` no longer silently overwrites recorded host-wrapper kwargs. When a key appears in both, emit `smem_check_skipped(reason="cfg_overrides_recorded_kwarg", config_idx=i, key=...)` and skip that Config. Telemetry surfaces shape-dependent constexpr conflicts (non-standard wrappers that fix `BLOCK_K=K_runtime`) that would otherwise produce false-positive/negative SMEM measurements at warmup.

**Legacy `_check_via_direct_warmup` REMOVED (fix #6).** The fallback path that re-introduced the original production-shape fail-open is gone. `check_autotune_smem_budget` now requires `host_wrapper_fn: Callable` (no longer `Optional`); any future caller must drive the host wrapper to capture JIT args. Removing the fallback removes the regression vector.

**`iter_no` plumbing (fixes #8 + #9).** Threaded through `CoderAgent.implement` / `translate` / `_run_tool_agent` / `_make_compile_tool` / `check_autotune_smem_budget` / `_capture_jit_args_via_host_wrapper`. All 9 SMEM-check `events.emit` sites now carry `iter=iter_no` instead of `iter=null`. Orchestrator K-way fan-out at `src/search/orchestrator.py` passes `iter_no=iter_no`; `baseline_generator.generate_triton_baseline` passes `iter_no=0`. `events.jsonl` is now joinable on `iter` for every SMEM-check event.

**3 new HardwareSpec derived properties (fix #11).** `peak_flops_tf32`, `peak_flops_fp8`, `peak_flops_nvfp4` added so `render_run_context()` dominant-dtype selection considers all six dtypes (was three). Hopper fp8 (1980 TFLOPS) and Blackwell nvfp4 are now represented when picking the dominant peak; previously the rendered "Peak FLOPS (fp16):" understated H100 throughput by 2× because the dominant-pick stopped at fp16.

**Symmetric `build_user_prompt` gate fix in both Coder and Planner (fixes #4 + #5).** `if bottleneck is not None:` → `if bottleneck is not None or (hardware is not None and hardware.name):`. The hw block now renders whenever hardware is configured, even on the very first iter before the once-per-run bottleneck classification has fired — symmetric with the `translate.md` fix from round-3 but applied to the steady-state user prompts.

**Other smaller fixes.** `sm_8.9` → `sm_89` format (fix #12, both `llm_backend.py` and `coder.py` — Triton's compute-capability strings drop the dot). `hw_for_check` semantic keyed on `shared_mem_per_block_bytes > 0` rather than empty-name (fix #13 — a present-but-unnamed hw spec with a real cap shouldn't be discarded). `Kernel(spec=, source_code=, triton_kernel_name=jit_names[0])` set at construction (fix #14) so `__post_init__` parses the right autotune block instead of the first `@triton.jit` it finds. `sample_args` lifted to orchestrator (fix #15) — K-way fan-out shares one allocation instead of K independent generator invocations, eliminating a per-candidate ~hundreds-of-ms allocation cost on large inputs. Zero-peak FLOPS line omitted from prompt render (fix #7 — `peak_flops_*=0` rendered as "Peak FLOPS (fp16): 0 TFLOPS" which the LLM read as a real signal). "- Bottleneck (this run):" documented across three doc/prompt files to match the code (fix #10).

**Pointer.** Findings JSON at `/code-review` output; spec updated in-place at `doc/specs/2026-05-24-coding-hw-spec-design.md` §6 (still uncommitted, gitignored). Plan at `doc/plans/2026-05-24-coding-hw-spec-plan.md` left stale — process exhaust per CLAUDE.md.

### Run-context enrichment: SM count, max threads/block, L2, TC tile, workload shapes (2026-05-25)

**One-sentence summary.** The `## Run context` block grew five new lines — `SM count`, `Max threads per block`, `L2 cache`, `Tensor Core tile (<dtype>)`, and `Workload shapes` — each chosen so the LLM has a single shot at the per-arch numeric budget it would otherwise either guess at or hallucinate from training-data priors that almost never match the live Ada-vs-Hopper-vs-Blackwell distinction.

**Why these five, and not others.** **SM count** drives every grid-sizing decision: a kernel choosing `grid = (M // BLOCK_M,)` against a 142-SM Ada part wants `M // BLOCK_M` to land somewhere in the `2×sm_count` to `4×sm_count` neighborhood for occupancy, and the LLM can't reason about that without seeing the count. **Max threads per block** lets the Coder ceiling its `num_warps` choice — Triton's `num_warps × 32 ≤ max_threads_per_block` is a hard launch-bound rule, and rendering the implied `num_warps ≤ 32` (Ada's 1024 / 32) inline beats the LLM extrapolating from a generic "warp size 32" prior. **L2 cache** capacity (96 MiB on Ada, 50 MiB on H100, ~80 MiB on B100) is the right level of abstraction for tile-reuse reasoning — "does the K-loop's reused operand fit in L2 across SMs?" is a question Triton can't make explicit (it allocates and prefetches but doesn't expose residency hints), so surfacing the cap lets the LLM make the working-set tradeoff calling-site instead of guessing a generic "tens of MB" anchor. **Tensor Core tile** is the BLOCK_K sizing anchor — Ada/Ampere fp16 want BLOCK_K as a multiple of 16, Ada fp8 wants 32, Hopper WGMMA pins K=16 for fp16 and 32 for fp8 — and without this line the Coder picks aspirational `BLOCK_K=64` configs that ptxas accepts but where the underlying MMA decomposes into N sub-instructions that wreck the achieved-TFLOPS curve. **Workload shapes** close the autotune-key alignment gap: when the LLM authors `@triton.autotune(key=["M", "K"])` against a workload set whose K is invariant across all selected workloads, the autotune partition is degenerate — surfacing the shape range lets the LLM (and the Reviewer's `pct_peak` prose) align the autotune key to the dimensions that actually vary.

**Why NOT others.** **Per-thread register limit (255 on Ada, 255 on Hopper)** is redundant with `max_threads_per_block` for Triton-level reasoning — Triton lowers to PTX with its own register-spill heuristics, and the LLM can't influence the register count at the source level without dropping below the abstraction. **PCIe / NVLink bandwidth** is irrelevant for kernel design (the kernel doesn't see host-device transfer); leaving it out keeps the block from accreting irrelevant numbers. **L1 cache** (per-SM, 128 KB on Ada) is not Triton-controllable — Triton's SMEM allocator lives in the 99-KB-per-block opt-in regime that the existing `shared_mem_per_block_bytes` line already covers, and the L1 vs SMEM partition isn't operator-tunable from Triton source — so adding an L1 line would be misleading prompt clutter.

**`peak_flops_int8` is still excluded** from the dominant-dtype selection — same rationale as fix #11's `int8` exclusion in the 15-fix sweep. Integer ops report as TOPS, not TFLOPS, and merging int8 into the floating-point peak table either forces the LLM to disambiguate the unit at every reference (and probably fail) or silently overstates the achievable headroom on a bf16 workload that picks `int8/bf16` ties via the rendering's `/`-join. The HardwareSpec field stays available for a future `query_hardware` tool (PROCESS backlog) that surfaces it on demand with the explicit "TOPS" unit attached.

**Tensor Core tile lookup is a renderer concern, not a HardwareSpec field.** The mapping from `(compute_capability, dominant_dtype)` to a tile string is a small per-arch lookup table — `m16n16k16` on Ampere/Ada fp16, `m16n16k32` on Ada fp8, the WGMMA prose for Hopper, the tcgen05 placeholder for Blackwell — and it lives in `_tensor_core_tile_for(...)` inside `llm_backend.py` rather than as a HardwareSpec property. The split keeps HardwareSpec focused on raw queryable counts (MAC/cycle, capacity, frequency) instead of accreting derived presentation strings; the `query_hardware` post-V1 tool wants the raw counts, not the prompt prose. Helper returns `None` when the dtype isn't supported on the arch (e.g. fp8 on Volta/Turing/Ampere), and the renderer omits the line entirely in that case — same fail-quietly rule as the zero-peak FLOPS omission from fix #7.

**Workload shapes are part of the once-per-run context block, not a separate "search context" block.** The semantic question is "what does the agent need to see about the immutable run setup?" — and the workload shape envelope is part of that immutable setup, even though it changes per problem. The threshold rendering — literal tuples up to 3, per-dim ranges with `N=<count>` beyond — is the same heuristic the Reviewer prose uses for "summarize when the list gets long enough to be noise." If the search-state info (current iter, last action, parent kernel age) ever needs its own broadcast block, that goes into a future `## Search context` section; the current single `## Run context` block stays focused on run-invariant facts.

**Implementation.** Two new fields on `HardwareSpec` (`sm_count`, `max_threads_per_block`); `detect_hardware` populates via `getattr(props, "multi_processor_count", 0)` / `getattr(props, "max_threads_per_block", 0)` matching the existing `shared_memory_per_block_optin` fallback pattern; `validate_hardware_spec` gets 10% tolerance checks on both new fields with skip-if-zero gating; `configs/arch/RTX6000Ada.yaml` carries `sm_count: 142` and `max_threads_per_block: 1024` from the AD102 datasheet. `render_run_context` adds four new lines under the existing `Shared mem per SM` line plus the Workload shapes line at the bottom of the block, each omitted independently when the underlying data is missing (`sm_count == 0`, `max_threads_per_block == 0`, `SRAM_capacity == 0`, helper returns None, or `workload_shapes` empty/None). The orchestrator computes `workload_shapes = [tuple(w.axes.values()) for w in workloads]` once per `run()` invocation and threads it through every Planner / Reviewer / Coder call site (4 total: baseline Reviewer at iter 0, per-iter Planner, K-way Coder fan-out, per-iter child Reviewer); the baseline generator computes the same and passes to `coder.translate`.

**Pointer.** Post-V1 backlog adds the `query_hardware(field)` tool that pulls additional HardwareSpec fields on demand — see PROCESS.md "Backlog (post-V1)" entry. That tool is the right place to surface `peak_flops_int8`, the full `MAC_per_cycle_*` table, and any field the curated block deliberately omits; gated on the same trigger as the Reviewer's metric-queries flag (real run where the LLM consistently asks for a field that's not in the broadcast block).

### Bench-subprocess isolation (Scope B+) — brainstorm outcomes (2026-05-24)

Closes the open questions reserved in the 2026-05-20 entry. Spec lives uncommitted at `doc/specs/2026-05-24-bench-subprocess-isolation-design.md`; plan at `doc/plans/2026-05-24-bench-subprocess-isolation-plan.md`.

- **Subprocess scope** — bench + NCU profile in one per-iter child ("Option C" in brainstorm). Initially scoped as bench-only ("Option A"); upgraded to include NCU because NCU is the leading suspect for the original context poisoner per the 2026-05-20 entry. Containing only the symptom (bench autotune burn-in) without the suspected source would predictably leave the same failure shape in a different layer.
- **Spawn + IPC** — `subprocess.Popen` running `python -m src.eval.bench_worker --request <path> --response <path>`; JSON via per-iter shared tempfiles under `<run_dir>/iter_<n>/worker/`. Chosen over `multiprocessing.Process`/pickle (debuggability) and stdin/stdout streaming (file persists for postmortem on child crash).
- **Failure vocabulary** — response.json is source of truth for per-candidate verdicts; non-zero exit / signal-kill means worker itself crashed → all K candidates treated as failed + `consecutive_worker_crashes` counter (threshold 3 → `WorkerProcessUnstable` whole-run abort, mirrors today's `CUDAContextPoisoned`). `consecutive_cuda_errors` retires from the bench path.
- **Channel-A semantics** — child aborts immediately on first `RewardHackDetected`; remaining candidates marked `status="not_run"`. Matches today's parent-side `break` semantics. Choice 2 ("run all K, report per-candidate flags") rejected — it would introduce a within-iter taint problem today's code does NOT have.
- **Artifact layout** — self-contained per-iter worker dir; parent merges `worker/events.jsonl` into canonical `<run_dir>/events.jsonl` on clean exit, copies `.ncu-rep` files into NCU cache + `tree/node_<id>/`. Non-clean exit leaves the worker dir alone as postmortem package.
- **Worker module structure** — `src/eval/bench_worker.py` with pure-fn `run_iter(request: dict) -> dict` + thin `__main__` CLI; orchestrator branches on `ACTSConfig.bench_use_subprocess: bool = True` (in-process bypass for debugging — pattern mirrors `use_operator_baseline`).
- **Free bonus** — the open "per_iter_anti_cheat state restoration" PROCESS.md tech-debt item (A2 follow-up) auto-retires for the *cross-iter* slice: child dies at end of iter, patched primitives die with the process, iter N+1 spawns a fresh child. Within-iter K-sibling pollution + parent-side cache drift remain open (spec §3 decision #9 residuals).
- **Clock-lock** — child inherits parent's device-level clock lock (nvidia-smi modifies device state, not process). No re-acquire.

Trigger reaffirmed by `runs/run_20260526T074709_405554Z` — iter 4 hit `cudaErrorIllegalAddress` on 3-of-4 K-way candidates inside autotune burn-in; iter 5 inherited the poisoned context and burned through Coder turn budgets (cands 1-3) without recovery. Second observed instance of the trigger pattern from `run_20260520T011532_929220Z`.

### Bench-subprocess isolation (Scope B+) — landed (2026-05-27)

The Scope B+ refactor (per-iter bench + NCU child process; spec at `doc/specs/2026-05-24-bench-subprocess-isolation-design.md`, plan at `doc/plans/2026-05-24-bench-subprocess-isolation-plan.md`, both uncommitted scratch per CLAUDE.md) has landed across `src/eval/bench_worker.py`, `src/eval/bench_subprocess.py`, `src/search/orchestrator.py`, `src/config.py`, and `src/runtime/events.py`, replacing the ~280-line K-way per-candidate try/except + profile-gauntlet path with a dispatch-and-rehydrate seam that contains CUDA context poisoning to a single iter and auto-retires the cross-iter slice of the open A2 anti-cheat tech debt as a free side effect.

**Module landscape.** `src/eval/bench_worker.py` is the new worker module — `run_iter(request: dict) -> dict` is the pure function the child executes for one iter's K-way bench + profile-gauntlet, `build_request(...)` constructs the request payload parent-side, `_encode` / `_decode` are the JSON serializers for nested dataclass shapes (KernelSpec, BenchmarkResult, ProfilingResult), and `_main_cli` is the `python -m src.eval.bench_worker --request <path> --response <path>` entry point. `src/eval/bench_subprocess.py` is the parent-side spawn helper — `WorkerCrashed(RuntimeError)` is the exit-code / signal-kill / malformed-response exception, async `run_bench_subprocess(*, request, worker_dir, worker_crash_threshold, worker_timeout_s)` does the Popen + wait + watchdog dance and returns the decoded response dict, and `merge_worker_artifacts(*, run_dir, worker_dir, iter_no, response, ncu_cache_dir)` concatenates `worker/events.jsonl` into canonical `<run_dir>/events.jsonl` and copies `.ncu-rep` files into both the NCU cache and `tree/node_<id>/`. `src/search/orchestrator.py` gained four module-level helpers for the IPC boundary — `_serialize_kernel_spec_for_request`, `_rebuild_cand_kernel`, `_rehydrate_bench_result`, and `_rehydrate_profiling_result` — plus a new `WorkerProcessUnstable(RuntimeError)` exception mirroring the `CUDAContextPoisoned` 3-strike escalation shape.

**Modified APIs.** `ACTSConfig` gained `bench_use_subprocess: bool = True`, `worker_crash_threshold: int = 3`, and `worker_timeout_s: float = 1800.0` — the timeout was originally landed as `worker_startup_timeout_s` with a 30s default, which the Codex 2026-05-26 adversarial round caught was killing healthy workers mid-NCU profile; the field is total-lifetime watchdog semantics, not startup-only, and the rename + default bump pins that intent. `Orchestrator.__init__` carries `self.consecutive_worker_crashes = 0` as a peer of `consecutive_cuda_errors`. `Orchestrator.run(...)` gained `run_dir: Path | None = None` and `ncu_cache_dir: Path | None = None` kwargs so the worker dir / NCU cache locations can be threaded through without going through ACTSConfig; `optimize(...)` gained `run_dir: Path | None = None` mirror, and `main()` threads `ctx.run_dir` from the pipeline context. `src/runtime/events.py::CORE_EVENT_KINDS` registered four new kinds — `bench_worker_spawned`, `bench_worker_exited`, `bench_worker_crashed`, `worker_chunk_merged` — for the spawn / clean-exit / crash / artifact-merge lifecycle.

**The K-way loop is replaced.** ~280 lines of OLD per-candidate try/except wrapping `benchmark_kernel → profile_gauntlet` are gone; in their place is dispatch-and-rehydrate (~290 lines net) that builds one request per iter, spawns the worker, awaits the response, and rebuilds per-candidate `BenchmarkResult` / `ProfilingResult` objects from the JSON shapes the helpers produce. `_CUDA_STICKY_PATTERNS` + `consecutive_cuda_errors` + the `except RuntimeError` recovery branch all retire from the bench path — those constants and the `CUDAContextPoisoned` class stay in source for the parent-side CUDA touches that the refactor doesn't move (the Coder's `run_correctness` tool path; Phase C winner re-profile).

**Event emission moves parent-side.** Per the Codex 2026-05-27 fix wave (items #1 + #2 + #4 in the `/code-review xhigh` round), per-candidate events that the parent can reconstruct from the response are now emitted parent-side from the response-handling loop instead of by the worker — `coder_failed` (channel-A + bench-failed + partial-bench branches), `reward_hack_detected`, the winner's `bench_done`, and the winner's `profile_done`. The worker still emits two events the parent cannot reconstruct: profile-gauntlet `coder_failed` for non-winner candidates that bench-succeeded but profile-failed (only the worker sees the per-candidate profile attempt). The worker's `_emit` now prepends `{"ts": iso_ts(), ...}` matching `runtime/events.py::emit`'s shape so post-merge timeline ordering is correct. `merge_worker_artifacts` is now ALWAYS called regardless of `bench_use_subprocess` — the in-process bypass also merges so events and `.ncu-rep` files reach canonical artifacts uniformly across both dispatch modes.

**Free bonus from the subprocess boundary.** The open A2 "per_iter_anti_cheat state restoration" tech debt is auto-retired for the cross-iter slice — child dies at end of iter, patched primitives die with the process, iter N+1 spawns a fresh child with an empty `sys.modules` and a fresh `torch.cuda` namespace. The within-iter K-sibling pollution residual and the parent-side `compile_kernel` cache drift residual remain open (catalogued in PROCESS.md A2 follow-ups + the Tier 3 list) — those require either explicit namespace restoration in the child between cands or per-candidate subprocess (Scope A in the original 2026-05-20 trade-off), neither of which is justified by current evidence.

**Trigger runs.** `runs/sweep_l1_048/regime_03_default/rep_0/run_20260520T011532_929220Z` was the original poster child — `cudaErrorInvalidAddressSpace` propagated across iters 1-5 with the smoking-gun sha256 evidence pinning the parent context as the source. `runs/run_20260526T074709_405554Z` was the second observed case from the brainstorm-round confirmation — `cudaErrorIllegalAddress` at iter 4, iter 5 inherited the poison and burned Coder turn budgets without recovery. Both runs are exactly the shape this refactor contains to a single iter.

**In-PR Codex review iterations.** Four review rounds landed before the PR closed; each round caught a distinct class of issue:

1. **Codex adversarial #1 (2026-05-26) — 3 fixes.** (a) `.ncu-rep` stale path on rename: the worker writes `.ncu-rep` under `worker/`, the parent renames into NCU cache, and the `ProfilingResult.ncu_rep_path` was still pointing at the worker-side path — `dataclasses.replace(profile, ncu_rep_path=dest)` is now applied before re-encoding. (b) `profile_config` was missing `blob_roots` + `t_sol_us` + `baseline_latency_us`: the worker was falling back to `1.0` defaults that diverged from parent ranking, and safetensors workloads couldn't reconstruct their blob locations — the request payload now carries all three explicitly. (c) Watchdog rename: `worker_startup_timeout_s` → `worker_timeout_s`, default `30s` → `1800s`, because the field is total-lifetime and 30s was killing healthy workers mid-NCU profile.
2. **Codex review #1 (2026-05-26) — 2 P1 fixes.** (a) Partial-bench `is_fully_successful=False` no longer leaks into the success path — the worker now mirrors the OLD orchestrator gate (`if not is_fully_successful: continue`) so a partially-successful candidate gets classified as a failure instead of being ranked. (b) `_load_definition(Path(""))` placeholder / no-workload crash is guarded — returns None for empty / missing / non-dir paths instead of raising.
3. **Codex review #2 (2026-05-26) — 2 P2 fixes.** (a) `WorkerCrashed` handler no longer bumps `parent.consecutive_agent_failures` — it was double-counting infra crashes against the agent-quarantine threshold, and 2 transient worker crashes on the only frontier node would end the search before the 3-strike `WorkerProcessUnstable` could fire. (b) Malformed `response.json` (truncated JSON from OOM-kill mid-write) now raises `WorkerCrashed` cleanly instead of leaking `JSONDecodeError` up through the orchestrator.
4. **Internal `/code-review xhigh` (2026-05-27) — 8 Tier 1+2 fixes.** Itemized:
   - **#1 Event-stream dedup** (Tier 1). Stripped 5 worker `_emit` calls that mirrored parent emits — channel-A `reward_hack_detected` + `coder_failed`, bench-failed `coder_failed`, partial-bench `coder_failed`, success-path `bench_done`, winner `profile_done`. Worker retains the 2 emits only it can see (profile-gauntlet `coder_failed` on non-winner cands that bench-succeeded).
   - **#2 `ts` in worker events** (Tier 1). Worker `_emit` now prepends `{"ts": iso_ts(), ...}` matching the parent shape so post-merge ordering is right.
   - **#3 Silent-downgrade WARN** (Tier 1). Orchestrator now logs a WARNING when `bench_use_subprocess=True && run_dir is None` — the previous silent fallback to in-process was a safety regression masquerading as graceful degradation.
   - **#4 Merge always** (Tier 1). Dropped the `if _subprocess_enabled:` gate on `merge_worker_artifacts`; in-process bypass merges too, so events + `.ncu-rep` reach canonical artifacts uniformly across both dispatch modes.
   - **#5 Defensive `_rehydrate_profiling_result`** (Tier 2). Nested `AnalyticalMetrics` / `NCUMetrics` construction is wrapped in try/except — missing required fields degrade to `None` instead of crashing the orchestrator on a malformed-but-recoverable response.
   - **#6 `_load_definition` guard** (Tier 2). Duplicated with the Codex #1 P2 fix; returns None for empty / missing / non-dir paths.
   - **#7 Tempdir hoist** (Tier 2). `_effective_run_dir` / `_effective_ncu_cache` / `_subprocess_enabled` resolution is hoisted outside the iter loop — was leaking `max_depth` tempdirs per run when `run_dir is None` — and `atexit.register(shutil.rmtree, ...)` handles cleanup at process exit.
   - **#8 Lazy-import test isolation** (Tier 2). Deferred — the current pattern works for all existing tests; the Codex concern is hypothetical for future test patterns that would patch upstream paths.

**Tests.** 408 passed across the Tier 1 cross-area sweep: `test_bench_worker`, `test_bench_subprocess_helper`, `test_orchestrator_bench_subprocess` for the new modules; `test_config`, `test_runtime_events` for the API additions; `test_tree_dump`, `test_failure_nodes`, `test_search_tree`, `test_run_context`, `test_planner`, `test_reviewer`, `test_coder` for the contract-adjacent surface. Two pre-existing unrelated `test_acts_config_reviewer_metric_queries_*` failures persist — those assert `reviewer_metric_queries=False` but the actual default has been `True` since well before this PR; they're config-default-drift unrelated to this refactor and tracked separately. The Tier 2 GPU smoke test file (`tests/test_bench_subprocess_gpu.py`) is deferred — not yet written; the cross-process bench + NCU path has been exercised end-to-end via the live optimize runs that motivated the refactor (the trigger runs above), but a focused mark-gpu test for the subprocess seam itself is the next obvious addition.

**Tier 3 catalogue.** 15 items are recorded under PROCESS.md "Trigger-gated tech-debt" → "Bench-subprocess isolation Tier 3 follow-ups" with explicit triggers + fix shapes: falsy-zero in `_bench_config_shim`, falsy-zero in `t_sol_us` / `baseline_latency_us`, asyncio.to_thread cancellation can't interrupt subprocess, non-atomic `response.json` write, `request["ncu_cache_dir"]` unread by child, profile-gauntlet placeholder lambda → ProfilerError, `coder_submitted.n_profile_attempts` hardcoded to 1, `_emit` locale encoding, dead `agent_failure_count` bump, entrypoint pre-filter silent on coder_failed, `_rehydrate_hardware_spec` zero-HardwareSpec, `_read_tail` full-file read, zombie process from `proc.kill()` without `proc.wait()`, `autotune_burn_in_done` semantic drift via `_child_autotuner`, and the lazy-import test isolation hypothetical from `/code-review` fix #8. Each entry carries a concrete trigger condition + a fix sketch so re-entry on any of them is cheap when evidence accumulates.

**Pointer.** Spec: `doc/specs/2026-05-24-bench-subprocess-isolation-design.md`. Plan: `doc/plans/2026-05-24-bench-subprocess-isolation-plan.md`. Both uncommitted per CLAUDE.md "specs and plans are never committed."

### Correctness-subprocess isolation — closing the last in-parent untrusted-launch site (2026-05-31)

**Root cause.** `runs/run_20260529T142421_439323Z` produced zero improvement across the whole run, and the diagnosis traced to a sticky CUDA-context poison in the long-lived parent process — the same failure *class* as the Scope B+ bench poisoning, but a different *site*. After Scope B+ moved bench + NCU profile into a per-iter child, correctness verification was the last untrusted-kernel-launch site still running IN-PARENT. One out-of-bounds candidate launched during the in-parent correctness check fired a device-side assert that poisoned the parent context for the rest of the run; every subsequent `torch.randn(device="cuda")` / input-gen in the parent failed from that point on. The failures were invisible because three fail-open input-generator sites caught and swallowed the sticky error silently — the run kept "advancing" through iters that could no longer produce a working candidate, with no surfaced signal that the context was dead.

**Decision — subprocess isolation (mirror Scope B+, applied to correctness).** Route all untrusted correctness launches through a crash-isolated `correctness_worker` subprocess via `run_correctness_subprocess`, fail-closed. Three parent-side launch sites are isolated: the Coder's `check_correctness` tool (now async), the orchestrator's reward-hack re-eval (`mode="strict_recheck"`), and the baseline post-verify. Subprocess-per-call was chosen over a persistent worker because the run is LLM-bound — cold-start is ~1% of wallclock, so the simplicity of spawn-per-call beats the lifecycle complexity of a pooled worker for no meaningful wallclock cost. It was also chosen over the alternative of dropping numeric verification entirely and running a compile-only Coder loop: keeping a live subprocess preserves real numeric-correctness feedback to the Coder, which is load-bearing signal we did not want to trade away to dodge the poison.

**Decision — trust gate by construction, not by coincidence.** `allow_in_parent_fallback=False` makes the "no untrusted launch in the parent" invariant enforced at construction time rather than relying on the coincidence that real runs happen to carry a `problem_definition_path` (which is what would route the launch to the child). Previously, a code path that forgot to thread the definition path would silently fall back to launching in the parent — re-opening the exact poison hole. With the gate, a forgotten path now raises `CorrectnessIsolationError` loudly instead of degrading silently into the failure mode this PR exists to close.

**Bounded timeout — do not inherit the bench timeout.** New `correctness_worker_timeout_s` (default 180s). The correctness path must NOT reuse the bench `worker_timeout_s` (~50h, sized for NCU profiling), because a hung correctness candidate under that ceiling would stall the run for ~2 days before the watchdog fired. Correctness is a fast check; 180s is generous for it and bounds the hang.

**Consolidations / hygiene.** Bench and correctness now share `worker_spawn.spawn_worker` for the Popen + watchdog + crash-classification dance (one implementation, two callers). New `strict_compare_one_workload` and `build_correctness_request` factor the per-workload compare and request-construction out of the worker. The three fail-open input-generator sites now swallow-LOG (the sticky error is recorded, not silently eaten) so a future poison surfaces in the logs even if some fail-open behavior is retained. `del _repr_inputs` after the dtype-probe frees the probe tensors' VRAM. Correctness scratch dirs are cleaned up after each call.

**Accepted residual (b) + revisit trigger.** The Coder's SMEM compile-tool still drives LLM-authored host-wrapper PYTHON in-parent. This is a deliberately-accepted residual, not an oversight: the untrusted Triton *kernel* itself does not launch there — the recorder no-ops `autotuner.run` and the per-config `warmup` is compile-only — so the poison risk is low (no device-side kernel execution on the untrusted path). It does, however, run arbitrary host Python with no timeout, so a hang is theoretically possible. It is deliberately NOT isolated now because the compile tool fires per self-check (high frequency), which makes subprocess-per-call a bad cost ratio against a low residual risk. **Revisit → full compile-tool subprocess isolation ONLY if** a host-wrapper poisoning OR a hang is observed in a real run; absent that evidence, the per-self-check spawn cost is not justified.

**Pointer.** Trigger run: `runs/run_20260529T142421_439323Z` (zero-improvement run, parent-context poison from the in-parent correctness launch). New modules: `src/eval/correctness_subprocess.py`, `src/eval/correctness_worker.py`, `src/eval/worker_spawn.py`.

### Opt-mem dedup — Codex adversarial-review fixes (2026-06-02)

The condition-keyed dedup feature drew a Codex adversarial review with four findings; this entry records the settled fix design (the feature's own design rationale is in the companion entry below).

**#1 — store write can wipe / race the shared store. Decision: keep the consolidated atomic rewrite, but make the write path read-merge from disk; no inter-process lock.** The bug splits in two. The *certain* half (#1a) is single-process: in write-only mode (`read_enabled=False, write_enabled=True`) `optimize.py` deliberately skips `store.load()`, so the in-memory cache is empty and T2's whole-file rewrite truncated the shared store down to just the current run's rows. The *conditional* half (#1b) — concurrent writers losing updates — was **confirmed out of scope**: there is exactly one store writer per run (`producer.flush() → store.add_many` in the parent at `orchestrator.py:691`, once at run-end; the bench/correctness worker subprocesses never touch opt-mem), and the user never runs two write-enabled `optimize` processes against the same store at once. With no concurrency, append-only's one real advantage (dodging the concurrent clobber) evaporates, so we keep the consolidated on-disk store the dedup feature was built to produce. The fix makes the write self-sufficient: `add`/`add_many` re-read the current on-disk rows (via a `_parse_rows` helper factored out of `load()` so both share the identical tolerant parser), `dedup_best(disk + cache + new)`, then atomic `tmp + os.replace` (per-pid temp name as stale-`.tmp` hygiene, not a lock). Truncation becomes impossible *by construction* regardless of whether `load()` ran — the robustness Codex was pushing for — at a cost of one extra whole-file read per run. **Append-only was considered and rejected**: it would abandon the bounded/consolidated store for no benefit absent concurrency, and unbounded on-disk history grows `load()` cost over time. **flock was considered and deferred**: revisit only if concurrent write-enabled runs are ever introduced.

**#2 — `condition` is an unvalidated prompt-injection channel.** `_render_past_experiences` rendered `condition` verbatim while title/lesson go through `_neutralize_prompt_markdown`. Fix: a new `_neutralize_metadata` (flatten to one line via `" ".join(text.split())` + collapse 3+ backtick fences) renders the condition; legitimate machine-generated conditions are already single-line so it's a no-op for them. Plus a `store.load()` guard: a non-string `condition` (hand-edited / malicious JSONL) is dropped to `""` with a warning so it can never reach `dedup_key()` as an unhashable tuple member and abort the load.

**#3 — legacy rows with distinct params collapse on migration.** `dedup_key` omits `action.parameters` and legacy rows default `condition=""`, so legacy edge rows with the same action_id but different param variants collapsed on `load()`. Fix matches the dedup spec's stated params-only legacy fallback: move `_format_condition` from `producer.py` to `experience.py` (co-located with `dedup_key`/`dedup_best`; producer keeps a re-export so existing imports resolve), and in `_row_to_experience` backfill `_format_condition(None, action)` (params-only) when the `condition` key is *absent* (true legacy row). New rows always serialize the key and are untouched.

**#4 — single-edge G3 suppression can discard the only lesson.** `finalize()` suppressed the run-scope G3 row whenever `best.parent_id == baseline.id`, assuming an edge row was captured — but with `cap == 1` `consider()` returns before buffering any edge (and edge-summary `None` is the same gap), so the documented G3-only mode flushed nothing. Fix: suppress G3 only when a baseline→best edge row *actually survives* in `_edge_buffer` (provenance match), not on the `parent_id` heuristic. Empty buffer → G3 is written.

**Pointer.** Review: Codex adversarial (verdict needs-attention, 4 findings). Spec: `doc/specs/2026-06-02-optmem-dedup-review-fixes-design.md` (uncommitted per CLAUDE.md). Touches `experience.py` (host `_format_condition`), `store.py` (#1/#2-guard/#3-backfill), `producer.py` (#3-reexport/#4), `planner.py` (#2-render), `optimize.py` (#1 comment).

**Round 2 — native Codex review of the #1 read-merge-rewrite (3 findings, all `store.py`).** The whole-file rewrite + `dedup_best` tuple ops are destructive/fragile against rows the old append-only parser never rewrote or hashed. **P1 (data-loss):** `_parse_rows` skips `schema_version > KNOWN_VERSION` rows, then the rewrite dropped them → an older binary silently deletes a newer binary's lessons. Fix: the parser returns `(experiences, passthrough_future_raw_lines)`; the write path carries the future rows through `_rewrite` verbatim — **preserve, not abort** (aborting compaction on any future row would stop consolidation the moment a newer-binary row lands). **P2a (availability):** a list/dict in any `dedup_key` identity field (`kernel_type`/`hardware_arch`/`scope`/`action_id`) makes the tuple unhashable and aborts the whole load/merge — Round-1 #2 only guarded `condition`; generalized to skip rows with non-string identity fields. **P2b (availability):** a non-string `created_at` breaks `dedup_best`'s `(speedup, created_at)` tie-break (`str` vs `dict` TypeError); coerce to `""` at parse. All fixes at the parse boundary — the single chokepoint every `dedup_best` caller (incl. retriever) flows through; `experience.py` unchanged.

### Opt-mem condition-keyed dedup — feature design (2026-06-02)

**Problem.** Analysis of a real run (`run_20260602T030232_055579Z`, opt-mem reads enabled) surfaced that the store accumulated near-duplicate lessons with no dedup at either layer. Two distinct sources: **(A) edge+run double-write** — a single-edge run (best-of-run node is a direct child of baseline) wrote *both* an edge row (`action_id` set) and a run-scope G3 row (`action_applied=None`) describing the same single improvement; and **(B) cross-run technique repeats** — the same technique winning on the same kernel/arch across runs produced the same insight reworded each time. Retrieval compounded it: `sample()` used `random.choices(pool, weights=…)` *with replacement*, so the Planner could be handed the literal same row twice.

**Decision — deterministic `condition` keys dedup at both layers.** Each `Experience` gains a deterministic `condition` string computed at write time from data already on hand — no LLM, no summarizer. The condition is the run bottleneck plus the action's sorted parameters: an edge row reads e.g. `"compute_bound | BLOCK_N=32"`, a run-scope row carries the bottleneck only (`"compute_bound"`), a legacy row with no stored bottleneck falls back to params-only. It does double duty: a **dedup discriminator** (same technique + same condition collapse, keeping the best speedup; *different* conditions are preserved as genuinely distinct lessons) and a **read-time disambiguator** (rendered to the Planner as `applies when: <condition>` so two surviving same-technique lessons are distinguishable).

**Mechanism.** The dedup key is `(kernel_type, hardware_arch, scope, action_id-or-"∅", condition)`; `dedup_best` keeps the highest `(speedup, created_at)` per key. Both layers consume it. The store consolidates by key on write and on load — the *final* write-path mechanism (read-merge from disk + atomic `tmp`/`os.replace` rewrite + forward-compatible passthrough of unknown-schema rows) is detailed in the review-fix entry above, which superseded the originally-built whole-file rewrite. The retriever dedups its candidate pool by key and then samples *without* replacement, so no literal repeat can reach the Planner. The producer threads the once-per-run bottleneck (classified once in the orchestrator) into `consider`/`finalize` to compute the conditions, and suppresses the redundant single-edge G3 row so the edge+run double-write is eliminated at the source.

**Both layers, not one.** Write-time consolidation is the source of truth and bounds the on-disk store; read-time dedup is defense-in-depth against un-compacted or legacy rows. Both are necessary because the store is shared across runs and across binaries, so a clean in-memory view at write time does not guarantee every row on disk has already been consolidated.

**Pointer.** Feature spec `doc/specs/2026-06-02-optmem-dedup-design.md` + plan `doc/plans/2026-06-02-optmem-dedup.md` are uncommitted scratch, deleted before commit per CLAUDE.md; the two Codex review rounds that hardened this feature are in the entry above.

### Autotune SMEM-budget check — removed (2026-06-03)

**Decision.** Removed `src/eval/smem_check.py` along with its Coder plumbing (`compile_kernel_tool`'s SMEM rejection path, `_maybe_synth_dps_outputs`, `_format_smem_violation`, the host-wrapper arg-capture recorder) and the two telemetry events the check emitted (`smem_check_skipped`, `smem_overflow_detected`). The hw-spec injection half of the 2026-05-24 feature — the two `HardwareSpec` SMEM fields, the `render_run_context()` prompt block, and the dtype-peak rendering — stays; only the compile-time check is gone. The Coder reverts to relying on Triton's runtime config pruning plus the compile/correctness/bench gauntlet.

**Why.** Three reasons compounded. (1) **Redundant.** The compile/correctness/bench gauntlet actually launches the kernel, so it catches a genuinely over-budget kernel by construction — morphology-independent, no static formula or shape dispatch needed. Triton's own autotuner is the other backstop: a `Config` that exceeds the SMEM cap raises "out of resource: shared memory" at runtime, the autotuner drops that `Config`, and the kernel still runs on the surviving configs. The check's only value-add over those two was earlier, more-structured feedback to the Coder. (2) **Chronically inert for the dominant kernel class.** The host-wrapper arg-capture path synthesized the destination-passing-style output buffer as `torch.empty_like(first_input)`, which has the wrong element count for any output-shape≠input-shape kernel (matmul, projection, reduction). The wrapper then crashed on `out.reshape(M, N)` *before* the kernel launch, the recorder reported `host_wrapper_failed`, and the check silently no-op'd — observed at 4/4 candidates/iter in `run_20260602T143405_680880Z`. So for the kernels that most needed the proactive guard, the check never fired. (3) **The morphology-robust fix wasn't worth the coupling.** Driving the wrapper with the oracle's output shape (instead of `empty_like`) would have fixed the inert case but tightened the eval↔coder coupling for a guard whose marginal value over gauntlet+pruning is only earlier/structured feedback — and there's no evidence in real runs that SMEM overcommit was a recurring time-sink that the reactive `autotune_exclude` path + Triton pruning weren't already absorbing.

**Kept.** The `shared_mem_per_block_bytes` hw-spec prompt injection (still a config-sizing hint surfaced to all three agents via `render_run_context`), `autotune_exclude` for runtime SMEM faults (the reactive safety net), the Triton-implicit-SMEM prompt guidance, and the `triton_autotuner` plumbing that records autotune winners (`autotune_burn_in_done`).

**Net.** The Coder is back to Triton runtime pruning + the gauntlet for SMEM safety; the prompts were updated and made self-consistent (the baseline-path `translate.md` had wrongly retired runtime-pruning as a "stale pattern" when the now-removed check was the policy — that framing is corrected so the prompt no longer implies a compile-time rejection exists).

### External reference baseline (Option C) — flashinfer as scoring T_b (2026-06-03/04)

**Problem.** FlashInfer-Bench challenges score a candidate against a *provided* flashinfer reference solution, but ACTS's SOL-score T_b was always the search-root latency. The flashinfer wrapper is a library call, not Triton, so it fits neither existing baseline path: the operator loader rejects it at the `@triton.jit` gate, and a non-Triton root can't be mutated by the Planner/Coder loop (nothing to expand). The challenge wants "did we beat flashinfer," and the root's own latency answers a different question.

**Options weighed.** (A) Ignore flashinfer, keep own-baseline scoring — no code but doesn't measure the thing the challenge asks. (B) Force flashinfer through the operator path — incoherent: a non-mutable root gives the search nothing to expand. (C, **chosen**) Treat the external reference as a scoring *overlay* — the SOL formula's own docstring already sanctions this: "T_b can be set as any fast implementation of the reference." So the reference replaces the *denominator* of the score without pretending to be a tree node.

**The locked decisions.** (1) **Global T_b** — the reference latency is the single scoring baseline, so the search root honestly scores `<0.5` whenever it's slower than flashinfer. (2) **Full 5-stage correctness gate on the reference, hard-fail** — a wrong scoring baseline silently corrupts *every* score, and the gate did real work here: it validated the hand-ported Blackwell→Ada wrapper on real GPU, including its base-2-LSE assumption (`reference_flashinfer.py` returns fa2's base-2 LSE matching the reference's natural-log division). (3) **Black-box `plan()`+`run()` timing through `benchmark_kernel`** — apples-to-apples with how candidates are timed; no plan/run split. (4) **Config = plain `.py` + entrypoint**, mirroring the operator-baseline shape. (5) **Side-channel, not a tree node** — the reference lives in `reference_baseline.py`, never enters the search tree. (6) **Report shows BOTH** the reference T_b *and* the ACTS root's own latency, so "score" and "how fast is our starting point" stay legible side by side. (7) **flashinfer-unavailable hard-fails** rather than silently degrading the baseline. (8) **Additive / opt-in** — `_dispatch_baseline` and both root paths are byte-untouched; the overlay only fires when configured.

**Review-driven hardening worth remembering.** The report initially *conflated* T_b with the root's latency — `score.baseline_latency_us` got repurposed for both, and our own test had encoded that bug as the expected value. Fixed with explicit `SearchResult.baseline_root_latency_us` → `OptimizationReport.acts_root_latency_us`, so the two numbers are separately sourced. The empty-workload path could fabricate a `benchmark_kernel` 100us-sentinel baseline while *silently skipping the correctness gate* — fixed fail-closed with `[validate]` guards (no definition / zero workloads / count mismatch all abort). And kda's `op_type: "dsa_paged"` fell back to `KernelType.CUSTOM`, which silently dropped the attention-gated Tier-6 actions and mis-keyed opt-mem; fixed with one mapping entry (`dsa_paged → ATTENTION`) — the fixture stays a verbatim kda transcode, so the new vocabulary belongs in code, not in the data.

**Cross-module correctness-loop dedup** (user-directed, separate mini-design). The five baseline/verify sites had each open-coded the correctness sweep. Consolidated into a result-returning `run_correctness_gate` that returns the first `CorrectnessGateFailure` (or `None`). Chosen over a callback/raising helper because one consumer — `generate_triton_baseline`'s post-verify — *breaks-with-string* rather than raising, and a raising helper would have inverted its control flow. The crash-capture stays inside the helper with a raw re-raise at the two baseline sites, preserving their behavior verbatim. Named "gate" not "sweep" — "sweep" is already the stage-2 shape-sweep. Free fix folded in: the reference-failure messages had been rendering an always-empty `failure_detail` via a `getattr` that never matched.

**Data-layout decisions.** Kept the per-problem directory convention (Option A) — declined both mirroring kda's HF-split layout and writing a native FlashInfer-trace loader; the existing convention already carries everything the reference path needs. The DSA blob is ONE container-level symlink (a sibling of the problem dirs), gitignored as machine-local (`benchmarks/flashinfer_trace/blob` → off-repo kda dataset), which is what forced `safetensors_blob_roots` to become cfg-loadable (it had been an implicit `definition_path.parent` default; the symlink lives outside that, so the override had to surface in `acts.cfg`).
