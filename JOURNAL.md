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

---

## Backend

### Triton (V1)

**Rationale**: From domain researchers: **agents are not good at writing CUDA-level code** — too complicated, small differences cause huge performance variation.

Triton effectively gives us Tiers 1-3.5. CUDA gives all 6 tiers — but the agent can't reliably use Tiers 4-6. Having knobs the agent can't turn wastes search budget: a failed Tier 5 CUDA attempt costs a full iteration while a successful Tier 2 Triton attempt adds a real tree node.

**Agent success rate matters more than peak performance ceiling.** KernelEvolve (Meta) validates this: uses Triton, achieves 100% pass rate on KernelBench, works cross-hardware. Tiers 1-3 already yield 10-50%+ gains for most kernels — sufficient to prove the ACTS architecture.

**Known limitation**: V1 cannot compete with hand-tuned libraries (cuBLAS, cuDNN, FlashAttention) on kernels requiring warp specialization or architecture-specific intrinsics. Deliberate tradeoff — prove framework first, chase peak performance later.

---

## Development Process

### Always-runnable framework

**Rationale**: Prevents the common failure mode of building a large codebase that doesn't run until everything is done. By keeping the framework complete-but-shallow, we test pipeline flow early and catch integration issues before investing in deep implementation.

### Logger system before first live GPU run (2026-04-23)

**Context**: The first multi-minute live run was about to kick off with zero progress signal — every `logger.info`/`logger.warning` was silently dropped (no `basicConfig`), reducing post-mortem to a single final exception line. Wrong forensic surface for a run spanning many LLM calls and GPU subprocesses.

**Three sinks, not one**: `run.log` (human tail-able), `events.jsonl` (structured snapshots for tooling/ablation), `traces/*.jsonl` (per-call SDK records, reusing `JSONLTraceProcessor` from `5281cdf`). Each answers a different question; collapsing them would force each consumer to reparse another's format.

**Coder event truthfulness** (Codex adversarial review catch): originally emitted `coder_compiled(passed=True)` + `coder_correctness(passed=True)` on `implement()` return, but the orchestrator cannot verify those gates from the return value — the SDK's `submit_kernel` validates the structured output, not the gates. Changed to `coder_submitted` (no pass claim) and `coder_failed(reason)`; per-tool-call detail lives in `traces/*.jsonl`.

**Microsecond-precision run-dir names**: second-precision collides when ablation scripts or CI jobs share `--run-dir`. Same format now used by `trace_processor.py`, consolidated via `src/runtime/timefmt.py::filename_ts`.

**RunContext owns trace wiring**: post-review refactor removed `main()`'s `explicit_trace_processor` + `_enable_traces_if_possible` helper. `RunContext.create(trace_dir=...)` now owns default and override paths.

**Deliberately out of scope (v1)**: no Rich/tqdm live terminal UI (plain stdlib + `jq` is enough), no log rotation / disk quota / size caps (one run ≈ a few MB), no remote log shipping (Loki / Datadog), no "resume a run into the same run-dir" (new `main()` always creates a fresh `run_<UTC>/` — resume is a checkpoint concern, not a logger one), no cross-run aggregation index, no per-agent sub-loggers beyond stdlib `getLogger(__name__)`. Revisit triggers: live UX pain during multi-hour batches (→ Rich), disk pressure on long CI (→ rotation), need to compare runs (→ index).

### Correctness tolerance — adopt SOL-ExecBench's defaults verbatim (2026-04-26)

**Context**: First successful logger run against `examples/triton/rmsnorm/` exposed that the Coder was producing structurally correct bf16 RMSNorm kernels — compile passing, math right — that all failed correctness with `max_abs ≈ 7.812e-3` on workload 2/3. That value is exactly `2^-7`, the bf16 ULP at unit magnitude. Our `verify_correctness` defaults (`atol=rtol=1e-3`, `required_matched_ratio=1.0` hardcoded in `TorchComparisonPolicy.compare`) sat *below* bf16's quantization noise floor, making the acceptance test mathematically unsatisfiable for the dtype. The Coder kept iterating until the turn budget ran out, producing the misleading symptom `MaxTurnsExceeded`.

**Decision**: align with SOL-ExecBench's `ToleranceSpec` defaults verbatim — `max_atol=max_rtol=1e-2`, `required_matched_ratio=0.99`. Three reasons:

1. **The bar SOL ships is the bar SOL expects to be tested at.** Our ACTS gate evaluates SOL problems; using a stricter bar than SOL itself rejects kernels SOL would accept and silently breaks the contract we're benchmarking against.
2. **bf16 ULP is a physical floor, not a tunable.** No amount of Coder iteration produces a bf16 output closer to the fp32 reference than ~7.8e-3 at magnitude 1. Tightening below this floor is asking for the impossible.
3. **The 1% slack absorbs outliers, not bugs.** SOL's `required_matched_ratio=0.99` lets ~1% of elements fail the per-element bound while still passing overall. The hard `max_error_cap` in `compute_error_stats` would still catch a kernel with rare catastrophic outliers, so the slack doesn't unsafe the gate — it just stops it from false-flagging fp32→bf16 round-trip noise.

**Implementation**: `TorchComparisonPolicy.compare` no longer overrides `required_matched_ratio` — passes `ToleranceSpec(max_atol=atol, max_rtol=rtol)` and lets SOL's default kick in. Single source of truth, zero literal `0.99` in our code. `verify_correctness` defaults `atol=rtol=1e-2` mirror SOL's `max_atol=max_rtol`. Anti-cheat (stage 5) keeps its independent `strict_atol=1e-5, strict_rtol=1e-4` — that gate is ours, not SOL's, and serves a different threat model (reward hacking under randomized inputs).

**Drift sentinel**: `tests/test_correctness.py::test_verify_correctness_atol_rtol_defaults_match_sol_execbench` reads `ToleranceSpec()` defaults at runtime and asserts the function signature defaults match. If SOL bumps to e.g. `1.5e-2`, the test fails and forces an update. Test skips gracefully when `sol_execbench` isn't importable (tier-1 venv).

**What's NOT in scope**: dtype-aware tolerance table (e.g., bf16→1e-2, fp16→5e-3, fp32→1e-4) — premature; SOL itself didn't bother and treats one set of defaults as universal. Per-problem `tolerance` overrides — schema-supported by SOL's `Workload.tolerance` field but never exercised in any shipped example, so plumbing it through buys nothing today. Loosening the anti-cheat strict tolerances — those are an independent gate and the previous strict values still match how the stage is documented in PROCESS / doc/eval.

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
