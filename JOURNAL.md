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

### Serial beam expansion (2026-04-19, /simplify review)

**Rationale**: `Orchestrator.run()` expands one frontier node per iteration despite `beam_width ≥ 1`. Parallelizing via `asyncio.gather` across the top-k picks would amortize three sequential LLM calls (Planner → Coder → Reviewer) across k concurrent branches — the largest wallclock-latency win available. Deliberately deferred because three downstream components assume single-writer semantics on the tree:

- **`beam_prune`**: the diversity-aware pass (see B2 above) ranks the current frontier once per iteration. Concurrent expansion would either need a frontier-snapshot-per-worker (stale rankings) or a post-join re-prune (defeats the parallelism win for small k).
- **`MemoryStore.add()`**: today a single-file JSON rewrite per add. Concurrent writers would race on the file. The deferred "batched flush" improvement (see `PROCESS.md` → Deferred Improvements) is a prerequisite — not a blocker, but parallelism pulls it onto the critical path.
- **Checkpoint writes**: atomic temp-file + `os.replace` is correct for one writer; N writers racing on the same checkpoint path would corrupt recovery state even with atomic replace.

**Decision**: keep expansion serial until a real benchmark shows LLM latency is the dominant cost. At that point, design the change as a coordinated restructure — frontier snapshots + batched memory flush + per-worker checkpoint slots — rather than dropping `asyncio.gather` into the hot path. Recorded with its trigger in `PROCESS.md` → Deferred Improvements → "Parallel beam expansion via asyncio.gather".

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
1. Tier 1 (GPU-free, fake-`ncu`, 5 test files) passes in `/tmp/acts_test_venv`.
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

---

## Optimization Memory

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

---

## Benchmark & Scoring

### SOL-ExecBench as benchmark suite (over KernelBench) (2026-04-14)

**Rationale**: KernelBench (Ouyang et al., 2025) measures speedup over PyTorch eager — a mutable software baseline that tells nothing about proximity to hardware limits. A 10x speedup over PyTorch can still be 100x away from hardware SOL. SOL-ExecBench (NVIDIA, 2026) reframes evaluation around closing the gap to hardware Speed-of-Light, providing 235 problems from 124 production AI models across BF16/FP8/NVFP4 precisions with forward and backward passes.

### HardwareSpec uses SOLAR arch YAML schema directly (2026-04-15)

**Rationale**: SOLAR arch config YAMLs (e.g., `H100_PCIe.yaml`, `B200.yaml`) define hardware in roofline-oriented terms: per-cycle throughput by precision (MAC/cycle for FP32, BF16, FP8, NVFP4, etc.), SRAM/DRAM capacities and bandwidth, and clock frequency. Rather than maintaining a separate `HardwareSpec` schema and translating between the two, `HardwareSpec` uses SOLAR's schema directly. This means:

- `load_hardware_spec(path)` reads a SOLAR YAML into a `HardwareSpec`
- SOLAR's Python API and ACTS's built-in roofline both consume the same data
- Derived properties (`peak_flops_fp32`, `peak_memory_bandwidth_gb_s`) are computed from the raw per-cycle fields + frequency, matching the formulas in SOLAR's comments (e.g., `MAC_per_cycle_bf16_tc * freq_GHz * 2` = PFLOPS)

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
