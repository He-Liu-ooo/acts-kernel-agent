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

**Retry budget**: Coder gets `max_debug_retries` self-correction attempts per iteration. If exhausted, the branch is marked dead. *Current implementation note (2026-04-18)*: `CoderAgent` hardcodes `_MAX_TURNS = 7` (= 2×3 + 1, derived from 3 compile+correctness tries). `max_debug_retries` from `ACTSConfig` is not yet read by the Coder; wiring is deferred to the same increment that turns the compile/correctness tool stubs into real calls.

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

### Coder: Pydantic output_type, tool placeholders, explicit failure contract (2026-04-18)

**Rationale**: Mirrored the Planner/Reviewer Pydantic structured-output pattern — `KernelCodeOutput(source_code: str)` is sent to the SDK via `output_type`, the tool loop is bounded by `max_turns`, and schema enforcement happens via constrained decoding. One field, no internal dataclass — the kernel source is the only thing the rest of the pipeline needs, so adding a translation layer would be cosmetic overhead.

**Tool placeholders, not scaffolding**: `compile_kernel_tool` and `check_correctness_tool` are module-level stubs returning success strings today. `@function_tool` wrapping, the Astra error-string pattern, and the Coder's workflow prompt are all real — only the tool bodies are stubbed. This lets the Coder land independently of `kernels/compiler.py` and `eval/correctness.py`, and the wiring work is mechanical when those modules arrive.

**Turn budget — `_MAX_TURNS = 7`, hardcoded for now**: derivation is 3 compile+correctness tries × 2 tool turns per cycle + 1 final structured-output turn. User framing: "3 tries means code can fail 2 times" — the third attempt must pass or the agent emits its best compiling effort. `ACTSConfig.max_debug_retries=3` already captures the user-facing concept but is not yet read by the Coder; wiring is deferred until the compiler/correctness integration (same increment that turns the tool stubs real). Recorded in `PROCESS.md` → Deferred Improvements.

**Failure contract — one sanctioned output in every case**:
- `run_agent()` returns `None` (transient retry exhaustion) → `implement()` raises `ImplementationError`.
- SDK tool loop hits `_MAX_TURNS` without a green correctness run → the prompt instructs the model to emit "the last version that compiled cleanly" as `source_code`. This is the *only* legal failure output, aligned explicitly with the `KernelCodeOutput` schema (which has no rationale field) and the hard rule that forbids emitting sources that were never compiled. No rationale side-channel, no multi-field schema, no prose stuffed into `source_code`.
- Without a model configured → returns the source unchanged.

Orchestrator-side handling of `ImplementationError` / `MaxTurnsExceeded` is deferred (same Deferred Improvements entry). Today these propagate and unwind the search run — acceptable while the Coder runs behind tool stubs, unacceptable once real failures are possible.

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

### Dynamic bottleneck reclassification — deferred to profiler implementation (2026-04-15)

**Context**: The orchestrator currently computes bottleneck classification once from the baseline roofline (via SOLAR) and reuses it for all iterations. This is correct for the skeleton phase — `profiler.py` is a placeholder returning zeros. However, the PRD specifies two bottleneck sources:

- **Static** (SOLAR, once at problem load): Is the *problem* fundamentally compute-bound or memory-bound?
- **Dynamic** (NCU profiling, each iteration): Is the *current candidate kernel* compute-bound or memory-bound?

Optimizations can shift a kernel's bottleneck (e.g., memory optimization moves it from memory-bound to compute-bound). When the real NCU profiler is implemented, the orchestrator loop should call `profile_kernel()` per candidate and pass the dynamic classification to memory retrieval, reviewer feedback, and planning. The static T_SOL remains constant — only the bottleneck classification updates.

**Decision**: Record and defer. No skeleton code change needed — would be routing placeholder data through a dynamic classification path. Wire when `profiler.py` gets real NCU integration.

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
