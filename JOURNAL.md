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

### Reviewer branch quality values

- `"promising"` — regression but underlying improvement visible (e.g., "memory traffic dropped 40%, one more fix should recover latency")
- `"blocked_potential"` — optimization is correct but benefit masked by a different bottleneck. E.g., memory optimization on compute-bound kernel shows no latency gain, but if compute bottleneck is resolved, the memory optimization would unlock 15-25% improvement. Must provide `conditional_assessment`: what follow-up action would unblock the potential.
- `"plateau"` — diminishing returns
- `"dead_end"` — fundamental mismatch, prune immediately

---

## Agents

### 4 LLM agents + deterministic orchestrator

**Rationale**: After analyzing AccelOpt (2-agent), STARK (3-agent), Astra (5-agent), we chose 4 agents to balance role specialization against communication overhead.

**Provider-agnostic**: The agent layer abstracts over LLM backends — no dependency on a specific provider's SDK.

### Why 4 and not 3 (merging Reviewer into Planner)

| Concern | 3-agent (merged) | 4-agent (separate Reviewer) |
|---------|------------------|-----------------------------|
| Planner prompt size | Large (profiling data + memory + action library + eval results) | Focused (memory + action library + Reviewer's distilled summary) |
| Auditability | Hard to tell if bad planning came from bad analysis or bad technique selection | Each agent's reasoning is isolated and inspectable |
| Model flexibility | Must use expensive model for both | Reviewer can use cheaper model |
| Extensibility | Adding future metrics requires changing Planner | Reviewer absorbs new metrics; Planner interface unchanged |

### Why 4 and not 5 (Astra-style)

Astra's Orchestrator agent is unreliable (better as deterministic code). Astra's separate Tester and Benchmarker are wasteful — correctness checking and benchmarking are deterministic operations that don't need LLM agents. Our eval harness runs these as code; the Reviewer interprets the results.

### Agent model choices

- *Planner*: Strongest reasoning model (planning quality is the bottleneck).
- *Coder*: Strong code model, speed matters (called every iteration).
- *Reviewer*: Can be cheaper model (analysis is easier than planning).
- *Debugger*: Strong reasoning model (debugging is hard).

### Future: context-adaptive agent specialization

From advisor discussion: agent specialization should be driven by LLM context window capacity as a finite resource. V1 uses 4 agents with large-context model. For smaller-context models, increase specialization:
- Large context (200K+): 4 agents
- Medium context (32-128K): 5-6 agents (Reviewer splits into Compute-Reviewer and Memory-Reviewer)
- Small context (8-32K): 7+ agents (further specialization, higher communication overhead)

**Hierarchical agent capabilities**: Upper-level agents (orchestrator, Planner) should be discriminative. Lower-level agents (Coder, Debugger) should be more capable with more tools.

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
