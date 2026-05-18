# ACTS — Research Goals

## Primary research target

**Derive an agent-system design methodology**: given a fixed resource budget,
how should the tradeoffs across the agents in a multi-agent system be made to
achieve optimal outputs? The goal is prescriptive — to teach practitioners how
to build an agent system once they know their budget.

The kernel-optimization setting (ACTS: Planner → Coder → Reviewer over a search
tree) is the empirical vehicle. The methodology should generalize beyond
kernels: the contribution is "given budget B, here is how to think about
allocating it across roles in your agent system," not "here is the best
ACTS configuration."

## Why this framing

Existing agent-system papers typically report a single operating point
("our system uses N LLM calls and achieves X"). They do not characterize:

- How performance degrades as the budget shrinks
- How performance shifts when budget is reallocated between roles at fixed total
- Whether the optimal allocation depends on the input class (e.g.,
  memory-bound vs. compute-bound kernels)
- Whether there is a transferable rule of thumb a practitioner can apply
  before running their own sweep

A methodology that answers these — even partially — is what the field is
missing.

## Resource axes under consideration

The methodology teaches practitioners how to split budget *across agents*. The
right primary distinction is therefore between **per-agent allocatable**
resources (the actual tradeoff levers) and **system-level constraints** (the
total under which the allocation is optimized). The first experiment fixes
one system-level constraint and sweeps a per-agent allocation.

### Per-agent allocatable — the tradeoff space

These are the knobs a practitioner turns to split budget across roles. They
are the methodology's primary design surface.

1. **LLM calls per agent** — how many invocations Planner / Coder / Reviewer
   each receive within a run. ACTS today: `max_depth` × per-agent participation
   per iteration; reallocation is straightforward.
2. **Context tokens per agent** — how much each agent sees in its prompt.
   A core agent-system design choice ("what does this role need to know?"),
   not a slippery parameter — the implementation friction in ACTS is an
   experiment concern, not a methodology concern.
3. **Turns per agent** (`max_turns`) — how long each agent can deliberate
   per invocation. Distinct from total calls: a Reviewer with 6 turns
   thinks differently than one with 2, even at the same call count.
4. **Retries per agent** — when an agent has a verifiable submission
   (Coder's compile / correctness gate). Captures "how forgiving is this
   role's success criterion?"
5. **Retrieval depth per agent** — if the agent does RAG / memory lookup
   (`optimization_memory_top_k` in ACTS). Narrow but real lever.

### System-level constraints — the total to allocate under

These are the totals a practitioner is given (or chooses as the deployment
target). The methodology's job is to allocate the per-agent budgets above
optimally *given* one of these constraints.

1. **Total LLM calls** — scalar, cleanest experimental constraint.
2. **Total cumulative tokens (input + output)** — closer to actual $$ than
   call count; matters when call sizes vary dramatically across roles.
3. **Total verifier / oracle calls** — count of external-verification cycles.
   ACTS's GPU evaluation budget is one instance; other agent systems
   instantiate this as human-in-the-loop checks, external graders, or API
   verifiers. Framing it generally is what lets the methodology travel
   beyond kernels.
4. **Total exploration budget** — tree-node / candidate count. Conditional:
   only applies when the agent system is search-shaped. ACTS is; many agent
   systems aren't. The methodology should flag this as system-shape-dependent
   rather than universal.
5. **Total wall-clock** — operator-facing, deployment-realistic, but a
   *consequence* of the others rather than a clean design lever.

### First-experiment shape

Fix a system-level constraint (start with **total LLM calls**) and sweep a
per-agent allocation across Planner / Coder / Reviewer. This is the simplest
possible instantiation of the methodology question.

## Open design questions

Recorded for the brainstorming pass that produces the first experiment spec:

- Which experiment shape best serves the methodology goal?
  - (A) ACTS-vs-baseline at fixed small budgets
  - (B) Fix total, vary split across Planner / Coder / Reviewer (Pareto)
  - (C) Factorial: problem class × allocation, look for interaction effects
  - (D) Diminishing-returns curve on one problem
- Which baseline(s) does the methodology need to beat or contextualize against?
  (e.g., one-shot Coder, no-Reviewer, no-Retriever ablations)
- What does "optimal output" mean for the methodology paper —
  best SOL-score? best speedup? area under the budget–score curve?
- Which problem set covers enough diversity to support a generalization
  claim without exploding cost? (SOL-ExecBench subset, balanced across
  memory-bound / compute-bound / mixed?)

## Candidate benchmarks for first experiments

Selected from SOL-ExecBench (`repo/benchmark/SOL-ExecBench/data/benchmark/`)
for the first budget-allocation sweeps. Selection criteria:

- **Forward-only** — backward-kernel SOLAR support is on PROCESS.md's Active
  queue, not yet shipped.
- **BF16 / FP32 only** — dev host is RTX 6000 Ada (Ada Lovelace); no native
  NVFP4 and limited FP8 tensor cores. Quant tier (33 problems) is skipped.
- **Max workload ≫ launch overhead** — at least one workload per problem
  delivers tens to hundreds of GFLOPs / GB so kernel runtime ≫ ~10 µs
  launch latency. ACTS's per-iter timing signal stays well above noise.
- **Known optimization headroom** — mixed compute + memory or fusion-rich
  patterns where the agentic baseline typically sits below SOL (paper §1
  reports median SOL score 0.732 — plenty of problems with room to grow).

### L1 — single-operation kernels

| Path (under `data/benchmark/`) | Why | Max-workload work | Optimization handles |
|---|---|---|---|
| `L1/048_fused_gate_up_projection_with_swiglu` | Gemma3 fused MLP gate+up+SwiGLU. Pure compute-bound, two parallel BF16 matmuls 3072→24576. | bs=4, seq=2048 → ~1.2 TFLOPs MM | Fused gate/up GEMM, SwiGLU epilogue fusion, tiling, TF32/BF16 trade. |
| `L1/067_flash_attention_gqa_ultralong` | Nemotron-8B-UltraLong, seq up to 16384. Naive attention falls off a cliff — flash is the canonical win. Largest expected SOL gap. | bs=1, seq=16384, hidden=4096, 32Q/8KV | Flash tiling, online softmax, causal short-circuit, KV repeat fusion. |
| `L1/092_gqa_attention_with_qk_norm` | GLM-4.5-Air GQA with extreme 96Q/8KV (12× repeat) + QK RMSNorm + RoPE. Multi-knob attention block. | bs=32, seq=256, hidden=4096 → ~30+ GFLOPs MM | QK-norm placement, GQA broadcast fusion, fused QKV GEMM. |
| `L1/074_fused_gated_mlp_silu` | Parakeet gated MLP at moderate scale (1024→4096). Same SwiGLU pattern as #1, ~10× smaller. | bs=16, seq=512 → ~140 GFLOPs | Same as #1; cross-scale comparison. |
| `L1/075_grouped_query_self_attention_with_rope` | Parakeet GQA self-attn, 16Q/4KV, hidden=1024. Smaller control attention workload. | bs=16, seq=512, hidden=1024 → ~25+ GFLOPs | Flash vs naive SDPA, RoPE precomp, output proj fusion. |

### L2 — multi-operation fused kernels

| Path (under `data/benchmark/`) | Why | Optimization handles |
|---|---|---|
| `L2/002_decoder_layer_full_block` | LLaMA-3 full decoder layer (RMSNorm + GQA + SwiGLU MLP + residuals). The canonical transformer step. | Every LLM-serving optimization that matters lives here: layer fusion, residual-in-norm, GEMM tiling, attention variants. |
| `L2/019_decoder_layer_fused_attention_mlp` | Qwen2VL decoder with multimodal 3D RoPE + 28Q/4KV GQA. Like above + extra fusion surface. | 3D RoPE fusion, GQA broadcast, SwiGLU epilogue. |
| `L2/062_decoder_complete_layer` | Canary-Qwen-2.5B with self-attn + **cross-attn** + MLP in one layer. Cross-attention is a fusion surface L1 doesn't expose. | KV-cache reuse, cross-attn K/V projection fusion, dual-RMSNorm placement. |
| `L2/070_basic_transformer_block` | SDXL Refiner UNet BasicTransformerBlock: self-attn + cross-attn + GEGLU. Diffusion workload pattern (spatial 2D) — different shape regime. | Cross-attn text conditioning, GEGLU vs SwiGLU activation, spatial-vs-sequence axis trade-offs. |
| `L2/082_moe_layer_complete_forward_with_residual` | Complete MoE layer (sparse routing + per-expert MLP + weighted combine). Distinct optimization domain from dense decoders. | Expert dispatch (batched vs scattered), routing softmax fusion, expert-parallel layout. |

### Caveats

- **Triton-only Coder.** MoE sparse dispatch (L2 #5) and `cu_seqlens` patterns
  are genuinely hard to express in Triton — expect more iter-0 baseline-gen
  failures. Fallback substitute for L2 #5 if instability dominates:
  `L2/059_decoder_layer_full_block` (another dense decoder, different model).
- **Per-problem SOL headroom is only visible after the iter-0 baseline.**
  Triage rule: anything with baseline SOL ≤ 0.5 (read from `report.txt`)
  has clear ACTS room. Use a single cheap pass per candidate to filter.
- **L2 is 3–10× heavier than L1** (paper §Table 2 caption). Wallclock-budget
  the L2 sweeps accordingly.
- **Per-workload variance.** Each spec carries ~16 dynamically-shaped
  workloads; the representative workload (first in `workload.jsonl`) is
  usually small-batch and may understate the win on bigger shapes. Phase C
  re-profiles the winner against every workload — read the per-workload
  speedup table to confirm the optimization generalized.

### Suggested order to try

For clearest first-pass signal:
1. `L1/048_fused_gate_up_projection_with_swiglu` — textbook SwiGLU+tiling win.
2. `L1/067_flash_attention_gqa_ultralong` — dramatic flash-attention before/after.
3. Remaining L1 picks once the Tier-2 venv + clock-lock + NCU pipeline is trusted.
4. L2 picks only after a clean L1 round — they're slower to debug.

## Capability directions

Directions the framework itself could grow in to raise the ceiling on what
the agent system is *capable* of achieving — orthogonal to the
budget-allocation methodology above. Each entry names a gap observed in
practice and the framework response it argues for.

1. **Global vision over local-knob optimization.** A kernel-level win on
   one knob — better coalescing, higher SM occupancy, fewer warp stalls
   — does **not** necessarily translate into an end-to-end kernel
   performance advance. A change that improves occupancy can spill more
   registers, evict L2 lines another fused stage depends on, or shift
   the bottleneck from compute to memory (or vice versa) without
   improving wallclock. The framework should teach the LLM to reason
   *globally* about the kernel: how a local change propagates through
   the rest of the program, which bottleneck it moves the kernel
   toward/away from, and whether the resulting operating point is
   actually closer to SOL. Concretely this means the Planner / Reviewer
   contract should surface bottleneck-shift evidence (not just
   per-iter metric deltas) and the prompt scaffolding should pose the
   optimization decision as "where does this move us in the
   compute-vs-memory plane and is that the right direction," not "is
   metric X higher than before."

2. **Failure patterns as a first-class search signal, not telemetry exhaust.**
   In the standard search-tree contract, an iteration is a unit that
   either produces a scored child (signal: SOL delta, bottleneck
   classification, Reviewer prose) or is dropped (signal: a line in
   `events.jsonl`). Failed candidates carry no weight in the Planner's
   next decision. This is empirically wrong: a postmortem on
   `run_20260517T044132_970459Z` (JOURNAL → Search → "Failure-node
   retention") found 4 consecutive Coder candidates at the same parent
   failing with the same `cudaErrorInvalidAddressSpace` during autotune
   burn-in — the same systematic failure class, repeated four times,
   because each new Planner call saw zero evidence the prior 4 had
   failed. The information content of "this branch is uncodable in this
   way" was generated and discarded four times. The framework response
   it argues for: **failed candidates persist as tree artifacts carrying
   a typed failure class**, and the Planner consumes them on equal
   footing with successful siblings. The methodology claim this surfaces
   is sharper than it looks — failure patterns are the *cheapest*
   high-signal observation an agent system produces (no profile run, no
   Reviewer LLM call, no correctness pass) and discarding them is
   wasting the most cost-efficient evidence the system generates.
   Concretely: (a) failed candidates produce non-expandable tree nodes
   carrying `(action, params, failure_class, turns_used)`; (b) the
   sibling-context render that A2 introduced is extended to include them
   in a parallel format to successful siblings; (c) "dead branch" detection
   becomes a property of the sibling list (N consecutive same-class
   failures at one parent), not a separate orchestrator counter; (d) the
   Planner→Coder contract widens so that Planner-level decisions
   informed by failure history (e.g., narrowing an autotune config grid
   after repeated `autotune_launch_fault`) flow into the next Coder
   plan's `params`. The general methodology lesson: in any search-shaped
   agent system, the *failure-signal-to-cost ratio* is a budget-allocation
   primitive comparable to per-agent context tokens or retrieval depth —
   surfacing failures cheaply and routing them into the next decision is
   a lever, not housekeeping.

## Status

Brainstorming in progress. Next step: pick the experiment shape, then write
the experiment spec under `doc/specs/`.
