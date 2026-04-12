# ACTS — Product Requirements Document

A framework for LLM-driven GPU kernel optimization that combines structured search, multi-agent coordination, and persistent optimization memory.

---

## Optimization Objective

**V1: Pure latency optimization only.** The sole metric is kernel execution time (μs), lower is better.

- **Evaluation harness**: Measures latency via CUDA Events + NCU hardware profiling. No power measurement in V1.
- **Beam scoring**: SOL Score — measures how much of the baseline-to-hardware-limit gap a candidate kernel closes. Range [0, 1].
- **Memory store**: Experiences record latency, speedup, and SOL score.
- **Move-on criteria**: SOL score plateau detection or SOL score approaching 1.0 (hardware limit reached).

---

## Search Strategy — Tree Search with Beam Pruning

Best-first tree search with beam constraint.

- **Structure**: Tree nodes = kernel versions, edges = optimization actions. Root = baseline kernel.
- **Selection**: Epsilon-greedy over frontier. With probability (1−ε) expand highest-scoring node; with probability ε expand a random node. Epsilon decays over iterations.
- **Parent retention**: Parent stays in frontier after expansion, enabling backtracking.
- **Child retention**: Children worse than their parent are kept by default. Regressed children are handled by: (1) score-based deprioritization, (2) beam constraint pruning, (3) Reviewer `branch_quality` override (`"promising"`, `"blocked_potential"`, `"plateau"`, `"dead_end"`).
- **Scoring**: SOL Score (see Roofline Model & Optimization Headroom section).
- **Termination**: All frontier nodes marked dead_end, iteration budget exhausted, SOL score approaching 1.0, or target speedup reached.
- **Single strategy**: Tree search with beam pruning only. No evolutionary fallback — keeps the search layer simple and debuggable.

---

## Agent Architecture — 4 LLM Agents + Deterministic Orchestrator

| Agent | Runs | Role |
|-------|------|------|
| **Planner** | Every iteration | Analyzes profiling data + optimization memory, selects technique from structured action library, produces structured plan `{tier, technique, params, target_region, rationale}` |
| **Coder** | Every iteration | Implements the plan into kernel code; one focused change per iteration |
| **Reviewer** | Every iteration (after eval) | Interprets eval results, produces structured feedback `{outcome, metric_deltas, bottleneck_classification, bottleneck_diagnosis, suggestions, branch_quality, conditional_assessment}` |
| **Debugger** | On failure only | Diagnoses compilation/correctness failures, produces fix plan for Coder; pruned after N retries |

Plus a deterministic orchestrator (code, not LLM) that manages tree state, beam selection, and move-on criteria.

**LLM provider**: Provider-agnostic. The agent layer abstracts over LLM backends — no dependency on a specific provider's SDK.

### Per-Iteration Communication Flow

```
Planner --> Coder --> [deterministic eval] --> Reviewer --> Planner (next iter)
                             |
                         (on failure)
                             |
                         Debugger --> Coder (retry)
```

### LLM Cost Estimate Per Iteration

| Agent | Calls/iter | Input tokens (est.) | Output tokens (est.) |
|-------|-----------|--------------------|--------------------|
| Planner | 1 | ~4K | ~500 |
| Coder | 1-2 (retry on debug) | ~3K | ~2K |
| Reviewer | 1 | ~2K | ~500 |
| Debugger | 0-1 (conditional) | ~3K | ~500 |
| **Total** | **3-5** | **~9-12K** | **~3-4K** |

~15K tokens per iteration. At beam width 3 and depth 20, ~900K tokens per kernel.

---

## Structured Action Library — 6-Tier System, Triton-First

Each action is a structured record: `{id, tier, name, description, applicable_to, preconditions, parameters, guidance, anti_patterns, expected_impact}`. Actions use high-level recipes (step-by-step guidance), not code templates — the Coder adapts the recipe to each kernel.

| Tier | Actions (examples) | Risk | Precondition |
|------|-------------------|------|-------------|
| 1 | block_size_tuning, grid_shape_optimization, occupancy_maximization | Low | None |
| 2 | shared_memory_tiling, global_memory_coalescing, register_caching, prefetching, bank_conflict_resolution | Low-Med | Memory bottleneck |
| 3 | tf32_accumulation, mixed_precision, fused_operations, vectorized_loads, loop_unrolling | Medium | Compute pattern |
| 4 | split_k_decomposition, persistent_kernel, warp_specialization, cooperative_groups, stream_k | High | Kernel structure |
| 5 | h100_tma_loads, h100_wgmma, a100_cp_async, hopper_cluster_launch | High | GPU arch |
| 6 | welford_online_stats, online_softmax, causal_mask_skip, flash_attention_tiling | High | Kernel type |

Tiers are not strictly sequential — Planner can pick any tier, but ordering encodes risk/reward.

### Spatial Grounding

Planner includes a `target_region` field — a natural language pointer to the code region the action should be applied to.

---

## Evaluation Harness — Correctness-First, Latency Profiling

Entirely deterministic (no LLM). Two stages:

### Stage 1: 5-Stage Correctness Gate

| Stage | What | Fail action |
|-------|------|------------|
| 1. Smoke test | Single input, check output matches baseline | Reject |
| 2. Shape sweep | Multiple input sizes (tiny → xlarge) | Reject |
| 3. Numerical stability | NaN/Inf detection, precision check | Reject |
| 4. Determinism | Repeated runs must produce identical outputs | Reject |
| 5. Anti-cheat | Randomized inputs, strict tolerance, no output caching | Reject |

Any failure → Debugger invoked. Fast-but-wrong kernels are never benchmarked.

### Stage 2: Latency + Hardware Profiling

| Metric | Tool | Method |
|--------|------|--------|
| **Latency** | CUDA Events | Median of N trials, 20 warmup + 100 timed |
| **Hardware profiling** | NCU (`ncu --set full`) | SM occupancy, memory throughput, compute throughput, cache hit rates, warp stall reasons |

### Profiling Feedback Pipeline

Full profiling → **Reviewer** → distilled summary → **Planner**. Reviewer acts as intelligent filter.

---

## Roofline Model & Optimization Headroom

Existing benchmarks (e.g., KernelBench) measure speedup over a mutable software baseline — beating PyTorch eager tells you nothing about how close you are to hardware limits. ACTS uses a roofline-based approach to derive an absolute performance target and measure remaining optimization headroom.

### Roofline Model

The roofline model (Williams et al., 2009) bounds achievable kernel performance by two hardware limits:

```
T_SOL = max(Total FLOPs / Peak Compute Throughput, Total Fused Bytes / Peak Memory Bandwidth)
```

- If `FLOPs / Throughput > Bytes / Bandwidth` → kernel is **compute-bound** (limited by ALU/tensor core throughput)
- If `Bytes / Bandwidth > FLOPs / Throughput` → kernel is **memory-bound** (limited by DRAM/cache bandwidth)

The crossover point is the hardware's **arithmetic intensity threshold** (ops/byte). Kernels above this threshold are compute-bound; below are memory-bound.

`T_SOL` is the theoretical minimum runtime — no software implementation can run faster than this on the given hardware. It provides a fixed target independent of any software baseline.

### Tighter Bounds via On-Chip Buffer Modeling

Naive roofline can overestimate achievable performance by assuming all data is reused from on-chip caches. ACTS uses Orogenesis-style analysis (Huang et al., 2024) to derive tighter bounds: model off-chip data movement as a function of on-chip buffer capacity, accounting for the reality that not all tensor data can be staged on-chip for full reuse. This prevents falsely classifying a memory-bound kernel as close to its SOL bound when cache capacity is the actual constraint.

### SOL Score

The SOL Score (SOL-ExecBench, NVIDIA 2026) measures how much of the baseline-to-hardware-limit gap a candidate kernel closes:

```
S(T_k) = (T_b - T_SOL) / ((T_k - T_SOL) + (T_b - T_SOL))
```

Where `T_b` = baseline runtime, `T_SOL` = roofline-derived hardware limit, `T_k` = candidate kernel runtime.

| Condition | SOL Score | Meaning |
|-----------|-----------|---------|
| `T_k = T_b` | 0.5 | Matches baseline (no improvement) |
| `T_k = T_SOL` | 1.0 | Reaches hardware Speed-of-Light |
| `T_k → ∞` | → 0 | Regression |

**Properties**:
- Bounded to [0, 1] — directly comparable across different kernels and problem sizes
- Nonlinear — the same ΔT yields a larger score gain near the SOL bound, rewarding diminishing-returns optimization
- Hardware-grounded — tells you *how much headroom remains* relative to physics, not relative to a mutable baseline

### How Roofline Integrates into the Pipeline

1. **At startup**: `config.py` detects hardware specs (peak FLOPS for FP16/BF16/FP32, peak memory bandwidth, cache sizes, SM count).
2. **At problem load**: `roofline.py` derives `T_SOL` for the kernel from its FLOP count, memory traffic, and hardware specs. Also classifies baseline as compute-bound or memory-bound.
3. **At each eval iteration**: `scorer.py` computes SOL Score for the candidate kernel. The bottleneck classification may shift (e.g., a memory optimization moves the kernel from memory-bound to compute-bound).
4. **Reviewer** receives the SOL score, bottleneck classification, and how far `T_k` is from `T_SOL`. Reports remaining headroom.
5. **Planner** receives distilled summary: "SOL score = 0.72, compute-bound, 28% headroom remaining, tensor core utilization at 60%."
6. **Move-on criteria**: SOL score plateau (consecutive iterations with < δ improvement) or SOL score > threshold (e.g., 0.95 — within 5% of hardware limit).
7. **Cross-kernel comparability**: SOL score of 0.9 on matmul is directly comparable to 0.9 on softmax — both are 90% of the way to their respective hardware limits.

### Bottleneck Classification

The roofline model drives action library tier selection:

| Classification | Condition | Primary Action Tiers |
|---------------|-----------|---------------------|
| Memory-bound | Arithmetic intensity < hardware threshold | Tier 2 (memory optimization) |
| Compute-bound | Arithmetic intensity > hardware threshold | Tier 3 (compute optimization) |
| Balanced | Near the ridge point | Either tier, guided by NCU sub-metrics |

The Reviewer tracks bottleneck transitions across iterations — a successful Tier 2 optimization can shift a kernel from memory-bound to compute-bound, signaling the Planner to switch to Tier 3 actions.

---

## Optimization Memory — Persistent Cross-Task Learning

### Experience Schema

```
Experience = {
    kernel_type: str,
    action_applied: ActionRecord,
    metrics: {latency, sol_score},
    speedup: float,
    reviewer_summary: str,
    bottleneck_before: str,
    bottleneck_after: str,
    hardware: str,
    success: bool
}
```

No kernel code stored — only summaries. Both successes and failures stored.

### Storage & Retrieval

- **Backend**: JSON files. Simple, git-friendly, human-readable. No database.
- **Retrieval**: (1) Filter by kernel type, (2) rank by bottleneck relevance, (3) select top-K (3-5 experiences).
- **Injection**: Planner only, contrastive summary format.

### Relationship to Search Tree

| | Search Tree | Optimization Memory |
|--|-------------|-------------------|
| Scope | Intra-task (one kernel) | Inter-task (all past kernels) |
| Lifetime | Task start → task end | Permanent |
| Granularity | Full state per node | Distilled summary |
| Consumer | Orchestrator | Planner |

---

## Backend — Triton (V1)

V1 uses Triton as the sole backend. Target hardware: NVIDIA GPUs.

Triton coverage by tier:

| Tier | Coverage |
|------|----------|
| 1: Block/grid sizing | Full |
| 2: Memory | Partial (coalescing automatic, num_stages for pipelining, no bank conflict control) |
| 3: Compute | Mostly full |
| 4: Advanced | Partial (split-K doable, persistent kernels awkward, warp specialization impossible) |
| 5: Arch-specific | Mostly blocked |
| 6: Kernel-specific | Partial |

**Known limitation**: V1 cannot compete with hand-tuned libraries on kernels requiring warp specialization or architecture-specific intrinsics.

---

## Hardware Specification Handling

1. **Detect hardware at startup** — GPU name, SM count, peak FLOPS (FP16/BF16/FP32), peak memory bandwidth, cache sizes, compute capability.
2. **Use specs internally** — feed into roofline model to derive `T_SOL` bound and bottleneck classification. Compute SOL Score for each candidate kernel.
3. **Reviewer sees** profiling results + roofline classification + SOL score + remaining headroom.
4. **Planner sees** Reviewer's distilled summary only. Agents never see raw hardware specs.

---

## Pipeline Flow

```
Phase A: Load Problem (lightweight)
  KernelBench problem -> load baseline kernel
  -> compile & verify baseline correctness
  -> baseline benchmark (latency)
  -> derive T_SOL via roofline model (FLOP count, memory traffic, hardware specs)
  -> classify baseline as compute-bound or memory-bound
  -> compute baseline SOL score (= 0.5 by definition)

Phase B: Search Loop (autonomous, 4-agent)
  orchestrator.py manages tree search:
  -> Retrieve similar past optimizations from memory
  -> PLANNER: profiling data + memory + feedback -> structured plan
  -> CODER: plan + kernel code -> optimized kernel
  -> [DETERMINISTIC EVAL]: compile -> 5-stage correctness -> benchmark
     -> NCU profiler -> roofline classification -> SOL score
  -> (on failure) DEBUGGER: error + code -> diagnosis -> re-invoke CODER
  -> REVIEWER: eval results + SOL score + headroom -> structured feedback + branch_quality
  -> Tree update: add node, score by SOL score, beam prune
  -> Memory update: store experience (including SOL score)
  -> Move-on criteria: SOL plateau or SOL > 0.95

Phase C: Report (autonomous)
  Best kernel selected from tree (highest SOL score)
  Report: baseline vs best, SOL score progression, bottleneck transitions,
          technique trace, remaining headroom to hardware limit
```

---

## Directory Structure

```
acts-kernel-agent/
|-- PRD.md
|-- JOURNAL.md
|-- PROCESS.md
|-- INSIGHTS.md
|-- pyproject.toml
|
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |
|   |-- agents/
|   |   |-- __init__.py
|   |   |-- planner.py
|   |   |-- coder.py
|   |   +-- evaluator.py
|   |
|   |-- search/
|   |   |-- __init__.py
|   |   |-- tree.py
|   |   |-- beam.py
|   |   +-- orchestrator.py
|   |
|   |-- eval/
|   |   |-- __init__.py
|   |   |-- correctness.py
|   |   |-- benchmark.py
|   |   |-- power.py
|   |   |-- profiler.py
|   |   |-- roofline.py
|   |   |-- scorer.py
|   |   +-- anti_cheat.py
|   |
|   |-- kernels/
|   |   |-- __init__.py
|   |   |-- kernel.py
|   |   |-- compiler.py
|   |   +-- starters/
|   |       |-- matmul.py
|   |       |-- softmax.py
|   |       |-- layernorm.py
|   |       |-- attention.py
|   |       +-- ...
|   |
|   |-- actions/
|   |   |-- __init__.py
|   |   |-- registry.py
|   |   |-- tier1_sizing.py
|   |   |-- tier2_memory.py
|   |   |-- tier3_compute.py
|   |   |-- tier4_advanced.py
|   |   |-- tier5_arch.py
|   |   +-- tier6_specific.py
|   |
|   |-- memory/
|   |   |-- __init__.py
|   |   |-- store.py
|   |   |-- retriever.py
|   |   +-- experience.py
|   |
|   |-- pipeline/
|   |   |-- __init__.py
|   |   |-- optimize.py
|   |   |-- verify.py
|   |   +-- report.py
|   |
|   +-- prompts/
|       |-- planner/
|       |   |-- system.md
|       |   +-- technique_select.md
|       |-- coder/
|       |   |-- system.md
|       |   +-- implement.md
|       |-- reviewer/
|       |   |-- system.md
|       |   +-- interpret.md
|       +-- debugger/
|           |-- system.md
|           +-- diagnose.md
|
|-- benchmarks/
|   |-- kernelbench/
|   +-- custom/
|
+-- tests/
    |-- test_correctness.py
    |-- test_search.py
    |-- test_memory.py
    +-- ...
```

---

## Development Constraint: Always-Runnable Framework

The framework must remain runnable at every development iteration. Unimplemented modules use placeholders so the full pipeline can execute end-to-end at all times.

- Iteration 0: All modules are placeholders, but `python -m src.pipeline.optimize` runs
- Iteration 1: Real eval/correctness.py, everything else placeholder
- Iteration 2: Real agents/coder.py, everything else placeholder or previously implemented
- ...each iteration deepening one module

---

## Differentiators vs. Reference Repos

| Aspect | Reference Repos | ACTS |
|--------|----------------|------|
| Search | Linear iteration or full evolutionary | Tree search + beam pruning |
| Actions | Free-form prompts or implicit | Explicit tiered action library |
| Memory | Per-run only or training data | Persistent cross-run optimization memory |
| Eval | Correctness only or no anti-cheat | 5-stage + anti-cheat + NCU profiling + roofline |
| Scoring | Relative speedup over software baseline | SOL Score — absolute headroom vs. hardware limit |
| Objective | Latency only | V1: latency. Interface reserves power/ELP |
| Orchestration | LLM-based or simple loop | Deterministic tree search |
| Extensibility | Monolithic scripts | SDK-style with clean abstractions |

---

## Future Directions

### Multi-Objective Optimization & Power Profiling

| Mode | Metric | Scoring | Use Case |
|------|--------|---------|----------|
| Pure Latency (V1) | Execution time (μs) | Lower is better | Inference serving |
| Pure Power | GPU power draw (W) via NVML | Lower is better | Edge deployment |
| Energy-Latency Product | Power × Latency² (J·s) | Lower is better | Data center efficiency |

### Action Library Extensions

- Code templates for tier 4-6 actions
- Free-form escape hatch for novel techniques
- CUDA C++ backend for full tier coverage

### Optimization Memory Extensions

- Embedding-based retrieval for cross-type transfer
- SQLite backend for scale

### Backend Alternatives

1. CUDA C++ (V2) — full tier 4-6 coverage
2. TileLang — tile-centric model
3. CuteDSL — near-PTX performance

### Context-Adaptive Agent Specialization

Agent count adapts to LLM context window: 4 agents at 200K+, 5-6 at 32-128K, 7+ at 8-32K.

### Reviewer Knowledge Base Architecture

Three-tier KB: Compute-Reviewer KB, Memory-Reviewer KB, Shared Interaction KB. Two-dimensional retrieval: metric-triggered + action-triggered.
