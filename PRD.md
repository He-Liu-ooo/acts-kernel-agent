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
- **Termination**: All frontier nodes marked dead_end, iteration budget exhausted, SOL score ≥ `sol_target`, or global plateau (best score stalled for `sol_plateau_window` iterations).
- **Single strategy**: Tree search with beam pruning only. No evolutionary fallback — keeps the search layer simple and debuggable.

---

## Agent Architecture — 3 LLM Agents + Deterministic Orchestrator

| Agent | Runs | Role |
|-------|------|------|
| **Planner** | Every iteration | Analyzes profiling data + optimization memory, selects technique from structured action library, produces structured plan `{tier, technique, params, target_region, rationale}` |
| **Coder** | Every iteration | Implements the plan into kernel code; one focused change per iteration. Has compile and correctness-check tools for self-correction within a retry budget. |
| **Reviewer** | Every iteration (after eval) | Interprets eval results, produces structured feedback `{outcome, metric_deltas, bottleneck_classification, bottleneck_diagnosis, suggestions, branch_quality, conditional_assessment}` |

Plus a deterministic orchestrator (code, not LLM) that manages tree state, beam selection, and move-on criteria.

**LLM SDK**: Agents are built on the OpenAI Agents SDK. The SDK provides the agent runtime (`Agent`, `Runner.run`, `function_tool`), structured output parsing via Pydantic `output_type`, and model-swapping via `OpenAIChatCompletionsModel` — any OpenAI-compatible API works by changing the base URL. The Coder uses SDK `function_tool` decorators for compile/correctness tools; Planner and Reviewer are single-call agents with no tools.

**LLM Backend**: Default model is **DeepSeek V3** for all three agents. Selection rationale:
- Triton/CUDA knowledge is strong and well-represented in pretraining data
- Reliable JSON mode for Pydantic structured output (critical for agent contracts)
- ~$0.27/1M input tokens — viable for 100+ planning iterations per kernel optimization
- Native OpenAI-compatible API — drops directly into `llm_backend.py` with zero adapter code
- Production-stable API with known reliability characteristics

**Evaluated alternatives**:
- *DeepSeek R1*: Stronger reasoning but 2x cost and chain-of-thought latency overhead. Overkill for technique selection; may be useful for Coder on complex kernel rewrites. Reserve for per-agent model specialization if V3 proves insufficient.
- *GLM-5.1 (Zhipu)*: Demonstrated kernel optimization capability (KernelBench L3: 3.6x geometric mean, 14h CUDA optimization reaching 35.7x speedup). SWE-bench Pro #1 (claimed). Open-source, self-hostable via vLLM. However: structured output reliability unverified, no production API pricing yet, new release without independent benchmarks. **Evaluate when API stabilizes** — kernel domain expertise may outperform DeepSeek V3 for the Coder agent.
- *Qwen2.5-Coder*: Good code generation, OpenAI-compatible. No differentiated kernel optimization capability.

**Model specialization** (future): `llm_backend.py` supports per-agent model configs. If evaluation shows benefit, use a stronger/domain-specialized model for Coder (where kernel expertise matters most) and a cheaper model for Planner/Reviewer (where structured output reliability matters most).

### Per-Iteration Communication Flow

```
Planner --> Coder (with compile/correctness tools) --> [deterministic eval] --> Reviewer --> Planner (next iter)
                  |                                |
                  +-- self-correction loop ---------+
                  (up to max_debug_retries attempts)
```

On compilation or correctness failure, the Coder's tool loop handles retries internally. If the retry budget is exhausted, the branch is marked dead — no separate Debugger agent.

### LLM Cost Estimate Per Iteration

| Agent | Calls/iter | Input tokens (est.) | Output tokens (est.) |
|-------|-----------|--------------------|--------------------|
| Planner | 1 | ~4K | ~500 |
| Coder | 1-3 (with tool-use self-correction) | ~3-6K | ~2-4K |
| Reviewer | 1 | ~2K | ~500 |
| **Total** | **3-5** | **~9-12K** | **~3-5K** |

~15K tokens per iteration. At beam width 3 and depth 20, ~900K tokens per kernel.

---

## Structured Action Library — 6-Tier System, Triton-First

Each action is a structured record: `{id, tier, name, description, applicable_to, preconditions, parameters, guidance, anti_patterns, expected_impact}`. Actions use high-level recipes (step-by-step guidance), not code templates — the Coder adapts the recipe to each kernel.

| Tier | Actions (examples) | Risk | Precondition |
|------|-------------------|------|-------------|
| 1 | block_size_tuning, grid_shape_optimization, occupancy_maximization | Low | None |
| 2 | shared_memory_tiling, global_memory_coalescing, register_caching, prefetching, bank_conflict_resolution | Low-Med | Memory bottleneck |
| 3 | tf32_accumulation, mixed_precision, fused_operations, vectorized_loads, loop_unrolling | Medium | Compute pattern |
| 4 | split_k_decomposition, persistent_kernel, warp_specialization, stream_k | High | Kernel structure |
| 5 | h100_tma_loads, h100_wgmma, a100_cp_async, hopper_cluster_launch | High | GPU arch |
| 6 | welford_online_stats, online_softmax, causal_mask_skip, flash_attention_tiling | High | Kernel type |

Tiers are not strictly sequential — Planner can pick any tier, but ordering encodes risk/reward.

### Spatial Grounding

Planner includes a `target_region` field — a natural language pointer to the code region the action should be applied to.

---

## Evaluation Harness — Correctness-First, Latency Profiling

Entirely deterministic (no LLM). Split across two call sites:

### Coder-Side Eval (via function_tools)

Compilation and correctness run inside the Coder's turn. The Coder calls these tools, sees errors, and self-corrects. By the time the Coder returns, the kernel is compiled and correct.

| Module | Called by | Purpose |
|--------|-----------|---------|
| `compiler.py` | Coder's `compile_kernel_tool` | Triton compilation |
| `correctness.py` + `anti_cheat.py` | Coder's `check_correctness_tool` | 5-stage correctness gate |

Note: `anti_cheat.py` has two surfaces. **Correctness-level** (above): randomized inputs, precision checks — runs inside the Coder's turn. **Performance-level**: `scorer.py` flags `reward_hack_suspect` when `T_k < T_SOL` — the orchestrator routes flagged candidates through additional anti-cheat inspection (see SOL Score Invariant Violations).

#### 5-Stage Correctness Gate

| Stage | What | Fail action |
|-------|------|------------|
| 1. Smoke test | Single input, check output matches baseline | Coder self-corrects |
| 2. Shape sweep | Multiple input sizes (tiny → xlarge) | Coder self-corrects |
| 3. Numerical stability | NaN/Inf detection, precision check | Coder self-corrects |
| 4. Determinism | Repeated runs must produce identical outputs | Coder self-corrects |
| 5. Anti-cheat | Randomized inputs, strict tolerance, no output caching | Coder self-corrects |

Any failure → Coder's tool loop retries (up to `max_debug_retries`). If budget exhausted, branch is marked dead. Fast-but-wrong kernels are never benchmarked.

### Problem-Load Eval (once per problem, Phase A)

Run once at startup before the search loop begins. Inputs are static (PyTorch reference + hardware config), so results are constant for the entire optimization.

| Module | Called by | Purpose |
|--------|-----------|---------|
| `roofline.py` | Orchestrator (Phase A) | SOLAR integration — derives T_SOL and initial bottleneck classification from PyTorch reference + hardware arch config |

### Orchestrator-Side Eval (after Coder returns, every iteration)

The orchestrator runs benchmarking and profiling on the Coder's output. These are never part of the Coder's tool loop — the Coder should not optimize for benchmark numbers directly.

| Module | Called by | Purpose |
|--------|-----------|---------|
| `benchmark.py` | Orchestrator | Latency measurement (CUDA events) |
| `profiler.py` | Orchestrator | Analytical roofline classification (every iter, free) + curated NCU section subprocess for occupancy/L2/TC/stall (every iter, representative workload); full-workload re-profile at Phase C. See JOURNAL "Profiler approach: analytical classification + curated NCU section (2026-04-20)". |
| `scorer.py` | Orchestrator | SOL score computation (using static T_SOL from roofline.py) |

| Metric | Tool | Method |
|--------|------|--------|
| **Latency** | CUDA Events | Median of N trials, 20 warmup + 100 timed |
| **Bottleneck classification** | Analytical (free) | `arithmetic_intensity vs ridge_point` from `(flops, bytes, measured latency, hardware_spec)`; yields `memory_bound` / `compute_bound` / `balanced` + achieved TFLOPs + achieved GB·s. Always populated. |
| **Hardware profiling** | NCU subprocess, curated sections (`Occupancy`, `WarpStateStats`, `MemoryWorkloadAnalysis`, `ComputeWorkloadAnalysis`) | SM occupancy, L2 hit rate, tensor-core utilization, dominant + runner-up warp stall class. Best-effort — NCU failures degrade the signal; analytical classification remains the floor. `ACTS_PROFILER_MODE=full` swaps to `--set full` for debug. |

### Profiling Feedback Pipeline

Full profiling → **Reviewer** → distilled summary → **Planner**. Reviewer acts as intelligent filter.

---

## Benchmark Source — SOL-ExecBench

ACTS uses SOL-ExecBench (NVIDIA, 2026) as its benchmark suite. SOL-ExecBench provides 235 CUDA kernel optimization problems extracted from 124 production AI models, organized into four categories (L1: single-op, L2: multi-op fused, Quant: FP8/NVFP4, FlashInfer-Bench: inference primitives). Each problem includes:

- **Definition** (`definition.json`): Problem name, input/output tensor shapes, dtypes, symbolic axes
- **Reference** (`reference.py`): PyTorch `run()` function — the ground-truth specification of the computation
- **Workloads** (`workload.jsonl`): 7-48 concrete shape instantiations per problem (varying batch size, sequence length, etc.)

### Triton Baseline Generation

SOL-ExecBench provides only PyTorch references. Since ACTS optimizes Triton code, each problem requires a Triton baseline as the root of the search tree. The Coder agent generates a one-shot PyTorch-to-Triton translation at problem load time:

1. Coder receives the PyTorch reference and problem definition
2. Coder produces a functionally equivalent Triton kernel
3. Correctness is verified against the PyTorch reference (same 5-stage gate)
4. If correctness fails, Coder retries (up to `max_baseline_retries` attempts)
5. If all retries fail, the problem is skipped

### Correctness Reference

The PyTorch reference is always the ground truth for correctness checking — both during baseline generation and during the optimization loop. The LLM-generated Triton baseline may have subtle numerical deviations; using it as correctness reference would propagate translation bugs as "correct" throughout optimization.

### SOL Score Baseline (T_b)

`T_b` is derived from the **Triton baseline**, not the PyTorch reference. The SOL score formula anchors S=0.5 at T_b, meaning "no improvement over starting point." Since ACTS optimizes Triton code, the meaningful zero-progress point is the Triton starting point. The SOL-ExecBench `sol_score.py` explicitly allows T_b to be set to any fast implementation.

`T_b` is measured once at problem load time with robust methodology (same warmup + timed iterations as candidate kernels, GPU clocks locked). It remains constant throughout the optimization search — recomputing T_b each iteration would introduce metric noise and break plateau detection.

---

## Roofline Model & Optimization Headroom

Existing benchmarks (e.g., KernelBench) measure speedup over a mutable software baseline — beating PyTorch eager tells you nothing about how close you are to hardware limits. ACTS uses a roofline-based approach to derive an absolute performance target and measure remaining optimization headroom.

### T_SOL Derivation via SOLAR

ACTS derives `T_SOL` using SOLAR (NVIDIA, 2026), a pipeline that analytically computes hardware-grounded SOL bounds from PyTorch programs. SOLAR operates in three stages:

1. **Graph Extractor**: Traces the PyTorch reference to produce an operator graph with tensor shapes and dtypes
2. **Agentic Einsum Converter**: Translates operators into extended einsum notation, deriving FLOP counts and memory traffic
3. **SOL Analyzer**: Computes roofline bound against target hardware architecture config

SOLAR produces three roofline models with progressively tighter bounds:
- **Unfused**: Each op in isolation, all tensors from DRAM
- **Fused**: Per-op roofline, intermediate tensors excluded from memory cost
- **Fused+Prefetched**: Single roofline for entire graph, perfect overlap assumed

ACTS uses the **fused** model as T_SOL. The fused_prefetched model assumes perfect overlap which is often unreachable in Triton; using it would make SOL scores pessimistic and trigger plateau detection prematurely.

`T_SOL` is the theoretical minimum runtime — no software implementation can run faster than this on the given hardware. It provides a fixed target independent of any software baseline.

### SOL Score

The SOL Score (SOL-ExecBench, NVIDIA 2026) measures how much of the baseline-to-hardware-limit gap a candidate kernel closes:

```
S(T_k) = (T_b - T_SOL) / ((T_k - T_SOL) + (T_b - T_SOL))
```

Where `T_b` = Triton baseline runtime, `T_SOL` = SOLAR-derived hardware limit, `T_k` = candidate kernel runtime.

| Condition | SOL Score | Meaning |
|-----------|-----------|---------|
| `T_k = T_b` | 0.5 | Matches Triton baseline (no improvement) |
| `T_k = T_SOL` | 1.0 | Reaches hardware Speed-of-Light |
| `T_k → ∞` | → 0 | Regression |

**Properties**:
- Bounded to [0, 1] under normal conditions — directly comparable across different kernels and problem sizes
- Nonlinear — the same ΔT yields a larger score gain near the SOL bound, rewarding diminishing-returns optimization
- Hardware-grounded — tells you *how much headroom remains* relative to physics, not relative to a mutable baseline

### SOL Score Invariant Violations — Audit Signals

The formula assumes `T_b > T_SOL` and `T_k ≥ T_SOL` (SOL-ExecBench paper, Section 4.3). When either assumption is violated, the scorer flags it as an audit signal rather than silently clamping:

| Violation | Flag | Score | Meaning |
|-----------|------|-------|---------|
| `T_k < T_SOL` | `reward_hack_suspect` | > 1.0 (raw, not clamped) | Candidate claims to beat hardware speed-of-light — almost certainly a measurement exploit (concurrency, caching, environment manipulation) |
| `T_b ≤ T_SOL` | `calibration_warning` | 1.0 | Baseline already at/below hardware limit — SOLAR bound may be too loose, or problem is already solved |

The `reward_hack_suspect` flag connects `scorer.py` to `anti_cheat.py` at the performance level — a second anti-cheat surface beyond the Coder-side correctness gate. The orchestrator should route flagged candidates through additional inspection before accepting.

### How Roofline Integrates into the Pipeline

1. **At startup**: `config.py` loads `HardwareSpec` from a SOLAR arch config YAML (or detects at runtime). ACTS and SOLAR share the same YAML schema.
2. **At problem load** (once): `roofline.py` runs SOLAR on the PyTorch reference + hardware arch config. Derives `T_SOL` and the run-level bottleneck classification via `classify_run`. Both are constant — the problem, representative workload, and hardware never change during optimization, so classifying per-iteration would only recompute the same answer.
3. **At each eval iteration**: `profiler.py` produces per-iter diagnostic signals — analytical roofline metrics (arithmetic intensity, achieved TFLOPS / GB/s, pct-of-peak) and curated NCU metrics (occupancy, L2 hit rate, tensor-core utilization, top-2 warp stalls). These refine the Reviewer's action-tier choice but do **not** re-classify the bottleneck. `scorer.py` computes SOL Score using the static `T_SOL` from step 2.
4. **Reviewer** receives the SOL score, the run-level bottleneck (threaded through from step 2), the per-iter analytical + NCU blocks, and how far `T_k` is from `T_SOL`. Reports remaining headroom.
5. **Planner** receives distilled summary: "SOL score = 0.72, compute-bound, 28% headroom remaining, tensor core utilization at 60%."
6. **Move-on criteria**: SOL score plateau (consecutive iterations with < δ improvement) or SOL score > threshold (e.g., 0.95 — within 5% of hardware limit).
7. **Cross-kernel comparability**: SOL score of 0.9 on matmul is directly comparable to 0.9 on softmax — both are 90% of the way to their respective hardware limits.

### Bottleneck Classification

Classification is once-per-run (via `classify_run` in `eval/roofline.py`) — invariant per `(problem, representative workload, hardware)`. The `BottleneckType` enum lives in `eval/types.py` so memory / search / pipeline can import it without pulling in the full roofline module:

| Classification | Condition | Primary Action Tiers |
|---------------|-----------|---------------------|
| Memory-bound | Arithmetic intensity < ridge point (outside balanced band) | Tier 2 (memory optimization) |
| Compute-bound | Arithmetic intensity > ridge point (outside balanced band) | Tier 3 (compute optimization) |
| Balanced | Near the ridge point | Either tier, guided by NCU sub-metrics |

The run-level label is threaded into retriever / planner / reviewer every iteration via `SearchResult.run_bottleneck` and surfaces in Phase C as `OptimizationReport.bottleneck`. A separate per-workload view — `OptimizationReport.winner_per_workload_bottlenecks`, populated by `classify_workload` — captures how individual workloads land relative to the ridge, which the single representative workload's label cannot show.

See JOURNAL → "Bottleneck classify-once (2026-04-22)" for why a per-iter dynamic reclassification (earlier design) was dropped.

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
    bottleneck_before: BottleneckType,  # run-level classification; invariant per run
    hardware: str,
    success: bool
}
```

No `bottleneck_after` — classification is once-per-run (invariant per `(problem, representative workload, hardware)`), so a pre/post pair would always carry the same value.

No kernel code stored — only summaries. Both successes and failures stored.

### Storage & Retrieval

- **Backend**: JSON files. Simple, git-friendly, human-readable. No database.
- **Retrieval**: (1) Filter by kernel type, (2) prefer same-hardware experiences (fall back to cross-hardware if insufficient), (3) score by bottleneck match + success + speedup, (4) select top-K with reserved failure slots so the Planner sees both what worked and what to avoid.
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

`HardwareSpec` uses the SOLAR arch YAML schema directly — both ACTS and SOLAR share the same hardware description format. This eliminates translation between two schemas and ensures roofline analysis uses consistent parameters.

1. **Load hardware spec at startup** — from a SOLAR arch config YAML (e.g., `configs/arch/H100_PCIe.yaml`, `configs/arch/B200.yaml`). The YAML provides per-cycle throughput by precision (MAC/cycle for FP32, BF16, FP8, etc.), memory hierarchy capacities, and clock frequency. Peak TFLOPS and bandwidth are derived properties.
2. **Use specs internally** — feed hardware spec to SOLAR for `T_SOL` derivation, and to the built-in roofline fallback for bottleneck classification. Compute SOL Score for each candidate kernel.
3. **Reviewer sees** profiling results + roofline classification + SOL score + remaining headroom.
4. **Planner sees** Reviewer's distilled summary only. Agents never see raw hardware specs.
5. **Fallback** — when no arch YAML is provided, `detect_hardware()` queries the CUDA runtime (placeholder in V1).

---

## Configuration

Run parameters are set through `.cfg` files (INI format, parsed via Python's `configparser`). Unspecified values fall back to built-in defaults. Hardware specs are loaded from a SOLAR arch YAML if `arch_config_path` is specified, otherwise detected at runtime.

```ini
[search]
beam_width = 3
beam_diversity = true
max_depth = 20
epsilon_start = 0.3
epsilon_end = 0.05

[eval]
warmup_runs = 20
timed_runs = 100

[move_on]
sol_plateau_window = 3
sol_plateau_delta = 0.01
sol_target = 0.95

[debug]
max_debug_retries = 3
max_baseline_retries = 3

[memory]
optimization_memory_top_k = 5

[benchmark]
benchmark_workload_count = 3

[hardware]
arch_config_path = configs/arch/H100_PCIe.yaml
```

---

## Pipeline Flow

```
Phase A: Load Problem
  SOL-ExecBench problem -> parse definition.json, reference.py, workload.jsonl
  -> derive T_SOL via SOLAR (PyTorch reference + hardware arch config)
  -> classify kernel as compute-bound or memory-bound
  -> generate Triton baseline via Coder (PyTorch -> Triton one-shot translation)
  -> verify Triton baseline correctness against PyTorch reference
     -> retry up to max_baseline_retries on failure; skip problem if exhausted
  -> measure T_b (Triton baseline latency, CUDA events, locked clocks)
  -> select representative workloads for iterative benchmarking (2-3 of 7-48)
  -> baseline SOL score = 0.5 by definition

Phase B: Search Loop (autonomous, 3-agent)
  orchestrator.py manages tree search. `run_bottleneck` (from Phase A
  `classify_run`) is threaded through every iteration — there is no
  per-iter re-classification, because `(flops, nbytes, hardware)` are
  invariant per run (see JOURNAL → "Bottleneck classify-once").
  -> Retrieve similar past optimizations from memory (filtered by run_bottleneck)
  -> PLANNER: profiling data + memory + run_bottleneck + feedback -> structured plan
  -> CODER (with tools): plan + kernel code -> compile -> correctness check
     -> correctness always checked against PyTorch reference on every selected workload
     -> self-correction loop on failure (up to max_debug_retries; SDK turn budget 2*N+1)
  -> [DETERMINISTIC EVAL]: benchmark (CUDA events) -> profiler (analytical roofline
     per-iter; curated NCU subprocess per-iter on representative workload) -> SOL score
  -> REVIEWER: eval results + SOL score + headroom + run_bottleneck + live
               ProfilingResult -> structured feedback + branch_quality
  -> Tree update: defer committing child.score + per_workload_latency_us until
               after the profile DEAD_END gauntlet clears; beam prune
  -> Memory update: store experience (including SOL score and run_bottleneck)
  -> Move-on criteria: SOL plateau, SOL >= sol_target, all-dead-end, or budget

Phase C: Report (autonomous)
  Best kernel selected from tree (highest SOL score)
  Run full workload suite on best kernel (all workloads, not just representative subset)
  Report: baseline vs best, SOL score progression, run-level bottleneck,
          per-workload bottlenecks, technique trace, remaining headroom
          to hardware limit
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
|   |   |-- reviewer.py
|   |   +-- llm_backend.py
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
|   |   |-- (power.py — V2, not in V1)
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
|       +-- debugger/           (reserved — Coder handles debugging via tools)
|
|-- benchmarks/
|   |-- sol_execbench/
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
| Benchmark | KernelBench PyTorch-to-CUDA | SOL-ExecBench — production AI model subgraphs, SOLAR-derived T_SOL |
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

### Parallel Kernel Candidate Generation

Generate multiple kernel candidates per iteration instead of one. The Coder produces N candidates from the same plan (varying implementation details), and all candidates are evaluated in parallel. The best-scoring candidate becomes the tree node. This trades LLM cost for search breadth — useful when a single Coder call has high variance in output quality.

### Multi-Technique Planning

Allow the Planner to select multiple optimization techniques per plan instead of exactly one. The current "one change at a time" constraint simplifies attribution (which technique helped?) but limits the search when techniques are complementary (e.g., shared memory tiling + prefetching). Multi-technique plans would require the Reviewer to decompose attribution across techniques, and the Coder to apply changes in a controlled sequence.

### Context-Adaptive Agent Specialization

Agent count adapts to LLM context window: 3 agents at 200K+, 5-6 at 32-128K, 7+ at 8-32K.

### Reviewer Knowledge Base Architecture

Three-tier KB: Compute-Reviewer KB, Memory-Reviewer KB, Shared Interaction KB. Two-dimensional retrieval: metric-triggered + action-triggered.

### Multi-Turn Reviewer with On-Demand Profiling Queries

Reviewer upgrades from single-call agent to tool-using agent (Coder-style) with bounded turn budget. Two query shapes: (A) lookup into `ProfilingResult.raw_metrics` for NCU metrics captured in the initial run but outside the curated subset (free, in-memory), and (B) on-demand re-profile with caller-specified `--section` / `--metrics` (expensive subprocess, cache-key expansion to include the metric set requested). Lets the Reviewer recover when the curated signals (occupancy, L2, tensor-core util, top-2 stalls) don't match the kernel's actual bottleneck signature. See JOURNAL → Agents → "Multi-turn Reviewer deferred" for why this is post-V1 and PROCESS.md Deferred Improvements for the trigger.
