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
- **Child retention**: Children worse than their parent are kept by default. Regressed children are handled by: (1) score-based deprioritization, (2) beam constraint pruning, (3) Reviewer `branch_quality` override (`"promising"`, `"blocked_potential"`, `"plateau"`, `"dead_end"`). K-way Coder/bench-layer failures collapse into ONE failure-summary node per iter (no score, `dead_reason=CODER_FAILED`, `kernel=None`) carrying a `failure_details: list[FailureDetail]` with one entry per failed candidate; the Planner sees prior failed attempts via `render_siblings`' FAILED block, which flattens summaries' `failure_details` into the existing dedup-on-`(action, params, reason)` + `×N` + `failure_sibling_cap` (default 8) pipeline.
- **Scoring**: SOL Score (see Roofline Model & Optimization Headroom section).
- **Termination**: All frontier nodes marked dead_end, iteration budget exhausted, SOL score ≥ `sol_target`, or global plateau (best score stalled for `sol_plateau_window` iterations).
- **Single strategy**: Tree search with beam pruning only. No evolutionary fallback — keeps the search layer simple and debuggable.

---

## Agent Architecture — 3 LLM Agents + Deterministic Orchestrator

| Agent | Runs | Role |
|-------|------|------|
| **Planner** | Every iteration | Analyzes profiling data + optimization memory, selects technique from structured action library, produces structured plan `{tier, technique, params, target_region, rationale}` |
| **Coder** | K times every iteration (K=4 default) | Implements the plan into kernel code; one focused change per iteration. Has compile and correctness-check tools for self-correction within a retry budget. K parallel calls per iter form a best-of-K fan-out (`ACTSConfig.coder_n_candidates`, opt-out via `=1`); decoder stochasticity at the forced T=1.0 of reasoning-mode models supplies the variance. |
| **Reviewer** | Every iteration (after eval) | Interprets eval results, produces structured feedback `{outcome, metric_deltas, bottleneck_classification, bottleneck_diagnosis, suggestions, branch_quality, conditional_assessment}`. Optional multi-turn capability via `query_metric` (gated by `ACTSConfig.reviewer_metric_queries`, default off) lets the Reviewer fetch additional `raw_metrics` from the iteration's profiling dump when the curated NCU subset is insufficient. |

Plus a deterministic orchestrator (code, not LLM) that manages tree state, beam selection, and move-on criteria.

**LLM SDK**: Agents are built on the OpenAI Agents SDK. The SDK provides the agent runtime (`Agent`, `Runner.run`, `function_tool`) and model-swapping via `OpenAIChatCompletionsModel` — any OpenAI-compatible API works by changing the base URL. All three agents are tool-using: Coder carries compile + correctness + `submit_kernel`; Planner carries `submit_plan`; Reviewer carries `submit_review` plus the optional `query_metric` tool (gated by `ACTSConfig.reviewer_metric_queries`, default off). Submit tools deliver each agent's structured output, retrying in-loop on Pydantic validation failure within a per-agent turn budget (4 by default; 6 for Reviewer when `query_metric` is enabled). Callers still treat each agent as one external call — multi-turn behavior is internal to the agent's tool loop. Reasoning-mode toggles and provider extras are plumbed through `ModelConfig` into `make_run_config`'s `ModelSettings` — `reasoning_effort` becomes `Reasoning(effort=...)` and `extra_body` is forwarded verbatim into the chat completions request — so thinking-mode models (DeepSeek v4-pro, OpenAI o-series) and provider-specific request-body extensions (DeepSeek's `{"thinking": {"type": "enabled"}}`) work without an adapter rewrite.

**LLM Backend**: Default model is **DeepSeek v4-pro** for all three agents (thinking-mode enabled, `reasoning_effort=high`, 384K max output, 1M context — see `configs/models/deepseek.json`). Selection rationale:
- Triton/CUDA knowledge is strong and well-represented in pretraining data
- Reliable JSON mode for Pydantic structured output (critical for agent contracts)
- Native thinking-mode lets a single model serve all three agent roles — heavier reasoning for Coder rewrites and Reviewer diagnosis, the same model for Planner technique selection
- Native OpenAI-compatible API — drops directly into `llm_backend.py` with zero adapter code
- Production-stable API with known reliability characteristics

**Evaluated alternatives**:
- *DeepSeek V3 (non-thinking)*: The prior default. Cheaper per token and lower latency without chain-of-thought, but weaker on the long-context kernel-rewrite path. Kept as a fallback option when budget or latency dominates correctness; switch by pointing `configs/models/deepseek.json` at `deepseek-chat` and clearing `reasoning_effort` / `extra_body`.
- *DeepSeek R1*: Stronger reasoning than V3 but historically 2x cost and chain-of-thought latency overhead; superseded by v4-pro for the default slot. Still a candidate for per-agent specialization if v4-pro proves insufficient on a specific role.
- *GLM-5.1 (Zhipu)*: Demonstrated kernel optimization capability (KernelBench L3: 3.6x geometric mean, 14h CUDA optimization reaching 35.7x speedup). SWE-bench Pro #1 (claimed). Open-source, self-hostable via vLLM. However: structured output reliability unverified, no production API pricing yet, new release without independent benchmarks. **Evaluate when API stabilizes** — kernel domain expertise may outperform DeepSeek v4-pro for the Coder agent.
- *Qwen2.5-Coder*: Good code generation, OpenAI-compatible. No differentiated kernel optimization capability.

**Model specialization** (future): `llm_backend.py` supports per-agent model configs. If evaluation shows benefit, use a stronger/domain-specialized model for Coder (where kernel expertise matters most) and a cheaper model for Planner/Reviewer (where structured output reliability matters most).

### Per-Iteration Communication Flow

```
Planner --> Coder x K (asyncio.gather, K=4 default) --> [per-candidate sequential
            (with compile/correctness tools)             anti-cheat + bench]
                  |                                |        --> rank-and-profile
                  +-- self-correction loop ---------+            (first profile-success
                  (up to max_debug_retries attempts)              wins) --> Reviewer
                                                                          --> Planner (next iter)
```

The Coder axis fans out to K parallel `Coder.implement()` calls per iter (best-of-K, `ACTSConfig.coder_n_candidates=4` default; set `=1` to opt out and recover the pre-A2 1-call shape). Compile + correctness self-correction happens inside each Coder call's own tool loop. The K outputs are then ranked sequentially through anti-cheat + benchmark + profile — the first candidate to clear profiling wins the iter and becomes the tree node. The Planner and Reviewer paths are unchanged; K-way multiplies cost only on the Coder axis. This converts the per-iter comparison from "median LLM draft vs baseline" to "best-of-K vs baseline," closing the failure mode where a regressing median Coder draft makes every iter lose. Failed Coder candidates (compile / correctness / entrypoint-binding / partial-bench / sticky-CUDA) collapse into ONE failure-summary sibling node per iter — its `failure_details: list[FailureDetail]` carries one entry per failed candidate, and per-candidate kernel sources live on disk under `tree/node_<id>/cand_<idx>/kernel.py` for postmortem. Profile-layer failures stay as `coder_failed` events only.

The K-way per-candidate bench + autotune burn-in + NCU profile gauntlet runs in a per-iter subprocess (`python -m src.eval.bench_worker`) so sticky CUDA-context errors (e.g. `cudaErrorIllegalAddress`) die with the child instead of poisoning the rest of the run. The parent orchestrates Planner / Coder K-way fan-out, dispatches the bench gauntlet via `src/eval/bench_subprocess.py`, and merges the child's results + events + `.ncu-rep` artifacts back into the live run on clean exit. See `doc/search.md` and `doc/eval.md` for the subprocess contract.

On compilation or correctness failure inside a Coder call, that call's tool loop handles retries internally. If the retry budget is exhausted, that candidate is dropped from the survivor set. If all K fail, the iter is marked SKIPPED; Coder-layer and bench-layer failures collapse into one failure-summary node per iter under the parent for Planner visibility (worst case 2 nodes per iter regardless of K), while profile-layer failures emit `coder_failed` events only without tree mutation. No separate Debugger agent.

**Operator caveat**: K-way assumes the LLM backend serves the K requests concurrently. Serial backends (locally hosted TGI with one slot, provider-side queues capped below K) turn the `asyncio.gather` into K sequential calls and pay K× wallclock for the same token cost as the parallel case — set `coder_n_candidates=1` in that environment.

### LLM Cost Estimate Per Iteration

| Agent | Calls/iter | Input tokens (est.) | Output tokens (est.) |
|-------|-----------|--------------------|--------------------|
| Planner | 1 | ~4K | ~500 |
| Coder | K × (1-3 with tool-use self-correction); K=4 default | ~12-24K | ~8-16K |
| Reviewer | 1 | ~2K | ~500 |
| **Total** | **~6-14** | **~18-30K** | **~9-17K** |

~60K tokens per iteration under K=4 (up from ~15K under K=1). At beam width 3 and depth 20, ~3.6M tokens per kernel. Set `coder_n_candidates=1` to recover the ~15K/iter pre-A2 budget at the cost of best-of-K variance reduction.

---

## Structured Action Library — 6-Tier System, Triton-First

Each action is a structured record: `{id, tier, name, description, applicable_to, preconditions, min_compute_capability, parameters, guidance, anti_patterns, expected_impact}`. Actions use high-level recipes (step-by-step guidance), not code templates — the Coder adapts the recipe to each kernel. The `preconditions: list[str]` field is LLM-visible documentation rendered into the Planner's system prompt (advisory free-text like `"memory_bound"` or `"compute_bound"`), whereas `min_compute_capability` is a structured enforcement gate consulted by `list_applicable` to deny actions whose hardware requirements the target GPU does not meet.

| Tier | Actions (examples) | Risk | Precondition |
|------|-------------------|------|-------------|
| 1 | block_size_tuning, grid_shape_optimization, occupancy_maximization (post-A1: these shape the `@triton.autotune` config list rather than picking single values; see Backend → Triton coverage) | Low | None |
| 2 | shared_memory_tiling, global_memory_coalescing, register_caching, prefetching, bank_conflict_resolution | Low-Med | Memory bottleneck |
| 3 | tf32_accumulation, mixed_precision, fused_operations, vectorized_loads, loop_unrolling | Medium | Compute pattern |
| 4 | split_k_decomposition, persistent_kernel, warp_specialization, stream_k | High | Kernel structure |
| 5 | h100_tma_loads, h100_wgmma, a100_cp_async, hopper_cluster_launch | High | GPU arch |
| 6 | welford_online_stats, online_softmax, causal_mask_skip, flash_attention_tiling | High | Kernel type |

The "Precondition" column above captures LLM-visible documentation strings surfaced to the Planner, not structurally enforced filters. The only structurally enforced hardware gate today is `min_compute_capability` on Tier-5 actions — `t5_h100_tma=9.0`, `t5_h100_wgmma=9.0`, `t5_hopper_cluster=9.0`, and `t5_a100_cp_async=8.0` — which `list_applicable` consults against `HardwareSpec.compute_capability` to drop unsupported actions before they reach the Planner.

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

**Correctness-isolation invariant (safety property):** No untrusted (agent-generated) kernel launch ever runs on the parent process's CUDA context. All correctness verification launches go to a crash-isolated subprocess (`src.eval.correctness_worker` via `correctness_subprocess.run_correctness_subprocess`), fail-closed on crash/timeout. The three previously in-parent launch sites — the Coder correctness tool, the orchestrator reward-hack re-eval, and baseline post-verification — are all isolated. The invariant is enforced by construction: a correctness call without a `problem_definition_path` and without an explicit in-parent opt-in raises `CorrectnessIsolationError` rather than silently launching in-parent. The Coder's `compile_kernel_tool` is compile-only — it neither launches the kernel nor drives the LLM-authored host wrapper — so it is not a parent-side launch site.

Note: `anti_cheat.py` has three landed surfaces (real, not skeleton). **Correctness-level** (above): `generate_randomized_inputs` + `strict_tolerance_check` provide randomized inputs and strict precision checks — runs inside the Coder's turn. **Performance-level**: `scorer.py` flags `reward_hack_suspect` when `T_k < T_SOL` — the orchestrator routes flagged candidates through additional anti-cheat inspection (see SOL Score Invariant Violations). **Process-level** (added 2026-04-27 scope expansion): the `per_iter_anti_cheat` context manager (yielding an `AntiCheatContext`) plus `check_lazy_outputs_after_bench` wrap the SOL `reward_hack` detector set (`check_monkey_patch`, `check_thread_injection`, `check_lazy_outputs`, `snapshot_critical_functions` + `check_eval_integrity`) — caught torch primitive rebinding, thread injection, lazy/deferred outputs, and namespace tampering between snapshot and check. See JOURNAL → "SOL integration scope expansion — adopt every applicable primitive (2026-04-27)" for the full integration plan.

#### 5-Stage Correctness Gate

| Stage | What | Fail action |
|-------|------|------------|
| 1. Smoke test | Single input, check output matches baseline | Coder self-corrects |
| 2. Shape sweep | Multiple input sizes (tiny → xlarge) | Coder self-corrects |
| 3. Numerical stability | NaN/Inf detection, precision check | Coder self-corrects |
| 4. Determinism | Repeated runs must produce identical outputs | Coder self-corrects |
| 5. Anti-cheat | Randomized inputs, strict tolerance, no output caching. When `workload.tolerance` (SOL `ToleranceSpec`) is set, its `max_atol` / `max_rtol` override every stage — including stage 5's `strict_atol` / `strict_rtol` — so anti-cheat matches the workload's per-problem spec rather than the hardcoded strict defaults. | Coder self-corrects |

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
| `benchmark.py` | Orchestrator (via `bench_worker` subprocess) | Latency measurement (CUDA events) |
| `profiler.py` | Orchestrator (via `bench_worker` subprocess for per-iter; parent for Phase C) | Per-iter analytical roofline diagnostics (arithmetic intensity, achieved TFLOPs / GB·s, pct-of-peak — free) + curated NCU section subprocess for occupancy/L2/TC/stall (every iter, representative workload); full-workload re-profile at Phase C. Bottleneck *classification* is computed once-per-run by `eval/roofline.py::classify_run` (Phase A) and threaded through; the per-iter analytical block refines diagnosis but never re-classifies. See JOURNAL "Profiler approach: analytical classification + curated NCU section (2026-04-20)" and "Bottleneck classify-once (2026-04-22)". |
| `scorer.py` | Orchestrator | SOL score computation (using static T_SOL from roofline.py) |

Per-iter K-way `benchmark.py` + `profiler.py` execution lives in a per-iter subprocess (`src.eval.bench_worker`, dispatched via `src.eval.bench_subprocess`) for blast-radius containment — sticky CUDA-context errors (`cudaErrorIllegalAddress` etc.) die with the child instead of poisoning the rest of the run. The parent handles dispatch + response-handling + artifact merge. Phase C winner re-profile in `pipeline/report.py` still runs in the parent (single trusted candidate, no isolation needed). The orchestrator's reward-hack re-eval (`strict_recheck` mode) is likewise crash-isolated via `correctness_subprocess`, per the correctness-isolation invariant in Coder-Side Eval.

| Metric | Tool | Method |
|--------|------|--------|
| **Latency** | CUDA Events | Median of N trials, 20 warmup + 100 timed + 1 burn-in launch (seed -1) so `@triton.autotune` compile + sweep occurs outside the timed window |
| **Bottleneck classification** | SOLAR / built-in roofline (run-level, Phase A) | `classify_run` computes `memory_bound` / `compute_bound` / `balanced` once per `(problem, representative workload, hardware)` and threads the label through every iteration via `SearchResult.run_bottleneck`. Per-iter analytical diagnostics (arithmetic intensity, achieved TFLOPs, achieved GB·s) refine the picture without re-classifying. |
| **Hardware profiling** | NCU subprocess, curated sections (`Occupancy`, `WarpStateStats`, `MemoryWorkloadAnalysis`, `ComputeWorkloadAnalysis`) | SM occupancy, L2 hit rate, tensor-core utilization, dominant + runner-up warp stall class. Best-effort — NCU failures degrade the signal; analytical classification remains the floor. `ACTS_PROFILER_MODE=full` swaps to `--set full` for debug. |

### Profiling Feedback Pipeline

Full profiling → **Reviewer** → curated prose subset (outcome / bottleneck_diagnosis / suggestions / conditional_assessment, rendered via `reviewer._render_review_for_planner`) → **Planner**. Reviewer acts as intelligent filter — Planner receives the parent node's prior review text, not a scalar.

---

## Benchmark Source — SOL-ExecBench

ACTS uses SOL-ExecBench (NVIDIA, 2026) as its benchmark suite. SOL-ExecBench provides 235 CUDA kernel optimization problems extracted from 124 production AI models, organized into four categories (L1: single-op, L2: multi-op fused, Quant: FP8/NVFP4, FlashInfer-Bench: inference primitives). Each problem includes:

- **Definition** (`definition.json`): Problem name, input/output tensor shapes, dtypes, symbolic axes
- **Reference** (`reference.py`): PyTorch `run()` function — the ground-truth specification of the computation
- **Workloads** (`workload.jsonl`): 7-48 concrete shape instantiations per problem (varying batch size, sequence length, etc.)

**In-memory representation**: ACTS adopts SOL's pydantic models (`sol_execbench.core.data.{Definition, Workload, Solution, Trace}` plus all input variants and trace types) directly as the canonical schema. There is no parallel ACTS dataclass layer — a `Definition` parsed from disk flows unchanged through orchestrator, agents, and report. Other benchmarks (KernelBench, custom problem sets) plug in via per-benchmark adapters under `src/benchmarks/<name>/load.py` that produce `tuple[Definition, list[Workload]]` from their native format.

**Benchmark-agnosticism guarantee** (architectural commitment, added 2026-04-27 scope expansion): the pipeline (`src/pipeline/`, `src/search/`, `src/eval/`, `src/agents/`, `src/kernels/`, `src/memory/`, `src/actions/`) operates exclusively on SOL pydantic types and does not import or reference any benchmark-specific on-disk format. The only place benchmark-format knowledge is allowed to live is `src/benchmarks/<name>/load.py`. Adding SOL-specific code paths outside the adapter (e.g., parsing `definition.json` directly in the orchestrator, or branching on benchmark category in `eval/`) is a violation of this contract. Adding a new benchmark is a one-file change: write `src/benchmarks/<name>/load.py` returning `tuple[Definition, list[Workload]]` from the native format. Today only the SOL-ExecBench adapter is real; `src/benchmarks/kernelbench/` and `src/benchmarks/custom/` are placeholder packages (future/scaffold). The intended KernelBench shape — wrapping `Model.forward` into a `def run(...)` string in `Definition.reference` with init params handled via `custom_inputs_entrypoint` — and the custom path (drop `definition.json` + `workload.jsonl` in a directory) are documented here as the target contract; both will land when those adapters are written.

**Constraints inherent to the canonical schema** (apply to all benchmarks, not just SOL-ExecBench):

- Reference is a pure function — `def run(*args)` from inputs to outputs, no shared state across calls. Stateful kernels (KV cache, accumulators) must encode state as inputs/outputs.
- Output count is static — declared in `Definition.outputs`; variable-output kernels don't fit.
- dtype must be in SOL's enum — fp64/32/16/bf16, fp8 e4m3/e5m2, fp4 e2m1, int64/32/16/8, bool. Other dtypes (complex, uint) need an upstream SOL schema PR.
- GPU-only correctness reference — `gen_inputs` and `verify_correctness` assume CUDA tensors.
- Reference operations must be Triton-translatable (per backend choice; see Backend section).

### Triton Baseline Generation

SOL-ExecBench provides only PyTorch references. Since ACTS optimizes Triton code, each problem requires a Triton baseline as the root of the search tree. The baseline can come from one of two sources, selected via configuration:

- **LLM translation** (default — `generate_triton_baseline`): the Coder agent produces a one-shot PyTorch-to-Triton translation at problem load time. Coder receives the PyTorch reference and problem definition, produces a functionally equivalent Triton kernel, correctness is verified against the PyTorch reference (same 5-stage gate), and Coder retries up to `max_baseline_retries` attempts on failure; if all retries fail, the problem is skipped.
- **Operator-supplied** (`load_operator_baseline`): when `[runtime] use_operator_baseline=true` in the run config, the file at `[runtime] triton_baseline_path` is loaded as the search-tree root after compile + per-workload correctness, bypassing the LLM Coder entirely. Any gate miss is a hard fail — no retry, no fallback. This is the sibling of investigation item C4 (curated per-op starter library): same idea, operator-supplied vs framework-baked.

See `doc/eval.md` and `doc/config.md` for the gate sequence and configuration details.

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
3. **Phase A baseline review** (iter=0, once per run): the orchestrator profiles the root via `profile_kernel(root)` and runs `Reviewer.review(prev_sol_score=None)`, attaching `root.last_review` + `root.branch_quality` + `root.profiling`. This primes the Planner-feedback channel so iter=1 sees a real review of the Triton baseline instead of `None`. A baseline `DEAD_END` verdict is clamped — the root stays expandable — and a `reviewer_feedback` event with `iter=0` is emitted.
4. **At each eval iteration**: `profiler.py` produces per-iter diagnostic signals — analytical roofline metrics (arithmetic intensity, achieved TFLOPS / GB/s, pct-of-peak) when `(flops, nbytes)` are derivable for the workload, and curated NCU metrics (occupancy, L2 hit rate, tensor-core utilization, top-2 warp stalls). The analytical block is optional: `ProfilingResult.analytical` is `None` for kernels where bytes cannot be derived from shapes / SOLAR, and NCU still runs in that case (renderers guard via `has_analytical`). These refine the Reviewer's action-tier choice but do **not** re-classify the bottleneck. `scorer.py` computes SOL Score using the static `T_SOL` from step 2.
5. **Reviewer** receives the SOL score, the run-level bottleneck (threaded through from step 2), the per-iter analytical + NCU blocks, and how far `T_k` is from `T_SOL`. Reports remaining headroom.
6. **Planner** receives the parent node's curated Reviewer prose — the `outcome`, `bottleneck_diagnosis`, `suggestions`, and `conditional_assessment` fields rendered by `reviewer._render_review_for_planner` — not a single distilled scalar. (Phase A's iter-0 baseline review in step 3 primes this channel so iter=1's Planner sees a real `last_review` rather than `None`.) In addition, the orchestrator threads the parent kernel's **condensed source** to the Planner (and Reviewer) via `Kernel.render_condensed_source(representative_workload_uuid=workloads[0].uuid)` — called at three sites in `src/search/orchestrator.py` (baseline review, per-iter Planner, per-iter Reviewer) — so the Planner sees the actual code it is mutating, not just the curated prose.
7. **Move-on criteria**: SOL score plateau (consecutive iterations with < δ improvement) or SOL score > threshold (e.g., 0.95 — within 5% of hardware limit).
8. **Cross-kernel comparability**: SOL score of 0.9 on matmul is directly comparable to 0.9 on softmax — both are 90% of the way to their respective hardware limits.

### Bottleneck Classification

Classification is once-per-run (via `classify_run` in `eval/roofline.py`) — invariant per `(problem, representative workload, hardware)`. The `BottleneckType` enum lives in `eval/types.py` so memory / search / pipeline can import it without pulling in the full roofline module:

| Classification | Condition | Primary Action Tiers |
|---------------|-----------|---------------------|
| Memory-bound | Arithmetic intensity < ridge point (outside balanced band) | Tier 2 (memory optimization) |
| Compute-bound | Arithmetic intensity > ridge point (outside balanced band) | Tier 3 (compute optimization) |
| Balanced | Near the ridge point | Either tier, guided by NCU sub-metrics |

The run-level label is threaded into retriever / planner / reviewer every iteration via `SearchResult.run_bottleneck` and surfaces in Phase C as `OptimizationReport.bottleneck`. A separate per-workload view — `OptimizationReport.winner_per_workload_bottlenecks`, populated by per-workload `derive_t_sol_from_solar(...).bottleneck` calls in `src/pipeline/report.py::generate_report` — captures how individual workloads land relative to the ridge, which the single representative workload's label cannot show. Workloads where SOLAR returns `None` are omitted from this map.

`OptimizationReport` also carries `hardware_spec: HardwareSpec | None` — the resolved spec used for the run (populated by `generate_report` from the orchestrator's resolved spec). `render_report` ends with a "Hardware spec" block summarizing the device the SOL bounds were computed against, and the rendered report is persisted to `<run_dir>/report.txt`. This makes the hardware target a first-class part of the artifact rather than an implicit assumption, so a report read months later still tells the reader which GPU the SOL score was anchored to.

`OptimizationReport` additionally carries `usage_stats: UsageSnapshot | None` — per-iter x per-agent LLM token accounting captured by `JSONLTraceProcessor`'s `UsageAccumulator` tap and surfaced via `RunContext.usage_snapshot()` (always returns a snapshot; never `None`). Two artifacts land in `<run_dir>`:

- **`report.txt` — `Resource usage (LLM)` block** sits between the `[AUDIT]` lines and the `Hardware spec` block. It renders a per-iter (rows) x per-agent (columns: Planner / Coder / Reviewer / baseline) grid plus row/column/grand totals, with each cell formatted as `<calls> (<turns>) / <input>-><output>` — call count, agent-loop turns in parentheses, input-to-output token counts. Empty cells render as `-`.
- **`<run_dir>/usage.json`** — machine-readable sidecar with schema `{schema_version: 1, columns: [...], by_iter: [...], by_agent: {...}, total: {...}}`. Best-effort write; OSError downgrades to a WARNING rather than aborting the run.

See `doc/runtime.md` and `doc/pipeline.md` for the full rendering rules and schema.

See JOURNAL → "Bottleneck classify-once (2026-04-22)" for why a per-iter dynamic reclassification (earlier design) was dropped.

### Per-Iteration Analytical Inputs — Shape-Based, Not SOLAR-Derived

The per-iter analytical profiler (`eval/profiler.py::_compute_analytical`) needs `(flops, nbytes)` to compute achieved-TFLOPS, achieved-bandwidth, and %peak. `src/benchmark/roofline_shapes.py::compute_roofline_inputs(definition, workload, *, roofline=None)` derives them with a two-source preference: when *roofline* is a SOLAR-sourced `RooflineResult` carrying positive `total_flops` and `total_fused_bytes`, those counts win (physics-accurate, fusion-aware, dtype-aware — already paid for at Phase A). Otherwise the function falls back to a small per-op-type shape-formula table (canonical `2·M·N·K` for matmul; `C·numel(output)` for elementwise / softmax / rmsnorm). The return shape encodes graceful degradation: `(0, nbytes)` when only flops can't be derived (fused / `op_type=None` kernels with resolvable shapes — `_compute_analytical` accepts `flops=0`), and `(0, 0)` only when nbytes is also unresolvable. `profile_kernel` accepts `nbytes=0` by skipping `_compute_analytical` and returning `analytical=None`; NCU still runs.

This split keeps two concerns separate:

- **SOLAR** owns score correctness — its FLOPs and bytes are physics-accurate, fusion-aware, dtype-aware, and view-elision-aware. Run once at problem load to derive `T_SOL`, and its counts are reused for per-iter analytical via the `roofline=` plumbing above.
- **Shape-based formulas** own the fallback diagnostic signal — coarse, O(1) per call, no SOLAR dependency. Used every iteration when SOLAR counts aren't available (placeholder mode, no-SOLAR installs, or SOLAR returned without populating the count fields). Tradeoff: overcount on view-heavy / mask-heavy kernels (where SOLAR's per-op overrides matter), but the Reviewer treats these numbers as hints, not ground truth, and bottleneck classification still flows from SOLAR.

Why per-iter SOLAR re-invocation is still off the table even though `SolarResult` now carries `total_flops` / `total_fused_bytes`: (a) the profiler must run on non-SOL paths (placeholder mode, no-SOLAR installs) where SOLAR returns `None` and only the shape-formula fallback is available; (b) per-iter SOLAR calls would dominate the wallclock. The Phase A SolarResult counts are threaded into per-iter `compute_roofline_inputs` via the `roofline=` argument instead. See JOURNAL → "Per-iteration analytical flops/nbytes — shape-based formulas, not SOLAR-derived (2026-05-10)" and the subsequent a+b decoupling pass for full rationale.

---

## Optimization Memory — Persistent Cross-Task Learning

Distilled lessons from prior runs in AccelOpt's row shape: `(title, lesson, snippet_before, snippet_after)` produced by a summarizer LLM from a `(parent, child, speedup)` triple. **No profile data, no full kernel source.** Profile data is live-only — consumed by the Planner via the current-iter profile dump, never persisted into opt-mem. Rows additionally carry a deterministic `condition` applicability signature (run bottleneck + action params, e.g. `"compute_bound | BLOCK_N=32"`) and are dedup-consolidated by `(kernel, arch, scope, action, condition)` — same technique under the same condition collapses to the highest-speedup row, while distinct conditions are preserved. Retrieval samples WITHOUT replacement so the Planner never sees a duplicate lesson, and the condition is surfaced to the Planner as "applies when: …". See `doc/memory.md` for the operational model.

### Experience Schema

```
Experience = {
    row_id: str,                 # sha256(run_id ‖ parent_id ‖ child_id ‖ scope)[:16] — idempotent; scope discriminates G1 edge vs G3 run on same (parent, child)
    schema_version: int,         # always KNOWN_VERSION=1 on write; tolerant on read
    kernel_type: str,            # retrieval-filter key
    hardware_arch: str,          # retrieval-preference key (e.g. "RTX6000Ada")
    scope: "edge" | "run",       # G1 per-iter edge vs G3 baseline→best-of-run
    speedup: float,              # always >= δ for stored rows
    action_applied: ActionRecord | None,  # None for scope=="run" (G3 cumulative — no single action)
    title: str,                  # short summarizer-emitted title
    lesson: str,                 # 2–5 sentences, no code
    snippet_before: str,         # changed region only — not the whole file
    snippet_after: str,
    provenance: dict[str, str],  # {run_id, parent_node_id, child_node_id, summarizer_model}
    created_at: str,             # ISO8601 UTC
}
```

Only successful improvements are stored: gate is `speedup >= opt_mem_min_improvement_ratio` (δ, default 1.05). The v1 `success: bool` field is gone — every stored row is a success by construction.

Under K-way Coder fan-out (`coder_n_candidates > 1`), opt-mem growth is unchanged: at most one G1 row per advanced iter (the winner that beat its parent by δ), not K. The K-1 losing candidates exist as `coder_failed` records in `events.jsonl` and `DeadReason.CODER_FAILED` failure-summary nodes in the search tree — they never enter opt-mem.

### Storage & Retrieval

- **Backend**: dedup-consolidated JSONL, one shared global file at `opt_mem/store.jsonl` (config-driven via `ACTSConfig.opt_mem_store_path`). `add` / `add_many` read-merge the on-disk rows + the in-memory cache + the new rows via `dedup_best` (keep the highest-speedup row per `(kernel, arch, scope, action, condition)` key, ties → most recent `created_at`), then atomically rewrite the whole file (`tmp` + `fsync` + `os.replace`) — crash-safety is the atomic replace, not per-row flush. Because the write path re-reads disk before merging, write-only mode (`read_enabled=False`, no prior `load()`) cannot truncate the shared store. Forward-compat rows (`schema_version > KNOWN_VERSION` this binary can't parse) are carried through the rewrite verbatim, so an older binary can't delete a newer binary's lessons. A single writer per store is assumed (no inter-process lock). The path is `.gitignored` so developer-local stores don't conflict.
- **Retrieval**: `MemoryRetriever.sample(kernel_type, hardware_arch)` — (1) filter by kernel type, (2) prefer same-`hardware_arch` rows (fall back to other archs if same-arch count < `top_k`), (3) iterative `random.choices` weighted by `speedup ** α`, drawing WITHOUT replacement (no lesson row repeats in one prompt); `α=0` is uniform random. Pool ≤ `top_k` returns the whole pool unsampled.
- **Injection**: Planner only. Rendered as `[L1]..[Lk]` blocks with structured fields (title / scope / speedup / arch / lesson / before / after). Empty `past_experiences` omits the block entirely.
- **Producer gating**: per-iter `Producer.consider(parent, child, action, *, bottleneck=None)` after the child is scored; gates on `opt_mem_write_enabled`, child compile + correctness (implicit in tree membership), timings present, `parent.runtime_ms / child.runtime_ms >= δ`, summarizer non-None. End-of-run `Producer.finalize(baseline, best_of_run, *, bottleneck=None)` writes one G3 row when cumulative ratio passes δ. Each row carries a `condition` computed via `_format_condition` from the run `bottleneck` + the action params (G3 run-scope rows carry bottleneck only). Cap reserves 1 slot for G3; G1 edges contend for `cap - 1`, with lowest-ratio rows evicting on overflow.

### Read/write flags

- `opt_mem_read_enabled` (default **True**): when False, `MemoryRetriever.sample()` short-circuits to `[]` without opening the store.
- `opt_mem_write_enabled` (default **False**): off by default so ad-hoc / ablation runs cannot pollute the shared store without explicit opt-in. Blessed runs flip True in their `.cfg`.

### Relationship to Search Tree

| | Search Tree | Optimization Memory |
|--|-------------|-------------------|
| Scope | Intra-run (one kernel optimization) | Inter-run (all past kernels) |
| Lifetime | Run start → run end (in-memory; tree-dump → run dir, cleaned later) | Permanent (shared global file) |
| Granularity | Full node state (kernel source, profile, score, branch quality) | Distilled lesson (no profile, no full source) |
| Consumer | Orchestrator | Planner |
| Producer | Orchestrator + Coder | Summarizer LLM (driven by Producer at improving-edge points) |

---

## Backend — Triton (V1)

V1 uses Triton as the sole backend. Target hardware: NVIDIA GPUs.

Triton coverage by tier:

| Tier | Coverage |
|------|----------|
| 1: Block/grid sizing | Full (value picking for `num_warps` / `num_stages` / `BLOCK_*` delegated to `@triton.autotune`; Tier-1 actions now shape the autotune search space rather than picking values directly) |
| 2: Memory | Partial (coalescing automatic, num_stages for pipelining, no bank conflict control) |
| 3: Compute | Mostly full |
| 4: Advanced | Partial (split-K doable, persistent kernels awkward, warp specialization impossible) |
| 5: Arch-specific | Mostly blocked |
| 6: Kernel-specific | Partial |

**Known limitation**: V1 cannot compete with hand-tuned libraries on kernels requiring warp specialization or architecture-specific intrinsics.

**Mandatory autotune (A1 PR 1/B, 2026-05-14)**: every Coder-emitted Triton kernel MUST carry an `@triton.autotune` decorator with at least 4 `triton.Config` entries and a non-empty `key=` list; a validator on `KernelCodeOutput` rejects submissions that omit the decorator, ship fewer than 4 configs, or leave `key=` empty. Post-2026-05-18, the same `submit_kernel` validator also hard-rejects any `triton.Config` entry matching a pattern in the Planner-supplied `autotune_exclude` field on the plan, so structurally-known-bad configs (e.g. ones that crashed sibling candidates) never reach `compile_kernel`. The eval harness performs a single burn-in launch before the warmup window so autotune's compile + config-pick cost lands outside the timed measurement, then attributes winners by diffing Triton's `autotuner.cache` around each workload. The orchestrator emits `autotune_burn_in_done` once per benched node — payload `{iter, workload_count, winner_count}`. Consequence for the action library: Tier-1 actions (`block_size_tuning`, etc.) are no longer manual value-picking levers; their semantics shift to defining the space the autotune sweep should cover. PR 1 is the foundation PR — the action library reshape and per-kernel-type recipe work are deferred and gated on PR 1 live-run evidence.

---

## Hardware Specification Handling

`HardwareSpec` uses the SOLAR arch YAML schema directly — both ACTS and SOLAR share the same hardware description format. This eliminates translation between two schemas and ensures roofline analysis uses consistent parameters.

1. **Load hardware spec at startup** — from a SOLAR arch config YAML (e.g., `configs/arch/H100_PCIe.yaml`, `configs/arch/B200.yaml`). The YAML provides per-cycle throughput by precision (MAC/cycle for FP32, BF16, FP8, etc.), memory hierarchy capacities, and clock frequency. Peak TFLOPS and bandwidth are derived properties.
2. **Use specs internally** — feed hardware spec to SOLAR for `T_SOL` derivation, and to the built-in roofline fallback for bottleneck classification. Compute SOL Score for each candidate kernel.
3. **Reviewer sees** profiling results + roofline classification + SOL score + remaining headroom.
4. **Planner sees** the parent node's curated Reviewer prose only — outcome / bottleneck_diagnosis / suggestions / conditional_assessment, rendered by `reviewer._render_review_for_planner`. Agents never see raw hardware specs.
5. **Fallback** — when no arch YAML is provided, `detect_hardware()` queries `torch.cuda` (best-effort) and returns a partially-populated `HardwareSpec` with the runtime-knowable fields filled (`name`, `freq_GHz` from boost clock, `SRAM_capacity` from L2, `DRAM_capacity`). Per-precision throughput tables (`MAC_per_cycle_*`) and bandwidth coefficients stay zero — those are arch-specific and `torch.cuda` cannot infer them, so real T_SOL / roofline math still requires a SOLAR arch YAML. When `torch` is unavailable, no CUDA device is visible, or the probe raises, a fully-zeroed `HardwareSpec` is returned (the orchestrator's zero-peak handling substitutes a populated placeholder downstream). When both an `arch_config_path` YAML and a runtime detection are available, `validate_hardware_spec()` cross-checks `DRAM_capacity` / `SRAM_capacity` / `freq_GHz` and warns on >10% mismatch — catches the silent-miscalibration case where the YAML doesn't match the GPU actually in the box.
6. **Single-GPU pinning** — ACTS pins itself to one physical GPU via the operator-visible `--gpu-index N` CLI flag on `python -m src.pipeline.optimize`. The flag is preparsed at module top (before any SOL or torch import) and exported as `CUDA_VISIBLE_DEVICES=N`, taking precedence over any pre-existing `CUDA_VISIBLE_DEVICES` in the environment so the operator's explicit choice wins over inherited shell state. A two-tier `_validate_gpu_visible` check enforces the pin: tier 1 confirms `CUDA_VISIBLE_DEVICES` resolves to exactly one device id; tier 2 confirms `torch.cuda.device_count() == 1` after import. `detect_hardware()` then queries device 0 of the visible set — which under the override is the selected physical GPU — so the resolved `HardwareSpec` describes the card ACTS will actually run on, not the host's GPU 0.

**Clock locking** (added 2026-04-27 scope expansion): GPU clocks are actively locked at startup so boost-clock variance stops contaminating timing — without this, T_k drifts run-to-run and the SOL score plateau detector fires on noise rather than real plateaus. ACTS owns the lock/verify/unlock cycle directly rather than delegating to SOL's clock-lock primitives, because every `nvidia-smi` invocation is scoped to the single GPU ACTS actually uses (GPU 0) via an `-i 0` flag injected through a single helper. SOL's unscoped calls would lock and unlock every visible GPU on the host, which is wrong on shared / multi-tenant boxes; the ACTS-side wrappers (`_lock_gpu0_clocks`, `_verify_gpu0_locked`, `_unlock_gpu0_clocks`) confine the side effect. Verification wakes GPU 0 with a tiny torch op before querying so an idle card doesn't report stale low clocks, and compares against the requested preset with a 50 MHz tolerance. The lock-success state is only flipped to True after verify succeeds; verify failure (returning False or raising) triggers a partial-lock rollback that issues the unlock and emits `clock_lock_unavailable` with a `verify_failed` / `verify_raised:<exc>` reason. Preset selection is ACTS-first (`_resolve_clock_preset` consults an internal table that currently covers RTX 6000 Ada at 2505 MHz core / 10001 MHz memory) and falls back to SOL's `get_clock_preset` for known datacenter cards (B200 / H100 / A100); SOL's `probe_clock_lock_available` is still used to detect missing privileges, defensively handling both bare-`bool` and future tuple-shaped returns. Lifecycle coverage extends beyond `atexit` to `SIGTERM` and `SIGHUP` handlers so `kill <pid>` and SSH disconnects unlock cleanly, and an operator escape hatch `python -m src.pipeline.optimize --reset-clocks` exists for the SIGKILL / segfault case where no in-process cleanup could run. The closed vocabulary of `clock_lock_unavailable` reasons (`OK`, `PROBE_RETURNED_FALSE`, `NO_PRESET`, `LOCK_FAILED`, `VERIFY_FAILED`, `UNKNOWN`) is enforced by a `ClockLockReason` enum, while exception-derived reasons remain free-form strings.

---

## Configuration

Run parameters are set through `.cfg` files (INI format, parsed via Python's `configparser`). Unspecified values fall back to built-in defaults. Hardware specs are loaded from a SOLAR arch YAML if `arch_config_path` is specified, otherwise detected at runtime.

```ini
[search]
beam_width = 3
beam_diversity = true
reviewer_metric_queries = false   ; opt-in: registers Reviewer's `query_metric` tool, max_turns=6
coder_n_candidates = 4            ; A2: K-way Coder fan-out per iter (best-of-K -> one child). Set =1 to opt out.
failure_sibling_cap = 8           ; per-parent cap on rendered DeadReason.CODER_FAILED failure siblings after flatten + dedup on (action, params, reason)
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

Several `ACTSConfig` fields are not surfaced via `.cfg` and are set programmatically: `benchmark_adapter` (`"sol_execbench"` / `"kernelbench"` override for `_load_problem` auto-detection), `safetensors_blob_roots` (override for safetensors blob lookup paths), and `anti_cheat_critical_names` (the list of `torch.cuda.Event` method names whose `id()` is snapshotted in the per-iteration anti-cheat context).

---

## Pipeline Flow

```
Phase A: Load SOL Definition + Workloads
  `_load_problem` dispatcher (`src/pipeline/optimize.py`): adapter selection
     1. `ACTSConfig.benchmark_adapter` override -> `'sol_execbench'` |
        `'kernelbench'` (NotImplementedError until that adapter ships).
     2. else `definition.json` present -> SOL-ExecBench adapter.
     3. else `model.py` present -> KernelBench (NotImplementedError).
     4. else raise `UnknownBenchmarkFormat`.
  SOL-ExecBench path -> `src/benchmarks/sol_execbench/load.py::load(...)`
       parses definition.json, reference.py, workload.jsonl into SOL pydantic
       `Definition` + `list[Workload]`; classify via `classify_run` once-per-run.
  -> derive T_SOL via SOLAR (PyTorch reference + hardware arch config)
     -> SOLAR is REQUIRED for SOL-ExecBench problems; `_load_sol_problem`
        fail-fasts via `is_solar_available()` with an install hint when
        missing (the prior silent fallback to built-in roofline produced
        t_sol_us=0.0 on SOL specs and corrupted every score)
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
  -> [DETERMINISTIC EVAL] (in per-iter `bench_worker` subprocess):
     benchmark (CUDA events) -> profiler (analytical roofline per-iter; curated
     NCU subprocess per-iter on representative workload) -> SOL score
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
|   |   |-- trace_processor.py  (SDK trace -> JSONL bridge + UsageAccumulator tap for token accounting)
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
|   |   |-- types.py            (BottleneckType + small shared enums; importable without roofline)
|   |   |-- correctness.py
|   |   |-- benchmark.py
|   |   |-- inputs.py           (input-tensor materialization for correctness/benchmark)
|   |   |-- (power.py — V2, not in V1)
|   |   |-- profiler.py
|   |   |-- _profiler_driver.py (subprocess entrypoint for NCU profiler runs)
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
|   |-- runtime/
|   |   |-- __init__.py
|   |   |-- timefmt.py          (shared UTC timestamp helpers: filename_ts, iso_ts)
|   |   |-- events.py           (emit/bind/unbind event bus + CORE_EVENT_KINDS + iter constants)
|   |   |-- usage.py            (UsageBucket / UsageSnapshot / UsageAccumulator / AgentLabel — per-iter x per-agent LLM token accounting)
|   |   |-- sdk_trace.py        (trace_span helper — shared SDK-trace shim with Tier-1 nullcontext fallback)
|   |   +-- run_context.py      (RunContext dataclass + create/close for trace capture)
|   |
|   |-- benchmark/              (kernel-side helpers — not benchmark-source loaders)
|   |   |-- __init__.py
|   |   |-- baseline_generator.py
|   |   |-- roofline_shapes.py
|   |   |-- solar_adapter.py
|   |   +-- workload_selector.py
|   |
|   |-- benchmarks/             (benchmark-source loaders — one adapter per source)
|   |   |-- sol_execbench/      (real adapter — load.py returns Definition + list[Workload])
|   |   |-- kernelbench/        (placeholder — NotImplementedError until adapter ships)
|   |   +-- custom/             (placeholder — drop definition.json + workload.jsonl)
|   |
|   +-- prompts/
|       |-- planner/
|       |   |-- system.md
|       |   +-- technique_select.md
|       |-- coder/
|       |   |-- system.md
|       |   |-- implement.md
|       |   +-- translate.md    (PyTorch -> Triton baseline translation prompt)
|       |-- reviewer/
|       |   |-- system.md
|       |   +-- interpret.md
|       +-- debugger/           (reserved — Coder handles debugging via tools)
|
+-- tests/
    |-- test_correctness.py
    |-- test_search.py
    |-- test_memory.py
    +-- ...
```

---

## Development Constraint: Always-Runnable Framework

`python -m src.pipeline.optimize` must execute end-to-end on a representative SOL-ExecBench problem at every development iteration — the smoke path. Modules under active development carry the smallest viable implementation (no full placeholders); any not-yet-real surface degrades to a documented no-op rather than blocking the run.

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

**Shipped (A2, default `coder_n_candidates = 4`).** Each iter dispatches K parallel Coder calls via `asyncio.gather`; survivors enter a sequential per-candidate bench loop and the highest-SOL-Score profileable candidate becomes the tree node. Diversity comes from LLM decoder stochasticity at the model's forced temperature, not from per-call prompt or temperature perturbation. See `doc/search.md` and `doc/agents.md` for the per-iter flow and the rank-and-profile-fallback semantics. Trades LLM cost (~K× tokens/iter on the Coder axis) for search breadth — the configured default of K=4 matches AccelOpt's plan-side cardinality and the canonical best-of-N value in code-gen literature.

**Future-work variants** (not shipped): per-call temperature schedule (`[0.3, 0.5, 0.7, 0.9]` style) and prompt-entropy perturbation. Both are decoder-diversity alternatives to A2's identical-call stochasticity; either could compose additively if reasoning-mode temperature constraints loosen.

### Multi-Technique Planning

Allow the Planner to select multiple optimization techniques per plan instead of exactly one. The current "one change at a time" constraint simplifies attribution (which technique helped?) but limits the search when techniques are complementary (e.g., shared memory tiling + prefetching). Multi-technique plans would require the Reviewer to decompose attribution across techniques, and the Coder to apply changes in a controlled sequence.

### Context-Adaptive Agent Specialization

Agent count adapts to LLM context window: 3 agents at 200K+, 5-6 at 32-128K, 7+ at 8-32K.

### Reviewer Knowledge Base Architecture

Three-tier KB: Compute-Reviewer KB, Memory-Reviewer KB, Shared Interaction KB. Two-dimensional retrieval: metric-triggered + action-triggered.

### Multi-Turn Reviewer with On-Demand Profiling Queries

**Variant A — shipped (commit 6d6e62d).** The Reviewer is now a tool-using agent with a bounded turn budget (4 default, 6 when `query_metric` is enabled). Variant A is the in-memory lookup: `query_metric` reads `ProfilingResult.raw_metrics` for NCU metrics captured in the initial run but outside the curated subset (free, no subprocess). Gated by `ACTSConfig.reviewer_metric_queries` (default off) — the existing single-submit path remains the verified default.

**Variant B — future work / on-demand reprofile.** On-demand re-profile with caller-specified `--section` / `--metrics` (expensive subprocess, cache-key expansion to include the metric set requested). Lets the Reviewer recover when the curated signals (occupancy, L2, tensor-core util, top-2 stalls) and the in-memory `raw_metrics` block both miss the kernel's actual bottleneck signature. See PROCESS.md Deferred Improvements for the trigger.
