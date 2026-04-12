# ACTS — Insights & Design Choices from References

Insights drawn from reference repos and 9 analyzed papers that informed ACTS design decisions.

---

## Reference Repo Analysis

### Summary

| Repo | Primary Lesson | Key Pattern to Borrow |
|------|---------------|----------------------|
| **AccelOpt** | Planner->Executor->Memory self-improvement loop | Optimization memory (slow-fast pairs), beam search, two-stage executor |
| **Astra** | Multi-agent role decomposition with shared state | 5-agent blackboard pattern, one-time test generation, clear tool boundaries |
| **AutoKernel** | Evaluation harness + autonomous orchestration | 5-stage correctness, Amdahl's law scheduling, single-file constraint, roofline analysis, 6-tier playbook |
| **EvoToolkit** | SDK design patterns for extensibility | Template method + adapter pattern, Method->Interface->Task stack, persistence layer |

### AccelOpt (MLSys 2026, Stanford/AWS)

Self-improving LLM system that autonomously optimizes AI accelerator kernels.

**Pipeline**: Collect candidates → Planner (LLM plans from baseline profiles) → Executor (parallel proposals → sequential profiling) → Select best → Re-profile → Construct experience memory → Loop.

**Key innovations**:
- **Optimization Memory**: Slow-fast kernel pairs for future guidance. Reduces exploration space. Enables transfer across tasks.
- **Two-Stage Executor**: Parallel proposal generation (speed) then sequential profiling (stability). Hard timeout prevents hangs.
- **Extended Thinking**: Temperature=1.0 with thinking budget for complex reasoning.
- **16-Metric Profiling**: hardware_flops, transpose_flops, hfu_estimated_percent, peak_flops_bandwidth_ratio, mm_arithmetic_intensity, engine utilization, memory transfer bytes, spill metrics, latency.
- **Correctness First**: 5-seed verification. L2 norm comparison. Rejects lower precision types.

**Agent design**: Planner reads baseline profiles, generates `breadth` plans (default 4), 10K thinking budget. Executor implements plans, two-stage with 900s hard timeout.

**Backends**: NKI (AWS Trainium), Triton (NVIDIA H100 via FlashInfer-Bench).

### Astra (Stanford, Production SGLang)

5-agent orchestrated system using OpenAI Agents SDK with `o4-mini`.

| Agent | Tools | Role |
|-------|-------|------|
| Orchestrator | None (text-only) | Coordinates workflow |
| CodeGenerator | save_kernel_code, compile_cuda_kernel | Generates CUDA kernel code |
| CorrectnessTester | generate_comprehensive_test_cases, verify_kernel_correctness | Creates tests (one-time), verifies all versions |
| Benchmarker | benchmark_kernel | Measures performance |
| OptimizationStrategist | None (text-only) | Analyzes performance, suggests optimizations |

**Key innovations**:
- **Shared Blackboard Pattern**: Module-level `optimization_state` dict. Agents read/write shared state, no direct coordination.
- **Test Reuse Architecture**: TestCollection generated once from v1 kernel, reused for all versions. Prevents test drift.
- **Runtime Compilation**: PyBind11 at runtime via `torch.utils.cpp_extension.load()`.
- **Type-Aware Verification**: fp16 → rtol=1e-2, atol=5e-3; fp32 → rtol=1e-6.
- **Benchmarking**: 20 warmup + 100 timed iterations. CUDA event timing. Mean + Std + P95.

**Supported kernels**: RMSNorm, SiLU, MergeState (extensible).

### AutoKernel (Karpathy-style autonomous agent)

Autonomous agent applying "autoresearch" philosophy to GPU kernel development. Agent modifies only `kernel.py`, runs fixed evaluation harness, keeps or reverts.

**Three-phase workflow**:
- **Phase A (Interactive, ~15 min)**: Profile PyTorch model → rank kernels by GPU time → Amdahl's law estimates → human confirms scope.
- **Phase B (Autonomous, 10+ hours)**: For each kernel by Amdahl priority: hypothesize optimization → edit kernel.py → git commit → 5-stage bench → keep if correct & faster, else revert → record result → check move-on criteria.
- **Phase C (Autonomous, ~30 min)**: End-to-end verification + aggregate report.

**Key innovations**:
- **Single-File Constraint**: Agent ONLY modifies `kernel.py`. Eval infrastructure is read-only.
- **5-Stage Correctness**: Smoke test → Shape sweep → Numerical stability → Determinism → Edge cases. Fast-but-wrong immediately reverted.
- **Amdahl's Law Orchestration**: `speedup = 1 / ((1 - f) + f/s)`. Prioritizes by end-to-end impact.
- **6-Tier Optimization Playbook** (900+ lines):

| Tier | Strategy | Gains | Risk |
|------|----------|-------|------|
| 1 | Block Size Tuning | 10-50% | Low |
| 2 | Memory Optimization | 10-30% | Low-Med |
| 3 | Compute Optimization | 5-15% | Medium |
| 4 | Advanced Techniques | 5-20% | High |
| 5 | Architecture-Specific | 5-15% | High |
| 6 | Kernel-Specific Tricks | Varies | High |

- **Dual Backend**: Triton (fast iteration, 80-95% cuBLAS) + CUDA C++ (max perf, tensor cores).
- **Roofline Analysis**: Classifies compute-bound vs memory-bound per kernel.
- **Move-On Criteria**: 5+ consecutive reverts, ≥90% theoretical peak, 2x target, 2hr budget, better ROI elsewhere.

**Built-in kernels**: matmul, softmax, layernorm, rmsnorm, flash_attention, fused_mlp, cross_entropy, rotary_embedding, reduce.

### EvoToolkit (Production-grade evolutionary SDK)

Three-layer stack: Method → MethodInterface → Task.

**Three built-in algorithms**:
1. **EvoEngineer**: PopulationMethod with task-specific operators (crossover, mutation). ThreadPoolExecutor for parallel generation + evaluation.
2. **EoH (Evolution of Heuristics)**: PopulationMethod with predefined operators (I1: init, E1: crossover, E2: guided crossover, M1: structural mutation, M2: parameter mutation).
3. **FunSearch**: IterativeMethod with island-based program evolution. Softmax sampling with temperature schedule. Island reset for diversity.

**Design patterns**:
- **Template Method**: `Method.run()` orchestrates: initialize → loop(step) → complete.
- **Adapter Pattern**: `MethodInterface` bridges Method ↔ Task.
- **Strategy Pattern**: `Operator` objects for genetic operators.
- **Repository/Store Pattern**: `RunStore` manages checkpoints (pickle), history (JSON), summaries.

**Extension points**: Subclass `PythonTask`/`StringTask` for custom evaluation, subclass `IterativeMethod`/`PopulationMethod` for custom algorithms, custom `MethodInterface` for prompt/response contracts.

---

## 9-Paper Knowledge Base

| Framework | Affiliation | Key Innovation | Search Strategy | Language |
|-----------|------------|-----------------|-----------------|----------|
| KernelEvolve | Meta | Heterogeneous accelerator support + persistent knowledge base | Tree search | Triton |
| CUDA-Agent | ByteDance | Only RL-trained approach; SKILL.md structured skills | PPO training | CUDA C++ |
| STARK | Meta AI/Duke | Grounded instruction markers; multi-agent | Epsilon-greedy tree | CUDA C++ |
| Astra | Stanford | Production SGLang kernels; 4-agent system | Iterative refinement | CUDA C++ |
| AVO | NVIDIA | Beats cuDNN/FlashAttention-4 via inline PTX | Evolutionary | CUDA + PTX |
| EvoEngineer | Anonymous | Structured mutations via traverse techniques | Evolutionary | CUDA C++ |
| AccelOpt | Stanford/AWS | First non-NVIDIA framework; beam search + optimization memory | Beam search | NKI |
| SwizzlePerf | Harvard/AMD/Stanford | Hardware-aware swizzle optimization for AMD MI300x | Iterative | Triton/HIP |
| robust-kbench | Sakana AI | Benchmark exploitation analysis; soft verification | Single-lineage evo | CUDA C++ |

---

## Critical Insights

### 1. Search Strategy is the Core Design Decision

The field has NOT converged on a winning search strategy:

- **Greedy/Iterative** (Astra): Fast, works for production code, simple coordination
- **Beam Search** (AccelOpt): Good middle ground, moderate compute
- **Tree Search** (KernelEvolve, STARK): Can backtrack, explores deeply, tree memory useful
- **Evolutionary** (AVO, EvoEngineer): High diversity, expensive, explores large spaces
- **RL-Trained** (CUDA-Agent): Fast inference but massive training cost; opaque strategy

### 2. Structured Actions >> Free-Form Prompts

All successful frameworks independently discovered that vague prompts fail:

- **CUDA-Agent's SKILL.md**: Predefined action templates
- **STARK's grounded instructions**: `<<<IMPROVE BEGINS>>> ... <<<IMPROVE ENDS>>>` markers anchor suggestions to code
- **EvoEngineer's traverse techniques**: Named optimization recipes (tiling, loop unrolling, warp-level reduction)

The LLM's job shifts from "figure out what to do" → "correctly apply this specific technique." Massively reduces hallucination.

### 3. Persistent Memory Enables Cross-Task Transfer

Three frameworks independently built this:
- **KernelEvolve**: Full optimization trajectories → few-shot context
- **AccelOpt**: Optimization patterns → ranked by similarity
- **SwizzlePerf**: Bottleneck-fix pairs → triggered by profiling

When optimizing a new kernel, retrieve similar past optimizations as examples. System improves over time.

### 4. Evaluation is Deeply Broken

robust-kbench exposed KernelBench exploitation via:
- Output caching (fast but wrong)
- Input-dependent shortcuts (hardcoding)
- Precision degradation (silently FP32 → FP16)
- Tolerance gaming (approximately-correct outputs)

Better alternatives:
- **SOL-ExecBench**: Measures vs. hardware theoretical peak (absolute, not relative)
- **Forward + backward testing**: Most frameworks only optimize forward pass
- **Robust verification**: Input randomization, strict numerical checks

### 5. Hardware Portability is a Fundamental Tradeoff

| Approach | Portability | Performance Ceiling |
|----------|------------|-------------------|
| **Triton** (KernelEvolve) | High (NVIDIA/AMD/MTIA) | Lower (abstractions limit vendor-specific ops) |
| **CUDA C++** (8/9 frameworks) | None (NVIDIA only) | High (full hardware access) |
| **CUDA + inline PTX** (AVO) | None (architecture-specific) | Maximum (direct instruction control) |

### 6. Multi-Agent vs. Single-Agent is Unresolved

**Multi-agent** (Astra 4-agent, STARK 3-agent): Role specialization, interpretability, different models per role. But communication overhead and coordination complexity.

**Single-agent with structured actions** (CUDA-Agent, AutoKernel): Similar decomposition without communication cost.
