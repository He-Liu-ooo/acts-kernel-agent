# ACTS — Implementation Status

## Completed

- [x] Reference repo analysis (AccelOpt, Astra, AutoKernel, EvoToolkit)
- [x] 9-paper knowledge base analysis
- [x] Architecture design (search strategy, agent architecture, action library, eval harness, optimization memory, backend choice, hardware handling)
- [x] Directory structure design
- [x] Pipeline flow design

## In Progress

- [ ] Project scaffolding (pyproject.toml, src/ skeleton with placeholder modules)

## Remaining (dependency-ordered)

### Phase 1: Foundation
No dependencies. Everything else builds on these.

- [ ] config.py — global config, hardware spec from SOLAR arch YAML (or runtime detection fallback)
- [ ] kernels/kernel.py — kernel abstraction (code + metadata)
- [ ] kernels/compiler.py — Triton compilation

### Phase 2: Evaluation Harness
Depends on: Phase 1 (kernel abstraction, config for hardware specs)

- [ ] eval/correctness.py — 5-stage correctness verification (against PyTorch reference)
- [ ] eval/benchmark.py — latency measurement (CUDA events, SOL-ExecBench timing methodology)
- [ ] eval/profiler.py — NCU integration + per-iteration bottleneck classification from candidate kernel metrics. Note: orchestrator must call per candidate and feed dynamic classification to retriever/reviewer/planner (see JOURNAL.md "Dynamic bottleneck reclassification")
- [ ] eval/roofline.py — SOLAR integration for T_SOL derivation + initial bottleneck classification (runs once at problem load, not per iteration)
- [ ] eval/scorer.py — SOL Score scoring (T_b from Triton baseline, T_SOL from roofline.py). Includes reward_hack_suspect / calibration_warning audit flags per SOL-ExecBench paper Section 4.3
- [ ] eval/anti_cheat.py — two surfaces: correctness-level (input randomization, precision checks in Coder's tool loop) + performance-level (T_k < T_SOL flagging from scorer, inspected by orchestrator)

### Phase 3: Actions & Memory
Depends on: Phase 1 (kernel abstraction for action targets, experience schema)

- [ ] memory/experience.py — experience dataclass
- [ ] memory/store.py — JSON storage backend
- [ ] memory/retriever.py — kernel-type filtering + bottleneck matching
- [ ] actions/registry.py — action registry + tier system
- [ ] actions/tier1_sizing.py — block/grid tuning
- [ ] actions/tier2_memory.py — memory optimization
- [ ] actions/tier3_compute.py — compute optimization
- [ ] actions/tier4_advanced.py — split-K, persistent kernels
- [ ] actions/tier5_arch.py — architecture-specific (H100/A100)
- [ ] actions/tier6_specific.py — kernel-specific tricks

### Phase 4: Agents & Prompts
Depends on: Phase 2 (eval results for Reviewer), Phase 3 (actions for Planner, memory for Planner)

- [ ] agents/llm_backend.py — OpenAI Agents SDK integration (Agent, Runner, function_tool, model config)
- [ ] prompts/planner/ — system + technique_select
- [ ] prompts/coder/ — system + implement
- [ ] prompts/reviewer/ — system + interpret
- [ ] agents/planner.py — profiling data + memory → structured plan (single-call, no tools)
- [ ] agents/coder.py — plan + kernel code → optimized kernel (tool-using: compile + correctness check, self-correction loop)
- [ ] agents/evaluator.py — interprets eval results (Reviewer, single-call, no tools)

### Phase 5: Search
Depends on: Phase 2 (scorer for node evaluation), Phase 4 (agents called during search)

- [ ] search/tree.py — tree search state management
- [ ] search/beam.py — beam pruning logic
- [ ] search/orchestrator.py — top-level search loop (3-agent: Planner → Coder → Reviewer)

### Phase 6: Pipeline & Integration
Depends on: all previous phases

- [ ] pipeline/optimize.py — main search loop entry point
- [ ] pipeline/verify.py — post-optimization verification
- [ ] pipeline/report.py — report generation
- [ ] benchmarks/sol_execbench/ — SOL-ExecBench problem loader (definition.json, reference.py, workload.jsonl)
- [ ] benchmarks/sol_execbench/ — Triton baseline generation (Coder one-shot translation + correctness gate)
- [ ] benchmarks/sol_execbench/ — workload selection (representative subset for iterative benchmarking)

### Future (Post-V1)
- [ ] Multi-objective optimization (power, energy-latency product)
- [ ] CUDA C++ backend (V2)
- [ ] Embedding-based memory retrieval
- [ ] Context-adaptive agent specialization
- [ ] Reviewer Knowledge Base architecture
