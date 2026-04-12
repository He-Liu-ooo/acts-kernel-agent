# ACTS — Implementation Status

## Completed

- [x] Reference repo analysis (AccelOpt, Astra, AutoKernel, EvoToolkit)
- [x] 9-paper knowledge base analysis
- [x] Architecture design (search strategy, agent architecture, action library, eval harness, optimization memory, backend choice, hardware handling)
- [x] Directory structure design
- [x] Pipeline flow design

## In Progress

- [ ] Project scaffolding (pyproject.toml, src/ skeleton with placeholder modules)

## Remaining

### Core Infrastructure
- [ ] config.py — global config, hardware detection
- [ ] kernels/kernel.py — kernel abstraction (code + metadata)
- [ ] kernels/compiler.py — Triton compilation

### Evaluation Harness
- [ ] eval/correctness.py — 5-stage correctness verification
- [ ] eval/benchmark.py — latency measurement (CUDA events)
- [ ] eval/profiler.py — NCU integration
- [ ] eval/roofline.py — roofline model analysis
- [ ] eval/scorer.py — scoring (V1: latency only)
- [ ] eval/anti_cheat.py — input randomization, precision checks

### Search
- [ ] search/tree.py — tree search state management
- [ ] search/beam.py — beam pruning logic
- [ ] search/orchestrator.py — top-level search loop

### Agents
- [ ] agents/planner.py — profiling data + memory → structured plan
- [ ] agents/coder.py — plan + kernel code → optimized kernel
- [ ] agents/evaluator.py — interprets eval results (Reviewer)

### Action Library
- [ ] actions/registry.py — action registry + tier system
- [ ] actions/tier1_sizing.py — block/grid tuning
- [ ] actions/tier2_memory.py — memory optimization
- [ ] actions/tier3_compute.py — compute optimization
- [ ] actions/tier4_advanced.py — split-K, persistent kernels
- [ ] actions/tier5_arch.py — architecture-specific (H100/A100)
- [ ] actions/tier6_specific.py — kernel-specific tricks

### Optimization Memory
- [ ] memory/store.py — JSON storage backend
- [ ] memory/retriever.py — kernel-type filtering + bottleneck matching
- [ ] memory/experience.py — experience dataclass

### Pipeline
- [ ] pipeline/optimize.py — main search loop entry point
- [ ] pipeline/verify.py — post-optimization verification
- [ ] pipeline/report.py — report generation

### Prompts
- [ ] prompts/planner/ — system + technique_select
- [ ] prompts/coder/ — system + implement
- [ ] prompts/reviewer/ — system + interpret
- [ ] prompts/debugger/ — system + diagnose

### Benchmarks & Tests
- [ ] benchmarks/kernelbench/ — KernelBench adapter
- [ ] Baseline kernel starters (matmul, softmax, layernorm, attention)
- [ ] tests/test_correctness.py
- [ ] tests/test_search.py
- [ ] tests/test_memory.py

### Future (Post-V1)
- [ ] Multi-objective optimization (power, energy-latency product)
- [ ] CUDA C++ backend (V2)
- [ ] Embedding-based memory retrieval
- [ ] SOL Score metric
- [ ] Context-adaptive agent specialization
- [ ] Reviewer Knowledge Base architecture
