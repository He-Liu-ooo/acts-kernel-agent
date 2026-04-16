# ACTS — Implementation Status

## Completed

- [x] Reference repo analysis (AccelOpt, Astra, AutoKernel, EvoToolkit)
- [x] 9-paper knowledge base analysis
- [x] Architecture design (search strategy, agent architecture, action library, eval harness, optimization memory, backend choice, hardware handling)
- [x] Directory structure design
- [x] Pipeline flow design
- [x] Project scaffolding (pyproject.toml, src/ skeleton with placeholder modules, pipeline runs end-to-end)

### Implemented during scaffolding (real logic, not placeholders)

- [x] config.py — HardwareSpec (SOLAR arch YAML schema), load_hardware_spec(), load_config(), ACTSConfig
- [x] kernels/kernel.py — Kernel, KernelSpec, KernelType dataclasses
- [x] eval/scorer.py — SOL Score formula + reward_hack_suspect / calibration_warning audit flags
- [x] eval/roofline.py — compute_roofline() (built-in fallback) + derive_t_sol_from_solar() wrapper
- [x] benchmark/problem_loader.py — load_problem(), load_definition(), load_workloads(), problem_to_kernel_spec(), op_type mapping
- [x] benchmark/workload_selector.py — select_workloads() (evenly-spaced sampling by problem size)
- [x] benchmark/solution_formatter.py — format_solution() (SOL-ExecBench solution JSON)
- [x] actions/registry.py — Action dataclass, ActionTier enum, build_default_registry()
- [x] actions/tier1-6 — action definitions (guidance text is placeholder, but structure/metadata is real)
- [x] memory/experience.py — Experience dataclass
- [x] memory/store.py — MemoryStore with save/load/add_experience/query (real JSON persistence)

### Implemented during search (real logic, not placeholders)

- [x] search/tree.py — path_to_node, checkpoint save/load (atomic writes)
- [x] search/beam.py — diversity-aware beam pruning (B2), branch-quality-weighted pruning (B3), configurable diversity (`beam_diversity`)
- [x] search/orchestrator.py — `detect_plateau` wired into search loop, plateau termination

## Next Up

### memory/retriever.py — experience retrieval

Pure Python, no GPU needed. Kernel-type filtering + bottleneck matching for optimization memory retrieval. Currently a skeleton.

## Remaining (dependency-ordered)

Items marked `(skeleton)` have interfaces + placeholder logic that keeps the pipeline runnable. Items marked `(done)` have real implementations. Unmarked items need real implementation.

### Phase 1: Foundation

- [x] config.py (done) — detect_hardware() is placeholder (deferred — YAML loading covers the primary path)
- [x] kernels/kernel.py (done) — dataclasses complete
- [ ] kernels/compiler.py (skeleton) — Triton compilation. Needs GPU for real tests.

### Phase 2: Evaluation Harness

- [ ] eval/correctness.py (skeleton) — 5-stage correctness verification. Needs compiler.py + GPU.
- [ ] eval/benchmark.py (skeleton) — latency measurement. Needs compiler.py + GPU.
- [ ] eval/profiler.py (skeleton) — NCU integration + per-iteration bottleneck classification. Needs GPU. Note: orchestrator must call per candidate and feed dynamic classification to retriever/reviewer/planner (see JOURNAL.md "Dynamic bottleneck reclassification")
- [x] eval/roofline.py (done) — two clean paths: SOLAR (T_SOL + bottleneck together) or built-in fallback. solar_adapter.py placeholder returns synthetic data until SOLAR is installed.
- [x] eval/scorer.py (done) — SOL Score with audit flags per SOL-ExecBench paper Section 4.3
- [ ] eval/anti_cheat.py (skeleton) — two surfaces: correctness-level (input randomization, precision checks) + performance-level (T_k < T_SOL flagging from scorer)

### Phase 3: Actions & Memory

- [x] memory/experience.py (done) — Experience dataclass
- [x] memory/store.py (done) — JSON persistence with save/load
- [ ] memory/retriever.py (skeleton) — kernel-type filtering + bottleneck matching. Pure Python, no GPU.
- [x] actions/registry.py (done) — registry + tier system
- [ ] actions/tier1-6 (skeleton) — action definitions exist but guidance text is placeholder

### Phase 4: Agents & Prompts

- [ ] agents/llm_backend.py (skeleton) — OpenAI Agents SDK integration
- [ ] prompts/planner/ (skeleton) — system + technique_select
- [ ] prompts/coder/ (skeleton) — system + implement
- [ ] prompts/reviewer/ (skeleton) — system + interpret
- [ ] agents/planner.py (skeleton) — returns default plan without LLM
- [ ] agents/coder.py (skeleton) — returns source unchanged without LLM
- [ ] agents/evaluator.py (skeleton) — returns neutral feedback without LLM

### Phase 5: Search

- [x] search/tree.py (done) — tree state, path_to_node, checkpoint save/load (atomic)
- [x] search/beam.py (done) — beam pruning (B3 quality-weighted + B2 diversity-aware, configurable), epsilon-greedy selection
- [ ] search/orchestrator.py (skeleton) — has real control flow but calls placeholder agents/eval

### Phase 6: Pipeline & Integration

- [ ] pipeline/optimize.py (skeleton) — has real Phase A flow (two load paths, roofline, workload selection) but calls placeholder baseline generator
- [ ] pipeline/verify.py (skeleton) — post-optimization verification
- [ ] pipeline/report.py (skeleton) — report generation
- [x] benchmark/problem_loader.py (done)
- [ ] benchmark/baseline_generator.py (skeleton) — Triton baseline generation. Needs Coder agent.
- [x] benchmark/workload_selector.py (done)
- [x] benchmark/solution_formatter.py (done)
- [ ] benchmark/solar_adapter.py (skeleton) — returns synthetic data. Needs SOLAR installed.

### Future (Post-V1)
- [ ] Multi-objective optimization (power, energy-latency product)
- [ ] CUDA C++ backend (V2)
- [ ] Embedding-based memory retrieval
- [ ] Context-adaptive agent specialization
- [ ] Reviewer Knowledge Base architecture
