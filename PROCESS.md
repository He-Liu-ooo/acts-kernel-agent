# ACTS — Implementation Status

## Completed

### Foundation
- **config.py** — `HardwareSpec` (SOLAR YAML schema), `load_hardware_spec()`, `load_config()`, `ACTSConfig`. `detect_hardware()` wires `torch.cuda.get_device_properties(0)` for runtime-knowable fields. `validate_hardware_spec()` catches YAML/GPU mismatches at config-load + pre-substitution time (DRAM/SRAM/freq with 10% tolerance, warn-don't-raise).
- **kernels/kernel.py** — `Kernel` / `KernelSpec` / `KernelType` dataclasses. `triton_kernel_name` field with `@model_validator` so the Coder declares the JIT name explicitly (removes silent NCU mis-profile on fused outputs with multiple `@triton.jit` defs).
- **kernels/compiler.py** — file-backed `importlib` load (`spec_from_file_location` + `exec_module`), hash-keyed cache, resolves `KernelSpec.entrypoint` via `getattr`.

### Evaluation Harness
- **eval/correctness.py** — 5-stage gate (smoke → shape-sweep → numerical stability → determinism → anti-cheat) with short-circuit failure attribution. Injectable `ComparisonPolicy` (torch-free at import); `TorchComparisonPolicy` delegates to `sol_execbench.compute_error_stats` and fails closed without SOL.
- **eval/inputs.py** — `build_reference_fn` (exec PyTorch reference, resolve `run`) + `build_input_generator` (wraps SOL's `gen_inputs` with seeding) + `allocate_dps_outputs` (Tier 6). Safetensors entries detected at build time and pre-loaded once via `load_safetensors` (Tier 7) so on-disk reads stay out of the per-iteration timing path.
- **eval/benchmark.py** — CUDA-event timing via injectable `BenchmarkTimer` Protocol. Production `_TorchCudaTimer` uses `torch.cuda.Event` pairs + 256MB int64 L2 thrasher. Multi-workload parallel-list contract with fresh-timer-per-workload isolation; fail-closed on partial-workload failures (<half survive → `BenchmarkError`); 100us sentinel on empty-workload path; `is_fully_successful` for orchestrator gating.
- **eval/profiler.py** — hybrid analytical roofline (required, fail-closed) + curated NCU subprocess (best-effort, degrades on failure). Source-hash-keyed cache. Per-iter signals feed Reviewer; run-level classification via `classify_run`.
- **eval/roofline.py** — two clean paths: SOLAR (`derive_t_sol_from_solar`, `source="solar"`) or built-in `compute_roofline()` fallback (`source="builtin"`). `RooflineResult.bottleneck` typed as `BottleneckType` enum end-to-end.
- **eval/scorer.py** — SOL Score formula + `reward_hack_suspect` / `calibration_warning` audit flags (SOL-ExecBench paper §4.3).
- **eval/anti_cheat.py** — three surfaces: correctness-level (Stage 5 in `eval/correctness.py`), performance-level (scorer audit flags), process-level (`per_iter_anti_cheat` context manager + `check_lazy_outputs_after_bench` wired to SOL's `reward_hack` detector set). Active clock-lock end-to-end on RTX 6000 Ada.

### Actions & Memory
- **actions/registry.py** + **tier1-6** — `Action` dataclass, `ActionTier` enum, `build_default_registry()`. `guidance` / `anti_patterns` / `expected_impact` text synthesized from the 9-paper KB + AccelOpt / Astra / autokernel / cuda-optimized-skill / evotoolkit catalogs.
- **memory/{experience,store,retriever}.py** — `Experience` dataclass, JSON persistence, scored retrieval (kernel-type + hardware filtering, bottleneck/success/speedup scoring, reserved failure slots). Pure Python.

### Agents & Prompts
- **agents/llm_backend.py** — OpenAI Agents SDK integration: `ModelConfig`, `create_model()`, `run_agent()` (narrow transient catch + jittered exponential backoff), `make_run_config()`. Shared helpers: `SUBMIT_OK_SENTINEL`, `format_submit_validation_error`, `render_kernel_section`, `max_turns` kwarg.
- **agents/planner.py** — tool-using Agent with `submit_plan` (option α). `PlannerOutput` Pydantic, technique validation, `max_turns=4`, `MaxTurnsExceeded → captured-output recovery → PlanningError`. `strict_mode=False` opt-out for `dict[str, X]` params.
- **agents/coder.py** — tool-using Agent with `submit_kernel` + `compile` + `run_correctness` tools. `KernelCodeOutput` Pydantic, `_max_turns = 2*max_debug_retries + 2` (= 8 by default; +2 covers `submit_kernel` + final plain-text confirmation). Tools closure-bind `KernelSpec` + `reference_fn` + `input_generators`. Second entry point `translate()` (PyTorch→Triton baseline port) shares wiring via `_run_tool_agent`. Temperature 0.0 for determinism.
- **agents/reviewer.py** — tool-using Agent with `submit_review` + optional `query_metric` (gated by `ACTSConfig.reviewer_metric_queries`, `max_turns=6`). `ReviewerFeedbackOutput` Pydantic, rule-based fallback with `degraded` / `error_reason` surfacing (tags include `max_turns_exceeded`, `missing_submit_review`, `llm_retries_exhausted`).
- **prompts/{planner,coder,reviewer}/** — system prompts (bottleneck→technique mapping, prescribed workflows, hard rules, anti-patterns, submit-tool contracts) + per-agent user-prompt format docs. Planner/Reviewer at temperature 0.3 for variance; Coder pinned at 0.0.

### Search
- **search/tree.py** — tree state, `path_to_node`, atomic checkpoint save/load. `TreeNode.consecutive_agent_failures` + module constant `QUARANTINE_THRESHOLD = 2`; `frontier()` excludes quarantined nodes, `best_node()` still considers them. Legacy checkpoints default the counter to 0.
- **search/beam.py** — diversity-aware beam pruning (B2) + branch-quality-weighted pruning (B3, configurable `beam_diversity`); epsilon-greedy selection.
- **search/orchestrator.py** — real control flow + agents + CUDA-event benchmarking + analytical profiling. Fail-closed baseline check (aborts run on partial-workload failure); branch-local `DEAD_END` on child partial failure / profile failure / missing latency. Calls `classify_run` once after roofline, threads `run_bottleneck` into retriever / planner / reviewer / `SearchResult`. Catches `ImplementationError` + `PlanningError` symmetrically; emits `coder_failed` + `planner_failed`; quarantine accounting on `parent.consecutive_agent_failures`. Plateau detection wired.

### Pipeline & Integration
- **pipeline/optimize.py** — Phase A real two-path load + roofline + workload selection + model-configured `CoderAgent` + fail-closed `generate_triton_baseline`. Phase B real CUDA-event benchmarking + analytical profiling. `_load_model_if_configured` reads `$ACTS_MODEL_CONFIG` / `configs/models/deepseek.json`, model load gated on SOL mode. `_load_sol_execbench` async; threads full `input_generators` list. Placeholder hardware substitution applies to caller-supplied zero-peak configs via `dataclasses.replace`. CLI: `--run-dir` + `--trace-dir` + positional `problem_path`. `validate_hardware_spec` runs before placeholder substitution.
- **pipeline/verify.py** — recompiles winner, reruns 5-stage correctness gate. Compile failures surface as `passed=False` with compile-phrased detail. Emits `verify_start` / `verify_done`.
- **pipeline/report.py** — `generate_report` walks `result.tree.path_to_node(best.id)` for `technique_trace`, propagates `reward_hack_suspect` / `calibration_warning`, surfaces run-level `bottleneck` (from `SearchResult.run_bottleneck`) and `winner_per_workload_bottlenecks` (fused with Phase C re-profile pass on every selected workload). `render_report` skips scoring block when `baseline_latency_us == 0`; surfaces `[AUDIT]` lines.
- **benchmark/baseline_generator.py** — drives `CoderAgent.translate()`, recompiles, reruns 5-stage gate against every selected workload. Post-verify catches SDK best-effort output when turn budget exhausted. `BaselineGenerationError` on no-model / retry exhaustion (no stub fallback).
- **benchmark/workload_selector.py** — `select_workloads()` evenly-spaced sampling by problem size.
- **benchmark/solar_adapter.py** — drives SOLAR's 4-stage Python pipeline (`PyTorchProcessor` → `PyTorchToEinsum` → `EinsumGraphAnalyzer` → `EinsumGraphPerfModel`). Bridge synthesizes SOLAR-shaped `Model` from SOL `Definition` + representative `Workload` (const/var/expr axes, 0-D tensors, int/bool dtypes). Arch resolution chain: explicit > SOLAR-bundled (H100_PCIe, B200) > ACTS-supplied YAML (incl. `placeholder-RTX6000Ada` alias) > H100_PCIe with WARNING. `configs/arch/RTX6000Ada.yaml` hand-authored. Forward-only.
- **benchmarks/sol_execbench/load.py** — `load(problem_path) -> tuple[Definition, list[Workload]]`, adopts SOL pydantic schema directly (replaced the deleted `benchmark/problem.py` + `problem_loader.py` + `solution_formatter.py` shims).

### Runtime & Logging
- **src/runtime/{__init__,timefmt,events,run_context}.py** — `RunContext.create(root, *, trace_dir=None, capture_traces=True)` owns per-invocation lifecycle: creates `./runs/run_<UTC>/` with microsecond-precision filename (parallel-safe), configures stdlib `FileHandler` + `StreamHandler`, silences `httpx`/`openai`/`agents` to WARNING, binds `events.jsonl`, wires SDK trace processor under `<run-dir>/traces/`. `close()` idempotent; `atexit` + `finally` for crashed-run flush. `_NullRunContext` keeps `emit()` in logger-only mode on setup OSError. `events.emit(kind, *, iter, **fields)` with `never-raise` discipline; 20 `CORE_EVENT_KINDS`. `finite_or_none()` sanitizes `inf`/`nan` for RFC-8259-valid JSON. See `doc/runtime.md`.
- **agents/trace_processor.py** — imports `filename_ts` from `src/runtime/timefmt.py` (deprecation-clean).

### Search-Tree Recording (landed 2026-05-02)
- **src/runtime/tree_dump.py** — new module, mirrors `events.py` bind/unbind/never-raise pattern. Exposes `bind(tree_root) / unbind() / is_bound() / dump_node(node, *, iter_no, ncu_rep_src, failure_reason=None, failure_detail=None) / finalize_tree(tree)`.
- **`RunContext.create / close`** — bind/unbind tree_dump alongside events; `_cleanup_partial_setup` includes tree_dump unbind.
- **src/search/tree.py** — `TreeNode.iter_no: int = -1` field; public `nodes()`, `has_node(node_id)`, `__len__` on `SearchTree`.
- **src/search/orchestrator.py** — three changes: `_iter_trace(iter_no, agent_name)` wraps each agent's `Runner.run()` in SDK `trace(workflow_name="acts_iter", metadata={...})` (Tier-1 fallback to `nullcontext`); `_kill_branch` calls `tree_dump.dump_node` after `branch_quality = DEAD_END` with `failure_reason`/`failure_detail` so dead-end nodes get persisted dirs; advance-path `dump_node` runs **after** `beam_prune` so streamed `meta.json.branch_quality` reflects post-prune state.
- **src/eval/profiler.py** — `.ncu-rep` capture decoupled from JSON cache (`(cache_dir or _ncu_tmpdir())`); `-f` force-overwrite flag in NCU argv to avoid stale-report collisions in persistent user-scoped tmpdir; `ProfilingResult.ncu_rep_path` populated whenever NCU produced a file; `_extract_ncu_csv` second-subprocess CSV re-extract via `ncu --import <rep> --csv --page details` because `-o` suppresses stdout CSV on NCU 2025.x (see JOURNAL 2026-05-08 entry).
- **src/pipeline/optimize.py** — `tree_dump.finalize_tree(result.tree)` runs after `optimize()` returns (before `ctx.close()` so bind is still live); writes end-of-run `index.json` + `tree.{txt,dot,mmd}` and rewrites each per-node `meta.json`'s `branch_quality` from final tree state.
- Per-node files: `<run_dir>/tree/node_<id>/{kernel.py, ncu.json, ncu.ncu-rep, meta.json}`. End-of-run files: `<run_dir>/tree/{index.json, tree.txt, tree.dot, tree.mmd}`. Trace cross-reference: `jq 'select(.metadata.iter == N and .metadata.agent == "<role>")' traces/*.jsonl`.
- 5 Codex review findings landed during the feature (3 adversarial + 2 non-adversarial): `.ncu-rep` orchestrator wiring, post-prune dump ordering, dead-end node persistence, beam-evicted-node meta rewrite at finalize, NCU `-f` flag.
- Tests: `tests/test_tree_dump.py` (new), `tests/test_search_tree.py` (new), plus extensions to `test_run_context.py`, `test_orchestrator_events.py`, `test_pipeline_optimize.py`, `test_profiler_subprocess.py`. Final state at this commit: 649 passed / 0 failed across non-GPU suite.
- Tree-dump root node persisted post-baseline (`src/search/orchestrator.py:425` calls `tree_dump.dump_node(root, iter_no=root.iter_no, ncu_rep_src=None)` immediately after the baseline `per_workload_latency_us` assignment); resolves prior 5/3 regression where `tree/node_0/` was missing. Verified in live run `runs/run_20260509T112843_330601Z/tree/node_0/` (both `kernel.py` + `meta.json` present).
- NCU silent-degradation diagnostic + `.ncu-rep` capture verified (2026-05-09). `src/eval/profiler.py` adds module-level `logger`; `ProfilingResult.make_degraded` emits `ncu degraded: <slug>` at WARNING from every degraded return path, plus an inline `logger.warning` in the parser-degraded branch (where `raw_metrics` is preserved so the factory can't be reused) (run.log of the 5/8 regression run logged 6 `ncu degraded` lines, confirming emission); 5/9 healthy run shows zero such lines. `nvidia-persistenced` enabled by the operator restored healthy NCU on host. The two-subprocess `ncu --import` design (`_extract_ncu_csv`) — needed because NCU 2025.x's `-o` suppresses stdout CSV — produces both `.ncu-rep` (~93 KB) and `.ncu.json` (~2.6 KB) per child node; verified in `tree/node_{1,2,3}/` of the same 5/9 run. Closes the deferred capture-path verification.

### SOL Integration (mega-PR PR2, landed 2026-04-28/29)
Adopted SOL primitives in-process; cu12.8-compatible, benchmark-agnostic.
- **Tier 1 schema** — `Definition` / `Workload` / `Solution` / `Trace` + all input/solution/trace variants.
- **Tier 3 sol_score** — `compute_sol_score` wraps `sol_execbench.sol_score.sol_score`, layers audit flags.
- **Tier 4 reward-hack** — `check_monkey_patch` / `check_thread_injection` / `check_lazy_outputs` / `snapshot_critical_functions` + `check_eval_integrity` wired through `eval/anti_cheat.py`. Active clock-lock end-to-end on RTX 6000 Ada (sudoers + GPU-0 scoping + signal handlers + `--reset-clocks`).
- **Tier 5 adapter** — `src/benchmarks/sol_execbench/load.py`; empty `kernelbench/` + `custom/` dirs scaffolded.
- **Tier 6 outputs** — `normalize_outputs` / `allocate_outputs` adopted in `eval/correctness.py` + `eval/benchmark.py`.
- **Tier 7 safetensors** — `load_safetensors` integrated into `build_input_generator`, pre-loaded once at problem-load.
- **Env bump** — Python 3.12 + torch cu128 + `pip install -e <SOL-ExecBench>` via `uv` (see `configs/venvs/3.12.md`).

Tier 2 (`do_bench` timing) and Tier 8 (subprocess) deferred — see Future.

### Verified milestones
- First live GPU run cleared 2026-04-26 (rmsnorm).
- V1 completion cleared 2026-04-27 (action library guidance + real `detect_hardware()` + SOLAR adapter).
- Multi-turn Reviewer (Variant A) shipped 2026-04-27.
- NCU enabled on host (`NVreg_RestrictProfilingToAdminUsers=0` + reboot, real metrics flowing).
- Live GPU run #3 (2026-05-02, `runs/run_20260502T060558_692431Z`) cleared 5 of 6 verification points; the per-workload-latency regression that surfaced has since landed as part of the search-tree-recording work (per-workload latencies populated on every committed node + emitted in `bench_done`).

## Next Up

### Active queue (in order)

1. **Backward-kernel SOLAR support**. Forward-shaped plan (zero structural change to SOL `Definition`): identify backward problems by spec-name suffix (`*_backward`); bridge gains `get_loss_fn` / `get_target` synthesis; `derive_t_sol` branches on suffix to select `BackwardProcessor`. Touches correctness + inputs + baseline generator + adapter — own brainstorming round. *Trigger to start*: first backward problem in SOL-ExecBench (likely attention-backward or layernorm-backward).

2. **`eval/anti_cheat.py` orchestration policy layer**. Process-level primitives shipped via Tier 4; remaining is the policy layer — route `reward_hack_suspect` / `calibration_warning` from `scorer.py` to a handler (mark branch dead vs warn-only vs human-review queue). Design discussion required.

### Backlog (post-V1)

- **Codex adversarial review of recent PR** — `/codex:adversarial-review` against `d9e6c4b..dd3220a`. Highest-value targets: deferred-`child.score` invariant (does any other call-site assume score is populated the moment the benchmark succeeds?), fused Phase C loop's `_resolve_workload_roofline` `(0, 0)` contract, `dataclasses.replace` non-mutation in `optimize.py`.
- **Variant B — `reprofile(sections, metrics) -> ProfilingResult`** (multi-turn Reviewer next step). On-demand `ncu` subprocess re-run with caller-specified `--section` / `--metrics`; ~30s/call; cache-key expansion. *Trigger*: real run where LLM consistently asks `query_metric` for keys not in `raw_metrics` AND the curated NCU section is wrong for the bottleneck shape.
- **`request_workload_variant(workload_idx)`** — re-bench against a different selected workload (Reviewer further extension). Pulls `BenchmarkTimer` + `input_generators[idx]` re-entry. *Trigger*: real run where Phase C reveals per-workload disagreement with the iteration's representative workload.
- **`ACTSConfig.reviewer_max_turns` operator-tunable**. Currently fixed at 6. *Trigger*: real run where the fixed budget shows pressure on legitimate paths (not pathological ones).
- **Hard cap on `query_metric` invocations per review** (independent of turn budget). *Trigger*: real run with pathological query loops within the turn budget.
- **`prompt_dir`-based Compute / Memory Reviewer split** consuming the same `query_metric` tool. *Trigger*: a run where one specialty class of bottleneck consistently warrants a different fetch heuristic.

## Future

### Post-V1 features
- Multi-objective optimization (power, energy-latency product)
- CUDA C++ backend (V2)
- Embedding-based memory retrieval
- Context-adaptive agent specialization
- Reviewer Knowledge Base architecture
- Parallel kernel candidate generation (Coder produces N candidates per plan)
- Multi-technique planning (Planner selects multiple complementary techniques)

### Trigger-gated tech-debt

Items surfaced by review passes; each has a **trigger** — the signal to act. Re-read the trigger before reaching for one of these.

- **Event-catalog enrichment — clock-lock / SOLAR-stage / NCU-metrics events**. Per-workload latency in `bench_done` already landed; the remaining sub-asks are deferred: `nvidia_smi_call` / `clock_lock_acquired` / `clock_lock_released` events around the clock-lock lifecycle; `solar_pipeline_stage` events per SOLAR pipeline stage (PyTorchProcessor → PyTorchToEinsum → EinsumGraphAnalyzer → EinsumGraphPerfModel); `ncu_metrics` event with parsed metric dict + `degraded_reason` (raw NCU dict is now persisted to `tree/node_<id>/ncu.json`, so the event is purely about scannable headline signal). *Trigger*: next live run where clock-lock or SOLAR-stage state is needed for postmortem from `events.jsonl` alone.

- **`.ncu-rep` and NCU JSON cache pruning** *(surfaced 2026-05-08 by /simplify efficiency reviewer)*. The search-tree-recording feature persists each profiled child node's `.ncu-rep` (~80–100 KB) into `tree/node_<id>/ncu.ncu-rep`, and the source `.ncu-rep` + `.ncu.json` cache files live in `(cache_dir or _ncu_tmpdir())` indefinitely. Single 30-iter run with ~3 children per iter produces ~9–27 MB inside the run directory. The cache dir grows once per unique cache key (source-hash × workload × mode) and is never pruned across `optimize.py` invocations. Long-running multi-problem operators will eventually hit disk pressure. Fix shape: an LRU/age-based prune in `_save_ncu_cache` (cap by count or age; document as operator policy if cap is configurable). Per-run `tree/` directories are operator concern (no auto-cleanup planned — they're audit artifacts). *Trigger* (any): cache-dir size exceeds an operator-defined threshold; observed disk-pressure warning during a live run; ACTS deployed to a multi-tenant box where shared `/tmp` cleanup matters; first batched-problem CLI lands and reuses the cache across many invocations.

- **SOL Tier 2 — `do_bench` timing + per-iter memory pool (PR 3)** *(deferred 2026-04-29)*. Replace `_TorchCudaTimer` in `eval/benchmark.py` with `sol_execbench.core.bench.timing.{time_runnable, do_bench, clone_args}`; wire `ShiftingMemoryPoolAllocator` into `do_bench`'s `setup` callback. Existing timer adequate for live runs (cleared 2026-04-26). Cost of deferring: ~1–5% ACTS-vs-SOL score gap (statistical estimator divergence); `ShiftingMemoryPoolAllocator` defends against result-cache exploits irrelevant for honest LLM kernels (no cross-iter memoization, L2 thrasher invalidates L2). 12 tests in `tests/test_benchmark.py` will need rewrite (Protocol → callable seam). *Trigger* (any): publish ACTS-vs-SOL head-to-head; anti-cheat catches result-cache exploit the L2 thrasher missed; a kernel's runtime is far enough outside fixed warmup/rep counts to skew the median; SOL upgrades `do_bench` with a feature we want.

- **Subprocess-isolated evaluation (SOL Tier 8)** *(deferred 2026-04-27)*. Adopt `ProblemPackager` + `_make_eval` + `make_eval` so each candidate eval runs in a fresh subprocess. Inline and subprocess produce identical results for ACTS's success path; subprocess unlocks clean kernel-crash recovery (vs sticky CUDA context state), GPU-memory isolation, cross-iteration global-state isolation, tampering robustness against state-based reward hacks (weakref caches, import-time hooks) that escape Tier 4. Cost: ~200–500ms per eval × ~5–10 candidates × ~20–50 iters × dozens of problems. For ACTS today (own LLM, bounded internal search, Triton-only, single-tenant dev box), Tier 4 + 5-stage gate cover the realistic threat surface. *Trigger A*: ACTS evaluates externally-sourced kernels (KernelBench external, RL-discovered, anything not our Coder). *Trigger B*: multi-tenant GPU hardware. *Trigger C*: live run with frequent kernel crashes (>1% of evals, or any case requiring manual intervention). See JOURNAL → "SOL integration scope refinement — Tier 8 (subprocess) deferred (2026-04-27)".

- **Per-dtype peak in `_compute_analytical()` ridge** — `eval/profiler.py:186` uses `peak_flops_fp32` regardless of workload dtype; off for tc-heavy fp16/bf16. Blast radius narrowed 2026-04-28 (bottleneck labels are SOLAR-sourced now); only Reviewer's analytical %peak signals affected. Fix: plumb `Workload.dtype` + `peak_for_dtype(hw, dtype)` against `HardwareSpec.MAC_per_cycle_{fp32_sm, fp16_tc, bf16_tc}`. *Trigger*: live run where Reviewer prose calls a tc workload compute-bound (analytical) while SOLAR calls it memory-bound (or vice-versa) and the disagreement drives wrong technique selection.

- **`MemoryStore.add()` batched flush** — currently rewrites full JSON per add (O(N²) write bytes per session). Split into in-memory `add()` + explicit `flush()` at iteration boundaries. *Trigger*: end-to-end run where store grows past ~500 experiences, OR rewrite shows up in a profile.

- **Tree serialization via `dataclasses.asdict`** *(partial)* — `_serialize_profiling` switched to `asdict` 2026-04-22. `_serialize_kernel` / `_serialize_score` still hand-roll; drift risk visible in `ScoreResult.get("reward_hack_suspect", False)` back-compat hook. Shared helper with enum/Path coercion would collapse the rest. *Trigger*: next field added to `TreeNode`, `Kernel`, `KernelSpec`, or `ScoreResult` — don't pre-refactor (back-compat risk).

- **`CorrectnessContext` dataclass** — **TRIGGER FIRED 2026-04-28**. SOL mega-PR added a fourth field (`policy: ComparisonPolicy`) alongside `definition` + `kernel` + `workload` at three call sites (`eval/correctness.py`, `eval/benchmark.py`, `search/orchestrator.py`). The `kernel_spec` + `reference_fn` + `input_generators` triad in `CoderAgent.implement/translate` + `Orchestrator.run` is the same parameter-sprawl shape. Side benefit: collapses double SOL-pydantic validation in `_load_sol_execbench` + `generate_triton_baseline`. Land independently — PR 3 deferral unblocked it.

- **`SearchResult.tree` → lighter path snapshot** — Phase C currently retains full `SearchTree`; cheap for one-shot CLI but holds every node's source. Lighter snapshot (precompute `best_path` or `technique_trace` in Orchestrator, drop tree reference) shrinks retained footprint and narrows the Phase-C abstraction. *Trigger*: long-lived/batch context where `SearchResult` outlives a single run, OR tree retention shows up in a memory profile.

- **Parallel beam expansion via `asyncio.gather`** — beam width ≥ k opens `asyncio.gather` of top-k frontier picks per iteration, amortizing LLM latency. *Trigger*: wallclock per iteration becomes dominant cost in a real run. Design pass required: serial expansion is load-bearing for `beam_prune` + `MemoryStore.add()` + checkpoint writes (single-writer assumptions). See JOURNAL → Search → "Serial beam expansion".

- **Test helper consolidation** — `_simulate_plan_submission` (`tests/test_planner.py`) and `_simulate_review_submission` (`tests/test_reviewer.py`) are near-identical (~30 lines each). Could share a parametrized helper in `tests/conftest.py`. *Trigger*: a third `_simulate_*_submission` helper appears (fourth agent migrates to submit-tool pattern), OR shared assertions diverge.

- **Action library KB refinement** — initial guidance text authored 2026-04-27 with intentional limitations: (a) `expected_impact` qualitative-only; (b) `anti_patterns` populated only where upstream repos gave explicit warnings (sparse-but-grounded). *Trigger (a)*: enough live runs to fit numeric ranges against observed `T_k / T_SOL` distributions per action. *Trigger (b)*: ≥10 live runs accumulating failed-kernel `Experience` records to ground anti-patterns from real ACTS failures.

- **`_SDK_AVAILABLE` patch fixture** — ~17 LLM-path tests across `test_planner.py` + `test_reviewer.py` repeat the same `patch("...._SDK_AVAILABLE", True) + patch(".Agent") + patch(".function_tool", side_effect=lambda f: f)` block. A `@pytest.fixture` returning the bundle would dedupe ~75 lines and remove the recurring miss-one-patch failure mode (forgetting `function_tool` silently turns the tool into a `FunctionTool` wrapper that breaks the simulator). *Trigger*: third agent needs the same mock, OR boilerplate masks a test failure.

### Skipped (decisions, not tech debt)

- **Tier action files → YAML catalog** — `src/actions/tier{1..6}*.py` are mostly data (~280 LOC). YAML would trade type-checking, IDE refactor support, and import-time error detection for slightly fewer lines. Only worth it if non-developers edit actions — not the case. Keep as Python.
