# Pipeline — `src/pipeline/`

End-to-end optimization entry points.

## optimize.py — Main Entry Point

`python -m src.pipeline.optimize`

### Phase A: Load Problem

1. Load baseline kernel (from starters or KernelBench)
2. Compile and verify baseline correctness
3. Benchmark baseline latency
4. Derive T_SOL via roofline model
5. Classify baseline as compute-bound or memory-bound
6. Compute baseline SOL score (= 0.5 by definition)

### Phase B: Search Loop

Delegates to `Orchestrator.run()`. Runs up to `max_depth` iterations with 3 agents (Planner → Coder → Reviewer).

### Phase C: Report

Generates `OptimizationReport` from the best node found.

## verify.py — Post-Optimization Verification

Re-runs the correctness gate on the best kernel to confirm results are reproducible. Recompiles the winner, then delegates to `verify_correctness` against the PyTorch reference. Compile failures surface as `passed=False` with a compile-phrased detail string.

`verify_optimized_kernel(optimized, *, reference_fn, input_generator, policy=None, cache_dir=None) -> VerificationResult`

## report.py — Report Generation

`generate_report(result) -> OptimizationReport`

| Field | Description |
|-------|-------------|
| `baseline_latency_us` | Starting latency |
| `best_latency_us` | Best achieved latency |
| `sol_score` | Final SOL score |
| `speedup` | Baseline / best |
| `technique_trace` | Sequence of actions applied |
| `bottleneck_transitions` | How bottleneck shifted |
| `remaining_headroom_pct` | Distance to hardware limit |
| `total_iterations` | Search iterations run |
| `termination_reason` | Why search stopped |

## Running the Pipeline

The pipeline runs end-to-end in placeholder mode without GPU or LLM:

```
$ python -m src.pipeline.optimize
Search completed: budget
  Iterations: 20
  Best SOL score: 0.5000
  Speedup: 1.00x
```

As modules are implemented, placeholder returns are replaced with real logic.
