"""Evaluation harness — correctness, benchmarking, profiling, and scoring.

Split across three call sites:

  Problem-load eval (once at startup, Phase A):
    - roofline.py: SOLAR integration for T_SOL derivation + initial
      bottleneck classification.  Runs on the PyTorch reference, result
      is constant for the entire optimization.

  Coder-side eval (via function_tools, every iteration):
    - compiler.py + correctness.py + anti_cheat.py: compile + 5-stage
      correctness gate.  Correctness is always checked against the
      PyTorch reference (not the Triton baseline).

  Orchestrator-side eval (after Coder returns, every iteration):
    - benchmark.py: latency measurement (CUDA events)
    - profiler.py: NCU profiling + per-iteration bottleneck classification
    - scorer.py: SOL score (T_b from Triton baseline, T_SOL from roofline)
"""
