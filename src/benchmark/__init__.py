"""SOL-ExecBench and SOLAR adapter layer.

Provides interfaces for:
  - Loading SOL-ExecBench problems (definition.json + workload.jsonl)
  - Formatting ACTS Triton output as SOL-ExecBench solution.json
  - Generating Triton baselines from PyTorch references
  - Selecting representative workloads for iterative benchmarking
  - Deriving T_SOL via SOLAR (external, optional dependency)
"""
