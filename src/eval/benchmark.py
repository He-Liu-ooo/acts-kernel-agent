"""Latency measurement via CUDA events.

Called by the orchestrator after the Coder returns a compiled, correct
kernel.  Not part of the Coder's tool loop.

Supports two modes:
  - **Single-workload** (legacy): benchmark with a fixed input shape
    derived from ``KernelSpec.input_shapes``.
  - **Multi-workload** (SOL-ExecBench): benchmark across a set of
    ``Workload`` instances, each providing concrete axis values / input
    descriptors.  Returns the aggregate (median-of-medians) result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.benchmark.problem import Workload
    from src.config import ACTSConfig
    from src.kernels.kernel import Kernel


@dataclass
class BenchmarkResult:
    """Latency benchmark result for a single kernel."""

    median_latency_us: float = 0.0
    min_latency_us: float = 0.0
    max_latency_us: float = 0.0
    warmup_runs: int = 0
    timed_runs: int = 0
    per_workload_latency_us: dict[str, float] = field(default_factory=dict)


def benchmark_kernel(
    kernel: Kernel,
    config: ACTSConfig,
    workloads: list[Workload] | None = None,
) -> BenchmarkResult:
    """Benchmark kernel latency using CUDA events.

    If *workloads* are provided (SOL-ExecBench mode), each workload is
    benchmarked independently and the result aggregates across all of
    them.  Otherwise, a single run with shapes from ``kernel.spec`` is
    used.

    Runs warmup iterations followed by timed iterations, measures
    latency via CUDA event elapsed time, returns median.
    """
    # Placeholder: return synthetic latency.
    per_wl: dict[str, float] = {}
    if workloads:
        for wl in workloads:
            per_wl[wl.uuid] = 100.0

    return BenchmarkResult(
        median_latency_us=100.0,
        min_latency_us=95.0,
        max_latency_us=110.0,
        warmup_runs=config.warmup_runs,
        timed_runs=config.timed_runs,
        per_workload_latency_us=per_wl,
    )
