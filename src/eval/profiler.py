"""NCU hardware profiling integration.

Called by the orchestrator after the Coder returns a compiled, correct kernel.
Not part of the Coder's tool loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.kernels.kernel import Kernel


@dataclass
class ProfilingResult:
    """Hardware profiling metrics from NCU."""

    sm_occupancy: float = 0.0
    memory_throughput_gb_s: float = 0.0
    compute_throughput_tflops: float = 0.0
    l2_cache_hit_rate: float = 0.0
    warp_stall_reasons: dict[str, float] = field(default_factory=dict)
    raw_metrics: dict[str, float] = field(default_factory=dict)


def profile_kernel(kernel: Kernel) -> ProfilingResult:
    """Run NCU profiling (``ncu --set full``) on the kernel.

    Extracts SM occupancy, memory throughput, compute throughput,
    cache hit rates, and warp stall reasons.
    """
    # Placeholder: return synthetic profiling data.
    return ProfilingResult(
        sm_occupancy=0.5,
        memory_throughput_gb_s=500.0,
        compute_throughput_tflops=50.0,
        l2_cache_hit_rate=0.6,
    )
