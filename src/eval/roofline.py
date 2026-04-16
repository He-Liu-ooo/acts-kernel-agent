"""Roofline model analysis and T_SOL derivation.

Two paths to T_SOL:

1. **SOLAR** (preferred): ``derive_t_sol_from_solar()`` calls the SOLAR
   adapter which runs the full pipeline on the PyTorch reference.  Result
   is tight and hardware-grounded.

2. **Built-in** (fallback): ``compute_roofline()`` does a simple
   FLOPs / peak_compute vs bytes / peak_bandwidth calculation from
   ``KernelSpec`` fields.  Used when SOLAR is not installed or when
   running on custom (non-SOL-ExecBench) problems where flop_count and
   memory_bytes are known.

Both paths produce a ``RooflineResult`` consumed by the scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.benchmark.problem import Problem
    from src.config import HardwareSpec
    from src.kernels.kernel import KernelSpec


class BottleneckType(Enum):
    """Kernel bottleneck classification from roofline model."""

    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    BALANCED = "balanced"


@dataclass
class RooflineResult:
    """Roofline analysis for a kernel on specific hardware."""

    t_sol_us: float  # Theoretical minimum runtime (microseconds)
    arithmetic_intensity: float  # ops/byte
    bottleneck: BottleneckType
    peak_achievable_tflops: float = 0.0
    source: str = "builtin"  # "solar" or "builtin"


def derive_t_sol_from_solar(
    problem: Problem,
    arch_config: str = "H100_PCIe",
) -> RooflineResult | None:
    """Derive T_SOL via the SOLAR pipeline (optional dependency).

    Returns ``None`` when SOLAR is not installed, signalling the caller
    to fall back to ``compute_roofline()``.
    """
    from src.benchmark.solar_adapter import derive_t_sol

    solar_result = derive_t_sol(problem, arch_config=arch_config)
    if solar_result is None:
        return None

    bottleneck_map = {
        "compute_bound": BottleneckType.COMPUTE_BOUND,
        "memory_bound": BottleneckType.MEMORY_BOUND,
        "balanced": BottleneckType.BALANCED,
    }
    return RooflineResult(
        t_sol_us=solar_result.t_sol_us,
        arithmetic_intensity=solar_result.arithmetic_intensity,
        bottleneck=bottleneck_map.get(solar_result.bottleneck, BottleneckType.MEMORY_BOUND),
        source="solar",
    )


def compute_roofline(
    spec: KernelSpec,
    hardware: HardwareSpec,
) -> RooflineResult:
    """Derive T_SOL and bottleneck classification from built-in roofline model.

    Fallback when SOLAR is not available.  Requires ``spec.flop_count`` and
    ``spec.memory_bytes`` to be populated.

    ``T_SOL = max(FLOPs / peak_compute, bytes / peak_bandwidth)``

    When SOLAR *is* available, callers should use
    ``derive_t_sol_from_solar()`` instead — it returns both T_SOL and
    bottleneck from SOLAR's more sophisticated analysis.
    """
    peak_compute = hardware.peak_flops_fp32  # TFLOPS
    peak_bw = hardware.peak_memory_bandwidth_gb_s  # GB/s

    if peak_compute > 0 and peak_bw > 0:
        t_compute_us = (spec.flop_count / (peak_compute * 1e12)) * 1e6
        t_memory_us = (spec.memory_bytes / (peak_bw * 1e9)) * 1e6
        t_sol_us = max(t_compute_us, t_memory_us)
        arithmetic_intensity = spec.flop_count / max(spec.memory_bytes, 1)
        ridge_point = (peak_compute * 1e12) / (peak_bw * 1e9)
        bottleneck = _classify_bottleneck(arithmetic_intensity, ridge_point)
    else:
        # No hardware specs — return synthetic values.
        t_sol_us = 10.0
        arithmetic_intensity = 0.0
        bottleneck = BottleneckType.MEMORY_BOUND

    return RooflineResult(
        t_sol_us=t_sol_us,
        arithmetic_intensity=arithmetic_intensity,
        bottleneck=bottleneck,
    )


def _classify_bottleneck(arithmetic_intensity: float, ridge_point: float) -> BottleneckType:
    if arithmetic_intensity > ridge_point * 1.1:
        return BottleneckType.COMPUTE_BOUND
    if arithmetic_intensity < ridge_point * 0.9:
        return BottleneckType.MEMORY_BOUND
    return BottleneckType.BALANCED
