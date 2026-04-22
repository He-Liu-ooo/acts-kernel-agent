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
from typing import TYPE_CHECKING

# Re-export for backward compatibility — callers importing ``BottleneckType``
# from ``src.eval.roofline`` still resolve to the shared definition in
# ``src.eval.types``.
from src.eval.types import BottleneckType

if TYPE_CHECKING:
    from src.benchmark.problem import Problem, Workload
    from src.config import HardwareSpec
    from src.kernels.kernel import KernelSpec

__all__ = [
    "BottleneckType",
    "RooflineResult",
    "classify_bottleneck",
    "classify_run",
    "classify_workload",
    "compute_roofline",
    "derive_t_sol_from_solar",
]


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
        bottleneck = classify_bottleneck(arithmetic_intensity, ridge_point)
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


def classify_bottleneck(arithmetic_intensity: float, ridge_point: float) -> BottleneckType:
    """Band-classify a kernel's bottleneck via its arithmetic intensity
    relative to the hardware ridge point. Shared by analytical profiler
    and SOLAR-less built-in roofline so the thresholds can't drift."""
    if arithmetic_intensity > ridge_point * 1.1:
        return BottleneckType.COMPUTE_BOUND
    if arithmetic_intensity < ridge_point * 0.9:
        return BottleneckType.MEMORY_BOUND
    return BottleneckType.BALANCED


def classify_run(
    *,
    hardware: HardwareSpec,
    roofline: RooflineResult | None,
    baseline_spec: KernelSpec | None = None,
) -> BottleneckType:
    """Once-per-run bottleneck classification.

    Classification is invariant per ``(problem, representative_workload,
    hardware)`` so the orchestrator computes it once at baseline time
    instead of re-deriving it on every profiled iteration.

    Preference order:

    1. If ``roofline`` is provided, return ``roofline.bottleneck`` verbatim
       — SOLAR is authoritative for SOL-ExecBench problems.
    2. Otherwise derive via ``compute_roofline(baseline_spec, hardware)``
       — the placeholder / non-SOL fallback path.

    Raises ``ValueError`` if neither is supplied; the orchestrator always
    has at least the baseline spec, so hitting this is a programmer error.
    """
    if roofline is not None:
        return roofline.bottleneck
    if baseline_spec is None:
        raise ValueError(
            "classify_run requires either a RooflineResult or a baseline "
            "KernelSpec; both were None"
        )
    return compute_roofline(baseline_spec, hardware).bottleneck


def classify_workload(
    problem: Problem,
    workload: Workload,
    hardware: HardwareSpec,
) -> BottleneckType:
    """Per-workload bottleneck classification from shape-derived flops/bytes.

    Uses ``compute_roofline_inputs`` to get ``(flops, nbytes)`` from the
    problem's op type + the concrete workload axes, then classifies via
    the shared ``classify_bottleneck`` band so thresholds stay consistent
    with the analytical profiler.

    Raises ``ValueError`` if ``problem.op_type`` has no roofline formula
    (``compute_roofline_inputs`` returns ``(0, 0)``) or if the hardware
    spec has zero peaks — both are config errors that should fail loud
    rather than silently returning ``MEMORY_BOUND``.
    """
    # Lazy import to keep module-load light and avoid any benchmark-side
    # import cycles on future refactors.
    from src.benchmark.roofline_shapes import compute_roofline_inputs

    flops, nbytes = compute_roofline_inputs(problem, workload)
    if flops == 0 and nbytes == 0:
        raise ValueError(f"no roofline formula for op_type={problem.op_type!r}")

    peak_compute = hardware.peak_flops_fp32  # TFLOPS
    peak_bw = hardware.peak_memory_bandwidth_gb_s  # GB/s
    if peak_compute <= 0 or peak_bw <= 0:
        raise ValueError("hardware peaks are zero")

    arithmetic_intensity = flops / nbytes
    ridge_point = (peak_compute * 1e12) / (peak_bw * 1e9)
    return classify_bottleneck(arithmetic_intensity, ridge_point)
