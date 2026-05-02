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

Both paths produce a ``RooflineResult`` consumed by the scorer. See
``RooflineResult`` for the unit policy on ``arithmetic_intensity`` and
``ridge_point`` (both in MACs/byte regardless of source; the built-in
path applies a FLOPs→MACs approximation, see ``compute_roofline``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Re-export for backward compatibility — callers importing ``BottleneckType``
# from ``src.eval.roofline`` still resolve to the shared definition in
# ``src.eval.types``.
from src.eval.types import BottleneckType

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload

    from src.config import HardwareSpec
    from src.kernels.kernel import KernelSpec

__all__ = [
    "BottleneckType",
    "RooflineResult",
    "classify_bottleneck",
    "classify_run",
    "compute_roofline",
    "derive_t_sol_from_solar",
]


@dataclass
class RooflineResult:
    """Roofline analysis for a kernel on specific hardware.

    Run-level invariants (a property of the kernel + workload + hardware,
    not of any single iteration):

    * ``arithmetic_intensity`` and ``ridge_point`` are both in **MACs/byte**
      regardless of which path produced them. SOLAR's einsum analyzer
      counts MACs natively (``total_macs / total_fused_bytes``); the
      built-in ``compute_roofline`` fallback approximates MACs from
      ``KernelSpec.flop_count`` via ``MACs ≈ FLOPs / 2`` (1 fused
      multiply-add = 2 FLOPs).

    * The FLOPs/2 approximation is **exact for MAC-dominated kernels**
      (matmul, attention, conv) where every floating-point op is an FMA,
      and an **over-estimate for non-MAC kernels** (rmsnorm, softmax,
      reductions) where elementwise + reduction ops inflate
      ``flop_count`` without contributing MACs. Consumers should treat
      built-in-sourced AI as approximate when ``source == "builtin"``;
      SOLAR-sourced AI (``source == "solar"``) is exact because its
      analyzer excludes non-MAC ops cleanly.

    * ``ridge_point`` source matters: SOLAR-sourced runs (``source ==
      "solar"``) carry SOLAR's **precision-aware** ridge — picked from
      the workload dtype's ``MAC_per_cycle`` table entry (bf16_tc,
      fp16_tc, int8_tc, fp32_sm, ...). Built-in-sourced runs use a
      ``peak_flops_fp32 / 2``-derived ridge as a fallback (see
      ``compute_roofline``); that's correct for FP32 workloads and
      approximate (over-estimates ridge) for tensor-core dtypes.
    """

    t_sol_us: float  # Theoretical minimum runtime (microseconds)
    bottleneck: BottleneckType
    source: str = "builtin"  # "solar" or "builtin"
    arithmetic_intensity: float = 0.0  # MACs/byte — see class docstring
    ridge_point: float = 0.0  # MACs/byte — see class docstring


def derive_t_sol_from_solar(
    definition: Definition,
    workload: Workload,
    hardware_spec: HardwareSpec,
    arch_yaml_path: Path | None = None,
) -> RooflineResult | None:
    """Derive T_SOL via the SOLAR pipeline (optional dependency).

    Bridges to ``solar_adapter.derive_t_sol`` which drives SOLAR's
    4-stage Python pipeline against the definition's reference + a
    representative workload's concrete shapes.

    *arch_yaml_path* overrides the arch resolution; otherwise the
    adapter looks up by ``hardware_spec.name``.

    Returns ``None`` when SOLAR is not installed (caller falls back to
    ``compute_roofline()``) or when any pipeline stage produces no
    result (per-stage diagnostics already logged inside the adapter).
    """
    from src.benchmark.solar_adapter import derive_t_sol

    solar_result = derive_t_sol(
        definition, workload, hardware_spec, arch_yaml_path=arch_yaml_path,
    )
    if solar_result is None:
        return None

    # SOLAR computes ``ridge_point`` using the workload's precision-aware
    # ``MAC_per_cycle`` (e.g. ``bf16_tc`` for a bf16 workload), not a
    # single FP32 peak. This is critical for tensor-core workloads where
    # an FP32-derived ridge would be up to 4× too low and silently
    # mis-classify them as compute-bound. See
    # ``repo/benchmark/SOLAR/solar/perf/perf_model.py`` L232.
    return RooflineResult(
        t_sol_us=solar_result.t_sol_us,
        bottleneck=solar_result.bottleneck,
        source="solar",
        arithmetic_intensity=solar_result.arithmetic_intensity,
        ridge_point=solar_result.ridge_point,
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
        # ``RooflineResult.arithmetic_intensity`` is **MACs/byte** by
        # contract (see class docstring). KernelSpec gives us FLOPs, not
        # MACs, so we approximate: ``MACs ≈ FLOPs / 2`` (1 fused
        # multiply-add = 2 FLOPs). Exact for MAC-dominated kernels
        # (matmul, attention, conv); over-estimates AI for non-MAC
        # kernels (rmsnorm, softmax, reductions) where elementwise +
        # reduction ops inflate flop_count without contributing MACs.
        # ``source="builtin"`` signals the approximation to consumers;
        # the SOLAR path produces exact MACs/byte without this fudge.
        # ``ridge_point`` is converted FLOPs→MACs the same way so the
        # two compare dimensionally inside ``classify_bottleneck``.
        arithmetic_intensity = (spec.flop_count / 2.0) / max(spec.memory_bytes, 1)
        # Built-in fallback: no SOLAR available, so we can't pick a
        # precision-aware ridge. FP32-derived ridge is correct for FP32
        # workloads, approximate (over-estimates ridge) for tensor-core
        # workloads — classifying them as compute-bound when they
        # shouldn't be. SOLAR-sourced runs (the normal path) use the
        # precision-aware ridge from ``solar_adapter.SolarResult``.
        ridge_point = (peak_compute / 2.0) * 1e12 / (peak_bw * 1e9)
        bottleneck = classify_bottleneck(arithmetic_intensity, ridge_point)
    else:
        # No hardware specs — return synthetic values.
        t_sol_us = 10.0
        arithmetic_intensity = 0.0
        ridge_point = 0.0
        bottleneck = BottleneckType.MEMORY_BOUND

    return RooflineResult(
        t_sol_us=t_sol_us,
        bottleneck=bottleneck,
        arithmetic_intensity=arithmetic_intensity,
        ridge_point=ridge_point,
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


