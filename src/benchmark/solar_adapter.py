"""SOLAR integration for T_SOL derivation.

SOLAR (Speed of Light Analysis for Runtime) is an optional external
dependency.  When installed, ACTS calls its Python API to derive
hardware-grounded T_SOL bounds from PyTorch references.  When absent,
ACTS falls back to its own roofline computation in ``eval/roofline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.benchmark.problem import Problem

# Guard: SOLAR is an optional dependency.
try:
    from solar.analysis import EinsumGraphAnalyzer  # noqa: F401
    from solar.einsum import PyTorchToEinsum  # noqa: F401
    from solar.graph import PyTorchProcessor  # noqa: F401
    from solar.perf import EinsumGraphPerfModel  # noqa: F401

    _SOLAR_AVAILABLE = True
except ModuleNotFoundError:
    _SOLAR_AVAILABLE = False


@dataclass
class SolarResult:
    """T_SOL and bottleneck classification from SOLAR."""

    t_sol_us: float
    bottleneck: str  # "compute_bound", "memory_bound", "balanced"
    arithmetic_intensity: float = 0.0
    roofline_model: str = "fused"  # which SOLAR model was used


def is_solar_available() -> bool:
    """Check whether the SOLAR package is importable."""
    return _SOLAR_AVAILABLE


def derive_t_sol(
    problem: Problem,
    arch_config: str = "H100_PCIe",
    roofline_model: str = "fused",
) -> SolarResult | None:
    """Derive T_SOL via the SOLAR pipeline.

    Runs SOLAR's 4-stage pipeline (graph extraction -> einsum conversion
    -> hardware-independent analysis -> performance prediction) on the
    PyTorch reference from *problem*.

    *arch_config* is the SOLAR arch name (e.g. ``"H100_PCIe"``,
    ``"B200"``).  SOLAR resolves this to its own YAML file internally.
    ACTS's ``HardwareSpec`` uses the same YAML schema, so both sides
    share the same hardware description.

    Returns ``None`` if SOLAR is not installed.

    SOLAR produces three roofline models (unfused, fused, fused+prefetched).
    ACTS uses **fused** by default — fused_prefetched assumes perfect overlap
    which is often unreachable in Triton.
    """
    if not _SOLAR_AVAILABLE:
        return None

    # Placeholder: SOLAR integration requires writing the reference to a
    # temp file and running the pipeline stages.  Return synthetic result
    # so the rest of the skeleton stays runnable.
    return SolarResult(
        t_sol_us=10.0,
        bottleneck="memory_bound",
        arithmetic_intensity=0.0,
        roofline_model=roofline_model,
    )
