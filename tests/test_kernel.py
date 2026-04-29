"""Tests for ``src/kernels/kernel.py`` — Kernel + KernelSpec dataclasses.

The legacy fields are exercised indirectly across the suite (compiler,
profiler, search). This file specifically pins the small contract bits
that need to round-trip: defaults, the ``dps`` flag for destination-
passing-style kernels, and the Coder-output → Kernel propagation.
"""

from __future__ import annotations

from src.agents.coder import KernelCodeOutput
from src.kernels.kernel import Kernel, KernelSpec, KernelType


def _spec() -> KernelSpec:
    return KernelSpec(name="k", kernel_type=KernelType.ELEMENTWISE)


# ── default field values ────────────────────────────────────────────────


def test_kernel_dps_defaults_to_false():
    """Hand-written starters and pre-DPS checkpoints must round-trip without
    setting ``dps``; the default has to stay False to preserve back-compat
    with every Kernel constructed before this field existed."""
    k = Kernel(spec=_spec(), source_code="def kernel_fn(x): return x")
    assert k.dps is False


def test_kernel_dps_can_be_set_true():
    """When the Coder declares destination-passing-style, the flag flows
    to the Kernel verbatim — the benchmark loop branches on it."""
    k = Kernel(spec=_spec(), source_code="def kernel_fn(x, out): pass", dps=True)
    assert k.dps is True


# ── KernelCodeOutput → Kernel propagation ──────────────────────────────


def test_kernel_code_output_dps_defaults_to_false():
    out = KernelCodeOutput(
        source_code="@triton.jit\ndef k(): pass",
        triton_kernel_name="k",
    )
    assert out.dps is False


def test_kernel_code_output_dps_round_trips_true():
    out = KernelCodeOutput(
        source_code="@triton.jit\ndef k(): pass",
        triton_kernel_name="k",
        dps=True,
    )
    assert out.dps is True
