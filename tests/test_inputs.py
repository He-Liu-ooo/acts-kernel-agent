"""Tests for eval/inputs.py — reference-function loader + input-generator factory.

The torch- and sol_execbench-backed factories can only run end-to-end with
torch + sol_execbench installed (and a GPU for real use). The torch-free
tests cover the reference loader, its error modes, and the structural
shape of the factory outputs. GPU-marked tests exercise the full
``build_input_generator`` path including safetensors blob loading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.inputs import (
    ReferenceLoadError,
    build_reference_fn,
)


# ── build_reference_fn — happy path ────────────────────────────────────


def test_build_reference_fn_returns_callable_from_source():
    source = "def run(x, y):\n    return x + y\n"
    fn = build_reference_fn(source)
    assert callable(fn)
    assert fn(2, 3) == 5


def test_build_reference_fn_respects_custom_entrypoint():
    source = "def reference(x):\n    return x * 3\n"
    fn = build_reference_fn(source, entrypoint="reference")
    assert fn(4) == 12


def test_build_reference_fn_preserves_module_scope_helpers():
    """Helpers + module-level constants in the reference source stay reachable."""
    source = (
        "SCALE = 5\n"
        "def _inner(x):\n    return x + SCALE\n"
        "def run(x):\n    return _inner(x)\n"
    )
    fn = build_reference_fn(source)
    assert fn(1) == 6


# ── build_reference_fn — failure modes ─────────────────────────────────


def test_build_reference_fn_rejects_missing_entrypoint():
    source = "def not_run(): pass\n"
    with pytest.raises(ReferenceLoadError, match="run"):
        build_reference_fn(source)


def test_build_reference_fn_rejects_non_callable_entrypoint():
    source = "run = 42\n"
    with pytest.raises(ReferenceLoadError, match="callable"):
        build_reference_fn(source)


def test_build_reference_fn_propagates_syntax_error():
    source = "def run(: invalid\n"
    with pytest.raises(SyntaxError):
        build_reference_fn(source)


# ── build_input_generator — safetensors blob loading (GPU-only) ────────


@pytest.mark.gpu
def test_allocate_dps_outputs_resolves_axes_and_returns_buffers():
    """``allocate_dps_outputs`` is the single source of truth for the DPS
    pre-allocation shape used by the correctness gate, the benchmark loop,
    and the NCU profiler subprocess. It must resolve the workload's axes
    against the definition and produce real on-device tensors that match
    the definition's output specs.
    """
    import torch

    from sol_execbench.core.data import Definition, Workload

    from src.eval.inputs import allocate_dps_outputs

    definition = Definition.model_validate({
        "name": "dps_alloc",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {
            "out1": {"shape": ["N"], "dtype": "float32"},
            "out2": {"shape": ["N"], "dtype": "float32"},
        },
        "reference": "def run(x):\n    return x.relu(), x.tanh()\n",
        "op_type": "elementwise",
    })
    workload = Workload.model_validate({
        "uuid": "wl-alloc", "axes": {"N": 128}, "inputs": {},
    })

    outputs = allocate_dps_outputs(definition, workload, device="cuda")

    assert len(outputs) == 2
    for buf in outputs:
        assert isinstance(buf, torch.Tensor)
        assert buf.shape == (128,)
        assert buf.dtype is torch.float32
        assert buf.device.type == "cuda"


@pytest.mark.gpu
def test_build_input_generator_loads_safetensors():
    """A SafetensorsInput-bearing workload triggers load_safetensors at build time.

    The frozen_weight tensor is loaded from the on-disk blob and must be
    identical across reseeded generations, while random inputs (``x``)
    differ — proves the safetensors path bypasses the per-seed RNG.
    """
    import torch

    from src.benchmarks.sol_execbench import load as sol_load
    from src.eval.inputs import build_input_generator

    fixture = Path(__file__).parent / "fixtures" / "sol_safetensors"
    definition, workloads = sol_load(fixture)
    workload = workloads[0]

    gen = build_input_generator(
        definition, workload, device="cuda", blob_roots=[fixture]
    )
    inputs = gen(seed=42)
    assert len(inputs) == 2  # x + frozen_weight in definition order

    inputs2 = gen(seed=43)
    # frozen_weight is loaded from the blob and stays bit-identical across seeds.
    assert torch.equal(inputs[1], inputs2[1])
    # ``x`` is RNG-driven and differs across seeds.
    assert not torch.equal(inputs[0], inputs2[0])
