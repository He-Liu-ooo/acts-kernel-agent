"""Tier-2 GPU end-to-end test for the external reference baseline (Option C).

Measures the real flashinfer DSA wrapper through ``measure_reference_baseline``
on the transcoded FlashInfer-Bench problem: loads the problem, builds the
PyTorch reference + safetensors-backed input generators (blobs resolved via
the sibling ``blob`` symlink at the container), correctness-gates the wrapper
against the reference, and benchmarks it. Requires flashinfer + a real GPU.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmark.reference_baseline import measure_reference_baseline
from src.benchmarks.sol_execbench.load import load as sol_load
from src.config import ACTSConfig, detect_hardware
from src.eval.inputs import build_input_generator, build_reference_fn
from src.kernels.kernel import KernelType

_REPO = Path(__file__).resolve().parents[1]
_PROBLEM = _REPO / "benchmarks/flashinfer_trace/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
_CONTAINER = _PROBLEM.parent  # holds the ``blob`` symlink → kda dataset blob tree


@pytest.mark.gpu
def test_flashinfer_reference_measures_and_passes_correctness():
    if (
        not (_PROBLEM / "definition.json").exists()
        or not (_PROBLEM / "reference_flashinfer.py").exists()
        or not (_CONTAINER / "blob").exists()  # follows the symlink; dangling → False
    ):
        pytest.skip("FlashInfer-trace DSA fixture/blob not present on this host")
    definition, workloads = sol_load(_PROBLEM)
    workloads = workloads[:2]  # decode workloads (num_tokens 1-2) — fast
    reference_fn = build_reference_fn(definition.reference)
    gens = [
        build_input_generator(definition, w, blob_roots=[_CONTAINER])
        for w in workloads
    ]
    result = measure_reference_baseline(
        definition,
        path=str(_PROBLEM / "reference_flashinfer.py"),
        entrypoint="run",
        kernel_type=KernelType.ATTENTION,
        workloads=workloads,
        input_generators=gens,
        reference_fn=reference_fn,
        config=ACTSConfig(hardware=detect_hardware()),
    )
    assert result.median_latency_us > 0
    assert set(result.per_workload_latency_us) == {w.uuid for w in workloads}
