"""Tests for benchmark/problem_loader.py — SOL-ExecBench problem parsing."""

import json
import tempfile
from pathlib import Path

from src.benchmark.problem_loader import (
    load_definition,
    load_problem,
    load_workloads,
    map_op_type_to_kernel_type,
    problem_to_kernel_spec,
)
from src.kernels.kernel import KernelType


# ── op_type mapping ────────────────────────────────────────────────────────

def test_map_known_op_types():
    assert map_op_type_to_kernel_type("gemm") == KernelType.GEMM
    assert map_op_type_to_kernel_type("rmsnorm") == KernelType.RMSNORM
    assert map_op_type_to_kernel_type("gqa_ragged") == KernelType.GQA
    assert map_op_type_to_kernel_type("moe") == KernelType.MOE
    assert map_op_type_to_kernel_type("swiglu") == KernelType.MLP


def test_map_unknown_op_type_falls_back_to_custom():
    assert map_op_type_to_kernel_type("fancy_new_op") == KernelType.CUSTOM


# ── definition parsing ─────────────────────────────────────────────────────

_SAMPLE_DEFINITION = {
    "name": "rmsnorm_h4096",
    "op_type": "rmsnorm",
    "description": "RMSNorm with hidden_size=4096",
    "axes": {
        "batch_size": {"type": "var"},
        "hidden_size": {"type": "const", "value": 4096},
    },
    "inputs": {
        "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
        "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
    },
    "outputs": {
        "output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
    },
    "reference": "import torch\n\ndef run(hidden_states, weight):\n    return hidden_states",
    "custom_inputs_entrypoint": None,
}

_SAMPLE_WORKLOADS = [
    {"uuid": "aaa", "axes": {"batch_size": 1}, "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}}},
    {"uuid": "bbb", "axes": {"batch_size": 64}, "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}}},
]


def test_load_definition():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "definition.json"
        path.write_text(json.dumps(_SAMPLE_DEFINITION))
        problem = load_definition(path)

    assert problem.name == "rmsnorm_h4096"
    assert problem.op_type == "rmsnorm"
    assert "batch_size" in problem.axes
    assert problem.axes["hidden_size"].value == 4096
    assert "hidden_states" in problem.inputs
    assert problem.inputs["hidden_states"].dtype == "bfloat16"
    assert problem.reference_source.startswith("import torch")


def test_load_workloads():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "workload.jsonl"
        path.write_text("\n".join(json.dumps(w) for w in _SAMPLE_WORKLOADS))
        workloads = load_workloads(path)

    assert len(workloads) == 2
    assert workloads[0].uuid == "aaa"
    assert workloads[1].axes["batch_size"] == 64


def test_load_problem_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "definition.json").write_text(json.dumps(_SAMPLE_DEFINITION))
        (d / "workload.jsonl").write_text("\n".join(json.dumps(w) for w in _SAMPLE_WORKLOADS))
        problem = load_problem(d)

    assert problem.name == "rmsnorm_h4096"
    assert len(problem.workloads) == 2
    assert problem.definition_path is not None


# ── KernelSpec conversion ──────────────────────────────────────────────────

def test_problem_to_kernel_spec():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "definition.json"
        path.write_text(json.dumps(_SAMPLE_DEFINITION))
        problem = load_definition(path)

    spec = problem_to_kernel_spec(problem)
    assert spec.name == "rmsnorm_h4096"
    assert spec.kernel_type == KernelType.RMSNORM
    assert spec.pytorch_reference.startswith("import torch")
    assert spec.input_shapes == [{"hidden_size": 4096}]
