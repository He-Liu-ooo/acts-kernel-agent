"""SOL-ExecBench problem loader.

Parses definition.json and workload.jsonl into ACTS internal data model.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.problem import AxisDef, Problem, TensorDef, Workload
from src.kernels.kernel import KernelSpec, KernelType

# ── op_type mapping ────────────────────────────────────────────────────────

_OP_TYPE_MAP: dict[str, KernelType] = {
    "gemm": KernelType.GEMM,
    "matmul": KernelType.MATMUL,
    "rmsnorm": KernelType.RMSNORM,
    "layernorm": KernelType.LAYERNORM,
    "softmax": KernelType.SOFTMAX,
    "gqa": KernelType.GQA,
    "gqa_ragged": KernelType.GQA,
    "gqa_paged": KernelType.GQA,
    "attention": KernelType.ATTENTION,
    "moe": KernelType.MOE,
    "moe_dispatch": KernelType.MOE,
    "embedding": KernelType.EMBEDDING,
    "rope": KernelType.EMBEDDING,
    "linear": KernelType.LINEAR,
    "mlp": KernelType.MLP,
    "swiglu": KernelType.MLP,
    "conv": KernelType.CONV,
    "ssm": KernelType.SSM,
    "mamba": KernelType.SSM,
}


def map_op_type_to_kernel_type(op_type: str) -> KernelType:
    """Map a SOL-ExecBench op_type string to an ACTS KernelType.

    Falls back to CUSTOM for unrecognised op_type values.
    """
    return _OP_TYPE_MAP.get(op_type.lower(), KernelType.CUSTOM)


# ── parsing helpers ────────────────────────────────────────────────────────

def _parse_axis(name: str, raw: dict) -> AxisDef:
    axis_type = raw["type"]
    return AxisDef(
        type=axis_type,
        value=raw.get("value"),
        expression=raw.get("expression"),
        description=raw.get("description", "") or "",
    )


def _parse_tensor_def(raw: dict) -> TensorDef:
    return TensorDef(
        shape=raw["shape"],
        dtype=raw["dtype"],
        description=raw.get("description", "") or "",
    )


# ── public API ─────────────────────────────────────────────────────────────

def load_problem(problem_dir: Path) -> Problem:
    """Load a SOL-ExecBench problem from a directory.

    Expects ``definition.json`` and ``workload.jsonl`` in *problem_dir*.
    """
    definition = load_definition(problem_dir / "definition.json")
    definition.workloads = load_workloads(problem_dir / "workload.jsonl")
    definition.definition_path = problem_dir / "definition.json"
    return definition


def load_definition(path: Path) -> Problem:
    """Parse a single ``definition.json`` into a Problem (no workloads)."""
    raw = json.loads(path.read_text())
    axes = {k: _parse_axis(k, v) for k, v in raw["axes"].items()}
    inputs = {k: _parse_tensor_def(v) for k, v in raw["inputs"].items()}
    outputs = {k: _parse_tensor_def(v) for k, v in raw["outputs"].items()}
    return Problem(
        name=raw["name"],
        axes=axes,
        inputs=inputs,
        outputs=outputs,
        reference_source=raw["reference"],
        op_type=raw.get("op_type", ""),
        description=raw.get("description", "") or "",
        constraints=raw.get("constraints", []),
        custom_inputs_entrypoint=raw.get("custom_inputs_entrypoint"),
    )


def load_workloads(path: Path) -> list[Workload]:
    """Parse a ``workload.jsonl`` file into a list of Workload objects."""
    workloads: list[Workload] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        workloads.append(
            Workload(
                uuid=raw["uuid"],
                axes=raw["axes"],
                inputs=raw.get("inputs", {}),
            )
        )
    return workloads


def problem_to_kernel_spec(problem: Problem) -> KernelSpec:
    """Convert a Problem to an ACTS KernelSpec.

    FLOPs and memory_bytes are left at 0 — they are derived by SOLAR,
    not from the problem definition.  The KernelSpec carries the
    ``pytorch_reference`` and ``definition_path`` so downstream code
    can access them.
    """
    kernel_type = map_op_type_to_kernel_type(problem.op_type)

    # Collect const-axis values as representative input shapes.
    const_axes = {
        name: axis.value
        for name, axis in problem.axes.items()
        if axis.type == "const" and axis.value is not None
    }

    return KernelSpec(
        name=problem.name,
        kernel_type=kernel_type,
        input_shapes=[const_axes] if const_axes else [],
        definition_path=problem.definition_path,
        pytorch_reference=problem.reference_source,
    )
