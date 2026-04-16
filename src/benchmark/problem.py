"""Data model for SOL-ExecBench problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AxisDef:
    """A single axis definition from definition.json."""

    type: str  # "const", "var", "expr"
    value: int | None = None  # for const axes
    expression: str | None = None  # for expr axes
    description: str = ""


@dataclass
class TensorDef:
    """An input or output tensor definition from definition.json."""

    shape: list[str] | None  # None = Python scalar, [] = 0-D tensor
    dtype: str
    description: str = ""


@dataclass
class Workload:
    """A single workload instance from workload.jsonl.

    Each workload provides concrete axis values and input descriptors
    (random, scalar, or safetensors) for one benchmark run.
    """

    uuid: str
    axes: dict[str, int]  # var axis name -> concrete value
    inputs: dict[str, dict] = field(default_factory=dict)  # input descriptors


@dataclass
class Problem:
    """A parsed SOL-ExecBench problem.

    Contains everything needed to set up an ACTS optimization run:
    the computational specification, the PyTorch reference (correctness
    oracle), and the workloads to benchmark against.
    """

    name: str
    axes: dict[str, AxisDef]
    inputs: dict[str, TensorDef]
    outputs: dict[str, TensorDef]
    reference_source: str  # PyTorch run() source — correctness oracle
    op_type: str = ""
    description: str = ""
    constraints: list[str] = field(default_factory=list)
    custom_inputs_entrypoint: str | None = None
    workloads: list[Workload] = field(default_factory=list)
    definition_path: Path | None = None
