"""Kernel abstraction — code + metadata for a single kernel version."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class KernelType(Enum):
    """Known kernel archetypes for memory retrieval matching.

    Core types cover common kernel patterns.  SOL-ExecBench op_type values
    (gemm, rmsnorm, gqa, moe, …) are mapped to these via
    ``map_op_type_to_kernel_type`` in ``src.benchmark.problem_loader``.
    """

    MATMUL = "matmul"
    GEMM = "gemm"
    SOFTMAX = "softmax"
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    ATTENTION = "attention"
    GQA = "gqa"
    MOE = "moe"
    EMBEDDING = "embedding"
    LINEAR = "linear"
    FUSED_BLOCK = "fused_block"
    MLP = "mlp"
    CONV = "conv"
    SSM = "ssm"
    REDUCTION = "reduction"
    ELEMENTWISE = "elementwise"
    CUSTOM = "custom"


@dataclass
class KernelSpec:
    """Static metadata about a kernel problem (does not change across versions).

    For SOL-ExecBench problems, ``definition_path`` points to the source
    ``definition.json``, ``pytorch_reference`` holds the PyTorch ``run()``
    source that serves as the correctness oracle, and ``t_sol_us`` is the
    SOLAR-derived hardware bound (populated at problem-load time).
    """

    name: str
    kernel_type: KernelType
    # Computational profile for roofline (may be 0 when SOLAR provides T_SOL)
    flop_count: int = 0
    memory_bytes: int = 0
    # Reference input shapes for correctness testing
    input_shapes: list[dict] = field(default_factory=list)
    # SOL-ExecBench integration
    definition_path: Path | None = None
    pytorch_reference: str = ""
    t_sol_us: float | None = None
    # Name of the callable the compiler should resolve from the loaded module.
    # Matches AutoKernel's convention; overridable for fused ops where the
    # launchable symbol is a host wrapper around one or more @triton.jit fns.
    entrypoint: str = "kernel_fn"


@dataclass
class Kernel:
    """A single kernel version: source code + metadata."""

    spec: KernelSpec
    source_code: str
    # Triton-specific
    num_warps: int = 4
    num_stages: int = 2
    block_size: dict[str, int] = field(default_factory=dict)
    # Bare name of the ``@triton.jit`` device function the profiler should
    # filter NCU on. Declared by the Coder via ``KernelCodeOutput`` so the
    # source-of-truth lives with the kernel that owns it; empty for hand-
    # written starters / test fixtures, where ``profile_kernel`` falls
    # back to source-regex extraction.
    triton_kernel_name: str = ""
