"""SOLAR integration for T_SOL derivation.

SOLAR (Speed of Light Analysis for Runtime) is an optional external
dependency. When installed, ACTS calls its 4-stage Python pipeline to
derive a hardware-grounded ``T_SOL`` bound from the PyTorch reference.
When absent, ``eval/roofline.py`` falls back to its built-in analytical
roofline.

Pipeline stages (all driven via SOLAR's published Python API — no
subprocess calls):

  1. ``PyTorchProcessor.process_model_file`` — extract the PyTorch graph.
  2. ``PyTorchToEinsum.convert``           — convert to einsum representation.
  3. ``EinsumGraphAnalyzer.analyze_graph`` — count MACs / memory elements.
  4. ``EinsumGraphPerfModel.predict``      — apply the arch YAML's roofline.

SOLAR expects a model file with ``class Model(nn.Module)`` + ``def
get_inputs()``. ACTS holds the reference as a ``def run(*tensors)``
function inside a ``Problem`` dataclass. The bridge in
``_write_model_bridge_file`` synthesizes the SOLAR-shaped wrapper from
the problem's ``reference_source`` + a representative ``Workload``'s
concrete axis values.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.eval.types import BottleneckType

if TYPE_CHECKING:
    from src.benchmark.problem import Problem, Workload
    from src.config import HardwareSpec

logger = logging.getLogger(__name__)

# Guard: SOLAR is an optional dependency.
try:
    from solar.analysis import EinsumGraphAnalyzer
    from solar.common.types import ProcessingConfig
    from solar.einsum import PyTorchToEinsum
    from solar.graph import PyTorchProcessor
    from solar.perf import EinsumGraphPerfModel

    _SOLAR_AVAILABLE = True
except ModuleNotFoundError:
    _SOLAR_AVAILABLE = False


@dataclass
class SolarResult:
    """T_SOL and bottleneck classification from SOLAR."""

    t_sol_us: float
    bottleneck: BottleneckType
    arithmetic_intensity: float = 0.0
    roofline_model: str = "fused"  # which SOLAR model was used


# SOLAR uses bare strings ("memory" / "compute" / "balanced"); ACTS uses
# BottleneckType. Map at the boundary so SolarResult is enum-typed and
# downstream consumers don't re-string-key.
_SOLAR_BOTTLENECK_TO_ENUM = {
    "memory": BottleneckType.MEMORY_BOUND,
    "compute": BottleneckType.COMPUTE_BOUND,
    "balanced": BottleneckType.BALANCED,
}


# ── dtype / arch lookup ────────────────────────────────────────────────

_TORCH_DTYPE_LITERALS = {
    "float32": "torch.float32", "fp32": "torch.float32",
    "float16": "torch.float16", "fp16": "torch.float16",
    "bfloat16": "torch.bfloat16", "bf16": "torch.bfloat16",
    "float64": "torch.float64", "fp64": "torch.float64",
    "int8": "torch.int8", "int32": "torch.int32", "int64": "torch.int64",
    "bool": "torch.bool",
}

_INT_DTYPES = {"int8", "int32", "int64"}
_BOOL_DTYPES = {"bool"}


def _torch_dtype_literal(dtype_name: str) -> str:
    """Map an ACTS/SOL-ExecBench dtype string to a torch.<dtype> literal
    safe to embed in the synthesized bridge file."""
    return _TORCH_DTYPE_LITERALS.get(dtype_name.lower(), "torch.float32")


def _tensor_constructor_call(dtype_name: str, shape_str: str) -> str:
    """Render the ``torch.<ctor>(...)`` literal for a tensor of the given
    dtype + concrete shape. ``torch.randn`` only accepts floating/complex
    dtypes; using it for int/bool raises ``RuntimeError`` deep in
    ``get_inputs()`` and silently bypasses SOLAR via the bridge soft-fall-
    back. Dispatch to dtype-appropriate constructors so SOLAR can trace
    int- and bool-input problems."""
    dtype_lit = _torch_dtype_literal(dtype_name)
    key = dtype_name.lower()
    if key in _INT_DTYPES:
        # Range [0, 2) keeps values valid for int8 too; SOLAR only inspects
        # the op graph, so the actual values don't matter.
        return f"torch.randint(0, 2, ({shape_str},), dtype={dtype_lit})"
    if key in _BOOL_DTYPES:
        return f"torch.zeros({shape_str}, dtype={dtype_lit})"
    return f"torch.randn({shape_str}, dtype={dtype_lit})"


_PRECISION_FOR_DTYPE = {
    "float32": "fp32", "fp32": "fp32",
    "float16": "fp16", "fp16": "fp16",
    "bfloat16": "bf16", "bf16": "bf16",
    "int8": "int8",
}


def _precision_for_first_input(problem: Problem) -> str:
    """Pick the SOLAR ``--precision`` flag from the first input tensor's
    dtype. SOLAR uses this to select which ``MAC_per_cycle_*`` field of
    the arch YAML to apply."""
    if not problem.inputs:
        return "fp16"
    first = next(iter(problem.inputs.values()))
    return _PRECISION_FOR_DTYPE.get(first.dtype.lower(), "fp16")


def is_solar_available() -> bool:
    """Check whether the SOLAR package is importable."""
    return _SOLAR_AVAILABLE


# ── bridge: ACTS Problem + Workload → SOLAR model file ──────────────────

def _write_model_bridge_file(
    problem: Problem, workload: Workload, target_path: Path
) -> Path:
    """Synthesize a SOLAR-compatible ``model.py`` from an ACTS Problem +
    Workload. The file contains the reference source verbatim, a
    ``class Model(nn.Module)`` whose ``forward`` calls ``run``, and a
    ``def get_inputs()`` that builds tensors per the workload's concrete
    axis values + the problem's input dtypes/shapes.

    Const axes (e.g. ``hidden_size: const 4096``) are folded in alongside
    the workload's var-axis values so every shape symbol resolves to an
    integer at synthesis time.
    """
    concrete: dict[str, int] = dict(workload.axes)
    for name, axis_def in problem.axes.items():
        if axis_def.type == "const" and axis_def.value is not None:
            concrete[name] = axis_def.value

    # Resolve ``expr`` axes (e.g. ``half_head_dim = attention_head_dim // 2``
    # in flux_rope) by fixed-point evaluation against ``concrete``. SOL-ExecBench
    # expressions are simple integer arithmetic over other axis names, so a
    # restricted ``eval`` with no builtins is sufficient and keeps the bridge
    # synchronous. Iterate until no progress to handle expr-on-expr chains.
    pending_exprs = {
        name: axis_def.expression
        for name, axis_def in problem.axes.items()
        if axis_def.type == "expr" and axis_def.expression is not None
    }
    while pending_exprs:
        progress = False
        for name in list(pending_exprs):
            try:
                value = eval(pending_exprs[name], {"__builtins__": {}}, concrete)
            except (NameError, SyntaxError):
                continue
            concrete[name] = int(value)
            del pending_exprs[name]
            progress = True
        if not progress:
            raise ValueError(
                f"unresolved expr axes {list(pending_exprs)} "
                f"(known concrete axes={sorted(concrete)})"
            )

    forward_args: list[str] = []
    inputs_lines: list[str] = []
    for tensor_name, tensor_def in problem.inputs.items():
        forward_args.append(tensor_name)
        if tensor_def.shape is None:
            # Python scalar input — placeholder 1.0 keeps SOLAR's tracer
            # happy without affecting the analyzed op graph.
            inputs_lines.append("        1.0,")
            continue
        if tensor_def.shape == []:
            # 0-D tensor — distinct from ``shape=None`` (Python scalar).
            # Emit a ``()``-shaped tensor; the dtype-aware constructor
            # avoids ``torch.randn(, ...)`` (SyntaxError) and also handles
            # int/bool dtypes that ``torch.randn`` rejects.
            inputs_lines.append(
                f"        {_tensor_constructor_call(tensor_def.dtype, '()')},"
            )
            continue
        resolved = []
        for axis in tensor_def.shape:
            if isinstance(axis, int):
                resolved.append(str(axis))
            elif axis in concrete:
                resolved.append(str(concrete[axis]))
            else:
                raise ValueError(
                    f"unresolved axis {axis!r} for input {tensor_name!r} "
                    f"(workload axes={workload.axes}, "
                    f"const axes={[n for n, a in problem.axes.items() if a.type == 'const']})"
                )
        shape_str = ", ".join(resolved)
        inputs_lines.append(
            f"        {_tensor_constructor_call(tensor_def.dtype, shape_str)},"
        )

    forward_args_str = ", ".join(forward_args)
    inputs_block = "\n".join(inputs_lines) if inputs_lines else "        # no inputs"

    source = (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        f"{problem.reference_source}\n"
        "\n"
        "class Model(nn.Module):\n"
        f"    def forward(self, {forward_args_str}):\n"
        f"        return run({forward_args_str})\n"
        "\n"
        "def get_inputs():\n"
        "    return [\n"
        f"{inputs_block}\n"
        "    ]\n"
    )
    target_path.write_text(source)
    return target_path


# ── arch resolution ─────────────────────────────────────────────────────

# Built-in SOLAR arch names that resolve internally without a path.
_SOLAR_BUNDLED_ARCHES = {"H100_PCIe", "B200"}

_ACTS_ARCH_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "arch"
_ADA_YAML = _ACTS_ARCH_DIR / "RTX6000Ada.yaml"

# ACTS-supplied arch YAMLs. When the hardware spec's name matches one of
# these, SOLAR receives the absolute YAML path so its perf model loads
# ACTS's hand-authored YAML directly. The placeholder alias covers the
# zero-peak path (no GPU / broken torch / no YAML configured) — without
# it, SOLAR's T_SOL would be computed against H100 while the in-process
# roofline already mirrors Ada peaks, silently miscalibrating sol_score.
_ACTS_ARCH_YAMLS = {
    "RTX6000Ada": _ADA_YAML,
    "NVIDIA RTX 6000 Ada Generation": _ADA_YAML,
    "placeholder-RTX6000Ada": _ADA_YAML,
}


def _resolve_arch_config(hardware_spec: HardwareSpec, arch_yaml_path: Path | None) -> str:
    """Resolve an ``arch_config`` string acceptable to
    ``EinsumGraphPerfModel.predict``. Caller-supplied path wins; else
    look up by ``hardware_spec.name``; else fall back to ``H100_PCIe``
    with a warning so SOLAR has something to load.
    """
    if arch_yaml_path is not None:
        return str(arch_yaml_path)
    if hardware_spec.name in _SOLAR_BUNDLED_ARCHES:
        return hardware_spec.name
    yaml_path = _ACTS_ARCH_YAMLS.get(hardware_spec.name)
    if yaml_path is not None and yaml_path.exists():
        return str(yaml_path)
    logger.warning(
        "no arch YAML for hardware_spec.name=%r — falling back to SOLAR's "
        "H100_PCIe profile. T_SOL will reflect H100 peaks, not the actual "
        "hardware. Author a YAML under configs/arch/<name>.yaml or pass "
        "arch_yaml_path to fix.",
        hardware_spec.name,
    )
    return "H100_PCIe"


# ── main entry point ────────────────────────────────────────────────────

def derive_t_sol(
    problem: Problem,
    workload: Workload,
    hardware_spec: HardwareSpec,
    arch_yaml_path: Path | None = None,
    roofline_model: str = "fused",
) -> SolarResult | None:
    """Derive T_SOL via SOLAR's 4-stage Python pipeline.

    Bridges the ACTS ``Problem`` + ``Workload`` to SOLAR's expected
    ``Model`` shape, drives all 4 SOLAR stages in a temp dir, and parses
    the perf YAML's ``fused`` runtime as ``T_SOL``.

    *roofline_model* selects which SOLAR variant to read — ``"fused"``
    is the documented default (intermediates stay in cache; matches what
    a well-fused Triton kernel can achieve). ``"unfused"`` is the
    worst-case bound (every tensor through DRAM); ``"fused_prefetched"``
    assumes perfect compute/memory overlap (often unreachable in Triton,
    so not the default).

    Returns ``None`` when SOLAR is not importable or when any pipeline
    stage produces no result. SOLAR's per-stage failures already log
    diagnostics; the ``None`` return signals the caller to fall back to
    the built-in roofline.
    """
    if not _SOLAR_AVAILABLE:
        return None

    arch_config = _resolve_arch_config(hardware_spec, arch_yaml_path)
    precision = _precision_for_first_input(problem)

    with tempfile.TemporaryDirectory(prefix="acts_solar_") as tmpdir:
        tmp = Path(tmpdir)
        try:
            model_file = _write_model_bridge_file(problem, workload, tmp / "model.py")
        except ValueError as exc:
            # Bridge couldn't synthesize a SOLAR model file (e.g. an axis
            # form the bridge doesn't know how to emit). Soft-fall-back to
            # the built-in roofline rather than crashing the load path.
            logger.warning(
                "SOLAR bridge failed for problem %r: %s — falling back to built-in roofline",
                problem.name, exc,
            )
            return None

        # Stage 1 — graph extraction.
        # """Configuration for processing models.
    
        # Attributes:
        #     save_graph: Whether to save graph visualizations.
        #     force_rerun: Force reprocessing even if output exists.
        #     batch_size: Number of models to process in parallel.
        #     timeout: Timeout for processing in seconds.
        #     output_dir: Directory for output files.
        #     debug: Enable debug output.
        # """
        proc = PyTorchProcessor(
            ProcessingConfig(
                save_graph=False, force_rerun=True, output_dir=str(tmp / "graph"),
                debug=False, safe_mode=False,
            )
        )
        if not proc.process_model_file(str(model_file), str(tmp / "graph")):
            logger.warning("SOLAR stage 1 (process_model_file) failed for problem %r", problem.name)
            return None

        # Stage 2 — einsum conversion.
        conv = PyTorchToEinsum(debug=False, enable_agent=False)
        einsum_dir = tmp / "einsum"
        conv_result = conv.convert(
            tmp / "graph" / "pytorch_graph.yaml", einsum_dir,
            copy_graph=False, expand_complex_ops=True, enable_rename=False,
        )
        if conv_result is None:
            logger.warning("SOLAR stage 2 (PyTorchToEinsum) failed for problem %r", problem.name)
            return None

        # Stage 3 — analysis.
        einsum_yaml = einsum_dir / "einsum_graph.yaml"
        if not einsum_yaml.exists():
            einsum_yaml = einsum_dir / "einsum_graph_renamed.yaml"
        analysis_dir = tmp / "analysis"
        analysis = EinsumGraphAnalyzer(debug=False).analyze_graph(
            einsum_yaml, analysis_dir, precision=precision, copy_graph=False,
        )
        if analysis is None:
            logger.warning("SOLAR stage 3 (EinsumGraphAnalyzer) failed for problem %r", problem.name)
            return None

        # Stage 4 — perf prediction.
        perf = EinsumGraphPerfModel(debug=False).predict(
            analysis_dir / "analysis.yaml", tmp / "perf",
            arch_config=arch_config, precision=precision, copy_analysis=False,
        )
        if perf is None:
            logger.warning("SOLAR stage 4 (EinsumGraphPerfModel) failed for problem %r", problem.name)
            return None

    section = perf.get(roofline_model, {})
    if not section:
        logger.warning("SOLAR perf YAML missing %r section for problem %r", roofline_model, problem.name)
        return None

    runtime_ms = float(section.get("runtime_ms", 0.0))
    bottleneck_str = str(section.get("bottleneck", "memory"))
    arithmetic_intensity = float(section.get("arithmetic_intensity", 0.0))

    return SolarResult(
        t_sol_us=runtime_ms * 1000.0,  # ms → us
        bottleneck=_SOLAR_BOTTLENECK_TO_ENUM.get(bottleneck_str, BottleneckType.MEMORY_BOUND),
        arithmetic_intensity=arithmetic_intensity,
        roofline_model=roofline_model,
    )
