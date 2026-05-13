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
get_inputs()``. ACTS receives the reference as a ``def run(*tensors)``
function inside a SOL ``Definition``. The bridge in
``_write_model_bridge_file`` synthesizes the SOLAR-shaped wrapper from
``definition.reference`` + a representative ``Workload``'s concrete
axis values.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.eval.types import BottleneckType

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload

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
    """T_SOL and bottleneck classification from SOLAR.

    ``arithmetic_intensity`` is in **MACs/byte** (SOLAR's native unit:
    ``total_macs / total_fused_bytes`` from the einsum analyzer). This is
    exact for MAC-dominated kernels (matmul, attention, conv) and reads
    ``0.0`` for pure-elementwise / reduction kernels (rmsnorm, softmax)
    where SOLAR's MAC counter excludes non-MAC ops cleanly.

    ``ridge_point`` is **precision-aware**: SOLAR computes it from the
    workload dtype's ``MAC_per_cycle`` (e.g. bf16 → ``MAC_per_cycle_bf16_tc``,
    int8 → ``MAC_per_cycle_int8_tc``), not from a single FP32 peak. This
    matters for tensor-core workloads where an FP32-derived ridge would
    be up to 4× too low and silently mis-classify them as compute-bound.
    Lifted directly from ``perf["arch"]["ridge_point"]``.
    """

    t_sol_us: float
    bottleneck: BottleneckType
    arithmetic_intensity: float = 0.0  # MACs/byte — see class docstring
    ridge_point: float = 0.0  # MACs/byte — precision-aware, see class docstring
    roofline_model: str = "fused"  # which SOLAR model was used
    # SOLAR-derived absolute counts (``total_flops`` = MAC×2 + non-MAC ops;
    # ``total_fused_bytes`` from the matching ``roofline_model`` section).
    # Preferred over shape formulas in ``compute_roofline_inputs`` when
    # positive; 0 means SOLAR didn't surface them and callers fall back.
    total_flops: int = 0
    total_fused_bytes: int = 0


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


def _dtype_str(dtype) -> str:
    """Coerce a SOL ``DType`` enum (or any string-like) to a lower-case
    string suitable for dtype lookups. ``DType`` is a ``str``-subclass
    enum, so ``getattr(dtype, "value", None)`` returns the underlying
    string; we fall through to ``str(dtype)`` for plain strings."""
    value = getattr(dtype, "value", None)
    return value if isinstance(value, str) else str(dtype)


def _precision_for_first_input(definition: Definition) -> str:
    """Pick the SOLAR ``--precision`` flag from the first input tensor's
    dtype. SOLAR uses this to select which ``MAC_per_cycle_*`` field of
    the arch YAML to apply."""
    if not definition.inputs:
        return "fp16"
    first = next(iter(definition.inputs.values()))
    return _PRECISION_FOR_DTYPE.get(_dtype_str(first.dtype).lower(), "fp16")


def is_solar_available() -> bool:
    """Check whether the SOLAR package is importable."""
    return _SOLAR_AVAILABLE


# ── bridge: SOL Definition + Workload → SOLAR model file ────────────────

def _write_model_bridge_file(
    definition: Definition, workload: Workload, target_path: Path
) -> Path:
    """Synthesize a SOLAR-compatible ``model.py`` from a SOL Definition +
    Workload. The file contains the reference source verbatim, a
    ``class Model(nn.Module)`` whose ``forward`` calls ``run``, and a
    ``def get_inputs()`` that builds tensors per the workload's concrete
    axis values + the definition's input dtypes/shapes.

    Uses ``definition.get_resolved_axes_values`` to fold const + expr axes
    into the workload's var-axis values in one shot, so every shape symbol
    resolves to an integer at synthesis time.
    """
    try:
        concrete = definition.get_resolved_axes_values(workload.axes)
    except (KeyError, NameError, ValueError) as exc:
        # SOL's shape-expression evaluator raises NameError for
        # unresolved symbols and ValueError for malformed expressions;
        # we collapse both at the adapter boundary so derive_t_sol can
        # fall back to the built-in roofline on either failure mode.
        raise ValueError(
            f"unresolved axes for {definition.name!r} "
            f"(workload axes={workload.axes}): {exc}"
        ) from exc

    forward_args: list[str] = []
    inputs_lines: list[str] = []
    for tensor_name, tensor_spec in definition.inputs.items():
        forward_args.append(tensor_name)
        dtype_str = _dtype_str(tensor_spec.dtype)
        if tensor_spec.shape is None:
            # Python scalar input — placeholder 1.0 keeps SOLAR's tracer
            # happy without affecting the analyzed op graph.
            inputs_lines.append("        1.0,")
            continue
        if tensor_spec.shape == []:
            # 0-D tensor — distinct from ``shape=None`` (Python scalar).
            # Emit a ``()``-shaped tensor; the dtype-aware constructor
            # avoids ``torch.randn(, ...)`` (SyntaxError) and also handles
            # int/bool dtypes that ``torch.randn`` rejects.
            inputs_lines.append(
                f"        {_tensor_constructor_call(dtype_str, '()')},"
            )
            continue
        resolved = []
        for axis in tensor_spec.shape:
            if isinstance(axis, int) or (isinstance(axis, str) and axis.isdigit()):
                resolved.append(str(axis))
            elif axis in concrete:
                resolved.append(str(concrete[axis]))
            else:
                raise ValueError(
                    f"unresolved axis {axis!r} for input {tensor_name!r} "
                    f"(workload axes={workload.axes}, "
                    f"resolved={sorted(concrete)})"
                )
        shape_str = ", ".join(resolved)
        inputs_lines.append(
            f"        {_tensor_constructor_call(dtype_str, shape_str)},"
        )

    forward_args_str = ", ".join(forward_args)
    inputs_block = "\n".join(inputs_lines) if inputs_lines else "        # no inputs"

    source = (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        f"{definition.reference}\n"
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

# Single source of truth lives in src/config.py — re-export so existing
# imports (``from src.benchmark.solar_adapter import _ACTS_ARCH_YAMLS``)
# keep working without duplicating the registry.
from src.config import _ACTS_ARCH_DIR, _ACTS_ARCH_YAMLS  # noqa: E402  (re-export)
from src.config import _ADA_YAML  # noqa: E402  (re-export)
from src.config import _lookup_arch_yaml  # noqa: E402


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
    yaml_path = _lookup_arch_yaml(hardware_spec.name)
    if yaml_path is not None:
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
    definition: Definition,
    workload: Workload,
    hardware_spec: HardwareSpec,
    arch_yaml_path: Path | None = None,
    roofline_model: str = "fused",
) -> SolarResult | None:
    """Derive T_SOL via SOLAR's 4-stage Python pipeline.

    Bridges the SOL ``Definition`` + ``Workload`` to SOLAR's expected
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
    precision = _precision_for_first_input(definition)

    with tempfile.TemporaryDirectory(prefix="acts_solar_") as tmpdir:
        tmp = Path(tmpdir)
        try:
            model_file = _write_model_bridge_file(definition, workload, tmp / "model.py")
        except ValueError as exc:
            # Bridge couldn't synthesize a SOLAR model file (e.g. an axis
            # form the bridge doesn't know how to emit). Soft-fall-back to
            # the built-in roofline rather than crashing the load path.
            logger.warning(
                "SOLAR bridge failed for definition %r: %s — falling back to built-in roofline",
                definition.name, exc,
            )
            return None

        # Stage 1 — graph extraction.
        proc = PyTorchProcessor(
            ProcessingConfig(
                save_graph=False, force_rerun=True, output_dir=str(tmp / "graph"),
                debug=False, safe_mode=False,
            )
        )
        if not proc.process_model_file(str(model_file), str(tmp / "graph")):
            logger.warning("SOLAR stage 1 (process_model_file) failed for problem %r", definition.name)
            return None

        # Stage 2 — einsum conversion.
        conv = PyTorchToEinsum(debug=False, enable_agent=False)
        einsum_dir = tmp / "einsum"
        conv_result = conv.convert(
            tmp / "graph" / "pytorch_graph.yaml", einsum_dir,
            copy_graph=False, expand_complex_ops=True, enable_rename=False,
        )
        if conv_result is None:
            logger.warning("SOLAR stage 2 (PyTorchToEinsum) failed for problem %r", definition.name)
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
            logger.warning("SOLAR stage 3 (EinsumGraphAnalyzer) failed for problem %r", definition.name)
            return None

        # Stage 4 — perf prediction.
        perf = EinsumGraphPerfModel(debug=False).predict(
            analysis_dir / "analysis.yaml", tmp / "perf",
            arch_config=arch_config, precision=precision, copy_analysis=False,
        )
        if perf is None:
            logger.warning("SOLAR stage 4 (EinsumGraphPerfModel) failed for problem %r", definition.name)
            return None

    section = perf.get(roofline_model, {})
    if not section:
        logger.warning("SOLAR perf YAML missing %r section for problem %r", roofline_model, definition.name)
        return None

    runtime_ms = float(section.get("runtime_ms", 0.0))
    bottleneck_str = str(section.get("bottleneck", "memory"))
    # SOLAR's per-section ``arithmetic_intensity`` is MACs/byte
    # (``total_macs / <section>_bytes``) — see solar/perf/perf_model.py.
    # 0.0 is a valid value for non-MAC kernels (rmsnorm, softmax,
    # reductions) where SOLAR's einsum analyzer reports no MAC ops.
    arithmetic_intensity = float(section.get("arithmetic_intensity", 0.0))
    # SOLAR's ``perf["arch"]["ridge_point"]`` is the precision-aware ridge
    # ``MAC_per_cycle / DRAM_byte_per_cycle`` where ``MAC_per_cycle`` is
    # selected by the workload's dtype (fp32_sm, bf16_tc, fp16_tc, int8_tc,
    # ...). Reading it here keeps us aligned with whatever precision SOLAR
    # actually used for compute_cycles — see solar/perf/perf_model.py L232.
    arch = perf.get("arch", {})
    ridge_point = float(arch.get("ridge_point", 0.0))

    # ``memory_bytes`` read from the SAME section as ``runtime_ms`` so AI
    # stays self-consistent (fused vs unfused differ). ``.get(..., 0)``
    # tolerates SOLAR schema drift — falls back to shape formulas.
    workload_section = perf.get("workload", {})
    total_flops = int(workload_section.get("total_flops", 0))
    total_fused_bytes = int(section.get("memory_bytes", 0))

    return SolarResult(
        t_sol_us=runtime_ms * 1000.0,  # ms → us
        bottleneck=_SOLAR_BOTTLENECK_TO_ENUM.get(bottleneck_str, BottleneckType.MEMORY_BOUND),
        arithmetic_intensity=arithmetic_intensity,
        ridge_point=ridge_point,
        roofline_model=roofline_model,
        total_flops=total_flops,
        total_fused_bytes=total_fused_bytes,
    )
