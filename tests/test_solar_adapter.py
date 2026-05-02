"""Tests for src/benchmark/solar_adapter.py — bridge + arch resolution.

Tier 1 (pure logic): bridge synthesis, dtype mapping, arch resolution,
SOLAR-absent guard. The full 4-stage pipeline drive lives in the GPU
test suite (requires SOLAR + torch installed).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sol_execbench.core.data import Definition, Workload

from src.benchmark.solar_adapter import (
    _ACTS_ARCH_YAMLS,
    _precision_for_first_input,
    _resolve_arch_config,
    _torch_dtype_literal,
    _write_model_bridge_file,
    derive_t_sol,
    is_solar_available,
)
from src.config import HardwareSpec


# ── pure helpers ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected",
    [
        ("float32", "torch.float32"),
        ("FP32", "torch.float32"),
        ("bfloat16", "torch.bfloat16"),
        ("bf16", "torch.bfloat16"),
        ("fp16", "torch.float16"),
        ("int8", "torch.int8"),
        ("unknown_dtype", "torch.float32"),  # safe fallback
    ],
)
def test_torch_dtype_literal_maps_known_and_unknown(name, expected):
    assert _torch_dtype_literal(name) == expected


def _definition_with_input_dtype(dtype: str) -> Definition:
    # SOL's DType enum doesn't include "unknown"; for that case, validate as
    # float32 then mutate the spec dtype to the unrecognized string so the
    # adapter's fallback path is exercised. Pydantic models are mutable by
    # default, so direct attribute assignment works.
    if dtype == "unknown":
        d = Definition.model_validate({
            "name": "t",
            "axes": {"n": {"type": "var"}},
            "inputs": {"x": {"shape": ["n"], "dtype": "float32"}},
            "outputs": {"y": {"shape": ["n"], "dtype": "float32"}},
            "reference": "def run(x): return x\n",
        })
        d.inputs["x"].dtype = "unknown"
        return d
    return Definition.model_validate({
        "name": "t",
        "axes": {"n": {"type": "var"}},
        "inputs": {"x": {"shape": ["n"], "dtype": dtype}},
        "outputs": {"y": {"shape": ["n"], "dtype": dtype}},
        "reference": "def run(x): return x\n",
    })


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("float32", "fp32"), ("bfloat16", "bf16"), ("float16", "fp16"),
        ("int8", "int8"), ("unknown", "fp16"),  # fallback
    ],
)
def test_precision_for_first_input(dtype, expected):
    assert _precision_for_first_input(_definition_with_input_dtype(dtype)) == expected


def test_precision_for_first_input_no_inputs_defaults_fp16():
    definition = Definition.model_validate({
        "name": "t",
        "axes": {},
        "inputs": {},
        "outputs": {"y": {"shape": ["1"], "dtype": "float32"}},
        "reference": "def run(): pass\n",
    })
    assert _precision_for_first_input(definition) == "fp16"


# ── arch resolution ───────────────────────────────────────────────────


def test_resolve_arch_config_explicit_path_wins():
    spec = HardwareSpec(name="anything")
    assert _resolve_arch_config(spec, Path("/tmp/foo.yaml")) == "/tmp/foo.yaml"


def test_resolve_arch_config_bundled_name_passes_through():
    spec = HardwareSpec(name="H100_PCIe")
    assert _resolve_arch_config(spec, None) == "H100_PCIe"


def test_resolve_arch_config_acts_yaml_resolves_to_path():
    spec = HardwareSpec(name="RTX6000Ada")
    resolved = _resolve_arch_config(spec, None)
    expected = _ACTS_ARCH_YAMLS["RTX6000Ada"]
    assert resolved == str(expected)
    assert expected.exists(), "configs/arch/RTX6000Ada.yaml must exist for the lookup"


def test_resolve_arch_config_unknown_falls_back_to_h100_with_warning(caplog):
    spec = HardwareSpec(name="some_random_gpu")
    with caplog.at_level("WARNING", logger="src.benchmark.solar_adapter"):
        resolved = _resolve_arch_config(spec, None)
    assert resolved == "H100_PCIe"
    assert any("no arch YAML" in rec.message for rec in caplog.records)


def test_resolve_arch_config_placeholder_name_resolves_to_ada_yaml():
    """The placeholder hardware spec used when peaks are zero
    (``placeholder-RTX6000Ada``) must resolve to the Ada YAML — otherwise
    SOLAR computes T_SOL against H100 peaks while the in-process built-in
    roofline uses RTX 6000 Ada peaks, silently miscalibrating sol_score
    on the default no-YAML smoke-run path."""
    spec = HardwareSpec(name="placeholder-RTX6000Ada")
    resolved = _resolve_arch_config(spec, None)
    expected = _ACTS_ARCH_YAMLS["placeholder-RTX6000Ada"]
    assert resolved == str(expected)
    assert expected.exists()


# ── bridge file synthesis ──────────────────────────────────────────────


def _rmsnorm_definition() -> Definition:
    """Mirror the SOL-ExecBench rmsnorm definition: const hidden_size,
    var batch_size, two bf16 inputs, scalar EPS folded into the source."""
    return Definition.model_validate({
        "name": "rmsnorm_h4096",
        "op_type": "rmsnorm",
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
        "reference": (
            "import torch\n"
            "@torch.no_grad()\n"
            "def run(hidden_states, weight):\n"
            "    return hidden_states * weight\n"
        ),
    })


def test_write_model_bridge_file_synthesizes_valid_python(tmp_path):
    definition = _rmsnorm_definition()
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {"batch_size": 7}, "inputs": {}
    })
    out = _write_model_bridge_file(definition, workload, tmp_path / "model.py")

    src = out.read_text()
    # Reference body inlined verbatim
    assert "def run(hidden_states, weight):" in src
    assert "return hidden_states * weight" in src
    # Model wrapper present with both args in forward signature
    assert "class Model(nn.Module):" in src
    assert "def forward(self, hidden_states, weight):" in src
    # get_inputs builds two tensors with concrete shapes folding both
    # the workload's var axis (batch_size=7) and the const axis (hidden_size=4096)
    assert "torch.randn(7, 4096, dtype=torch.bfloat16)" in src
    assert "torch.randn(4096, dtype=torch.bfloat16)" in src
    # Synthesized file must be syntactically valid Python
    compile(src, str(out), "exec")


def test_write_model_bridge_file_unresolved_axis_raises(tmp_path):
    """If the workload doesn't supply a value for a var axis the input
    shape references, the synth must fail loudly — silently leaving an
    unresolved symbol in get_inputs() would crash SOLAR's tracer."""
    definition = Definition.model_validate({
        "name": "t",
        "axes": {"unknown_axis": {"type": "var"}},
        "inputs": {"x": {"shape": ["unknown_axis"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["unknown_axis"], "dtype": "float32"}},
        "reference": "def run(x): return x\n",
    })
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {}, "inputs": {}
    })  # no value for unknown_axis
    with pytest.raises(ValueError, match="unresolved axis"):
        _write_model_bridge_file(definition, workload, tmp_path / "model.py")


def test_write_model_bridge_file_resolves_expr_axis(tmp_path):
    """Real SOL-ExecBench problems (e.g. flux_rope) define ``expr`` axes
    like ``half_head_dim = attention_head_dim // 2``. The bridge must
    evaluate them against the concrete environment so SOLAR sees integer
    shapes — otherwise valid problems crash Phase A."""
    definition = Definition.model_validate({
        "name": "t",
        "axes": {
            "attention_head_dim": {"type": "const", "value": 128},
            "half_head_dim": {"type": "expr", "expression": "attention_head_dim // 2"},
            "batch": {"type": "var"},
        },
        "inputs": {
            "x": {"shape": ["batch", "half_head_dim"], "dtype": "float32"},
        },
        "outputs": {
            "y": {"shape": ["batch", "half_head_dim"], "dtype": "float32"},
        },
        "reference": "def run(x): return x\n",
    })
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {"batch": 8}, "inputs": {}
    })
    out = _write_model_bridge_file(definition, workload, tmp_path / "model.py")
    src = out.read_text()
    assert "torch.randn(8, 64, dtype=torch.float32)" in src
    compile(src, str(out), "exec")


def test_write_model_bridge_file_resolves_chained_expr_axes(tmp_path):
    """Expr axes can depend on other expr axes; the resolver must reach
    fixed-point rather than failing on declaration order."""
    definition = Definition.model_validate({
        "name": "t",
        "axes": {
            "n": {"type": "var"},
            "n2": {"type": "expr", "expression": "n * 2"},
            "n4": {"type": "expr", "expression": "n2 * 2"},
        },
        "inputs": {"x": {"shape": ["n4"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["n4"], "dtype": "float32"}},
        "reference": "def run(x): return x\n",
    })
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {"n": 5}, "inputs": {}
    })
    out = _write_model_bridge_file(definition, workload, tmp_path / "model.py")
    src = out.read_text()
    assert "torch.randn(20, dtype=torch.float32)" in src


def test_write_model_bridge_file_unresolvable_expr_axis_raises(tmp_path):
    """An expr axis whose dependencies aren't satisfied must raise — the
    caller in ``derive_t_sol`` catches this and falls back to the built-in
    roofline. Silent emission of an unresolved symbol would crash SOLAR
    deeper in the pipeline with a less actionable error."""
    definition = Definition.model_validate({
        "name": "t",
        "axes": {
            "missing": {"type": "var"},  # not supplied by workload
            "derived": {"type": "expr", "expression": "missing * 2"},
        },
        "inputs": {"x": {"shape": ["derived"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["derived"], "dtype": "float32"}},
        "reference": "def run(x): return x\n",
    })
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {}, "inputs": {}
    })
    with pytest.raises(ValueError, match="unresolved"):
        _write_model_bridge_file(definition, workload, tmp_path / "model.py")


def test_write_model_bridge_file_int_dtype_uses_randint(tmp_path):
    """Integer dtypes must NOT be emitted as ``torch.randn(...)`` — that
    raises ``RuntimeError`` for non-floating dtypes and would silently
    bypass SOLAR via the bridge soft-fallback. Use ``torch.randint`` for
    int dtypes so SOLAR can actually trace integer-input problems."""
    definition = Definition.model_validate({
        "name": "t",
        "axes": {"n": {"type": "var"}},
        "inputs": {"idx": {"shape": ["n"], "dtype": "int32"}},
        "outputs": {"out": {"shape": ["n"], "dtype": "int32"}},
        "reference": "def run(idx): return idx\n",
    })
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {"n": 8}, "inputs": {}
    })
    out = _write_model_bridge_file(definition, workload, tmp_path / "model.py")
    src = out.read_text()
    assert "torch.randint" in src
    assert "dtype=torch.int32" in src
    assert "torch.randn(8, dtype=torch.int32" not in src
    compile(src, str(out), "exec")


def test_write_model_bridge_file_bool_dtype_uses_zeros(tmp_path):
    """Bool dtype must NOT be emitted as ``torch.randn(...)``. Use
    ``torch.zeros(..., dtype=torch.bool)`` so the synthesized ``get_inputs()``
    actually executes."""
    definition = Definition.model_validate({
        "name": "t",
        "axes": {"n": {"type": "var"}},
        "inputs": {"mask": {"shape": ["n"], "dtype": "bool"}},
        "outputs": {"out": {"shape": ["n"], "dtype": "bool"}},
        "reference": "def run(mask): return mask\n",
    })
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {"n": 4}, "inputs": {}
    })
    out = _write_model_bridge_file(definition, workload, tmp_path / "model.py")
    src = out.read_text()
    assert "torch.zeros(4, dtype=torch.bool)" in src
    assert "torch.randn" not in src
    compile(src, str(out), "exec")


def test_write_model_bridge_file_zero_d_tensor_input(tmp_path):
    """``shape=[]`` is a 0-D tensor (distinct from ``shape=None`` which is
    a Python scalar). The bridge must emit ``torch.randn((), dtype=...)``,
    not ``torch.randn(, dtype=...)`` — the latter is a SyntaxError."""
    definition = Definition.model_validate({
        "name": "t",
        "axes": {},
        "inputs": {"alpha": {"shape": [], "dtype": "float32"}},
        "outputs": {"out": {"shape": [], "dtype": "float32"}},
        "reference": "def run(alpha): return alpha\n",
    })
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {}, "inputs": {}
    })
    out = _write_model_bridge_file(definition, workload, tmp_path / "model.py")
    src = out.read_text()
    assert "torch.randn((), dtype=torch.float32)" in src
    # Critical: synthesized file must compile (the bug emits invalid Python).
    compile(src, str(out), "exec")


def test_write_model_bridge_file_scalar_input_uses_placeholder(tmp_path):
    """Tensor with shape=None is a Python scalar; bridge emits a 1.0
    placeholder so SOLAR's tracer doesn't choke on a missing arg."""
    definition = Definition.model_validate({
        "name": "t",
        "axes": {"n": {"type": "var"}},
        "inputs": {
            "x": {"shape": ["n"], "dtype": "float32"},
            "eps": {"shape": None, "dtype": "float32"},
        },
        "outputs": {"out": {"shape": ["n"], "dtype": "float32"}},
        "reference": "def run(x, eps): return x * eps\n",
    })
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {"n": 4}, "inputs": {}
    })
    out = _write_model_bridge_file(definition, workload, tmp_path / "model.py")
    src = out.read_text()
    assert "torch.randn(4, dtype=torch.float32)" in src
    assert "1.0," in src  # scalar placeholder emitted on its own line


# ── SOLAR-absent guard ─────────────────────────────────────────────────


def test_derive_t_sol_returns_none_when_solar_unavailable():
    """Adapter must short-circuit cleanly when SOLAR isn't importable —
    the no-SOLAR fallback path in roofline.py depends on this contract."""
    definition = _rmsnorm_definition()
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {"batch_size": 7}, "inputs": {}
    })
    spec = HardwareSpec(name="RTX6000Ada")
    with patch("src.benchmark.solar_adapter._SOLAR_AVAILABLE", False):
        result = derive_t_sol(definition, workload, spec)
    assert result is None


def test_is_solar_available_is_a_bool():
    """Public availability check must always return a bool — no truthy
    side-channel through SOLAR import objects."""
    assert isinstance(is_solar_available(), bool)


def test_derive_t_sol_soft_fails_on_bridge_value_error():
    """Bridge ValueError (e.g. unresolvable expr axis on a future schema
    addition) must downgrade to ``None`` so the caller falls back to the
    built-in roofline rather than crashing the whole load path."""
    definition = _rmsnorm_definition()
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {"batch_size": 7}, "inputs": {}
    })
    spec = HardwareSpec(name="RTX6000Ada")
    with (
        patch("src.benchmark.solar_adapter._SOLAR_AVAILABLE", True),
        patch(
            "src.benchmark.solar_adapter._write_model_bridge_file",
            side_effect=ValueError("unresolved axis 'mystery'"),
        ),
    ):
        result = derive_t_sol(definition, workload, spec)
    assert result is None


# ── perf["arch"]["ridge_point"] propagation ────────────────────────────


def test_derive_t_sol_populates_ridge_point_from_perf_arch():
    """``SolarResult.ridge_point`` must come from
    ``perf["arch"]["ridge_point"]`` — SOLAR's precision-aware value, which
    uses the workload-dtype's ``MAC_per_cycle`` (e.g. ``bf16_tc``) rather
    than a single FP32 peak. Without this, tensor-core workloads get an
    FP32-derived ridge that's up to 4× too low and silently classify as
    compute-bound."""
    definition = _rmsnorm_definition()
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {"batch_size": 7}, "inputs": {}
    })
    spec = HardwareSpec(name="RTX6000Ada")

    # bf16 tensor-core ridge for RTX 6000 Ada is ~190 MACs/byte —
    # distinct from the FP32-derived ~47.5 the old code path produced,
    # so a wrong-source bug would be visible in the assertion.
    fake_perf = {
        "arch": {"name": "RTX6000Ada", "ridge_point": 189.8},
        "fused": {
            "runtime_ms": 0.5,
            "bottleneck": "memory",
            "arithmetic_intensity": 12.5,
        },
    }

    # Stub each SOLAR stage so we never actually exercise the real
    # pipeline. Stages 1–3 just need a truthy return; stage 4 returns
    # the fake perf dict.
    proc_stub = type("P", (), {
        "process_model_file": lambda self, *a, **k: True,
    })
    conv_stub = type("C", (), {
        "convert": lambda self, *a, **k: object(),
    })
    analyzer_stub = type("A", (), {
        "analyze_graph": lambda self, *a, **k: object(),
    })
    perf_stub = type("M", (), {
        "predict": lambda self, *a, **k: fake_perf,
    })

    # Patch ``Path.exists`` for the einsum_yaml lookup so stage 3's
    # output path doesn't need to physically exist.
    with (
        patch("src.benchmark.solar_adapter._SOLAR_AVAILABLE", True),
        patch("src.benchmark.solar_adapter.PyTorchProcessor", lambda *a, **k: proc_stub()),
        patch("src.benchmark.solar_adapter.PyTorchToEinsum", lambda *a, **k: conv_stub()),
        patch("src.benchmark.solar_adapter.EinsumGraphAnalyzer", lambda *a, **k: analyzer_stub()),
        patch("src.benchmark.solar_adapter.EinsumGraphPerfModel", lambda *a, **k: perf_stub()),
        patch("src.benchmark.solar_adapter.ProcessingConfig", lambda *a, **k: None),
        patch.object(Path, "exists", lambda self: True),
    ):
        result = derive_t_sol(definition, workload, spec)

    assert result is not None
    assert result.ridge_point == pytest.approx(189.8)
    # Sanity: arithmetic_intensity still flows through too (fixture
    # value distinct from default 0.0).
    assert result.arithmetic_intensity == pytest.approx(12.5)


def test_derive_t_sol_ridge_point_defaults_to_zero_when_arch_missing():
    """If SOLAR's ``perf`` dict somehow lacks ``arch.ridge_point`` (older
    SOLAR build, partial dict), default cleanly to 0.0 rather than
    raising — the caller's downstream classifier handles 0-ridge gracefully."""
    definition = _rmsnorm_definition()
    workload = Workload.model_validate({
        "uuid": "w1", "axes": {"batch_size": 7}, "inputs": {}
    })
    spec = HardwareSpec(name="RTX6000Ada")

    fake_perf = {
        # No "arch" key at all
        "fused": {
            "runtime_ms": 0.5,
            "bottleneck": "memory",
            "arithmetic_intensity": 0.0,
        },
    }

    proc_stub = type("P", (), {"process_model_file": lambda self, *a, **k: True})
    conv_stub = type("C", (), {"convert": lambda self, *a, **k: object()})
    analyzer_stub = type("A", (), {"analyze_graph": lambda self, *a, **k: object()})
    perf_stub = type("M", (), {"predict": lambda self, *a, **k: fake_perf})

    with (
        patch("src.benchmark.solar_adapter._SOLAR_AVAILABLE", True),
        patch("src.benchmark.solar_adapter.PyTorchProcessor", lambda *a, **k: proc_stub()),
        patch("src.benchmark.solar_adapter.PyTorchToEinsum", lambda *a, **k: conv_stub()),
        patch("src.benchmark.solar_adapter.EinsumGraphAnalyzer", lambda *a, **k: analyzer_stub()),
        patch("src.benchmark.solar_adapter.EinsumGraphPerfModel", lambda *a, **k: perf_stub()),
        patch("src.benchmark.solar_adapter.ProcessingConfig", lambda *a, **k: None),
        patch.object(Path, "exists", lambda self: True),
    ):
        result = derive_t_sol(definition, workload, spec)

    assert result is not None
    assert result.ridge_point == 0.0
