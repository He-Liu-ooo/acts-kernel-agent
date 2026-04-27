"""Tests for config.py — HardwareSpec from SOLAR arch YAML."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config import HardwareSpec, detect_hardware, load_hardware_spec, validate_hardware_spec


_H100_YAML = """\
name: "H100_PCIe"
SRAM_capacity: 52428800
SRAM_byte_per_cycle: 10000
DRAM_capacity: 85899345920
DRAM_byte_per_cycle: 1019.4
freq_GHz: 2
MAC_per_cycle_fp32_sm: 25500
MAC_per_cycle_int8_tc: 756000
MAC_per_cycle_fp8_tc: 756000
MAC_per_cycle_fp16_tc: 378000
MAC_per_cycle_bf16_tc: 378000
MAC_per_cycle_tf32_tc: 189000
"""


def test_load_hardware_spec_from_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(_H100_YAML)
        f.flush()
        spec = load_hardware_spec(Path(f.name))

    assert spec.name == "H100_PCIe"
    assert spec.freq_GHz == 2.0
    assert spec.DRAM_capacity == 85899345920
    assert spec.MAC_per_cycle_bf16_tc == 378000


def test_derived_peak_bandwidth():
    spec = load_hardware_spec(_write_yaml(_H100_YAML))
    # DRAM_byte_per_cycle * freq_GHz = 1019.4 * 2 = 2038.8 GB/s
    assert abs(spec.peak_memory_bandwidth_gb_s - 2038.8) < 0.1


def test_derived_peak_flops_fp32():
    spec = load_hardware_spec(_write_yaml(_H100_YAML))
    # MAC_per_cycle_fp32_sm * freq_GHz * 2 / 1e6 = 25500 * 2 * 2 / 1e6 = 0.102 TFLOPS
    assert abs(spec.peak_flops_fp32 - 0.102) < 0.001


def test_derived_peak_flops_bf16():
    spec = load_hardware_spec(_write_yaml(_H100_YAML))
    # MAC_per_cycle_bf16_tc * freq_GHz * 2 / 1e6 = 378000 * 2 * 2 / 1e6 = 1.512 TFLOPS
    assert abs(spec.peak_flops_bf16 - 1.512) < 0.001


def test_missing_nvfp4_defaults_to_zero():
    """H100 YAML has no MAC_per_cycle_nvfp4_tc — should default to 0."""
    spec = load_hardware_spec(_write_yaml(_H100_YAML))
    assert spec.MAC_per_cycle_nvfp4_tc == 0.0


def test_default_hardware_spec_all_zero():
    spec = HardwareSpec()
    assert spec.peak_flops_fp32 == 0.0
    assert spec.peak_memory_bandwidth_gb_s == 0.0


def _write_yaml(content: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        return Path(f.name)


# ── detect_hardware() ──────────────────────────────────────────────────


def _fake_torch(*, cuda_available: bool, device_count: int = 1, props=None):
    """Build a fake ``torch`` module mimicking the surface ``detect_hardware``
    touches. ``props`` is the value returned by
    ``torch.cuda.get_device_properties(0)``."""
    cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: device_count,
        get_device_properties=lambda _idx: props,
    )
    return SimpleNamespace(cuda=cuda)


def test_detect_hardware_no_torch_returns_zeroed_spec():
    """If ``import torch`` raises (CPU-only env), return zeroed HardwareSpec
    so callers fall through to the YAML / placeholder path without error."""
    with patch.dict(sys.modules, {"torch": None}):
        spec = detect_hardware()
    assert spec == HardwareSpec()


def test_detect_hardware_no_gpu_returns_zeroed_spec():
    """If torch imports but CUDA isn't available, return zeroed spec."""
    fake = _fake_torch(cuda_available=False)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()
    assert spec == HardwareSpec()


def test_detect_hardware_zero_devices_returns_zeroed_spec():
    """Defensive: ``is_available()`` true but ``device_count()`` zero."""
    fake = _fake_torch(cuda_available=True, device_count=0)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()
    assert spec == HardwareSpec()


def test_detect_hardware_with_gpu_populates_runtime_fields():
    """With a real GPU, populate name / freq_GHz / SRAM_capacity /
    DRAM_capacity from torch.cuda.get_device_properties. Per-precision
    throughput tables stay at zero — those need a SOLAR arch YAML."""
    props = SimpleNamespace(
        name="NVIDIA RTX 6000 Ada Generation",
        clock_rate=2_505_000,        # kHz → 2.505 GHz
        L2_cache_size=100_663_296,
        total_memory=50_876_841_984,
    )
    fake = _fake_torch(cuda_available=True, props=props)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()

    assert spec.name == "NVIDIA RTX 6000 Ada Generation"
    assert abs(spec.freq_GHz - 2.505) < 1e-6
    assert spec.SRAM_capacity == 100_663_296
    assert spec.DRAM_capacity == 50_876_841_984
    # Per-precision tables intentionally left at zero.
    assert spec.MAC_per_cycle_fp32_sm == 0.0
    assert spec.MAC_per_cycle_bf16_tc == 0.0
    assert spec.DRAM_byte_per_cycle == 0.0
    assert spec.SRAM_byte_per_cycle == 0.0


def test_detect_hardware_torch_probe_failure_returns_zeroed_spec():
    """If get_device_properties raises (driver mismatch, etc.), don't
    crash the whole load path — return zeroed spec."""
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=MagicMock(side_effect=RuntimeError("driver mismatch")),
    )
    fake = SimpleNamespace(cuda=cuda)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()
    assert spec == HardwareSpec()


@pytest.mark.parametrize(
    "exc",
    [
        OSError("libcuda.so.1: cannot open shared object file"),
        RuntimeError("torch ABI mismatch"),
    ],
)
def test_detect_hardware_broken_torch_import_returns_zeroed_spec(exc):
    """Broken torch installs (missing CUDA shared lib, ABI mismatch) raise
    OSError/RuntimeError from ``import torch`` — not just ImportError. The
    docstring promises a zeroed HardwareSpec fallback for "torch cannot be
    imported"; the catch must be wide enough to honor that contract,
    otherwise ``load_config()`` crashes on broken-driver dev boxes."""
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "torch":
            raise exc
        return real_import(name, *args, **kwargs)

    # Ensure the `torch` slot is unpopulated so the import statement
    # actually re-runs `__import__` instead of returning a cached module.
    with patch.dict(sys.modules, {}, clear=False):
        sys.modules.pop("torch", None)
        with patch.object(builtins, "__import__", side_effect=_import):
            spec = detect_hardware()
    assert spec == HardwareSpec()


# ── reviewer multi-turn flag ──────────────────────────────────────────


def test_acts_config_reviewer_metric_queries_default_false():
    """`reviewer_metric_queries` defaults to False — multi-turn Reviewer is
    opt-in. Existing single-call path is the verified default."""
    from src.config import ACTSConfig

    cfg = ACTSConfig()
    assert cfg.reviewer_metric_queries is False


def test_load_config_reviewer_metric_queries_from_ini():
    """`load_config` parses [search] reviewer_metric_queries via the
    existing boolean coercion path used by `beam_diversity`."""
    from src.config import load_config

    ini = """\
[search]
reviewer_metric_queries = true
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(ini)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.reviewer_metric_queries is True


def test_load_config_reviewer_metric_queries_omitted_uses_default():
    """When [search] omits the key, fall back to the dataclass default."""
    from src.config import load_config

    ini = """\
[search]
beam_width = 5
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(ini)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.reviewer_metric_queries is False


# ── validate_hardware_spec() ───────────────────────────────────────────


def test_validate_hardware_spec_no_detected_dram_skips_checks():
    """If no GPU was detected (or detection failed), DRAM_capacity is 0
    and we have nothing to validate against — return [] rather than
    flagging every config as a mismatch."""
    spec = HardwareSpec(name="RTX6000Ada", DRAM_capacity=48 * 1024**3)
    detected = HardwareSpec()  # all zeros
    assert validate_hardware_spec(spec, detected) == []


def test_validate_hardware_spec_no_spec_dram_skips_checks():
    """If the source spec has no DRAM info (e.g. partial/identity-only
    spec), there's nothing meaningful to compare — skip rather than warn."""
    spec = HardwareSpec(name="something", DRAM_capacity=0)
    detected = HardwareSpec(name="x", DRAM_capacity=80 * 1024**3)
    assert validate_hardware_spec(spec, detected) == []


def test_validate_hardware_spec_dram_match_within_tolerance_passes():
    """torch.cuda's reported total_memory is slightly under nameplate
    (some bytes reserved). 10% tolerance keeps the check robust."""
    spec = HardwareSpec(DRAM_capacity=51_539_607_552)         # 48 GiB exact
    detected = HardwareSpec(DRAM_capacity=50_876_841_984)     # ~47.4 GiB
    assert validate_hardware_spec(spec, detected) == []


def test_validate_hardware_spec_dram_mismatch_returns_message():
    """48 GiB (Ada) vs 80 GiB (H100) is the canonical wrong-YAML failure
    mode that silently miscalibrates T_SOL math."""
    spec = HardwareSpec(name="RTX6000Ada", DRAM_capacity=48 * 1024**3)
    detected = HardwareSpec(name="NVIDIA H100", DRAM_capacity=80 * 1024**3)
    issues = validate_hardware_spec(spec, detected)
    assert len(issues) == 1
    assert "DRAM capacity mismatch" in issues[0]
    assert "48" in issues[0] and "80" in issues[0]


def test_validate_hardware_spec_l2_mismatch_returns_message():
    """SRAM_capacity (L2 cache) is the discriminator that DRAM can't
    catch on its own — Ada and L40S both have 48 GiB DRAM, but Ada has
    96 MB L2 vs L40S's 96 MB (same), Ada vs H100 is 96 MB vs 50 MB.
    Catches wrong-YAML where DRAM happens to match."""
    spec = HardwareSpec(SRAM_capacity=96 * 1024 * 1024, DRAM_capacity=48 * 1024**3)
    detected = HardwareSpec(SRAM_capacity=50 * 1024 * 1024, DRAM_capacity=48 * 1024**3)
    issues = validate_hardware_spec(spec, detected)
    assert any("SRAM" in m or "L2" in m for m in issues)


def test_validate_hardware_spec_freq_mismatch_returns_message():
    """``torch.cuda.get_device_properties(0).clock_rate`` reports the
    boost clock; the YAML's ``freq_GHz`` is also boost. A 30% delta is
    well outside driver-reported precision — flag it."""
    spec = HardwareSpec(freq_GHz=2.5, DRAM_capacity=48 * 1024**3)
    detected = HardwareSpec(freq_GHz=1.5, DRAM_capacity=48 * 1024**3)
    issues = validate_hardware_spec(spec, detected)
    assert any("freq" in m or "frequency" in m.lower() for m in issues)


def test_validate_hardware_spec_all_three_mismatch():
    """When every overlapping field disagrees (clearly the wrong YAML
    for this GPU), report all three so the user sees the full picture
    rather than fixing one and re-running into the next."""
    spec = HardwareSpec(
        name="RTX6000Ada",
        freq_GHz=2.5,
        SRAM_capacity=96 * 1024 * 1024,
        DRAM_capacity=48 * 1024**3,
    )
    detected = HardwareSpec(
        name="NVIDIA H100 PCIe",
        freq_GHz=1.98,
        SRAM_capacity=50 * 1024 * 1024,
        DRAM_capacity=80 * 1024**3,
    )
    issues = validate_hardware_spec(spec, detected)
    assert len(issues) == 3


def test_load_config_warns_on_yaml_vs_detected_mismatch(caplog):
    """When ``arch_config_path`` points at a YAML for hardware that
    doesn't match the actually-detected GPU, ``load_config`` must log a
    warning so the user sees the silent miscalibration before sol_score
    starts producing garbage."""
    from src.config import load_config

    # Ada YAML (48 GiB) but the "GPU" reports 80 GiB.
    ada_yaml = """\
name: "RTX6000Ada"
SRAM_capacity: 100663296
DRAM_capacity: 51539607552
freq_GHz: 2.505
"""
    yaml_path = _write_yaml(ada_yaml)
    cfg_text = f"[hardware]\narch_config_path = {yaml_path}\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        cfg_path = Path(f.name)

    h100_props = SimpleNamespace(
        name="NVIDIA H100 PCIe",
        clock_rate=1_980_000,
        L2_cache_size=52_428_800,
        total_memory=85_899_345_920,
    )
    fake = _fake_torch(cuda_available=True, props=h100_props)
    with (
        patch.dict(sys.modules, {"torch": fake}),
        caplog.at_level("WARNING", logger="src.config"),
    ):
        config = load_config(cfg_path)

    assert config.hardware.name == "RTX6000Ada"  # YAML still wins
    assert any("DRAM capacity mismatch" in rec.message for rec in caplog.records)
