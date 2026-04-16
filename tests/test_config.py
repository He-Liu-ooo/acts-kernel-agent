"""Tests for config.py — HardwareSpec from SOLAR arch YAML."""

import tempfile
from pathlib import Path

from src.config import HardwareSpec, load_hardware_spec


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
