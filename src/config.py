"""Global configuration and hardware detection."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HardwareSpec:
    """GPU hardware specification — matches SOLAR arch YAML schema.

    Load from a SOLAR arch config YAML via ``load_hardware_spec(path)``,
    or construct directly for testing.  Derived properties compute peak
    TFLOPS and bandwidth from the raw per-cycle fields + frequency.
    """

    name: str = ""
    freq_GHz: float = 0.0
    # Memory hierarchy
    SRAM_capacity: int = 0          # L2 cache bytes
    SRAM_byte_per_cycle: float = 0.0
    DRAM_capacity: int = 0          # Total GPU memory bytes
    DRAM_byte_per_cycle: float = 0.0
    # MAC per cycle by precision (SM = CUDA cores, TC = Tensor Cores)
    MAC_per_cycle_fp32_sm: float = 0.0
    MAC_per_cycle_tf32_tc: float = 0.0
    MAC_per_cycle_fp16_tc: float = 0.0
    MAC_per_cycle_bf16_tc: float = 0.0
    MAC_per_cycle_fp8_tc: float = 0.0
    MAC_per_cycle_int8_tc: float = 0.0
    MAC_per_cycle_nvfp4_tc: float = 0.0  # Blackwell only

    # ── derived properties ────────────────────────────────────────────────

    @property
    def peak_memory_bandwidth_gb_s(self) -> float:
        """Peak DRAM bandwidth in GB/s."""
        return self.DRAM_byte_per_cycle * self.freq_GHz

    @property
    def peak_sram_bandwidth_gb_s(self) -> float:
        """Peak SRAM (L2) bandwidth in GB/s."""
        return self.SRAM_byte_per_cycle * self.freq_GHz

    @property
    def peak_flops_fp32(self) -> float:
        """Peak FP32 throughput in TFLOPS (CUDA cores)."""
        return self.MAC_per_cycle_fp32_sm * self.freq_GHz * 2 / 1e6

    @property
    def peak_flops_bf16(self) -> float:
        """Peak BF16 throughput in TFLOPS (Tensor Cores)."""
        return self.MAC_per_cycle_bf16_tc * self.freq_GHz * 2 / 1e6

    @property
    def peak_flops_fp16(self) -> float:
        """Peak FP16 throughput in TFLOPS (Tensor Cores)."""
        return self.MAC_per_cycle_fp16_tc * self.freq_GHz * 2 / 1e6


def load_hardware_spec(path: Path) -> HardwareSpec:
    """Load a HardwareSpec from a SOLAR arch config YAML.

    Example: ``load_hardware_spec(Path("configs/arch/H100_PCIe.yaml"))``
    """
    import yaml

    raw = yaml.safe_load(path.read_text())
    return HardwareSpec(
        name=raw.get("name", ""),
        freq_GHz=raw.get("freq_GHz", 0.0),
        SRAM_capacity=raw.get("SRAM_capacity", 0),
        SRAM_byte_per_cycle=raw.get("SRAM_byte_per_cycle", 0.0),
        DRAM_capacity=raw.get("DRAM_capacity", 0),
        DRAM_byte_per_cycle=raw.get("DRAM_byte_per_cycle", 0.0),
        MAC_per_cycle_fp32_sm=raw.get("MAC_per_cycle_fp32_sm", 0.0),
        MAC_per_cycle_tf32_tc=raw.get("MAC_per_cycle_tf32_tc", 0.0),
        MAC_per_cycle_fp16_tc=raw.get("MAC_per_cycle_fp16_tc", 0.0),
        MAC_per_cycle_bf16_tc=raw.get("MAC_per_cycle_bf16_tc", 0.0),
        MAC_per_cycle_fp8_tc=raw.get("MAC_per_cycle_fp8_tc", 0.0),
        MAC_per_cycle_int8_tc=raw.get("MAC_per_cycle_int8_tc", 0.0),
        MAC_per_cycle_nvfp4_tc=raw.get("MAC_per_cycle_nvfp4_tc", 0.0),
    )


@dataclass
class ACTSConfig:
    """Top-level configuration for an ACTS optimization run."""

    # Search parameters
    beam_width: int = 3
    beam_diversity: bool = True
    max_depth: int = 20
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05

    # Evaluation parameters
    warmup_runs: int = 20
    timed_runs: int = 100

    # Move-on criteria
    sol_plateau_window: int = 3
    sol_plateau_delta: float = 0.01
    sol_target: float = 0.95

    # Debug retry budget
    max_debug_retries: int = 3
    max_baseline_retries: int = 3

    # Memory retrieval
    optimization_memory_top_k: int = 5

    # Benchmark
    benchmark_workload_count: int = 3

    # Hardware — loaded from SOLAR arch YAML, or detected at runtime
    hardware: HardwareSpec = field(default_factory=HardwareSpec)
    arch_config_path: str = ""  # Path to SOLAR arch YAML (e.g. "configs/arch/H100_PCIe.yaml")


def load_config(path: Path) -> ACTSConfig:
    """Load ACTSConfig from a .cfg file via configparser.

    Values not specified in the file fall back to ACTSConfig defaults.
    Hardware specs are loaded from a SOLAR arch YAML if ``[hardware]
    arch_config_path`` is set, otherwise detected at runtime.
    """
    cfg = configparser.ConfigParser()
    cfg.read(path)
    kwargs: dict = {}
    _section_map = {
        "search": ["beam_width", "beam_diversity", "max_depth", "epsilon_start", "epsilon_end"],
        "eval": ["warmup_runs", "timed_runs"],
        "move_on": ["sol_plateau_window", "sol_plateau_delta", "sol_target"],
        "debug": ["max_debug_retries", "max_baseline_retries"],
        "memory": ["optimization_memory_top_k"],
        "benchmark": ["benchmark_workload_count"],
    }
    defaults = ACTSConfig()
    for section, keys in _section_map.items():
        if not cfg.has_section(section):
            continue
        for key in keys:
            if cfg.has_option(section, key):
                default_val = getattr(defaults, key)
                if isinstance(default_val, bool):
                    kwargs[key] = cfg.getboolean(section, key)
                else:
                    kwargs[key] = type(default_val)(cfg.get(section, key))
    # Hardware: load from SOLAR arch YAML if specified, else detect at runtime
    arch_path_str = cfg.get("hardware", "arch_config_path", fallback="")
    if arch_path_str:
        kwargs["arch_config_path"] = arch_path_str
        kwargs["hardware"] = load_hardware_spec(Path(arch_path_str))
    else:
        kwargs["hardware"] = detect_hardware()
    return ACTSConfig(**kwargs)


def detect_hardware() -> HardwareSpec:
    """Detect GPU hardware and return a HardwareSpec.

    Fallback when no SOLAR arch YAML is provided.  Returns a default
    (zeroed) spec if no GPU is available.
    """
    # Placeholder: return zeroed spec until CUDA detection is implemented.
    # Real implementation would query torch.cuda / pynvml and map to
    # SOLAR-schema fields.
    return HardwareSpec()
