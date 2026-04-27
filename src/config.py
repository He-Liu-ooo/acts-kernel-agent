"""Global configuration and hardware detection."""

from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


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
    # When True, the Reviewer registers a `query_metric` tool alongside
    # `submit_review` and runs with `max_turns=6` instead of `4`. Default
    # off — the existing single-call path is the verified default.
    reviewer_metric_queries: bool = False
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
        "search": [
            "beam_width", "beam_diversity", "reviewer_metric_queries",
            "max_depth", "epsilon_start", "epsilon_end",
        ],
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
        yaml_spec = load_hardware_spec(Path(arch_path_str))
        for msg in validate_hardware_spec(yaml_spec, detect_hardware()):
            logger.warning("arch_config_path %s: %s", arch_path_str, msg)
        kwargs["hardware"] = yaml_spec
    else:
        kwargs["hardware"] = detect_hardware()
    return ACTSConfig(**kwargs)


def detect_hardware() -> HardwareSpec:
    """Detect GPU hardware via ``torch.cuda`` and return a HardwareSpec.

    **Best-effort, partial spec.** Populates only the runtime-knowable
    fields (``name``, ``freq_GHz``, ``SRAM_capacity``, ``DRAM_capacity``).
    Per-precision throughput tables (``MAC_per_cycle_*``) and bandwidth
    coefficients (``DRAM_byte_per_cycle``, ``SRAM_byte_per_cycle``) stay
    at zero — those depend on architecture details ``torch.cuda`` cannot
    infer (boost-clock vs base-clock, tensor-core configurations, memory
    subsystem peak vs effective). For real ``T_SOL`` / roofline math,
    callers must load a SOLAR arch YAML via ``load_hardware_spec()``.

    Returns a fully-zeroed ``HardwareSpec`` (matches the no-config
    placeholder) when:

    - ``torch`` cannot be imported (CPU-only environment),
    - ``torch.cuda.is_available()`` is False,
    - no CUDA devices are visible, or
    - the device-property probe raises (driver mismatch, etc.).

    The orchestrator's existing zero-peak handling (substituting a
    populated placeholder when peaks are zero) covers all of these.
    """
    try:
        import torch
    except Exception:
        # Broken torch installs raise OSError (missing libcuda.so) or
        # RuntimeError (ABI mismatch), not just ImportError. The docstring
        # promises a zeroed-spec fallback for "torch cannot be imported"
        # — honor that contract for the broken-driver case too.
        return HardwareSpec()

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return HardwareSpec()
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return HardwareSpec()

    return HardwareSpec(
        name=props.name,
        freq_GHz=props.clock_rate / 1_000_000,  # kHz → GHz
        SRAM_capacity=props.L2_cache_size,
        DRAM_capacity=props.total_memory,
    )


def validate_hardware_spec(spec: HardwareSpec, detected: HardwareSpec) -> list[str]:
    """Compare a config-source HardwareSpec against the runtime-detected
    spec and return a list of mismatch messages (empty = no mismatch).

    Catches the silent-miscalibration class of bugs where an
    ``arch_config_path`` YAML or the placeholder-substitution path doesn't
    match the GPU actually running the workload (e.g. ``RTX6000Ada.yaml``
    configured but H100 in the box → all T_SOL math is wrong but the run
    completes "successfully").

    Checks every field both sources populate: ``DRAM_capacity`` (discriminates
    GPU family), ``SRAM_capacity`` (L2 — discriminates within a family that
    shares DRAM, e.g. Ada 96 MB vs H100 50 MB), ``freq_GHz`` (boost clock —
    both sources report boost, so they should match within driver precision).
    Per-field skip-if-zero so the validator stays silent when either side
    is unpopulated (no GPU, partial spec) — its job is to flag
    *demonstrable* mismatches, not noise on missing data.
    """
    issues: list[str] = []
    if spec.DRAM_capacity > 0 and detected.DRAM_capacity > 0:
        ratio = spec.DRAM_capacity / detected.DRAM_capacity
        if ratio < 0.9 or ratio > 1.1:
            issues.append(
                f"DRAM capacity mismatch: spec={spec.DRAM_capacity / 1024**3:.1f} GiB "
                f"(name={spec.name!r}), detected={detected.DRAM_capacity / 1024**3:.1f} GiB "
                f"(name={detected.name!r}) — the configured hardware probably "
                f"doesn't match the actual GPU; T_SOL and sol_score will be wrong"
            )
    if spec.SRAM_capacity > 0 and detected.SRAM_capacity > 0:
        ratio = spec.SRAM_capacity / detected.SRAM_capacity
        if ratio < 0.9 or ratio > 1.1:
            issues.append(
                f"SRAM (L2) capacity mismatch: spec={spec.SRAM_capacity / 1024**2:.0f} MiB "
                f"(name={spec.name!r}), detected={detected.SRAM_capacity / 1024**2:.0f} MiB "
                f"(name={detected.name!r})"
            )
    if spec.freq_GHz > 0 and detected.freq_GHz > 0:
        ratio = spec.freq_GHz / detected.freq_GHz
        if ratio < 0.9 or ratio > 1.1:
            issues.append(
                f"freq_GHz mismatch: spec={spec.freq_GHz:.3f} GHz "
                f"(name={spec.name!r}), detected={detected.freq_GHz:.3f} GHz "
                f"(name={detected.name!r}) — both sources report boost clock, "
                f"so a >10% delta likely means wrong YAML"
            )
    return issues
