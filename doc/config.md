# Config — `src/config.py`

Global configuration and hardware detection.

## Configuration File

Run parameters are set through `.cfg` files (INI format, parsed via `configparser`). Unspecified values fall back to built-in defaults. Hardware specs are loaded from a SOLAR arch YAML if specified, otherwise detected at runtime.

```ini
[search]
beam_width = 3
beam_diversity = true
max_depth = 20
epsilon_start = 0.3
epsilon_end = 0.05

[eval]
warmup_runs = 20
timed_runs = 100

[move_on]
sol_plateau_window = 3
sol_plateau_delta = 0.01
sol_target = 0.95

[debug]
max_debug_retries = 3
max_baseline_retries = 3

[memory]
optimization_memory_top_k = 5

[benchmark]
benchmark_workload_count = 3

[hardware]
arch_config_path = configs/arch/H100_PCIe.yaml
```

## HardwareSpec

Frozen dataclass using the SOLAR arch YAML schema. Load from YAML via `load_hardware_spec(path)`, or construct directly for testing. Peak TFLOPS and bandwidth are derived properties.

### Raw fields (from SOLAR YAML)

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | GPU model name (e.g. "H100_PCIe", "B200") |
| `freq_GHz` | float | Clock frequency in GHz |
| `SRAM_capacity` | int | L2 cache size in bytes |
| `SRAM_byte_per_cycle` | float | L2 bandwidth per cycle |
| `DRAM_capacity` | int | Total GPU memory in bytes |
| `DRAM_byte_per_cycle` | float | DRAM bandwidth per cycle |
| `MAC_per_cycle_fp32_sm` | float | FP32 MACs/cycle (CUDA cores) |
| `MAC_per_cycle_tf32_tc` | float | TF32 MACs/cycle (Tensor Cores) |
| `MAC_per_cycle_fp16_tc` | float | FP16 MACs/cycle (Tensor Cores) |
| `MAC_per_cycle_bf16_tc` | float | BF16 MACs/cycle (Tensor Cores) |
| `MAC_per_cycle_fp8_tc` | float | FP8 MACs/cycle (Tensor Cores) |
| `MAC_per_cycle_int8_tc` | float | INT8 MACs/cycle (Tensor Cores) |
| `MAC_per_cycle_nvfp4_tc` | float | NVFP4 MACs/cycle (Blackwell only) |

### Derived properties

| Property | Formula | Unit |
|----------|---------|------|
| `peak_memory_bandwidth_gb_s` | `DRAM_byte_per_cycle * freq_GHz` | GB/s |
| `peak_sram_bandwidth_gb_s` | `SRAM_byte_per_cycle * freq_GHz` | GB/s |
| `peak_flops_fp32` | `MAC_per_cycle_fp32_sm * freq_GHz * 2 / 1e6` | TFLOPS |
| `peak_flops_bf16` | `MAC_per_cycle_bf16_tc * freq_GHz * 2 / 1e6` | TFLOPS |
| `peak_flops_fp16` | `MAC_per_cycle_fp16_tc * freq_GHz * 2 / 1e6` | TFLOPS |

## ACTSConfig

Mutable dataclass. All parameters for a single optimization run.

**Search parameters** — control tree search in `search/orchestrator.py`:
- `beam_width` (3): max active frontier nodes after beam pruning.
- `beam_diversity` (True): enable the diversity-aware rescue pass (B2) in `beam_prune`. Disable for ablation or pure-exploitation runs.
- `max_depth` (20): max tree depth (longest root-to-leaf path).
- `epsilon_start` (0.3): initial exploration rate for epsilon-greedy selection.
- `epsilon_end` (0.05): final exploration rate after decay.

**Evaluation parameters** — control `eval/benchmark.py`:
- `warmup_runs` (20): CUDA warmup iterations before timing.
- `timed_runs` (100): measured iterations; median latency taken.

**Move-on criteria** — when to stop optimizing:
- `sol_plateau_window` (3): consecutive iterations to check for plateau.
- `sol_plateau_delta` (0.01): minimum SOL improvement to not count as plateau.
- `sol_target` (0.95): SOL score threshold for "close enough to hardware limit."

**Other:**
- `max_debug_retries` (3): Coder's self-correction attempts per iteration before marking branch dead. *Current gap*: `CoderAgent` hardcodes the derived SDK turn budget (`_MAX_TURNS = 7`) rather than reading this field; see `PROCESS.md` → Deferred Improvements.
- `max_baseline_retries` (3): Triton baseline generation attempts before skipping problem.
- `optimization_memory_top_k` (5): past experiences injected into Planner's context.
- `benchmark_workload_count` (3): representative workloads for iterative benchmarking.
- `arch_config_path` (""): path to SOLAR arch YAML. If empty, `detect_hardware()` is used.
- `hardware`: populated from arch YAML or `detect_hardware()` at startup.

## Functions

- `load_config(path) -> ACTSConfig`: Parse `.cfg` file, fall back to defaults. Loads arch YAML if `[hardware] arch_config_path` is set.
- `load_hardware_spec(path) -> HardwareSpec`: Parse a SOLAR arch config YAML into a `HardwareSpec`.
- `detect_hardware() -> HardwareSpec`: Query CUDA runtime. Returns zeroed spec if no GPU. Fallback when no arch YAML is provided.
