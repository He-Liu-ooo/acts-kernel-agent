# Config — `src/config.py`

Global configuration and hardware detection.

## Configuration File

Run parameters are set through `.cfg` files (libconfig format, parsed via `libconf` in `load_config()`). Unspecified groups + keys fall back to `ACTSConfig` dataclass defaults. Hardware specs are loaded from a SOLAR arch YAML when `hardware.arch_config_path` is set, otherwise detected at runtime. See `configs/example.cfg` for the canonical reference.

```libconfig
runtime:
{
    problem_path = "placeholder";
    reset_clocks = false;
};

hardware:
{
    gpu_index = 0;
    arch_config_path = "configs/arch/H100_PCIe.yaml";
};

search:
{
    beam_width = 3;
    beam_diversity = true;
    reviewer_metric_queries = false;
    failure_sibling_cap = 8;
    max_depth = 20;
    epsilon_start = 0.3;
    epsilon_end = 0.05;
};

eval:        { warmup_runs = 20; timed_runs = 100; };
move_on:     { sol_plateau_window = 3; sol_plateau_delta = 0.01; sol_target = 0.95; };
debug:       { max_debug_retries = 3; max_baseline_retries = 3; };
memory:      { optimization_memory_top_k = 5; };
benchmark:   { benchmark_workload_count = 3; };
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
| `shared_mem_per_block_bytes` | int | Per-block shared memory budget (opt-in dynamic SMEM ceiling). Default `0` = unknown. Populated by `detect_hardware()` from `props.shared_memory_per_block_optin` (defensive `getattr` fallback to `.shared_memory_per_block` on older torch). Read from YAML via `raw.get("shared_mem_per_block_bytes", 0)`. Consumed by the Phase B SMEM check in `compile_kernel_tool` and the `## Run context` prompt block rendered by `render_run_context()`. |
| `shared_mem_per_multiprocessor_bytes` | int | Per-SM shared memory budget (architectural ceiling). Default `0` = unknown. Populated by `detect_hardware()` from `props.shared_memory_per_multiprocessor`. Read from YAML via `raw.get("shared_mem_per_multiprocessor_bytes", 0)`. Same Phase B / run-context consumers as the per-block field. |
| `MAC_per_cycle_fp32_sm` | float | FP32 MACs/cycle (CUDA cores) |
| `MAC_per_cycle_tf32_tc` | float | TF32 MACs/cycle (Tensor Cores) |
| `MAC_per_cycle_fp16_tc` | float | FP16 MACs/cycle (Tensor Cores) |
| `MAC_per_cycle_bf16_tc` | float | BF16 MACs/cycle (Tensor Cores) |
| `MAC_per_cycle_fp8_tc` | float | FP8 MACs/cycle (Tensor Cores) |
| `MAC_per_cycle_int8_tc` | float | INT8 MACs/cycle (Tensor Cores) |
| `MAC_per_cycle_nvfp4_tc` | float | NVFP4 MACs/cycle (Blackwell only) |
| `compute_capability` | float | Compute capability (e.g. 8.9 for Ada, 9.0 for Hopper); `0.0` = unknown — callers default-deny on `min_compute_capability`-gated actions. Populated from YAML or `props.major + props.minor / 10` by `detect_hardware()`. |

### Derived properties

| Property | Formula | Unit |
|----------|---------|------|
| `peak_memory_bandwidth_gb_s` | `DRAM_byte_per_cycle * freq_GHz` | GB/s |
| `peak_sram_bandwidth_gb_s` | `SRAM_byte_per_cycle * freq_GHz` | GB/s |
| `peak_flops_fp32` | `MAC_per_cycle_fp32_sm * freq_GHz * 2 / 1e3` | TFLOPS |
| `peak_flops_tf32` | `MAC_per_cycle_tf32_tc * freq_GHz * 2 / 1e3` | TFLOPS |
| `peak_flops_bf16` | `MAC_per_cycle_bf16_tc * freq_GHz * 2 / 1e3` | TFLOPS |
| `peak_flops_fp16` | `MAC_per_cycle_fp16_tc * freq_GHz * 2 / 1e3` | TFLOPS |
| `peak_flops_fp8` | `MAC_per_cycle_fp8_tc * freq_GHz * 2 / 1e3` | TFLOPS |
| `peak_flops_nvfp4` | `MAC_per_cycle_nvfp4_tc * freq_GHz * 2 / 1e3` | TFLOPS (Blackwell-only; zero on pre-Blackwell arches) |

Dimensional analysis for the FLOPS rows: `MAC/cycle * GHz * 2` → `1e9 MACs/sec * 2 ops/MAC` → ops/sec; divide by `1e12` for TFLOPS, i.e. `* 1e9 * 2 / 1e12 = * 2 / 1e3`. The previous `/1e6` divisor returned PFLOPS while claiming TFLOPS — a 1000× overstatement.

**INT8 deliberately excluded** from the `peak_flops_*` family. INT8 Tensor Core throughput is reported in TOPS (integer ops/sec), not TFLOPS — mixing it into a TFLOPS-typed property family would be a unit-error footgun for downstream consumers (e.g. the multi-dtype peak rendering in `render_run_context()`). Operators that need INT8 peak should compute it directly from `MAC_per_cycle_int8_tc * freq_GHz * 2 / 1e3` and label the result TOPS.

**Multi-dtype peak rendering.** `render_run_context()` in `src/agents/llm_backend.py` consumes the full `{fp32, tf32, bf16, fp16, fp8, nvfp4}` dtype-peaks dict and renders **all non-zero peaks** on one line sorted by value descending, with tied dtypes grouped (alphabetical, `/`-joined). Format: `Peak FLOPS (TFLOPS): fp8=728.4 · bf16/fp16=364.2 · fp32=91.1`. Replaces the prior single-dominant-dtype pick (spec §3 decision 8) — on Ada/Hopper the dominant pick was fp8, which understated achievable headroom for non-fp8 workloads (a bf16 kernel's Reviewer pct_peak drifted against the wrong ceiling). Showing every peak lets the LLM (and the Reviewer's pct_peak prose) pick the right one for its workload dtype. When **all** peaks evaluate to 0.0 (uninitialized `MAC_per_cycle_*` fields on a named `HardwareSpec`), the entire `Peak FLOPS` line is omitted rather than printed as a misleading `0.0 TFLOPS`. The bandwidth lines render independently and are unaffected.

## ACTSConfig

Mutable dataclass. All parameters for a single optimization run.

**Search parameters** — control tree search in `search/orchestrator.py`:
- `beam_width` (3): max active frontier nodes after beam pruning.
- `beam_diversity` (True): enable the diversity-aware rescue pass (B2) in `beam_prune`. Disable for ablation or pure-exploitation runs.
- `reviewer_metric_queries` (False): If True, Reviewer can fetch additional NCU metrics via the `query_metric` tool (multi-turn agent loop, `max_turns=6`). Default off; the default submit-only path is the verified default. When True, `ReviewerAgent` also appends the `system_metric_queries.md` addendum to its system prompt; when False, the base `system.md` is used alone (mirrors tool-registration to prevent prompt-leak regressions).
- `planner_max_turns` (None: `int | None`): override for the Planner's SDK turn-budget. `None` preserves the hardcoded default of 4 (one in-band validation retry + confirmation). Non-None overrides; the orchestrator threads the value into `plan(max_turns=...)`. Used by the LLM-call Pareto experiment — see `doc/specs/2026-05-19-llm-call-pareto-experiment-design.md`.
- `reviewer_max_turns` (None: `int | None`): override for the Reviewer's SDK turn-budget. `None` preserves the 4-or-6 toggle conditioned on `reviewer_metric_queries`. Non-None overrides both branches uniformly; the orchestrator threads the value into `review(max_turns=...)`.
- `failure_sibling_cap` (8): max number of failure-sibling entries rendered into Planner prompts via `SearchTree.render_siblings(consumer='planner', failure_cap=...)`. Above the cap, failure siblings are deduped on `(action, params, failure_reason)` and the most-recent groups (by `(iter_no, child_id)`) survive. Reviewer consumer omits failure siblings regardless of cap. `0` = uncapped.
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
- `max_debug_retries` (3): Coder's self-correction attempts per iteration. `CoderAgent` reads this field at construction and derives the SDK tool-loop bound as `max_turns = 2 * max_debug_retries + 2` (default 8 = 3 compile+correctness cycles × 2 turns + 1 `submit_kernel` tool call + 1 final plain-text confirmation). See `doc/agents.md` → "Turn budget" for the full derivation.
- `max_baseline_retries` (3): Triton baseline generation attempts before skipping problem.
- `optimization_memory_top_k` (5): past experiences injected into Planner's context.
- `benchmark_workload_count` (3): representative workloads for iterative benchmarking.
- `arch_config_path` (""): path to SOLAR arch YAML. If empty, `detect_hardware()` is used.
- `hardware`: populated from arch YAML or `detect_hardware()` at startup.
- `safetensors_blob_roots` (None: `list[Path] | None`): override the default `[problem_dir]` blob_roots used by `build_input_generator` to resolve `SafetensorsInput` workloads. When `None`, the dispatcher falls back to `[problem_path]`, preserving the in-tree fixture layout. Useful when blobs live outside the problem directory (e.g. a shared model-weight staging area).
- `benchmark_adapter` (None: `str | None`): explicit override for the `_load_problem` dispatcher's adapter choice. Values: `"sol_execbench"` (also auto-selected when `definition.json` is present), `"kernelbench"` (raises `NotImplementedError` until that adapter ships). When `None`, the dispatcher inspects the problem directory and either picks an adapter or raises `UnknownBenchmarkFormat`.
- `anti_cheat_critical_names` (`list[str]`, default `["elapsed_time", "synchronize", "wait", "record", "query"]`): names of methods on `torch.cuda.Event` whose `id()` is snapshotted on entry to `per_iter_anti_cheat` and re-checked on exit. A monkey-patch substitution between snapshot and check raises `RewardHackDetected` and the orchestrator marks the branch DEAD_END. Default covers the timing primitives a candidate would need to patch to fake faster-than-SOL latencies; operators can extend without code change.

### Operator-supplied Triton baseline

Skip `CoderAgent.translate()` and seat a pre-written Triton kernel as the search-tree root. Read from the `.cfg` `[runtime]` section. See `doc/eval.md` for the loader gate sequence (compile + per-workload `verify_correctness`) and `doc/pipeline.md` for the dispatch seam that branches on `use_operator_baseline`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `use_operator_baseline` | `bool` | `False` | **Dispatch flag.** When True, `_dispatch_baseline` skips `CoderAgent.translate()` and loads the file at `triton_baseline_path` as the search-tree root. When False (default), the LLM-translation path runs and all other `triton_baseline_*` fields are ignored. |
| `triton_baseline_path` | `str \| None` | `None` | Path to a pre-written Triton `.py` file (file location, no longer the toggle). CWD-relative when not absolute. `load_config` raises `FileNotFoundError` at cfg-load time only when `use_operator_baseline=true` and the file is missing on disk. |
| `triton_baseline_dps` | `bool` | `False` | Host-wrapper destination-passing-style contract (matches `Kernel.dps`). Cfg-supplied because dps is unknowable from source. |
| `triton_baseline_kernel_name` | `str \| None` | `None` | Override for multi-`@triton.jit def` files. Auto-detected from source when exactly one JIT def is present. |
| `triton_baseline_enforce_autotune` | `bool` | `False` | Opt-in to `KernelCodeOutput._autotune_decorator_well_formed` (≥4 `triton.Config` entries, non-empty `key=[...]`). Default skip — operator kernels may legitimately use `@triton.heuristics` or hand-tuned single configs. |

```libconfig
runtime:
{
    use_operator_baseline = true;
    triton_baseline_path = "kernels/my_op.py";
    triton_baseline_dps = false;
    triton_baseline_enforce_autotune = false;
};
```

**Asymmetric consistency rules.** `__post_init__` enforces the dispatch flag with deliberately uneven strictness — raise where the misconfig would silently waste a run, warn where it's harmless:

- **`use_operator_baseline=true` + `triton_baseline_path` empty → `ValueError` at cfg load.** The operator declared intent to use a pre-written kernel but didn't supply the file; falling through to the LLM path would silently contradict the declared intent. Raise.
- **`use_operator_baseline=false` + any of `{triton_baseline_path, triton_baseline_dps, triton_baseline_kernel_name, triton_baseline_enforce_autotune}` set → `logger.warning` "dead config".** The flag is the only thing that gates dispatch; stray fields are inert. Warn so operators can toggle modes by flipping `use_operator_baseline` alone (without scrubbing the rest of the cfg) and so mid-iteration mode-flipping during a debug session is tolerated. Mirrors the `validate_hardware_spec` warn-not-raise pattern.

The `load_config` `FileNotFoundError` guard is correspondingly scoped: it fires only when `use_operator_baseline=true` and `triton_baseline_path` is set but missing on disk. A stray path under `use_operator_baseline=false` is dead config, not a misconfig — `__post_init__` already warns; `load_config` does not re-validate.

**Autotune-enforce trade-off.** Default skip preserves compatibility with operator kernels that intentionally bypass `@triton.autotune` (e.g. `@triton.heuristics`, single hand-tuned config). Set `triton_baseline_enforce_autotune = true` to apply the same well-formedness gate the LLM-translation path uses when the operator wants matching strictness.

`load_config` also extends the `_section_map` `[runtime]` keys to absorb the five fields, with a third coercion branch: when `default_val is None` (Optional fields), libconf's native value is stored directly — the previous `type(default_val)(value)` path tried `NoneType(value)` → `TypeError`.

### Bench-subprocess isolation knobs

Per-iter K-way bench, autotune burn-in, and the NCU profile gauntlet run in a short-lived `python -m src.eval.bench_worker` subprocess by default, isolating the orchestrator from CUDA-context poisoning, driver-level crashes, and ptxas/SOL hangs. The three knobs below (all read from the `.cfg` `[runtime]` section, mirroring the `use_operator_baseline` pattern) tune the isolation envelope. See `doc/eval.md` for the IPC contract + crash taxonomy and `doc/specs/2026-05-24-bench-subprocess-isolation-design.md` for the design rationale.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `bench_use_subprocess` | `bool` | `True` | **Dispatch flag.** When True, per-iter bench + autotune burn-in + NCU profile gauntlet run in a `python -m src.eval.bench_worker` subprocess. When False, runs in-process (single-step debugging without IPC overhead); the orchestrator loses crash isolation. |
| `worker_crash_threshold` | `int` | `3` | Consecutive worker-process non-zero exits (or signal-kills) before raising `WorkerProcessUnstable` and aborting the whole run. Mirrors `CUDAContextPoisoned`'s 3-strike escalation. Must be `>= 1` (enforced in `__post_init__`). |
| `worker_timeout_s` | `float` | `180000.0` | Total-lifetime watchdog passed to `proc.wait(timeout=...)`. On expiry the helper calls `terminate()` then `kill()` and raises `WorkerCrashed`. Default 180000s (~50 h) **effectively disables the watchdog** while the subprocess refactor beds in — there is no evidence yet of a frozen-child failure mode that needs a tight watchdog, and the original 30s default killed healthy workers mid-NCU (Codex 2026-05-26). Operators with hard wallclock budgets should set this explicitly to the actual envelope (worst-case K-way bench (warmup + timed × workloads × K) + NCU profile (~60s) + import overhead is in the low thousands of seconds). **Renamed from `worker_startup_timeout_s`** (Codex 2026-05-26: original name implied startup-only scope but the field has always been the total-lifetime watchdog). Must be `> 0` (enforced in `__post_init__`). A spawn-vs-lifetime split is deferred — see PROCESS / spec §13. |

```libconfig
runtime:
{
    bench_use_subprocess = true;
    worker_crash_threshold = 3;
    worker_timeout_s = 180000.0;
};
```

`load_config` extends the `_section_map` `[runtime]` keys to include all three fields; the cfg loader does not accept the old name `worker_startup_timeout_s`.

## Functions

- `load_config(path) -> ACTSConfig`: Parse `.cfg` file, fall back to defaults. Loads arch YAML if `[hardware] arch_config_path` is set; after load, calls `validate_hardware_spec()` against `detect_hardware()` and logs a `WARNING` per mismatch.
- `load_hardware_spec(path) -> HardwareSpec`: Parse a SOLAR arch config YAML into a `HardwareSpec`.
- `detect_hardware() -> HardwareSpec`: Query CUDA runtime via `torch.cuda.get_device_properties(0)` (device 0 of the visible set; ACTS pins this via `--gpu-index N` at module top, so device 0 is the operator-selected GPU), then auto-load the registered arch YAML for the detected device (if any) and merge. Steps:
  1. Probe runtime-knowable fields:
     - `name` — GPU model string (e.g. "NVIDIA RTX 6000 Ada Generation")
     - `freq_GHz` — boost clock from `clock_rate / 1_000_000` (kHz → GHz)
     - `SRAM_capacity` — `L2_cache_size`
     - `DRAM_capacity` — `total_memory`
     - `shared_mem_per_block_bytes` — `props.shared_memory_per_block_optin` (defensive `getattr` fallback to `.shared_memory_per_block` on older torch that predates the opt-in dynamic-SMEM attribute).
     - `shared_mem_per_multiprocessor_bytes` — `props.shared_memory_per_multiprocessor`.
     - `compute_capability` — derived as `props.major + props.minor / 10` (defensive `getattr` for test stubs missing the attrs).
  2. Look up the device name in `_ACTS_ARCH_YAMLS` via `_lookup_arch_yaml(detected.name)`.
  3. If a YAML is registered AND on disk, load it via `load_hardware_spec` and merge with the runtime spec via `dataclasses.replace`. **Runtime ground-truth wins** for `name` / `freq_GHz` / `SRAM_capacity` / `DRAM_capacity`; the YAML supplies `MAC_per_cycle_*` + `*_byte_per_cycle` + tier ratios.
  4. Run `validate_hardware_spec(yaml_spec, runtime_spec)` and log `WARNING` on any mismatch (DRAM/SRAM/freq, 10% tolerance) — never raises.
  5. If the device name isn't registered, OR the YAML load raises, OR torch isn't available / no CUDA, return the runtime-only spec with throughput tables and bandwidth coefficients zeroed (the previous behavior). The placeholder substitution path in `pipeline/optimize.py` covers this fallback for SOL-ExecBench live runs.

  Returns a fully-zeroed `HardwareSpec` on torch import error (catches `Exception`, covering broken-driver `OSError`/`RuntimeError`), `torch.cuda.is_available()` False, no CUDA devices, or device-property probe raises.

### `validate_hardware_spec(spec, detected) -> list[str]`

Compares config-source `spec` against runtime-`detected` spec and returns a list of mismatch messages (empty = no mismatch). Catches the silent-miscalibration class of bugs where the YAML or placeholder substitution doesn't match the actual GPU.

Five checks, each with **10% tolerance** and **per-field skip-if-zero**:
- `DRAM_capacity` — GPU-family fingerprint (Ada 48 GiB ≠ H100 80 GiB).
- `SRAM_capacity` (L2) — discriminates within a family that shares DRAM (Ada 96 MiB vs H100 50 MiB).
- `freq_GHz` — both sources report boost clock, so >10% delta likely means wrong YAML.
- `shared_mem_per_block_bytes` — per-block opt-in SMEM ceiling; >10% delta flags a wrong-YAML / wrong-arch pairing (e.g. Ada 99 KiB vs Hopper 227 KiB).
- `shared_mem_per_multiprocessor_bytes` — per-SM SMEM ceiling; same wrong-arch signal as the per-block field. Both new checks are warn-don't-raise, parallel to the existing three. See `doc/specs/2026-05-24-coding-hw-spec-design.md` §4 for the design rationale.

Call sites:
1. `load_config()` — after loading a YAML via `arch_config_path`, validates against `detect_hardware()`.
2. `pipeline/optimize.py::optimize()` — before substituting `_PLACEHOLDER_HARDWARE_SPEC` for zero-peak specs.

Both call sites **warn (don't raise)** — sometimes you legitimately model GPU X while running on GPU Y for ablation.

### Arch YAML registry

`_ACTS_ARCH_YAMLS: dict[str, Path]` in `src/config.py` maps detected GPU device names to canonical arch YAML paths. Currently registered:

- `RTX6000Ada` → `configs/arch/RTX6000Ada.yaml`
- `NVIDIA RTX 6000 Ada Generation` → `configs/arch/RTX6000Ada.yaml`
- `placeholder-RTX6000Ada` → `configs/arch/RTX6000Ada.yaml`

Lookup goes through `_lookup_arch_yaml(detected_name) -> Path | None`, which returns the registered path if the YAML exists on disk, else `None`. The registry is consulted by both `detect_hardware()` (auto-load on runtime detection) and `solar_adapter._resolve_arch_config` (explicit SOLAR adapter path) — single source of truth; `solar_adapter` re-exports the symbol for back-compat.

To add a new GPU: drop a YAML in `configs/arch/<Name>.yaml` matching SOLAR's arch schema (per-precision `MAC_per_cycle` table + `DRAM_byte_per_cycle` + `SRAM_byte_per_cycle` + `freq_GHz` + capacities), then add the device-name substring to `_ACTS_ARCH_YAMLS`.
