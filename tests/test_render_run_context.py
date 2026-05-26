"""Tests for the render_run_context() helper extension carrying HardwareSpec.

Lives alongside the existing Planner/Reviewer tests; the helper is shared by
all three agents per the hw-spec-injection spec (doc/specs/2026-05-24-coding-
hw-spec-design.md §5.1).
"""

from src.agents.llm_backend import render_run_context
from src.config import HardwareSpec
from src.eval.types import BottleneckType


def test_render_run_context_omits_hw_block_when_name_empty():
    """Empty HardwareSpec (no spec configured) falls back to bottleneck-only block."""
    out = render_run_context(BottleneckType.MEMORY_BOUND, hardware=HardwareSpec())
    assert "Bottleneck" in out
    assert "Hardware:" not in out
    assert "Shared mem per block" not in out


def test_render_run_context_hardware_kwarg_defaults_none():
    """Existing single-arg call sites continue working (hardware kwarg is optional)."""
    out = render_run_context(BottleneckType.MEMORY_BOUND)
    assert "Bottleneck" in out
    assert "Hardware:" not in out


def test_render_run_context_includes_hw_block_when_spec_present():
    """Spec with non-empty name renders the 6-line hw block under the bottleneck."""
    hw = HardwareSpec(
        name="NVIDIA RTX 6000 Ada Generation",
        compute_capability=8.9,
        freq_GHz=2.505,
        DRAM_byte_per_cycle=383,
        MAC_per_cycle_bf16_tc=72695,
        MAC_per_cycle_fp16_tc=72695,
        MAC_per_cycle_fp32_sm=18185,
        shared_mem_per_block_bytes=101376,
        shared_mem_per_multiprocessor_bytes=102400,
    )
    out = render_run_context(BottleneckType.MEMORY_BOUND, hardware=hw)
    assert "Hardware: NVIDIA RTX 6000 Ada Generation (sm_89)" in out
    assert "Shared mem per block: 101376 B" in out
    assert "Shared mem per SM: 102400 B" in out
    assert "Peak DRAM bandwidth" in out
    assert "Per-Config shared-mem rule" in out
    # Rule line uses naming-agnostic prose per spec §5.1.
    assert "input_tile_elements" in out


def test_render_run_context_renders_all_nonzero_peaks_sorted_desc():
    """All non-zero dtype peaks render on the same line, sorted by value
    descending. The LLM picks the peak that matches its workload dtype
    instead of seeing only the device-max dtype.

    Replaces the prior "dominant-dtype selection" behavior (spec §3
    decision 8). The dominant-pick understated achievable headroom on
    workloads whose dtype wasn't the device's highest-peak dtype — e.g.
    on Ada, a bf16 workload saw "Peak FLOPS (fp8): 728 TFLOPS" and the
    Reviewer's pct_peak interpretation drifted accordingly.
    """
    hw = HardwareSpec(
        name="X", compute_capability=9.0, freq_GHz=1.0,
        MAC_per_cycle_fp32_sm=1000,
        MAC_per_cycle_bf16_tc=500,
        MAC_per_cycle_fp16_tc=400,
        shared_mem_per_block_bytes=1, shared_mem_per_multiprocessor_bytes=1,
    )
    out = render_run_context(BottleneckType.COMPUTE_BOUND, hardware=hw)
    # New format: one "Peak FLOPS (TFLOPS):" line listing every non-zero
    # dtype, descending by peak value, with ` · ` separators.
    peak_line = [ln for ln in out.splitlines() if ln.startswith("- Peak FLOPS")]
    assert len(peak_line) == 1, peak_line
    line = peak_line[0]
    # All three non-zero dtypes appear with their values.
    assert "fp32=" in line
    assert "bf16=" in line
    assert "fp16=" in line
    # Descending order: fp32's peak (largest at 1000 MAC/cycle × 2 FLOPS/MAC
    # × 1.0 GHz = 2.0 TFLOPS) comes first, then bf16, then fp16.
    assert line.index("fp32=") < line.index("bf16=") < line.index("fp16=")
    # Unit declared once in the label, not repeated per entry.
    assert line.startswith("- Peak FLOPS (TFLOPS):")
    # No old-style dominant-only output.
    assert "Peak FLOPS (fp32):" not in out


def test_render_run_context_renders_ties_joined_alphabetically():
    """Tied dtypes (same peak value) group into one entry joined by '/'
    in alphabetical order.
    """
    hw = HardwareSpec(
        name="X", compute_capability=8.9, freq_GHz=2.505,
        MAC_per_cycle_fp32_sm=18185,
        MAC_per_cycle_bf16_tc=72695,
        MAC_per_cycle_fp16_tc=72695,  # tied with bf16
        shared_mem_per_block_bytes=1, shared_mem_per_multiprocessor_bytes=1,
    )
    out = render_run_context(BottleneckType.COMPUTE_BOUND, hardware=hw)
    peak_line = [ln for ln in out.splitlines() if ln.startswith("- Peak FLOPS")][0]
    # bf16 and fp16 share a peak value → joined as "bf16/fp16=...".
    assert "bf16/fp16=" in peak_line
    # fp32 is separate (lower peak).
    assert "fp32=" in peak_line
    # The tied entry comes before the lower-peak fp32.
    assert peak_line.index("bf16/fp16=") < peak_line.index("fp32=")


def test_render_run_context_renders_hw_when_bottleneck_none(monkeypatch):
    """Baseline generation path: bottleneck not yet classified, but hw spec is
    available — still render the hw budget block so the Coder sees the SMEM
    cap before drafting the baseline.

    Regression for Codex 2026-05-25 finding: translate path didn't render
    hw context at all, leaving baseline Coder blind to the cap on first attempt.
    """
    hw = HardwareSpec(
        name="NVIDIA RTX 6000 Ada Generation",
        compute_capability=8.9,
        freq_GHz=2.505,
        DRAM_byte_per_cycle=383,
        MAC_per_cycle_bf16_tc=72695,
        MAC_per_cycle_fp16_tc=72695,
        MAC_per_cycle_fp32_sm=18185,
        shared_mem_per_block_bytes=101376,
        shared_mem_per_multiprocessor_bytes=102400,
    )
    out = render_run_context(None, hardware=hw)
    # Hw block IS rendered.
    assert "Hardware: NVIDIA RTX 6000 Ada Generation" in out
    assert "Shared mem per block: 101376 B" in out
    # Bottleneck line says "not yet classified" instead of a value.
    assert "not yet classified" in out or "baseline" in out.lower()


def test_render_run_context_returns_empty_when_both_none():
    """Both bottleneck and hardware unset → empty string (no '## Run context'
    heading-only section)."""
    out = render_run_context(None, hardware=None)
    assert out == ""


def test_render_run_context_omits_peak_flops_line_when_all_zero():
    """When the operator hasn't populated any MAC_per_cycle_* fields, every
    derived peak is 0. Rendering "Peak FLOPS (bf16/fp16/fp32): 0.0 TFLOPS"
    is misleading — skip the line entirely (Fix #7)."""
    hw = HardwareSpec(
        name="PartiallyConfiguredGPU",
        compute_capability=8.9,
        freq_GHz=2.0,
        # All MAC_per_cycle_* fields left at 0 — derived peaks all 0.
        shared_mem_per_block_bytes=101376,
        shared_mem_per_multiprocessor_bytes=102400,
    )
    out = render_run_context(BottleneckType.MEMORY_BOUND, hardware=hw)
    # Hw block still rendered (name is non-empty).
    assert "Hardware: PartiallyConfiguredGPU" in out
    assert "Shared mem per block:" in out
    # But Peak FLOPS line is omitted — no meaningless "0.0 TFLOPS" output.
    assert "Peak FLOPS" not in out


def test_render_run_context_includes_sm_count_and_max_threads():
    """Full hw spec renders both the SM count and max-threads/block lines."""
    hw = HardwareSpec(
        name="RTX6000Ada",
        compute_capability=8.9,
        freq_GHz=2.505,
        DRAM_byte_per_cycle=383,
        MAC_per_cycle_bf16_tc=72695,
        MAC_per_cycle_fp16_tc=72695,
        shared_mem_per_block_bytes=101376,
        shared_mem_per_multiprocessor_bytes=102400,
        sm_count=142,
        max_threads_per_block=1024,
    )
    out = render_run_context(BottleneckType.MEMORY_BOUND, hardware=hw)
    assert "- SM count: 142" in out
    # Max-threads line annotates the implied num_warps ceiling (1024 // 32 = 32).
    assert "- Max threads per block: 1024" in out
    assert "num_warps" in out
    assert "32" in out


def test_render_run_context_includes_l2_cache_line():
    """SRAM_capacity (L2 cache) renders as its own line with MiB approximation."""
    hw = HardwareSpec(
        name="RTX6000Ada",
        compute_capability=8.9,
        freq_GHz=2.505,
        SRAM_capacity=100663296,  # 96 MiB
        DRAM_byte_per_cycle=383,
        MAC_per_cycle_bf16_tc=72695,
        shared_mem_per_block_bytes=101376,
        shared_mem_per_multiprocessor_bytes=102400,
        sm_count=142,
        max_threads_per_block=1024,
    )
    out = render_run_context(BottleneckType.MEMORY_BOUND, hardware=hw)
    assert "- L2 cache:" in out
    assert "100663296" in out
    assert "96" in out  # MiB approximation
    assert "MiB" in out


def test_render_run_context_includes_tensor_core_tile_for_ada_fp16():
    """Ada (sm_89) + fp16 dominant dtype → m16n16k16 tile string."""
    hw = HardwareSpec(
        name="RTX6000Ada",
        compute_capability=8.9,
        freq_GHz=2.505,
        MAC_per_cycle_fp16_tc=72695,
        MAC_per_cycle_bf16_tc=72695,
        shared_mem_per_block_bytes=101376,
        shared_mem_per_multiprocessor_bytes=102400,
    )
    out = render_run_context(BottleneckType.MEMORY_BOUND, hardware=hw)
    assert "- Tensor Core tile" in out
    assert "m16n16k16" in out


def test_render_run_context_omits_tensor_core_tile_when_unknown_arch():
    """Turing (sm_75) + fp8 (unsupported on Turing) → no Tensor Core tile line."""
    hw = HardwareSpec(
        name="T4Like",
        compute_capability=7.5,
        freq_GHz=1.5,
        MAC_per_cycle_fp8_tc=100,  # bogus on Turing; just to force fp8-as-dominant
        shared_mem_per_block_bytes=49152,
        shared_mem_per_multiprocessor_bytes=65536,
    )
    out = render_run_context(BottleneckType.MEMORY_BOUND, hardware=hw)
    assert "Tensor Core tile" not in out


def test_render_run_context_renders_tensor_core_tile_for_hopper_wgmma():
    """Hopper (sm_90) + fp8 dominant → WGMMA prose tile descriptor."""
    hw = HardwareSpec(
        name="H100Like",
        compute_capability=9.0,
        freq_GHz=2.0,
        MAC_per_cycle_fp8_tc=756000,
        shared_mem_per_block_bytes=232448,
        shared_mem_per_multiprocessor_bytes=233472,
    )
    out = render_run_context(BottleneckType.COMPUTE_BOUND, hardware=hw)
    assert "Tensor Core tile" in out
    assert "WGMMA" in out
    # WGMMA prose lists the K-shape rule for fp8.
    assert "32" in out


def test_render_run_context_omits_sm_count_when_zero():
    """sm_count=0 → no SM count line (mirrors zero-peak FLOPS omit policy)."""
    hw = HardwareSpec(
        name="PartialGPU",
        compute_capability=8.9,
        freq_GHz=2.0,
        shared_mem_per_block_bytes=101376,
        shared_mem_per_multiprocessor_bytes=102400,
        # sm_count=0 (default), max_threads_per_block=0 (default)
    )
    out = render_run_context(BottleneckType.MEMORY_BOUND, hardware=hw)
    assert "SM count" not in out
    assert "Max threads per block" not in out


def test_render_run_context_renders_workload_shapes_few():
    """workload_shapes kwarg with ≤3 shapes renders each tuple literally."""
    hw = HardwareSpec(
        name="X",
        compute_capability=8.9,
        freq_GHz=2.0,
        shared_mem_per_block_bytes=1,
        shared_mem_per_multiprocessor_bytes=1,
    )
    out = render_run_context(
        BottleneckType.MEMORY_BOUND,
        hardware=hw,
        workload_shapes=[(1024, 4096, 2048), (2048, 4096, 2048), (4096, 4096, 2048)],
    )
    assert "- Workload shapes:" in out
    assert "(1024, 4096, 2048)" in out
    assert "(2048, 4096, 2048)" in out
    assert "(4096, 4096, 2048)" in out


def test_render_run_context_renders_workload_shapes_many():
    """>3 shapes summarize into per-dim min-max ranges with N=<count>."""
    hw = HardwareSpec(
        name="X",
        compute_capability=8.9,
        freq_GHz=2.0,
        shared_mem_per_block_bytes=1,
        shared_mem_per_multiprocessor_bytes=1,
    )
    shapes = [(m, 4096, 2048) for m in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)]
    out = render_run_context(
        BottleneckType.MEMORY_BOUND,
        hardware=hw,
        workload_shapes=shapes,
    )
    assert "- Workload shapes:" in out
    assert "N=10" in out
    assert "128-65536" in out


def test_render_run_context_omits_workload_shapes_when_none():
    """workload_shapes=None (default) → no Workload shapes line."""
    hw = HardwareSpec(
        name="X",
        compute_capability=8.9,
        freq_GHz=2.0,
        shared_mem_per_block_bytes=1,
        shared_mem_per_multiprocessor_bytes=1,
    )
    out = render_run_context(BottleneckType.MEMORY_BOUND, hardware=hw)
    assert "Workload shapes" not in out


def test_render_run_context_shows_fp8_alongside_lower_precisions_on_hopper():
    """On Hopper/Blackwell, fp8 peak is typically 2× the fp16 peak. The
    LLM must see ALL non-zero peaks — bf16/fp16 alongside fp8 — so a
    bf16-workload kernel doesn't get measured against the fp8 ceiling.
    Replaces the prior fp8-only "dominant pick" (fix #11 from 2026-05-25)
    which made the rendered peak misleading for non-fp8 workloads.
    """
    hw = HardwareSpec(
        name="HopperLike",
        compute_capability=9.0,
        freq_GHz=2.0,
        MAC_per_cycle_fp32_sm=25500,
        MAC_per_cycle_bf16_tc=378000,
        MAC_per_cycle_fp16_tc=378000,
        MAC_per_cycle_fp8_tc=756000,  # 2× fp16
        shared_mem_per_block_bytes=1,
        shared_mem_per_multiprocessor_bytes=1,
    )
    out = render_run_context(BottleneckType.COMPUTE_BOUND, hardware=hw)
    peak_line = [ln for ln in out.splitlines() if ln.startswith("- Peak FLOPS")][0]
    # fp8 (highest) appears first, then the tied bf16/fp16, then fp32.
    assert "fp8=" in peak_line
    assert "bf16/fp16=" in peak_line
    assert "fp32=" in peak_line
    assert (peak_line.index("fp8=")
            < peak_line.index("bf16/fp16=")
            < peak_line.index("fp32="))
    # Old single-dtype line format is gone.
    assert "Peak FLOPS (fp8):" not in out
