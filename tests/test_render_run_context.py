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


def test_render_run_context_dominant_dtype_single_highest():
    """When one dtype's peak strictly exceeds the others, only that name is rendered."""
    hw = HardwareSpec(
        name="X", compute_capability=9.0, freq_GHz=1.0,
        MAC_per_cycle_fp32_sm=1000,
        MAC_per_cycle_bf16_tc=500,
        MAC_per_cycle_fp16_tc=400,
        shared_mem_per_block_bytes=1, shared_mem_per_multiprocessor_bytes=1,
    )
    out = render_run_context(BottleneckType.COMPUTE_BOUND, hardware=hw)
    assert "Peak FLOPS (fp32):" in out
    # No spurious join when there's no tie.
    assert "Peak FLOPS (fp32/" not in out
    assert "Peak FLOPS (bf16/fp16):" not in out


def test_render_run_context_dominant_dtype_tie_renders_slash_joined():
    """Ada-like case where bf16 == fp16: renders both joined alphabetically with /.

    Per spec §3 decision 8 — tied dtypes joined by '/' in alphabetical order.
    """
    hw = HardwareSpec(
        name="X", compute_capability=8.9, freq_GHz=2.505,
        MAC_per_cycle_fp32_sm=18185,
        MAC_per_cycle_bf16_tc=72695,
        MAC_per_cycle_fp16_tc=72695,  # tied with bf16
        shared_mem_per_block_bytes=1, shared_mem_per_multiprocessor_bytes=1,
    )
    out = render_run_context(BottleneckType.COMPUTE_BOUND, hardware=hw)
    assert "Peak FLOPS (bf16/fp16):" in out


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


def test_render_run_context_dominant_dtype_picks_fp8_on_hopper_spec():
    """On Hopper/Blackwell, fp8 peak is typically 2× the fp16 peak. The
    dominant-dtype renderer must consider fp8 (and nvfp4 on Blackwell) —
    not just fp32/bf16/fp16 — otherwise the rendered peak understates the
    real compute throughput (Fix #11)."""
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
    assert "Peak FLOPS (fp8):" in out
    assert "Peak FLOPS (fp16):" not in out
    assert "Peak FLOPS (bf16/fp16):" not in out
