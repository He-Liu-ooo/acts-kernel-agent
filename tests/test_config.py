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
    # MAC_per_cycle_fp32_sm * freq_GHz * 2 / 1e3 = 25500 * 2 * 2 / 1e3 = 102.0 TFLOPS
    assert abs(spec.peak_flops_fp32 - 102.0) < 0.1


def test_derived_peak_flops_bf16():
    spec = load_hardware_spec(_write_yaml(_H100_YAML))
    # MAC_per_cycle_bf16_tc * freq_GHz * 2 / 1e3 = 378000 * 2 * 2 / 1e3 = 1512.0 TFLOPS
    assert abs(spec.peak_flops_bf16 - 1512.0) < 0.1


def test_peak_flops_unit_is_tflops_not_pflops():
    """Regression: ``peak_flops_*`` must return TFLOPS, not PFLOPS.

    The formula is ops/sec / 1e12 = MAC_per_cycle * freq_GHz * 1e9 * 2 / 1e12
    = MAC_per_cycle * freq_GHz * 2 / 1e3. A previous version divided by 1e6
    (PFLOPS) — downstream consumers in eval/profiler.py and eval/roofline.py
    treat the value as TFLOPS, so the off-by-1000x silently broke roofline /
    bottleneck classification on hosts that loaded a real arch YAML.

    Hand-constructed spec keeps this torch-free."""
    spec = HardwareSpec(
        MAC_per_cycle_fp32_sm=10000,
        MAC_per_cycle_bf16_tc=20000,
        MAC_per_cycle_fp16_tc=30000,
        freq_GHz=2.0,
    )
    # 10000 * 2.0 * 2 / 1e3 = 40.0 TFLOPS
    assert spec.peak_flops_fp32 == pytest.approx(40.0)
    # 20000 * 2.0 * 2 / 1e3 = 80.0 TFLOPS
    assert spec.peak_flops_bf16 == pytest.approx(80.0)
    # 30000 * 2.0 * 2 / 1e3 = 120.0 TFLOPS
    assert spec.peak_flops_fp16 == pytest.approx(120.0)


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
    """With a GPU whose name is NOT registered in ``_ACTS_ARCH_YAMLS``,
    populate only name / freq_GHz / SRAM_capacity / DRAM_capacity from
    torch.cuda.get_device_properties. Per-precision throughput tables
    stay at zero — those need a registered SOLAR arch YAML.

    (For the merged-with-YAML path, see
    ``test_detect_hardware_merges_yaml_when_name_matches_known_stem``.)"""
    props = SimpleNamespace(
        name="NVIDIA Unregistered Test GPU",
        clock_rate=2_505_000,        # kHz → 2.505 GHz
        L2_cache_size=100_663_296,
        total_memory=50_876_841_984,
    )
    fake = _fake_torch(cuda_available=True, props=props)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()

    assert spec.name == "NVIDIA Unregistered Test GPU"
    assert abs(spec.freq_GHz - 2.505) < 1e-6
    assert spec.SRAM_capacity == 100_663_296
    assert spec.DRAM_capacity == 50_876_841_984
    # Per-precision tables intentionally left at zero (no YAML merged).
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


def test_load_config_reviewer_metric_queries_from_cfg():
    """`load_config` parses search.reviewer_metric_queries via the
    existing boolean coercion path used by `beam_diversity`."""
    from src.config import load_config

    cfg_text = "search: { reviewer_metric_queries = true; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.reviewer_metric_queries is True


def test_load_config_reviewer_metric_queries_omitted_uses_default():
    """When ``search`` omits the key, fall back to the dataclass default."""
    from src.config import load_config

    cfg_text = "search: { beam_width = 5; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.reviewer_metric_queries is False


# ── A2: coder_n_candidates (K-way Coder fan-out) ──────────────────────


def test_acts_config_coder_n_candidates_default_4():
    """A2: default K is 4 — matches AccelOpt num_samples=2 × breadth=2 = 4
    plan-side cardinality and the canonical best-of-N value in code-gen
    literature."""
    from src.config import ACTSConfig

    cfg = ACTSConfig()
    assert cfg.coder_n_candidates == 4


def test_load_config_coder_n_candidates_from_cfg():
    """A2: ``[search] coder_n_candidates = 8`` overrides the default."""
    from src.config import load_config

    cfg_text = "search: { coder_n_candidates = 8; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.coder_n_candidates == 8


def test_load_config_coder_n_candidates_omitted_uses_default():
    """A2: omitted setting → dataclass default (4)."""
    from src.config import load_config

    cfg_text = "search: { beam_width = 5; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.coder_n_candidates == 4


def test_acts_config_coder_n_candidates_zero_raises():
    """A2: K=0 would silently skip every iter ("all 0 candidates failed");
    __post_init__ fails fast instead. Codex review #3."""
    import pytest
    from src.config import ACTSConfig

    with pytest.raises(ValueError, match="coder_n_candidates"):
        ACTSConfig(coder_n_candidates=0)


def test_acts_config_coder_n_candidates_negative_raises():
    """A2: a negative typo would mean ``range(K)`` dispatches no Coder
    calls — same silent-skip failure mode. Codex review #3."""
    import pytest
    from src.config import ACTSConfig

    with pytest.raises(ValueError, match="coder_n_candidates"):
        ACTSConfig(coder_n_candidates=-1)


def test_load_config_coder_n_candidates_zero_raises(tmp_path):
    """A2: K=0 from the cfg file fails fast at load_config too."""
    import pytest
    from src.config import load_config

    cfg_text = "search: { coder_n_candidates = 0; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        with pytest.raises(ValueError, match="coder_n_candidates"):
            load_config(Path(f.name))


def test_acts_config_coder_n_candidates_non_int_raises():
    """A2: a float (e.g. typo'd ``1.5``) would crash ``range(K)`` deep in
    the iter loop with a confusing TypeError. Reject at config load."""
    import pytest
    from src.config import ACTSConfig

    with pytest.raises(TypeError, match="coder_n_candidates"):
        ACTSConfig(coder_n_candidates=1.5)


def test_acts_config_coder_n_candidates_bool_accepted():
    """A2: ``bool`` is a subclass of ``int`` in Python, so ``True``
    passes both the isinstance and the ``>= 1`` checks (``True == 1``).
    Treat this as benign — no need to special-case bool out."""
    from src.config import ACTSConfig

    cfg = ACTSConfig(coder_n_candidates=True)
    assert cfg.coder_n_candidates == 1  # noqa: E712 — bool→int coercion intent


# ── planner_max_turns / reviewer_max_turns (cfg-tunable agent budgets) ──


def test_acts_config_planner_max_turns_default_none():
    """Default = None means \"preserve hardcoded planner.py budget of 4\".
    Non-None overrides. Keeps existing-runs behavior identical."""
    from src.config import ACTSConfig

    cfg = ACTSConfig()
    assert cfg.planner_max_turns is None


def test_acts_config_reviewer_max_turns_default_none():
    """Default = None means \"preserve hardcoded reviewer.py 4/6 toggle\".
    Non-None overrides both branches of the metric_queries conditional."""
    from src.config import ACTSConfig

    cfg = ACTSConfig()
    assert cfg.reviewer_max_turns is None


def test_load_config_planner_max_turns_from_cfg():
    """`[search] planner_max_turns = 6` overrides the default."""
    from src.config import load_config

    cfg_text = "search: { planner_max_turns = 6; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.planner_max_turns == 6


def test_load_config_reviewer_max_turns_from_cfg():
    """`[search] reviewer_max_turns = 8` overrides the default."""
    from src.config import load_config

    cfg_text = "search: { reviewer_max_turns = 8; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.reviewer_max_turns == 8


def test_load_config_planner_max_turns_omitted_uses_none():
    """Omitted setting → None → preserves planner.py's hardcoded budget."""
    from src.config import load_config

    cfg_text = "search: { beam_width = 5; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.planner_max_turns is None


def test_load_config_reviewer_max_turns_omitted_uses_none():
    """Omitted setting → None → preserves reviewer.py's 4/6 toggle."""
    from src.config import load_config

    cfg_text = "search: { beam_width = 5; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.reviewer_max_turns is None


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
    cfg_text = f'hardware: {{ arch_config_path = "{yaml_path}"; }};\n'
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


# ── detect_hardware() auto-merge of configs/arch/<name>.yaml ──────────


def test_detect_hardware_merges_yaml_when_name_matches_known_stem():
    """When the detected GPU name is registered in ``_ACTS_ARCH_YAMLS``
    and the YAML exists, ``detect_hardware`` returns a fully-populated
    HardwareSpec — runtime ground-truth for name/freq/capacities, YAML
    for the per-precision throughput tables. This is the bug fix: prior
    to the merge, callers got ``MAC_per_cycle_bf16_tc=0`` and
    ``DRAM_byte_per_cycle=0``, which routed pipeline/optimize.py through
    the placeholder substitution and skipped SOLAR's roofline."""
    props = SimpleNamespace(
        name="NVIDIA RTX 6000 Ada Generation",
        clock_rate=2_505_000,        # kHz → 2.505 GHz
        L2_cache_size=100_663_296,
        total_memory=51_539_607_552,
    )
    fake = _fake_torch(cuda_available=True, props=props)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()

    # Runtime fields preserved (the GPU is ground-truth for these).
    assert spec.name == "NVIDIA RTX 6000 Ada Generation"
    assert abs(spec.freq_GHz - 2.505) < 1e-6
    assert spec.SRAM_capacity == 100_663_296
    assert spec.DRAM_capacity == 51_539_607_552
    # YAML-supplied throughput tables non-zero — SOLAR roofline + sol_score
    # math now have real numbers to work with.
    assert spec.MAC_per_cycle_bf16_tc > 0
    assert spec.MAC_per_cycle_fp32_sm > 0
    assert spec.DRAM_byte_per_cycle > 0
    assert spec.SRAM_byte_per_cycle > 0
    # Derived peaks must therefore be non-zero too.
    assert spec.peak_flops_bf16 > 0
    assert spec.peak_memory_bandwidth_gb_s > 0


def test_detect_hardware_unknown_name_returns_zero_peaks():
    """When the detected GPU isn't registered in ``_ACTS_ARCH_YAMLS``,
    behavior is unchanged from the pre-fix code: runtime fields populated,
    throughput tables zero. The orchestrator's placeholder-substitution
    path then engages downstream — the existing fallback contract."""
    props = SimpleNamespace(
        name="NVIDIA RTX 9999 Made Up GPU",
        clock_rate=1_500_000,
        L2_cache_size=50_000_000,
        total_memory=24 * 1024**3,
    )
    fake = _fake_torch(cuda_available=True, props=props)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()

    assert spec.name == "NVIDIA RTX 9999 Made Up GPU"
    assert spec.SRAM_capacity == 50_000_000
    assert spec.MAC_per_cycle_bf16_tc == 0.0
    assert spec.DRAM_byte_per_cycle == 0.0
    assert spec.peak_flops_bf16 == 0.0


def test_detect_hardware_yaml_mismatch_logs_warning_does_not_raise(caplog):
    """If the YAML registered for the detected name disagrees with the
    runtime fields by >10% (stale YAML / wrong registry entry), log a
    WARNING per mismatched field but still return the merged spec. The
    operator sees the silent miscalibration without the load path
    breaking on a partial-mismatch."""
    # Simulate: detected name matches the Ada YAML registry entry, but
    # the device reports H100 capacities (~80 GiB DRAM, ~50 MiB L2).
    # validate_hardware_spec should flag DRAM + SRAM mismatches.
    props = SimpleNamespace(
        name="NVIDIA RTX 6000 Ada Generation",
        clock_rate=2_505_000,
        L2_cache_size=52_428_800,           # ~50 MiB (H100, not Ada's 96 MiB)
        total_memory=85_899_345_920,        # ~80 GiB (H100, not Ada's 48 GiB)
    )
    fake = _fake_torch(cuda_available=True, props=props)
    with (
        patch.dict(sys.modules, {"torch": fake}),
        caplog.at_level("WARNING", logger="src.config"),
    ):
        spec = detect_hardware()

    assert any("DRAM capacity mismatch" in rec.message for rec in caplog.records)
    # Spec still returned (warning, not raise) and runtime ground-truth
    # for the mismatched fields wins.
    assert spec.DRAM_capacity == 85_899_345_920
    assert spec.SRAM_capacity == 52_428_800
    # YAML-supplied throughput tables still merge in.
    assert spec.MAC_per_cycle_bf16_tc > 0


def test_detect_hardware_no_gpu_path_unchanged_after_merge():
    """No-GPU regression: with CUDA unavailable, detect_hardware still
    returns a fully-zeroed spec — the YAML merge logic must not engage
    when there's no detected name to look up."""
    fake = _fake_torch(cuda_available=False)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()
    assert spec == HardwareSpec()
    assert spec.MAC_per_cycle_bf16_tc == 0.0


def test_detect_hardware_yaml_load_failure_falls_back_to_runtime_only(
    tmp_path, monkeypatch, caplog,
):
    """If the registered YAML exists but ``yaml.safe_load`` chokes on it
    (corrupted file, encoding mismatch), don't crash — log a warning and
    return the runtime-only HardwareSpec so the placeholder path can
    still engage downstream."""
    from src import config as cfg_module

    bogus_yaml = tmp_path / "bogus.yaml"
    bogus_yaml.write_text("name: 'X\nthis_is: not_yaml: at_all\n")

    monkeypatch.setitem(
        cfg_module._ACTS_ARCH_YAMLS, "NVIDIA Test Bogus", bogus_yaml,
    )

    props = SimpleNamespace(
        name="NVIDIA Test Bogus",
        clock_rate=2_000_000,
        L2_cache_size=50_000_000,
        total_memory=24 * 1024**3,
    )
    fake = _fake_torch(cuda_available=True, props=props)
    with (
        patch.dict(sys.modules, {"torch": fake}),
        caplog.at_level("WARNING", logger="src.config"),
    ):
        spec = detect_hardware()

    assert spec.name == "NVIDIA Test Bogus"
    assert spec.MAC_per_cycle_bf16_tc == 0.0
    assert any("failed to load" in rec.message for rec in caplog.records)


# ── new fields absorbed from argparse (2026-05-11) ───────────────────────────


def test_load_config_gpu_index_from_cfg():
    """`load_config` parses hardware.gpu_index — previously a CLI flag."""
    from src.config import load_config

    cfg_text = "hardware: { gpu_index = 3; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.gpu_index == 3


def test_load_config_gpu_index_omitted_uses_default():
    """Absent gpu_index falls back to ACTSConfig default (0)."""
    from src.config import load_config

    cfg_text = "hardware: { };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.gpu_index == 0


def test_load_config_reset_clocks_from_cfg():
    """`load_config` parses runtime.reset_clocks — previously a CLI flag."""
    from src.config import load_config

    cfg_text = "runtime: { reset_clocks = true; };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.reset_clocks is True


def test_load_config_reset_clocks_omitted_uses_default():
    """Absent reset_clocks falls back to ACTSConfig default (False)."""
    from src.config import load_config

    cfg_text = "runtime: { };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.reset_clocks is False


def test_load_config_problem_path_from_cfg():
    """`load_config` parses runtime.problem_path — previously a CLI positional."""
    from src.config import load_config

    cfg_text = (
        'runtime: { problem_path = '
        '"repo/benchmark/SOL-ExecBench/examples/triton/rmsnorm"; };\n'
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.problem_path == "repo/benchmark/SOL-ExecBench/examples/triton/rmsnorm"


def test_load_config_problem_path_omitted_uses_default():
    """Absent problem_path falls back to ACTSConfig default ('placeholder')."""
    from src.config import load_config

    cfg_text = "runtime: { };\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(cfg_text)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.problem_path == "placeholder"


# ── HardwareSpec shared-memory fields (hw-spec injection Task 1) ───────


_SMEM_YAML = """\
name: "TestGPU"
freq_GHz: 2.0
compute_capability: 8.9
shared_mem_per_block_bytes: 101376
shared_mem_per_multiprocessor_bytes: 102400
"""


def test_hardware_spec_new_smem_fields_load_from_yaml():
    """YAML carries the two new shared-memory fields; load_hardware_spec reads them."""
    spec = load_hardware_spec(_write_yaml(_SMEM_YAML))
    assert spec.shared_mem_per_block_bytes == 101376
    assert spec.shared_mem_per_multiprocessor_bytes == 102400


def test_detect_hardware_populates_smem_fields():
    """detect_hardware reads shared_memory_per_block_optin + per_multiprocessor."""
    props = SimpleNamespace(
        name="FakeGPU",
        major=8, minor=9,
        total_memory=48 * 1024**3,
        L2_cache_size=96 * 1024**2,
        clock_rate=2_505_000,  # kHz → 2.505 GHz
        shared_memory_per_block_optin=101376,
        shared_memory_per_multiprocessor=102400,
    )
    fake = _fake_torch(cuda_available=True, props=props)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()
    assert spec.shared_mem_per_block_bytes == 101376
    assert spec.shared_mem_per_multiprocessor_bytes == 102400


def test_detect_hardware_falls_back_when_optin_missing():
    """Older torch lacking ``shared_memory_per_block_optin`` falls back to
    ``shared_memory_per_block``. The ``_optin`` suffix was added in recent
    torch; the docstring promise of "older driver compat" requires the
    fallback path."""
    props = SimpleNamespace(
        name="FakeGPU",
        major=8, minor=9,
        total_memory=48 * 1024**3,
        L2_cache_size=96 * 1024**2,
        clock_rate=2_505_000,
        # NO shared_memory_per_block_optin attribute on purpose
        shared_memory_per_block=49152,
        shared_memory_per_multiprocessor=102400,
    )
    fake = _fake_torch(cuda_available=True, props=props)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()
    assert spec.shared_mem_per_block_bytes == 49152  # fallback path
    assert spec.shared_mem_per_multiprocessor_bytes == 102400


def test_validate_hardware_spec_tolerates_smem_mismatch_under_threshold():
    """A ~2% diff in ``shared_mem_per_block_bytes`` does NOT flag (under 10%)."""
    yaml_spec = HardwareSpec(name="X", shared_mem_per_block_bytes=101376)
    detected = HardwareSpec(name="X", shared_mem_per_block_bytes=99000)
    issues = validate_hardware_spec(yaml_spec, detected)
    assert not any("shared_mem_per_block" in i for i in issues)


def test_validate_hardware_spec_flags_smem_mismatch_over_threshold():
    """A >2x diff in ``shared_mem_per_block_bytes`` triggers a warning."""
    yaml_spec = HardwareSpec(name="X", shared_mem_per_block_bytes=49152)
    detected = HardwareSpec(name="X", shared_mem_per_block_bytes=101376)
    issues = validate_hardware_spec(yaml_spec, detected)
    assert any("shared_mem_per_block" in i for i in issues)


# ── HardwareSpec count fields (run-context enrichment, 2026-05-25) ─────


_COUNT_YAML = """\
name: "TestGPU"
freq_GHz: 2.0
compute_capability: 8.9
sm_count: 142
max_threads_per_block: 1024
"""


def test_hardware_spec_new_count_fields_load_from_yaml():
    """YAML carries the two new count fields; load_hardware_spec reads them."""
    spec = load_hardware_spec(_write_yaml(_COUNT_YAML))
    assert spec.sm_count == 142
    assert spec.max_threads_per_block == 1024


def test_detect_hardware_populates_count_fields():
    """detect_hardware reads multi_processor_count + max_threads_per_block via getattr."""
    props = SimpleNamespace(
        name="FakeGPU",
        major=8, minor=9,
        total_memory=48 * 1024**3,
        L2_cache_size=96 * 1024**2,
        clock_rate=2_505_000,
        shared_memory_per_block_optin=101376,
        shared_memory_per_multiprocessor=102400,
        multi_processor_count=142,
        max_threads_per_block=1024,
    )
    fake = _fake_torch(cuda_available=True, props=props)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()
    assert spec.sm_count == 142
    assert spec.max_threads_per_block == 1024


def test_validate_hardware_spec_flags_sm_count_mismatch():
    """A >10% diff in ``sm_count`` triggers a warning."""
    yaml_spec = HardwareSpec(name="X", sm_count=80)
    detected = HardwareSpec(name="X", sm_count=142)
    issues = validate_hardware_spec(yaml_spec, detected)
    assert any("sm_count" in i for i in issues)


# ── Bench-subprocess isolation knobs (2026-05-24) ──────────────────────


def test_acts_config_bench_use_subprocess_defaults_true():
    """Per-iter bench + NCU subprocess is on by default (production path);
    flip to False for in-process debugging. Mirrors use_operator_baseline
    pattern. See doc/specs/2026-05-24-bench-subprocess-isolation-design.md §3."""
    from src.config import ACTSConfig
    cfg = ACTSConfig()
    assert cfg.bench_use_subprocess is True


def test_acts_config_worker_crash_threshold_defaults_three():
    """3 consecutive worker crashes → WorkerProcessUnstable whole-run abort.
    Mirrors CUDAContextPoisoned's 3-strike escalation."""
    from src.config import ACTSConfig
    cfg = ACTSConfig()
    assert cfg.worker_crash_threshold == 3


def test_acts_config_worker_timeout_defaults_180000():
    """Total-lifetime watchdog on ``proc.wait()``. Default ~50h
    effectively disables the watchdog while the subprocess refactor
    beds in — the original 30s default killed healthy workers mid-NCU
    (Codex 2026-05-26). Operators with hard wallclock budgets override
    via ``[runtime] worker_timeout_s`` in cfg. Field was renamed from
    ``worker_startup_timeout_s`` to reflect actual total-lifetime
    semantics."""
    from src.config import ACTSConfig
    cfg = ACTSConfig()
    assert cfg.worker_timeout_s == 180000.0


def test_acts_config_worker_crash_threshold_rejects_zero():
    import pytest
    from src.config import ACTSConfig
    with pytest.raises(ValueError, match="worker_crash_threshold"):
        ACTSConfig(worker_crash_threshold=0)


def test_acts_config_worker_timeout_rejects_non_positive():
    import pytest
    from src.config import ACTSConfig
    with pytest.raises(ValueError, match="worker_timeout_s"):
        ACTSConfig(worker_timeout_s=0.0)
