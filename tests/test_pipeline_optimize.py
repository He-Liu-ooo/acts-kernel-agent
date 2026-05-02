"""Tests for pipeline/optimize.py — Phase A → Phase B wiring.

Verifies that (1) Phase A produces a reference_fn and *every* selected workload's
input generator and forwards the full list into ``Orchestrator.run``, and
(2) the placeholder path never loads a model, so the default CLI smoke-path
(`python -m src.pipeline.optimize`) stays runnable once a model config exists
on disk — a model-backed Coder would raise ImplementationError against the
stub baseline on the first iteration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import ACTSConfig, HardwareSpec
from src.kernels.kernel import Kernel, KernelSpec, KernelType
from src.pipeline.optimize import _load_problem, optimize


def _spec() -> KernelSpec:
    return KernelSpec(
        name="t",
        kernel_type=KernelType.ELEMENTWISE,
        entrypoint="kernel_fn",
        pytorch_reference="def run(x):\n    return x * 2.0\n",
    )


@pytest.mark.asyncio
async def test_load_problem_returns_reference_fn_and_all_generators():
    """Phase A must return a reference_fn and one generator per selected workload
    so Phase B's correctness tool binds to the full coverage set. Collapsing to
    just workloads[0] lets kernels that pass workload 1 but break 2..N slip through."""
    from sol_execbench.core.data import Definition, Workload

    definition = Definition.model_validate({
        "name": "p",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x): return x * 2.0\n",
        "op_type": "elementwise",
    })
    workloads = [
        Workload.model_validate({"uuid": f"wl-{i}", "axes": {"N": 8}, "inputs": {}})
        for i in range(3)
    ]
    spec = _spec()
    baseline = Kernel(spec=spec, source_code="src")

    ref_fn = lambda x: x * 2.0
    gens = [lambda seed, i=i: (i, seed) for i in range(3)]

    # Explicit ``benchmark_adapter="sol_execbench"`` so the dispatcher
    # routes to the SOL adapter without needing a real definition.json
    # on disk (path is /fake).
    config = ACTSConfig(benchmark_adapter="sol_execbench")

    with (
        patch(
            "src.benchmarks.sol_execbench.load",
            return_value=(definition, workloads),
        ),
        patch(
            "src.pipeline.optimize._definition_to_kernel_spec", return_value=spec,
        ),
        patch("src.benchmark.workload_selector.select_workloads", return_value=workloads),
        patch("src.benchmark.solar_adapter.is_solar_available", return_value=True),
        patch("src.eval.roofline.derive_t_sol_from_solar", return_value=None),
        patch(
            "src.benchmark.baseline_generator.generate_triton_baseline",
            new_callable=AsyncMock,
            return_value=baseline,
        ),
        patch("src.eval.inputs.build_reference_fn", return_value=ref_fn),
        patch("src.eval.inputs.build_input_generator", side_effect=gens),
    ):
        result = await _load_problem(Path("/fake"), config, MagicMock())

    # Expect 7-tuple: (baseline, definition, workloads, roofline,
    #                   reference_fn, input_generators, definition_path)
    assert len(result) == 7
    (
        _baseline, _definition, _workloads, _roofline,
        got_ref, got_gens, _definition_path,
    ) = result
    assert got_ref is ref_fn
    assert got_gens == gens  # all three, in workload order


@pytest.mark.asyncio
async def test_load_problem_forwards_arch_config_path_to_solar():
    """Regression for the silent-arch-fallback bug: when the user configures
    ``[hardware] arch_config_path`` (i.e. ``ACTSConfig.arch_config_path`` is
    non-empty), Phase A must forward that path into the SOLAR adapter as
    ``arch_yaml_path``. Otherwise the adapter falls back to name-based
    lookup and lands on H100_PCIe for any unrecognized hardware name —
    silently corrupting T_SOL / sol_score for the entire run."""
    from sol_execbench.core.data import Definition, Workload

    definition = Definition.model_validate({
        "name": "p",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x): return x\n",
    })
    workloads = [
        Workload.model_validate({"uuid": "w1", "axes": {"N": 8}, "inputs": {}})
    ]
    spec = _spec()
    baseline = Kernel(spec=spec, source_code="src")

    config = ACTSConfig(
        hardware=HardwareSpec(name="some_custom_gpu"),
        arch_config_path="/some/custom/arch.yaml",
        benchmark_adapter="sol_execbench",
    )

    captured: dict = {}

    def _capture(definition_, workload_, hardware_, arch_yaml_path=None):
        captured["arch_yaml_path"] = arch_yaml_path
        return None

    with (
        patch(
            "src.benchmarks.sol_execbench.load",
            return_value=(definition, workloads),
        ),
        patch(
            "src.pipeline.optimize._definition_to_kernel_spec", return_value=spec,
        ),
        patch("src.benchmark.workload_selector.select_workloads", return_value=workloads),
        patch("src.benchmark.solar_adapter.is_solar_available", return_value=True),
        patch("src.eval.roofline.derive_t_sol_from_solar", side_effect=_capture),
        patch(
            "src.benchmark.baseline_generator.generate_triton_baseline",
            new_callable=AsyncMock,
            return_value=baseline,
        ),
        patch("src.eval.inputs.build_reference_fn", return_value=lambda x: x),
        patch("src.eval.inputs.build_input_generator", return_value=lambda s: ()),
    ):
        await _load_problem(Path("/fake"), config, MagicMock())

    assert captured["arch_yaml_path"] == Path("/some/custom/arch.yaml")


@pytest.mark.asyncio
async def test_load_problem_passes_none_when_arch_config_path_empty():
    """When ``arch_config_path`` is empty (the default), Phase A passes
    ``arch_yaml_path=None`` and the adapter resolves by name. This keeps
    the runtime-detected hardware path working without a config file."""
    from sol_execbench.core.data import Definition, Workload

    definition = Definition.model_validate({
        "name": "p",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x): return x\n",
    })
    workloads = [
        Workload.model_validate({"uuid": "w1", "axes": {"N": 8}, "inputs": {}})
    ]
    spec = _spec()
    baseline = Kernel(spec=spec, source_code="src")

    captured: dict = {}

    def _capture(definition_, workload_, hardware_, arch_yaml_path=None):
        captured["arch_yaml_path"] = arch_yaml_path
        return None

    with (
        patch(
            "src.benchmarks.sol_execbench.load",
            return_value=(definition, workloads),
        ),
        patch(
            "src.pipeline.optimize._definition_to_kernel_spec", return_value=spec,
        ),
        patch("src.benchmark.workload_selector.select_workloads", return_value=workloads),
        patch("src.benchmark.solar_adapter.is_solar_available", return_value=True),
        patch("src.eval.roofline.derive_t_sol_from_solar", side_effect=_capture),
        patch(
            "src.benchmark.baseline_generator.generate_triton_baseline",
            new_callable=AsyncMock,
            return_value=baseline,
        ),
        patch("src.eval.inputs.build_reference_fn", return_value=lambda x: x),
        patch("src.eval.inputs.build_input_generator", return_value=lambda s: ()),
    ):
        await _load_problem(
            Path("/fake"),
            ACTSConfig(benchmark_adapter="sol_execbench"),
            MagicMock(),
        )

    assert captured["arch_yaml_path"] is None


@pytest.mark.asyncio
async def test_load_sol_problem_fails_fast_when_solar_unavailable():
    """SOL-ExecBench problems leave ``KernelSpec.flop_count`` /
    ``memory_bytes`` at zero — the orchestrator's built-in
    ``compute_roofline`` fallback (which fires when SOLAR returns
    ``None``) silently produces ``t_sol_us=0.0``, corrupting every
    score with no visible diagnostic. The adapter must refuse to load
    the problem so the operator sees the actionable install hint
    immediately, not at score-emit time.

    Regression for the 2026-04-30 live-run bug where every
    ``score_computed`` event reported ``t_sol_source="builtin"`` and
    ``t_sol_us=0.0`` because SOLAR was missing from the run venv."""
    config = ACTSConfig(benchmark_adapter="sol_execbench")

    with patch(
        "src.benchmark.solar_adapter.is_solar_available", return_value=False
    ):
        with pytest.raises(RuntimeError, match=r"SOLAR is required"):
            await _load_problem(Path("/fake"), config, MagicMock())


@pytest.mark.asyncio
async def test_load_sol_problem_threads_solar_source_into_roofline():
    """Regression: when SOLAR succeeds, the ``RooflineResult`` returned
    to the orchestrator must carry ``source="solar"`` so the per-iter
    ``score_computed`` event reports the correct provenance.

    Together with the fail-fast test above, this pins the two halves of
    the live-run bug: (1) SOLAR-absent must not silently land on
    ``builtin`` + ``t_sol_us=0.0``; (2) SOLAR-present must thread
    ``source="solar"`` end-to-end."""
    from sol_execbench.core.data import Definition, Workload

    from src.eval.roofline import BottleneckType, RooflineResult

    definition = Definition.model_validate({
        "name": "p",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x): return x\n",
    })
    workloads = [
        Workload.model_validate({"uuid": "w1", "axes": {"N": 8}, "inputs": {}})
    ]
    spec = _spec()
    baseline = Kernel(spec=spec, source_code="src")

    solar_roofline = RooflineResult(
        t_sol_us=12.5,
        bottleneck=BottleneckType.MEMORY_BOUND,
        source="solar",
    )

    config = ACTSConfig(benchmark_adapter="sol_execbench")

    with (
        patch(
            "src.benchmarks.sol_execbench.load",
            return_value=(definition, workloads),
        ),
        patch(
            "src.pipeline.optimize._definition_to_kernel_spec", return_value=spec,
        ),
        patch("src.benchmark.workload_selector.select_workloads", return_value=workloads),
        patch("src.benchmark.solar_adapter.is_solar_available", return_value=True),
        patch(
            "src.eval.roofline.derive_t_sol_from_solar",
            return_value=solar_roofline,
        ),
        patch(
            "src.benchmark.baseline_generator.generate_triton_baseline",
            new_callable=AsyncMock,
            return_value=baseline,
        ),
        patch("src.eval.inputs.build_reference_fn", return_value=lambda x: x),
        patch("src.eval.inputs.build_input_generator", return_value=lambda s: ()),
    ):
        result = await _load_problem(Path("/fake"), config, MagicMock())

    _baseline, _definition, _workloads, roofline, *_rest = result
    assert roofline is solar_roofline
    assert roofline.source == "solar"
    assert roofline.t_sol_us == 12.5


@pytest.mark.asyncio
async def test_optimize_forwards_correctness_context_to_orchestrator():
    """reference_fn + full generator list from Phase A reach Orchestrator.run()
    as kwargs so the Coder's correctness tool binds to every selected workload."""
    ref_fn = lambda x: x
    gens = [lambda seed, i=i: (i, seed) for i in range(3)]
    baseline = Kernel(spec=_spec(), source_code="src")

    fake_result = MagicMock()
    fake_orch = MagicMock()
    fake_orch.run = AsyncMock(return_value=fake_result)
    store_instance = MagicMock()

    # Placeholder path: reference_fn is None, input_generators is empty — the
    # baseline is a stub so there's nothing the correctness tool could bind to.
    with (
        patch("src.pipeline.optimize._load_model_if_configured", return_value=None),
        patch("src.search.orchestrator.Orchestrator", return_value=fake_orch),
        patch("src.memory.store.MemoryStore", return_value=store_instance),
        patch(
            "src.kernels.starters.matmul.make_matmul_kernel", return_value=baseline,
        ),
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
    ):
        await optimize("placeholder")

    kwargs = fake_orch.run.call_args.kwargs
    assert "reference_fn" in kwargs
    assert "input_generators" in kwargs
    assert kwargs["reference_fn"] is None
    assert kwargs["input_generators"] == []

    # SOL-ExecBench path: reference_fn and the full generator list come back
    # from _load_problem and reach Orchestrator.run as kwargs.
    fake_orch.run.reset_mock()
    with (
        patch("src.pipeline.optimize._load_model_if_configured", return_value=None),
        patch("src.search.orchestrator.Orchestrator", return_value=fake_orch),
        patch("src.memory.store.MemoryStore", return_value=store_instance),
        patch(
            "src.pipeline.optimize._load_problem",
            new_callable=AsyncMock,
            return_value=(
                baseline, MagicMock(), [MagicMock()] * 3, None,
                ref_fn, gens, Path("/fake/definition.json"),
            ),
        ),
        patch("pathlib.Path.is_dir", return_value=True),
        patch.object(Path, "exists", autospec=True, return_value=True),
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
    ):
        await optimize("/fake/problem")

    kwargs = fake_orch.run.call_args.kwargs
    assert kwargs["reference_fn"] is ref_fn
    assert kwargs["input_generators"] == gens


@pytest.mark.asyncio
async def test_placeholder_substitutes_nonzero_hardware_spec():
    """``detect_hardware()`` returns a zeroed HardwareSpec until real detection
    lands; feeding that into ``Orchestrator.run`` trips the fail-fast guard and
    the placeholder CLI dies before the first iteration. ``optimize()`` must
    substitute a populated placeholder so ``python -m src.pipeline.optimize``
    stays runnable on a machine without an arch YAML."""
    baseline = Kernel(spec=_spec(), source_code="src")
    fake_orch = MagicMock()
    fake_orch.run = AsyncMock(return_value=MagicMock())

    with (
        patch("src.pipeline.optimize._load_model_if_configured", return_value=None),
        patch("src.config.detect_hardware", return_value=HardwareSpec()),  # zeros
        patch("src.search.orchestrator.Orchestrator", return_value=fake_orch) as mock_orch_cls,
        patch("src.memory.store.MemoryStore", return_value=MagicMock()),
        patch("src.kernels.starters.matmul.make_matmul_kernel", return_value=baseline),
    ):
        await optimize("placeholder")

    config = mock_orch_cls.call_args.kwargs["config"]
    assert config.hardware.peak_flops_fp32 > 0, (
        "optimize() must substitute a populated placeholder HardwareSpec when "
        "detect_hardware() returns zeroed peaks"
    )
    assert config.hardware.peak_memory_bandwidth_gb_s > 0


@pytest.mark.asyncio
async def test_zero_peak_caller_config_also_gets_placeholder_substituted():
    """A caller who passes a bare ``ACTSConfig()`` (or any config whose
    HardwareSpec has zero peaks) must NOT trip the orchestrator's fail-fast
    guard — the same placeholder substitution that runs for the
    ``config is None`` path must apply to caller-supplied configs too.
    Before this fix, the ``config is None`` branch was skipped and the
    orchestrator raised ``ValueError`` before the first iteration."""
    baseline = Kernel(spec=_spec(), source_code="src")
    fake_orch = MagicMock()
    fake_orch.run = AsyncMock(return_value=MagicMock())

    caller_config = ACTSConfig()  # HardwareSpec() → zero peaks

    with (
        patch("src.pipeline.optimize._load_model_if_configured", return_value=None),
        patch("src.search.orchestrator.Orchestrator", return_value=fake_orch) as mock_orch_cls,
        patch("src.memory.store.MemoryStore", return_value=MagicMock()),
        patch("src.kernels.starters.matmul.make_matmul_kernel", return_value=baseline),
    ):
        await optimize("placeholder", config=caller_config)

    passed = mock_orch_cls.call_args.kwargs["config"]
    assert passed.hardware.peak_flops_fp32 > 0
    assert passed.hardware.peak_memory_bandwidth_gb_s > 0
    # Caller's config object must not be mutated — substitution returns a
    # new ACTSConfig via ``dataclasses.replace``.
    assert caller_config.hardware.peak_flops_fp32 == 0


@pytest.mark.asyncio
async def test_placeholder_substitution_warns_on_dram_mismatch(caplog):
    """When the placeholder substitution fires but the detected GPU
    obviously doesn't match the placeholder's DRAM (e.g. running on H100
    where ``placeholder-RTX6000Ada`` would silently route SOLAR to the
    Ada YAML), log a warning so the user sees the mismatch instead of
    discovering it via wrong sol_score numbers."""
    baseline = Kernel(spec=_spec(), source_code="src")
    fake_orch = MagicMock()
    fake_orch.run = AsyncMock(return_value=MagicMock())

    # Detected: H100 (80 GiB) — peaks=0 because per-precision tables aren't
    # populated by ``detect_hardware()``, so substitution will fire.
    detected_h100 = HardwareSpec(
        name="NVIDIA H100 PCIe",
        freq_GHz=1.98,
        SRAM_capacity=52_428_800,
        DRAM_capacity=85_899_345_920,
    )

    with (
        patch("src.pipeline.optimize._load_model_if_configured", return_value=None),
        patch("src.config.detect_hardware", return_value=detected_h100),
        patch("src.search.orchestrator.Orchestrator", return_value=fake_orch),
        patch("src.memory.store.MemoryStore", return_value=MagicMock()),
        patch("src.kernels.starters.matmul.make_matmul_kernel", return_value=baseline),
        caplog.at_level("WARNING", logger="src.pipeline.optimize"),
    ):
        await optimize("placeholder")

    assert any("DRAM capacity mismatch" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_placeholder_substitution_silent_when_no_gpu_detected(caplog):
    """When ``detect_hardware()`` returns a fully-zeroed spec (no GPU at
    all), the placeholder substitution should NOT log a mismatch warning
    — there's nothing to compare against, and warning here would noise
    every CPU-only smoke run."""
    baseline = Kernel(spec=_spec(), source_code="src")
    fake_orch = MagicMock()
    fake_orch.run = AsyncMock(return_value=MagicMock())

    with (
        patch("src.pipeline.optimize._load_model_if_configured", return_value=None),
        patch("src.config.detect_hardware", return_value=HardwareSpec()),
        patch("src.search.orchestrator.Orchestrator", return_value=fake_orch),
        patch("src.memory.store.MemoryStore", return_value=MagicMock()),
        patch("src.kernels.starters.matmul.make_matmul_kernel", return_value=baseline),
        caplog.at_level("WARNING", logger="src.pipeline.optimize"),
    ):
        await optimize("placeholder")

    assert not any("DRAM capacity mismatch" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_populated_hardware_spec_from_caller_preserved():
    """When the caller supplies a populated HardwareSpec via ``config``, the
    placeholder substitution must NOT fire — caller's spec wins."""
    baseline = Kernel(spec=_spec(), source_code="src")
    fake_orch = MagicMock()
    fake_orch.run = AsyncMock(return_value=MagicMock())
    # Non-zero but obviously synthetic peaks so we can detect pass-through.
    custom_hw = HardwareSpec(
        name="CustomTest",
        freq_GHz=1.0,
        DRAM_byte_per_cycle=100.0,
        MAC_per_cycle_fp32_sm=50.0,
    )
    custom_config = ACTSConfig(hardware=custom_hw)

    with (
        patch("src.pipeline.optimize._load_model_if_configured", return_value=None),
        patch("src.search.orchestrator.Orchestrator", return_value=fake_orch) as mock_orch_cls,
        patch("src.memory.store.MemoryStore", return_value=MagicMock()),
        patch("src.kernels.starters.matmul.make_matmul_kernel", return_value=baseline),
    ):
        await optimize("placeholder", config=custom_config)

    passed = mock_orch_cls.call_args.kwargs["config"]
    assert passed.hardware.name == "CustomTest"
    assert passed.hardware.DRAM_byte_per_cycle == 100.0


# ── CLI argument parsing (T2) ─────────────────────────────────────────


def test_main_defaults_to_placeholder_when_no_arg(tmp_path, monkeypatch):
    """`python -m src.pipeline.optimize` with no args must keep the historical
    placeholder smoke-path so existing CI / docs invocations don't break."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    async def fake_optimize(problem_path, config=None):
        captured["problem_path"] = problem_path
        return MagicMock(), MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main([])

    assert captured["problem_path"] == "placeholder"


def test_main_forwards_problem_path_to_optimize(tmp_path, monkeypatch):
    """Positional argument selects which SOL-ExecBench problem to run.
    Forwarded verbatim — the optimize() coroutine handles directory vs literal."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    async def fake_optimize(problem_path, config=None):
        captured["problem_path"] = problem_path
        return MagicMock(), MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main(["repo/benchmark/SOL-ExecBench/examples/triton/rmsnorm"])

    assert captured["problem_path"] == "repo/benchmark/SOL-ExecBench/examples/triton/rmsnorm"


# ── run directory ─────────────────────────────────────────────────────


def test_main_creates_run_dir(tmp_path, monkeypatch):
    """--run-dir creates a timestamped sub-directory with events.jsonl,
    run.log, and traces/."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)

    async def fake_optimize(problem_path, config=None):
        return MagicMock(), MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", False),
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main(["placeholder", "--run-dir", str(tmp_path / "runs")])

    run_dirs = list((tmp_path / "runs").glob("run_*"))
    assert len(run_dirs) == 1, run_dirs
    rd = run_dirs[0]
    assert (rd / "events.jsonl").exists()
    assert (rd / "run.log").exists()
    assert (rd / "traces").is_dir()


def test_main_trace_dir_defaults_under_run_dir(tmp_path, monkeypatch):
    """When --trace-dir is omitted, SDK trace capture targets
    <run-dir>/traces/ rather than the old ``./traces/`` default."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)
    fake_processor = MagicMock()
    fake_processor.path = tmp_path / "some" / "trace.jsonl"

    async def fake_optimize(problem_path, config=None):
        return MagicMock(), MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", True),
        patch(
            "src.agents.trace_processor.enable_local_trace_capture",
            return_value=fake_processor,
        ) as mock_enable,
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main(["placeholder", "--run-dir", str(tmp_path / "runs")])

    # enable_local_trace_capture called with <run-dir>/traces/, not ./traces/
    mock_enable.assert_called_once()
    (called_path,) = mock_enable.call_args.args
    rd = next((tmp_path / "runs").glob("run_*"))
    assert called_path == rd / "traces"


def test_main_emits_run_start_and_run_end(tmp_path, monkeypatch):
    """main() emits exactly one run_start at entry and one run_end at exit,
    bracketing the full pipeline in events.jsonl.

    ``run_end`` reads the real ``SearchResult`` field names
    (``total_iterations``, ``best_node.score.sol_score``) and the
    ``TerminationReason`` enum's ``.value``. Using ``MagicMock()`` here
    would silently mask attribute-name bugs — a previous getattr-with-
    defaults variant would emit ``best_score=0.0, total_iters=0`` even
    after real iterations. Use a SimpleNamespace stand-in instead."""
    import json
    from types import SimpleNamespace

    from src.pipeline import optimize as opt_mod
    from src.search.orchestrator import TerminationReason

    monkeypatch.chdir(tmp_path)

    # SearchResult-shaped stand-in: exercises the field-name paths that
    # the real dataclass uses so the event extraction is actually tested.
    fake_score = SimpleNamespace(sol_score=0.73)
    fake_best_node = SimpleNamespace(score=fake_score)
    fake_result = SimpleNamespace(
        best_node=fake_best_node,
        total_iterations=5,
        termination_reason=TerminationReason.BUDGET,
        tree=MagicMock(),
    )

    async def fake_optimize(problem_path, config=None):
        return fake_result, MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", False),
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main(["placeholder", "--run-dir", str(tmp_path / "runs")])

    rd = next((tmp_path / "runs").glob("run_*"))
    lines = [json.loads(line) for line in (rd / "events.jsonl").read_text().splitlines() if line.strip()]
    kinds = [e["kind"] for e in lines]
    assert kinds.count("run_start") == 1
    assert kinds.count("run_end") == 1
    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_end"
    start = lines[0]
    assert start["problem_path"] == "placeholder"
    assert "model_configured" in start

    end = lines[-1]
    # termination_reason uses the enum's .value, not str(enum)
    assert end["termination_reason"] == "budget", end
    assert end["best_score"] == 0.73
    assert end["total_iterations"] == 5
    # Regression guards against the previous schema drift:
    assert "total_iters" not in end
    assert "best_iter" not in end


def test_main_emits_run_end_on_exception(tmp_path, monkeypatch):
    """If optimize() raises, run_end still fires with termination_reason=ERROR
    so post-mortems can distinguish normal exit from crashes."""
    import json
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)

    async def raising_optimize(problem_path, config=None):
        raise RuntimeError("boom")

    with (
        patch.object(opt_mod, "optimize", side_effect=raising_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", False),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            opt_mod.main(["placeholder", "--run-dir", str(tmp_path / "runs")])

    rd = next((tmp_path / "runs").glob("run_*"))
    lines = [json.loads(line) for line in (rd / "events.jsonl").read_text().splitlines() if line.strip()]
    kinds = [e["kind"] for e in lines]
    assert "run_end" in kinds
    end = next(e for e in lines if e["kind"] == "run_end")
    assert end["termination_reason"] == "ERROR"


def test_main_explicit_trace_dir_override(tmp_path, monkeypatch):
    """--trace-dir <path> still honors the explicit path (escape hatch for
    users who want traces outside the run-dir)."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)
    external = tmp_path / "external_traces"
    fake_processor = MagicMock()
    fake_processor.path = external / "trace.jsonl"

    async def fake_optimize(problem_path, config=None):
        return MagicMock(), MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", True),
        patch(
            "src.agents.trace_processor.enable_local_trace_capture",
            return_value=fake_processor,
        ) as mock_enable,
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main([
            "placeholder",
            "--run-dir", str(tmp_path / "runs"),
            "--trace-dir", str(external),
        ])

    # Explicit override wins — RunContext must not have also registered a
    # second processor under <run-dir>/traces/.
    mock_enable.assert_called_once_with(external)


# ── trace capture wiring ──────────────────────────────────────────────


def test_main_enables_trace_capture_when_sdk_available(tmp_path, monkeypatch):
    """Explicit ``--trace-dir <path>`` fires ``enable_local_trace_capture``
    when the SDK is present and shuts the processor down after the run."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)
    fake_processor = MagicMock()
    fake_processor.path = tmp_path / "trace.jsonl"

    async def fake_optimize(problem_path, config=None):
        return MagicMock(), MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", True),
        patch(
            "src.agents.trace_processor.enable_local_trace_capture",
            return_value=fake_processor,
        ) as mock_enable,
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main(["placeholder", "--trace-dir", str(tmp_path)])

    mock_enable.assert_called_once_with(tmp_path)
    fake_processor.shutdown.assert_called_once()


def test_main_skips_trace_capture_when_sdk_absent(tmp_path, monkeypatch):
    """Tier 1 venv has no SDK — capture must silently no-op rather than
    crash the placeholder smoke path."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)

    async def fake_optimize(problem_path, config=None):
        return MagicMock(), MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", False),
        patch(
            "src.agents.trace_processor.enable_local_trace_capture",
        ) as mock_enable,
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main([])

    mock_enable.assert_not_called()


def test_main_skips_trace_capture_when_disabled_explicitly(tmp_path, monkeypatch):
    """``--trace-dir=`` (empty string) is the user-facing kill-switch."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)

    async def fake_optimize(problem_path, config=None):
        return MagicMock(), MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", True),
        patch(
            "src.agents.trace_processor.enable_local_trace_capture",
        ) as mock_enable,
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main(["--trace-dir="])

    mock_enable.assert_not_called()


def test_main_completes_run_even_if_trace_setup_raises(tmp_path, monkeypatch):
    """Trace capture is best-effort diagnostics — a setup failure must not
    abort the actual optimization run."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)

    async def fake_optimize(problem_path, config=None):
        return MagicMock(), MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", True),
        patch(
            "src.agents.trace_processor.enable_local_trace_capture",
            side_effect=RuntimeError("trace dir not writable"),
        ),
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        # Must not raise.
        opt_mod.main(["placeholder", "--trace-dir", str(tmp_path)])


@pytest.mark.asyncio
async def test_placeholder_mode_never_loads_model():
    """Placeholder baseline is a stub. If a model config exists on disk and we
    load it, the model-backed Coder will raise ImplementationError on the first
    iteration (no oracle to bind). Gate model loading behind SOL-ExecBench mode
    so the default CLI smoke path stays runnable."""
    baseline = Kernel(spec=_spec(), source_code="src")
    fake_orch = MagicMock()
    fake_orch.run = AsyncMock(return_value=MagicMock())

    with (
        patch("src.pipeline.optimize._load_model_if_configured") as mock_load_model,
        patch("src.search.orchestrator.Orchestrator", return_value=fake_orch),
        patch("src.memory.store.MemoryStore", return_value=MagicMock()),
        patch("src.kernels.starters.matmul.make_matmul_kernel", return_value=baseline),
    ):
        await optimize("placeholder")

    mock_load_model.assert_not_called()


# ── SOL integration: import-order, clock-lock, adapter dispatch ───────


def test_import_order_contract_sol_first():
    """``import sol_execbench`` must be the first non-stdlib import in
    ``pipeline/optimize.py``. SOL's ``core.bench.reward_hack`` snapshots
    ``torch.cuda.Event.elapsed_time`` at module load — any user-supplied
    torch import landing first would let the snapshot capture a tampered
    address.

    We verify this by reading the source: the first non-stdlib ``import``
    line (after the docstring) must reference ``sol_execbench``.
    """
    from src.pipeline import optimize as opt_mod

    src = Path(opt_mod.__file__).read_text()
    # Use the AST so docstrings, multi-line strings, and conditional
    # blocks can't trip the parser. Walk top-level body in order; first
    # ``Import`` / ``ImportFrom`` whose top-level package is not stdlib
    # must be sol_execbench.
    import ast
    import sys

    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    tree = ast.parse(src)
    found_third_party = None
    for node in tree.body:
        if isinstance(node, ast.Import):
            mod = node.names[0].name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or node.module is None:
                continue
            mod = node.module.split(".")[0]
        else:
            continue
        if mod in stdlib:
            continue
        found_third_party = mod
        break
    assert found_third_party == "sol_execbench", (
        f"first non-stdlib import must be sol_execbench, got {found_third_party!r}"
    )


@pytest.mark.asyncio
async def test_load_problem_dispatches_to_sol_when_definition_present(tmp_path):
    """When ``definition.json`` exists in the problem dir, the dispatcher
    routes to the SOL adapter without needing ``benchmark_adapter`` set."""
    from src.pipeline.optimize import _load_problem

    (tmp_path / "definition.json").write_text("{}")  # presence-only check
    captured: dict = {}

    async def fake_sol_loader(problem_dir, config, coder):
        captured["called"] = problem_dir
        return ("sentinel-tuple",)

    with patch(
        "src.pipeline.optimize._load_sol_problem", side_effect=fake_sol_loader,
    ):
        result = await _load_problem(tmp_path, ACTSConfig(), MagicMock())

    assert result == ("sentinel-tuple",)
    assert captured["called"] == tmp_path


@pytest.mark.asyncio
async def test_load_problem_raises_unknown_format_when_no_markers(tmp_path):
    """Empty directory → no definition.json, no model.py → raises
    ``UnknownBenchmarkFormat`` with a useful message."""
    from src.pipeline.optimize import UnknownBenchmarkFormat, _load_problem

    with pytest.raises(UnknownBenchmarkFormat, match="Cannot determine"):
        await _load_problem(tmp_path, ACTSConfig(), MagicMock())


@pytest.mark.asyncio
async def test_load_problem_raises_on_unknown_adapter_value():
    """Explicit override with a typo → ``UnknownBenchmarkFormat``."""
    from src.pipeline.optimize import UnknownBenchmarkFormat, _load_problem

    config = ACTSConfig(benchmark_adapter="kerneblench-typo")
    with pytest.raises(UnknownBenchmarkFormat, match="Unknown benchmark_adapter"):
        await _load_problem(Path("/fake"), config, MagicMock())


@pytest.mark.asyncio
async def test_load_problem_kernelbench_not_implemented():
    """Setting ``benchmark_adapter='kernelbench'`` raises NotImplementedError."""
    from src.pipeline.optimize import _load_problem

    config = ACTSConfig(benchmark_adapter="kernelbench")
    with pytest.raises(NotImplementedError):
        await _load_problem(Path("/fake"), config, MagicMock())


def test_main_emits_clock_lock_unavailable_when_probe_fails(tmp_path, monkeypatch):
    """When ``probe_clock_lock_available()`` returns False (no sudo or
    unsupported GPU), ``main()`` must log a warning and emit a
    ``clock_lock_unavailable`` event so post-run analysis sees the lack
    of timing isolation."""
    import json

    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)

    async def fake_optimize(problem_path, config=None):
        return MagicMock(), MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", False),
        patch.object(opt_mod, "probe_clock_lock_available", return_value=False),
        patch.object(opt_mod, "_lock_gpu0_clocks") as mock_lock,
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main(["placeholder", "--run-dir", str(tmp_path / "runs")])

    # _lock_gpu0_clocks must NOT be called when the probe fails.
    mock_lock.assert_not_called()
    rd = next((tmp_path / "runs").glob("run_*"))
    lines = [
        json.loads(line)
        for line in (rd / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kinds = [e["kind"] for e in lines]
    assert "clock_lock_unavailable" in kinds, kinds


def test_clock_lock_unavailable_event_includes_reason_when_probe_returns_bare_false():
    """Regression: SOL's ``probe_clock_lock_available()`` returns a bare
    ``bool`` today, not a ``(bool, str)`` tuple. When it returns ``False``,
    the ``clock_lock_unavailable`` event must still carry a synthesized
    ``reason`` field so post-run analysis sees *why* clock locking was
    skipped — earlier code dropped the reason and emitted only ``device``.
    """
    from src.pipeline import optimize as opt_mod

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 1
    fake_torch.cuda.get_device_name.return_value = "FakeGPU"

    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch.object(opt_mod, "probe_clock_lock_available", return_value=False),
        patch("src.runtime.events.emit") as mock_emit,
    ):
        opt_mod._try_acquire_clock_lock()

    # Find the clock_lock_unavailable emit call.
    matching = [
        c for c in mock_emit.call_args_list
        if c.args and c.args[0] == "clock_lock_unavailable"
    ]
    assert len(matching) == 1, mock_emit.call_args_list
    kwargs = matching[0].kwargs
    assert kwargs.get("device") == "FakeGPU"
    assert kwargs.get("reason") == "probe_returned_false", kwargs


def test_clock_lock_unavailable_event_handles_tuple_probe_return():
    """Defensive: if a future SOL release lands a ``(bool, str)`` tuple
    return shape for ``probe_clock_lock_available()`` (its docstring hints
    at this), ACTS must unpack it and forward the SOL-supplied reason
    string into the event payload — not crash on tuple-vs-bool unpacking
    and not synthesize a generic placeholder."""
    from src.pipeline import optimize as opt_mod

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 1
    fake_torch.cuda.get_device_name.return_value = "FakeGPU"

    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch.object(
            opt_mod,
            "probe_clock_lock_available",
            return_value=(False, "custom reason"),
        ),
        patch("src.runtime.events.emit") as mock_emit,
    ):
        opt_mod._try_acquire_clock_lock()

    matching = [
        c for c in mock_emit.call_args_list
        if c.args and c.args[0] == "clock_lock_unavailable"
    ]
    assert len(matching) == 1, mock_emit.call_args_list
    assert matching[0].kwargs.get("reason") == "custom reason"


def test_try_acquire_clock_lock_rolls_back_on_verify_false():
    """Regression: when ``_lock_gpu0_clocks`` succeeds but ``verify_clocks``
    reports drift (``False``), the lock attempt must be treated as a
    *failure* — roll back the partial pin via ``_unlock_gpu0_clocks``,
    emit ``clock_lock_unavailable`` with ``reason='verify_failed'``, and
    leave ``_clock_lock_state['locked']`` as ``False``. The earlier
    behavior swallowed the drift warning and still flipped
    ``locked=True``, which let plateau/scoring logic proceed as if
    clocks were pinned even though verification proved otherwise.
    """
    from src.pipeline import optimize as opt_mod

    # Reset module-global state so prior tests don't leak in.
    opt_mod._clock_lock_state["locked"] = False
    opt_mod._clock_lock_state["device_name"] = ""

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 1
    fake_torch.cuda.get_device_name.return_value = "FakeGPU"
    fake_preset = MagicMock(gpu_clk_mhz=2505, dram_clk_mhz=10001)

    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch.object(opt_mod, "probe_clock_lock_available", return_value=True),
        patch.object(opt_mod, "get_clock_preset", return_value=fake_preset),
        patch.object(opt_mod, "_lock_gpu0_clocks", return_value=True),
        patch.object(opt_mod, "_verify_gpu0_locked", return_value=False),
        patch.object(opt_mod, "_unlock_gpu0_clocks") as mock_unlock,
        patch("src.runtime.events.emit") as mock_emit,
    ):
        opt_mod._try_acquire_clock_lock()

    # Rollback fired exactly once.
    assert mock_unlock.call_count == 1
    # Event emitted with the verify_failed reason.
    matching = [
        c for c in mock_emit.call_args_list
        if c.args and c.args[0] == "clock_lock_unavailable"
    ]
    assert len(matching) == 1, mock_emit.call_args_list
    kwargs = matching[0].kwargs
    assert kwargs.get("device") == "FakeGPU"
    assert kwargs.get("reason") == "verify_failed", kwargs
    # locked-state must remain False.
    assert opt_mod._clock_lock_state["locked"] is False


def test_try_acquire_clock_lock_rolls_back_on_verify_exception():
    """Regression: when ``verify_clocks`` *raises*, ACTS must treat that
    the same as drift — roll back the partial pin, emit
    ``clock_lock_unavailable`` with a ``reason`` that starts with
    ``verify_raised:`` and carries a snippet of the exception message,
    and leave ``_clock_lock_state['locked']`` False. The earlier
    behavior logged a warning and then unconditionally flipped
    ``locked=True``, hiding a real driver-level disagreement from
    downstream consumers.
    """
    from src.pipeline import optimize as opt_mod

    # Reset module-global state so prior tests don't leak in.
    opt_mod._clock_lock_state["locked"] = False
    opt_mod._clock_lock_state["device_name"] = ""

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 1
    fake_torch.cuda.get_device_name.return_value = "FakeGPU"
    fake_preset = MagicMock(gpu_clk_mhz=2505, dram_clk_mhz=10001)

    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch.object(opt_mod, "probe_clock_lock_available", return_value=True),
        patch.object(opt_mod, "get_clock_preset", return_value=fake_preset),
        patch.object(opt_mod, "_lock_gpu0_clocks", return_value=True),
        patch.object(
            opt_mod, "_verify_gpu0_locked",
            side_effect=RuntimeError("driver disagrees"),
        ),
        patch.object(opt_mod, "_unlock_gpu0_clocks") as mock_unlock,
        patch("src.runtime.events.emit") as mock_emit,
    ):
        opt_mod._try_acquire_clock_lock()

    # Rollback fired exactly once.
    assert mock_unlock.call_count == 1
    # Event emitted with a verify_raised:* reason that carries exception text.
    matching = [
        c for c in mock_emit.call_args_list
        if c.args and c.args[0] == "clock_lock_unavailable"
    ]
    assert len(matching) == 1, mock_emit.call_args_list
    kwargs = matching[0].kwargs
    assert kwargs.get("device") == "FakeGPU"
    reason = kwargs.get("reason", "")
    assert reason.startswith("verify_raised:"), reason
    assert "driver disagrees" in reason, reason
    # locked-state must remain False.
    assert opt_mod._clock_lock_state["locked"] is False


def test_main_unlocks_clocks_on_normal_exit(tmp_path, monkeypatch):
    """On normal exit, ``main()`` must call ``_unlock_gpu0_clocks`` at
    least once via the ``finally`` path. The atexit registration is a
    safety net for abnormal exits; the explicit call ensures unlock
    happens before the interpreter teardown."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)

    async def fake_optimize(problem_path, config=None):
        return MagicMock(), MagicMock()

    fake_preset = MagicMock(gpu_clk_mhz=2505, dram_clk_mhz=10001)

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", False),
        patch.object(opt_mod, "probe_clock_lock_available", return_value=True),
        patch.object(opt_mod, "get_clock_preset", return_value=fake_preset),
        patch.object(opt_mod, "_lock_gpu0_clocks", return_value=True),
        patch.object(opt_mod, "_verify_gpu0_locked", return_value=True),
        patch.object(opt_mod, "_unlock_gpu0_clocks") as mock_unlock,
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main(["placeholder", "--run-dir", str(tmp_path / "runs")])

    # First call from the explicit ``finally`` block; the atexit-registered
    # call sees ``locked=False`` (idempotent flag) and no-ops.
    assert mock_unlock.call_count >= 1


def test_main_unlocks_clocks_when_optimize_raises(tmp_path, monkeypatch):
    """If ``optimize()`` raises, the clock-lock cleanup must still fire so
    the GPU isn't left pinned across runs."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)

    async def raising_optimize(problem_path, config=None):
        raise RuntimeError("boom")

    fake_preset = MagicMock(gpu_clk_mhz=2505, dram_clk_mhz=10001)

    with (
        patch.object(opt_mod, "optimize", side_effect=raising_optimize),
        patch("src.agents.llm_backend._SDK_AVAILABLE", False),
        patch.object(opt_mod, "probe_clock_lock_available", return_value=True),
        patch.object(opt_mod, "get_clock_preset", return_value=fake_preset),
        patch.object(opt_mod, "_lock_gpu0_clocks", return_value=True),
        patch.object(opt_mod, "_verify_gpu0_locked", return_value=True),
        patch.object(opt_mod, "_unlock_gpu0_clocks") as mock_unlock,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            opt_mod.main(["placeholder", "--run-dir", str(tmp_path / "runs")])

    assert mock_unlock.call_count >= 1


def test_unlock_clocks_safe_is_idempotent():
    """``_unlock_clocks_safe`` swallows exceptions and clears the flag so
    a second call is a no-op even if the underlying unlock raises on the
    first call."""
    from src.pipeline import optimize as opt_mod

    opt_mod._clock_lock_state["locked"] = True
    opt_mod._clock_lock_state["device_name"] = "TestGPU"
    with patch.object(
        opt_mod, "_unlock_gpu0_clocks", side_effect=RuntimeError("nope"),
    ):
        # First call: unlock raises but we swallow it; flag is cleared.
        opt_mod._unlock_clocks_safe()
    assert opt_mod._clock_lock_state["locked"] is False
    # Second call: flag cleared, no-op.
    with patch.object(opt_mod, "_unlock_gpu0_clocks") as mock_unlock:
        opt_mod._unlock_clocks_safe()
        mock_unlock.assert_not_called()


# ── GPU-0-scoped clock lock helpers ────────────────────────────────────


def test_lock_gpu0_clocks_issues_two_scoped_subprocess_calls():
    """``_lock_gpu0_clocks`` must invoke ``nvidia-smi`` twice — once for
    -lgc (graphics clock) and once for -lmc (memory clock) — and both
    invocations must carry ``-i 0`` so the lock applies to GPU 0 only,
    not to every GPU on the host."""
    from src.pipeline import optimize as opt_mod

    with patch.object(opt_mod, "subprocess") as mock_sp:
        # Default MagicMock return is a successful CompletedProcess-like.
        mock_sp.CalledProcessError = __import__("subprocess").CalledProcessError
        ok = opt_mod._lock_gpu0_clocks(2505, 10001)

    assert ok is True
    assert mock_sp.run.call_count == 2
    first_args = mock_sp.run.call_args_list[0].args[0]
    second_args = mock_sp.run.call_args_list[1].args[0]
    assert first_args == [
        "sudo", "-n", "nvidia-smi", "-lgc", "2505,2505", "-i", "0",
    ]
    assert second_args == [
        "sudo", "-n", "nvidia-smi", "-lmc", "10001,10001", "-i", "0",
    ]


def test_unlock_gpu0_clocks_issues_two_scoped_subprocess_calls():
    """``_unlock_gpu0_clocks`` must invoke ``nvidia-smi`` twice — once for
    -rgc and once for -rmc — both scoped to ``-i 0`` so we never reset
    application clocks on GPUs ACTS isn't using."""
    from src.pipeline import optimize as opt_mod

    with patch.object(opt_mod, "subprocess") as mock_sp:
        opt_mod._unlock_gpu0_clocks()

    assert mock_sp.run.call_count == 2
    first_args = mock_sp.run.call_args_list[0].args[0]
    second_args = mock_sp.run.call_args_list[1].args[0]
    assert first_args == ["sudo", "-n", "nvidia-smi", "-rgc", "-i", "0"]
    assert second_args == ["sudo", "-n", "nvidia-smi", "-rmc", "-i", "0"]


def test_lock_gpu0_clocks_rolls_back_on_dram_failure():
    """If the DRAM lock (``-lmc``) fails after the GPU clock lock
    (``-lgc``) succeeded, ``_lock_gpu0_clocks`` must (a) return False
    and (b) issue a ``-rgc -i 0`` call to roll the GPU lock back —
    otherwise we'd leave a half-locked GPU."""
    import subprocess as real_subprocess

    from src.pipeline import optimize as opt_mod

    def run_side_effect(cmd, **_kwargs):
        # First call: -lgc succeeds. Second: -lmc fails. Third: -rgc rollback.
        if "-lmc" in cmd:
            raise real_subprocess.CalledProcessError(
                returncode=1, cmd=cmd, stderr=b"locked",
            )
        return MagicMock()  # CompletedProcess stand-in

    with patch.object(opt_mod, "subprocess") as mock_sp:
        mock_sp.CalledProcessError = real_subprocess.CalledProcessError
        mock_sp.run.side_effect = run_side_effect
        ok = opt_mod._lock_gpu0_clocks(2505, 10001)

    assert ok is False
    # Three calls: lgc (ok), lmc (raise), rgc rollback.
    assert mock_sp.run.call_count == 3
    rollback_cmd = mock_sp.run.call_args_list[2].args[0]
    assert rollback_cmd == ["sudo", "-n", "nvidia-smi", "-rgc", "-i", "0"]


# ── ACTS-side verify wrapper ──────────────────────────────────────────


def test_verify_gpu0_locked_returns_true_when_clocks_match():
    """``_verify_gpu0_locked`` must return True when nvidia-smi reports
    current graphics + memory clocks within tolerance of the expected
    values. The wake-op (a tiny torch kernel launch) is mocked away —
    we're testing the parse + compare logic, not torch."""
    from src.pipeline import optimize as opt_mod

    fake_torch = MagicMock()
    fake_completed = MagicMock()
    fake_completed.stdout = "2505, 10001\n"

    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch.object(opt_mod.subprocess, "run", return_value=fake_completed) as mock_run,
    ):
        ok = opt_mod._verify_gpu0_locked(2505, 10001)

    assert ok is True
    # The nvidia-smi query must be scoped to GPU 0 (mirroring the
    # GPU-0-only lock) — that's the H2 fix.
    cmd = mock_run.call_args.args[0]
    assert "-i" in cmd and cmd[cmd.index("-i") + 1] == "0", cmd
    assert "nvidia-smi" in cmd
    assert "--query-gpu=clocks.current.graphics,clocks.current.memory" in cmd


def test_verify_gpu0_locked_returns_false_when_clocks_drift_beyond_tolerance():
    """When nvidia-smi reports a current clock outside the 50-MHz
    tolerance band, ``_verify_gpu0_locked`` must return False so the
    caller rolls back the partial lock. This is the H1 false-negative
    case we used to silently accept (idle GPU at 210 MHz vs locked
    target 2505 MHz)."""
    from src.pipeline import optimize as opt_mod

    fake_torch = MagicMock()
    fake_completed = MagicMock()
    # Idle clock: 210 MHz graphics, target was 2505 — well outside tolerance.
    fake_completed.stdout = "210, 10001\n"

    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch.object(opt_mod.subprocess, "run", return_value=fake_completed),
    ):
        ok = opt_mod._verify_gpu0_locked(2505, 10001)

    assert ok is False


def test_verify_gpu0_locked_returns_false_on_subprocess_error():
    """If nvidia-smi fails (CalledProcessError, FileNotFoundError),
    ``_verify_gpu0_locked`` must return False rather than raising —
    callers treat False as drift and roll back the partial lock, which
    is the correct conservative outcome when we can't read the clocks."""
    import subprocess as real_subprocess

    from src.pipeline import optimize as opt_mod

    fake_torch = MagicMock()

    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch.object(
            opt_mod.subprocess, "run",
            side_effect=real_subprocess.CalledProcessError(returncode=1, cmd="x"),
        ),
    ):
        ok = opt_mod._verify_gpu0_locked(2505, 10001)

    assert ok is False


def test_signal_unlock_handler_unlocks_then_propagates():
    """The SIGTERM/SIGHUP handler must (1) call ``_unlock_clocks_safe``,
    (2) restore the default disposition for the signal, and (3) re-raise
    the signal via ``os.kill`` so the process dies with the conventional
    signal-class exit code (128 + signum) rather than via ``sys.exit``."""
    import signal as real_signal

    from src.pipeline import optimize as opt_mod

    with (
        patch.object(opt_mod, "_unlock_clocks_safe") as mock_unlock,
        patch.object(opt_mod.signal, "signal") as mock_sig,
        patch.object(opt_mod.os, "kill") as mock_kill,
        patch.object(opt_mod.os, "getpid", return_value=12345),
    ):
        opt_mod._signal_unlock_handler(real_signal.SIGTERM, None)

    mock_unlock.assert_called_once()
    # Default disposition must be restored for the specific signal.
    mock_sig.assert_called_once_with(real_signal.SIGTERM, real_signal.SIG_DFL)
    # Re-raise via os.kill against this pid.
    mock_kill.assert_called_once_with(12345, real_signal.SIGTERM)


# ── ACTS-first clock preset resolution ────────────────────────────────


def test_resolve_clock_preset_acts_table_hits_for_rtx_6000_ada():
    """ACTS's table covers workstation + Pro cards SOL omits. RTX 6000
    Ada must resolve to the design-boost values (2505 / 10001) so
    clock-lock activates on the dev host instead of falling through to
    ``no_preset``."""
    from sol_execbench.core.bench.config.device_config import ClockPreset

    from src.pipeline.optimize import _resolve_clock_preset

    result = _resolve_clock_preset("NVIDIA RTX 6000 Ada Generation")
    assert result == ClockPreset(gpu_clk_mhz=2505, dram_clk_mhz=10001)


def test_resolve_clock_preset_falls_through_to_sol_for_h100():
    """When the ACTS table misses, SOL's ``get_clock_preset`` is
    consulted. H100 is in SOL's table → returns the SOL preset
    (gpu_clk_mhz=1410); we don't assert the exact dram value to keep
    the test loosely coupled to SOL's internal numbers."""
    from src.pipeline.optimize import _resolve_clock_preset

    result = _resolve_clock_preset("NVIDIA H100 PCIe")
    assert result is not None
    assert result.gpu_clk_mhz == 1410


def test_resolve_clock_preset_returns_none_for_unknown_device():
    """Both ACTS and SOL tables miss → None propagates so the caller
    emits ``clock_lock_unavailable`` with ``reason='no_preset'`` and
    continues with unlocked clocks."""
    from src.pipeline.optimize import _resolve_clock_preset

    assert _resolve_clock_preset("Made-Up GPU Model") is None


def test_resolve_clock_preset_acts_table_takes_precedence():
    """Defensive: ACTS-first ordering means a colliding key in ACTS's
    table shadows SOL's value. No current entry collides, but the
    contract is worth pinning so future overrides behave predictably."""
    from sol_execbench.core.bench.config.device_config import ClockPreset

    from src.pipeline import optimize as opt_mod

    fake_table = {
        "NVIDIA H100": ClockPreset(gpu_clk_mhz=9999, dram_clk_mhz=8888),
    }
    with patch.object(opt_mod, "_ACTS_CLOCK_PRESETS", fake_table):
        result = opt_mod._resolve_clock_preset("NVIDIA H100 PCIe")

    assert result is not None
    assert result.gpu_clk_mhz == 9999, (
        "ACTS table entry must shadow SOL's H100 preset (1410)"
    )


def test_main_reset_clocks_short_circuits_pipeline(tmp_path, monkeypatch, capsys):
    """``--reset-clocks`` is the operator escape hatch for the SIGKILL /
    segfault case: it must reset GPU 0 clocks and exit immediately,
    without creating a RunContext, loading a model, or invoking
    ``asyncio.run(optimize(...))``."""
    from src.pipeline import optimize as opt_mod

    monkeypatch.chdir(tmp_path)

    with (
        patch.object(opt_mod, "subprocess") as mock_sp,
        patch("src.runtime.run_context.RunContext.create") as mock_rc_create,
        patch.object(opt_mod.asyncio, "run") as mock_async_run,
    ):
        opt_mod.main(["--reset-clocks"])

    # Two subprocess calls: -rgc + -rmc, both scoped to GPU 0.
    assert mock_sp.run.call_count == 2
    rgc_cmd = mock_sp.run.call_args_list[0].args[0]
    rmc_cmd = mock_sp.run.call_args_list[1].args[0]
    assert rgc_cmd == ["sudo", "-n", "nvidia-smi", "-rgc", "-i", "0"]
    assert rmc_cmd == ["sudo", "-n", "nvidia-smi", "-rmc", "-i", "0"]
    # No pipeline side effects.
    mock_rc_create.assert_not_called()
    mock_async_run.assert_not_called()
    # Confirmation line on stdout.
    captured = capsys.readouterr()
    assert "GPU 0 clocks reset." in captured.out
