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
from src.pipeline.optimize import _load_sol_execbench, optimize


def _spec() -> KernelSpec:
    return KernelSpec(
        name="t",
        kernel_type=KernelType.ELEMENTWISE,
        entrypoint="kernel_fn",
        pytorch_reference="def run(x):\n    return x * 2.0\n",
    )


@pytest.mark.asyncio
async def test_load_sol_execbench_returns_reference_fn_and_all_generators():
    """Phase A must return a reference_fn and one generator per selected workload
    so Phase B's correctness tool binds to the full coverage set. Collapsing to
    just workloads[0] lets kernels that pass workload 1 but break 2..N slip through."""
    from src.benchmark.problem import Problem, Workload

    problem = Problem(
        name="p", axes={}, inputs={}, outputs={},
        reference_source="def run(x): return x * 2.0\n",
        op_type="elementwise",
    )
    workloads = [Workload(uuid=f"wl-{i}", axes={}, inputs={}) for i in range(3)]
    spec = _spec()
    baseline = Kernel(spec=spec, source_code="src")

    ref_fn = lambda x: x * 2.0
    gens = [lambda seed, i=i: (i, seed) for i in range(3)]

    with (
        patch("src.benchmark.problem_loader.load_problem", return_value=problem),
        patch("src.benchmark.problem_loader.problem_to_kernel_spec", return_value=spec),
        patch("src.benchmark.workload_selector.select_workloads", return_value=workloads),
        patch("src.eval.roofline.derive_t_sol_from_solar", return_value=None),
        patch(
            "src.benchmark.baseline_generator.generate_triton_baseline",
            new_callable=AsyncMock,
            return_value=baseline,
        ),
        patch("src.eval.inputs.build_reference_fn", return_value=ref_fn),
        patch("src.eval.inputs.build_input_generator", side_effect=gens),
    ):
        result = await _load_sol_execbench(Path("/fake"), ACTSConfig(), MagicMock())

    # Expect 6-tuple: (baseline, problem, workloads, roofline, reference_fn, input_generators)
    assert len(result) == 6
    _baseline, _problem, _workloads, _roofline, got_ref, got_gens = result
    assert got_ref is ref_fn
    assert got_gens == gens  # all three, in workload order


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
    ):
        await optimize("placeholder")

    kwargs = fake_orch.run.call_args.kwargs
    assert "reference_fn" in kwargs
    assert "input_generators" in kwargs
    assert kwargs["reference_fn"] is None
    assert kwargs["input_generators"] == []

    # SOL-ExecBench path: reference_fn and the full generator list come back
    # from _load_sol_execbench and reach Orchestrator.run as kwargs.
    fake_orch.run.reset_mock()
    with (
        patch("src.pipeline.optimize._load_model_if_configured", return_value=None),
        patch("src.search.orchestrator.Orchestrator", return_value=fake_orch),
        patch("src.memory.store.MemoryStore", return_value=store_instance),
        patch(
            "src.pipeline.optimize._load_sol_execbench",
            new_callable=AsyncMock,
            return_value=(baseline, MagicMock(), [MagicMock()] * 3, None, ref_fn, gens),
        ),
        patch("pathlib.Path.is_dir", return_value=True),
        patch.object(Path, "exists", autospec=True, return_value=True),
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


def test_main_defaults_to_placeholder_when_no_arg():
    """`python -m src.pipeline.optimize` with no args must keep the historical
    placeholder smoke-path so existing CI / docs invocations don't break."""
    from src.pipeline import optimize as opt_mod

    captured: dict = {}

    async def fake_optimize(problem_path, config=None):
        captured["problem_path"] = problem_path
        return MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main([])

    assert captured["problem_path"] == "placeholder"


def test_main_forwards_problem_path_to_optimize():
    """Positional argument selects which SOL-ExecBench problem to run.
    Forwarded verbatim — the optimize() coroutine handles directory vs literal."""
    from src.pipeline import optimize as opt_mod

    captured: dict = {}

    async def fake_optimize(problem_path, config=None):
        captured["problem_path"] = problem_path
        return MagicMock()

    with (
        patch.object(opt_mod, "optimize", side_effect=fake_optimize),
        patch("src.pipeline.report.generate_report", return_value=MagicMock()),
        patch("src.pipeline.report.render_report", return_value=""),
    ):
        opt_mod.main(["repo/benchmark/SOL-ExecBench/examples/triton/rmsnorm"])

    assert captured["problem_path"] == "repo/benchmark/SOL-ExecBench/examples/triton/rmsnorm"


# ── trace capture wiring ──────────────────────────────────────────────


def test_main_enables_trace_capture_when_sdk_available(tmp_path):
    """Default ``--trace-dir traces`` should fire ``enable_local_trace_capture``
    when the SDK is present and shutdown the processor after the run."""
    from src.pipeline import optimize as opt_mod

    fake_processor = MagicMock()
    fake_processor.path = tmp_path / "trace.jsonl"

    async def fake_optimize(problem_path, config=None):
        return MagicMock()

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


def test_main_skips_trace_capture_when_sdk_absent():
    """Tier 1 venv has no SDK — capture must silently no-op rather than
    crash the placeholder smoke path."""
    from src.pipeline import optimize as opt_mod

    async def fake_optimize(problem_path, config=None):
        return MagicMock()

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


def test_main_skips_trace_capture_when_disabled_explicitly():
    """``--trace-dir=`` (empty string) is the user-facing kill-switch."""
    from src.pipeline import optimize as opt_mod

    async def fake_optimize(problem_path, config=None):
        return MagicMock()

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


def test_main_completes_run_even_if_trace_setup_raises(tmp_path):
    """Trace capture is best-effort diagnostics — a setup failure must not
    abort the actual optimization run."""
    from src.pipeline import optimize as opt_mod

    async def fake_optimize(problem_path, config=None):
        return MagicMock()

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
