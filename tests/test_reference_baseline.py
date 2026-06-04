from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmark.reference_baseline import (
    ReferenceBaselineError,
    ReferenceBaselineResult,
    load_reference_callable,
    measure_reference_baseline,
)
from src.eval.benchmark import BenchmarkError
from src.eval.correctness import CorrectnessGateFailure
from src.kernels.kernel import KernelType

FIX = Path(__file__).parent / "fixtures"


def test_load_reference_callable_resolves_entrypoint():
    _kernel, fn = load_reference_callable(
        path=FIX / "reference_ok.py",
        entrypoint="run",
        kernel_type=KernelType.ELEMENTWISE,
    )
    assert fn(1) == 2


def test_load_reference_missing_file_raises():
    with pytest.raises(ReferenceBaselineError, match="load"):
        load_reference_callable(
            path=FIX / "nope.py",
            entrypoint="run",
            kernel_type=KernelType.ELEMENTWISE,
        )


def test_load_reference_bad_import_raises():
    with pytest.raises(ReferenceBaselineError, match="load"):
        load_reference_callable(
            path=FIX / "reference_badimport.py",
            entrypoint="run",
            kernel_type=KernelType.ELEMENTWISE,
        )


def _fake_workload(uuid="w0"):
    wl = MagicMock()
    wl.uuid = uuid
    return wl


@patch("src.benchmark.reference_baseline.benchmark_kernel")
@patch("src.benchmark.reference_baseline.run_correctness_gate")
@patch("src.benchmark.reference_baseline.load_reference_callable")
def test_measure_success_returns_median(mock_load, mock_gate, mock_bench, tmp_path):
    ref_py = tmp_path / "ref.py"
    ref_py.write_text("def run(x):\n    return x\n")
    mock_load.return_value = (MagicMock(), lambda *a: None)
    mock_gate.return_value = None
    mock_bench.return_value = MagicMock(
        median_latency_us=42.0,
        per_workload_latency_us={"w0": 42.0},
        is_fully_successful=True,
    )
    res = measure_reference_baseline(
        definition=MagicMock(),
        path=str(ref_py),
        entrypoint="run",
        kernel_type=MagicMock(),
        workloads=[_fake_workload()],
        input_generators=[lambda s: ()],
        reference_fn=lambda *a: None,
        config=MagicMock(),
    )
    assert isinstance(res, ReferenceBaselineResult)
    assert res.median_latency_us == 42.0
    # benchmark called with autotuner=None and the loaded callable as kernel_fn
    assert mock_bench.call_args.kwargs["autotuner"] is None
    assert mock_bench.call_args.kwargs["kernel_fn"] is mock_load.return_value[1]


@patch("src.benchmark.reference_baseline.benchmark_kernel")
@patch("src.benchmark.reference_baseline.run_correctness_gate")
@patch("src.benchmark.reference_baseline.load_reference_callable")
def test_measure_correctness_fail_hard_fails(
    mock_load, mock_gate, mock_bench, tmp_path
):
    ref_py = tmp_path / "ref.py"
    ref_py.write_text("def run(x):\n    return x\n")
    wl = _fake_workload()
    mock_load.return_value = (MagicMock(), lambda *a: None)
    mock_gate.return_value = CorrectnessGateFailure(
        index=0,
        workload=wl,
        result=MagicMock(passed=False, error_message="LSE mismatch"),
    )
    with pytest.raises(ReferenceBaselineError, match="correctness"):
        measure_reference_baseline(
            definition=MagicMock(),
            path=str(ref_py),
            entrypoint="run",
            kernel_type=MagicMock(),
            workloads=[_fake_workload()],
            input_generators=[lambda s: ()],
            reference_fn=lambda *a: None,
            config=MagicMock(),
        )
    mock_bench.assert_not_called()


@patch("src.benchmark.reference_baseline.benchmark_kernel")
@patch("src.benchmark.reference_baseline.load_reference_callable")
def test_measure_empty_workloads_validates_before_any_call(mock_load, mock_bench):
    with pytest.raises(ReferenceBaselineError, match="validate"):
        measure_reference_baseline(
            definition=MagicMock(),
            path="ref.py",
            entrypoint="run",
            kernel_type=MagicMock(),
            workloads=[],
            input_generators=[],
            reference_fn=lambda *a: None,
            config=MagicMock(),
        )
    mock_load.assert_not_called()
    mock_bench.assert_not_called()


def test_measure_length_mismatch_validates():
    with pytest.raises(ReferenceBaselineError, match="validate"):
        measure_reference_baseline(
            definition=MagicMock(),
            path="ref.py",
            entrypoint="run",
            kernel_type=MagicMock(),
            workloads=[_fake_workload("w0"), _fake_workload("w1")],
            input_generators=[lambda s: ()],
            reference_fn=lambda *a: None,
            config=MagicMock(),
        )


def test_measure_none_reference_fn_validates():
    with pytest.raises(ReferenceBaselineError, match="validate"):
        measure_reference_baseline(
            definition=MagicMock(),
            path="ref.py",
            entrypoint="run",
            kernel_type=MagicMock(),
            workloads=[_fake_workload()],
            input_generators=[lambda s: ()],
            reference_fn=None,
            config=MagicMock(),
        )


@patch("src.benchmark.reference_baseline.benchmark_kernel")
@patch("src.benchmark.reference_baseline.run_correctness_gate")
@patch("src.benchmark.reference_baseline.load_reference_callable")
def test_measure_benchmark_error_wrapped(mock_load, mock_gate, mock_bench, tmp_path):
    ref_py = tmp_path / "ref.py"
    ref_py.write_text("def run(x):\n    return x\n")
    mock_load.return_value = (MagicMock(), lambda *a: None)
    mock_gate.return_value = None
    mock_bench.side_effect = BenchmarkError("boom")
    with pytest.raises(ReferenceBaselineError, match=r"\[benchmark\]"):
        measure_reference_baseline(
            definition=MagicMock(),
            path=str(ref_py),
            entrypoint="run",
            kernel_type=MagicMock(),
            workloads=[_fake_workload()],
            input_generators=[lambda s: ()],
            reference_fn=lambda *a: None,
            config=MagicMock(),
        )


@patch("src.benchmark.reference_baseline.benchmark_kernel")
@patch("src.benchmark.reference_baseline.run_correctness_gate")
@patch("src.benchmark.reference_baseline.load_reference_callable")
def test_measure_correctness_crash_wrapped(mock_load, mock_gate, mock_bench, tmp_path):
    ref_py = tmp_path / "ref.py"
    ref_py.write_text("def run(x):\n    return x\n")
    wl = _fake_workload()
    mock_load.return_value = (MagicMock(), lambda *a: None)
    mock_gate.return_value = CorrectnessGateFailure(
        index=0,
        workload=wl,
        exception=RuntimeError("CUDA device-side assert"),
    )
    with pytest.raises(ReferenceBaselineError, match=r"\[correctness\]"):
        measure_reference_baseline(
            definition=MagicMock(),
            path=str(ref_py),
            entrypoint="run",
            kernel_type=MagicMock(),
            workloads=[_fake_workload()],
            input_generators=[lambda s: ()],
            reference_fn=lambda *a: None,
            config=MagicMock(),
        )
    mock_bench.assert_not_called()
