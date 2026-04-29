"""Tests for src.benchmarks.sol_execbench.load (the SOL ExecBench
fixture loader). Despite the filename, this module does NOT cover the
SOLAR adapter — see test_solar_adapter.py for that surface."""
from pathlib import Path

import pydantic
import pytest
from sol_execbench.core.data import Definition, Workload

from src.benchmarks.sol_execbench import load

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sol_simple"


def test_load_returns_definition_and_workloads():
    definition, workloads = load(FIXTURE_DIR)
    assert isinstance(definition, Definition)
    assert all(isinstance(w, Workload) for w in workloads)


def test_loaded_definition_has_expected_fields():
    definition, _ = load(FIXTURE_DIR)
    assert definition.name == "test_elementwise_add"
    assert definition.op_type == "elementwise"
    assert "N" in definition.axes
    assert "a" in definition.inputs and "b" in definition.inputs
    assert "y" in definition.outputs


def test_loaded_workloads_match_fixture():
    _, workloads = load(FIXTURE_DIR)
    assert len(workloads) == 2
    assert workloads[0].uuid == "w1"
    assert workloads[0].axes == {"N": 1024}
    assert workloads[1].uuid == "w2"
    assert workloads[1].axes == {"N": 4096}


def test_load_raises_on_missing_definition(tmp_path):
    (tmp_path / "workload.jsonl").write_text("")
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_load_raises_on_malformed_definition(tmp_path):
    (tmp_path / "definition.json").write_text("{not valid json")
    (tmp_path / "workload.jsonl").write_text("")
    with pytest.raises(pydantic.ValidationError):
        load(tmp_path)
