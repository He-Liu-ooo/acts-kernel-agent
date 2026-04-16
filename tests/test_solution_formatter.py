"""Tests for benchmark/solution_formatter.py — SOL-ExecBench solution output."""

from src.benchmark.problem import AxisDef, Problem, TensorDef
from src.benchmark.solution_formatter import format_solution


def _make_problem() -> Problem:
    return Problem(
        name="rmsnorm_h4096",
        axes={"batch_size": AxisDef(type="var"), "hidden_size": AxisDef(type="const", value=4096)},
        inputs={"x": TensorDef(shape=["batch_size", "hidden_size"], dtype="bfloat16")},
        outputs={"y": TensorDef(shape=["batch_size", "hidden_size"], dtype="bfloat16")},
        reference_source="def run(x): return x",
        op_type="rmsnorm",
    )


def test_format_solution_schema():
    sol = format_solution(_make_problem(), triton_source="# triton kernel")
    assert sol["definition"] == "rmsnorm_h4096"
    assert sol["spec"]["languages"] == ["triton"]
    assert sol["spec"]["destination_passing_style"] is True
    assert len(sol["sources"]) == 1
    assert sol["sources"][0]["content"] == "# triton kernel"


def test_format_solution_custom_name():
    sol = format_solution(_make_problem(), triton_source="# k", name="my_sol_v2")
    assert sol["name"] == "my_sol_v2"
