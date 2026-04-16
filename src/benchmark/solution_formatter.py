"""Format ACTS Triton output as a SOL-ExecBench solution.json."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.benchmark.problem import Problem


def format_solution(
    problem: Problem,
    triton_source: str,
    entry_point: str = "kernel.py::run",
    author: str = "acts-agent",
    name: str | None = None,
) -> dict:
    """Build a SOL-ExecBench solution dict from an ACTS Triton kernel.

    The returned dict conforms to the SOL-ExecBench solution schema and
    can be written to ``solution.json`` for evaluation by the harness.
    """
    solution_name = name or f"{problem.name}_triton_acts"
    return {
        "name": solution_name,
        "definition": problem.name,
        "author": author,
        "description": "Triton kernel optimised by ACTS",
        "spec": {
            "languages": ["triton"],
            "target_hardware": ["LOCAL"],
            "entry_point": entry_point,
            "dependencies": ["torch", "triton"],
            "destination_passing_style": True,
        },
        "sources": [
            {
                "path": entry_point.split("::")[0],
                "content": triton_source,
            },
        ],
    }
