"""SOL-ExecBench problem loader.

Parses the on-disk SOL format (definition.json + workload.jsonl) into SOL
pydantic types. The output flows unchanged through the rest of the ACTS
pipeline — this is the ONLY place benchmark-format knowledge is allowed
to live.
"""
from __future__ import annotations

from pathlib import Path

from sol_execbench.core.data import Definition, Workload
from sol_execbench.core.data.json_utils import load_json_file, load_jsonl_file


def load(path: Path) -> tuple[Definition, list[Workload]]:
    """Load a SOL-ExecBench problem directory.

    Args:
        path: Directory containing definition.json + workload.jsonl.

    Returns:
        (definition, workloads) — pydantic-validated SOL types.

    Raises:
        FileNotFoundError: definition.json or workload.jsonl missing.
        pydantic.ValidationError: malformed JSON or schema mismatch.
            (Upstream's `load_json_file` uses `model_validate_json`, which
            wraps both syntactic and semantic errors as `ValidationError`.)
    """
    definition_path = path / "definition.json"
    workload_path = path / "workload.jsonl"
    definition = load_json_file(Definition, definition_path)
    workloads = load_jsonl_file(Workload, workload_path)
    return definition, workloads
