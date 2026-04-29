"""Per-benchmark adapter modules.

Each subpackage exposes `load(path: Path) -> tuple[Definition, list[Workload]]`
that converts the benchmark's native on-disk format into SOL pydantic types.

This is the ONLY place benchmark-format knowledge is allowed to live —
see PRD "Benchmark-agnosticism guarantee" for the architectural contract.
"""
