"""Custom-format adapter — unimplemented placeholder namespace.

No ``load`` callable is exported yet; importing this module is a no-op.
For one-off problems written in SOL shape (definition.json +
workload.jsonl in a directory), use ``src.benchmarks.sol_execbench.load(path)``
directly. This namespace exists for discoverability; future custom
formats (non-SOL-shaped) plug in here.
"""
