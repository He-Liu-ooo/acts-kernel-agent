"""Reference-function and input-generator helpers for correctness verification.

Bridges the gap between a SOL-ExecBench ``Problem`` and the pair of
callables consumed by ``verify_correctness``:

- ``reference_fn(*args) -> output`` — the PyTorch oracle from definition.json.
- ``input_generator(seed) -> args`` — fresh input tuple for a trial.

``build_reference_fn`` is pure-Python (it just execs the reference
source into a namespace), so the module imports cleanly in torch-less
test venvs. ``build_input_generator`` requires ``torch`` +
``sol_execbench`` — those imports happen at call time (not at module
import), and the SOL pydantic models are validated once so per-seed
calls stay tight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from src.benchmark.problem import Problem, Workload


class ReferenceLoadError(RuntimeError):
    """Raised when the PyTorch reference source cannot be turned into a callable."""


def build_reference_fn(
    source: str,
    entrypoint: str = "run",
) -> Callable[..., Any]:
    """Exec a PyTorch reference source and return its entrypoint callable.

    The source string comes from ``definition.json``'s ``reference`` field
    (or, equivalently, from ``reference.py`` in the SOL-ExecBench layout).
    It is expected to define ``def run(*args): ...`` at module scope; the
    returned callable is the correctness oracle used by the 5-stage gate.

    Raises ``ReferenceLoadError`` when the entrypoint symbol is missing
    or non-callable. ``SyntaxError`` / ``ImportError`` from the source
    propagate directly so the caller sees the real cause.
    """
    namespace: dict[str, Any] = {"__name__": "__acts_reference__"}
    exec(compile(source, "<acts-reference>", "exec"), namespace)

    fn = namespace.get(entrypoint)
    if fn is None:
        raise ReferenceLoadError(
            f"Reference entrypoint '{entrypoint}' not found in definition source. "
            f"The PyTorch reference must define `def {entrypoint}(...):` at module scope."
        )
    if not callable(fn):
        raise ReferenceLoadError(
            f"Reference entrypoint '{entrypoint}' is not callable (got {type(fn).__name__})."
        )
    return fn


def build_input_generator(
    problem: Problem,
    workload: Workload,
    *,
    device: str = "cuda",
) -> Callable[[int], tuple]:
    """Build an input generator backed by ``sol_execbench.core.bench.io.gen_inputs``.

    Reseeds the global torch/python/cuda RNG before each generation so
    trials at different seeds produce distinct inputs. The returned
    callable yields a tuple of positional args suitable for both the
    reference and the candidate.

    Requires ``torch`` and ``sol_execbench`` installed — lazy-imported
    so this module stays importable in torch-less environments. SOL's
    pydantic models are validated once at build time so per-seed calls
    only pay the RNG reset + input generation.
    """
    from sol_execbench.core.bench.correctness import set_seed
    from sol_execbench.core.bench.io import gen_inputs
    from sol_execbench.core.data.definition import Definition
    from sol_execbench.core.data.workload import Workload as SOLWorkload

    sol_def = Definition.model_validate(_problem_to_sol_dict(problem))
    sol_wkl = SOLWorkload.model_validate(_workload_to_sol_dict(workload))

    def _generator(seed: int) -> tuple:
        set_seed(seed)
        return tuple(gen_inputs(sol_def, sol_wkl, device=device))

    return _generator


def _problem_to_sol_dict(problem: Problem) -> dict:
    """Convert an ACTS ``Problem`` dataclass to a dict compatible with SOL's Definition."""
    axes: dict[str, dict] = {}
    for name, axis in problem.axes.items():
        entry: dict[str, Any] = {"type": axis.type, "description": axis.description or None}
        if axis.value is not None:
            entry["value"] = axis.value
        if axis.expression is not None:
            entry["expression"] = axis.expression
        axes[name] = entry

    def _tensor_spec(defn) -> dict:
        return {
            "shape": defn.shape,
            "dtype": defn.dtype,
            "description": defn.description or None,
        }

    return {
        "name": problem.name,
        "op_type": problem.op_type,
        "axes": axes,
        "inputs": {k: _tensor_spec(v) for k, v in problem.inputs.items()},
        "outputs": {k: _tensor_spec(v) for k, v in problem.outputs.items()},
        "reference": problem.reference_source,
        "description": problem.description or None,
        "custom_inputs_entrypoint": problem.custom_inputs_entrypoint,
    }


def _workload_to_sol_dict(workload: Workload) -> dict:
    """Convert an ACTS ``Workload`` dataclass to a dict compatible with SOL's Workload."""
    return {
        "uuid": workload.uuid,
        "axes": workload.axes,
        "inputs": workload.inputs,
    }
