"""Reference-function and input-generator helpers for correctness verification.

Bridges the gap between a SOL ``Definition`` + ``Workload`` and the pair
of callables consumed by ``verify_correctness``:

- ``reference_fn(*args) -> output`` — the PyTorch oracle from definition.json.
- ``input_generator(seed) -> args`` — fresh input tuple for a trial.

``build_reference_fn`` is pure-Python (it just execs the reference
source into a namespace), so the module imports cleanly in torch-less
test venvs. ``build_input_generator`` requires ``torch`` +
``sol_execbench`` — those imports happen at call time (not at module
import). The SOL pydantic models flow through directly, no shim needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload


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
    definition: Definition,
    workload: Workload,
    *,
    device: str = "cuda",
    blob_roots: list[Path] | None = None,
) -> Callable[[int], tuple]:
    """Build an input generator backed by ``sol_execbench.core.bench.io.gen_inputs``.

    Reseeds the global torch/python/cuda RNG before each generation so
    trials at different seeds produce distinct inputs. The returned
    callable yields a tuple of positional args suitable for both the
    reference and the candidate.

    Requires ``torch`` and ``sol_execbench`` installed — lazy-imported
    so this module stays importable in torch-less environments. SOL
    pydantic types flow through unchanged (no dict shimming needed).

    *blob_roots* is forwarded to ``load_safetensors`` when the workload
    declares any ``SafetensorsInput``: blobs are resolved against these
    roots in order, with the first existing match winning. The blobs are
    loaded once at build time (not per ``_generator(seed)`` call) so the
    on-disk read does not enter the per-iteration timing path.
    """
    from sol_execbench.core.bench.correctness import set_seed
    from sol_execbench.core.bench.io import gen_inputs
    from sol_execbench.core.data.workload import SafetensorsInput

    safe_tensors: dict | None = None
    if any(isinstance(v, SafetensorsInput) for v in workload.inputs.values()):
        from sol_execbench.core.bench.io import load_safetensors

        safe_tensors = load_safetensors(
            definition, workload, blob_roots=blob_roots or []
        )

    def _generator(seed: int) -> tuple:
        set_seed(seed)
        return tuple(
            gen_inputs(definition, workload, device=device, safe_tensors=safe_tensors)
        )

    return _generator


def allocate_dps_outputs(
    definition: Definition,
    workload: Workload,
    *,
    device: str = "cuda",
) -> list:
    """Pre-allocate DPS output buffers for ``kernel_fn(*inputs, *outputs)`` calls.

    Resolves the workload's axes against the definition once, then delegates to
    ``sol_execbench.core.bench.io.allocate_outputs``. Single source of truth for
    the DPS allocation shape — used by the correctness gate, the benchmark loop,
    and the NCU profiler subprocess.
    """
    from sol_execbench.core.bench.io import allocate_outputs as _allocate_outputs

    resolved_axes = definition.get_resolved_axes_values(workload.axes)
    return _allocate_outputs(definition, resolved_axes, device=device)
