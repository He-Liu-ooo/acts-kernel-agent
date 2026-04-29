"""Inner process that NCU profiles.

Executed as ``python -m src.eval._profiler_driver <spec_json_path>`` by
``_run_ncu``. NCU wraps this invocation; the driver's only job is to
import the compiled kernel, build inputs, warm up once, synchronize, and
launch the kernel a single time (the measured launch NCU captures).

Spec JSON contract (mirror of the docstring on ``_run_ncu``):

.. code-block:: json

    {
      "kernel_source_path": "<abs path to compiled .py>",
      "entrypoint": "kernel_fn",
      "workload": {"uuid": "...", "axes": {...}, "inputs": {...}},
      "mode": "curated",
      "problem_dir": "<abs path to the SOL problem directory>",
      "blob_roots": ["<abs dir 1>", "<abs dir 2>"],
      "dps": false,
      "seed": 0
    }

* ``blob_roots`` (optional, list[str]): rehydrated to ``list[Path]`` and
  forwarded as ``blob_roots=`` into ``build_input_generator`` so workloads
  with ``SafetensorsInput`` resolve real on-disk weights inside the
  subprocess. Absent → ``None``; safe for non-safetensors workloads.
* ``dps`` (optional, bool, default False): when True the host wrapper
  takes pre-allocated output buffers as positional args after the inputs;
  the driver allocates them via ``src.eval.inputs.allocate_dps_outputs``
  and calls ``kernel_fn(*inputs, *outputs)``. Mirrors the wiring in
  ``src/eval/benchmark.py::_wrap_dps_generator`` and
  ``src/eval/correctness.py::_maybe_wrap_dps_candidate``.

Input resolution priority:

1. ``problem_dir`` present → build via
   ``src.eval.inputs.build_input_generator`` (orchestrator path). The
   directory must contain ``definition.json`` + ``workload.jsonl``; the
   driver calls ``load_problem`` on it.
2. ``module.make_inputs(seed)`` defined in the kernel source → call it
   (self-contained kernel convention — primary Tier 2 path).
3. ``spec["args"]`` present → use as positional args (ad-hoc smoke tests).
4. Otherwise ``()`` — only safe when ``run()`` takes no arguments.

The driver is intentionally minimal — anything that can happen in the
parent process should happen there so this is the shortest possible
path between ``ncu`` and the kernel launch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(path: Path):
    """Import ``path`` as a private module object. Used by both
    ``_load_callable`` and the ``module.make_inputs`` lookup so a single
    ``exec_module`` call serves both."""
    module_name = f"_acts_profiler_target_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import kernel source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_host_callable(module, entrypoint: str):
    """Return the host wrapper callable from ``module``.

    Convention (aligned with ``src/eval/inputs.py`` and the Coder
    translate prompt): a compiled kernel module exposes ``def run(...)``
    as the host wrapper that launches the ``@triton.jit`` kernel with
    ``fn[grid](...)`` syntax. Prefers ``module.run``; falls back to
    ``module.<entrypoint>`` for non-Triton kernels and the historical
    shape where ``entrypoint`` itself was the callable.
    """
    fn = getattr(module, "run", None)
    if fn is None or not callable(fn):
        fn = getattr(module, entrypoint, None)
    if fn is None or not callable(fn):
        raise RuntimeError(
            f"no host callable in module: expected ``def run(...)`` "
            f"(preferred) or ``def {entrypoint}(...)``"
        )
    return fn


def _build_inputs(
    problem_dir: Path,
    workload_dict: dict,
    seed: int,
    blob_roots: list[Path] | None = None,
) -> tuple:
    """Rebuild inputs for the profiled workload and surface the parsed SOL objects.

    ``problem_dir`` is the directory containing ``definition.json`` and
    ``workload.jsonl`` — the SOL adapter's ``load`` expects a directory,
    not a file.

    ``blob_roots`` is forwarded to ``build_input_generator`` so workloads
    with ``SafetensorsInput`` resolve real on-disk weights inside the
    subprocess. ``None`` is safe for any workload without safetensors-backed
    inputs.

    Returns ``(definition, workload, inputs)``: ``definition`` and
    ``workload`` are reused by the DPS path in ``main()`` to allocate output
    buffers; ``inputs`` is the tuple produced by the generator at ``seed``.

    Torch + sol_execbench are imported lazily here so Tier 1 tests can
    import the driver without the GPU stack installed.
    """
    # Lazy imports — not available in the Tier 1 test venv.
    from sol_execbench.core.data import Workload

    from src.benchmarks.sol_execbench import load as sol_load
    from src.eval.inputs import build_input_generator

    definition, _all_workloads = sol_load(Path(problem_dir))
    # The orchestrator hands us the representative workload's pydantic
    # model_dump output; SOL's ``Workload`` validator handles the typed
    # input variants (random / scalar / safetensors / custom).
    workload = Workload.model_validate(
        {
            "uuid": workload_dict.get("uuid", "profile-0"),
            "axes": workload_dict.get("axes", {}),
            "inputs": workload_dict.get("inputs", {}),
        }
    )
    generator = build_input_generator(definition, workload, blob_roots=blob_roots)
    return definition, workload, generator(seed)


def _call_kernel(
    kernel_fn,
    *,
    inputs: tuple,
    definition,
    workload,
    dps: bool,
    device: str = "cuda",
):
    """Invoke ``kernel_fn`` honoring the kernel's DPS flag.

    Non-DPS path: ``kernel_fn(*inputs)`` — return value flows back
    unchanged. Mirrors the legacy single-return shape used in the
    benchmark loop for non-DPS kernels.

    DPS path: pre-allocate output buffers via
    ``src.eval.inputs.allocate_dps_outputs(definition, workload, device=device)``,
    then call ``kernel_fn(*inputs, *outputs)``. The kernel populates the
    buffers in place; we return the outputs tuple (the kernel's actual
    return value is conventionally None and is discarded).
    """
    if not dps:
        return kernel_fn(*inputs)

    # Lazy import — sol_execbench is not in the Tier 1 test venv.
    from src.eval.inputs import allocate_dps_outputs

    if definition is None or workload is None:
        raise ValueError(
            "_call_kernel(dps=True) requires definition and workload — "
            "allocate_dps_outputs needs the definition's output schema and the "
            "workload's axes. Spec contract bug at the call site."
        )

    outputs = allocate_dps_outputs(definition, workload, device=device)
    kernel_fn(*inputs, *outputs)
    return tuple(outputs)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m src.eval._profiler_driver <spec_json>", file=sys.stderr)
        return 2

    spec = json.loads(Path(argv[1]).read_text())
    entrypoint = spec["entrypoint"]
    module = _load_module(Path(spec["kernel_source_path"]))
    kernel_fn = _resolve_host_callable(module, entrypoint)

    seed = spec.get("seed", 0)
    dps = bool(spec.get("dps", False))
    # Rehydrate blob_roots back to ``list[Path] | None``. Absent → ``None``
    # for back-compat with cached profiler specs that predate the field.
    blob_roots: list[Path] | None
    raw_blob_roots = spec.get("blob_roots")
    if raw_blob_roots:
        blob_roots = [Path(p) for p in raw_blob_roots]
    else:
        blob_roots = None

    # ``definition`` + ``workload`` are needed only on the DPS path
    # (``allocate_dps_outputs`` needs the output schema + workload axes).
    definition = None
    workload_obj = None

    if "problem_dir" in spec:
        definition, workload_obj, inputs = _build_inputs(
            Path(spec["problem_dir"]),
            spec.get("workload", {}),
            seed,
            blob_roots=blob_roots,
        )
    elif callable(getattr(module, "make_inputs", None)):
        # Self-contained kernel convention: the source defines its own
        # ``make_inputs(seed) -> tuple`` so the driver can rebuild inputs
        # without the parent process's (unpicklable) closure.
        inputs = tuple(module.make_inputs(seed))
    else:
        inputs = tuple(spec.get("args", ()))

    # Warmup launch — establishes caches and JIT-compiles the kernel so
    # the measured launch is steady-state. ``_call_kernel`` honors the
    # DPS flag so the warmup matches the kernel's real call shape;
    # otherwise a DPS kernel would TypeError on the warmup call before
    # NCU could even record the measured launch.
    _call_kernel(
        kernel_fn,
        inputs=inputs,
        definition=definition,
        workload=workload_obj,
        dps=dps,
    )
    _synchronize()
    # Measured launch — this is the one NCU profiles.
    _call_kernel(
        kernel_fn,
        inputs=inputs,
        definition=definition,
        workload=workload_obj,
        dps=dps,
    )
    _synchronize()
    print("ok")
    return 0


def _synchronize() -> None:
    """``torch.cuda.synchronize()`` — driver only runs inside NCU with
    the GPU stack present."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
