"""Inner process that NCU profiles.

Executed as ``python -m src.eval._profiler_driver <spec_json_path>`` by
``_run_ncu``. NCU wraps this invocation; the driver's only job is to
import the compiled kernel, build inputs, warm up once, synchronize, and
launch the kernel a single time (the measured launch NCU captures).

The spec JSON shape (produced by ``_run_ncu``):

.. code-block:: json

    {
      "kernel_source_path": "<abs path to compiled .py>",
      "entrypoint": "kernel_fn",
      "workload": {"uuid": "...", "axes": {...}, "inputs": {...}},
      "problem_dir": "<abs path to the SOL problem directory>",
      "seed": 0
    }

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


def _build_inputs(problem_dir: Path, workload_dict: dict, seed: int) -> tuple:
    """Rebuild the input generator for the profiled workload and draw one batch.

    ``problem_dir`` is the directory containing ``definition.json`` and
    ``workload.jsonl`` — ``load_problem`` expects a directory, not a file.

    Torch + sol_execbench are imported lazily here so Tier 1 tests can
    import the driver without the GPU stack installed.
    """
    # Lazy imports — not available in the Tier 1 test venv.
    from src.benchmark.problem_loader import load_problem
    from src.benchmark.problem import Workload
    from src.eval.inputs import build_input_generator

    problem = load_problem(Path(problem_dir))
    workload = Workload(
        uuid=workload_dict.get("uuid", "profile-0"),
        axes=workload_dict.get("axes", {}),
        inputs=workload_dict.get("inputs", {}),
    )
    generator = build_input_generator(problem, workload)
    return generator(seed)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m src.eval._profiler_driver <spec_json>", file=sys.stderr)
        return 2

    spec = json.loads(Path(argv[1]).read_text())
    entrypoint = spec["entrypoint"]
    module = _load_module(Path(spec["kernel_source_path"]))
    kernel_fn = _resolve_host_callable(module, entrypoint)

    seed = spec.get("seed", 0)
    if "problem_dir" in spec:
        inputs = _build_inputs(
            Path(spec["problem_dir"]),
            spec.get("workload", {}),
            seed,
        )
    elif callable(getattr(module, "make_inputs", None)):
        # Self-contained kernel convention: the source defines its own
        # ``make_inputs(seed) -> tuple`` so the driver can rebuild inputs
        # without the parent process's (unpicklable) closure.
        inputs = tuple(module.make_inputs(seed))
    else:
        inputs = tuple(spec.get("args", ()))

    # Warmup launch — establishes caches and JIT-compiles the kernel so
    # the measured launch is steady-state.
    kernel_fn(*inputs)
    _synchronize()
    # Measured launch — this is the one NCU profiles.
    kernel_fn(*inputs)
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
