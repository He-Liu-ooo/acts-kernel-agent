"""Triton kernel compilation.

Used by both the Coder's compile_kernel_tool (during its turn) and the
orchestrator (when materializing kernels for benchmarking / profiling).

Strategy (matches AutoKernel/Astra pattern):
1. Write source to a file under ``cache_dir`` so tracebacks carry real
   filenames (bare ``exec()`` loses this, which makes Coder self-correction
   much harder).
2. Load via ``importlib.util.spec_from_file_location`` + ``exec_module``.
3. Resolve ``kernel.spec.entrypoint`` via ``getattr``.

Triton's ``@triton.jit`` is lazy — specialization happens on first launch
with typed args, so shape/dtype-dependent compile errors surface later in
``eval/correctness.py``, not here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.kernels.kernel import Kernel

DEFAULT_CACHE_DIR = Path(".acts_cache/compiled")


@dataclass
class CompilationResult:
    success: bool
    error_message: str = ""
    compiled_fn: Callable | None = None
    source_path: Path | None = None
    # A1 PR 1: when the source declares an ``@triton.autotune``-wrapped
    # ``@triton.jit`` kernel, this holds the resolved Autotuner instance
    # (i.e. ``module.<kernel.triton_kernel_name>``). The orchestrator
    # passes it into ``benchmark_kernel`` so the benchmark loop can diff
    # Triton's per-config cache around each workload. ``None`` for kernels
    # without an autotune decorator or when ``triton_kernel_name`` is empty
    # (legacy starters, hand-written test fixtures).
    triton_autotuner: Callable | None = None


def compile_kernel(
    kernel: Kernel,
    cache_dir: Path | None = None,
) -> CompilationResult:
    """Compile a kernel's source and return its entrypoint callable.

    Parse-time errors (syntax, imports, missing/non-callable entrypoint)
    surface as ``success=False``. Launch-time compile errors are Triton's
    problem — they appear during correctness/benchmark runs.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    source = kernel.source_code
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    stem = f"{kernel.spec.name}_{source_hash}"
    source_path = cache_dir / f"{stem}.py"
    entrypoint = kernel.spec.entrypoint
    module_name = f"acts_compiled_{stem}"

    # Source hash pins (name, content) → (path, module_name). If the
    # module is already imported, Phase C's N re-profiles of the same
    # winning kernel reuse the exec_module result instead of re-parsing.
    cached_module = sys.modules.get(module_name)
    if cached_module is not None and source_path.exists():
        fn = getattr(cached_module, entrypoint, None)
        if callable(fn):
            return CompilationResult(
                success=True,
                compiled_fn=fn,
                source_path=source_path,
                triton_autotuner=_resolve_triton_autotuner(cached_module, kernel),
            )

    source_path.write_text(source)

    try:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            return CompilationResult(
                success=False,
                error_message=f"Could not build import spec for {source_path}",
                source_path=source_path,
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        return CompilationResult(
            success=False,
            error_message=traceback.format_exc(),
            source_path=source_path,
        )

    fn = getattr(module, entrypoint, None)
    if fn is None:
        sys.modules.pop(module_name, None)
        return CompilationResult(
            success=False,
            error_message=(
                f"Entrypoint '{entrypoint}' not found in compiled kernel. "
                f"Define `def {entrypoint}(...):` at module scope."
            ),
            source_path=source_path,
        )
    if not callable(fn):
        sys.modules.pop(module_name, None)
        return CompilationResult(
            success=False,
            error_message=(
                f"Entrypoint '{entrypoint}' is not callable "
                f"(got {type(fn).__name__})."
            ),
            source_path=source_path,
        )

    return CompilationResult(
        success=True,
        compiled_fn=fn,
        source_path=source_path,
        triton_autotuner=_resolve_triton_autotuner(module, kernel),
    )


def _resolve_triton_autotuner(module, kernel: "Kernel"):
    """Locate the ``@triton.autotune``-wrapped JIT kernel in *module*.

    Returns ``module.<kernel.triton_kernel_name>`` when that name resolves
    to an object that looks like a Triton ``Autotuner`` (carries a
    ``.cache`` attribute, even if empty). Returns ``None`` for:
      - Legacy starters / test fixtures without ``triton_kernel_name``.
      - Kernels whose declared name is a bare ``@triton.jit`` def with no
        autotune wrapper (no ``.cache`` attribute).
      - Anything that fails attribute resolution.

    Best-effort: failures degrade to ``None`` so ``compile_kernel`` stays
    success-path even when autotune introspection isn't available.
    """
    name = getattr(kernel, "triton_kernel_name", "") or ""
    if not name:
        return None
    try:
        candidate = getattr(module, name, None)
        if candidate is None:
            return None
        # Duck-type: Triton's Autotuner exposes ``.cache``; the bare
        # JITFunction does not. We only want the Autotuner.
        if not hasattr(candidate, "cache"):
            return None
        return candidate
    except Exception:
        return None
