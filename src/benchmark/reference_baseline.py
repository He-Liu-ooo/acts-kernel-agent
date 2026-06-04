"""External reference baseline measurement (Option C).

Loads a non-Triton reference implementation (e.g. a flashinfer wrapper),
verifies it against the PyTorch reference via the 5-stage correctness gate,
and times it black-box through ``benchmark_kernel``. Its median latency
becomes the SOL-score T_b. Any failure raises ``ReferenceBaselineError``,
which the pipeline lets abort the run — a wrong scoring baseline silently
corrupts every score. See doc/specs/2026-06-03-external-reference-baseline-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.eval.benchmark import benchmark_kernel
from src.eval.correctness import run_correctness_gate
from src.kernels.compiler import compile_kernel
from src.kernels.kernel import Kernel, KernelSpec, KernelType

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload

    from src.config import ACTSConfig
    from src.eval.correctness import ComparisonPolicy


class ReferenceBaselineError(Exception):
    """Raised when the external reference baseline cannot be measured."""


@dataclass
class ReferenceBaselineResult:
    median_latency_us: float
    per_workload_latency_us: dict[str, float]


def _build_reference_kernel(
    path: Path, entrypoint: str, kernel_type: KernelType
) -> Kernel:
    source = path.read_text()
    spec = KernelSpec(
        name="reference_baseline",
        kernel_type=kernel_type,
        flop_count=0,  # unused — reference is never profiled/roofline-scored
        memory_bytes=0,
        entrypoint=entrypoint,
    )
    return Kernel(spec=spec, source_code=source, triton_kernel_name="")


def load_reference_callable(
    *, path: Path, entrypoint: str, kernel_type: KernelType
) -> tuple[Kernel, Callable[..., Any]]:
    """Resolve the reference entrypoint, returning its Kernel and compiled callable.

    Hard-fails on any load error.
    """
    if not Path(path).exists():
        raise ReferenceBaselineError(
            f"[load] reference_baseline_path does not exist: {path}"
        )
    kernel = _build_reference_kernel(Path(path), entrypoint, kernel_type)
    result = compile_kernel(kernel)
    if not result.success or result.compiled_fn is None:
        raise ReferenceBaselineError(
            f"[load] failed to load reference at {path} (entrypoint={entrypoint!r}): "
            f"{result.error_message}"
        )
    return kernel, result.compiled_fn


def measure_reference_baseline(
    definition: "Definition",
    *,
    path: str,
    entrypoint: str,
    kernel_type: KernelType,
    workloads: list["Workload"],
    input_generators: list[Callable[[int], tuple]],
    reference_fn: Callable[..., Any],
    config: "ACTSConfig",
    policy: "ComparisonPolicy | None" = None,
    cache_dir: Path | None = None,
) -> ReferenceBaselineResult:
    """Load → correctness-gate (all workloads) → benchmark. Hard-fail on any miss."""
    # Fail-closed validation: an empty/mismatched workload list would skip the
    # correctness loop entirely and hit benchmark_kernel's 100us sentinel,
    # fabricating a scoring baseline. Reject before any load/bench call.
    if definition is None or reference_fn is None:
        raise ReferenceBaselineError(
            "[validate] reference baseline requires a SOL definition and a "
            "PyTorch reference_fn."
        )
    if not workloads:
        raise ReferenceBaselineError(
            "[validate] reference baseline requires at least one workload."
        )
    if len(workloads) != len(input_generators):
        raise ReferenceBaselineError(
            f"[validate] workloads ({len(workloads)}) != input_generators "
            f"({len(input_generators)})."
        )

    kernel, fn = load_reference_callable(
        path=Path(path), entrypoint=entrypoint, kernel_type=kernel_type
    )

    # Gate: 5-stage correctness vs the PyTorch reference on every workload.
    failure = run_correctness_gate(
        fn, reference_fn, input_generators, workloads,
        definition=definition, kernel=kernel, policy=policy,
    )
    if failure is not None:
        wl_tag = f"wl {failure.index + 1}/{len(workloads)} (uuid={failure.workload.uuid})"
        if failure.exception is not None:
            raise ReferenceBaselineError(
                f"[correctness] reference raised on {wl_tag}: {failure.exception}"
            ) from failure.exception
        raise ReferenceBaselineError(
            f"[correctness] reference failed {wl_tag}: {failure.result.error_message}"
        )

    try:
        bench = benchmark_kernel(
            kernel,
            config,
            workloads=workloads,
            input_generators=input_generators,
            kernel_fn=fn,
            definition=definition,
            autotuner=None,
        )
    except Exception as exc:  # BenchmarkError (>half workloads failed) etc.
        raise ReferenceBaselineError(
            f"[benchmark] reference benchmark failed: {exc}"
        ) from exc
    if not bench.is_fully_successful:
        raise ReferenceBaselineError(
            f"[benchmark] reference bench had partial-workload failures: "
            f"{getattr(bench, 'workload_errors', None)}"
        )
    return ReferenceBaselineResult(
        median_latency_us=bench.median_latency_us,
        per_workload_latency_us=dict(bench.per_workload_latency_us),
    )
