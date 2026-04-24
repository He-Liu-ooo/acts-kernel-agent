"""Triton baseline generation from a PyTorch reference.

The Coder produces a one-shot PyTorch→Triton port. The result becomes the
root of the search tree and the anchor for T_b in the SOL-score formula
(S = 0.5 at baseline). Each attempt goes through ``CoderAgent.translate``
(tool-loop over compile + correctness bound to every selected workload)
and a post-verify pass that re-runs correctness on every workload — the
post-verify catches SDK best-effort output when the turn budget was
exhausted. Raises ``BaselineGenerationError`` on no-model or retry
exhaustion; there is no stub fallback because search against a fake
baseline would silently report progress.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.coder import CoderAgent, ImplementationError
from src.eval.correctness import verify_correctness
from src.eval.inputs import build_input_generator, build_reference_fn
from src.kernels.compiler import compile_kernel
from src.kernels.kernel import Kernel
from src.runtime.events import emit

if TYPE_CHECKING:
    from pathlib import Path

    from src.benchmark.problem import Problem, Workload
    from src.eval.correctness import ComparisonPolicy
    from src.kernels.kernel import KernelSpec


class BaselineGenerationError(Exception):
    """Raised when a verified Triton baseline cannot be produced."""


async def generate_triton_baseline(
    problem: Problem,
    spec: KernelSpec,
    *,
    coder: CoderAgent | None,
    workloads: list[Workload],
    max_retries: int = 3,
    cache_dir: Path | None = None,
    policy: ComparisonPolicy | None = None,
) -> Kernel:
    """Translate a PyTorch reference into a verified Triton baseline.

    Returns the first candidate that compiles and passes correctness on
    every workload in *workloads*. Raises ``BaselineGenerationError``
    when no model is configured or when the attempt budget is exhausted.
    """
    if coder is None or not coder.has_model:
        raise BaselineGenerationError(
            f"No model configured for '{problem.name}' — set ACTS_MODEL_CONFIG "
            "or drop configs/models/<provider>.json in place.",
        )

    if not workloads:
        raise ValueError(
            "generate_triton_baseline requires at least one workload.",
        )

    reference_fn = build_reference_fn(problem.reference_source)
    input_generators = [build_input_generator(problem, w) for w in workloads]

    for attempt in range(max_retries):
        emit("baseline_attempt", attempt=attempt + 1, max_attempts=max_retries)
        try:
            output = await coder.translate(
                reference_source=problem.reference_source,
                kernel_spec=spec,
                reference_fn=reference_fn,
                input_generators=input_generators,
            )
        except ImplementationError as exc:
            emit(
                "baseline_failure",
                attempt=attempt + 1,
                reason=f"ImplementationError: {str(exc)[:200]}",
            )
            continue

        candidate = Kernel(
            spec=spec,
            source_code=output.source_code,
            triton_kernel_name=output.triton_kernel_name,
        )
        compiled = compile_kernel(candidate, cache_dir=cache_dir)
        if not compiled.success:
            emit(
                "baseline_failure",
                attempt=attempt + 1,
                reason=f"CompileError: {str(compiled.error_message or '')[:200]}",
            )
            continue

        if all(
            verify_correctness(
                candidate_fn=compiled.compiled_fn,
                reference_fn=reference_fn,
                input_generator=gen,
                policy=policy,
            ).passed
            for gen in input_generators
        ):
            emit(
                "baseline_success",
                source_bytes=len(output.source_code),
                triton_kernel_name=output.triton_kernel_name or "",
            )
            return candidate

        emit(
            "baseline_failure",
            attempt=attempt + 1,
            reason="CorrectnessError: post-verify failed on one or more workloads",
        )

    raise BaselineGenerationError(
        f"Baseline translation for '{problem.name}' failed after "
        f"{max_retries} attempts.",
    )
