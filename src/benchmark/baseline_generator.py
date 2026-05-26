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

from src.agents.coder import AttemptFailure, CoderAgent, ImplementationError
from src.eval.correctness import verify_correctness
from src.eval.inputs import build_input_generator, build_reference_fn
from src.eval.profiler import find_jit_name_in_entrypoint, triton_kernel_names_in
from src.kernels.compiler import compile_kernel
from src.kernels.kernel import Kernel
from src.runtime.events import emit
from src.runtime.sdk_trace import trace_span
from src.runtime.usage import AgentLabel

if TYPE_CHECKING:
    from pathlib import Path

    from sol_execbench.core.data import Definition, Workload

    from src.eval.correctness import ComparisonPolicy
    from src.kernels.kernel import KernelSpec


_POST_VERIFY_COMPILE_FAILED = "Post-verify Compile FAILED"
_POST_VERIFY_CORRECTNESS_FAILED = "Post-verify Correctness FAILED"

# Stage tags for ``load_operator_baseline`` failures. Each tag appears
# both in the BaselineGenerationError message (as ``[<stage>]``) and in
# the ``operator_baseline_failure`` event payload's ``stage`` field, so
# postmortem grep on either surface lands on the same vocabulary.
_OPERATOR_STAGE_EMPTY = "empty"
_OPERATOR_STAGE_MISSING_FILE = "missing_file"
_OPERATOR_STAGE_NAME_RESOLVE = "name_resolve"
_OPERATOR_STAGE_AUTOTUNE = "autotune_validate"
_OPERATOR_STAGE_COMPILE = "compile"
_OPERATOR_STAGE_CORRECTNESS = "correctness"


class BaselineGenerationError(Exception):
    """Raised when a verified Triton baseline cannot be produced."""


async def generate_triton_baseline(
    definition: Definition,
    spec: KernelSpec,
    *,
    coder: CoderAgent | None,
    workloads: list[Workload],
    max_retries: int = 3,
    cache_dir: Path | None = None,
    policy: ComparisonPolicy | None = None,
    blob_roots: list[Path] | None = None,
) -> Kernel:
    """Translate a PyTorch reference into a verified Triton baseline.

    Returns the first candidate that compiles and passes correctness on
    every workload in *workloads*. Raises ``BaselineGenerationError``
    when no model is configured or when the attempt budget is exhausted.

    *blob_roots* is forwarded to ``build_input_generator`` so workloads
    that declare ``SafetensorsInput`` can resolve their on-disk weights
    during this baseline-translation step. Mirrors the same kwarg flow
    the search-loop input generators use in ``_load_sol_problem``;
    omitting it here would make any safetensors-bearing problem fail
    before Phase B starts.
    """
    if coder is None or not coder.has_model:
        raise BaselineGenerationError(
            f"No model configured for '{definition.name}' — set ACTS_MODEL_CONFIG "
            "or drop configs/models/<provider>.json in place.",
        )

    if not workloads:
        raise ValueError(
            "generate_triton_baseline requires at least one workload.",
        )

    reference_fn = build_reference_fn(definition.reference)
    input_generators = [
        build_input_generator(definition, w, blob_roots=blob_roots) for w in workloads
    ]

    # Accumulator threaded into each attempt's translate() call. Grows by one
    # AttemptFailure per failed attempt (ImplementationError, post-verify
    # compile, post-verify correctness) so the next attempt's user prompt
    # carries a "## Prior attempt failures" section listing what didn't work
    # in earlier sessions. See doc/specs/2026-05-13-cross-attempt-memory-design.md.
    prior_failures: list[AttemptFailure] = []
    for attempt in range(max_retries):
        emit("baseline_attempt", attempt=attempt + 1, max_attempts=max_retries)
        try:
            with trace_span(
                "acts_baseline",
                iter_no=0,
                agent=AgentLabel.CODER_TRANSLATE,
                attempt=attempt + 1,
            ):
                output = await coder.translate(
                    reference_source=definition.reference,
                    kernel_spec=spec,
                    reference_fn=reference_fn,
                    input_generators=input_generators,
                    definition=definition,
                    workloads=workloads,
                    prior_failures=prior_failures,
                    # Baseline is conventionally iter 0 — threads through
                    # to compile_kernel_tool's events.emit calls so the
                    # SMEM telemetry on baseline-attempt failures lands
                    # with iter=0 instead of iter=null. Fix #8 plumbing.
                    iter_no=0,
                    workload_shapes=[tuple(w.axes.values()) for w in workloads],
                )
        except ImplementationError as exc:
            prior_failures.append(
                AttemptFailure(
                    attempt_no=attempt + 1,
                    tool_errors=list(exc.tool_errors),
                )
            )
            emit(
                "baseline_failure",
                attempt=attempt + 1,
                reason=f"ImplementationError: {str(exc)[:200]}",
            )
            continue

        # The KernelCodeOutput validator only checks that the declared
        # triton_kernel_name appears as an @triton.jit def somewhere in
        # source — a multi-kernel source can still declare one JIT def
        # while spec.entrypoint launches a different one, skewing
        # profiler/autotune attribution. Bind-check here before the
        # Kernel is constructed so the retry loop carries the diagnostic
        # in prior_failures.
        ok, reason = find_jit_name_in_entrypoint(
            output.source_code, spec.entrypoint, output.triton_kernel_name,
        )
        if not ok:
            prior_failures.append(
                AttemptFailure(
                    attempt_no=attempt + 1,
                    tool_errors=[f"Entrypoint-binding FAILED:\n{reason}"],
                )
            )
            emit(
                "baseline_failure",
                attempt=attempt + 1,
                reason=f"EntrypointBinding: {reason[:160]}",
            )
            continue

        candidate = Kernel(
            spec=spec,
            source_code=output.source_code,
            triton_kernel_name=output.triton_kernel_name,
            dps=output.dps,
        )
        compiled = compile_kernel(candidate, cache_dir=cache_dir)
        if not compiled.success:
            prior_failures.append(
                AttemptFailure(
                    attempt_no=attempt + 1,
                    tool_errors=[
                        f"{_POST_VERIFY_COMPILE_FAILED}:\n{compiled.error_message}"
                    ],
                )
            )
            emit(
                "baseline_failure",
                attempt=attempt + 1,
                reason=f"CompileError: {str(compiled.error_message or '')[:200]}",
            )
            continue

        # Walk explicitly so the first failure can be captured for prior_failures.
        first_failure: "CorrectnessResult | None" = None
        first_failure_idx: int = -1
        for idx, (gen, wl) in enumerate(zip(input_generators, workloads)):
            result = verify_correctness(
                candidate_fn=compiled.compiled_fn,
                reference_fn=reference_fn,
                input_generator=gen,
                definition=definition,
                kernel=candidate,
                workload=wl,
                policy=policy,
            )
            if not result.passed:
                first_failure = result
                first_failure_idx = idx
                break

        if first_failure is None:
            emit(
                "baseline_success",
                source_bytes=len(output.source_code),
                triton_kernel_name=output.triton_kernel_name or "",
            )
            return candidate

        prior_failures.append(
            AttemptFailure(
                attempt_no=attempt + 1,
                tool_errors=[
                    f"{_POST_VERIFY_CORRECTNESS_FAILED} on workload "
                    f"{first_failure_idx + 1}/{len(workloads)}:\n"
                    f"{first_failure.error_message}"
                ],
            )
        )
        emit(
            "baseline_failure",
            attempt=attempt + 1,
            reason="CorrectnessError: post-verify failed on one or more workloads",
        )

    raise BaselineGenerationError(
        f"Baseline translation for '{definition.name}' failed after "
        f"{max_retries} attempts.",
    )


def _fail_operator(stage: str, message: str) -> "BaselineGenerationError":
    """Emit the failure event and build the BaselineGenerationError.

    Caller raises the returned exception; emission happens inside this
    helper so every failure path keeps the event-and-raise pair in sync.
    """
    reason = message[:200]
    emit("operator_baseline_failure", stage=stage, reason=reason)
    return BaselineGenerationError(f"[{stage}] {message}")


async def load_operator_baseline(
    definition: "Definition",
    spec: "KernelSpec",
    *,
    path: "Path",
    dps: bool,
    kernel_name_override: str | None,
    enforce_autotune: bool,
    workloads: list["Workload"],
    cache_dir: "Path | None" = None,
    policy: "ComparisonPolicy | None" = None,
    blob_roots: list["Path"] | None = None,
) -> Kernel:
    """Load an operator-supplied Triton kernel as the search-tree root.

    Bypasses ``CoderAgent.translate()``. Source is read from *path* and
    verified via the same compile + per-workload ``verify_correctness``
    gate the LLM path runs after ``translate()``. Raises
    ``BaselineGenerationError`` on any gate failure — no retry loop, no
    fallback to the LLM path. See
    ``doc/specs/2026-05-16-operator-supplied-triton-baseline-design.md``.
    """
    if not workloads:
        raise ValueError(
            "load_operator_baseline requires at least one workload.",
        )

    # Gate 1: read source + missing-file / empty guard.
    if not path.exists():
        raise _fail_operator(
            _OPERATOR_STAGE_MISSING_FILE,
            f"Operator baseline file missing: {path}",
        )
    source = path.read_text()
    if not source.strip():
        raise _fail_operator(
            _OPERATOR_STAGE_EMPTY,
            f"Operator baseline file empty: {path}",
        )

    # Gate 2 + 3: resolve triton_kernel_name (override -> auto-detect)
    # and verify the resolved name actually appears in source.
    names = triton_kernel_names_in(source)
    if kernel_name_override:
        if kernel_name_override not in names:
            raise _fail_operator(
                _OPERATOR_STAGE_NAME_RESOLVE,
                f"triton_baseline_kernel_name={kernel_name_override!r} not "
                f"found in source; @triton.jit defs present: {names}",
            )
        resolved_name = kernel_name_override
    else:
        if not names:
            raise _fail_operator(
                _OPERATOR_STAGE_NAME_RESOLVE,
                f"Operator baseline has no @triton.jit def in {path}",
            )
        if len(names) > 1:
            raise _fail_operator(
                _OPERATOR_STAGE_NAME_RESOLVE,
                f"Operator baseline has {len(names)} @triton.jit defs "
                f"({names}); set [runtime] triton_baseline_kernel_name "
                f"to disambiguate",
            )
        resolved_name = names[0]

    # Gate 3b: entrypoint-binding check. Symmetric with the
    # ``generate_triton_baseline`` and ``Orchestrator`` per-iter gates;
    # see ``find_jit_name_in_entrypoint`` for the contract.
    ok, reason = find_jit_name_in_entrypoint(
        source, spec.entrypoint, resolved_name,
    )
    if not ok:
        raise _fail_operator(_OPERATOR_STAGE_NAME_RESOLVE, reason)

    emit(
        "operator_baseline_load",
        path=str(path),
        kernel_name=resolved_name,
        dps=dps,
        enforce_autotune=enforce_autotune,
    )

    # Gate 4: autotune validator — opt-in. When enforced, reuse the
    # KernelCodeOutput validator's logic by constructing the Pydantic
    # model (its model_validator chain runs both name-match — already
    # passed — and autotune well-formedness). On ValidationError, wrap
    # as BaselineGenerationError.
    if enforce_autotune:
        from pydantic import ValidationError

        from src.agents.coder import KernelCodeOutput
        try:
            KernelCodeOutput(
                source_code=source,
                triton_kernel_name=resolved_name,
                dps=dps,
            )
        except ValidationError as exc:
            raise _fail_operator(
                _OPERATOR_STAGE_AUTOTUNE,
                f"Autotune validator failed: {exc}",
            ) from exc

    # Gate 5: construct Kernel (its __post_init__ parses autotune
    # metadata from source; lenient when no decorator is present).
    kernel = Kernel(
        spec=spec,
        source_code=source,
        triton_kernel_name=resolved_name,
        dps=dps,
    )

    # Gate 6: compile.
    with trace_span(
        "acts_operator_baseline",
        iter_no=0,
        agent=AgentLabel.CODER_TRANSLATE,
    ):
        compiled = compile_kernel(kernel, cache_dir=cache_dir)
    if not compiled.success:
        raise _fail_operator(
            _OPERATOR_STAGE_COMPILE,
            f"Operator baseline compile FAILED: {compiled.error_message}",
        )

    # Gate 7: per-workload correctness (first failure wins).
    reference_fn = build_reference_fn(definition.reference)
    input_generators = [
        build_input_generator(definition, w, blob_roots=blob_roots)
        for w in workloads
    ]
    for idx, (gen, wl) in enumerate(zip(input_generators, workloads)):
        result = verify_correctness(
            candidate_fn=compiled.compiled_fn,
            reference_fn=reference_fn,
            input_generator=gen,
            definition=definition,
            kernel=kernel,
            workload=wl,
            policy=policy,
        )
        if not result.passed:
            raise _fail_operator(
                f"{_OPERATOR_STAGE_CORRECTNESS} wl {idx + 1}/{len(workloads)}",
                f"Operator baseline correctness FAILED on workload "
                f"{idx + 1}/{len(workloads)}: {result.error_message}",
            )

    emit(
        "operator_baseline_success",
        source_bytes=len(source),
        triton_kernel_name=resolved_name,
    )
    return kernel
