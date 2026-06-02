"""5-stage correctness verification gate.

Called by Coder's ``check_correctness_tool`` during its turn. By the time
the Coder returns, correctness is guaranteed (or the branch fails).

Correctness is checked against a caller-supplied ``reference_fn`` — the
PyTorch reference from ``definition.json`` in production. Using the Triton
baseline as oracle would propagate its translation bugs through the run.

Stages (short-circuit on first failure):
    1. Smoke test          — single input, output matches reference
    2. Shape sweep         — N trials, varying seeds / input shapes
    3. Numerical stability — no NaN / Inf on normal inputs
    4. Determinism         — repeated runs on identical input match bitwise
    5. Anti-cheat          — randomized inputs under strict tolerance

Tensor comparison is delegated to a ``ComparisonPolicy`` — the production
torch policy is lazy-built so this module stays import-clean when torch
is absent (unit tests inject a scalar policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload

    from src.kernels.kernel import Kernel


__all__ = [
    "ComparisonPolicy",
    "ComparisonResult",
    "CorrectnessResult",
    "CorrectnessStage",
    "TorchComparisonPolicy",
    "build_normalize_context",
    "compare_outputs",
    "maybe_wrap_dps_candidate",
    "strict_compare_one_workload",
    "verify_correctness",
]


class CorrectnessStage(Enum):
    SMOKE_TEST = "smoke_test"
    SHAPE_SWEEP = "shape_sweep"
    NUMERICAL_STABILITY = "numerical_stability"
    DETERMINISM = "determinism"
    ANTI_CHEAT = "anti_cheat"


@dataclass
class ComparisonResult:
    match: bool
    max_abs_error: float = 0.0
    reason: str = ""


@dataclass
class CorrectnessResult:
    passed: bool
    failed_stage: CorrectnessStage | None = None
    error_message: str = ""
    max_abs_error: float = 0.0


class ComparisonPolicy(Protocol):
    """Tensor-comparison abstraction.

    Production: torch-backed (``TorchComparisonPolicy``).
    Tests: scalar-backed, no torch dependency.
    """

    def compare(
        self, output: Any, expected: Any, *, atol: float, rtol: float
    ) -> ComparisonResult: ...

    def contains_non_finite(self, output: Any) -> bool: ...

    def bitwise_equal(self, a: Any, b: Any) -> bool: ...


@lru_cache(maxsize=1)
def _try_import_sol():
    """Resolve sol_execbench symbols once per process. Raises ImportError
    when absent — SOL is required for the production policy."""
    try:
        from sol_execbench.core.bench.correctness import compute_error_stats
        from sol_execbench.core.data.workload import ToleranceSpec
    except ImportError as exc:
        raise ImportError(
            "TorchComparisonPolicy requires sol_execbench. "
            "Install via: pip install -e <path-to-SOL-ExecBench> --no-deps"
        ) from exc
    return compute_error_stats, ToleranceSpec


class TorchComparisonPolicy:
    """Default production policy. Torch is imported lazily inside methods.

    Element-wise comparison is delegated to SOL-ExecBench's
    ``compute_error_stats`` — gives us matched-ratio tolerance, separate
    NaN/Inf flags, and a hard max-error cap. SOL is required: a
    ``torch.allclose`` fallback's pass criterion (every element within
    tolerance) diverges from SOL's matched-ratio rule and would hide bf16
    outliers from non-SOL test runs. Constructing the policy when SOL is
    unimportable raises ``ImportError`` at first ``compare`` call rather
    than silently using a different rule.
    """

    def compare(
        self, output: Any, expected: Any, *, atol: float, rtol: float
    ) -> ComparisonResult:
        if output.shape != expected.shape:
            return ComparisonResult(
                match=False,
                reason=f"shape mismatch: {tuple(output.shape)} vs {tuple(expected.shape)}",
            )
        compute_error_stats, ToleranceSpec = _try_import_sol()

        # SOL-ExecBench element-wise pass condition (compute_error_stats):
        #   tol_bound       = max_atol + max_rtol * |reference|
        #   element_passes  iff |output - reference| <= tol_bound
        #   overall_passes  iff (passing / total) >= required_matched_ratio
        # required_matched_ratio is left at SOL's default — the 1% slack
        # absorbs bf16 quantization outliers (~1 ULP ≈ 7.8e-3 at magnitude 1)
        # without rejecting mathematically correct kernels.
        tolerance = ToleranceSpec(max_atol=atol, max_rtol=rtol)
        correctness, exceeds = compute_error_stats(output, expected, tolerance)
        max_err = float(correctness.max_absolute_error or 0.0)
        if not exceeds:
            return ComparisonResult(match=True, max_abs_error=max_err)
        if correctness.has_nan:
            reason = "NaN in output or reference"
        elif correctness.has_inf:
            reason = "Inf in output or reference"
        else:
            # ``tol_bound`` at the worst-absolute-error position: SOL's
            # per-element bound is ``atol + rtol * |reference|``, so its
            # numerical value depends on the reference magnitude at the
            # specific element that drove ``max_abs``. Reported so the
            # LLM (and post-run debug) can see how far over the threshold
            # the worst element actually was — ``max_abs`` alone hides
            # whether the kernel was 2× over or 500× over.
            tol_bound_at_max = _tol_bound_at_max_abs(
                output, expected, atol=atol, rtol=rtol,
            )
            reason = (
                f"tolerance exceeded: max_abs={max_err:.3e}, "
                f"max_rel={float(correctness.max_relative_error or 0.0):.3e} "
                f"(atol={atol}, rtol={rtol}, "
                f"tol_bound@max_abs={tol_bound_at_max:.3e})"
            )
        return ComparisonResult(match=False, max_abs_error=max_err, reason=reason)

    def contains_non_finite(self, output: Any) -> bool:
        import torch

        return bool(torch.isnan(output).any() or torch.isinf(output).any())

    def bitwise_equal(self, a: Any, b: Any) -> bool:
        import torch

        return bool(torch.equal(a, b))


def _tol_bound_at_max_abs(
    output: Any, expected: Any, *, atol: float, rtol: float,
) -> float:
    """``atol + rtol * |reference|`` at the position where ``|output - reference|``
    is largest. Matches SOL's per-element ``tol_bound`` (computed in fp32
    inside ``compute_error_stats``); the cast mirrors SOL's so the reported
    bound aligns with the threshold SOL actually compared against.
    """
    import torch

    out_f32 = output.to(torch.float32)
    ref_f32 = expected.to(torch.float32)
    diff = torch.abs(out_f32 - ref_f32)
    idx = int(diff.flatten().argmax().item())
    ref_at_max = float(torch.abs(ref_f32.flatten()[idx]).item())
    return atol + rtol * ref_at_max


@dataclass
class _StageOutcome:
    match: bool
    max_abs_error: float = 0.0
    reason: str = ""


@dataclass
class _NormalizeContext:
    """Cached wiring needed to route candidate/reference outputs through
    SOL's ``normalize_outputs`` before per-name comparison.

    Built once per ``verify_correctness`` call when a ``Definition`` is
    supplied; ``output_names`` and ``output_dtypes`` are derived from
    the definition's ``outputs`` mapping. ``device`` is resolved lazily
    from the first reference output (a tensor) so we don't need to
    thread the workload's device explicitly — SOL already pins inputs
    to the bench device, so reference outputs land there too.
    """

    output_names: list[str]
    output_dtypes: dict[str, Any]
    normalize_outputs: Callable[..., Any]


def build_normalize_context(definition: Definition | None) -> _NormalizeContext | None:
    """Resolve SOL's ``normalize_outputs`` + the per-name dtype map.

    Returns ``None`` when *definition* is ``None`` so the legacy
    "compare raw outputs directly" path stays available for unit tests
    that drive the gate with scalar policies and have no Definition. In
    production (where the orchestrator threads ``definition`` through),
    this is always populated.
    """
    if definition is None:
        return None
    from sol_execbench.core.bench.io import normalize_outputs
    from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype

    output_names = list(definition.outputs.keys())
    output_dtypes = {
        name: dtype_str_to_torch_dtype(spec.dtype)
        for name, spec in definition.outputs.items()
    }
    return _NormalizeContext(
        output_names=output_names,
        output_dtypes=output_dtypes,
        normalize_outputs=normalize_outputs,
    )


def maybe_wrap_dps_candidate(
    candidate_fn: Callable[..., Any],
    *,
    kernel: Kernel | None,
    workload: Workload | None,
    definition: Definition | None,
) -> Callable[..., Any]:
    """Wrap *candidate_fn* to allocate per-call output buffers when the
    kernel uses destination-passing-style.

    Mirrors ``_wrap_dps_generator`` in ``src.eval.benchmark`` but acts on
    the *candidate* call inside the correctness gate (the benchmark loop
    shapes args via the input generator; the correctness gate calls
    ``candidate_fn(*args)`` directly). Returns ``candidate_fn`` unchanged
    when the kernel is not DPS or when no kernel was supplied — preserves
    the legacy single-return path for scalar-policy unit tests and
    non-SOL benchmarks. When ``kernel.dps`` is True, *definition* and
    *workload* are required so ``allocate_outputs`` can resolve output
    shapes; missing them is a contract bug at the call site, surfaced as
    ``ValueError`` rather than a silent TypeError in the candidate body.
    """
    if kernel is None or not kernel.dps:
        return candidate_fn
    if definition is None or workload is None:
        raise ValueError(
            "verify_correctness with kernel.dps=True requires both "
            "definition and workload — allocate_outputs needs the "
            "definition's output schema and the workload's resolved axes."
        )

    # Lazy import — keep the torch-less unit tests importing correctness.py
    # without paying for sol_execbench at module load time.
    from src.eval.inputs import allocate_dps_outputs

    def _dps_candidate(*inputs: Any) -> Any:
        import torch

        device = "cuda"
        for arg in inputs:
            if isinstance(arg, torch.Tensor):
                device = str(arg.device)
                break
        outputs = allocate_dps_outputs(definition, workload, device=device)
        candidate_fn(*inputs, *outputs)
        # Tuple matches the reference's return-by-value shape so
        # ``normalize_outputs`` lines both sides up by output index.
        return tuple(outputs)

    return _dps_candidate


def compare_outputs(
    candidate_out: Any,
    reference_out: Any,
    *,
    policy: ComparisonPolicy,
    atol: float,
    rtol: float,
    norm: _NormalizeContext | None,
) -> _StageOutcome:
    """Run policy.compare against either raw outputs (legacy path) or
    each named output of the normalized dicts. On multi-output runs the
    first failure short-circuits and reports its own reason; success
    folds the worst per-name error into the returned ``max_abs_error``.
    """
    if norm is None:
        cmp = policy.compare(candidate_out, reference_out, atol=atol, rtol=rtol)
        return _StageOutcome(
            match=cmp.match, max_abs_error=cmp.max_abs_error, reason=cmp.reason
        )

    # Resolve the device from the reference's first tensor — SOL's
    # normalize_outputs needs an explicit ``device`` arg to coerce raw
    # scalars / cpu tensors back onto the kernel's device.
    device = _infer_device(reference_out, candidate_out)
    cand_dict = norm.normalize_outputs(
        candidate_out,
        device=device,
        output_names=norm.output_names,
        output_dtypes=norm.output_dtypes,
    )
    ref_dict = norm.normalize_outputs(
        reference_out,
        device=device,
        output_names=norm.output_names,
        output_dtypes=norm.output_dtypes,
    )
    worst_err = 0.0
    for name in norm.output_names:
        cmp = policy.compare(cand_dict[name], ref_dict[name], atol=atol, rtol=rtol)
        if not cmp.match:
            return _StageOutcome(
                match=False,
                max_abs_error=cmp.max_abs_error,
                reason=f"output[{name!r}]: {cmp.reason}",
            )
        worst_err = max(worst_err, cmp.max_abs_error)
    return _StageOutcome(match=True, max_abs_error=worst_err, reason="")


def strict_compare_one_workload(
    *,
    candidate_fn: Callable[..., Any],
    reference_fn: Callable[..., Any],
    input_generator: Callable[[int], tuple],
    definition: Definition | None,
    kernel: Kernel | None,
    workload: Workload | None,
    seed: int,
    atol: float,
    rtol: float,
) -> bool:
    """Run candidate + reference once at ``seed`` and compare with the given
    strict tolerance. Shared by the correctness worker's ``strict_recheck``
    mode and the orchestrator's in-parent reward-hack re-eval fallback so the
    two cannot drift on wrap/seed/tolerance semantics. Returns True iff the
    outputs match. Anti-cheat context + RewardHackDetected handling stay at
    the call sites."""
    from src.eval.anti_cheat import generate_randomized_inputs  # lazy: avoid import cycle
    policy = TorchComparisonPolicy()
    norm = build_normalize_context(definition)
    wrapped = maybe_wrap_dps_candidate(
        candidate_fn, kernel=kernel, workload=workload, definition=definition,
    )
    inputs = generate_randomized_inputs(input_generator, seed=seed)
    cand_out = wrapped(*inputs)
    ref_out = reference_fn(*inputs)
    outcome = compare_outputs(cand_out, ref_out, policy=policy, atol=atol, rtol=rtol, norm=norm)
    return bool(outcome.match)


def _infer_device(*outputs: Any) -> Any:
    """Pick a torch.device to pass to ``normalize_outputs``.

    Walks the supplied outputs (in order) and returns the first tensor's
    device. Falls back to CUDA-or-CPU when none of the outputs is a
    tensor (e.g., scalar return) — SOL's normalize_outputs uses *device*
    only when it must coerce a non-tensor scalar onto the bench device,
    so the fallback only matters in that path.
    """
    import torch

    for out in outputs:
        if isinstance(out, torch.Tensor):
            return out.device
        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item.device
        if isinstance(out, dict):
            for item in out.values():
                if isinstance(item, torch.Tensor):
                    return item.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_compare_trial(
    candidate_fn: Callable[..., Any],
    reference_fn: Callable[..., Any],
    input_generator: Callable[[int], tuple],
    *,
    seed: int,
    policy: ComparisonPolicy,
    atol: float,
    rtol: float,
    norm: _NormalizeContext | None,
) -> _StageOutcome:
    try:
        args = input_generator(seed)
        expected = reference_fn(*args)
        output = candidate_fn(*args)
    except Exception as exc:
        return _StageOutcome(match=False, reason=f"{type(exc).__name__}: {exc}")
    return compare_outputs(
        output, expected, policy=policy, atol=atol, rtol=rtol, norm=norm,
    )


def _fail(
    stage: CorrectnessStage,
    reason: str,
    max_abs_error: float,
    *,
    trial: int | None = None,
) -> CorrectnessResult:
    prefix = f"[{stage.value}]"
    if trial is not None:
        prefix = f"{prefix} trial {trial}:"
    return CorrectnessResult(
        passed=False,
        failed_stage=stage,
        error_message=f"{prefix} {reason}",
        max_abs_error=max_abs_error,
    )


def verify_correctness(
    candidate_fn: Callable[..., Any],
    reference_fn: Callable[..., Any],
    input_generator: Callable[[int], tuple],
    *,
    definition: Definition | None = None,
    kernel: Kernel | None = None,
    workload: Workload | None = None,
    policy: ComparisonPolicy | None = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    strict_atol: float = 1e-5,
    strict_rtol: float = 1e-4,
    n_sweep_trials: int = 5,
    n_anti_cheat_trials: int = 3,
) -> CorrectnessResult:
    """Run the 5-stage correctness gate.

    ``input_generator(seed)`` returns the args tuple for a trial. Seeds
    used: 42 (smoke), 0..n_sweep_trials-1 (sweep), 7 (stability), 11
    (determinism), 1000..1000+n_anti_cheat_trials-1 (anti-cheat).

    ``atol`` / ``rtol`` mirror SOL-ExecBench's ``ToleranceSpec`` defaults
    (1e-2) — loose enough for bf16 storage roundtrip (~1 ULP ≈ 7.8e-3
    at magnitude 1), tight enough to catch fp32 math errors. Stages 1–4
    use these; stage 5 (anti-cheat) uses ``strict_atol`` / ``strict_rtol``
    to catch kernels that pass on canned seeds but fail on randomized inputs.

    *definition* is the SOL-ExecBench Definition for the workload. When
    provided, both candidate and reference outputs flow through SOL's
    ``normalize_outputs`` so multi-output (tuple / dict) returns are
    compared name-by-name under per-name dtypes. When ``None`` the gate
    falls back to comparing raw outputs directly — preserves back-compat
    with non-SOL benchmarks and scalar-policy unit tests that drive the
    gate without a Definition.

    *kernel* + *workload* are required when the candidate's host wrapper
    is destination-passing-style (``kernel.dps=True``). The gate then
    allocates output buffers per call via
    ``sol_execbench.core.bench.io.allocate_outputs(definition,
    resolved_axes, device)`` and invokes ``candidate_fn(*inputs, *outputs)``;
    the filled buffers serve as the candidate's outputs for the per-stage
    comparison. The reference oracle (``reference_fn``) is always
    return-by-value — it's the PyTorch ``run()`` from ``definition.json``
    and is never DPS — so the comparison sides line up via SOL's
    ``normalize_outputs``. When ``kernel`` is None or ``kernel.dps`` is
    False, the gate calls ``candidate_fn(*inputs)`` and treats the return
    value as the output (legacy / non-DPS path).

    When *workload* carries a ``tolerance`` (SOL-ExecBench's
    ``ToleranceSpec``), its ``max_atol`` / ``max_rtol`` override **every**
    stage's tolerance — stages 1–4 *and* the anti-cheat stage 5. Anti-cheat
    therefore relaxes from the hardcoded ``strict_atol`` / ``strict_rtol``
    to the workload's per-problem spec. This is the deliberate "match the
    workload exactly" policy; callers that want the prior tighter
    anti-cheat behaviour pass ``workload=None`` (the override is opt-in via
    workload presence).
    """
    if workload is not None and getattr(workload, "tolerance", None) is not None:
        atol = workload.tolerance.max_atol
        rtol = workload.tolerance.max_rtol
        strict_atol = workload.tolerance.max_atol
        strict_rtol = workload.tolerance.max_rtol
    policy = policy or TorchComparisonPolicy()
    norm = build_normalize_context(definition)
    candidate_fn = maybe_wrap_dps_candidate(
        candidate_fn,
        kernel=kernel,
        workload=workload,
        definition=definition,
    )
    worst_error = 0.0

    # Stage 1: Smoke test
    stage = CorrectnessStage.SMOKE_TEST
    r = _run_compare_trial(
        candidate_fn, reference_fn, input_generator,
        seed=42, policy=policy, atol=atol, rtol=rtol, norm=norm,
    )
    if not r.match:
        return _fail(stage, r.reason, r.max_abs_error)
    worst_error = max(worst_error, r.max_abs_error)

    # Stage 2: Shape sweep
    stage = CorrectnessStage.SHAPE_SWEEP
    for i in range(n_sweep_trials):
        r = _run_compare_trial(
            candidate_fn, reference_fn, input_generator,
            seed=i, policy=policy, atol=atol, rtol=rtol, norm=norm,
        )
        if not r.match:
            return _fail(stage, r.reason, r.max_abs_error, trial=i)
        worst_error = max(worst_error, r.max_abs_error)

    # Stage 3: Numerical stability — candidate must match oracle AND be finite.
    # The oracle compare guards against seed-7-specific wrong answers that
    # would otherwise slip past a pure NaN/Inf check.
    stage = CorrectnessStage.NUMERICAL_STABILITY
    try:
        args = input_generator(7)
        expected = reference_fn(*args)
        output = candidate_fn(*args)
    except Exception as exc:
        return _fail(stage, f"{type(exc).__name__}: {exc}", worst_error)
    cmp = compare_outputs(
        output, expected, policy=policy, atol=atol, rtol=rtol, norm=norm,
    )
    if not cmp.match:
        return _fail(stage, cmp.reason, cmp.max_abs_error)
    worst_error = max(worst_error, cmp.max_abs_error)
    if _stage_output_has_nonfinite(output, policy=policy, norm=norm):
        return _fail(stage, "Output contains NaN or Inf on normal input.", worst_error)

    # Stage 4: Determinism — candidate must match oracle AND reproduce bitwise.
    # The oracle compare guards against seed-11-specific wrong answers that
    # would otherwise slip past a pure self-equality check.
    stage = CorrectnessStage.DETERMINISM
    try:
        args1 = input_generator(11)
        expected = reference_fn(*args1)
        out1 = candidate_fn(*args1)
        args2 = input_generator(11)
        out2 = candidate_fn(*args2)
    except Exception as exc:
        return _fail(stage, f"{type(exc).__name__}: {exc}", worst_error)
    cmp = compare_outputs(
        out1, expected, policy=policy, atol=atol, rtol=rtol, norm=norm,
    )
    if not cmp.match:
        return _fail(stage, cmp.reason, cmp.max_abs_error)
    worst_error = max(worst_error, cmp.max_abs_error)
    if not _stage_outputs_bitwise_equal(out1, out2, policy=policy, norm=norm):
        return _fail(
            stage,
            "Repeated runs on identical input produced different outputs.",
            worst_error,
        )

    # Stage 5: Anti-cheat (strict tolerance, fresh seeds)
    stage = CorrectnessStage.ANTI_CHEAT
    for i in range(n_anti_cheat_trials):
        r = _run_compare_trial(
            candidate_fn, reference_fn, input_generator,
            seed=1000 + i, policy=policy,
            atol=strict_atol, rtol=strict_rtol, norm=norm,
        )
        if not r.match:
            return _fail(stage, r.reason, r.max_abs_error, trial=i)
        worst_error = max(worst_error, r.max_abs_error)

    return CorrectnessResult(passed=True, max_abs_error=worst_error)


def _stage_output_has_nonfinite(
    output: Any,
    *,
    policy: ComparisonPolicy,
    norm: _NormalizeContext | None,
) -> bool:
    """Return True if any tensor in the output has NaN/Inf.

    Multi-output (tuple/list/dict) returns reach this check via ``norm``;
    each named output is fed through ``policy.contains_non_finite``. The
    legacy single-output path delegates straight to the policy.
    """
    if norm is None:
        return policy.contains_non_finite(output)
    cand_dict = norm.normalize_outputs(
        output,
        device=_infer_device(output),
        output_names=norm.output_names,
        output_dtypes=norm.output_dtypes,
    )
    return any(policy.contains_non_finite(t) for t in cand_dict.values())


def _stage_outputs_bitwise_equal(
    out1: Any,
    out2: Any,
    *,
    policy: ComparisonPolicy,
    norm: _NormalizeContext | None,
) -> bool:
    """Bitwise-equal check that handles multi-output normalization.

    Multi-output paths normalize both calls and compare element-wise per
    output name. Single-output legacy path delegates to the policy.
    """
    if norm is None:
        return policy.bitwise_equal(out1, out2)
    d1 = norm.normalize_outputs(
        out1,
        device=_infer_device(out1, out2),
        output_names=norm.output_names,
        output_dtypes=norm.output_dtypes,
    )
    d2 = norm.normalize_outputs(
        out2,
        device=_infer_device(out1, out2),
        output_names=norm.output_names,
        output_dtypes=norm.output_dtypes,
    )
    return all(policy.bitwise_equal(d1[name], d2[name]) for name in norm.output_names)
