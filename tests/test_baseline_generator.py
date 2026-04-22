"""Tests for benchmark/baseline_generator.py — PyTorch→Triton translation loop.

Covers:
- no Coder / no model is configured → raises BaselineGenerationError (fail closed
  so real SOL runs can't silently search against a fake baseline)
- happy path: Coder returns source, all selected workloads verify → Kernel returned
- correctness failure on any workload consumes one attempt and triggers retry
- `ImplementationError` (transient LLM failure) consumes one attempt and triggers retry
- compile failure during post-verification is treated the same as correctness failure
- all attempts fail → raises BaselineGenerationError (problem gets skipped by caller)
- ``CoderAgent.translate`` is invoked with the PyTorch reference source, the
  KernelSpec, and an input generator built from the first selected workload
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.coder import CoderAgent, ImplementationError, KernelCodeOutput
from src.benchmark.baseline_generator import (
    BaselineGenerationError,
    generate_triton_baseline,
)


def _coder_output(source_code: str, triton_kernel_name: str = "kernel_fn") -> KernelCodeOutput:
    """Test helper: build a KernelCodeOutput without paying the Pydantic
    validator's cost. Most tests here drive the retry/loop control flow
    with placeholder source like ``"good source"`` that wouldn't satisfy
    the ``@triton.jit`` cross-validation."""
    return KernelCodeOutput.model_construct(
        source_code=source_code,
        triton_kernel_name=triton_kernel_name,
    )
from src.benchmark.problem import Problem, Workload
from src.eval.correctness import CorrectnessResult, CorrectnessStage
from src.kernels.compiler import CompilationResult
from src.kernels.kernel import KernelSpec, KernelType


# ── fixtures ───────────────────────────────────────────────────────────

def _make_problem(
    name: str = "test_prob",
    reference: str = "def run(x):\n    return x * 2.0\n",
) -> Problem:
    return Problem(
        name=name,
        axes={},
        inputs={},
        outputs={},
        reference_source=reference,
        op_type="elementwise",
    )


def _make_workloads(n: int = 3) -> list[Workload]:
    return [Workload(uuid=f"wl-{i}", axes={}, inputs={}) for i in range(n)]


def _make_spec(name: str = "test_prob", entrypoint: str = "kernel_fn") -> KernelSpec:
    return KernelSpec(
        name=name,
        kernel_type=KernelType.ELEMENTWISE,
        entrypoint=entrypoint,
        pytorch_reference="def run(x):\n    return x * 2.0\n",
    )


def _pass() -> CorrectnessResult:
    return CorrectnessResult(passed=True, max_abs_error=0.0)


def _fail(stage: CorrectnessStage = CorrectnessStage.SMOKE_TEST) -> CorrectnessResult:
    return CorrectnessResult(
        passed=False, failed_stage=stage, error_message="mismatch", max_abs_error=1.0,
    )


@pytest.fixture
def patched_io():
    """Patch torch-dependent helpers so tests run in the torch-less venv."""
    with (
        patch(
            "src.benchmark.baseline_generator.build_reference_fn",
            return_value=lambda x: x * 2.0,
        ),
        patch(
            "src.benchmark.baseline_generator.build_input_generator",
            side_effect=lambda problem, workload, **_: lambda seed: (float(seed),),
        ),
    ):
        yield


def _compile_ok() -> CompilationResult:
    return CompilationResult(success=True, compiled_fn=lambda x: x * 2.0)


def _compile_fail() -> CompilationResult:
    return CompilationResult(success=False, error_message="SyntaxError: bad")


# ── guards ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_workloads_raises_value_error():
    """An empty workload list is a config/loader bug — fail fast before retries."""
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(return_value=_coder_output("src"))
    with pytest.raises(ValueError, match="workload"):
        await generate_triton_baseline(
            _make_problem(), _make_spec(), coder=coder, workloads=[],
        )


# ── no-model / fail-closed path ────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_coder_raises_baseline_error():
    """Without a coder we must fail closed, not fabricate a stub kernel.
    A stub would let a real SOL run silently "search" against fake baseline
    source — the surrounding pipeline has no way to know it was never translated.
    """
    with pytest.raises(BaselineGenerationError, match="No model"):
        await generate_triton_baseline(
            _make_problem(), _make_spec(), coder=None, workloads=_make_workloads(),
        )


@pytest.mark.asyncio
async def test_coder_without_model_raises_baseline_error():
    """CoderAgent(model=None) has no oracle to bind to — same fail-closed rule."""
    coder = CoderAgent(model=None)
    with pytest.raises(BaselineGenerationError, match="No model"):
        await generate_triton_baseline(
            _make_problem(), _make_spec(), coder=coder, workloads=_make_workloads(),
        )


# ── LLM path — happy case ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_successful_translate_returns_verified_kernel(patched_io):
    """Coder returns good source; all selected workloads pass → return that Kernel."""
    spec = _make_spec()
    workloads = _make_workloads(n=3)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(
        return_value=_coder_output("@triton.jit\ndef kernel_fn(x): pass")
    )

    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.verify_correctness",
            return_value=_pass(),
        ) as mock_verify,
    ):
        result = await generate_triton_baseline(
            _make_problem(), spec, coder=coder, workloads=workloads,
        )

    assert result is not None
    assert result.source_code == "@triton.jit\ndef kernel_fn(x): pass"
    assert result.spec is spec
    assert mock_verify.call_count == 3  # once per workload
    coder.translate.assert_awaited_once()


@pytest.mark.asyncio
async def test_verify_uses_all_selected_workloads(patched_io):
    """One input_generator per selected workload; verify_correctness runs once per."""
    workloads = _make_workloads(n=3)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(return_value=_coder_output("src"))

    with (
        patch(
            "src.benchmark.baseline_generator.build_input_generator",
            side_effect=lambda p, w, **_: lambda seed: (w.uuid, seed),
        ) as mock_build_gen,
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.verify_correctness",
            return_value=_pass(),
        ) as mock_verify,
    ):
        result = await generate_triton_baseline(
            _make_problem(), _make_spec(), coder=coder, workloads=workloads,
        )

    assert result is not None
    assert mock_build_gen.call_count == 3  # one generator per workload
    assert mock_verify.call_count == 3


@pytest.mark.asyncio
async def test_translate_receives_reference_source_and_all_generators(patched_io):
    """translate() gets reference_source, spec, reference_fn, and every selected
    workload's generator — so its correctness tool can catch cross-workload bugs."""
    problem = _make_problem(reference="def run(x):\n    return x + 1\n")
    spec = _make_spec()
    workloads = _make_workloads(n=3)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(return_value=_coder_output("src"))

    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.verify_correctness",
            return_value=_pass(),
        ),
    ):
        await generate_triton_baseline(
            problem, spec, coder=coder, workloads=workloads,
        )

    kwargs = coder.translate.call_args.kwargs
    assert kwargs["reference_source"] == problem.reference_source
    assert kwargs["kernel_spec"] is spec
    assert callable(kwargs["reference_fn"])
    assert isinstance(kwargs["input_generators"], list)
    assert len(kwargs["input_generators"]) == 3
    assert all(callable(g) for g in kwargs["input_generators"])


# ── T4: triton_kernel_name propagation ─────────────────────────────────

@pytest.mark.asyncio
async def test_translate_kernel_name_propagates_to_kernel(patched_io):
    """T4: ``triton_kernel_name`` declared in the translate output must land on
    the returned Kernel — otherwise the profiler's regex fallback re-extracts
    (correct today, but a silent contract break) and a fused baseline could
    pick the wrong jit'd function."""
    spec = _make_spec()
    workloads = _make_workloads(n=1)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(
        return_value=_coder_output(
            "@triton.jit\ndef _epilogue(): pass\n@triton.jit\ndef main_k(): pass\n",
            triton_kernel_name="main_k",
        )
    )

    with (
        patch("src.benchmark.baseline_generator.compile_kernel", return_value=_compile_ok()),
        patch("src.benchmark.baseline_generator.verify_correctness", return_value=_pass()),
    ):
        result = await generate_triton_baseline(
            _make_problem(), spec, coder=coder, workloads=workloads,
        )

    assert result.triton_kernel_name == "main_k"


# ── retry semantics ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_correctness_failure_on_any_workload_triggers_retry(patched_io):
    """A single failed workload consumes one attempt; a retry is taken."""
    workloads = _make_workloads(n=3)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(
        side_effect=[_coder_output("bad source"), _coder_output("good source")]
    )

    # Attempt 1: workload 0 passes, workload 1 fails (short-circuit).
    # Attempt 2: all three pass.
    correctness_sequence = [_pass(), _fail(), _pass(), _pass(), _pass()]
    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.verify_correctness",
            side_effect=correctness_sequence,
        ),
    ):
        result = await generate_triton_baseline(
            _make_problem(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
        )

    assert result is not None
    assert result.source_code == "good source"
    assert coder.translate.await_count == 2


@pytest.mark.asyncio
async def test_implementation_error_triggers_retry(patched_io):
    """Transient ImplementationError consumes one attempt; then a retry is taken."""
    workloads = _make_workloads(n=2)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(
        side_effect=[ImplementationError("LLM failed"), _coder_output("good source")]
    )

    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.verify_correctness",
            return_value=_pass(),
        ),
    ):
        result = await generate_triton_baseline(
            _make_problem(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
        )

    assert result is not None
    assert result.source_code == "good source"
    assert coder.translate.await_count == 2


@pytest.mark.asyncio
async def test_compile_failure_in_post_verify_is_treated_as_attempt_failure(patched_io):
    """If the translated source won't compile, skip verify and retry."""
    workloads = _make_workloads(n=1)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(
        side_effect=[_coder_output("won't compile"), _coder_output("good source")]
    )

    compile_sequence = [_compile_fail(), _compile_ok()]
    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            side_effect=compile_sequence,
        ),
        patch(
            "src.benchmark.baseline_generator.verify_correctness",
            return_value=_pass(),
        ) as mock_verify,
    ):
        result = await generate_triton_baseline(
            _make_problem(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
        )

    assert result is not None
    assert result.source_code == "good source"
    # verify_correctness never ran on the failed-compile attempt
    assert mock_verify.call_count == 1
    assert coder.translate.await_count == 2


@pytest.mark.asyncio
async def test_all_attempts_fail_raises_baseline_error(patched_io):
    """Budget exhausted (mix of failures) → raise BaselineGenerationError with the
    attempt count, so the caller can skip this problem via a typed exception
    instead of silently continuing on a sentinel value."""
    workloads = _make_workloads(n=2)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(
        side_effect=[
            ImplementationError("transient"),
            _coder_output("bad1"),
            _coder_output("bad2"),
        ]
    )

    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.verify_correctness",
            return_value=_fail(),
        ),
        pytest.raises(BaselineGenerationError, match="3 attempts"),
    ):
        await generate_triton_baseline(
            _make_problem(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
        )

    assert coder.translate.await_count == 3
