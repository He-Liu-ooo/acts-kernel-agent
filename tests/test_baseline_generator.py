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
- ``blob_roots`` kwarg is forwarded to ``build_input_generator`` so safetensors-
  backed workloads resolve their on-disk weights before Phase B starts
  (regression: ``generate_triton_baseline``'s rebuild path must preserve the
  same ``blob_roots`` thread-through that ``_load_sol_problem`` does for the
  search-loop generators; dropping it makes safetensors-backed workloads
  fail to load on the baseline path).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sol_execbench.core.data import Definition, Workload

from src.agents.coder import CoderAgent, ImplementationError, KernelCodeOutput
from src.benchmark.baseline_generator import (
    BaselineGenerationError,
    generate_triton_baseline,
)
from src.eval.correctness import (
    CorrectnessGateFailure,
    CorrectnessResult,
    CorrectnessStage,
)
from src.kernels.compiler import CompilationResult
from src.kernels.kernel import KernelSpec, KernelType


def _coder_output(source_code: str, triton_kernel_name: str = "kernel_fn") -> KernelCodeOutput:
    """Test helper: build a KernelCodeOutput without paying the Pydantic
    validator's cost. Most tests here drive the retry/loop control flow
    with placeholder source like ``"good source"`` that wouldn't satisfy
    the ``@triton.jit`` cross-validation."""
    return KernelCodeOutput.model_construct(
        source_code=source_code,
        triton_kernel_name=triton_kernel_name,
    )


# ── fixtures ───────────────────────────────────────────────────────────

def _make_definition(
    name: str = "test_prob",
    reference: str = "def run(x):\n    return x * 2.0\n",
) -> Definition:
    return Definition.model_validate({
        "name": name,
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": reference,
        "op_type": "elementwise",
    })


def _make_workloads(n: int = 3) -> list[Workload]:
    return [
        Workload.model_validate({"uuid": f"wl-{i}", "axes": {"N": 8}, "inputs": {}})
        for i in range(n)
    ]


def _make_spec(name: str = "test_prob", entrypoint: str = "kernel_fn") -> KernelSpec:
    return KernelSpec(
        name=name,
        kernel_type=KernelType.ELEMENTWISE,
        entrypoint=entrypoint,
        pytorch_reference="def run(x):\n    return x * 2.0\n",
    )


def _fail(stage: CorrectnessStage = CorrectnessStage.SMOKE_TEST) -> CorrectnessResult:
    return CorrectnessResult(
        passed=False, failed_stage=stage, error_message="mismatch", max_abs_error=1.0,
    )


def _gate_pass() -> None:
    """run_correctness_gate returns None when every workload passes."""
    return None


def _gate_fail(
    index: int = 0,
    workloads: list | None = None,
    stage: CorrectnessStage = CorrectnessStage.SMOKE_TEST,
) -> CorrectnessGateFailure:
    """run_correctness_gate's first-failure report (verify ran and failed)."""
    wl = workloads[index] if workloads is not None else _make_workloads(index + 1)[index]
    return CorrectnessGateFailure(index=index, workload=wl, result=_fail(stage))


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
            side_effect=lambda definition, workload, **_: lambda seed: (float(seed),),
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
            _make_definition(), _make_spec(), coder=coder, workloads=[],
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
            _make_definition(), _make_spec(), coder=None, workloads=_make_workloads(),
        )


@pytest.mark.asyncio
async def test_coder_without_model_raises_baseline_error():
    """CoderAgent(model=None) has no oracle to bind to — same fail-closed rule."""
    coder = CoderAgent(model=None)
    with pytest.raises(BaselineGenerationError, match="No model"):
        await generate_triton_baseline(
            _make_definition(), _make_spec(), coder=coder, workloads=_make_workloads(),
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
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ) as mock_verify,
    ):
        result = await generate_triton_baseline(
            _make_definition(), spec, coder=coder, workloads=workloads,
            allow_in_parent_fallback=True,
        )

    assert result is not None
    assert result.source_code == "@triton.jit\ndef kernel_fn(x): pass"
    assert result.spec is spec
    assert mock_verify.call_count == 1  # gate runs once per attempt
    coder.translate.assert_awaited_once()


@pytest.mark.asyncio
async def test_verify_uses_all_selected_workloads(patched_io):
    """One input_generator per selected workload; the gate receives every one."""
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
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ) as mock_verify,
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(), coder=coder, workloads=workloads,
            allow_in_parent_fallback=True,
        )

    assert result is not None
    assert mock_build_gen.call_count == 3  # one generator per workload
    assert mock_verify.call_count == 1  # gate runs once per attempt
    gate_args = mock_verify.call_args
    # The gate gets all three generators + all three workloads to walk.
    assert len(gate_args.args[2]) == 3  # input_generators
    assert len(gate_args.args[3]) == 3  # workloads


@pytest.mark.asyncio
async def test_translate_receives_reference_source_and_all_generators(patched_io):
    """translate() gets reference_source, spec, reference_fn, and every selected
    workload's generator — so its correctness tool can catch cross-workload bugs."""
    definition = _make_definition(reference="def run(x):\n    return x + 1\n")
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
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ),
    ):
        await generate_triton_baseline(
            definition, spec, coder=coder, workloads=workloads,
            allow_in_parent_fallback=True,
        )

    kwargs = coder.translate.call_args.kwargs
    assert kwargs["reference_source"] == definition.reference
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
        patch("src.benchmark.baseline_generator.run_correctness_gate", return_value=_gate_pass()),
    ):
        result = await generate_triton_baseline(
            _make_definition(), spec, coder=coder, workloads=workloads,
            allow_in_parent_fallback=True,
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

    # Attempt 1: gate reports workload 1 failed (short-circuit inside the gate).
    # Attempt 2: gate passes (None).
    correctness_sequence = [_gate_fail(index=1, workloads=workloads), _gate_pass()]
    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.run_correctness_gate",
            side_effect=correctness_sequence,
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
            allow_in_parent_fallback=True,
        )

    assert result is not None
    assert result.source_code == "good source"
    assert coder.translate.await_count == 2


@pytest.mark.asyncio
async def test_entrypoint_binding_mismatch_triggers_retry(patched_io):
    """Codex 2026-05-16: LLM declares triton_kernel_name=`other` but the
    host wrapper launches `my_kernel`. Without the binding gate this
    would pass bench+correctness while NCU filters on a never-launched
    kernel. Treated as a post-verify failure: one attempt consumed, the
    retry sees a clean source and succeeds.
    """
    workloads = _make_workloads(n=1)
    misleading_source = (
        "@triton.jit\n"
        "def my_kernel(x): pass\n"
        "\n"
        "@triton.jit\n"
        "def other(x): pass\n"
        "\n"
        "def kernel_fn(x):\n"
        "    my_kernel[(1,)](x)\n"
    )
    good_source = "@triton.jit\ndef kernel_fn(x): pass\n"
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(
        side_effect=[
            _coder_output(misleading_source, triton_kernel_name="other"),
            _coder_output(good_source, triton_kernel_name="kernel_fn"),
        ]
    )

    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
            allow_in_parent_fallback=True,
        )

    assert result.source_code == good_source
    assert coder.translate.await_count == 2
    # Attempt-2 prompt must surface the entrypoint-binding diagnostic so
    # the LLM knows what to fix.
    second_call_kwargs = coder.translate.await_args_list[1].kwargs
    prior_failures = second_call_kwargs["prior_failures"]
    assert len(prior_failures) == 1
    assert any(
        "Entrypoint-binding FAILED" in err for err in prior_failures[0].tool_errors
    )


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
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
            allow_in_parent_fallback=True,
        )

    assert result is not None
    assert result.source_code == "good source"
    assert coder.translate.await_count == 2


@pytest.mark.asyncio
async def test_translate_receives_accumulated_prior_failures(patched_io):
    """Each failed attempt's tool_errors must be carried into the next
    attempt's prior_failures kwarg, in order, cumulative across attempts.
    """
    from src.agents.coder import AttemptFailure

    workloads = _make_workloads(n=1)
    coder = CoderAgent(model=MagicMock())

    captured_calls: list[list[AttemptFailure]] = []

    async def fake_translate(*args, prior_failures=(), **kwargs):
        # Copy so later mutations of the caller's accumulator don't bleed in.
        captured_calls.append(list(prior_failures))
        if len(captured_calls) == 1:
            raise ImplementationError(
                "attempt 1 failed",
                tool_errors=["err1a", "err1b"],
            )
        if len(captured_calls) == 2:
            raise ImplementationError(
                "attempt 2 failed",
                tool_errors=["err2a"],
            )
        return _coder_output("good source")

    coder.translate = fake_translate

    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
            allow_in_parent_fallback=True,
        )

    assert result is not None
    assert len(captured_calls) == 3
    # Attempt 1: empty (no prior failures).
    assert captured_calls[0] == []
    # Attempt 2: one prior failure carrying attempt-1 tool_errors.
    assert len(captured_calls[1]) == 1
    assert captured_calls[1][0].attempt_no == 1
    assert captured_calls[1][0].tool_errors == ["err1a", "err1b"]
    # Attempt 3: cumulative — both attempts 1 and 2.
    assert len(captured_calls[2]) == 2
    assert captured_calls[2][0].attempt_no == 1
    assert captured_calls[2][1].attempt_no == 2
    assert captured_calls[2][1].tool_errors == ["err2a"]


@pytest.mark.asyncio
async def test_post_verify_compile_failure_synthesizes_prior_failure(patched_io):
    """When translate() succeeds but post-verify compile fails, the next
    attempt's prior_failures should carry a synthetic 'Post-verify Compile
    FAILED' entry so the model sees the failure mode."""
    from src.agents.coder import AttemptFailure

    workloads = _make_workloads(n=1)
    coder = CoderAgent(model=MagicMock())

    captured_calls: list[list[AttemptFailure]] = []

    async def fake_translate(*args, prior_failures=(), **kwargs):
        captured_calls.append(list(prior_failures))
        return _coder_output("src")

    coder.translate = fake_translate

    # First compile fails; second succeeds.
    compile_outcomes = iter([_compile_fail(), _compile_ok()])

    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            side_effect=lambda *a, **k: next(compile_outcomes),
        ),
        patch(
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
            allow_in_parent_fallback=True,
        )

    assert result is not None
    assert len(captured_calls) == 2
    assert len(captured_calls[1]) == 1
    failure = captured_calls[1][0]
    assert failure.attempt_no == 1
    assert len(failure.tool_errors) == 1
    assert "Post-verify Compile FAILED" in failure.tool_errors[0]
    assert "SyntaxError: bad" in failure.tool_errors[0]


@pytest.mark.asyncio
async def test_post_verify_correctness_failure_synthesizes_prior_failure(patched_io):
    """When translate() + compile succeed but correctness fails on any
    workload, synthesize a 'Post-verify Correctness FAILED' entry into
    the next attempt's prior_failures."""
    from src.agents.coder import AttemptFailure

    workloads = _make_workloads(n=2)
    coder = CoderAgent(model=MagicMock())

    captured_calls: list[list[AttemptFailure]] = []

    async def fake_translate(*args, prior_failures=(), **kwargs):
        captured_calls.append(list(prior_failures))
        return _coder_output("src")

    coder.translate = fake_translate

    # Attempt 1: gate reports wl 1 failed. Attempt 2: gate passes.
    correctness_outcomes = iter([_gate_fail(index=1, workloads=workloads), _gate_pass()])

    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.run_correctness_gate",
            side_effect=lambda *a, **kw: next(correctness_outcomes),
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
            allow_in_parent_fallback=True,
        )

    assert result is not None
    assert len(captured_calls) == 2
    assert len(captured_calls[1]) == 1
    failure = captured_calls[1][0]
    assert failure.attempt_no == 1
    assert len(failure.tool_errors) == 1
    assert "Post-verify Correctness FAILED" in failure.tool_errors[0]


# ── subprocess-isolated post-verify (Codex P1) ─────────────────────────
#
# When a SOL ``problem_definition_path`` is bound, the LLM baseline's
# post-verify must run in the crash-isolated worker
# (``run_correctness_subprocess``, mode ``gate``) instead of launching the
# untrusted kernel in-parent via ``verify_correctness`` — an out-of-bounds
# LLM baseline would otherwise poison the parent CUDA context before Phase
# B starts. Without a definition_path (unit tests / placeholder runs) the
# in-parent loop is preserved.


def _sub_pass(max_err: float = 0.0, total: int = 1):
    from src.eval.correctness_subprocess import CorrectnessResult as SubResult
    return SubResult(passed=True, max_err=max_err, total_workloads=total)


def _sub_fail(stage: str = "numerical", idx: int = 1, total: int = 1):
    from src.eval.correctness_subprocess import CorrectnessResult as SubResult
    return SubResult(
        passed=False, failed_stage=stage, error_message="mismatch at [0]",
        total_workloads=total, failed_workload_idx=idx,
    )


@pytest.mark.asyncio
async def test_post_verify_uses_subprocess_when_definition_path_set(patched_io):
    """With a definition_path bound, post-verify delegates to the
    crash-isolated worker and NOT to in-parent verify_correctness."""
    spec = _make_spec()
    workloads = _make_workloads(n=3)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(
        return_value=_coder_output("@triton.jit\ndef kernel_fn(x): pass")
    )

    seen = {}

    async def _fake_sub(*, request, worker_dir, timeout_s):
        seen["mode"] = request["mode"]
        seen["definition_path"] = request["definition_path"]
        seen["timeout_s"] = timeout_s
        seen["n_workloads"] = len(request["workloads"])
        return _sub_pass(total=3)

    with (
        patch(
            "src.eval.correctness_subprocess.run_correctness_subprocess",
            side_effect=_fake_sub,
        ),
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.run_correctness_gate",
            side_effect=AssertionError("in-parent correctness gate must not run"),
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(), spec, coder=coder, workloads=workloads,
            problem_definition_path=Path("/problem/definition.json"),
            worker_timeout_s=42.0,
        )

    assert result is not None
    assert seen["mode"] == "gate"
    assert seen["definition_path"] == "/problem/definition.json"
    assert seen["timeout_s"] == 42.0
    assert seen["n_workloads"] == 3


@pytest.mark.asyncio
async def test_post_verify_subprocess_failure_feeds_prior_failures_and_retries(patched_io):
    """A failing subprocess result consumes one attempt, synthesizes a
    Post-verify Correctness FAILED entry into prior_failures, and a retry
    is taken — same control flow as the in-parent path."""
    from src.agents.coder import AttemptFailure

    workloads = _make_workloads(n=2)
    coder = CoderAgent(model=MagicMock())

    captured_calls: list[list[AttemptFailure]] = []

    async def fake_translate(*args, prior_failures=(), **kwargs):
        captured_calls.append(list(prior_failures))
        return _coder_output("src")

    coder.translate = fake_translate

    sub_outcomes = iter([_sub_fail(stage="numerical", idx=2, total=2), _sub_pass(total=2)])

    async def _fake_sub(*, request, worker_dir, timeout_s):
        return next(sub_outcomes)

    with (
        patch(
            "src.eval.correctness_subprocess.run_correctness_subprocess",
            side_effect=_fake_sub,
        ),
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.run_correctness_gate",
            side_effect=AssertionError("in-parent correctness gate must not run"),
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
            problem_definition_path=Path("/problem/definition.json"),
        )

    assert result is not None
    assert len(captured_calls) == 2
    # Attempt 2 carries the synthetic post-verify failure from attempt 1.
    assert len(captured_calls[1]) == 1
    failure = captured_calls[1][0]
    assert failure.attempt_no == 1
    assert "Post-verify Correctness FAILED" in failure.tool_errors[0]
    assert "mismatch at [0]" in failure.tool_errors[0]


@pytest.mark.asyncio
async def test_post_verify_subprocess_crash_consumes_attempt(patched_io):
    """A worker_crashed result (out-of-bounds LLM baseline) is a post-verify
    failure: the parent stays alive (crash was isolated) and the loop
    retries instead of propagating a poisoned context."""
    workloads = _make_workloads(n=1)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(side_effect=[_coder_output("oob"), _coder_output("ok")])

    sub_outcomes = iter([
        _sub_fail(stage="worker_crashed", idx=None, total=1),
        _sub_pass(total=1),
    ])

    async def _fake_sub(*, request, worker_dir, timeout_s):
        return next(sub_outcomes)

    with (
        patch(
            "src.eval.correctness_subprocess.run_correctness_subprocess",
            side_effect=_fake_sub,
        ),
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
            problem_definition_path=Path("/problem/definition.json"),
        )

    assert result is not None
    assert result.source_code == "ok"
    assert coder.translate.await_count == 2


@pytest.mark.asyncio
async def test_post_verify_in_parent_when_no_definition_path(patched_io):
    """No definition_path → keep the existing in-parent verify_correctness
    loop (fallback for tests / placeholder runs with no SOL problem dir)."""
    workloads = _make_workloads(n=2)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(return_value=_coder_output("src"))

    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ) as mock_verify,
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads,
            allow_in_parent_fallback=True,
        )

    assert result is not None
    # In-parent gate ran once for the attempt — no subprocess delegation.
    assert mock_verify.call_count == 1


# ── correctness-isolation trust gate (post-verify) ─────────────────────
#
# Absent a definition_path the post-verify can't crash-isolate the
# candidate launch in a subprocess; launching it in-parent is only safe
# in a deliberately-trusted/mocked context. The three-way gate raises a
# typed ``CorrectnessIsolationError`` unless ``allow_in_parent_fallback``
# opts in — so a dropped path fails loud instead of silently launching an
# untrusted kernel in the parent CUDA context.


@pytest.mark.asyncio
async def test_generate_triton_baseline_raises_without_isolation_or_optin(patched_io):
    """No definition_path + no opt-in → the post-verify gate raises
    CorrectnessIsolationError. translate() is mocked to return a compiling
    candidate so we reach the post-verify gate (not translate's own
    construction guard)."""
    from src.eval.correctness_subprocess import CorrectnessIsolationError

    workloads = _make_workloads(n=1)
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
            "src.benchmark.baseline_generator.run_correctness_gate",
            side_effect=AssertionError("in-parent gate must not run without opt-in"),
        ),
        pytest.raises(CorrectnessIsolationError),
    ):
        await generate_triton_baseline(
            _make_definition(), _make_spec(), coder=coder, workloads=workloads,
            problem_definition_path=None,  # absent
            # allow_in_parent_fallback defaults False
        )


@pytest.mark.asyncio
async def test_generate_triton_baseline_in_parent_with_optin(patched_io):
    """No definition_path but explicit opt-in → the in-parent
    verify_correctness loop runs and a verified baseline is returned."""
    workloads = _make_workloads(n=2)
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
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ) as mock_verify,
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(), coder=coder, workloads=workloads,
            problem_definition_path=None, allow_in_parent_fallback=True,
        )

    assert result is not None
    # In-parent gate ran once for the attempt (the opt-in path).
    assert mock_verify.call_count == 1


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
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ) as mock_verify,
    ):
        result = await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
            allow_in_parent_fallback=True,
        )

    assert result is not None
    assert result.source_code == "good source"
    # The gate never ran on the failed-compile attempt; only the second.
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
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_fail(index=0, workloads=workloads),
        ),
        pytest.raises(BaselineGenerationError, match="3 attempts"),
    ):
        await generate_triton_baseline(
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
            allow_in_parent_fallback=True,
        )

    assert coder.translate.await_count == 3


# ── event emission ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_triton_baseline_emits_attempt_events(tmp_path, patched_io):
    """The retry loop emits one ``baseline_attempt`` per attempt, one
    ``baseline_failure`` per non-final failure, and exactly one
    ``baseline_success`` when a verified candidate is returned.
    """
    import json

    from src.runtime import events

    workloads = _make_workloads(n=1)
    coder = CoderAgent(model=MagicMock())
    # Two transient LLM failures then one good candidate.
    coder.translate = AsyncMock(
        side_effect=[
            ImplementationError("transient 1"),
            ImplementationError("transient 2"),
            _coder_output("@triton.jit\ndef kernel_fn(x): pass"),
        ]
    )

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with (
            patch(
                "src.benchmark.baseline_generator.compile_kernel",
                return_value=_compile_ok(),
            ),
            patch(
                "src.benchmark.baseline_generator.run_correctness_gate",
                return_value=_gate_pass(),
            ),
        ):
            await generate_triton_baseline(
                _make_definition(), _make_spec(),
                coder=coder, workloads=workloads, max_retries=3,
                allow_in_parent_fallback=True,
            )
    finally:
        events.unbind()
        fh.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kinds = [r["kind"] for r in records]
    assert kinds.count("baseline_attempt") == 3
    assert kinds.count("baseline_failure") == 2
    assert kinds.count("baseline_success") == 1
    # Ordering: each attempt comes before its failure/success marker.
    assert kinds[0] == "baseline_attempt"
    assert kinds[-1] == "baseline_success"
    # Failure records carry the ImplementationError name.
    failures = [r for r in records if r["kind"] == "baseline_failure"]
    assert all("ImplementationError" in r["reason"] for r in failures)


# ── blob_roots threading for safetensors workloads ────────────────────


@pytest.mark.asyncio
async def test_blob_roots_forwarded_to_build_input_generator():
    """``_load_sol_problem`` computes ``blob_roots = config.safetensors_blob_roots
    or [problem_dir]`` and must thread it into every input-generator
    construction site. ``generate_triton_baseline`` rebuilds its own
    generators in the baseline path, so it must preserve the kwarg too —
    without it, any safetensors-bearing workload trips
    FileNotFoundError before Phase B can start.

    This test mocks ``build_input_generator`` and asserts the kwarg
    survives the call. The strict spy signature below (keyword-only
    ``blob_roots``) raises TypeError if a future refactor drops the
    kwarg, surfacing the regression on import-time behavior rather than
    on the safetensors filesystem branch.
    """
    workloads = _make_workloads(n=2)
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(return_value=_coder_output("src"))

    fake_roots = [Path("/tmp/fake_blob_root")]
    seen_roots: list[list[Path] | None] = []

    def _spy(definition, workload, *, blob_roots):
        # Strict signature: positional-or-keyword `definition`/`workload`,
        # keyword-only `blob_roots`. If the production code drops the
        # kwarg this raises TypeError, surfacing the bug on import-time
        # behavior rather than on the safetensors filesystem branch.
        seen_roots.append(blob_roots)
        return lambda seed: (workload.uuid, seed)

    with (
        patch(
            "src.benchmark.baseline_generator.build_reference_fn",
            return_value=lambda x: x * 2.0,
        ),
        patch(
            "src.benchmark.baseline_generator.build_input_generator",
            side_effect=_spy,
        ),
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ),
    ):
        await generate_triton_baseline(
            _make_definition(),
            _make_spec(),
            coder=coder,
            workloads=workloads,
            blob_roots=fake_roots,
            allow_in_parent_fallback=True,
        )

    assert len(seen_roots) == 2  # one build per workload
    assert all(r is fake_roots for r in seen_roots)


@pytest.mark.asyncio
async def test_blob_roots_defaults_to_none_when_omitted(patched_io):
    """Default ``blob_roots=None`` keeps the existing call sites working
    so the new kwarg is non-breaking for non-safetensors callers."""
    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(return_value=_coder_output("src"))

    with (
        patch(
            "src.benchmark.baseline_generator.compile_kernel",
            return_value=_compile_ok(),
        ),
        patch(
            "src.benchmark.baseline_generator.run_correctness_gate",
            return_value=_gate_pass(),
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(),
            _make_spec(),
            coder=coder,
            workloads=_make_workloads(n=1),
            allow_in_parent_fallback=True,
        )

    assert result is not None


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_safetensors_workload_resolves_blob_with_roots(monkeypatch):
    """End-to-end GPU regression: a safetensors-bearing workload survives
    the input-generator construction step in ``generate_triton_baseline``
    iff ``blob_roots`` points at the staging directory.

    Without ``blob_roots`` threaded through, ``load_safetensors`` falls
    back to resolving ``"weights.safetensors"`` against CWD and raises
    ``FileNotFoundError``. We swap CWD to a temp directory to guarantee
    the relative path can't accidentally resolve, then run the baseline
    far enough to confirm the input-generator stage succeeds. The Coder
    is mocked to return failing source so the test exits via
    ``BaselineGenerationError`` rather than a fake-success path — what
    matters is that ``FileNotFoundError`` does not surface on the
    safetensors-resolution path.
    """
    import tempfile

    from src.benchmarks.sol_execbench import load as sol_load

    fixture = Path(__file__).parent / "fixtures" / "sol_safetensors"
    definition, workloads = sol_load(fixture)

    spec = KernelSpec(
        name=definition.name,
        kernel_type=KernelType.MATMUL,
        entrypoint="kernel_fn",
        pytorch_reference=definition.reference,
    )

    coder = CoderAgent(model=MagicMock())
    coder.translate = AsyncMock(side_effect=ImplementationError("forced fail"))

    # Make CWD a directory with no weights file so that the broken
    # (no-blob_roots) path provably can't resolve the relative blob.
    with tempfile.TemporaryDirectory() as cwd:
        monkeypatch.chdir(cwd)

        # With the fix, blob_roots steers load_safetensors to the fixture
        # dir; the input-generator construction succeeds and the loop
        # proceeds to translate(), which fails with the mocked
        # ImplementationError → BaselineGenerationError after retries.
        # Without the fix, build_input_generator raises FileNotFoundError
        # *before* ever reaching the retry loop.
        with pytest.raises(BaselineGenerationError, match="failed after"):
            await generate_triton_baseline(
                definition,
                spec,
                coder=coder,
                workloads=workloads,
                max_retries=1,
                blob_roots=[fixture],
            )


import contextlib


class TestBaselineTraceWrap:
    """The baseline coder.translate() call must run inside a trace
    tagged {iter: 0, agent: 'coder-translate'} so the resource
    accumulator can attribute its usage. Tier-1 mocked — verifies the
    consolidated ``trace_span`` helper is invoked with the right kwargs.
    """

    @pytest.mark.asyncio
    async def test_baseline_translate_wrapped_in_acts_baseline_trace(
        self, monkeypatch, patched_io
    ):
        import src.benchmark.baseline_generator as bg
        from src.runtime.usage import AgentLabel

        captured: list[dict] = []

        @contextlib.contextmanager
        def fake_trace_span(workflow_name, **kwargs):
            captured.append({"workflow_name": workflow_name, **kwargs})
            yield

        # Monkey-patch the module-level alias imported from
        # ``src.runtime.sdk_trace``. Patching at the import site rather
        # than the source module ensures the rebinding takes effect for
        # the ``bg`` module's lookup.
        monkeypatch.setattr(bg, "trace_span", fake_trace_span, raising=True)

        workloads = _make_workloads(n=1)
        coder = CoderAgent(model=MagicMock())
        coder.translate = AsyncMock(return_value=_coder_output("good source"))

        with (
            patch(
                "src.benchmark.baseline_generator.compile_kernel",
                return_value=_compile_ok(),
            ),
            patch(
                "src.benchmark.baseline_generator.run_correctness_gate",
                return_value=_gate_pass(),
            ),
        ):
            await generate_triton_baseline(
                _make_definition(),
                _make_spec(),
                coder=coder,
                workloads=workloads,
                max_retries=1,
                allow_in_parent_fallback=True,
            )

        # Exactly one trace wrap — happy path takes one attempt.
        assert len(captured) == 1
        assert captured[0] == {
            "workflow_name": "acts_baseline",
            "iter_no": 0,
            "agent": AgentLabel.CODER_TRANSLATE,
            "attempt": 1,
        }
