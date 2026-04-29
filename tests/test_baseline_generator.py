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
  (regression test for Codex G2: prior code dropped this kwarg in
  ``generate_triton_baseline``'s rebuild path even though
  ``_load_sol_problem`` threaded it into the search-loop generators).
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
from src.eval.correctness import CorrectnessResult, CorrectnessStage
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
            "src.benchmark.baseline_generator.verify_correctness",
            return_value=_pass(),
        ) as mock_verify,
    ):
        result = await generate_triton_baseline(
            _make_definition(), spec, coder=coder, workloads=workloads,
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
            _make_definition(), _make_spec(), coder=coder, workloads=workloads,
        )

    assert result is not None
    assert mock_build_gen.call_count == 3  # one generator per workload
    assert mock_verify.call_count == 3


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
            "src.benchmark.baseline_generator.verify_correctness",
            return_value=_pass(),
        ),
    ):
        await generate_triton_baseline(
            definition, spec, coder=coder, workloads=workloads,
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
        patch("src.benchmark.baseline_generator.verify_correctness", return_value=_pass()),
    ):
        result = await generate_triton_baseline(
            _make_definition(), spec, coder=coder, workloads=workloads,
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
            _make_definition(), _make_spec(),
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
            _make_definition(), _make_spec(),
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
            _make_definition(), _make_spec(),
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
            _make_definition(), _make_spec(),
            coder=coder, workloads=workloads, max_retries=3,
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
                "src.benchmark.baseline_generator.verify_correctness",
                return_value=_pass(),
            ),
        ):
            await generate_triton_baseline(
                _make_definition(), _make_spec(),
                coder=coder, workloads=workloads, max_retries=3,
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


# ── G2 regression: blob_roots threading for safetensors workloads ─────


@pytest.mark.asyncio
async def test_blob_roots_forwarded_to_build_input_generator():
    """Regression for Codex G2 (P1-2).

    ``_load_sol_problem`` computes ``blob_roots = config.safetensors_blob_roots
    or [problem_dir]`` and must thread it into every input-generator
    construction site. The earlier fix only patched the search-loop site;
    ``generate_triton_baseline`` rebuilt its own generators and dropped
    the kwarg, so any safetensors-bearing workload tripped a
    FileNotFoundError before Phase B could start.

    This test mocks ``build_input_generator`` and asserts the kwarg
    survives the call so the missing-kwarg form (which used to raise
    TypeError under the strict-spy below) cannot regress silently.
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
            "src.benchmark.baseline_generator.verify_correctness",
            return_value=_pass(),
        ),
    ):
        await generate_triton_baseline(
            _make_definition(),
            _make_spec(),
            coder=coder,
            workloads=workloads,
            blob_roots=fake_roots,
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
            "src.benchmark.baseline_generator.verify_correctness",
            return_value=_pass(),
        ),
    ):
        result = await generate_triton_baseline(
            _make_definition(),
            _make_spec(),
            coder=coder,
            workloads=_make_workloads(n=1),
        )

    assert result is not None


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_safetensors_workload_resolves_blob_with_roots(monkeypatch):
    """End-to-end GPU regression: a safetensors-bearing workload survives
    the input-generator construction step in ``generate_triton_baseline``
    iff ``blob_roots`` points at the staging directory.

    On the broken pre-fix code (no kwarg threading), this raises
    ``FileNotFoundError`` at line 66 because ``load_safetensors`` falls
    back to resolving ``"weights.safetensors"`` against CWD. We swap CWD
    to a temp directory to guarantee the relative path can't accidentally
    resolve, then run the baseline far enough to confirm the input-
    generator stage succeeds. The Coder is mocked to return failing
    source so the test exits via ``BaselineGenerationError`` rather than
    a fake-success path — what matters is that the FileNotFoundError
    from the pre-fix code never surfaces.
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
