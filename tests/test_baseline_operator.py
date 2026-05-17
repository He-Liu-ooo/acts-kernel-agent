"""Tier 1 tests for operator-supplied Triton baseline.

Covers cfg schema, load_operator_baseline gate sequence, and pipeline
dispatch. Mocked compile / correctness — runs in the torchless venv.
See doc/specs/2026-05-16-operator-supplied-triton-baseline-design.md.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest

from src.config import ACTSConfig, load_config


# ── cfg schema ──────────────────────────────────────────────────────


def _write_cfg(tmp_path: Path, body: str) -> Path:
    cfg_path = tmp_path / "test.cfg"
    cfg_path.write_text(body)
    return cfg_path


def _write_baseline(tmp_path: Path, source: str = "# stub\n") -> Path:
    p = tmp_path / "baseline.py"
    p.write_text(source)
    return p


def test_acts_config_defaults_for_operator_baseline_fields():
    cfg = ACTSConfig()
    assert cfg.use_operator_baseline is False
    assert cfg.triton_baseline_path is None
    assert cfg.triton_baseline_dps is False
    assert cfg.triton_baseline_kernel_name is None
    assert cfg.triton_baseline_enforce_autotune is False


def test_load_config_roundtrips_operator_baseline_fields(tmp_path):
    baseline = _write_baseline(tmp_path)
    cfg_body = textwrap.dedent(f"""\
        runtime: {{
            problem_path = "placeholder";
            use_operator_baseline = true;
            triton_baseline_path = "{baseline}";
            triton_baseline_dps = true;
            triton_baseline_kernel_name = "matmul_kernel";
            triton_baseline_enforce_autotune = true;
        }};
    """)
    cfg = load_config(_write_cfg(tmp_path, cfg_body))
    assert cfg.use_operator_baseline is True
    assert cfg.triton_baseline_path == str(baseline)
    assert cfg.triton_baseline_dps is True
    assert cfg.triton_baseline_kernel_name == "matmul_kernel"
    assert cfg.triton_baseline_enforce_autotune is True


def test_load_config_raises_when_flag_true_but_path_empty(tmp_path):
    cfg_body = textwrap.dedent("""\
        runtime: {
            problem_path = "placeholder";
            use_operator_baseline = true;
        };
    """)
    with pytest.raises(ValueError, match="use_operator_baseline.*triton_baseline_path"):
        load_config(_write_cfg(tmp_path, cfg_body))


def test_load_config_raises_when_baseline_file_missing(tmp_path):
    missing = tmp_path / "no_such.py"
    cfg_body = textwrap.dedent(f"""\
        runtime: {{
            problem_path = "placeholder";
            use_operator_baseline = true;
            triton_baseline_path = "{missing}";
        }};
    """)
    with pytest.raises(FileNotFoundError, match="triton_baseline_path"):
        load_config(_write_cfg(tmp_path, cfg_body))


def test_post_init_raises_when_flag_true_path_empty():
    with pytest.raises(ValueError, match="use_operator_baseline.*triton_baseline_path"):
        ACTSConfig(use_operator_baseline=True)


def test_post_init_warns_on_stray_path_without_flag(caplog, tmp_path):
    caplog.set_level(logging.WARNING, logger="src.config")
    ACTSConfig(triton_baseline_path=str(_write_baseline(tmp_path)))
    assert any(
        "triton_baseline_path" in r.message and "dead config" in r.message
        for r in caplog.records
    )


def test_post_init_warns_on_stray_dps_without_flag(caplog):
    caplog.set_level(logging.WARNING, logger="src.config")
    ACTSConfig(triton_baseline_dps=True)
    assert any(
        "triton_baseline_dps" in r.message and "dead config" in r.message
        for r in caplog.records
    )


def test_post_init_warns_on_stray_kernel_name_without_flag(caplog):
    caplog.set_level(logging.WARNING, logger="src.config")
    ACTSConfig(triton_baseline_kernel_name="foo")
    assert any(
        "triton_baseline_kernel_name" in r.message and "dead config" in r.message
        for r in caplog.records
    )


def test_post_init_warns_on_stray_enforce_without_flag(caplog):
    caplog.set_level(logging.WARNING, logger="src.config")
    ACTSConfig(triton_baseline_enforce_autotune=True)
    assert any(
        "triton_baseline_enforce_autotune" in r.message and "dead config" in r.message
        for r in caplog.records
    )


def test_post_init_silent_when_all_defaults(caplog):
    caplog.set_level(logging.WARNING, logger="src.config")
    ACTSConfig()
    assert not any("triton_baseline" in r.message for r in caplog.records)


def test_post_init_silent_when_flag_true_path_set(caplog, tmp_path):
    caplog.set_level(logging.WARNING, logger="src.config")
    ACTSConfig(
        use_operator_baseline=True,
        triton_baseline_path=str(_write_baseline(tmp_path)),
        triton_baseline_dps=True,
        triton_baseline_enforce_autotune=True,
    )
    assert not any("dead config" in r.message for r in caplog.records)


# ── event kinds ─────────────────────────────────────────────────────


def test_operator_baseline_event_kinds_registered():
    from src.runtime.events import CORE_EVENT_KINDS
    assert "operator_baseline_load" in CORE_EVENT_KINDS
    assert "operator_baseline_success" in CORE_EVENT_KINDS
    assert "operator_baseline_failure" in CORE_EVENT_KINDS


# ── load_operator_baseline ──────────────────────────────────────────

from unittest.mock import patch

from sol_execbench.core.data import Definition, Workload

from src.benchmark.baseline_generator import (
    BaselineGenerationError,
    load_operator_baseline,
)
from src.eval.correctness import CorrectnessResult, CorrectnessStage
from src.kernels.compiler import CompilationResult
from src.kernels.kernel import KernelSpec, KernelType


_AUTOTUNE_KERNEL = """\
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}, num_warps=4),
        triton.Config({'BLOCK': 128}, num_warps=4),
        triton.Config({'BLOCK': 256}, num_warps=8),
        triton.Config({'BLOCK': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def my_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x * 2.0, mask=mask)
"""

_NO_AUTOTUNE_KERNEL = """\
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x * 2.0, mask=mask)
"""

_MULTI_KERNEL = _NO_AUTOTUNE_KERNEL + """

@triton.jit
def other_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + offs, tl.load(x_ptr + offs))
"""

# Multi-kernel where the host wrapper kernel_fn ACTUALLY launches
# other_kernel — proves the override picks a function that drives
# computation, not just a symbol that happens to appear in source.
# Used by test_loader_accepts_multi_kernel_with_override (positive)
# and as a contrast to _MULTI_KERNEL_MISLEADING_OVERRIDE (negative).
_MULTI_KERNEL_ENTRYPOINT_LAUNCHES_OTHER = _MULTI_KERNEL + """

def kernel_fn(x, y, N):
    BLOCK = 64
    other_kernel[(1,)](x, y, N, BLOCK)
"""

# Multi-kernel where kernel_fn launches my_kernel; override naming
# other_kernel would mis-attribute. Used by the negative entrypoint-
# binding test.
_MULTI_KERNEL_ENTRYPOINT_LAUNCHES_MY = _MULTI_KERNEL + """

def kernel_fn(x, y, N):
    BLOCK = 64
    my_kernel[(1,)](x, y, N, BLOCK)
"""


def _make_definition() -> Definition:
    return Definition.model_validate({
        "name": "operator_test",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x * 2.0\n",
        "op_type": "elementwise",
    })


def _make_workloads(n: int = 3) -> list[Workload]:
    return [
        Workload.model_validate({"uuid": f"wl-{i}", "axes": {"N": 8}, "inputs": {}})
        for i in range(n)
    ]


def _make_spec(entrypoint: str = "my_kernel") -> KernelSpec:
    """Default entrypoint='my_kernel' matches the direct-launch convention
    used by the single-kernel fixtures (_NO_AUTOTUNE_KERNEL / _AUTOTUNE_KERNEL),
    where the @triton.jit def IS the entrypoint. Multi-kernel fixtures
    define a separate `kernel_fn` host wrapper; pass entrypoint='kernel_fn'.
    """
    return KernelSpec(
        name="operator_test",
        kernel_type=KernelType.ELEMENTWISE,
        entrypoint=entrypoint,
        pytorch_reference="def run(x):\n    return x * 2.0\n",
    )


def _pass() -> CorrectnessResult:
    return CorrectnessResult(passed=True, max_abs_error=0.0)


def _fail() -> CorrectnessResult:
    return CorrectnessResult(
        passed=False,
        failed_stage=CorrectnessStage.SMOKE_TEST,
        error_message="mismatch",
        max_abs_error=1.0,
    )


def _compile_ok() -> CompilationResult:
    return CompilationResult(success=True, compiled_fn=lambda x: x * 2.0)


def _compile_fail() -> CompilationResult:
    return CompilationResult(success=False, error_message="SyntaxError: bad")


@pytest.fixture
def patched_io():
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


@pytest.mark.asyncio
async def test_loader_happy_path_single_kernel_no_autotune(tmp_path, patched_io):
    path = tmp_path / "kernel.py"
    path.write_text(_NO_AUTOTUNE_KERNEL)
    with (
        patch("src.benchmark.baseline_generator.compile_kernel", return_value=_compile_ok()),
        patch("src.benchmark.baseline_generator.verify_correctness", return_value=_pass()),
    ):
        kernel = await load_operator_baseline(
            _make_definition(), _make_spec(),
            path=path, dps=False, kernel_name_override=None,
            enforce_autotune=False, workloads=_make_workloads(2),
        )
    assert kernel.triton_kernel_name == "my_kernel"
    assert kernel.source_code == _NO_AUTOTUNE_KERNEL
    assert kernel.dps is False


@pytest.mark.asyncio
async def test_loader_happy_path_with_autotune_enforced(tmp_path, patched_io):
    path = tmp_path / "kernel.py"
    path.write_text(_AUTOTUNE_KERNEL)
    with (
        patch("src.benchmark.baseline_generator.compile_kernel", return_value=_compile_ok()),
        patch("src.benchmark.baseline_generator.verify_correctness", return_value=_pass()),
    ):
        kernel = await load_operator_baseline(
            _make_definition(), _make_spec(),
            path=path, dps=False, kernel_name_override=None,
            enforce_autotune=True, workloads=_make_workloads(1),
        )
    assert kernel.triton_kernel_name == "my_kernel"


@pytest.mark.asyncio
async def test_loader_rejects_empty_file(tmp_path, patched_io):
    path = tmp_path / "empty.py"
    path.write_text("   \n  \n")
    with pytest.raises(BaselineGenerationError, match=r"\[empty\]"):
        await load_operator_baseline(
            _make_definition(), _make_spec(),
            path=path, dps=False, kernel_name_override=None,
            enforce_autotune=False, workloads=_make_workloads(),
        )


@pytest.mark.asyncio
async def test_loader_rejects_missing_file(tmp_path, patched_io):
    path = tmp_path / "no_such.py"
    with pytest.raises(BaselineGenerationError, match=r"\[missing_file\]"):
        await load_operator_baseline(
            _make_definition(), _make_spec(),
            path=path, dps=False, kernel_name_override=None,
            enforce_autotune=False, workloads=_make_workloads(),
        )


@pytest.mark.asyncio
async def test_loader_rejects_no_triton_jit_def(tmp_path, patched_io):
    path = tmp_path / "kernel.py"
    path.write_text("def not_a_kernel(): pass\n")
    with pytest.raises(BaselineGenerationError, match=r"\[name_resolve\]"):
        await load_operator_baseline(
            _make_definition(), _make_spec(),
            path=path, dps=False, kernel_name_override=None,
            enforce_autotune=False, workloads=_make_workloads(),
        )


@pytest.mark.asyncio
async def test_loader_rejects_multi_kernel_without_override(tmp_path, patched_io):
    path = tmp_path / "kernel.py"
    path.write_text(_MULTI_KERNEL)
    with pytest.raises(
        BaselineGenerationError,
        match=r"my_kernel.*other_kernel|other_kernel.*my_kernel",
    ):
        await load_operator_baseline(
            _make_definition(), _make_spec(),
            path=path, dps=False, kernel_name_override=None,
            enforce_autotune=False, workloads=_make_workloads(),
        )


@pytest.mark.asyncio
async def test_loader_accepts_multi_kernel_with_override(tmp_path, patched_io):
    """Override names other_kernel AND the host wrapper actually launches it."""
    path = tmp_path / "kernel.py"
    path.write_text(_MULTI_KERNEL_ENTRYPOINT_LAUNCHES_OTHER)
    with (
        patch("src.benchmark.baseline_generator.compile_kernel", return_value=_compile_ok()),
        patch("src.benchmark.baseline_generator.verify_correctness", return_value=_pass()),
    ):
        kernel = await load_operator_baseline(
            _make_definition(), _make_spec(entrypoint="kernel_fn"),
            path=path, dps=False, kernel_name_override="other_kernel",
            enforce_autotune=False, workloads=_make_workloads(1),
        )
    assert kernel.triton_kernel_name == "other_kernel"


@pytest.mark.asyncio
async def test_loader_rejects_override_when_entrypoint_launches_other_kernel(
    tmp_path, patched_io,
):
    """Codex 2026-05-16: override names other_kernel but kernel_fn launches my_kernel.

    Without the entrypoint-binding gate this would pass correctness while
    NCU filters on a never-launched kernel — silent profiler/autotune
    mis-attribution.
    """
    path = tmp_path / "kernel.py"
    path.write_text(_MULTI_KERNEL_ENTRYPOINT_LAUNCHES_MY)
    with pytest.raises(
        BaselineGenerationError,
        match=r"\[name_resolve\].*does not launch @triton\.jit def 'other_kernel'",
    ):
        await load_operator_baseline(
            _make_definition(), _make_spec(entrypoint="kernel_fn"),
            path=path, dps=False, kernel_name_override="other_kernel",
            enforce_autotune=False, workloads=_make_workloads(1),
        )


# The binding check requires launch-position match, not any reference.
# Fixtures below cover the three accepted patterns + the rejected
# "mentioned but not launched" case.

_MULTI_KERNEL_DOT_RUN = _MULTI_KERNEL + """

def kernel_fn(x, y, N):
    BLOCK = 64
    other_kernel.run((1,), x, y, N, BLOCK)
"""

_MULTI_KERNEL_ALIAS_LAUNCH = _MULTI_KERNEL + """

def kernel_fn(x, y, N):
    fn = other_kernel
    BLOCK = 64
    fn[(1,)](x, y, N, BLOCK)
"""

# The Codex P2 pathological case: declared name appears in the
# entrypoint body but only as a non-launch reference; a DIFFERENT kernel
# actually launches. Pre-tightening this slipped through; post-tightening
# this must be rejected.
_MULTI_KERNEL_MENTIONED_BUT_NOT_LAUNCHED = _MULTI_KERNEL + """

def kernel_fn(x, y, N):
    unused = other_kernel  # mentioned but never launched
    BLOCK = 64
    my_kernel[(1,)](x, y, N, BLOCK)
"""


@pytest.mark.asyncio
async def test_loader_accepts_override_via_dot_run_launch(tmp_path, patched_io):
    """Pattern 2: jit_name.run(grid, args) is a valid launch position."""
    path = tmp_path / "kernel.py"
    path.write_text(_MULTI_KERNEL_DOT_RUN)
    with (
        patch("src.benchmark.baseline_generator.compile_kernel", return_value=_compile_ok()),
        patch("src.benchmark.baseline_generator.verify_correctness", return_value=_pass()),
    ):
        kernel = await load_operator_baseline(
            _make_definition(), _make_spec(entrypoint="kernel_fn"),
            path=path, dps=False, kernel_name_override="other_kernel",
            enforce_autotune=False, workloads=_make_workloads(1),
        )
    assert kernel.triton_kernel_name == "other_kernel"


@pytest.mark.asyncio
async def test_loader_accepts_override_via_single_level_alias(tmp_path, patched_io):
    """Pattern 3: <alias> = jit_name; <alias>[grid](...) is accepted."""
    path = tmp_path / "kernel.py"
    path.write_text(_MULTI_KERNEL_ALIAS_LAUNCH)
    with (
        patch("src.benchmark.baseline_generator.compile_kernel", return_value=_compile_ok()),
        patch("src.benchmark.baseline_generator.verify_correctness", return_value=_pass()),
    ):
        kernel = await load_operator_baseline(
            _make_definition(), _make_spec(entrypoint="kernel_fn"),
            path=path, dps=False, kernel_name_override="other_kernel",
            enforce_autotune=False, workloads=_make_workloads(1),
        )
    assert kernel.triton_kernel_name == "other_kernel"


@pytest.mark.asyncio
async def test_loader_rejects_override_mentioned_but_not_launched(
    tmp_path, patched_io,
):
    """Codex P2: declared name appears in body but only as a passing
    reference; a different kernel actually launches. Must reject.
    """
    path = tmp_path / "kernel.py"
    path.write_text(_MULTI_KERNEL_MENTIONED_BUT_NOT_LAUNCHED)
    with pytest.raises(
        BaselineGenerationError,
        match=r"\[name_resolve\].*does not launch @triton\.jit def 'other_kernel'",
    ):
        await load_operator_baseline(
            _make_definition(), _make_spec(entrypoint="kernel_fn"),
            path=path, dps=False, kernel_name_override="other_kernel",
            enforce_autotune=False, workloads=_make_workloads(1),
        )


@pytest.mark.asyncio
async def test_loader_rejects_unknown_override_name(tmp_path, patched_io):
    path = tmp_path / "kernel.py"
    path.write_text(_NO_AUTOTUNE_KERNEL)
    with pytest.raises(
        BaselineGenerationError, match=r"\[name_resolve\].*not_in_source"
    ):
        await load_operator_baseline(
            _make_definition(), _make_spec(),
            path=path, dps=False, kernel_name_override="not_in_source",
            enforce_autotune=False, workloads=_make_workloads(),
        )


@pytest.mark.asyncio
async def test_loader_rejects_missing_autotune_when_enforced(tmp_path, patched_io):
    path = tmp_path / "kernel.py"
    path.write_text(_NO_AUTOTUNE_KERNEL)
    with pytest.raises(BaselineGenerationError, match=r"\[autotune_validate\]"):
        await load_operator_baseline(
            _make_definition(), _make_spec(),
            path=path, dps=False, kernel_name_override=None,
            enforce_autotune=True, workloads=_make_workloads(),
        )


@pytest.mark.asyncio
async def test_loader_compile_failure(tmp_path, patched_io):
    path = tmp_path / "kernel.py"
    path.write_text(_NO_AUTOTUNE_KERNEL)
    with patch(
        "src.benchmark.baseline_generator.compile_kernel",
        return_value=_compile_fail(),
    ):
        with pytest.raises(BaselineGenerationError, match=r"\[compile\]"):
            await load_operator_baseline(
                _make_definition(), _make_spec(),
                path=path, dps=False, kernel_name_override=None,
                enforce_autotune=False, workloads=_make_workloads(),
            )


@pytest.mark.asyncio
async def test_loader_correctness_failure_first_workload_wins(tmp_path, patched_io):
    path = tmp_path / "kernel.py"
    path.write_text(_NO_AUTOTUNE_KERNEL)
    with (
        patch("src.benchmark.baseline_generator.compile_kernel", return_value=_compile_ok()),
        patch(
            "src.benchmark.baseline_generator.verify_correctness",
            side_effect=[_pass(), _fail(), _pass()],
        ),
    ):
        with pytest.raises(BaselineGenerationError, match=r"\[correctness wl 2/3\]"):
            await load_operator_baseline(
                _make_definition(), _make_spec(),
                path=path, dps=False, kernel_name_override=None,
                enforce_autotune=False, workloads=_make_workloads(3),
            )


@pytest.mark.asyncio
async def test_loader_emits_events_on_success(tmp_path, patched_io):
    path = tmp_path / "kernel.py"
    path.write_text(_NO_AUTOTUNE_KERNEL)
    events: list[tuple[str, dict]] = []

    def _capture(kind, **fields):
        events.append((kind, fields))

    with (
        patch("src.benchmark.baseline_generator.compile_kernel", return_value=_compile_ok()),
        patch("src.benchmark.baseline_generator.verify_correctness", return_value=_pass()),
        patch("src.benchmark.baseline_generator.emit", side_effect=_capture),
    ):
        await load_operator_baseline(
            _make_definition(), _make_spec(),
            path=path, dps=False, kernel_name_override=None,
            enforce_autotune=False, workloads=_make_workloads(1),
        )
    kinds = [k for k, _ in events]
    assert kinds == ["operator_baseline_load", "operator_baseline_success"]
    load_payload = events[0][1]
    assert load_payload["kernel_name"] == "my_kernel"
    assert load_payload["dps"] is False
    assert load_payload["enforce_autotune"] is False


@pytest.mark.asyncio
async def test_loader_emits_failure_event_with_stage(tmp_path, patched_io):
    path = tmp_path / "kernel.py"
    path.write_text(_NO_AUTOTUNE_KERNEL)
    events: list[tuple[str, dict]] = []

    def _capture(kind, **fields):
        events.append((kind, fields))

    with (
        patch("src.benchmark.baseline_generator.compile_kernel", return_value=_compile_fail()),
        patch("src.benchmark.baseline_generator.emit", side_effect=_capture),
    ):
        with pytest.raises(BaselineGenerationError):
            await load_operator_baseline(
                _make_definition(), _make_spec(),
                path=path, dps=False, kernel_name_override=None,
                enforce_autotune=False, workloads=_make_workloads(),
            )
    kinds_and_stages = [(k, f.get("stage")) for k, f in events]
    assert ("operator_baseline_load", None) in kinds_and_stages
    assert ("operator_baseline_failure", "compile") in kinds_and_stages


# ── pipeline dispatch ───────────────────────────────────────────────

from unittest.mock import AsyncMock

from src.kernels.kernel import Kernel


def _stub_kernel(spec) -> Kernel:
    return Kernel(
        spec=spec,
        source_code=_NO_AUTOTUNE_KERNEL,
        triton_kernel_name="my_kernel",
        dps=False,
    )


_PATCH_OPERATOR = "src.benchmark.baseline_generator.load_operator_baseline"
_PATCH_GENERATOR = "src.benchmark.baseline_generator.generate_triton_baseline"


@pytest.mark.asyncio
async def test_dispatch_picks_operator_path_when_flag_true(tmp_path):
    from src.pipeline import optimize
    baseline = tmp_path / "k.py"
    baseline.write_text(_NO_AUTOTUNE_KERNEL)
    config = ACTSConfig(
        use_operator_baseline=True,
        triton_baseline_path=str(baseline),
        triton_baseline_dps=False,
    )
    spec = _make_spec()
    operator_mock = AsyncMock(return_value=_stub_kernel(spec))
    generator_mock = AsyncMock(return_value=_stub_kernel(spec))
    with (
        patch(_PATCH_OPERATOR, operator_mock),
        patch(_PATCH_GENERATOR, generator_mock),
    ):
        result = await optimize._dispatch_baseline(
            config, _make_definition(), spec,
            coder=None, workloads=_make_workloads(1), blob_roots=[tmp_path],
        )
    operator_mock.assert_awaited_once()
    generator_mock.assert_not_awaited()
    assert result.triton_kernel_name == "my_kernel"


@pytest.mark.asyncio
async def test_dispatch_picks_llm_path_when_flag_false(tmp_path):
    from src.pipeline import optimize
    config = ACTSConfig()  # use_operator_baseline defaults to False
    spec = _make_spec()
    operator_mock = AsyncMock(return_value=_stub_kernel(spec))
    generator_mock = AsyncMock(return_value=_stub_kernel(spec))
    with (
        patch(_PATCH_OPERATOR, operator_mock),
        patch(_PATCH_GENERATOR, generator_mock),
    ):
        await optimize._dispatch_baseline(
            config, _make_definition(), spec,
            coder=None, workloads=_make_workloads(1), blob_roots=[tmp_path],
        )
    operator_mock.assert_not_awaited()
    generator_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_picks_llm_path_when_flag_false_even_with_stray_path(tmp_path, caplog):
    """Flag is the single source of truth; stray path is dead config (warned)."""
    from src.pipeline import optimize
    caplog.set_level(logging.WARNING, logger="src.config")
    baseline = tmp_path / "k.py"
    baseline.write_text(_NO_AUTOTUNE_KERNEL)
    config = ACTSConfig(
        use_operator_baseline=False,
        triton_baseline_path=str(baseline),  # stray — warned, ignored for dispatch
    )
    spec = _make_spec()
    operator_mock = AsyncMock(return_value=_stub_kernel(spec))
    generator_mock = AsyncMock(return_value=_stub_kernel(spec))
    with (
        patch(_PATCH_OPERATOR, operator_mock),
        patch(_PATCH_GENERATOR, generator_mock),
    ):
        await optimize._dispatch_baseline(
            config, _make_definition(), spec,
            coder=None, workloads=_make_workloads(1), blob_roots=[tmp_path],
        )
    operator_mock.assert_not_awaited()
    generator_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_propagates_dps_and_override(tmp_path):
    from src.pipeline import optimize
    baseline = tmp_path / "k.py"
    baseline.write_text(_MULTI_KERNEL)
    config = ACTSConfig(
        use_operator_baseline=True,
        triton_baseline_path=str(baseline),
        triton_baseline_dps=True,
        triton_baseline_kernel_name="other_kernel",
        triton_baseline_enforce_autotune=False,
    )
    spec = _make_spec()
    operator_mock = AsyncMock(return_value=_stub_kernel(spec))
    with patch(_PATCH_OPERATOR, operator_mock):
        await optimize._dispatch_baseline(
            config, _make_definition(), spec,
            coder=None, workloads=_make_workloads(1), blob_roots=[tmp_path],
        )
    call_kwargs = operator_mock.await_args.kwargs
    assert call_kwargs["dps"] is True
    assert call_kwargs["kernel_name_override"] == "other_kernel"
    assert call_kwargs["enforce_autotune"] is False
    assert call_kwargs["path"] == Path(str(baseline)).resolve()
