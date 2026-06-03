"""Tests for agents/coder.py — Coder agent with tool-using LLM loop.

These tests exercise the tool factories and the `implement()` flow
without requiring `torch` or the OpenAI Agents SDK. The factory
closures delegate to `src.kernels.compiler.compile_kernel` (already
covered by `test_compiler.py`) and `src.eval.correctness.verify_correctness`
(covered by `test_correctness.py`), so tests here focus on wiring and
error-string shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.coder import (
    AttemptFailure,
    CoderAgent,
    ImplementationError,
    KernelCodeOutput,
    _make_compile_tool,
    _make_correctness_tool,
)
from src.agents.planner import OptimizationPlan
from src.config import ACTSConfig
from tests.conftest import (
    ScalarPolicy as _ScalarPolicy,
    make_kernel_spec as _make_spec,
    scalar_gen as _gen,
    scalar_ref as _ref,
)


# ── test helpers for the submit-tool flow ──────────────────────────────
#
# The Coder routes its final answer through a ``submit_kernel`` tool
# call (DeepSeek-reasoner compatibility), not Pydantic ``output_type=``
# enforcement. To simulate the LLM calling that tool inside the SDK
# loop, tests pair a synthetic ``Agent`` factory that captures the
# constructed tools list with a synthetic ``run_agent`` that finds
# ``submit_kernel`` in that list and invokes it directly.


def _simulate_submission(source_code: str, triton_kernel_name: str):
    """Return ``(capture_agent, fake_run_agent)`` patch side-effects that
    simulate the LLM calling ``submit_kernel(source_code, triton_kernel_name)``
    once during the agent run.

    The captured tools list survives across both side-effects via closure
    so the second side-effect can find ``submit_kernel`` even though the
    test patches ``Agent`` itself out (so ``agent.tools`` on the mock isn't
    populated).
    """
    captured_tools: list[list] = []

    def capture_agent(*args, **kwargs):
        captured_tools.append(kwargs.get("tools", []))
        return MagicMock()

    async def fake_run_agent(agent, prompt, **kwargs):
        for tool in captured_tools[-1]:
            if getattr(tool, "__name__", "") == "submit_kernel":
                tool(
                    source_code=source_code,
                    triton_kernel_name=triton_kernel_name,
                )
                break
        # The SDK's ``RunResult.final_output`` is a plain text confirmation;
        # coder.py reads from the captured submission via the tool call, not
        # from ``result.final_output``, so its content is irrelevant.
        return MagicMock(final_output="done")

    return capture_agent, fake_run_agent


# A1 PR 1: every Coder-emitted source must carry @triton.autotune with
# >=4 configs + non-empty key=. ``_VALID_SOURCE`` is the canonical
# "accept" shape used by every test that wants the model to validate
# successfully; the autotune block stays present at module-import time
# so subsequent test-bodies don't have to repeat it.
_VALID_SOURCE = """\
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=4),
    ],
    key=["M"],
)
@triton.jit
def k(X, M, BLOCK_M: tl.constexpr):
    pass
"""
_VALID_NAME = "k"


# ── Cross-attempt memory foundation types ──────────────────────────────


def test_attempt_failure_dataclass_is_frozen():
    af = AttemptFailure(attempt_no=1, tool_errors=["err1", "err2"])
    assert af.attempt_no == 1
    assert af.tool_errors == ["err1", "err2"]
    with pytest.raises(Exception):  # FrozenInstanceError
        af.attempt_no = 2  # type: ignore[misc]


def test_attempt_failure_default_tool_errors_is_empty_list():
    af = AttemptFailure(attempt_no=3)
    assert af.attempt_no == 3
    assert af.tool_errors == []


def test_implementation_error_default_tool_errors_is_empty():
    err = ImplementationError("oops")
    assert err.tool_errors == []
    assert str(err) == "oops"


def test_implementation_error_carries_tool_errors_kwarg():
    err = ImplementationError("budget exhausted", tool_errors=["e1", "e2", "e3"])
    assert err.tool_errors == ["e1", "e2", "e3"]
    assert str(err) == "budget exhausted"


# ── Pydantic output model ──────────────────────────────────────────────


def test_output_model_accepts_valid_data():
    out = KernelCodeOutput(
        source_code=_VALID_SOURCE,
        triton_kernel_name="k",
    )
    assert "@triton.autotune" in out.source_code
    assert out.triton_kernel_name == "k"


def test_output_model_requires_source_code():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KernelCodeOutput()  # type: ignore[call-arg]


def test_output_model_requires_triton_kernel_name():
    """T4: Coder must declare which @triton.jit kernel NCU should profile.
    Empty / missing triton_kernel_name is a Pydantic validation failure
    so the SDK's tool loop retries within the existing turn budget."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KernelCodeOutput(source_code="@triton.jit\ndef k(): pass")  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="required"):
        KernelCodeOutput(
            source_code="@triton.jit\ndef k(): pass",
            triton_kernel_name="",
        )


def test_output_model_rejects_kernel_name_not_in_source():
    """The declared kernel name must literally appear in source_code as
    ``@triton.jit\\ndef <name>``. Mismatch → silent NCU mis-profile in
    production, so we surface it as a validation failure."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="not found"):
        KernelCodeOutput(
            source_code="@triton.jit\ndef actual_name(): pass",
            triton_kernel_name="claimed_name",
        )


def test_output_model_rejects_source_with_no_triton_jit():
    """The Coder writes Triton kernels — pure-PyTorch source means it
    skipped its job. Reject before the kernel reaches the orchestrator."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="@triton.jit"):
        KernelCodeOutput(
            source_code="def run(x):\n    return x * 2.0\n",
            triton_kernel_name="run",
        )


def test_output_model_accepts_multiple_jit_defs_with_matching_name():
    """Fused kernels can declare ``@triton.jit`` helpers alongside the main
    kernel. The Coder picks the dominant-work kernel; we only verify the
    declared name is one of the jit'd defs (not necessarily the first).

    A1 PR 1: ``@triton.autotune`` is required directly above the named
    ``@triton.jit def``; helper kernels may carry their own decorator or
    none. Both the main kernel (autotune-bearing) and the helper (bare
    ``@triton.jit``) below should be acceptable as ``triton_kernel_name``
    targets — the autotune validator only fires on whichever one the
    Coder names as the primary kernel.
    """
    src = """\
import triton
import triton.language as tl

@triton.jit
def _epilogue(): pass

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=4),
    ],
    key=["M"],
)
@triton.jit
def main_kernel(X, M, BLOCK_M: tl.constexpr): pass
"""
    out = KernelCodeOutput(source_code=src, triton_kernel_name="main_kernel")
    assert out.triton_kernel_name == "main_kernel"

    # Naming the helper instead: the helper has no @triton.autotune above
    # it, so the autotune validator rejects (Coder must autotune the
    # benchmarked kernel; if _epilogue is the primary, IT needs autotune).
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        KernelCodeOutput(source_code=src, triton_kernel_name="_epilogue")


def test_output_model_jit_decorator_with_args_recognized():
    """``@triton.jit(do_not_specialize=...)`` should still match — the
    regex tolerates decorator arguments."""
    src = """\
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=4),
    ],
    key=["n"],
)
@triton.jit(do_not_specialize=['n'])
def tuned_kernel(n): pass
"""
    out = KernelCodeOutput(source_code=src, triton_kernel_name="tuned_kernel")
    assert out.triton_kernel_name == "tuned_kernel"


# ── A1 PR 1: KernelCodeOutput @triton.autotune validator ───────────────


def test_validator_accepts_valid_autotune():
    out = KernelCodeOutput(source_code=_VALID_SOURCE, triton_kernel_name="k")
    assert out.triton_kernel_name == "k"


def test_validator_rejects_source_without_autotune():
    from pydantic import ValidationError
    src = "@triton.jit\ndef my_kernel(X): pass"
    with pytest.raises(ValidationError) as exc:
        KernelCodeOutput(source_code=src, triton_kernel_name="my_kernel")
    assert "@triton.autotune" in str(exc.value)


def test_validator_rejects_fewer_than_four_configs():
    from pydantic import ValidationError
    src = """\
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
    ],
    key=["M"],
)
@triton.jit
def my_kernel(X, M, BLOCK_M): pass
"""
    with pytest.raises(ValidationError) as exc:
        KernelCodeOutput(source_code=src, triton_kernel_name="my_kernel")
    assert "at least 4" in str(exc.value).lower()


def test_validator_rejects_empty_key():
    from pydantic import ValidationError
    src = """\
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=4),
    ],
    key=[],
)
@triton.jit
def my_kernel(X, M, BLOCK_M): pass
"""
    with pytest.raises(ValidationError) as exc:
        KernelCodeOutput(source_code=src, triton_kernel_name="my_kernel")
    assert "non-empty" in str(exc.value).lower() or "key=" in str(exc.value)


def test_validator_rejects_autotune_on_unnamed_kernel():
    """The decorator must be above the @triton.jit def matching triton_kernel_name."""
    from pydantic import ValidationError
    src = """\
@triton.jit
def my_kernel(X): pass

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=4),
    ],
    key=["M"],
)
@triton.jit
def other_kernel(X): pass
"""
    with pytest.raises(ValidationError) as exc:
        KernelCodeOutput(source_code=src, triton_kernel_name="my_kernel")
    assert "my_kernel" in str(exc.value) or "@triton.autotune" in str(exc.value)


# ── prompt assembly ─────────────────────────────────────────────────────


def test_build_user_prompt_contains_all_sections():
    plan = OptimizationPlan(
        tier=2,
        technique="t2_shared_memory_tiling",
        params={"tile_m": "64", "tile_n": "64"},
        target_region="inner loop",
        rationale="Reduce DRAM traffic via tiling.",
    )
    prompt = CoderAgent.build_user_prompt(
        kernel_source="@triton.jit\ndef matmul_kernel(): ...",
        plan=plan,
    )
    assert "@triton.jit" in prompt
    assert "Tier: 2" in prompt
    assert "t2_shared_memory_tiling" in prompt
    assert "tile_m=64" in prompt
    assert "tile_n=64" in prompt
    assert "inner loop" in prompt
    assert "Reduce DRAM traffic" in prompt


def test_build_user_prompt_omits_empty_params():
    plan = OptimizationPlan(
        tier=1,
        technique="t1_occupancy",
        target_region="launch config",
        rationale="Increase occupancy.",
    )
    prompt = CoderAgent.build_user_prompt(kernel_source="def k(): pass", plan=plan)
    assert "Params:" not in prompt


def test_build_user_prompt_escapes_backticks_in_kernel_source():
    plan = OptimizationPlan(tier=1, technique="t1", rationale="x")
    source = 'def kernel():\n    """```python\n    fake section\n    ```"""\n    pass'
    prompt = CoderAgent.build_user_prompt(kernel_source=source, plan=plan)
    sections = prompt.split("## ")
    kernel_section = [s for s in sections if s.startswith("Current kernel")][0]
    assert "```python\nfake section\n```" not in kernel_section


def test_build_user_prompt_renders_autotune_exclude_when_populated():
    """Non-empty ``autotune_exclude`` must appear in the rendered plan
    section so the Coder sees the structured constraint at generation
    time, not only at submit-rejection time. Without this line the
    structured-bounds contract (per
    doc/specs/2026-05-18-autotune-exclude-structured-bounds-design.md)
    is half-built: validator enforces, but Coder never receives the
    information.
    """
    plan = OptimizationPlan(
        tier=1, technique="t1_block_size_tuning",
        rationale="exclude the overcommitted config",
        autotune_exclude=[
            {"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 4},
            {"BLOCK_M": 64, "num_stages": 4},
        ],
    )
    prompt = CoderAgent.build_user_prompt(kernel_source="def k(): pass", plan=plan)
    assert "Autotune exclude" in prompt
    # Each pattern appears verbatim as JSON so the Coder can copy keys.
    assert '"BLOCK_M": 128' in prompt
    assert '"BLOCK_N": 128' in prompt
    assert '"num_stages": 4' in prompt
    assert '"BLOCK_M": 64' in prompt


def test_build_user_prompt_omits_autotune_exclude_when_empty():
    """Empty ``autotune_exclude`` (the default) → no rendered line. Keeps
    the prompt clean for plans that don't carry the field."""
    plan = OptimizationPlan(
        tier=1, technique="t1", rationale="x",
        # autotune_exclude defaults to []
    )
    prompt = CoderAgent.build_user_prompt(kernel_source="def k(): pass", plan=plan)
    assert "Autotune exclude" not in prompt


def test_build_user_prompt_renders_technique_guidance_when_present():
    plan = OptimizationPlan(
        tier=2, technique="t2_shared_memory_tiling",
        target_region="inner loop", rationale="reuse",
    )
    prompt = CoderAgent.build_user_prompt(
        kernel_source="@triton.jit\ndef k(): ...",
        plan=plan,
        technique_guidance="In Triton this is implicit — use tl.dot and num_stages.",
    )
    assert "## Technique guidance" in prompt
    assert "tl.dot" in prompt


def test_build_user_prompt_omits_technique_guidance_when_empty():
    plan = OptimizationPlan(
        tier=2, technique="t2_shared_memory_tiling",
        target_region="inner loop", rationale="reuse",
    )
    prompt = CoderAgent.build_user_prompt(
        kernel_source="def k(): pass", plan=plan,
    )
    assert "## Technique guidance" not in prompt


def test_build_translate_prompt_no_section_when_prior_failures_empty():
    """Default path (no prior attempts) — section must be absent."""
    prompt = CoderAgent.build_translate_prompt(
        reference_source="def run(x): return x",
        kernel_spec=_make_spec(),
    )
    assert "Prior attempt failures" not in prompt
    assert "PyTorch reference" in prompt


def test_build_translate_prompt_renders_single_attempt_section():
    failures = [
        AttemptFailure(
            attempt_no=1,
            tool_errors=[
                "Correctness FAILED on workload 1/3 at stage [smoke_test]: tl.tanh",
                "Compilation FAILED:\nshape mismatch",
            ],
        ),
    ]
    prompt = CoderAgent.build_translate_prompt(
        reference_source="def run(x): return x",
        kernel_spec=_make_spec(),
        prior_failures=failures,
    )
    assert "## Prior attempt failures" in prompt
    assert "### Attempt 1" in prompt
    assert "tl.tanh" in prompt
    assert "shape mismatch" in prompt
    # Section must come BEFORE the PyTorch reference block.
    assert prompt.index("## Prior attempt failures") < prompt.index("## PyTorch reference")


def test_build_translate_prompt_renders_multiple_attempts_in_order():
    failures = [
        AttemptFailure(attempt_no=1, tool_errors=["err1"]),
        AttemptFailure(attempt_no=2, tool_errors=["err2"]),
    ]
    prompt = CoderAgent.build_translate_prompt(
        reference_source="def run(x): return x",
        kernel_spec=_make_spec(),
        prior_failures=failures,
    )
    assert "### Attempt 1" in prompt
    assert "### Attempt 2" in prompt
    assert prompt.index("### Attempt 1") < prompt.index("### Attempt 2")
    assert "err1" in prompt
    assert "err2" in prompt


def test_build_translate_prompt_empty_tool_errors_renders_placeholder():
    """Reasoning-content truncation pathology: the attempt's tool_errors list
    is empty. The block must still render (so the model sees the attempt
    happened) with an explanatory placeholder bullet."""
    failures = [AttemptFailure(attempt_no=1, tool_errors=[])]
    prompt = CoderAgent.build_translate_prompt(
        reference_source="def run(x): return x",
        kernel_spec=_make_spec(),
        prior_failures=failures,
    )
    assert "### Attempt 1" in prompt
    assert "no tool errors recorded" in prompt


# ── compile tool factory ────────────────────────────────────────────────


def test_compile_tool_factory_returns_callable(tmp_path):
    tool = _make_compile_tool(_make_spec(), cache_dir=tmp_path)
    assert callable(tool)


def test_compile_tool_reports_success_on_good_source(tmp_path):
    tool = _make_compile_tool(_make_spec(), cache_dir=tmp_path)
    msg = tool("def kernel_fn(x):\n    return x + 1\n")
    assert "success" in msg.lower()
    assert "kernel_fn" in msg  # entrypoint surfaced so Coder knows what it resolved


def test_compile_tool_reports_error_on_syntax_error(tmp_path):
    tool = _make_compile_tool(_make_spec(), cache_dir=tmp_path)
    msg = tool("def kernel_fn(: invalid\n")
    assert "fail" in msg.lower() or "error" in msg.lower()
    assert "SyntaxError" in msg


def test_compile_tool_reports_error_on_missing_entrypoint(tmp_path):
    tool = _make_compile_tool(_make_spec(entrypoint="run"), cache_dir=tmp_path)
    msg = tool("def kernel_fn(x): return x\n")  # wrong symbol name
    assert "run" in msg  # the missing entrypoint name


def test_compile_tool_appends_to_error_log_on_failure(tmp_path):
    log: list[str] = []
    tool = _make_compile_tool(_make_spec(), cache_dir=tmp_path, error_log=log)
    msg = tool("def kernel_fn(: broken\n")
    assert msg.startswith("Compilation FAILED:")
    assert log == [msg]


def test_compile_tool_does_not_append_on_success(tmp_path):
    log: list[str] = []
    tool = _make_compile_tool(_make_spec(), cache_dir=tmp_path, error_log=log)
    msg = tool("def kernel_fn(x):\n    return x + 1\n")
    assert "success" in msg.lower()
    assert log == []


def test_compile_tool_works_with_no_error_log(tmp_path):
    """error_log=None must not raise on failure paths."""
    tool = _make_compile_tool(_make_spec(), cache_dir=tmp_path, error_log=None)
    msg = tool("def kernel_fn(: broken\n")
    assert msg.startswith("Compilation FAILED:")  # no AttributeError raised


# ── correctness tool factory ────────────────────────────────────────────


def test_correctness_tool_factory_returns_callable(tmp_path):
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
        allow_in_parent_fallback=True,
    )
    assert callable(tool)


@pytest.mark.asyncio
async def test_correctness_tool_reports_compile_error_without_running_correctness(tmp_path):
    """If the candidate source won't compile, surface that — don't try to run it."""
    calls = {"ref": 0}

    def ref(x):
        calls["ref"] += 1
        return x * 2.0

    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=ref,
        input_generators=[_gen],
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
        allow_in_parent_fallback=True,
    )
    msg = await tool("def kernel_fn(: broken\n")
    assert "compile" in msg.lower()
    assert calls["ref"] == 0  # reference was never invoked


@pytest.mark.asyncio
async def test_correctness_tool_reports_success_on_matching_candidate(tmp_path):
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
        allow_in_parent_fallback=True,
    )
    msg = await tool("def kernel_fn(x):\n    return x * 2.0\n")
    assert "pass" in msg.lower()


@pytest.mark.asyncio
async def test_correctness_tool_reports_failure_stage_on_mismatch(tmp_path):
    """Failure messages surface the failed stage so the Coder can diagnose."""
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
        allow_in_parent_fallback=True,
    )
    msg = await tool("def kernel_fn(x):\n    return x * 3.0\n")
    assert "fail" in msg.lower()
    assert "smoke_test" in msg  # first-stage failure for a uniformly-wrong candidate


@pytest.mark.asyncio
async def test_correctness_tool_appends_to_error_log_on_failure(tmp_path):
    log: list[str] = []
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
        error_log=log,
        allow_in_parent_fallback=True,
    )
    msg = await tool("def kernel_fn(x):\n    return x * 3.0\n")
    assert "FAILED" in msg
    assert log == [msg]


@pytest.mark.asyncio
async def test_correctness_tool_appends_to_error_log_on_compile_abort(tmp_path):
    """Compile-abort branch inside the correctness tool also logs to error_log."""
    log: list[str] = []
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
        error_log=log,
        allow_in_parent_fallback=True,
    )
    msg = await tool("def kernel_fn(: broken\n")
    assert "Correctness aborted" in msg
    assert log == [msg]


@pytest.mark.asyncio
async def test_correctness_tool_does_not_append_on_success(tmp_path):
    log: list[str] = []
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
        policy=_ScalarPolicy(),
        cache_dir=tmp_path,
        error_log=log,
        allow_in_parent_fallback=True,
    )
    msg = await tool("def kernel_fn(x):\n    return x * 2.0\n")
    assert "pass" in msg.lower()
    assert log == []


def test_correctness_tool_empty_generators_raises():
    """Building a correctness tool with no workloads is a contract violation."""
    with pytest.raises(ValueError, match="generator"):
        _make_correctness_tool(
            _make_spec(),
            reference_fn=_ref,
            input_generators=[],
            policy=_ScalarPolicy(),
        )


# ── correctness-isolation trust gate (construction guard) ──────────────
#
# Absent a ``problem_definition_path`` the candidate launch can't be
# crash-isolated in a subprocess, so launching it in-parent is only safe
# in a deliberately-trusted/mocked context. The factory raises a typed
# ``CorrectnessIsolationError`` unless ``allow_in_parent_fallback=True``
# opts in — converting the "no untrusted launch in the parent CUDA
# context" invariant from coincidence to construction.


def _dummy_corr_tool(**overrides):
    kw = dict(
        kernel_spec=_make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
        definition=object(),
        workloads=[object()],
        problem_definition_path=None,
        blob_roots=None,
    )
    kw.update(overrides)
    return _make_correctness_tool(**kw)


def test_correctness_tool_raises_without_path_or_optin():
    """No definition path + no opt-in → the construction guard raises."""
    from src.eval.correctness_subprocess import CorrectnessIsolationError

    with pytest.raises(CorrectnessIsolationError):
        _dummy_corr_tool()  # no path, flag defaults False


def test_correctness_tool_allows_in_parent_with_explicit_optin():
    """Explicit opt-in (trusted/mocked context) → tool constructs in-parent."""
    tool = _dummy_corr_tool(allow_in_parent_fallback=True)
    assert callable(tool)  # constructed, no raise


def test_correctness_tool_with_path_constructs_without_optin():
    """A bound definition path → subprocess branch; the flag is irrelevant."""
    tool = _dummy_corr_tool(problem_definition_path="/p")
    assert callable(tool)


@pytest.mark.asyncio
async def test_correctness_tool_iterates_all_generators_and_reports_first_failure(tmp_path):
    """Tool must run each generator until one fails — its output tells the Coder
    which workload broke so retries can actually correct multi-workload bugs."""
    from src.eval.correctness import CorrectnessResult, CorrectnessStage

    gens = [MagicMock(name=f"gen_{i}") for i in range(3)]
    results = [
        CorrectnessResult(passed=True, max_abs_error=0.0),
        CorrectnessResult(
            passed=False,
            failed_stage=CorrectnessStage.NUMERICAL_STABILITY,
            error_message="numerical mismatch",
            max_abs_error=1.0,
        ),
    ]
    with patch("src.agents.coder.verify_correctness", side_effect=results) as mock_verify:
        tool = _make_correctness_tool(
            _make_spec(),
            reference_fn=_ref,
            input_generators=gens,
            cache_dir=tmp_path,
            allow_in_parent_fallback=True,
        )
        msg = await tool("def kernel_fn(x):\n    return x * 2.0\n")

    assert mock_verify.call_count == 2  # short-circuit after first failure
    assert "workload 2" in msg.lower()
    assert "numerical_stability" in msg


@pytest.mark.asyncio
async def test_correctness_tool_reports_success_when_all_generators_pass(tmp_path):
    """All workloads clean → single success message (not one per workload)."""
    from src.eval.correctness import CorrectnessResult

    gens = [MagicMock(name=f"gen_{i}") for i in range(3)]
    with patch(
        "src.agents.coder.verify_correctness",
        return_value=CorrectnessResult(passed=True, max_abs_error=0.0),
    ) as mock_verify:
        tool = _make_correctness_tool(
            _make_spec(),
            reference_fn=_ref,
            input_generators=gens,
            cache_dir=tmp_path,
            allow_in_parent_fallback=True,
        )
        msg = await tool("def kernel_fn(x):\n    return x * 2.0\n")

    assert mock_verify.call_count == 3
    assert "pass" in msg.lower()


# ── correctness tool — subprocess isolation (gate mode) ─────────────────
#
# When a real ``problem_definition_path`` is supplied, the tool must
# delegate the candidate launch to the crash-isolated worker
# (``run_correctness_subprocess``, mode ``gate``) instead of compiling +
# launching the kernel on the parent process's CUDA context. The returned
# tool becomes ``async``; the helper is mocked so these stay Tier-1.


class _DummyWorkload:
    def model_dump(self, mode="json"):
        return {"uuid": "w0"}


async def _aresult(r):  # coroutine returning a pre-built result
    return r


@pytest.mark.asyncio
async def test_correctness_tool_delegates_to_subprocess(monkeypatch):
    from src.eval.correctness_subprocess import CorrectnessResult

    seen = {}

    async def _fake_helper(*, request, worker_dir, timeout_s):
        seen["mode"] = request["mode"]
        seen["seed"] = request["input_seed"]
        return CorrectnessResult(
            passed=False, failed_stage="numerical",
            error_message="bad at [3]", total_workloads=3,
            failed_workload_idx=2,
        )

    monkeypatch.setattr(
        "src.agents.coder.run_correctness_subprocess", _fake_helper,
        raising=False,
    )
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
        definition=object(),
        workloads=[_DummyWorkload()],
        problem_definition_path="/p",
        blob_roots=["/p"],
        worker_timeout_s=180.0,
    )
    msg = await tool("source", dps=False)
    assert seen["mode"] == "gate"
    assert seen["seed"] == 0
    assert "FAILED on workload 2/3" in msg
    assert "numerical" in msg


@pytest.mark.asyncio
async def test_correctness_tool_passes_message(monkeypatch):
    from src.eval.correctness_subprocess import CorrectnessResult

    monkeypatch.setattr(
        "src.agents.coder.run_correctness_subprocess",
        lambda **kw: _aresult(
            CorrectnessResult(passed=True, max_err=2e-3, total_workloads=3),
        ),
        raising=False,
    )
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
        definition=object(),
        workloads=[_DummyWorkload()],
        problem_definition_path="/p",
        blob_roots=["/p"],
        worker_timeout_s=180.0,
    )
    msg = await tool("source", dps=False)
    assert "passed on all 3 workloads" in msg
    assert "2.000e-03" in msg


@pytest.mark.asyncio
async def test_correctness_tool_reports_gpu_crash(monkeypatch):
    """worker_crashed / timeout → explicit 'crashed the GPU' message."""
    from src.eval.correctness_subprocess import CorrectnessResult

    monkeypatch.setattr(
        "src.agents.coder.run_correctness_subprocess",
        lambda **kw: _aresult(
            CorrectnessResult(
                passed=False, failed_stage="worker_crashed",
                error_message="device-side assert triggered",
            ),
        ),
        raising=False,
    )
    log: list[str] = []
    tool = _make_correctness_tool(
        _make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
        definition=object(),
        workloads=[_DummyWorkload()],
        problem_definition_path="/p",
        blob_roots=["/p"],
        error_log=log,
    )
    msg = await tool("source", dps=False)
    assert "crashed the GPU" in msg
    assert log == [msg]


# ── implement() — placeholder path (no model) ───────────────────────────


@pytest.mark.asyncio
async def test_implement_without_model_returns_source_unchanged():
    agent = CoderAgent(model=None)
    plan = OptimizationPlan(tier=1, technique="t1_occupancy")
    src = "@triton.jit\ndef k(): ..."
    out = await agent.implement(
        kernel_source=src,
        plan=plan,
        kernel_spec=_make_spec(),
        reference_fn=_ref,
        input_generators=[_gen],
    )
    assert isinstance(out, KernelCodeOutput)
    assert out.source_code == src
    # No-model placeholder path can't declare a kernel name (no LLM to ask) —
    # downstream profiler falls back to source-regex extraction when this is empty.
    assert out.triton_kernel_name == ""


# ── implement() — LLM path (mocked) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_implement_calls_llm_and_returns_modified_source():
    """With a model, implement() builds the Agent with bound tools and runs it.
    The Coder emits its answer by calling submit_kernel, not via output_type."""
    capture_agent, fake_run = _simulate_submission(
        source_code=_VALID_SOURCE,
        triton_kernel_name=_VALID_NAME,
    )

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent) as mock_agent_cls,
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run

        agent = CoderAgent(model=MagicMock())
        plan = OptimizationPlan(
            tier=1,
            technique="t1_block_size_tuning",
            params={"block_size": "128"},
            target_region="main loop",
            rationale="Bigger tile => more reuse.",
        )
        result = await agent.implement(
            kernel_source="original source",
            plan=plan,
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    assert isinstance(result, KernelCodeOutput)
    assert result.source_code == _VALID_SOURCE
    assert result.triton_kernel_name == _VALID_NAME
    mock_run.assert_awaited_once()
    # Agent gets compile + correctness + submit_kernel tools (3) and is built
    # without ``output_type=`` so the SDK doesn't request response_format=json_schema.
    kwargs = mock_agent_cls.call_args.kwargs
    assert len(kwargs["tools"]) == 3
    assert "output_type" not in kwargs
    assert any(
        getattr(t, "__name__", "") == "submit_kernel" for t in kwargs["tools"]
    )


@pytest.mark.asyncio
async def test_implement_raises_when_input_generators_missing_or_empty():
    """A model-backed implement() cannot bind its correctness tool without at least
    one generator — refuse fast so Phase B doesn't silently score broken children."""
    agent = CoderAgent(model=MagicMock())
    plan = OptimizationPlan(tier=1, technique="t1")

    with pytest.raises(ImplementationError, match="input_generators"):
        await agent.implement(
            kernel_source="src",
            plan=plan,
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=None,
        )

    with pytest.raises(ImplementationError, match="input_generators"):
        await agent.implement(
            kernel_source="src",
            plan=plan,
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[],
        )


@pytest.mark.asyncio
async def test_implement_binds_all_generators_to_correctness_tool():
    """Phase B must bind every selected workload's generator to the correctness
    tool — else a kernel that passes workload 1 but breaks 2..N slips through."""
    gens = [MagicMock(name=f"gen_{i}") for i in range(3)]
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)

    captured = {}
    def capture_factory(*args, **kwargs):
        captured["input_generators"] = kwargs["input_generators"]
        return lambda src: "passed"

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
        patch("src.agents.coder._make_correctness_tool", side_effect=capture_factory),
    ):
        mock_run.side_effect = fake_run
        agent = CoderAgent(model=MagicMock())
        await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=gens,
        )

    assert captured["input_generators"] is gens


@pytest.mark.asyncio
async def test_implement_raises_on_llm_failure():
    """If run_agent returns None (retries exhausted), raise ImplementationError."""
    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = None

        agent = CoderAgent(model=MagicMock())
        plan = OptimizationPlan(tier=1, technique="t1")
        with pytest.raises(ImplementationError, match="LLM"):
            await agent.implement(
                kernel_source="src",
                plan=plan,
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
                allow_in_parent_fallback=True,
            )


@pytest.mark.asyncio
async def test_implement_passes_default_max_turns_when_no_config():
    """No config → default ACTSConfig.max_debug_retries=3 → max_turns = 2*3+2 = 8.
    The +2 (vs. the historical +1) reserves one turn for ``submit_kernel`` and
    one for the final plain-text confirmation that terminates the SDK loop."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run

        agent = CoderAgent(model=MagicMock())
        await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    assert mock_run.await_args.kwargs.get("max_turns") == 8


@pytest.mark.asyncio
async def test_implement_max_turns_derived_from_config():
    """max_debug_retries=5 → max_turns = 2*5+2 = 12 (+2 for submit + confirm)."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run

        agent = CoderAgent(model=MagicMock(), config=ACTSConfig(max_debug_retries=5))
        await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    assert mock_run.await_args.kwargs.get("max_turns") == 12


@pytest.mark.asyncio
async def test_implement_uses_zero_temperature():
    """Coder runs with temperature=0.0 — deterministic code generation."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config") as mock_cfg,
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
        mock_cfg.return_value = None

        agent = CoderAgent(model=MagicMock())
        await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    mock_cfg.assert_called_once_with(temperature=0.0)


@pytest.mark.asyncio
async def test_implement_raises_when_agent_terminates_without_submitting():
    """Submit-tool contract: if the LLM exits the tool loop without ever
    calling submit_kernel, we have no Coder output. Raising
    ImplementationError lets the caller surface the failure rather than
    silently treating an empty submission as a degraded best-effort."""
    # Capture-agent path (no fake_run side_effect that calls submit_kernel) —
    # mock_run returns a normal RunResult but the captured dict stays empty.
    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = MagicMock(final_output="done")

        agent = CoderAgent(model=MagicMock())
        with pytest.raises(ImplementationError, match="submit_kernel"):
            await agent.implement(
                kernel_source="src",
                plan=OptimizationPlan(tier=1, technique="t1"),
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
                allow_in_parent_fallback=True,
            )


# ── MaxTurnsExceeded handling ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_implement_converts_max_turns_exceeded_to_implementation_error():
    """SDK ``MaxTurnsExceeded`` (raised mid-tool-loop when the LLM burns
    through the budget without ever submitting) must be converted to
    ``ImplementationError`` at the Coder boundary so the orchestrator /
    baseline_generator catch sites work uniformly. Without this
    conversion, the SDK exception propagates straight out of
    ``optimize()`` and aborts the entire run instead of dead-ending one
    branch."""
    from src.agents.coder import MaxTurnsExceeded

    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = MaxTurnsExceeded("Max turns (8) exceeded")

        agent = CoderAgent(model=MagicMock())
        with pytest.raises(ImplementationError, match="turn budget"):
            await agent.implement(
                kernel_source="src",
                plan=OptimizationPlan(tier=1, technique="t1"),
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
                allow_in_parent_fallback=True,
            )


@pytest.mark.asyncio
async def test_implement_returns_partial_output_when_max_turns_after_submission():
    """If the LLM submitted a valid kernel before the SDK loop hit max_turns
    (e.g., it kept calling tools after submit despite the system-prompt rule),
    treat that submission as the answer rather than raising. The kernel was
    Pydantic-validated when submit_kernel ran; the run merely went over budget."""
    from src.agents.coder import MaxTurnsExceeded

    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)

    async def submit_then_exhaust(agent, prompt, **kwargs):
        # First simulate a successful submission, then raise as if the
        # SDK kept spinning after submit and burned the budget.
        await fake_run(agent, prompt, **kwargs)
        raise MaxTurnsExceeded("Max turns exceeded")

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = submit_then_exhaust

        agent = CoderAgent(model=MagicMock())
        result = await agent.implement(
            kernel_source="src",
            plan=OptimizationPlan(tier=1, technique="t1"),
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    assert isinstance(result, KernelCodeOutput)
    assert result.source_code == _VALID_SOURCE
    assert result.triton_kernel_name == _VALID_NAME


# ── tool_errors propagation through ImplementationError ────────────────


def _capture_error_log_factories(seed_errors: list[str]):
    """Build patch side-effects for _make_compile_tool / _make_correctness_tool
    that capture the error_log list bound by ``_run_tool_agent`` and seed it
    with *seed_errors* so the raised ``ImplementationError.tool_errors``
    deterministically carries them.

    Returns ``(compile_side_effect, correctness_side_effect, captured)``;
    ``captured["error_log"]`` is populated after the first factory call so
    tests can assert against the same list object the SDK loop saw.
    """
    captured: dict = {}

    def compile_side_effect(*args, error_log=None, **kwargs):
        if error_log is not None:
            captured.setdefault("error_log", error_log)
            for e in seed_errors:
                error_log.append(e)
        return MagicMock(return_value="Compilation FAILED:\nseeded")

    def correctness_side_effect(*args, error_log=None, **kwargs):
        if error_log is not None:
            captured.setdefault("error_log", error_log)
        return MagicMock(return_value="Correctness FAILED: seeded")

    return compile_side_effect, correctness_side_effect, captured


@pytest.mark.asyncio
async def test_max_turns_exceeded_carries_tool_errors():
    """When MaxTurnsExceeded fires without a captured submit, the raised
    ImplementationError must carry the tool-error log so baseline_generator
    can thread it into the next attempt's prompt."""
    from src.agents.coder import MaxTurnsExceeded

    seeded = ["compile FAILED #1", "correctness FAILED #2", "compile FAILED #3"]
    compile_se, correctness_se, captured = _capture_error_log_factories(seeded)

    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder._make_compile_tool", side_effect=compile_se),
        patch("src.agents.coder._make_correctness_tool", side_effect=correctness_se),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = MaxTurnsExceeded("budget hit")

        agent = CoderAgent(model=MagicMock())
        with pytest.raises(ImplementationError) as exc_info:
            await agent.implement(
                kernel_source="src",
                plan=OptimizationPlan(tier=1, technique="t1"),
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
            )

    err = exc_info.value
    assert "turn budget" in str(err)
    assert err.tool_errors == seeded


@pytest.mark.asyncio
async def test_did_not_submit_carries_tool_errors():
    """The "agent terminated without calling submit_kernel" path also carries
    whatever errors the tools logged before the agent gave up."""
    seeded = ["correctness FAILED: tl.tanh"]
    compile_se, correctness_se, _ = _capture_error_log_factories(seeded)

    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder._make_compile_tool", side_effect=compile_se),
        patch("src.agents.coder._make_correctness_tool", side_effect=correctness_se),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = MagicMock(final_output="done")  # no submit captured

        agent = CoderAgent(model=MagicMock())
        with pytest.raises(ImplementationError) as exc_info:
            await agent.implement(
                kernel_source="src",
                plan=OptimizationPlan(tier=1, technique="t1"),
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
            )

    err = exc_info.value
    assert "submit_kernel" in str(err)
    assert err.tool_errors == seeded


@pytest.mark.asyncio
async def test_submit_validation_failure_lands_in_tool_errors():
    """An attempt that fails ONLY at submit time (Pydantic validation
    reject) must carry that validation error in ImplementationError.tool_errors
    so the next baseline-generator retry sees it. Without this fix, the
    next attempt's "## Prior attempt failures" block reads the misleading
    'no tool errors recorded' placeholder even though the attempt did
    invoke submit_kernel."""
    captured_logs: dict = {}

    def capture_submit_factory(captured_dict, *, error_log=None, **kwargs):
        captured_logs["error_log"] = error_log
        # Simulate the SDK loop dispatching submit_kernel mid-run with a
        # validation-failing payload (kernel_name absent from source).
        if error_log is not None:
            error_log.append(
                "submit_kernel FAILED:\nValidation: triton_kernel_name "
                "'claimed' not found in source"
            )
        return MagicMock(return_value="submit_kernel FAILED")

    def noop_compile_factory(*args, error_log=None, **kwargs):
        return MagicMock(return_value="Compilation successful")

    def noop_correctness_factory(*args, error_log=None, **kwargs):
        return MagicMock(return_value="passed")

    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder._make_compile_tool", side_effect=noop_compile_factory),
        patch("src.agents.coder._make_correctness_tool", side_effect=noop_correctness_factory),
        patch("src.agents.coder._make_submit_tool", side_effect=capture_submit_factory),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = MagicMock(final_output="done")  # no captured["output"]

        agent = CoderAgent(model=MagicMock())
        with pytest.raises(ImplementationError) as exc_info:
            await agent.implement(
                kernel_source="src",
                plan=OptimizationPlan(tier=1, technique="t1"),
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
            )

    err = exc_info.value
    # The submit-tool's validation failure must be in tool_errors.
    assert len(err.tool_errors) == 1
    assert "submit_kernel FAILED" in err.tool_errors[0]
    assert "triton_kernel_name" in err.tool_errors[0]


@pytest.mark.asyncio
async def test_empty_tool_errors_when_no_tool_calls_happened():
    """Reasoning-content truncation pathology: agent returns without invoking
    any tool. The factories never get error_log entries appended, so the
    raised ImplementationError carries an empty tool_errors list — the
    placeholder rendering in build_translate_prompt covers this case."""

    def noop_compile_side_effect(*args, error_log=None, **kwargs):
        # Capture the slot but DON'T append — simulates no tool calls.
        return MagicMock(return_value="Compilation successful (entrypoint: 'kernel_fn').")

    def noop_correctness_side_effect(*args, error_log=None, **kwargs):
        return MagicMock(return_value="passed")

    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder._make_compile_tool", side_effect=noop_compile_side_effect),
        patch("src.agents.coder._make_correctness_tool", side_effect=noop_correctness_side_effect),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = MagicMock(final_output="done")

        agent = CoderAgent(model=MagicMock())
        with pytest.raises(ImplementationError) as exc_info:
            await agent.implement(
                kernel_source="src",
                plan=OptimizationPlan(tier=1, technique="t1"),
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
            )

    assert exc_info.value.tool_errors == []


# ── has_model property ─────────────────────────────────────────────────


def test_has_model_reflects_configuration():
    """baseline_generator branches on has_model — must be a public, stable signal."""
    assert CoderAgent(model=None).has_model is False
    assert CoderAgent(model=MagicMock()).has_model is True


# ── translate() — PyTorch→Triton one-shot port ─────────────────────────


@pytest.mark.asyncio
async def test_translate_without_model_raises():
    """translate() has no sensible no-op fallback — a model is required."""
    agent = CoderAgent(model=None)
    with pytest.raises(ImplementationError, match="model"):
        await agent.translate(
            reference_source="def run(x):\n    return x * 2.0\n",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
        )


@pytest.mark.asyncio
async def test_translate_builds_agent_with_three_tools_and_returns_source():
    """translate() constructs a fresh Agent with compile + correctness +
    submit tools and returns the captured Coder submission."""
    capture_agent, fake_run = _simulate_submission(
        source_code=_VALID_SOURCE,
        triton_kernel_name=_VALID_NAME,
    )

    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent) as mock_agent_cls,
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
        agent = CoderAgent(model=MagicMock())
        result = await agent.translate(
            reference_source="def run(x):\n    return x * 2.0\n",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    assert isinstance(result, KernelCodeOutput)
    assert result.source_code == _VALID_SOURCE
    assert result.triton_kernel_name == _VALID_NAME
    mock_run.assert_awaited_once()
    kwargs = mock_agent_cls.call_args.kwargs
    assert len(kwargs["tools"]) == 3  # compile + correctness + submit
    assert "output_type" not in kwargs


@pytest.mark.asyncio
async def test_translate_threads_prior_failures_into_user_prompt():
    """translate() must forward prior_failures to build_translate_prompt
    so the rendered user prompt carries the cross-attempt memory section."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)
    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x): return x",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            prior_failures=[
                AttemptFailure(attempt_no=1, tool_errors=["tl.tanh AttributeError"]),
            ],
            allow_in_parent_fallback=True,
        )

    prompt = mock_run.await_args.args[1]
    assert "## Prior attempt failures" in prompt
    assert "### Attempt 1" in prompt
    assert "tl.tanh AttributeError" in prompt
    # Empty default still works — caller may omit prior_failures.


@pytest.mark.asyncio
async def test_translate_no_prior_failures_section_by_default():
    """Existing callers that omit prior_failures must get the original
    prompt shape — no section rendered."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)
    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x): return x",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    prompt = mock_run.await_args.args[1]
    assert "Prior attempt failures" not in prompt


@pytest.mark.asyncio
async def test_translate_user_prompt_contains_reference_and_entrypoint():
    """Prompt must surface the source to translate and the target entrypoint."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)
    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x):\n    return x * 2.0\n",
            kernel_spec=_make_spec(entrypoint="my_kernel"),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    prompt = mock_run.await_args.args[1]
    assert "def run(x)" in prompt
    assert "my_kernel" in prompt


@pytest.mark.asyncio
async def test_translate_uses_distinct_translate_instructions():
    """translate() loads translate.md — separate from the optimize system.md."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)
    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent) as mock_agent_cls,
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x): return x",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    instructions = mock_agent_cls.call_args.kwargs["instructions"]
    # translate.md is the from-scratch port prompt; must surface both dialects.
    assert "Triton" in instructions
    assert "PyTorch" in instructions
    # Must NOT carry optimize-mode framing that contradicts the translation task.
    assert "one focused change" not in instructions.lower()


@pytest.mark.asyncio
async def test_translate_raises_on_llm_failure():
    """run_agent returning None (retries exhausted) → ImplementationError."""
    with (
        patch("src.agents.coder.Agent"),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config", return_value=None),
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.return_value = None
        agent = CoderAgent(model=MagicMock())
        with pytest.raises(ImplementationError, match="LLM"):
            await agent.translate(
                reference_source="def run(x): return x",
                kernel_spec=_make_spec(),
                reference_fn=_ref,
                input_generators=[_gen],
                allow_in_parent_fallback=True,
            )


@pytest.mark.asyncio
async def test_translate_uses_zero_temperature():
    """Like implement(), translate() pins temperature=0.0 for determinism."""
    capture_agent, fake_run = _simulate_submission(_VALID_SOURCE, _VALID_NAME)
    with (
        patch("src.agents.coder.Agent", side_effect=capture_agent),
        patch("src.agents.coder.run_agent", new_callable=AsyncMock) as mock_run,
        patch("src.agents.coder.make_run_config") as mock_cfg,
        patch("src.agents.coder.function_tool", side_effect=lambda f: f),
    ):
        mock_run.side_effect = fake_run
        mock_cfg.return_value = None
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x): return x",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    mock_cfg.assert_called_once_with(temperature=0.0)


@pytest.mark.asyncio
async def test_translate_forwards_definition_path_and_blob_roots_to_run_tool_agent():
    """translate() must thread problem_definition_path/blob_roots into
    _run_tool_agent so the baseline-generation correctness tool takes the
    crash-isolated subprocess path (mode gate) instead of the in-parent
    _legacy_in_parent_check fallback. Codex P1: an out-of-bounds LLM
    baseline would otherwise poison the parent CUDA context before Phase B.
    """
    captured: dict = {}

    async def fake_run_tool_agent(self, **kwargs):
        captured.update(kwargs)
        return KernelCodeOutput.model_construct(
            source_code=_VALID_SOURCE, triton_kernel_name=_VALID_NAME, dps=False,
        )

    with patch.object(CoderAgent, "_run_tool_agent", fake_run_tool_agent):
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x): return x",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            problem_definition_path="/problem/definition.json",
            blob_roots=["/problem"],
        )

    assert captured["problem_definition_path"] == "/problem/definition.json"
    assert captured["blob_roots"] == ["/problem"]


@pytest.mark.asyncio
async def test_translate_definition_path_defaults_to_none():
    """Omitting the new kwargs keeps the existing in-parent fallback shape —
    non-breaking for callers that don't bind a SOL problem dir."""
    captured: dict = {}

    async def fake_run_tool_agent(self, **kwargs):
        captured.update(kwargs)
        return KernelCodeOutput.model_construct(
            source_code=_VALID_SOURCE, triton_kernel_name=_VALID_NAME, dps=False,
        )

    with patch.object(CoderAgent, "_run_tool_agent", fake_run_tool_agent):
        agent = CoderAgent(model=MagicMock())
        await agent.translate(
            reference_source="def run(x): return x",
            kernel_spec=_make_spec(),
            reference_fn=_ref,
            input_generators=[_gen],
            allow_in_parent_fallback=True,
        )

    assert captured["problem_definition_path"] is None
    assert captured["blob_roots"] is None


# ── submit-tool factory ────────────────────────────────────────────────


def test_make_submit_tool_captures_valid_output():
    """Direct unit test of the submit-tool factory: a valid (source, name)
    pair populates the captured dict and returns the success sentinel."""
    from src.agents.coder import _make_submit_tool

    captured: dict = {}
    submit = _make_submit_tool(captured)
    msg = submit(
        source_code=_VALID_SOURCE,
        triton_kernel_name=_VALID_NAME,
    )
    assert "submitted" in msg.lower()
    assert "output" in captured
    assert isinstance(captured["output"], KernelCodeOutput)
    assert captured["output"].source_code == _VALID_SOURCE
    assert captured["output"].triton_kernel_name == _VALID_NAME


def test_make_submit_tool_returns_validation_error_string_on_mismatch():
    """A name-not-in-source mismatch must NOT raise — instead the tool returns
    the error string so the SDK hands it back to the LLM as the tool-call
    response, prompting an in-loop retry within the existing turn budget."""
    from src.agents.coder import _make_submit_tool

    captured: dict = {}
    submit = _make_submit_tool(captured)
    msg = submit(
        source_code="@triton.jit\ndef actual_name(): pass",
        triton_kernel_name="claimed_name",
    )
    assert "FAILED" in msg
    assert "claimed_name" in msg
    # On failure the captured dict stays empty so coder.py raises
    # ImplementationError after the run if the LLM never recovered.
    assert "output" not in captured


def test_make_submit_tool_returns_error_when_source_lacks_triton_jit():
    """The Coder writes Triton kernels — pure-PyTorch source is rejected
    by the same Pydantic validator the old output_type= path used."""
    from src.agents.coder import _make_submit_tool

    captured: dict = {}
    submit = _make_submit_tool(captured)
    msg = submit(
        source_code="def run(x): return x * 2.0",
        triton_kernel_name="run",
    )
    assert "FAILED" in msg


def test_make_submit_tool_appends_validation_failure_to_error_log():
    """Pydantic validation failures must land in error_log so cross-attempt
    memory carries the actual reason an attempt failed when the failure
    happens at submit time. Without this the prior-failures section
    renders the misleading 'no tool errors recorded' placeholder for an
    attempt that DID invoke submit_kernel but had its payload rejected."""
    from src.agents.coder import _make_submit_tool

    captured: dict = {}
    log: list[str] = []
    submit = _make_submit_tool(captured, error_log=log)
    msg = submit(
        source_code="@triton.jit\ndef actual_name(): pass",
        triton_kernel_name="claimed_name",
    )
    assert "FAILED" in msg
    assert log == [msg]


def test_make_submit_tool_does_not_append_on_success():
    """Successful submissions are not failures to remember."""
    from src.agents.coder import _make_submit_tool

    captured: dict = {}
    log: list[str] = []
    submit = _make_submit_tool(captured, error_log=log)
    msg = submit(
        source_code=_VALID_SOURCE,
        triton_kernel_name=_VALID_NAME,
    )
    assert "submitted" in msg.lower()
    assert log == []


def test_make_submit_tool_works_with_no_error_log():
    """error_log=None keeps the original two-arg call site working."""
    from src.agents.coder import _make_submit_tool

    captured: dict = {}
    submit = _make_submit_tool(captured, error_log=None)
    msg = submit(
        source_code="@triton.jit\ndef actual_name(): pass",
        triton_kernel_name="claimed_name",
    )
    assert "FAILED" in msg  # no AttributeError raised
    assert "@triton.jit" in msg
    assert "output" not in captured


# ── autotune_exclude validator tests ──────────────────────────────────────────


def _kernel_source_with_configs(configs_block: str) -> str:
    """Build a minimal Coder-valid kernel source with a custom autotune
    configs list. Inserts the literal text into a ≥4-config-valid template.
    """
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.autotune(\n"
        "    configs=[\n"
        f"{configs_block}\n"
        "    ],\n"
        "    key=[\"M\", \"N\", \"K\"],\n"
        ")\n"
        "@triton.jit\n"
        "def my_kernel(x_ptr, M, N, K, BLOCK_M: tl.constexpr, "
        "BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):\n"
        "    pass\n"
    )


_FOUR_CONFIGS_NO_STAGES4 = (
    '        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),\n'
    '        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),\n'
    '        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),\n'
    '        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),'
)

_FOUR_CONFIGS_WITH_STAGES4 = (
    '        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),\n'
    '        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),\n'
    '        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),\n'
    '        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),'
)


def test_submit_kernel_no_op_when_autotune_exclude_empty():
    """Empty ``autotune_exclude`` (the default) → validator skips the
    exclude check even on a kernel whose configs WOULD match a hypothetical
    populated pattern."""
    from src.agents.coder import _make_submit_tool
    from src.agents.llm_backend import SUBMIT_OK_SENTINEL
    from src.agents.planner import OptimizationPlan

    captured: dict = {}
    source = _kernel_source_with_configs(_FOUR_CONFIGS_WITH_STAGES4)
    plan = OptimizationPlan(tier=1, technique="t1")  # autotune_exclude=[]
    submit = _make_submit_tool(captured, plan=plan)
    result = submit(source_code=source, triton_kernel_name="my_kernel")
    assert result == SUBMIT_OK_SENTINEL
    assert "output" in captured


def test_submit_kernel_rejects_single_key_exclude_match():
    """Single-key pattern ``{"num_stages": 4}`` matches any config with
    that value."""
    from src.agents.coder import _make_submit_tool
    from src.agents.planner import OptimizationPlan

    captured: dict = {}
    source = _kernel_source_with_configs(_FOUR_CONFIGS_WITH_STAGES4)
    plan = OptimizationPlan(
        tier=1, technique="t1",
        autotune_exclude=[{"num_stages": 4}],
    )
    submit = _make_submit_tool(captured, plan=plan)
    result = submit(source_code=source, triton_kernel_name="my_kernel")
    assert "submit_kernel FAILED" in result
    assert "autotune_exclude violation" in result
    assert "num_stages" in result
    assert "output" not in captured  # captured stays empty on rejection


def test_submit_kernel_rejects_multi_key_exclude_match():
    """Multi-key pattern requires ALL listed keys to match the same config."""
    from src.agents.coder import _make_submit_tool
    from src.agents.planner import OptimizationPlan

    captured: dict = {}
    source = _kernel_source_with_configs(_FOUR_CONFIGS_WITH_STAGES4)
    plan = OptimizationPlan(
        tier=1, technique="t1",
        autotune_exclude=[{"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 4}],
    )
    submit = _make_submit_tool(captured, plan=plan)
    result = submit(source_code=source, triton_kernel_name="my_kernel")
    assert "submit_kernel FAILED" in result
    assert "autotune_exclude violation" in result


def test_submit_kernel_no_match_when_partial_key_mismatch():
    """Pattern ``{"BLOCK_M": 128, "num_stages": 4}`` does NOT match a
    config with BLOCK_M=128 but num_stages=3 — partial-match requires
    ALL listed keys equal."""
    from src.agents.coder import _make_submit_tool
    from src.agents.llm_backend import SUBMIT_OK_SENTINEL
    from src.agents.planner import OptimizationPlan

    captured: dict = {}
    source = _kernel_source_with_configs(_FOUR_CONFIGS_NO_STAGES4)
    plan = OptimizationPlan(
        tier=1, technique="t1",
        autotune_exclude=[{"BLOCK_M": 128, "num_stages": 4}],
    )
    submit = _make_submit_tool(captured, plan=plan)
    result = submit(source_code=source, triton_kernel_name="my_kernel")
    assert result == SUBMIT_OK_SENTINEL


def test_submit_kernel_error_message_names_violations_and_patterns():
    """Error message lists BOTH the exclude patterns AND the specific
    configs that violated, plus the closing reminder that the ≥4-config
    minimum still applies."""
    from src.agents.coder import _make_submit_tool
    from src.agents.planner import OptimizationPlan

    captured: dict = {}
    source = _kernel_source_with_configs(_FOUR_CONFIGS_WITH_STAGES4)
    plan = OptimizationPlan(
        tier=1, technique="t1",
        autotune_exclude=[{"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 4}],
    )
    submit = _make_submit_tool(captured, plan=plan)
    result = submit(source_code=source, triton_kernel_name="my_kernel")
    assert "'BLOCK_M': 128" in result
    assert "'BLOCK_N': 128" in result
    assert "'num_stages': 4" in result
    assert "exclude pattern:" in result
    assert "≥4-config minimum still applies" in result


def test_submit_kernel_rejects_multiple_violations_in_one_call():
    """Multiple violating configs render as multiple lines in the error
    message — the Coder sees all offenders at once."""
    from src.agents.coder import _make_submit_tool
    from src.agents.planner import OptimizationPlan

    captured: dict = {}
    source = _kernel_source_with_configs(
        '        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),\n'
        '        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),\n'
        '        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),\n'
        '        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),'
    )
    plan = OptimizationPlan(
        tier=1, technique="t1",
        autotune_exclude=[{"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 4}],
    )
    submit = _make_submit_tool(captured, plan=plan)
    result = submit(source_code=source, triton_kernel_name="my_kernel")
    assert result.count("matches autotune_exclude") == 2


def test_submit_kernel_no_op_when_plan_is_none():
    """``plan=None`` (translate baseline path's default) → exclude check
    is skipped regardless of source contents."""
    from src.agents.coder import _make_submit_tool
    from src.agents.llm_backend import SUBMIT_OK_SENTINEL

    captured: dict = {}
    source = _kernel_source_with_configs(_FOUR_CONFIGS_WITH_STAGES4)
    submit = _make_submit_tool(captured, plan=None)  # baseline path
    result = submit(source_code=source, triton_kernel_name="my_kernel")
    assert result == SUBMIT_OK_SENTINEL


# ── hw-spec injection (hw-spec injection Task 3) ───────────────────────


def test_coder_build_user_prompt_includes_run_context_when_bottleneck_and_hardware_present():
    """Coder prompt now carries the run-context block when both bottleneck+hardware passed."""
    from src.agents.coder import CoderAgent
    from src.agents.planner import OptimizationPlan
    from src.config import HardwareSpec
    from src.eval.types import BottleneckType

    plan = OptimizationPlan(
        tier=1, technique="t1_coalesced_load",
        params={}, target_region="", rationale="r",
    )
    hw = HardwareSpec(
        name="TestGPU",
        compute_capability=8.9,
        freq_GHz=2.0,
        DRAM_byte_per_cycle=400,
        MAC_per_cycle_fp32_sm=1000,
        shared_mem_per_block_bytes=101376,
        shared_mem_per_multiprocessor_bytes=102400,
    )
    prompt = CoderAgent.build_user_prompt(
        kernel_source="def x(): pass",
        plan=plan,
        bottleneck=BottleneckType.MEMORY_BOUND,
        hardware=hw,
    )
    assert "## Run context" in prompt
    assert "Hardware: TestGPU" in prompt
    assert "Shared mem per block: 101376 B" in prompt


def test_coder_build_user_prompt_omits_run_context_when_bottleneck_none():
    """Backward-compat: no bottleneck AND no hardware → no run-context section."""
    from src.agents.coder import CoderAgent
    from src.agents.planner import OptimizationPlan

    plan = OptimizationPlan(
        tier=1, technique="t1_coalesced_load",
        params={}, target_region="", rationale="r",
    )
    prompt = CoderAgent.build_user_prompt(kernel_source="def x(): pass", plan=plan)
    assert "## Run context" not in prompt


def test_coder_build_user_prompt_renders_hw_block_when_bottleneck_none():
    """Pre-classification path: hardware populated but bottleneck=None.

    The Coder still needs to see the SMEM cap so it can author autotune
    configs within budget on the first attempt. The previous gate
    (``if bottleneck is not None``) dropped the entire Run-context block
    when only hardware was provided; the new gate mirrors
    ``build_translate_prompt`` and renders when EITHER signal is set.
    Codex P-HIGH 2026-05-25, fix #4.
    """
    from src.agents.coder import CoderAgent
    from src.agents.planner import OptimizationPlan
    from src.config import HardwareSpec

    plan = OptimizationPlan(
        tier=1, technique="t1_coalesced_load",
        params={}, target_region="", rationale="r",
    )
    hw = HardwareSpec(
        name="TestGPU",
        compute_capability=8.9,
        freq_GHz=2.0,
        DRAM_byte_per_cycle=400,
        MAC_per_cycle_fp32_sm=1000,
        shared_mem_per_block_bytes=101376,
        shared_mem_per_multiprocessor_bytes=102400,
    )
    prompt = CoderAgent.build_user_prompt(
        kernel_source="def x(): pass",
        plan=plan,
        bottleneck=None,
        hardware=hw,
    )
    assert "## Run context" in prompt
    assert "Hardware: TestGPU" in prompt
    assert "Shared mem per block: 101376 B" in prompt


def test_coder_build_user_prompt_threads_workload_shapes():
    """workload_shapes kwarg threads through both build_user_prompt and
    build_translate_prompt to render_run_context."""
    from src.agents.coder import CoderAgent
    from src.agents.planner import OptimizationPlan
    from src.config import HardwareSpec
    from src.eval.types import BottleneckType
    from src.kernels.kernel import KernelSpec, KernelType

    plan = OptimizationPlan(
        tier=1, technique="t1_coalesced_load",
        params={}, target_region="", rationale="r",
    )
    hw = HardwareSpec(
        name="TestGPU",
        compute_capability=8.9,
        freq_GHz=2.0,
        DRAM_byte_per_cycle=400,
        MAC_per_cycle_fp32_sm=1000,
        shared_mem_per_block_bytes=101376,
        shared_mem_per_multiprocessor_bytes=102400,
    )
    shapes = [(1024, 4096, 2048), (2048, 4096, 2048)]
    prompt = CoderAgent.build_user_prompt(
        kernel_source="def x(): pass",
        plan=plan,
        bottleneck=BottleneckType.MEMORY_BOUND,
        hardware=hw,
        workload_shapes=shapes,
    )
    assert "Workload shapes:" in prompt
    assert "(1024, 4096, 2048)" in prompt

    spec = KernelSpec(name="k", kernel_type=KernelType.MATMUL, entrypoint="x")
    translate_prompt = CoderAgent.build_translate_prompt(
        reference_source="def x(): pass",
        kernel_spec=spec,
        bottleneck=None,
        hardware=hw,
        workload_shapes=shapes,
    )
    assert "Workload shapes:" in translate_prompt
    assert "(1024, 4096, 2048)" in translate_prompt


# ── compile_kernel_tool kernel-name auto-derivation ───────────────────


def _make_kernel_spec_for_smem_tests():
    """Minimal KernelSpec for the compile-tool auto-derivation tests."""
    from src.kernels.kernel import KernelSpec, KernelType
    return KernelSpec(
        name="smem_test_kernel",
        kernel_type=KernelType.MATMUL,
        entrypoint="x",
    )


def test_compile_tool_auto_derives_triton_kernel_name_from_source(monkeypatch):
    """In the production Coder tool flow, triton_kernel_name is declared
    only at submit_kernel time, not at compile_kernel_tool time. So
    Kernel(source) would be constructed with kernel_name='' unless the
    name is auto-derived from the source.

    The tool auto-derives the name from source (exactly one @triton.jit
    def → use it; multiple or zero → leave empty) so __post_init__ parses
    autotune_configs / autotune_keys against the right def.
    """
    from src.agents.coder import _make_compile_tool
    from src.kernels.compiler import CompilationResult

    captured_kernel_names: list[str] = []

    def _capture_compile(kernel, cache_dir=None):
        captured_kernel_names.append(kernel.triton_kernel_name)
        return CompilationResult(
            success=True, error_message="", compiled_fn=lambda: None,
            triton_autotuner=None,
        )

    monkeypatch.setattr("src.agents.coder.compile_kernel", _capture_compile)

    tool = _make_compile_tool(_make_kernel_spec_for_smem_tests())
    src_one_jit = (
        "import triton\n"
        "@triton.jit\n"
        "def my_kernel(x_ptr, BLOCK: tl.constexpr):\n"
        "    pass\n"
    )
    out = tool(src_one_jit)
    assert "Compilation successful" in out
    # compile_kernel was called with kernel.triton_kernel_name='my_kernel'.
    assert captured_kernel_names == ["my_kernel"]


def test_compile_tool_leaves_kernel_name_empty_on_multi_jit_source(monkeypatch):
    """Two @triton.jit defs → can't auto-disambiguate → leave kernel_name
    empty."""
    from src.agents.coder import _make_compile_tool
    from src.kernels.compiler import CompilationResult

    captured_kernel_names: list[str] = []

    def _capture_compile(kernel, cache_dir=None):
        captured_kernel_names.append(kernel.triton_kernel_name)
        return CompilationResult(
            success=True, error_message="", compiled_fn=lambda: None,
            triton_autotuner=None,
        )

    monkeypatch.setattr("src.agents.coder.compile_kernel", _capture_compile)

    tool = _make_compile_tool(_make_kernel_spec_for_smem_tests())
    src_two_jits = (
        "import triton\n"
        "@triton.jit\n"
        "def helper_kernel(x_ptr, BLOCK: tl.constexpr):\n"
        "    pass\n"
        "@triton.jit\n"
        "def main_kernel(x_ptr, BLOCK: tl.constexpr):\n"
        "    pass\n"
    )
    out = tool(src_two_jits)
    assert "Compilation successful" in out
    assert captured_kernel_names == [""]


# ── kernel.autotune_configs target-aware parse (fix #14) ───────────────


def test_compile_tool_kernel_autotune_configs_match_resolved_name(monkeypatch):
    """A source with two ``@triton.autotune`` decorators must parse
    ``kernel.autotune_configs`` against the resolved ``triton_kernel_name``
    rather than picking up the first decorator it walks. Previously,
    ``Kernel(spec, source)`` ran ``__post_init__`` with an empty name,
    matching the FIRST autotuned function and silently mis-attributing
    its configs to whatever the auto-derivation later resolves as the
    primary kernel. The fix passes the resolved name to the constructor
    so ``__post_init__`` parses with the right target."""
    from src.agents.coder import _make_compile_tool
    from src.kernels.compiler import CompilationResult

    captured_kernels: list = []

    def _capture_compile(kernel, cache_dir=None):
        captured_kernels.append(kernel)
        return CompilationResult(
            success=True, error_message="", compiled_fn=lambda: None,
            triton_autotuner=None,
        )

    monkeypatch.setattr("src.agents.coder.compile_kernel", _capture_compile)

    # Source has two @triton.jit defs, but ONLY the second has @triton.autotune.
    # Auto-derive: ``triton_kernel_names_in`` returns [helper, main]; len !=1 →
    # resolved_name stays empty. Verify the empty-name case still produces
    # ZERO autotune_configs from the helper (no @autotune on helper, only on
    # main_kernel), avoiding the legacy "first @autotune anywhere" attribution.
    src = (
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def helper_kernel(x_ptr, BLOCK: tl.constexpr):\n"
        "    pass\n"
        "\n"
        "@triton.autotune(\n"
        "    configs=[\n"
        '        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),\n'
        '        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),\n'
        '        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=3),\n'
        '        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=4),\n'
        "    ],\n"
        '    key=["x"],\n'
        ")\n"
        "@triton.jit\n"
        "def main_kernel(x_ptr, BLOCK: tl.constexpr):\n"
        "    pass\n"
    )

    tool = _make_compile_tool(_make_kernel_spec_for_smem_tests())
    out = tool(src)
    assert "Compilation successful" in out

    # Two JIT defs → resolved_name=='' → Kernel parses autotune with no
    # target filter; legacy "first @autotune anywhere" behavior surfaces
    # the SECOND function's decorator because that's the only one.
    k = captured_kernels[0]
    assert k.triton_kernel_name == ""
    assert len(k.autotune_configs) == 4  # parsed from main_kernel's decorator


def test_compile_tool_kernel_autotune_configs_target_aware_on_single_jit(monkeypatch):
    """Single @triton.jit → resolved_name is that name. The constructor
    parses ``autotune_configs`` filtered against that name, so a source
    with a misplaced (and irrelevant) decorator above an unrelated function
    doesn't pollute the primary kernel's configs.

    This is the meaningful regression: previously ``Kernel(spec, src)``
    parsed with name='', could pick up wrong @autotune; now the resolved
    name flows into __post_init__ so the right target is matched.
    """
    from src.agents.coder import _make_compile_tool
    from src.kernels.compiler import CompilationResult

    captured_kernels: list = []

    def _capture_compile(kernel, cache_dir=None):
        captured_kernels.append(kernel)
        return CompilationResult(
            success=True, error_message="", compiled_fn=lambda: None,
            triton_autotuner=None,
        )

    monkeypatch.setattr("src.agents.coder.compile_kernel", _capture_compile)

    # Exactly one @triton.jit def → auto-derived name flows into ctor.
    src = (
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.autotune(\n"
        "    configs=[\n"
        '        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),\n'
        '        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),\n'
        '        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=3),\n'
        '        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=4),\n'
        "    ],\n"
        '    key=["x"],\n'
        ")\n"
        "@triton.jit\n"
        "def main_kernel(x_ptr, BLOCK: tl.constexpr):\n"
        "    pass\n"
    )

    tool = _make_compile_tool(_make_kernel_spec_for_smem_tests())
    out = tool(src)
    assert "Compilation successful" in out

    k = captured_kernels[0]
    assert k.triton_kernel_name == "main_kernel"
    # autotune_configs was parsed against the resolved name, picks up
    # the decorator above main_kernel.
    assert len(k.autotune_configs) == 4
