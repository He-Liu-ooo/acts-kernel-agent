"""Tier 2 real-GPU tests for ``src.eval.profiler``.

Gated by ``@pytest.mark.gpu`` at module scope — skipped by the default
``/tmp/acts_test_venv`` Tier 1 run. These tests exercise the full
analytical + NCU pipeline against the live ``ncu`` binary and a real
Triton kernel on the RTX 6000 Ada.

Guarded invariants (from memory file ``profiler_impl_progress.md`` — the
7 NCU discoveries the original spec didn't anticipate):

1. End-to-end: ``profile_kernel`` returns ``ncu is not None`` on a real
   Triton kernel — analytical + NCU paths compose correctly.
2. NCU metric-name stability on CUDA 12.8 / NCU 2025.1.1.0 — the curated
   raw-dotted metric names actually appear in output (guards against
   metric rename across NCU versions; Tier 1 cannot detect this because
   the fake CSV always has the right names).
3. Triton kernel name is mangled (e.g. ``vectorized_elementwise_kernel
   <4, FillFunctor<float>, ...>``); the substring match in
   ``_parse_ncu_csv`` handles it correctly in real output.
4. Cache hit on the second call skips the subprocess.
5. NCU-side failures degrade the result but do NOT raise.
6. Analytical failures ARE branch-killing (``ProfilerError``).
7. Warp-stall top-1 + runner-up are extracted and non-empty from real HW
   counter samples.

Environment notes
-----------------
- Astra's venv (``/home/hel19/workspace/projects/self-evolved-llm/repo/
  Astra/.venv``) has a torch / triton / sol_execbench stack known to
  work with the host's CUDA 12.8 driver. SOL-ExecBench's own venv ships
  a too-new torch ("NVIDIA driver too old, found 12080").
- Astra's venv does not include ``pytest`` by default — install with
  ``pip install pytest pyyaml`` inside that venv (leaves the torch
  install untouched).
- Run with ``PYTHONPATH=.`` from the project root so ``src.*`` imports
  resolve without a full editable install.
- NCU lock-file workaround (memory discovery #7): export
  ``TMPDIR=/tmp/<user>_ncu`` before invocation. The profiler's own
  ``_ncu_tmpdir`` already does this.

Kernel-source convention
------------------------
These tests exercise the full production path: ``profile_kernel`` calls
``compile_kernel`` to materialise the kernel under
``.acts_cache/compiled/``, extracts the Triton GPU symbol via the
``@triton.jit`` heuristic for NCU's ``--kernel-name regex:`` filter,
and launches ``src.eval._profiler_driver``, which imports the module
and calls ``module.run(*inputs)`` (host-wrapper convention). The test
kernel therefore exposes ``def run(x, y)`` as its host wrapper.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


# ── hard environment preconditions ────────────────────────────────────────

torch = pytest.importorskip("torch", reason="Tier 2 requires torch + CUDA")
triton = pytest.importorskip("triton", reason="Tier 2 requires Triton")

if not torch.cuda.is_available():  # pragma: no cover — Tier 2 gated on live GPU
    pytest.skip("No CUDA GPU available", allow_module_level=True)

if shutil.which("ncu") is None:  # pragma: no cover — Tier 2 gated on ncu binary
    pytest.skip("ncu binary not on PATH", allow_module_level=True)


# Project modules are safe to import after the gate passes.
from conftest import rtx6000_ada_hardware as _rtx6000_ada  # noqa: E402
from src.eval import profiler as profiler_mod  # noqa: E402
from src.eval.profiler import (  # noqa: E402
    ProfilerError,
    ProfilingResult,
    profile_kernel,
)
from src.kernels.kernel import Kernel, KernelSpec, KernelType  # noqa: E402


# ── Triton elementwise kernel: deliberately memory-bound ──────────────────

_ELEMENTWISE_SOURCE = '''\
"""Memory-bound Triton elementwise kernel used by the Tier 2 profiler
smoke test. ``run`` is the host wrapper the profiler driver calls;
NCU sees the inner ``@triton.jit`` kernel by its mangled symbol."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def elementwise_add_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    elementwise_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def make_inputs(seed: int = 0):
    """Driver convention: the kernel source is self-contained — expose a
    ``make_inputs`` so the NCU subprocess can reconstruct inputs without
    the parent's unpicklable closure."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n = 1 << 19
    x = torch.empty(n, device="cuda", dtype=torch.float32).normal_(generator=gen)
    y = torch.empty(n, device="cuda", dtype=torch.float32).normal_(generator=gen)
    return (x, y)
'''

# spec.entrypoint names the host wrapper (Python-side callable). The
# profiler extracts the Triton GPU symbol from source for NCU filtering,
# so ``run`` here corresponds to ``module.run`` in the compiled kernel.
_KERNEL_ENTRYPOINT = "run"


def _input_generator(seed: int = 0) -> tuple:
    """Return two CUDA fp32 tensors of 2 MiB each. Large enough for the
    kernel to exhibit real memory-bandwidth behaviour; small enough that
    NCU's replay overhead stays under the default timeout."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n = 1 << 19  # 524_288 elements × 4 bytes = 2 MiB per tensor
    x = torch.empty(n, device="cuda", dtype=torch.float32).normal_(generator=gen)
    y = torch.empty(n, device="cuda", dtype=torch.float32).normal_(generator=gen)
    return (x, y)


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tier2_kernel() -> Kernel:
    """Tier 2 kernel wrapping ``_ELEMENTWISE_SOURCE``. ``profile_kernel``
    materialises the source via ``compile_kernel`` itself — no test-side
    tmp_path plumbing is needed."""
    return Kernel(
        spec=KernelSpec(
            name="elementwise_add_tier2",
            kernel_type=KernelType.ELEMENTWISE,
            entrypoint=_KERNEL_ENTRYPOINT,
        ),
        source_code=_ELEMENTWISE_SOURCE,
    )


@pytest.fixture
def measured_latency_s(tier2_kernel) -> float:
    """Measure one warm-up then a small timed burst with torch.cuda events.
    Used as ``latency_s`` input to the analytical path; needs to be the
    real per-launch latency so the roofline ratios are sensible."""
    from src.kernels.compiler import compile_kernel as _compile_kernel

    result = _compile_kernel(tier2_kernel)
    assert result.success, f"compile failed: {result.error_message}"
    run = result.compiled_fn

    inputs = _input_generator(seed=0)
    for _ in range(5):
        run(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iters = 50
    start.record()
    for _ in range(iters):
        run(*inputs)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return (ms / iters) / 1000.0


@pytest.fixture
def ncu_call_counter(monkeypatch):
    """Count ``_run_ncu`` invocations without altering behaviour. Used by
    the cache-hit test to prove the second ``profile_kernel`` call does
    NOT re-enter the subprocess path."""
    state = {"calls": 0}
    real = profiler_mod._run_ncu

    def spy(*args, **kwargs):
        state["calls"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(profiler_mod, "_run_ncu", spy)
    monkeypatch.setattr(profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False)
    return state


# ── Test 1: real Triton kernel end-to-end ────────────────────────────────


def test_profile_real_triton_kernel_end_to_end(
    tmp_path, tier2_kernel, measured_latency_s
):
    """End-to-end: analytical + NCU paths compose and return a populated
    ``ProfilingResult`` on a real Triton kernel. Guards against
    regressions in how the orchestrator would call profile_kernel — any
    failure in analytical math, driver invocation, or CSV parsing
    surfaces here."""
    result = profile_kernel(
        tier2_kernel,
        {"uuid": "tier2-0", "axes": {"N": 524288}, "inputs": {}},
        _input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=524_288,          # 1 add per element
        nbytes=524_288 * 4 * 3, # 2 reads + 1 write, fp32
        latency_s=measured_latency_s,
        mode="curated",
        timeout_s=120.0,
        cache_dir=tmp_path / "cache",
    )
    assert isinstance(result, ProfilingResult)
    assert result.degraded is False, f"NCU path degraded: {result.degraded_reason}"
    assert result.ncu is not None
    assert result.analytical is not None
    assert result.analytical.arithmetic_intensity >= 0
    assert result.analytical.achieved_bandwidth_gb_s > 0


# ── Test 2: curated NCU metric names still exist on this NCU version ─────


def test_curated_metric_names_present_in_real_ncu_output(
    tmp_path, tier2_kernel, measured_latency_s
):
    """On NCU 2025.1.1.0 / RTX 6000 Ada the curated raw-dotted metric
    names must appear in ``result.raw_metrics``. If NVIDIA renames a
    metric between NCU releases, this test fails and we know to update
    ``_CURATED_REQUIRED``. Tier 1 can't catch this — the fake CSV always
    has the curated names hard-coded."""
    result = profile_kernel(
        tier2_kernel,
        {"uuid": "tier2-1", "axes": {"N": 524288}, "inputs": {}},
        _input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=524_288,
        nbytes=524_288 * 4 * 3,
        latency_s=measured_latency_s,
        mode="curated",
        timeout_s=120.0,
        cache_dir=tmp_path / "cache",
    )
    assert result.degraded is False, f"NCU path degraded: {result.degraded_reason}"
    assert "sm__warps_active.avg.pct_of_peak_sustained_active" in result.raw_metrics
    assert "lts__t_sector_hit_rate.pct" in result.raw_metrics
    for name, value in result.raw_metrics.items():
        assert value == value, f"{name} is NaN"


# ── Test 3: mangled Triton kernel name substring-matches the extracted name ──


def test_triton_mangled_name_matches_extracted_name(
    tmp_path, tier2_kernel, measured_latency_s
):
    """Triton mangles the kernel name into a C++-style template
    expansion (e.g. ``elementwise_add_kernel_0d1d...``). The parser
    substring-matches on the name extracted from ``@triton.jit def`` —
    this test proves the real mangling survives that match on-GPU."""
    result = profile_kernel(
        tier2_kernel,
        {"uuid": "tier2-2", "axes": {"N": 524288}, "inputs": {}},
        _input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=524_288,
        nbytes=524_288 * 4 * 3,
        latency_s=measured_latency_s,
        mode="curated",
        timeout_s=120.0,
        cache_dir=tmp_path / "cache",
    )
    assert result.degraded is False, (
        f"substring match failed on real Triton mangled name: {result.degraded_reason}"
    )
    assert result.ncu is not None


# ── Test 4: second call is a cache hit, subprocess NOT spawned ───────────


def test_cache_hit_skips_subprocess_on_real_gpu(
    tmp_path, tier2_kernel, measured_latency_s, ncu_call_counter
):
    """First ``profile_kernel`` call populates the on-disk cache; second
    identical call must rehydrate from JSON without spawning ncu. Uses
    a ``_run_ncu`` invocation counter + a wallclock sanity check (cache
    hit should be orders of magnitude faster than a real NCU run)."""
    cache_dir = tmp_path / "cache"
    workload = {"uuid": "tier2-3", "axes": {"N": 524288}, "inputs": {}}

    t0 = time.perf_counter()
    first = profile_kernel(
        tier2_kernel, workload, _input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=524_288, nbytes=524_288 * 4 * 3,
        latency_s=measured_latency_s,
        mode="curated", timeout_s=120.0,
        cache_dir=cache_dir,
    )
    first_elapsed = time.perf_counter() - t0
    assert first.degraded is False
    assert ncu_call_counter["calls"] == 1

    t1 = time.perf_counter()
    second = profile_kernel(
        tier2_kernel, workload, _input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=524_288, nbytes=524_288 * 4 * 3,
        latency_s=measured_latency_s,
        mode="curated", timeout_s=120.0,
        cache_dir=cache_dir,
    )
    second_elapsed = time.perf_counter() - t1

    assert ncu_call_counter["calls"] == 1, "cache hit must not invoke _run_ncu"
    assert second.ncu is not None
    assert second_elapsed < first_elapsed / 5, (
        f"cache hit took {second_elapsed:.3f}s vs first call "
        f"{first_elapsed:.3f}s — suspicious"
    )


# ── Test 5: NCU-side failure degrades but does not raise ─────────────────


def test_ncu_failure_degrades_but_does_not_raise(
    tmp_path, tier2_kernel, measured_latency_s
):
    """Forcing an absurdly tight NCU timeout makes the subprocess fail
    with ``ncu_timeout``. Invariant: ``profile_kernel`` returns a
    degraded result with ``ncu=None`` and ``analytical`` still
    populated — it does NOT raise. Orchestrator relies on this to keep
    the branch alive when only the NCU signal is missing."""
    result = profile_kernel(
        tier2_kernel,
        {"uuid": "tier2-4", "axes": {"N": 524288}, "inputs": {}},
        _input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=524_288,
        nbytes=524_288 * 4 * 3,
        latency_s=measured_latency_s,
        mode="curated",
        timeout_s=0.001,
        cache_dir=tmp_path / "cache",
    )
    assert result.degraded is True
    assert result.ncu is None
    assert result.degraded_reason is not None
    assert (
        "timeout" in result.degraded_reason
        or "nonzero_exit" in result.degraded_reason
    )
    assert result.analytical is not None


# ── Test 6: analytical failure IS branch-killing ─────────────────────────


def test_analytical_failure_raises_profiler_error(tmp_path, tier2_kernel):
    """``latency_s=0.0`` makes the roofline math undefined; invariant:
    ``profile_kernel`` raises ``ProfilerError`` so the orchestrator can
    kill the branch. Contrasts with NCU failures, which degrade."""
    with pytest.raises(ProfilerError):
        profile_kernel(
            tier2_kernel,
            {"uuid": "tier2-5", "axes": {"N": 524288}, "inputs": {}},
            _input_generator,
            hardware_spec=_rtx6000_ada(),
            flops=524_288,
            nbytes=524_288 * 4 * 3,
            latency_s=0.0,
            mode="curated",
            timeout_s=120.0,
            cache_dir=tmp_path / "cache",
        )


# ── Test 7: warp stall top-1 + runner-up are extracted from real run ─────


def test_warp_stall_dominant_and_runner_up_populated(
    tmp_path, tier2_kernel, measured_latency_s
):
    """Real HW-counter sampling populates the 18 ``smsp__average_warp_
    latency_issue_stalled_*.pct`` metrics; the parser ranks them and
    exposes top-1 / runner-up. Values depend on sampling noise — we
    assert only that both are non-empty strings and ``> 0``, not on
    specific stall reasons."""
    result = profile_kernel(
        tier2_kernel,
        {"uuid": "tier2-6", "axes": {"N": 524288}, "inputs": {}},
        _input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=524_288,
        nbytes=524_288 * 4 * 3,
        latency_s=measured_latency_s,
        mode="curated",
        timeout_s=120.0,
        cache_dir=tmp_path / "cache",
    )
    assert result.degraded is False, f"NCU path degraded: {result.degraded_reason}"
    assert result.ncu is not None
    assert isinstance(result.ncu.warp_stall_dominant, str)
    assert result.ncu.warp_stall_dominant != ""
    assert result.ncu.warp_stall_dominant_pct > 0
    assert result.ncu.warp_stall_runner_up != ""
    assert (
        result.ncu.warp_stall_dominant_pct
        >= result.ncu.warp_stall_runner_up_pct
    )


# ── Test 8: SOL-problem path end-to-end (problem_definition_path) ─────────


def _write_vecadd_sol_problem(root: Path, n: int = 524_288) -> Path:
    """Write a minimal SOL-ExecBench problem matching the elementwise
    kernel's ``run(x, y)`` signature: two fp32 vectors of length ``n`` in,
    one out. Returns the directory (the driver expects a *directory*,
    not the definition.json path — regression pin from Codex review)."""
    problem_dir = root / "sol_vecadd"
    problem_dir.mkdir()
    (problem_dir / "definition.json").write_text(json.dumps({
        "name": "vecadd_sol_tier2",
        "op_type": "elementwise",
        "axes": {"n": {"type": "const", "value": n}},
        "inputs": {
            "x": {"shape": ["n"], "dtype": "float32"},
            "y": {"shape": ["n"], "dtype": "float32"},
        },
        "outputs": {"z": {"shape": ["n"], "dtype": "float32"}},
        "reference": (
            "import torch\n"
            "def run(x, y):\n"
            "    return x + y\n"
        ),
    }))
    (problem_dir / "workload.jsonl").write_text(
        json.dumps({
            "uuid": "tier2-sol-0",
            "axes": {},
            "inputs": {"x": {"type": "random"}, "y": {"type": "random"}},
        }) + "\n"
    )
    return problem_dir


def test_profile_with_problem_definition_path_is_not_degraded(
    tmp_path, tier2_kernel, measured_latency_s
):
    """End-to-end regression for two Codex review findings:

    1. ``_run_ncu`` must serialize the problem *directory* (not
       ``definition.json``) into the spec JSON so the driver's
       ``load_problem`` call succeeds.
    2. When ``problem_definition_path`` is set, the in-process
       ``input_generator`` is ignored at the subprocess boundary — the
       driver must rebuild inputs from the SOL definition. This test
       passes an ``input_generator`` that would raise if called, proving
       that path is taken and non-degraded.

    Requires ``sol_execbench`` in the Tier 2 venv (it's what builds the
    input generator inside the NCU subprocess)."""
    pytest.importorskip(
        "sol_execbench",
        reason="SOL-ExecBench input generator required for problem_dir path",
    )

    problem_dir = _write_vecadd_sol_problem(tmp_path, n=524_288)
    definition_path = problem_dir / "definition.json"

    def _must_not_be_called(seed: int) -> tuple:
        raise AssertionError(
            "in-process input_generator was invoked — the subprocess "
            "should have rebuilt inputs from problem_dir instead"
        )

    result = profile_kernel(
        tier2_kernel,
        {
            "uuid": "tier2-sol-0",
            "axes": {},
            "inputs": {"x": {"type": "random"}, "y": {"type": "random"}},
        },
        _must_not_be_called,
        hardware_spec=_rtx6000_ada(),
        flops=524_288,
        nbytes=524_288 * 4 * 3,
        latency_s=measured_latency_s,
        mode="curated",
        timeout_s=120.0,
        cache_dir=tmp_path / "cache",
        problem_definition_path=definition_path,
    )

    assert result.degraded is False, (
        f"SOL problem_dir path degraded: {result.degraded_reason} "
        "— driver likely crashed on load_problem (pre-Codex-fix regression) "
        "or sol_execbench input generation failed"
    )
    assert result.ncu is not None
    assert result.analytical is not None
