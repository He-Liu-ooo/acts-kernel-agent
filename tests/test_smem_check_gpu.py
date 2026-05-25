"""Tier 2 GPU tests for the compile-side SMEM check (recorder-patch redesign).

Run via:
    source ~/.venvs/acts_run_venv/bin/activate
    python -m pytest tests/test_smem_check_gpu.py -v -m gpu

Requires NVIDIA GPU + Triton install (cu128 venv per configs/venvs/3.12.md).
The mocked Tier 1 counterpart lives in tests/test_smem_check.py.

These tests pin the Triton-side contract:

* ``CompiledKernel.metadata.shared`` is the canonical ptxas-SMEM accessor
  (verified on Triton 3.6.0; legacy ``.shared`` fallback covered by Tier-1).
* ``JITFunction.device_caches[dev_idx][0]`` is the
  ``dict[signature_str, CompiledKernel]`` populated by ``warmup()``. Older
  Triton's ``JITFunction.cache`` is also supported via fallback in
  ``_latest_cache_entry``.
* The instance-attribute monkey-patch on ``autotuner.run`` is honored by
  Triton's ``KernelInterface.__getitem__`` late-bound ``self.run`` lookup.

The recorder-patch redesign (spec §6, 2026-05-25) fixed the prior
limitation where SMEM check was vacuous on host-wrapper matmul kernels.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.gpu


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_triton(), reason="Triton not installed (Tier-1 venv)")
def test_real_triton_autotuner_metadata_shared_present():
    """Pins the duck-typed contract: current Triton exposes
    ``CompiledKernel.metadata.shared`` reachable through
    ``JITFunction.device_caches`` (modern) or ``JITFunction.cache`` (legacy).
    """
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _noop_kernel(x_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, v)

    x = torch.zeros((16,), device="cuda")
    _noop_kernel[(1,)](x, BLOCK=16, num_warps=1, num_stages=1)

    from src.eval.smem_check import _latest_cache_entry, _read_compiled_smem

    compiled = _latest_cache_entry(_noop_kernel)
    assert compiled is not None, (
        "JITFunction cache empty after launch — duck-typed accessor in "
        "_latest_cache_entry doesn't match this Triton version."
    )
    smem = _read_compiled_smem(compiled)
    assert smem is not None
    assert smem >= 0


@pytest.mark.skipif(not _has_triton(), reason="Triton not installed (Tier-1 venv)")
def test_recorder_patch_takes_effect_on_real_autotuner():
    """Pins the instance-attribute monkey-patch contract on real Triton.

    If a future Triton changes ``Autotuner.__getitem__`` or ``KernelInterface``
    to bypass the late-bound ``self.run`` lookup, this test fails BEFORE the
    SMEM check silently no-ops in production.
    """
    import torch
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 32}, num_warps=1, num_stages=1),
        ],
        key=["N"],
    )
    @triton.jit
    def _at_noop(x_ptr, N, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < N)
        tl.store(x_ptr + offs, v, mask=offs < N)

    captured: dict = {"called": 0, "args": None}

    def recording_run(*args, **kwargs):
        captured["called"] += 1
        captured["args"] = args
        return None  # bail without launching

    try:
        _at_noop.run = recording_run
        x = torch.zeros((32,), device="cuda")
        _at_noop[(1,)](x, 32)
    finally:
        try:
            del _at_noop.run
        except AttributeError:
            pass

    assert captured["called"] == 1, (
        "Recorder was NOT invoked — Autotuner.__getitem__ no longer does "
        "late-bound self.run lookup. The instance-attribute patch design "
        "is broken on this Triton version."
    )
    assert captured["args"] is not None
    # After del, autotuner.run should be the class method again.
    assert "run" not in _at_noop.__dict__


@pytest.mark.skipif(not _has_triton(), reason="Triton not installed (Tier-1 venv)")
def test_check_autotune_smem_budget_rejects_host_wrapper_matmul_overcommit():
    """End-to-end recorder-patch happy path on a host-wrapper matmul.

    The recorder drives the host wrapper once, captures full JIT args
    (including ``c``, ``M``, ``N``, ``K`` derived inside the wrapper),
    then warmups each Config with the captured args and reads ptxas
    SMEM. The deliberately overcommitted Config (256/256/64 fp32 +
    num_stages=4) overflows Ada's 99 KB per-block cap.

    Pins the fix for the prior production limitation (JOURNAL 2026-05-24
    "Phase B production-shape limitation").
    """
    import torch
    import triton
    import triton.language as tl

    from src.config import detect_hardware
    from src.eval.smem_check import check_autotune_smem_budget

    hw = detect_hardware()
    if hw.shared_mem_per_block_bytes == 0:
        pytest.skip("GPU detection returned 0 SMEM cap")

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16},
                          num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64},
                          num_warps=8, num_stages=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        c = tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c)

    def matmul(a, b):
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        matmul_kernel[(1,)](a, b, c, M, N, K)
        return c

    a = torch.randn((32, 32), device="cuda", dtype=torch.float32)
    b = torch.randn((32, 32), device="cuda", dtype=torch.float32)

    violations = check_autotune_smem_budget(
        matmul_kernel, matmul, sample_args=(a, b),
        cap_bytes=hw.shared_mem_per_block_bytes,
    )
    assert len(violations) >= 1, (
        f"Expected ≥1 SMEM violation for overcommitted Config "
        f"(BLOCK_M=256/BLOCK_N=256/BLOCK_K=64/num_stages=4, fp32) but got "
        f"none. cap={hw.shared_mem_per_block_bytes}; violations={violations}"
    )
    # config_idx=1 is the overcommit
    assert any(v.config_idx == 1 for v in violations), (
        f"Expected config_idx=1 in violations; got "
        f"{[v.config_idx for v in violations]}"
    )


@pytest.mark.skipif(not _has_triton(), reason="Triton not installed (Tier-1 venv)")
def test_compile_kernel_tool_rejects_smem_overflow_end_to_end(tmp_path):
    """End-to-end production-path integration test.

    Exercises the full ``_make_compile_tool → Kernel(source) →
    compile_kernel → _resolve_triton_autotuner → check_autotune_smem_budget
    → rejection`` chain on a real host-wrapper matmul with an
    overcommitted Config. This is the path Codex's P2 finding flagged
    as broken because ``triton_kernel_name`` was empty at
    compile_kernel_tool time and ``_resolve_triton_autotuner`` returned
    None.

    Earlier unit tests (in tests/test_smem_check_gpu.py and the Tier-1
    coder mocks) bypassed this chain — they called the helpers directly
    or stubbed compile_kernel to return a result with a pre-resolved
    triton_autotuner. This test runs the production path top-to-bottom
    so a future regression in autotuner resolution surfaces here, not
    via empty events.jsonl entries in a real run.
    """
    import torch
    from src.agents.coder import _make_compile_tool
    from src.config import detect_hardware
    from src.kernels.kernel import KernelSpec, KernelType

    hw = detect_hardware()
    if hw.shared_mem_per_block_bytes == 0:
        pytest.skip("GPU detection returned 0 SMEM cap")

    # Real host-wrapper matmul kernel source. The deliberately
    # overcommitted Config (256/256/64 fp32, num_stages=4) overflows
    # Ada's 99 KB per-block cap. The host wrapper passes positional ptrs
    # and positional dims (so kwargs replay isn't being tested here —
    # the keyword-launch path is covered in tests/test_smem_check.py
    # ``test_check_autotune_smem_budget_replays_recorded_kwargs``).
    src = '''
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c)


def matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    matmul_kernel[(1,)](a, b, c, M, N, K)
    return c
'''
    src_path = tmp_path / "overcommit_matmul.py"
    src_path.write_text(src)

    spec = KernelSpec(
        name="overcommit_matmul",
        kernel_type=KernelType.MATMUL,
        entrypoint="matmul",
    )

    a = torch.randn((32, 32), device="cuda", dtype=torch.float32)
    b = torch.randn((32, 32), device="cuda", dtype=torch.float32)

    error_log: list[str] = []
    tool = _make_compile_tool(
        spec, error_log=error_log, hardware=hw, sample_args=(a, b),
    )
    out = tool(src)

    assert out.startswith("Compile FAILED: shared-memory budget exceeded"), (
        f"Production-path SMEM rejection did NOT fire. Tool returned: {out[:300]}\n"
        "If this is 'Compilation successful', the autotuner was never resolved "
        "(check triton_kernel_name auto-derivation in _make_compile_tool) or "
        "the SMEM helper failed open silently."
    )
    assert error_log == [out]  # cross-attempt memory captures the rejection
