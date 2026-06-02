"""Tier-2: subprocess isolation survives a context-poisoning candidate.

This is the regression proof for the correctness-subprocess isolation work
(plan ``doc/plans/2026-05-30-correctness-subprocess-isolation.md`` Task 6): a
candidate kernel that issues an out-of-bounds ``tl.store`` faults the CUDA
context. If that launch happened in-parent, every subsequent CUDA op in the
parent process would error ("CUDA error: an illegal memory access was
encountered"). By routing the launch through ``run_correctness_subprocess``,
the fault dies in the (now-dead) CHILD process and the PARENT's CUDA context
stays usable.

Expect noisy stderr: the child faulting prints device-side assert / illegal
memory access banners. That is the test working as designed — the only thing
that matters is the final pytest verdict.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

L1_DIR = Path(
    "/home/hel19/workspace/projects/self-evolved-llm/repo/benchmark/"
    "SOL-ExecBench/data/benchmark/L1/048_fused_gate_up_projection_with_swiglu"
)


# A minimal Triton kernel whose host entrypoint matches the spec's
# ``kernel_fn`` (x, gate_proj, up_proj) and that COMPILES cleanly, but whose
# device kernel issues an UNMASKED out-of-bounds ``tl.store`` at a wildly
# out-of-range offset (base + 2**42 elements ~= 16 TB past the buffer — far
# beyond any GPU address space, so the access is guaranteed unmapped and
# faults, rather than landing in some other mapped allocation). The host
# wrapper then calls ``torch.cuda.synchronize()`` so the device-side illegal
# memory access surfaces SYNCHRONOUSLY inside the (child) process — turning the
# fault into a deterministic worker crash instead of an async poison that might
# slip past the correctness comparison. The autotune block carries a
# well-formed >=4-config sweep with a non-empty key= to satisfy the repo's
# compiler contract.
_OOB_KERNEL_SOURCE = '''
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def oob_store_kernel(out_ptr, N, BLOCK: tl.constexpr):
    # Deliberately ignore N / masking: store unconditionally to an offset
    # 2**42 elements past the (tiny) output buffer. This is a launch-time
    # illegal memory access, not a compile error.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bad = offs.to(tl.int64) + 4398046511104  # 2**42
    tl.store(out_ptr + bad, tl.zeros((BLOCK,), dtype=tl.float32))


def kernel_fn(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    # Allocate a real (small) output so the kernel launches against a valid
    # base pointer; the in-kernel +2**42 offset is what faults.
    batch_size, seq_len, hidden_size = x.shape
    intermediate_size = gate_proj.shape[0]
    M = batch_size * seq_len
    out = torch.empty((M, intermediate_size), dtype=x.dtype, device=x.device)
    N = out.numel()
    grid = (1,)
    oob_store_kernel[grid](out, N)
    # Force the illegal-access fault to surface synchronously in THIS process.
    torch.cuda.synchronize()
    return out.reshape(batch_size, seq_len, intermediate_size)
'''


def _build_request(mode: str, source: str, dps: bool = False, seed: int = 0) -> dict:
    from src.benchmarks.sol_execbench import load as sol_load
    from src.search.orchestrator import _serialize_kernel_spec_for_request
    from src.pipeline.optimize import _definition_to_kernel_spec

    definition, workloads = sol_load(L1_DIR)
    spec = _definition_to_kernel_spec(definition, L1_DIR / "definition.json")
    return {
        "schema_version": 1,
        "mode": mode,
        "kernel_spec": _serialize_kernel_spec_for_request(spec),
        "source_code": source,
        "dps": dps,
        "definition_path": str(L1_DIR),
        "workloads": [w.model_dump(mode="json") for w in workloads[:1]],
        "blob_roots": [str(L1_DIR)],
        "input_seed": seed,
        "anti_cheat_critical_names": [],
    }


def test_oob_kernel_crashes_child_parent_survives(tmp_path):
    from src.eval.correctness_subprocess import run_correctness_subprocess

    res = asyncio.new_event_loop().run_until_complete(
        run_correctness_subprocess(
            request=_build_request("gate", _OOB_KERNEL_SOURCE),
            worker_dir=tmp_path / "w1",
            timeout_s=120.0,
        )
    )
    # The candidate must NOT pass. The IDEAL signal is worker_crashed (the
    # child process itself died on the illegal access); but the worker's
    # verify_correctness gate may instead CATCH the CUDA illegal-access as a
    # per-stage exception and report a clean correctness-gate failure (e.g.
    # smoke_test) — that is equally valid isolation: the poison was contained
    # in the child, which exited normally with a non-pass response. Any
    # non-pass stage is acceptable; what is NOT acceptable is the launch
    # poisoning the PARENT (asserted below).
    from src.eval.correctness import CorrectnessStage

    accepted_stages = (
        {"worker_crashed", "timeout", "compile"}
        | {s.value for s in CorrectnessStage}
    )
    assert res.passed is False
    assert res.failed_stage in accepted_stages, (
        f"unexpected failed_stage={res.failed_stage!r}; "
        f"error_message={res.error_message!r}"
    )

    # THE PROOF: the parent's CUDA context is intact afterward. If the OOB
    # launch had happened in-parent, these would raise "CUDA error: an
    # illegal memory access was encountered".
    import torch

    x = torch.randn(8, device="cuda")
    torch.cuda.synchronize()
    assert x.shape == (8,)

    # And the SOL input generator still produces healthy bfloat16 tensors in
    # the parent process.
    from src.eval.inputs import build_input_generator
    from src.benchmarks.sol_execbench import load as sol_load

    definition, workloads = sol_load(L1_DIR)
    gen = build_input_generator(definition, workloads[0], blob_roots=[L1_DIR])
    inputs = gen(0)
    tensor_inputs = [t for t in inputs if hasattr(t, "dtype")]
    assert tensor_inputs, "expected at least one tensor input from the generator"
    assert all(t.dtype == torch.bfloat16 for t in tensor_inputs)
