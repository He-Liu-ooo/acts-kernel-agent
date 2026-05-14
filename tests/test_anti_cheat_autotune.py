"""Tier-2 GPU: ``per_iter_anti_cheat`` tolerance for Triton compile threads (A1 PR 1).

Validates that running an autotune-bearing kernel inside the anti-cheat
context manager does not trigger spurious ``check_thread_injection``
failures from Triton's own compile-time thread pool. If this test
fails, that's the §1.5 signal in the spec to widen tolerance or move
the thread-count check to a post-burn-in baseline.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu]


def test_autotune_inside_anti_cheat_no_spurious_failure():
    """An autotune burn-in inside ``per_iter_anti_cheat`` completes cleanly.

    Failure mode worth watching: Triton may spawn parallel compile
    threads during a 4-config autotune; if SOL's ``check_thread_injection``
    has zero tolerance, the exit-side check would raise on the elevated
    ``threading.active_count()``. We accept the test failing as the
    legitimate trigger for Task 9 (tolerance widening or check
    relocation), not a bug.
    """
    import torch
    import triton
    import triton.language as tl

    from src.eval.anti_cheat import per_iter_anti_cheat

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 64},  num_warps=2, num_stages=2),
            triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        y = tl.load(Y + offs, mask=mask)
        tl.store(Z + offs, x + y, mask=mask)

    N = 8192
    X = torch.randn(N, device="cuda")
    Y = torch.randn(N, device="cuda")
    Z = torch.empty(N, device="cuda")

    critical_names = ["elapsed_time", "synchronize", "wait", "record", "query"]

    with per_iter_anti_cheat(critical_names):
        add_kernel[(triton.cdiv(N, 64),)](X, Y, Z, N)
        torch.cuda.synchronize()
