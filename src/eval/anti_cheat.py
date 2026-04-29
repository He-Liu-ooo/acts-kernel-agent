"""ACTS anti-cheat surface — coordinates correctness-level, performance-level,
and process-level reward-hack defenses.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator

import torch

from sol_execbench.core.bench.reward_hack import (
    RewardHackDetected,
    check_eval_integrity,
    check_lazy_outputs as _sol_check_lazy_outputs,
    check_monkey_patch,
    check_thread_injection,
    snapshot_critical_functions,
)

if TYPE_CHECKING:
    from src.kernels.kernel import Kernel


@dataclass
class AntiCheatContext:
    snapshot: dict[str, int]
    namespace: MappingProxyType[str, Any]
    threads_before: int


@contextmanager
def per_iter_anti_cheat(critical_names: list[str]) -> Iterator[AntiCheatContext]:
    threads_before = threading.active_count()
    namespace = vars(torch.cuda.Event)
    snapshot = snapshot_critical_functions(namespace, critical_names)
    # ``check_monkey_patch`` validates the module-load ``_ELAPSED_TIME_ADDR``
    # snapshot — fires only if a candidate kernel patched
    # ``torch.cuda.Event.elapsed_time`` after sol_execbench import. The
    # import-order contract in ``pipeline/optimize.py`` guards against this
    # by importing SOL first.
    check_monkey_patch()
    ctx = AntiCheatContext(snapshot=snapshot, namespace=namespace, threads_before=threads_before)
    try:
        yield ctx
    finally:
        # Run exit-side checks unconditionally so a body that raised for an
        # unrelated reason still gets validated; a tampered run is more
        # important to surface than whatever caused the body to error, so
        # letting these shadow an in-body exception is acceptable.
        check_thread_injection(threads_before, threading.active_count())
        check_eval_integrity(snapshot, namespace)


def check_lazy_outputs_after_bench(outputs: list) -> None:
    # SOL's check uses strict ``type(t) is torch.Tensor`` — any subclass
    # (FakeTensor, lazy proxies) is rejected.
    if not outputs:
        return
    _sol_check_lazy_outputs(outputs)


def generate_randomized_inputs(
    input_generator,
    seed: int,
) -> list:
    return list(input_generator(seed))


def strict_tolerance_check(
    candidate_output: "torch.Tensor",
    reference_output: "torch.Tensor",
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    from sol_execbench.core.bench.correctness import compute_error_stats
    from sol_execbench.core.data.workload import ToleranceSpec

    spec = ToleranceSpec(max_atol=atol, max_rtol=rtol, required_matched_ratio=1.0)
    _correctness, exceeds = compute_error_stats(candidate_output, reference_output, spec)
    return not exceeds
