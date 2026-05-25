"""Compile-time shared-memory budget check for @triton.autotune configs.

Reads ptxas-emitted SMEM via Triton's ``CompiledKernel.metadata.shared``
instead of estimating from source — duck-typed across Triton versions per
the hw-spec-injection spec (doc/specs/2026-05-24-coding-hw-spec-design.md
§6.2). Called from ``src/agents/coder.py::_make_compile_tool`` after the
existing ``compile_kernel()`` success path; rejections ride the same
in-loop tool-error retry path as ``autotune_exclude`` violations.

Why ptxas truth vs static formula:
- ``ptxas`` decides actual SMEM allocation (fuses loads, hoists allocations,
  shares buffers across pipeline stages). A static formula
  ``num_stages × (M×K + N×K) × dtype_bytes`` is an upper bound but can
  over- AND undercount.
- Reads ``.metadata.shared`` directly. No parser, no formula, no
  shape-dispatch (matmul/reduction/elementwise). Catches every kernel
  shape because ``ptxas`` already accounts for whatever shape Triton's
  compiler chose.

The check is fail-open by design: warmup failures and missing-metadata
cases skip the Config rather than rejecting it (events.emit a
``smem_check_skipped`` for post-run analysis). The compile / correctness
gauntlet downstream catches real-but-undetectable overflows; the SMEM
check exists to catch the OBVIOUS cases proactively, not to be the sole
guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.runtime import events


@dataclass(frozen=True)
class SMEMViolation:
    """One @triton.autotune Config whose ptxas SMEM exceeds the device cap."""

    config_idx: int
    config_kwargs: dict
    num_warps: int
    num_stages: int
    footprint: int      # ptxas-reported bytes


def _read_compiled_smem(compiled: Any) -> int | None:
    """Duck-typed across Triton versions: modern ``.metadata.shared``,
    older ``.shared``. Returns None when neither is present (caller skips
    that Config — fail-open, not fail-closed)."""
    md = getattr(compiled, "metadata", None)
    if md is not None and hasattr(md, "shared"):
        return int(md.shared)
    legacy = getattr(compiled, "shared", None)
    return int(legacy) if legacy is not None else None


def _latest_cache_entry(jit_fn: Any) -> Any | None:
    """Most recent CompiledKernel in the JITFunction's per-device cache.

    Triton's cache shape changed across versions; this helper accepts both:

    * **Triton ≥ 3.6** uses ``jit_fn.device_caches: defaultdict[int, tuple]``
      where the first tuple entry (``entry[0]``) is
      ``dict[signature_str, CompiledKernel]``. Subsequent tuple entries
      are signature-canonicalization maps and the backend/binder objects.
    * **Older Triton** used ``jit_fn.cache: dict[device, dict[hash, CompiledKernel]]``.

    After ``jit_fn.warmup(...)`` populates a new entry, this returns the
    freshest one (last-insertion-order via ``reversed``). Returns None when
    no compiled kernel is reachable through either attribute.
    """
    # Modern Triton (≥3.6): device_caches[dev_idx] is a 5-tuple; entry[0]
    # is the actual dict[signature_str -> CompiledKernel].
    device_caches = getattr(jit_fn, "device_caches", None)
    if device_caches:
        for entry in device_caches.values():
            # Defensive: only modern Triton stores a tuple here. Older
            # Triton-version code never reaches device_caches in the first
            # place, so the isinstance guard is just paranoia.
            kernel_dict = entry[0] if isinstance(entry, tuple) and entry else None
            if kernel_dict:
                return next(reversed(kernel_dict.values()))
    # Legacy Triton path: cache[device] is a dict[hash, CompiledKernel].
    cache = getattr(jit_fn, "cache", None)
    if cache:
        for device_cache in cache.values():
            if device_cache:
                return next(reversed(device_cache.values()))
    return None


def _capture_jit_args_via_host_wrapper(
    autotuner: Any,
    host_wrapper_fn: Callable,
    sample_args: tuple,
    *,
    iter_no: int | None = None,
) -> tuple[tuple, dict] | None:
    """Drive ``host_wrapper_fn(*sample_args)`` once with ``autotuner.run``
    replaced by a no-launch recorder. Returns ``(args, kwargs)`` — the
    positional args and keyword args the host wrapper passed to the
    autotuner on the FIRST internal kernel invocation — or ``None`` on
    capture failure.

    Capturing BOTH positional and keyword args matters because real host
    wrappers commonly use keyword launches for shape/stride parameters
    (``matmul_kernel[grid](a, b, c, M=M, K=K, stride_am=a.stride(0))``).
    Without the kwargs, the per-Config warmup downstream raises on
    signature mismatch, the helper emits ``warmup_failed`` for every
    Config, and the SMEM check fails open on the production case. Regression
    surfaced by Codex 2026-05-25 adversarial review.

    Multi-call host wrappers (split-K matmul, multi-stage fusions): only
    the first recorded invocation is used. SMEM is per-launch invariant
    in autotune — every call shares the same Config set — so one
    representative args set is sufficient.

    Restore-on-exception via try/finally with ``del autotuner.run``
    (deleting the instance attribute restores the class-method descriptor).
    See spec §6.2 for Triton 3.6 ground truth on the late-bound
    ``self.run`` lookup inside ``KernelInterface.__getitem__``.
    """
    captured: dict = {"args": None, "kwargs": None, "called": False}

    def recording_run(*args, **kwargs):
        if not captured["called"]:
            captured["args"] = args
            captured["kwargs"] = dict(kwargs)
            captured["called"] = True
        return None  # host wrapper allocates c BEFORE this call; tolerates None

    try:
        autotuner.run = recording_run
        try:
            host_wrapper_fn(*sample_args)
        except Exception:
            # Crash AFTER a successful first-call capture is still useful:
            # only the first invocation is replayed downstream (SMEM is
            # per-launch invariant across the Config set), so we honor that
            # capture and emit a softer telemetry signal. Crash BEFORE any
            # capture is the hard fail-open case.
            if captured["called"]:
                events.emit(
                    "smem_check_skipped",
                    iter=iter_no,
                    role="coder",
                    reason="host_wrapper_crashed_after_capture",
                )
                return captured["args"], captured["kwargs"]
            events.emit(
                "smem_check_skipped",
                iter=iter_no,
                role="coder",
                reason="host_wrapper_failed",
            )
            return None
    finally:
        try:
            del autotuner.run
        except AttributeError:
            pass

    if not captured["called"]:
        events.emit(
            "smem_check_skipped",
            iter=iter_no,
            role="coder",
            reason="recorder_no_capture",
        )
        return None
    return captured["args"], captured["kwargs"]


def check_autotune_smem_budget(
    autotuner: Any,
    host_wrapper_fn: Callable,
    sample_args: tuple,
    *,
    cap_bytes: int,
    iter_no: int | None = None,
) -> list[SMEMViolation]:
    """Top-level SMEM check. Drives the host wrapper to capture full JIT
    args (production path), then warmups each Config with the captured
    args and reads ptxas-reported SMEM.

    ``host_wrapper_fn`` is REQUIRED — the legacy direct-warmup fallback
    was removed because it silently failed open on production
    host-wrapper kernels (the exact bug the recorder-patch redesign was
    meant to close). All callers must pass a callable that drives the
    autotuner with the user-facing inputs in ``sample_args``.

    Returns a list of ``SMEMViolation`` entries (empty when all configs
    fit or when the check skipped). Emits ``smem_check_skipped`` events
    on every fail-open path (``host_wrapper_failed``,
    ``host_wrapper_crashed_after_capture``, ``recorder_no_capture``,
    ``cfg_overrides_recorded_kwarg``, ``warmup_failed``); never raises.

    *sample_args* — the user-facing inputs the host wrapper accepts
    (``(a, b)`` for matmul). The recorder uses these to drive the host
    wrapper once; the wrapper internally derives the full JIT call args
    (``c, M, N, K, strides``).

    *iter_no* — optional iteration index threaded into every
    ``events.emit`` so per-iter event correlation in ``events.jsonl``
    works. ``None`` (default) preserves call sites that don't track an
    iteration.
    """
    captured = _capture_jit_args_via_host_wrapper(
        autotuner, host_wrapper_fn, sample_args, iter_no=iter_no,
    )
    if captured is None:
        return []  # event already emitted by capture helper
    recorded_args, recorded_kwargs = captured

    # Strip framework-injected kwargs (``grid``, ``warmup``) added by
    # ``KernelInterface.__getitem__``; we set our own ``grid=(1,)`` for the
    # compile-only warmup and Triton's ``warmup`` arg is handled implicitly
    # by ``JITFunction.warmup``.
    base_kwargs = {
        k: v for k, v in recorded_kwargs.items() if k not in ("grid", "warmup")
    }

    violations: list[SMEMViolation] = []
    for i, cfg in enumerate(autotuner.configs):
        # Conflict detection: if the host wrapper explicitly passed a
        # constexpr that also appears in cfg.kwargs (e.g. a shape-dependent
        # ``BLOCK_K=K_runtime`` launched by the wrapper, autotuned over a
        # different ``BLOCK_K`` set by the Config), blindly preferring
        # cfg.kwargs would silently diverge the measured SMEM from
        # production. Skip the Config and emit telemetry so the user can
        # investigate. Fail-open by design (per module docstring).
        conflicts = sorted(set(base_kwargs).intersection(cfg.kwargs))
        if conflicts:
            for key in conflicts:
                events.emit(
                    "smem_check_skipped",
                    iter=iter_no,
                    role="coder",
                    reason="cfg_overrides_recorded_kwarg",
                    config_idx=i,
                    key=key,
                )
            continue

        # Build the warmup call kwargs: recorded host-wrapper kwargs first
        # (shape/stride args), then cfg.kwargs (BLOCK_M etc.), then
        # num_warps/num_stages/grid for the warmup itself.
        call_kwargs = dict(base_kwargs)
        call_kwargs.update(cfg.kwargs)
        call_kwargs.update(
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            grid=(1,),
        )
        try:
            autotuner.fn.warmup(*recorded_args, **call_kwargs)
        except Exception:
            events.emit(
                "smem_check_skipped",
                iter=iter_no,
                role="coder",
                reason="warmup_failed",
                config_idx=i,
            )
            continue
        compiled = _latest_cache_entry(autotuner.fn)
        smem = _read_compiled_smem(compiled)
        if smem is None or smem <= cap_bytes:
            continue
        violations.append(
            SMEMViolation(
                config_idx=i,
                config_kwargs=dict(cfg.kwargs),
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                footprint=smem,
            )
        )
    return violations
