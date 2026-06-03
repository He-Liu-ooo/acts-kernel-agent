"""ACTS-narrative event emission.

One ``emit()`` call fans out to two sinks: stdlib ``logger.info`` (human
text -> stderr + run.log) and a module-level JSONL file handle
(structured -> events.jsonl). Unbound => JSONL write is skipped; the
logger line still goes out. See ``doc/runtime.md`` for full semantics.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from enum import Enum
from typing import Any, IO

from src.runtime.timefmt import iso_ts

logger = logging.getLogger(__name__)

CORE_EVENT_KINDS: frozenset[str] = frozenset({
    "run_start", "baseline_attempt", "baseline_success", "baseline_failure",
    "baseline_ready", "iter_start", "planner_selected",
    # ``planner_failed`` covers any ``PlanningError`` cause (turn-budget,
    # missing ``submit_plan``, transient retry exhaustion, available-
    # actions guard). Mirrors ``coder_failed``.
    "planner_failed",
    # ``coder_submitted`` marks ``implement()`` returning a kernel — it
    # does NOT prove the compile/correctness tools ran or passed.
    # Ground-truth per-tool-call records are in ``traces/*.jsonl``.
    # K-way fan-out payload: ``{winner_candidate_idx, n_candidates,
    # n_survivors, n_profile_attempts}`` — ``n_survivors`` counts
    # bench-survivors, ``n_profile_attempts`` is the winner's 1-indexed
    # rank (>1 means profile-fallback engaged).
    # ``coder_failed`` payload: ``{iter, reason}`` (reason truncated to
    # 200 chars), plus ``candidate_idx: int | None`` (set per K-way
    # candidate; None for legacy single-candidate paths like baseline
    # ``translate()``).
    "coder_submitted", "coder_failed",
    "bench_done", "profile_done", "score_computed",
    "reviewer_feedback",
    # Per-call event from the multi-turn Reviewer's ``query_metric`` tool —
    # records ``{iter, count, names[:8]}`` so post-run analysis can see
    # what the LLM actually asked for. Emitted only when
    # ``ACTSConfig.reviewer_metric_queries`` is True.
    "reviewer_metric_query",
    "branch_dead_end", "iter_end", "verify_start",
    "verify_done", "run_end",
    # SOL integration (2026-04-27) — Tier 1 Trace + Tier 4 anti-cheat /
    # clock-lock observability. ``trace_emitted`` is fired once per
    # evaluation with the SOL ``Trace`` payload. The ``reward_hack_*``
    # triplet covers the two-channel detector flow:
    #   channel A — process-level detector raised inside the eval block
    #     → ``reward_hack_detected``. K-way payload carries
    #     ``candidate_idx``; the iter aborts (no sibling fallback —
    #     process state may be tainted). A matching ``coder_failed``
    #     fires alongside for the all-failures bookkeeping.
    #   channel B — sol_score flagged ``reward_hack_suspect`` and the
    #     re-eval was either confirmed (``reward_hack_confirmed`` →
    #     DEAD_END) or cleared (``reward_hack_cleared`` → accept score).
    # ``calibration_warning`` fires when sol_score's ``calibration_warning``
    # bit is set (T_k < ~T_SOL margin).
    "trace_emitted",
    "clock_lock_unavailable",
    # Reserved — emission deferred to clock-lock implementation.
    "clock_drift_detected",
    "reward_hack_detected",
    "reward_hack_confirmed",
    "reward_hack_cleared",
    "calibration_warning",
    # Sibling-aware Planner/Reviewer contracts (see
    # doc/specs/2026-05-13-sibling-aware-agent-contracts-design.md).
    # ``sibling_context_rendered`` fires once per Planner call AND once
    # per Reviewer call whenever the parent has been expanded before
    # (sibling_context != ""). Payload: ``parent_node_id``,
    # ``sibling_count``, ``regressed_actions``, ``consumer``
    # (``"planner"`` | ``"reviewer"``).
    "sibling_context_rendered",
    # Fires once per iter when ≥1 K-way Coder/bench-layer candidates
    # failed and a failure-summary node was attached. Payload:
    # ``node_id``, ``parent_id``, ``action``, ``params``,
    # ``candidate_count``. Profile-layer failures don't fire this.
    "failure_summary_added",
    # Fires when Reviewer verdict is dead_end AND child.action_applied
    # already regressed on a sibling — the existing system.md L61-62 rule
    # has now actually fired. Payload: ``action``, ``sibling_iter``.
    "repeated_pathway_dead_end",
    # A1 PR 1/B (Triton autotune integration) — fires once per iteration
    # after benchmark-captured autotune winners are copied to the Kernel.
    # Payload: ``workload_count: int`` (workloads benchmarked this iter),
    # ``winner_count: int`` (per-workload winners captured from cache
    # deltas). Per-workload winners themselves are persisted on the Kernel
    # (``autotune_winner: dict[uuid, cfg]``); per-workload burn-in elapsed
    # is captured by ``bench_done`` upstream.
    "autotune_burn_in_done",
    # Operator-supplied Triton baseline (bypass CoderAgent.translate()).
    # See doc/specs/2026-05-16-operator-supplied-triton-baseline-design.md.
    # ``operator_baseline_load`` payload: path (str), kernel_name (str),
    # dps (bool), enforce_autotune (bool). Fired once at loader entry
    # after kernel-name resolution.
    # ``operator_baseline_success`` payload: source_bytes (int),
    # triton_kernel_name (str). Fired after gate 7 (correctness) passes.
    # ``operator_baseline_failure`` payload: stage (one of "empty",
    # "missing_file", "name_resolve", "autotune_validate", "compile",
    # "correctness wl K/N"), reason (str, truncated to 200 chars). Fired
    # immediately before BaselineGenerationError is raised.
    "operator_baseline_load",
    "operator_baseline_success",
    "operator_baseline_failure",
    # Bench-subprocess isolation (2026-05-24). Per-iter K-way bench + NCU
    # profile runs in a fresh ``python -m src.eval.bench_worker`` child;
    # parent merges artifacts on clean exit, bumps a crash counter on
    # non-zero exit. See doc/specs/2026-05-24-bench-subprocess-isolation-
    # design.md §5.5.
    # ``bench_worker_spawned`` payload: ``iter``, ``worker_dir``,
    # ``request_path``. Fired by parent immediately before Popen.
    # ``bench_worker_exited`` payload: ``iter``, ``returncode``,
    # ``walltime_s``. Fired by parent after proc.wait() regardless of
    # outcome (clean exit AND crashes).
    # ``bench_worker_crashed`` payload: ``iter``, ``returncode``,
    # ``stderr_tail`` (last 2KB of worker.log), ``consecutive``. Fired
    # ONLY on non-zero / signal-kill exit (or missing response.json /
    # watchdog timeout).
    # ``worker_chunk_merged`` payload: ``iter``, ``event_count``,
    # ``ncu_rep_count``. Fired by parent after merging ``worker/
    # events.jsonl`` + copying ``.ncu-rep`` files on clean exit.
    "bench_worker_spawned",
    "bench_worker_exited",
    "bench_worker_crashed",
    "worker_chunk_merged",
})

# ``iter_end.outcome`` values. Kept as string constants (not an enum) so
# emit payloads stay trivially JSON-serializable and callers don't pay an
# import tax. Typos are caught via `CORE_EVENT_KINDS`-style review, not
# the type system.
ITER_ADVANCED = "advanced"
ITER_DEAD_END = "dead_end"
ITER_SKIPPED = "skipped"

class DeadReason(str, Enum):
    """Why a branch was marked DEAD_END.

    Inherits from ``str`` so members JSON-serialize as their string value
    directly — telemetry payloads (``branch_dead_end.reason``) and
    persisted tree checkpoints (``TreeNode.dead_reason``) store the same
    stable wire form. Telemetry consumers (log parsers, regression tests)
    key on these strings; keep values stable.

    Three causes today (kept distinct so downstream readers — ``best_node``,
    memory distillation, tree viz — can treat them differently rather than
    collapsing them into a single dead-end signal):

      - Reviewer-judged: kernel ran with a valid score, Reviewer
        classified the branch as regressed/over → REVIEWER_JUDGED.
      - Beam-pruned: kernel ran with a valid score, lost the beam
        competition → BEAM_PRUNED. Score remains trustworthy as a
        final answer.
      - Infrastructure error: kernel never produced a trustworthy score
        (CUDA error, profiler crash, bench failure, reward-hack
        confirmation, …). One member per error class.
    """

    REWARD_HACK = "reward_hack"
    REWARD_HACK_CONFIRMED = "reward_hack_confirmed"
    CUDA_ERROR = "cuda_error"
    PROFILER_ERROR = "profiler_error"
    BENCH_FAILURE = "bench_failure"
    REPR_LATENCY_UNAVAILABLE = "repr_workload_latency_unavailable"
    # Reserved — emission deferred to dead-agent failure handling.
    AGENT_FAILURE = "agent_failure"
    BEAM_PRUNED = "beam_pruned"
    REVIEWER_JUDGED = "reviewer_judged"
    # K-way candidate that failed at the Coder or bench layer. No score,
    # so excluded from ``best_node`` like other infra-error reasons.
    CODER_FAILED = "coder_failed"


_events_fh: IO[str] | None = None
_lock = threading.Lock()


def finite_or_none(x: float | int | None) -> float | None:
    """Map non-finite floats (``inf``/``-inf``/``nan``) to ``None``.

    ``BenchmarkResult.per_workload_latency_us`` uses ``math.inf`` as a
    launch-failure sentinel; forwarding that verbatim produces the
    non-standard ``Infinity`` token in ``events.jsonl``, which RFC-8259
    parsers reject.
    """
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def bind(fh: IO[str]) -> None:
    """Register a file handle for JSONL writes. Called by ``RunContext.create``."""
    global _events_fh
    with _lock:
        _events_fh = fh


def unbind() -> None:
    """Clear the registered file handle. Called by ``RunContext.close`` before FH close."""
    global _events_fh
    with _lock:
        _events_fh = None


def _compact_json(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, default=str, separators=(",", ":"))
    except Exception:
        return "{}"


def emit(kind: str, *, iter: int | None = None, **fields: Any) -> None:
    """Emit a narrative event to both sinks. Never raises."""
    if kind not in CORE_EVENT_KINDS:
        try:
            logger.warning("unknown event kind: %s", kind)
        except Exception:
            pass
    # Skip the serialize-for-log work if nobody at INFO is listening.
    if logger.isEnabledFor(logging.INFO):
        # Merge ``iter`` into the log payload so ``run.log`` shows the
        # iteration for iter-scoped events; omitted when ``iter is None``
        # so per-iter greps aren't polluted with a null key.
        log_payload: dict[str, Any] = (
            {"iter": iter, **fields} if iter is not None else dict(fields)
        )
        try:
            logger.info("%s %s", kind, _compact_json(log_payload))
        except Exception:
            pass
    fh = _events_fh
    if fh is None:
        return
    # Build the record (including timestamp) outside the write lock so
    # contending threads don't serialize on datetime formatting.
    try:
        record = {"ts": iso_ts(), "kind": kind, "iter": iter, **fields}
        # ``allow_nan=False`` enforces RFC-8259: NaN/Inf raise ValueError
        # rather than emitting the non-standard ``NaN``/``Infinity`` tokens
        # that strict parsers reject. ``finite_or_none`` is the upstream
        # sanitizer; this is the backstop.
        payload = json.dumps(record, default=str, allow_nan=False) + "\n"
    except Exception:
        return
    try:
        with _lock:
            fh.write(payload)
    except Exception:
        pass
