"""Async helper around ``src/eval/bench_worker.py`` subprocess.

This module owns the parent side of the bench-subprocess isolation
refactor: build ``request.json``, spawn the bench worker as
``python -m src.eval.bench_worker``, await its exit, parse
``response.json`` on clean termination, and merge per-iter artifacts
(``events.jsonl`` chunk + ``cand_<idx>.ncu-rep`` files) into the run's
canonical files.

The orchestrator integrates this helper via ``await
run_bench_subprocess(...)`` inside its per-iter K-way eval branch and
then calls ``merge_worker_artifacts(...)`` on clean exit. On crash
(``WorkerCrashed`` raised) the orchestrator bumps its
``consecutive_worker_crashes`` counter and escalates to
``WorkerProcessUnstable`` at the configured threshold. See spec §4
(architecture), §7 (merge mechanics), §8 (error path matrix) of
``doc/specs/2026-05-24-bench-subprocess-isolation-design.md``.

The helper deliberately does **not** own the crash-counter state.
``worker_crash_threshold`` is accepted as a parameter for contract
symmetry (and to make a future relocation of the counter trivial) but
is currently unused inside this module — each call signals exactly one
crash via ``WorkerCrashed``; the orchestrator owns counter bump +
``WorkerProcessUnstable`` escalation.
"""
from __future__ import annotations

from pathlib import Path

from src.eval.worker_spawn import spawn_worker


class WorkerCrashed(RuntimeError):
    """Internal helper exception signaling a single failed worker invocation.

    Distinct from the public ``WorkerProcessUnstable`` raised by the
    orchestrator after the counter trips the configured threshold. This
    exception is per-call; the orchestrator translates it to the
    public exception after the counter bump (see spec §5.6).

    Attributes:
        returncode: Process exit code. ``0`` if the worker exited
            cleanly but failed to produce a response.json; positive
            for unhandled Python exceptions; negative for
            signal-kills (e.g., ``-9`` for SIGKILL); ``-1`` for the
            watchdog-timeout path.
        stderr_tail: Last ~2KB of the worker's stderr/stdout log
            (decoded UTF-8, errors replaced) for the parent's
            ``bench_worker_crashed`` event payload.
    """

    def __init__(self, *, returncode: int, stderr_tail: str):
        super().__init__(f"bench worker crashed: returncode={returncode}")
        self.returncode = returncode
        self.stderr_tail = stderr_tail


async def run_bench_subprocess(
    *,
    request: dict,
    worker_dir: Path,
    worker_crash_threshold: int,
    worker_timeout_s: float,
) -> dict:
    """Spawn the bench worker subprocess and return parsed response.json.

    The helper:

    1. ``mkdir -p worker_dir``.
    2. Writes ``request.json`` under ``worker_dir``.
    3. Spawns ``python -m src.eval.bench_worker --request <r> --response <w>``
       with both stdout and stderr redirected to ``worker.log``.
    4. ``await``s ``proc.wait(timeout=worker_timeout_s)`` via
       ``asyncio.to_thread`` (the orchestrator stays async-friendly
       with the K-way fan-out phase).
    5. On clean exit (returncode == 0 AND response.json exists),
       parses and returns the response dict.
    6. On non-zero exit, signal-kill, missing response, or watchdog
       timeout, raises ``WorkerCrashed`` carrying the returncode and
       the tail of ``worker.log``.

    On watchdog timeout the helper first ``terminate()``s the child,
    waits ~2s for graceful shutdown, then ``kill()``s any stragglers
    before raising.

    Args:
        request: Request payload to serialize into request.json. See
            spec §5.1 for the schema.
        worker_dir: Per-iter directory under ``<run_dir>/iter_<n>/``
            where request/response/events/worker.log/.ncu-rep all live.
        worker_crash_threshold: Threshold the orchestrator uses to
            escalate to ``WorkerProcessUnstable``. Currently unused
            inside this helper — the helper raises ``WorkerCrashed``
            unconditionally on failure; the orchestrator owns counter
            state. Kept in the signature for contract symmetry per the
            plan.
        worker_timeout_s: Total-lifetime ``proc.wait()`` watchdog
            (covers compile + autotune burn-in + K-way bench + NCU
            profile end-to-end). Codex 2026-05-26 fix #2 — formerly
            ``worker_startup_timeout_s`` with a 30s default that
            killed healthy workers mid-NCU; renamed + default bumped
            in ``ACTSConfig.worker_timeout_s`` to 180000s (~50h,
            effectively disabling the watchdog while the refactor
            beds in; see ``ACTSConfig.worker_timeout_s`` docstring).

    Returns:
        Parsed response.json as a dict (schema per spec §5.2).

    Raises:
        WorkerCrashed: On any failure mode (non-zero exit, signal,
            missing response, timeout).
    """
    del worker_crash_threshold  # see docstring — orchestrator owns counter

    # The spawn / wait / timeout-kill / malformed-guard sequence is shared
    # with the correctness helper; ``spawn_worker`` owns it and returns a
    # structured outcome. The bench contract maps any non-``ok`` status to
    # ``WorkerCrashed`` — preserving the exact prior payloads: timeout →
    # returncode -1, crashed/missing → child returncode, malformed → child
    # returncode with the ``malformed response.json (...)`` prefix.
    outcome = await spawn_worker(
        module="src.eval.bench_worker",
        request=request,
        worker_dir=worker_dir,
        timeout_s=worker_timeout_s,
    )
    if outcome.status != "ok":
        raise WorkerCrashed(
            returncode=outcome.returncode,
            stderr_tail=outcome.log_tail,
        )
    return outcome.response


def merge_worker_artifacts(
    *,
    run_dir: Path,
    worker_dir: Path,
    iter_no: int,
    response: dict,
    ncu_cache_dir: Path,
) -> dict:
    """Merge per-iter worker chunk into canonical run artifacts.

    Two operations, both idempotent and crash-tolerant:

    1. **events.jsonl concat** — read ``worker_dir / events.jsonl``
       line-by-line and append each line to ``run_dir / events.jsonl``.
       No JSON parsing; events.jsonl is one-object-per-line by
       contract. Missing chunk (no events fired) → count=0, no error.
    2. **.ncu-rep cache copy** — glob ``worker_dir / cand_*.ncu-rep``
       and copy each into ``ncu_cache_dir``. The per-tree-node copy
       to ``tree/node_<id>/ncu.ncu-rep`` happens later via
       ``tree_dump.dump_node(..., ncu_rep_src=worker_dir / "cand_<winner>.ncu-rep")``
       — this helper only handles the shared NCU cache.

    Args:
        run_dir: Canonical run root.
        worker_dir: ``<run_dir>/iter_<iter_no>/worker``.
        iter_no: Current iter index. Accepted for symmetry with the
            event payload schema; not used in the merge itself.
        response: Parsed response.json. Accepted for symmetry with
            the orchestrator call site (the parent may use
            ``winner_idx`` etc. when emitting follow-up events);
            this helper does not introspect it.
        ncu_cache_dir: Shared NCU cache root.

    Returns:
        ``{"event_count": int, "ncu_rep_count": int}`` for the
        parent's ``worker_chunk_merged`` event payload (spec §5.5).
    """
    del iter_no, response  # see docstring — accepted for caller symmetry

    event_count = 0
    ncu_rep_count = 0

    # 1. events.jsonl concat (line-by-line append; no parsing).
    chunk = worker_dir / "events.jsonl"
    if chunk.exists():
        canonical = run_dir / "events.jsonl"
        with chunk.open("r") as src, canonical.open("a") as dst:
            for line in src:
                dst.write(line)
                event_count += 1

    # 2. .ncu-rep copy to shared cache. Per-tree-node copy is handled
    # by tree_dump.dump_node from the worker dir directly (see Task 9).
    ncu_cache_dir.mkdir(parents=True, exist_ok=True)
    for ncu_rep in sorted(worker_dir.glob("cand_*.ncu-rep")):
        dest = ncu_cache_dir / ncu_rep.name
        dest.write_bytes(ncu_rep.read_bytes())
        ncu_rep_count += 1

    return {"event_count": event_count, "ncu_rep_count": ncu_rep_count}
