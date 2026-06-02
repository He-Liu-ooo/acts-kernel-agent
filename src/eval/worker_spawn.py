"""Shared parent-side worker-spawn skeleton.

Both ``bench_subprocess.run_bench_subprocess`` and
``correctness_subprocess.run_correctness_subprocess`` drive an identical
spawn sequence: mkdir the worker dir, write ``request.json``, ``Popen`` a
``python -m <worker> --request <r> --response <w>`` child with both
streams redirected to ``worker.log``, await its exit via
``asyncio.to_thread(proc.wait, timeout)``, and on a watchdog timeout
terminate / sleep(2) / kill the straggler. The *only* real differences
between the two callers are the worker module name and the OUTCOME
contract (bench raises ``WorkerCrashed``; correctness returns a
fail-closed ``CorrectnessResult``).

This module owns that shared sequence and returns a structured
``WorkerOutcome``; each caller maps the outcome onto its own contract.
The four observable outcomes — and their exact (returncode, log_tail)
payloads — are preserved byte-for-byte from the pre-refactor inline
bodies:

- ``"timeout"`` → returncode ``-1``, tail of ``worker.log``.
- ``"crashed"`` (non-zero exit or missing response.json) → the child's
  returncode, tail of ``worker.log``.
- ``"crashed"`` (clean exit but unreadable/malformed response.json) →
  the child's returncode, with the ``malformed response.json
  (<ExcType>: <msg[:200]>)\\n`` prefix prepended to the log tail.
- ``"ok"`` → returncode, parsed ``response.json`` dict, empty tail.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkerOutcome:
    """Structured result of one worker invocation.

    Attributes:
        status: ``"ok"`` | ``"timeout"`` | ``"crashed"``.
        returncode: Child exit code. ``-1`` on the watchdog-timeout path;
            the child's actual code (positive for unhandled exceptions,
            negative for signal-kills, ``0`` for clean-exit-no-response or
            malformed-response) otherwise.
        response: Parsed ``response.json`` dict when ``status == "ok"``;
            ``None`` otherwise.
        log_tail: Last ~2KB of ``worker.log`` (UTF-8, errors replaced).
            On the malformed-response path this is prefixed with a
            ``malformed response.json (...)`` line. Empty string when
            ``status == "ok"``.
    """

    status: str
    returncode: int
    response: dict | None
    log_tail: str


async def spawn_worker(
    *,
    module: str,
    request: dict,
    worker_dir: Path,
    timeout_s: float,
    argv_prefix: list[str] | None = None,
) -> WorkerOutcome:
    """Spawn a worker subprocess and return its structured outcome.

    Args:
        module: Worker module run as ``python -m <module>`` (e.g.
            ``"src.eval.bench_worker"``). Ignored when ``argv_prefix`` is
            supplied.
        request: Payload serialized to ``worker_dir / request.json``.
        worker_dir: Per-call directory holding request/response/worker.log.
        timeout_s: Total-lifetime ``proc.wait()`` watchdog.
        argv_prefix: Optional override of the default
            ``[sys.executable, "-m", module]`` argv prefix — used by
            ``correctness_subprocess`` to inject a stub worker script in
            tests. ``--request``/``--response`` flags are appended either
            way.
    """
    worker_dir.mkdir(parents=True, exist_ok=True)
    request_path = worker_dir / "request.json"
    response_path = worker_dir / "response.json"
    log_path = worker_dir / "worker.log"
    request_path.write_text(json.dumps(request))

    prefix = argv_prefix if argv_prefix is not None else [sys.executable, "-m", module]
    argv = [
        *prefix,
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]

    # Keep the log file handle open across the wait — the worker writes
    # incrementally and we want everything captured even if the child gets
    # killed mid-write.
    logfile = log_path.open("w")
    try:
        proc = subprocess.Popen(
            argv, stdout=logfile, stderr=logfile, start_new_session=True
        )

        def _wait() -> int:
            return proc.wait(timeout=timeout_s)

        try:
            returncode = await asyncio.to_thread(_wait)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            return WorkerOutcome("timeout", -1, None, _read_tail(log_path, 2048))

        if returncode != 0 or not response_path.exists():
            return WorkerOutcome("crashed", returncode, None, _read_tail(log_path, 2048))

        try:
            response = json.loads(response_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return WorkerOutcome(
                "crashed",
                returncode,
                None,
                f"malformed response.json ({type(exc).__name__}: "
                f"{str(exc)[:200]})\n" + _read_tail(log_path, 2048),
            )
        return WorkerOutcome("ok", returncode, response, "")
    finally:
        logfile.close()


def _kill_process_group(proc: subprocess.Popen, grace_s: float = 2.0) -> None:
    """SIGTERM the worker's process group, give it a grace period, then ALWAYS
    SIGKILL the group and reap the direct child. ``start_new_session=True``
    makes the child a group leader, so its pid is the pgid and signalling the
    group reaches every descendant (e.g. NCU children). The final SIGKILL is
    unconditional: a descendant can ignore SIGTERM and survive even after the
    direct child has already exited, so the group kill must not be gated on the
    child still being alive. Best-effort: a process/group that already exited
    raises ProcessLookupError, which we swallow."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace_s)  # let the group exit gracefully on SIGTERM
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)  # ALWAYS — descendants may ignore SIGTERM
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace_s)  # reap the direct child; no zombie
    except subprocess.TimeoutExpired:
        pass


def _read_tail(path: Path, max_bytes: int) -> str:
    """Return the last ``max_bytes`` of ``path`` as UTF-8 (errors replaced).

    Empty string if the file is missing — defensive for the
    crash-before-log-open edge case.
    """
    if not path.exists():
        return ""
    return path.read_bytes()[-max_bytes:].decode("utf-8", errors="replace")
