"""Tier 1 tests for src/eval/bench_subprocess.py (Popen mocked).

Covers Tasks 6-8 of the bench-subprocess isolation refactor plan
(``doc/plans/2026-05-24-bench-subprocess-isolation-plan.md``):

- Task 6: ``run_bench_subprocess`` happy path — fake Popen exits 0 +
  writes canned response.json; helper returns parsed dict.
- Task 7: crash + timeout handling — non-zero exit, signal-kill,
  missing response.json, and watchdog timeout all raise
  ``WorkerCrashed`` with ``returncode`` + ``stderr_tail`` attributes.
- Task 8: ``merge_worker_artifacts`` — events.jsonl line-by-line
  concat preserves order; ``cand_*.ncu-rep`` files copy to the shared
  NCU cache; returned counts feed the parent's
  ``worker_chunk_merged`` event.

All tests are Tier 1 (torchless). ``subprocess.Popen`` is monkeypatched
to a fake that mimics the bench-worker contract without spawning a
real process.
"""
from __future__ import annotations

import asyncio
import json
import subprocess as _subprocess
from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# Test helpers
# --------------------------------------------------------------------------


def _minimal_request(worker_dir: Path, run_dir: Path, iter_no: int = 0) -> dict:
    """Build a minimal request dict — the helper doesn't introspect it."""
    return {
        "schema_version": 1,
        "iter_no": iter_no,
        "worker_dir": str(worker_dir),
        "run_dir": str(run_dir),
    }


def _canned_response(iter_no: int = 0) -> dict:
    """Canned response.json payload — matches spec §5.2 shape."""
    return {
        "schema_version": 1,
        "iter_no": iter_no,
        "candidates": [],
        "winner_idx": None,
        "winner_profile": None,
        "aborted_by_channel_A": False,
        "child_walltime_s": 0.1,
    }


def _make_fake_popen_class(
    *,
    returncode: int = 0,
    write_response: bool = True,
    response_payload: dict | None = None,
    stderr_text: str = "",
    timeout_on_wait: bool = False,
):
    """Build a FakePopen class with the requested behavior.

    The fake mirrors the argv shape this helper builds:
    ``[python, -m, src.eval.bench_worker, --request, R, --response, W]``.
    On instantiation it (optionally) writes ``response_payload`` to W
    and (optionally) writes ``stderr_text`` to the ``stderr`` file
    handle passed by the helper.
    """

    class FakePopen:
        def __init__(self, args, **kwargs):
            # Pull the --response path out of argv (mirrors the real argv layout).
            response_path = Path(args[args.index("--response") + 1])
            if write_response:
                payload = response_payload if response_payload is not None else _canned_response()
                response_path.write_text(json.dumps(payload))
            # Write to the stderr file handle so _read_tail picks it up.
            if stderr_text and "stderr" in kwargs and kwargs["stderr"] is not None:
                kwargs["stderr"].write(stderr_text)
                kwargs["stderr"].flush()
            self.returncode = returncode
            self._timeout_on_wait = timeout_on_wait
            self._alive = True

        def wait(self, timeout=None):
            if self._timeout_on_wait:
                raise _subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            return self.returncode

        def terminate(self):
            self._alive = False

        def kill(self):
            self._alive = False

        def poll(self):
            return None if self._alive else self.returncode

    return FakePopen


# --------------------------------------------------------------------------
# Task 6 — happy path
# --------------------------------------------------------------------------


def test_run_bench_subprocess_happy_path(tmp_path, monkeypatch):
    """Fake Popen exits 0 + writes canned response.json; helper returns parsed dict."""
    from src.eval.bench_subprocess import run_bench_subprocess

    worker_dir = tmp_path / "iter_0" / "worker"
    worker_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "src.eval.bench_subprocess.subprocess.Popen",
        _make_fake_popen_class(returncode=0, write_response=True),
    )

    request = _minimal_request(worker_dir, tmp_path)
    result = asyncio.run(
        run_bench_subprocess(
            request=request,
            worker_dir=worker_dir,
            worker_crash_threshold=3,
            worker_timeout_s=30.0,
        )
    )
    assert result["schema_version"] == 1
    assert result["iter_no"] == 0
    # Helper must have written request.json to the worker dir.
    assert (worker_dir / "request.json").exists()
    written = json.loads((worker_dir / "request.json").read_text())
    assert written["iter_no"] == 0


def test_run_bench_subprocess_creates_worker_dir_if_missing(tmp_path, monkeypatch):
    """Helper mkdirs worker_dir even when caller forgot to."""
    from src.eval.bench_subprocess import run_bench_subprocess

    worker_dir = tmp_path / "iter_0" / "worker"  # does NOT pre-exist
    monkeypatch.setattr(
        "src.eval.bench_subprocess.subprocess.Popen",
        _make_fake_popen_class(returncode=0, write_response=True),
    )

    request = _minimal_request(worker_dir, tmp_path)
    result = asyncio.run(
        run_bench_subprocess(
            request=request,
            worker_dir=worker_dir,
            worker_crash_threshold=3,
            worker_timeout_s=30.0,
        )
    )
    assert result["schema_version"] == 1
    assert worker_dir.exists()


# --------------------------------------------------------------------------
# Task 7 — crash + timeout handling
# --------------------------------------------------------------------------


def test_non_zero_exit_raises_worker_crashed(tmp_path, monkeypatch):
    """Worker exits with returncode=1 → WorkerCrashed with stderr_tail populated."""
    from src.eval.bench_subprocess import run_bench_subprocess, WorkerCrashed

    worker_dir = tmp_path / "w"
    worker_dir.mkdir()
    monkeypatch.setattr(
        "src.eval.bench_subprocess.subprocess.Popen",
        _make_fake_popen_class(
            returncode=1,
            write_response=False,
            stderr_text="boom traceback\n",
        ),
    )

    with pytest.raises(WorkerCrashed) as exc_info:
        asyncio.run(
            run_bench_subprocess(
                request=_minimal_request(worker_dir, tmp_path),
                worker_dir=worker_dir,
                worker_crash_threshold=3,
                worker_timeout_s=30.0,
            )
        )
    assert exc_info.value.returncode == 1
    assert "boom" in exc_info.value.stderr_tail


def test_signal_killed_returns_negative_returncode(tmp_path, monkeypatch):
    """Worker SIGKILL → returncode == -9 surfaces on WorkerCrashed."""
    from src.eval.bench_subprocess import run_bench_subprocess, WorkerCrashed

    worker_dir = tmp_path / "w"
    worker_dir.mkdir()
    monkeypatch.setattr(
        "src.eval.bench_subprocess.subprocess.Popen",
        _make_fake_popen_class(
            returncode=-9,
            write_response=False,
            stderr_text="killed\n",
        ),
    )

    with pytest.raises(WorkerCrashed) as exc_info:
        asyncio.run(
            run_bench_subprocess(
                request=_minimal_request(worker_dir, tmp_path),
                worker_dir=worker_dir,
                worker_crash_threshold=3,
                worker_timeout_s=30.0,
            )
        )
    assert exc_info.value.returncode == -9
    assert "killed" in exc_info.value.stderr_tail


def test_missing_response_json_treated_as_crash(tmp_path, monkeypatch):
    """Worker exits 0 but didn't write response.json → WorkerCrashed."""
    from src.eval.bench_subprocess import run_bench_subprocess, WorkerCrashed

    worker_dir = tmp_path / "w"
    worker_dir.mkdir()
    monkeypatch.setattr(
        "src.eval.bench_subprocess.subprocess.Popen",
        _make_fake_popen_class(
            returncode=0,
            write_response=False,
            stderr_text="exited cleanly but produced nothing\n",
        ),
    )

    with pytest.raises(WorkerCrashed) as exc_info:
        asyncio.run(
            run_bench_subprocess(
                request=_minimal_request(worker_dir, tmp_path),
                worker_dir=worker_dir,
                worker_crash_threshold=3,
                worker_timeout_s=30.0,
            )
        )
    assert exc_info.value.returncode == 0
    assert "exited cleanly" in exc_info.value.stderr_tail


def test_timeout_terminates_and_raises(tmp_path, monkeypatch):
    """proc.wait() raises TimeoutExpired → helper terminate()+kill()s, raises WorkerCrashed."""
    from src.eval.bench_subprocess import run_bench_subprocess, WorkerCrashed

    worker_dir = tmp_path / "w"
    worker_dir.mkdir()

    terminated = {"flag": False}
    killed = {"flag": False}

    class HangingPopen:
        def __init__(self, args, **kwargs):
            if kwargs.get("stderr") is not None:
                kwargs["stderr"].write("hang\n")
                kwargs["stderr"].flush()
            self.returncode = None
            self._alive = True

        def wait(self, timeout=None):
            raise _subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

        def terminate(self):
            terminated["flag"] = True

        def kill(self):
            killed["flag"] = True
            self._alive = False

        def poll(self):
            # Still alive after terminate() — forces helper to escalate to kill().
            return None if self._alive else 0

    monkeypatch.setattr("src.eval.bench_subprocess.subprocess.Popen", HangingPopen)
    # Stub asyncio.sleep so the test doesn't actually wait 2 real seconds.
    monkeypatch.setattr("src.eval.bench_subprocess.asyncio.sleep", _stub_sleep)

    with pytest.raises(WorkerCrashed) as exc_info:
        asyncio.run(
            run_bench_subprocess(
                request=_minimal_request(worker_dir, tmp_path),
                worker_dir=worker_dir,
                worker_crash_threshold=3,
                worker_timeout_s=0.01,
            )
        )
    assert exc_info.value.returncode == -1
    assert terminated["flag"] is True
    assert killed["flag"] is True
    assert "hang" in exc_info.value.stderr_tail


async def _stub_sleep(_seconds):
    """No-op replacement for asyncio.sleep in the timeout test."""
    return None


# --------------------------------------------------------------------------
# Task 8 — merge_worker_artifacts
# --------------------------------------------------------------------------


def test_events_jsonl_merge_preserves_line_order(tmp_path):
    """worker/events.jsonl is appended line-for-line to canonical events.jsonl."""
    from src.eval.bench_subprocess import merge_worker_artifacts

    run_dir = tmp_path
    worker_dir = run_dir / "iter_3" / "worker"
    worker_dir.mkdir(parents=True)
    (worker_dir / "events.jsonl").write_text(
        '{"kind":"bench_done","iter":3,"candidate_idx":0}\n'
        '{"kind":"profile_done","iter":3,"candidate_idx":0}\n'
    )
    canonical = run_dir / "events.jsonl"
    canonical.write_text('{"kind":"iter_start","iter":3}\n')

    counts = merge_worker_artifacts(
        run_dir=run_dir,
        worker_dir=worker_dir,
        iter_no=3,
        response={"candidates": []},
        ncu_cache_dir=run_dir / "ncu_cache",
    )

    lines = canonical.read_text().splitlines()
    assert lines[0] == '{"kind":"iter_start","iter":3}'
    assert "bench_done" in lines[1]
    assert "profile_done" in lines[2]
    assert counts["event_count"] == 2


def test_ncu_rep_copied_to_cache_after_merge(tmp_path):
    """cand_<idx>.ncu-rep files in worker_dir copy into the shared NCU cache."""
    from src.eval.bench_subprocess import merge_worker_artifacts

    run_dir = tmp_path
    worker_dir = run_dir / "iter_3" / "worker"
    worker_dir.mkdir(parents=True)
    ncu_cache = run_dir / "ncu_cache"  # intentionally not pre-created
    (worker_dir / "cand_2.ncu-rep").write_bytes(b"fake ncu rep contents")
    response = {"candidates": [], "winner_idx": 2}

    counts = merge_worker_artifacts(
        run_dir=run_dir,
        worker_dir=worker_dir,
        iter_no=3,
        response=response,
        ncu_cache_dir=ncu_cache,
    )

    cached = sorted(ncu_cache.glob("*.ncu-rep"))
    assert len(cached) == 1
    assert cached[0].name == "cand_2.ncu-rep"
    assert cached[0].read_bytes() == b"fake ncu rep contents"
    assert counts["ncu_rep_count"] == 1


def test_merge_returns_event_count_and_ncu_rep_count(tmp_path):
    """Counts populate the parent's worker_chunk_merged event payload."""
    from src.eval.bench_subprocess import merge_worker_artifacts

    run_dir = tmp_path
    worker_dir = run_dir / "iter_3" / "worker"
    worker_dir.mkdir(parents=True)
    ncu_cache = run_dir / "ncu_cache"
    ncu_cache.mkdir()
    (worker_dir / "events.jsonl").write_text('{"a":1}\n{"a":2}\n')
    (worker_dir / "cand_0.ncu-rep").write_bytes(b"x")
    (worker_dir / "cand_1.ncu-rep").write_bytes(b"y")

    counts = merge_worker_artifacts(
        run_dir=run_dir,
        worker_dir=worker_dir,
        iter_no=3,
        response={"candidates": []},
        ncu_cache_dir=ncu_cache,
    )
    assert counts["event_count"] == 2
    assert counts["ncu_rep_count"] == 2


def test_merge_handles_missing_events_jsonl_gracefully(tmp_path):
    """No worker/events.jsonl → event_count=0, no error."""
    from src.eval.bench_subprocess import merge_worker_artifacts

    run_dir = tmp_path
    worker_dir = run_dir / "iter_3" / "worker"
    worker_dir.mkdir(parents=True)
    ncu_cache = run_dir / "ncu_cache"

    counts = merge_worker_artifacts(
        run_dir=run_dir,
        worker_dir=worker_dir,
        iter_no=3,
        response={"candidates": []},
        ncu_cache_dir=ncu_cache,
    )
    assert counts["event_count"] == 0
    assert counts["ncu_rep_count"] == 0


def test_malformed_response_json_treated_as_crash(tmp_path, monkeypatch):
    """Codex 2026-05-26 review P2 fix #2.

    If the worker exits cleanly (returncode 0) but writes a truncated or
    otherwise invalid ``response.json``, the helper must surface this as
    ``WorkerCrashed`` so the orchestrator's crash-recovery path engages
    (emit ``bench_worker_crashed``, bump ``consecutive_worker_crashes``,
    honor the configured threshold). Previously, ``json.loads`` raised
    ``JSONDecodeError`` which the orchestrator's ``except WorkerCrashed``
    didn't catch — the whole run aborted on the very first malformed
    response, ignoring the threshold entirely.
    """
    import asyncio
    from pathlib import Path
    from src.eval.bench_subprocess import run_bench_subprocess, WorkerCrashed

    worker_dir = tmp_path / "iter_3" / "worker"
    worker_dir.mkdir(parents=True)

    class FakePopen:
        def __init__(self, args, **kwargs):
            # Clean exit, but write garbage to response.json.
            response_path = Path(args[args.index("--response") + 1])
            response_path.write_text("{this is not valid json")
            # Also write something to stderr so the tail is recoverable.
            kwargs["stderr"].write("worker exited cleanly but produced bad json\n")
            kwargs["stderr"].flush()
            self.returncode = 0
        def wait(self, timeout=None): return 0
        def terminate(self): pass
        def kill(self): pass
        def poll(self): return 0

    monkeypatch.setattr("src.eval.bench_subprocess.subprocess.Popen", FakePopen)

    with pytest.raises(WorkerCrashed) as exc_info:
        asyncio.run(run_bench_subprocess(
            request={"iter_no": 3},
            worker_dir=worker_dir,
            worker_crash_threshold=3,
            worker_timeout_s=30.0,
        ))
    # Clean exit code surfaces as 0 — distinguishable from non-zero
    # crashes if a consumer wants to special-case "malformed but exited
    # cleanly" later (the orchestrator currently treats both as crashes).
    assert exc_info.value.returncode == 0
    # stderr_tail must be populated so the orchestrator's
    # ``bench_worker_crashed`` event carries postmortem detail.
    assert "cleanly but produced bad json" in exc_info.value.stderr_tail
