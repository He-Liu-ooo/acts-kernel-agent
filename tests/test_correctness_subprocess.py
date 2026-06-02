"""Tier-1 tests for run_correctness_subprocess (no torch; stub child scripts)."""
from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

from src.eval.correctness_subprocess import (
    CorrectnessResult,
    run_correctness_subprocess,
)


def _run(coro):
    return asyncio.run(coro)


def _stub_worker(tmp_path: Path, body: str) -> Path:
    """Write a fake `python <script> --request r --response w` worker."""
    script = tmp_path / "stub_worker.py"
    script.write_text(textwrap.dedent(body))
    return script


def test_clean_response_is_parsed(tmp_path, monkeypatch):
    # Stub child writes a valid response.json and exits 0.
    script = _stub_worker(tmp_path, """
        import argparse, json
        p = argparse.ArgumentParser()
        p.add_argument("--request"); p.add_argument("--response")
        a = p.parse_args()
        json.dump({"schema_version": 1, "passed": True, "failed_stage": None,
                   "error_message": None, "max_err": 1.5e-3,
                   "total_workloads": 3, "failed_workload_idx": None},
                  open(a.response, "w"))
    """)
    monkeypatch.setattr(
        "src.eval.correctness_subprocess._WORKER_ARGV",
        [sys.executable, str(script)],
    )
    res = _run(run_correctness_subprocess(
        request={"mode": "gate"}, worker_dir=tmp_path, timeout_s=30.0,
    ))
    assert isinstance(res, CorrectnessResult)
    assert res.passed is True
    assert res.max_err == pytest.approx(1.5e-3)
    assert res.total_workloads == 3


def test_nonzero_exit_is_fail_closed_worker_crashed(tmp_path, monkeypatch):
    script = _stub_worker(tmp_path, """
        import sys
        sys.stderr.write("boom: device-side assert triggered\\n")
        sys.exit(1)
    """)
    monkeypatch.setattr(
        "src.eval.correctness_subprocess._WORKER_ARGV",
        [sys.executable, str(script)],
    )
    res = _run(run_correctness_subprocess(
        request={"mode": "gate"}, worker_dir=tmp_path, timeout_s=30.0,
    ))
    assert res.passed is False
    assert res.failed_stage == "worker_crashed"
    assert "device-side assert" in res.error_message


def test_timeout_kills_child_and_fails_closed(tmp_path, monkeypatch):
    script = _stub_worker(tmp_path, """
        import time
        time.sleep(30)
    """)
    monkeypatch.setattr(
        "src.eval.correctness_subprocess._WORKER_ARGV",
        [sys.executable, str(script)],
    )
    res = _run(run_correctness_subprocess(
        request={"mode": "gate"}, worker_dir=tmp_path, timeout_s=1.0,
    ))
    assert res.passed is False
    assert res.failed_stage == "timeout"


def test_correctness_isolation_error_is_importable_and_typed():
    from src.eval.correctness_subprocess import CorrectnessIsolationError
    assert issubclass(CorrectnessIsolationError, RuntimeError)
    with pytest.raises(CorrectnessIsolationError):
        raise CorrectnessIsolationError("x")
