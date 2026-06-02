"""Regression test: spawn_worker must kill the worker's entire process group
(descendants included) and reap the worker on timeout. Codex review fix."""

import asyncio
import os
import sys
import time
import pytest

from src.eval.worker_spawn import spawn_worker


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.asyncio
async def test_timeout_kills_descendants_and_reaps(tmp_path):
    # Fake worker: spawn a long-sleeping grandchild, record its pid, then sleep
    # well past the watchdog timeout. If spawn_worker only kills the direct
    # child, the grandchild survives.
    pidfile = tmp_path / "grandchild.pid"
    child_py = (
        "import subprocess, sys, time;"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        f"open(r'{pidfile}', 'w').write(str(g.pid));"
        "time.sleep(60)"
    )
    argv_prefix = [sys.executable, "-c", child_py]

    outcome = await spawn_worker(
        module="unused.because.argv_prefix.set",
        request={},
        worker_dir=tmp_path,
        timeout_s=2.0,
        argv_prefix=argv_prefix,
    )

    assert outcome.status == "timeout"

    # Give the OS a moment to tear down the group.
    for _ in range(50):
        if pidfile.exists():
            break
        time.sleep(0.05)
    assert pidfile.exists(), "grandchild never recorded its pid"
    grandchild_pid = int(pidfile.read_text().strip())

    # The grandchild must have been killed via the process-group signal.
    for _ in range(40):
        if not _alive(grandchild_pid):
            break
        time.sleep(0.05)
    assert not _alive(grandchild_pid), (
        f"grandchild {grandchild_pid} survived — process group was not killed"
    )


def test_kill_process_group_always_sigkills_even_if_child_exits_after_sigterm(monkeypatch):
    """If the direct child exits after SIGTERM but a descendant ignored it,
    the final group SIGKILL must still fire (not be gated on the child being
    alive). Codex 2026-06-02 review."""
    import signal
    from src.eval import worker_spawn

    signals = []
    monkeypatch.setattr(worker_spawn.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(worker_spawn.os, "killpg", lambda pgid, sig: signals.append(sig))

    class FakeProc:
        pid = 12345
        def wait(self, timeout=None):
            return 0  # direct child exits promptly after SIGTERM (no TimeoutExpired)

    worker_spawn._kill_process_group(FakeProc())
    assert signal.SIGTERM in signals
    assert signal.SIGKILL in signals  # pre-fix: skipped because the child exited
