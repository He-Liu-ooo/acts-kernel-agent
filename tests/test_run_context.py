"""Unit tests for ``src.runtime.run_context.RunContext``."""
import logging
import sys

import pytest

from src.runtime.run_context import RunContext


def test_create_builds_dir_layout(tmp_path):
    ctx = RunContext.create(root=tmp_path)
    try:
        assert ctx.run_dir is not None
        assert ctx.run_dir.exists() and ctx.run_dir.is_dir()
        assert ctx.run_dir.parent == tmp_path
        assert ctx.run_dir.name.startswith("run_")
        assert ctx.events_path == ctx.run_dir / "events.jsonl"
        assert ctx.events_path.exists()
        assert ctx.log_path == ctx.run_dir / "run.log"
        assert ctx.traces_dir == ctx.run_dir / "traces"
        assert ctx.traces_dir.exists()
    finally:
        ctx.close()


def test_create_configures_stdlib_logging(tmp_path):
    ctx = RunContext.create(root=tmp_path)
    try:
        logging.getLogger("src.runtime.test_only").info("hello-logger")
        for h in logging.getLogger().handlers:
            h.flush()
        content = ctx.log_path.read_text()
        assert "hello-logger" in content
    finally:
        ctx.close()


def test_create_silences_sdk_loggers(tmp_path):
    ctx = RunContext.create(root=tmp_path)
    try:
        for name in ("httpx", "openai", "agents"):
            assert logging.getLogger(name).level == logging.WARNING
    finally:
        ctx.close()


def test_create_sdk_absent_degrades_gracefully(tmp_path, monkeypatch):
    # Simulate the ``agents`` package being unavailable.
    for key in list(sys.modules):
        if key == "agents" or key.startswith("agents."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    monkeypatch.setitem(sys.modules, "agents", None)  # import -> ImportError

    ctx = RunContext.create(root=tmp_path)
    try:
        assert ctx.run_dir is not None
        assert ctx.traces_dir is not None and ctx.traces_dir.exists()
    finally:
        ctx.close()


def test_create_mkdir_failure_returns_null_context(tmp_path, caplog):
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o555)  # no write bit
    try:
        with caplog.at_level(logging.WARNING, logger="src.runtime.run_context"):
            ctx = RunContext.create(root=read_only)
        try:
            assert ctx.run_dir is None
            assert ctx.events_path is None
            assert ctx.log_path is None
            assert any("run-dir" in r.getMessage().lower() for r in caplog.records)
            # emit() is logger-only in this mode — no crash.
            from src.runtime.events import emit
            emit("run_start")
        finally:
            ctx.close()
    finally:
        read_only.chmod(0o755)  # let pytest clean up


def test_close_is_idempotent(tmp_path):
    ctx = RunContext.create(root=tmp_path)
    ctx.close()
    ctx.close()  # must not raise


def test_create_mid_setup_osError_falls_back_to_null_context(tmp_path, monkeypatch, caplog):
    """A failure AFTER mkdir (disk quota hit between mkdir and events FH
    open, FD exhaustion, mid-setup perm change) must degrade to the same
    null-context fallback as an mkdir failure. Before this fix the error
    propagated and aborted the entire run — Codex review 2026-04-23
    Finding 3."""
    import src.runtime.run_context as rc_mod

    # Let mkdir succeed, then make FileHandler construction (which opens
    # the log file internally) fail with OSError. Simulates disk quota
    # or FD exhaustion that strikes after the dir exists.
    orig_file_handler = logging.FileHandler

    def exploding_file_handler(*args, **kwargs):
        raise OSError("simulated quota exhaustion")

    monkeypatch.setattr(rc_mod.logging, "FileHandler", exploding_file_handler)

    with caplog.at_level(logging.WARNING, logger="src.runtime.run_context"):
        ctx = RunContext.create(root=tmp_path)
    try:
        # Null-context fallback — same shape as the mkdir-failure path.
        assert ctx.run_dir is None
        assert ctx.events_path is None
        assert ctx.log_path is None
        assert ctx.traces_dir is None
        assert any("run-dir setup failed" in r.getMessage() for r in caplog.records)
        # emit() must still be safe in logger-only mode.
        from src.runtime.events import emit
        emit("run_start")
    finally:
        ctx.close()
        # Safety net: cleanup has unbound events FH.
        assert rc_mod.events._events_fh is None


def test_create_twice_in_same_second_does_not_collide(tmp_path):
    """Two back-to-back invocations sharing a --run-dir must each get a
    distinct run_<UTC>/ dir — second-precision timestamps previously
    trip FileExistsError and dump the second run into the null-context
    fallback."""
    ctx_a = RunContext.create(root=tmp_path)
    ctx_b = RunContext.create(root=tmp_path)
    try:
        assert ctx_a.run_dir is not None
        assert ctx_b.run_dir is not None
        assert ctx_a.run_dir != ctx_b.run_dir
        assert ctx_a.run_dir.exists()
        assert ctx_b.run_dir.exists()
    finally:
        ctx_a.close()
        ctx_b.close()
