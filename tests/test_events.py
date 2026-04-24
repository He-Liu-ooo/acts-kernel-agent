"""Unit tests for ``src.runtime.events.emit``."""
import json
import logging
from enum import Enum
from pathlib import Path

import pytest

from src.runtime import events


def test_emit_unbound_is_logger_only(caplog):
    events.unbind()  # ensure clean state
    with caplog.at_level(logging.INFO, logger="src.runtime.events"):
        events.emit("run_start", problem_path="foo/bar", model_configured=True)
    messages = [r.getMessage() for r in caplog.records]
    assert any("run_start" in m and '"problem_path"' in m for m in messages)


def test_emit_bound_writes_jsonl_line(tmp_path):
    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        events.emit("planner_selected", iter=3, technique="tiling", confidence=0.8)
    finally:
        events.unbind()
        fh.close()
    raw = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(raw) == 1
    rec = json.loads(raw[0])
    assert rec["kind"] == "planner_selected"
    assert rec["iter"] == 3
    assert rec["technique"] == "tiling"
    assert rec["confidence"] == 0.8
    assert "ts" in rec


def test_emit_unknown_kind_warns_but_writes(caplog, tmp_path):
    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        with caplog.at_level(logging.WARNING, logger="src.runtime.events"):
            events.emit("typo_kind_not_in_catalog", foo=1)
    finally:
        events.unbind()
        fh.close()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("unknown event kind" in r.getMessage() for r in warnings)
    raw = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(raw) == 1  # still written
    assert json.loads(raw[0])["kind"] == "typo_kind_not_in_catalog"


def test_emit_never_raises_on_io_error():
    class BrokenFH:
        def write(self, _):
            raise OSError("disk full")

    events.bind(BrokenFH())
    try:
        events.emit("run_start")  # must not raise
    finally:
        events.unbind()


def test_emit_log_line_includes_iter_for_iter_scoped_events(caplog):
    """run.log must surface ``iter`` for iteration-scoped events so
    operators tailing the file during a multi-iter run can tell which
    iteration produced each line. The JSONL sink always carried ``iter``;
    the human log sink previously dropped it because ``emit`` serialized
    only ``**fields`` (Codex review 2026-04-23)."""
    events.unbind()
    with caplog.at_level(logging.INFO, logger="src.runtime.events"):
        events.emit("iter_start", iter=3, parent_node_id="abc", parent_score=0.5)
    messages = [r.getMessage() for r in caplog.records]
    match = next((m for m in messages if m.startswith("iter_start")), None)
    assert match is not None, messages
    assert '"iter":3' in match, match
    assert '"parent_node_id":"abc"' in match


def test_emit_log_line_omits_iter_for_run_scoped_events(caplog):
    """Run-scoped events pass ``iter=None`` and the log line should not
    carry an ``iter`` key at all — serializing ``"iter":null`` would be
    noise that makes per-iter greps less reliable."""
    events.unbind()
    with caplog.at_level(logging.INFO, logger="src.runtime.events"):
        events.emit("run_start", problem_path="foo/bar")
    messages = [r.getMessage() for r in caplog.records]
    match = next((m for m in messages if m.startswith("run_start")), None)
    assert match is not None, messages
    assert '"iter"' not in match, match
    assert '"problem_path":"foo/bar"' in match


def test_emit_coerces_nonjson_values(tmp_path):
    class MyEnum(Enum):
        A = "a"

    fh = (tmp_path / "events.jsonl").open("w", buffering=1)
    events.bind(fh)
    try:
        events.emit("run_start", path=Path("/tmp/x"), tag=MyEnum.A)
    finally:
        events.unbind()
        fh.close()
    rec = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])
    assert rec["path"] == "/tmp/x"
    assert rec["tag"] == "MyEnum.A" or "MyEnum" in rec["tag"]
