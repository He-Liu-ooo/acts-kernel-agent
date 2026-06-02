"""Tests for the JSONL append-only MemoryStore."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.memory.experience import ActionRecord, Experience
from src.memory.store import MemoryStore


def _exp(row_id: str = "r_abc", **overrides) -> Experience:
    defaults = dict(
        row_id=row_id,
        schema_version=1,
        kernel_type="matmul",
        hardware_arch="RTX6000Ada",
        scope="edge",
        speedup=1.5,
        action_applied=ActionRecord(action_id="a1", tier=3, name="vectorize"),
        title="Test lesson",
        lesson="A lesson body.",
        snippet_before="x = 1",
        snippet_after="x = 2",
        provenance={"run_id": "r1"},
        created_at="2026-05-24T00:00:00Z",
    )
    defaults.update(overrides)
    return Experience(**defaults)


def test_add_then_load_round_trip(tmp_path: Path):
    p = tmp_path / "store.jsonl"
    s = MemoryStore(p)
    s.add(_exp())
    s2 = MemoryStore(p)
    s2.load()
    rows = s2.all()
    assert len(rows) == 1
    assert rows[0].row_id == "r_abc"
    assert rows[0].action_applied.action_id == "a1"


def test_add_appends_one_line_per_call(tmp_path: Path):
    p = tmp_path / "store.jsonl"
    s = MemoryStore(p)
    s.add(_exp(row_id="r1"))
    s.add(_exp(row_id="r2"))
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["row_id"] == "r1"
    assert json.loads(lines[1])["row_id"] == "r2"


def test_add_many_single_open(tmp_path: Path):
    p = tmp_path / "store.jsonl"
    s = MemoryStore(p)
    s.add_many([_exp(row_id="r1"), _exp(row_id="r2"), _exp(row_id="r3")])
    lines = p.read_text().splitlines()
    assert [json.loads(line)["row_id"] for line in lines] == ["r1", "r2", "r3"]


def test_load_missing_file_is_empty(tmp_path: Path):
    p = tmp_path / "does_not_exist.jsonl"
    s = MemoryStore(p)
    s.load()
    assert s.all() == []


def test_tolerant_load_missing_optional_field(tmp_path: Path, caplog):
    p = tmp_path / "store.jsonl"
    # row missing provenance + created_at — should still load with defaults
    p.write_text(
        json.dumps({
            "row_id": "r1",
            "schema_version": 1,
            "kernel_type": "matmul",
            "hardware_arch": "RTX6000Ada",
            "scope": "edge",
            "speedup": 1.5,
            "action_applied": {
                "action_id": "a1", "tier": 3, "name": "x", "parameters": {},
            },
            "title": "t",
            "lesson": "l",
            "snippet_before": "a",
            "snippet_after": "b",
        }) + "\n"
    )
    s = MemoryStore(p)
    with caplog.at_level(logging.WARNING):
        s.load()
    assert len(s.all()) == 1
    assert s.all()[0].provenance == {}
    # one warn per missing field name in this load
    assert any("provenance" in r.message for r in caplog.records)


def test_skip_and_warn_on_non_integer_schema_version(tmp_path: Path, caplog):
    """Regression for ultra-review finding: a stringly-typed
    ``schema_version`` previously crashed ``load()`` with ``TypeError``
    on the ``>`` comparison, aborting the whole store read. The
    coerce-with-try inside the loop must skip just the offending row."""
    p = tmp_path / "store.jsonl"
    good = json.dumps({
        "row_id": "r1", "schema_version": 1, "kernel_type": "matmul",
        "hardware_arch": "RTX6000Ada", "scope": "edge", "speedup": 1.5,
        "action_applied": {"action_id": "a", "tier": 1, "name": "n", "parameters": {}},
        "title": "t", "lesson": "l", "snippet_before": "a", "snippet_after": "b",
        "provenance": {}, "created_at": "",
    })
    bad = json.dumps({"row_id": "r2", "schema_version": "2"})  # string!
    p.write_text(good + "\n" + bad + "\n")
    s = MemoryStore(p)
    with caplog.at_level(logging.WARNING):
        s.load()
    assert [r.row_id for r in s.all()] == ["r1"]
    assert any("schema_version" in r.message for r in caplog.records)


def test_skip_and_warn_on_future_schema_version(tmp_path: Path, caplog):
    p = tmp_path / "store.jsonl"
    row_v1 = json.dumps({
        "row_id": "r1", "schema_version": 1, "kernel_type": "matmul",
        "hardware_arch": "RTX6000Ada", "scope": "edge", "speedup": 1.5,
        "action_applied": {"action_id": "a", "tier": 1, "name": "n", "parameters": {}},
        "title": "t", "lesson": "l", "snippet_before": "a", "snippet_after": "b",
        "provenance": {}, "created_at": "",
    })
    row_v99 = json.dumps({"row_id": "r2", "schema_version": 99})
    p.write_text(row_v1 + "\n" + row_v99 + "\n")
    s = MemoryStore(p)
    with caplog.at_level(logging.WARNING):
        s.load()
    assert [r.row_id for r in s.all()] == ["r1"]
    assert any("schema_version" in r.message for r in caplog.records)


def test_skip_and_warn_on_malformed_line(tmp_path: Path, caplog):
    p = tmp_path / "store.jsonl"
    good = json.dumps({
        "row_id": "r1", "schema_version": 1, "kernel_type": "matmul",
        "hardware_arch": "RTX6000Ada", "scope": "edge", "speedup": 1.5,
        "action_applied": {"action_id": "a", "tier": 1, "name": "n", "parameters": {}},
        "title": "t", "lesson": "l", "snippet_before": "a", "snippet_after": "b",
        "provenance": {}, "created_at": "",
    })
    p.write_text(good + "\n" + "{not json\n" + good + "\n")
    s = MemoryStore(p)
    with caplog.at_level(logging.WARNING):
        s.load()
    assert len(s.all()) == 2
    assert any(
        "malformed" in r.message.lower() or "line" in r.message.lower()
        for r in caplog.records
    )


def test_load_skips_non_dict_json_rows(tmp_path: Path, caplog):
    """Regression for adversarial-review finding 3: a syntactically valid
    but non-object JSON row (``[]`` / ``null`` / scalar / string) decodes
    cleanly, so the ``json.JSONDecodeError`` guard does not fire — but
    ``d.get(...)`` then raises ``AttributeError``, escaping ``load()`` and
    aborting the whole store read. The tolerant reader must skip the bad
    row and keep going."""
    p = tmp_path / "store.jsonl"
    s = MemoryStore(p)
    s.add(_exp(row_id="r1"))  # one valid Experience row, serialized by add()
    valid_line = p.read_text().rstrip("\n")
    p.write_text(valid_line + '\n[]\nnull\n42\n"hello"\n')
    s2 = MemoryStore(p)
    with caplog.at_level(logging.WARNING):
        s2.load()  # MUST NOT raise
    assert len(s2.all()) == 1  # only the valid row survives
    assert s2.all()[0].row_id == "r1"


def test_run_scope_row_with_none_action_round_trips(tmp_path: Path):
    """G3 rows carry ``action_applied=None``; the store must serialize
    + deserialize ``None`` cleanly so the schema invariant survives
    a write→read cycle."""
    p = tmp_path / "store.jsonl"
    s = MemoryStore(p)
    run_row = _exp(row_id="g3", scope="run", action_applied=None)
    s.add(run_row)
    s2 = MemoryStore(p)
    s2.load()
    rows = s2.all()
    assert len(rows) == 1
    assert rows[0].scope == "run"
    assert rows[0].action_applied is None


def _valid_row(**overrides) -> dict:
    """A schema-valid JSONL row dict carrying every required field.

    Mirrors ``_CANONICAL_FIELDS`` + the keys ``_row_to_experience`` reads.
    Callers override individual fields to inject malformed values.
    """
    row = {
        "row_id": "r1",
        "schema_version": 1,
        "kernel_type": "matmul",
        "hardware_arch": "RTX6000Ada",
        "scope": "edge",
        "speedup": 1.5,
        "action_applied": {
            "action_id": "a", "tier": 1, "name": "n", "parameters": {},
        },
        "title": "t",
        "lesson": "l",
        "snippet_before": "a",
        "snippet_after": "b",
        "provenance": {},
        "created_at": "",
    }
    row.update(overrides)
    return row


def test_load_skips_row_with_nonfinite_or_nonpositive_speedup(tmp_path: Path):
    """Regression: a NaN / inf / 0 / negative ``speedup`` previously loaded,
    then ``random.choices()`` in the retriever raised ``Total of weights
    must be finite``, aborting the run. ``load()`` must skip+warn such rows."""
    import math  # noqa: F401  (parity with implementation guard)

    path = tmp_path / "store.jsonl"
    good = _valid_row(speedup=1.5)
    rows = [
        _valid_row(speedup=float("nan")),
        _valid_row(speedup=float("inf")),
        _valid_row(speedup=0.0),
        _valid_row(speedup=-1.0),
        good,
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    store = MemoryStore(path)
    store.load()
    speedups = [e.speedup for e in store.all()]
    assert speedups == [1.5]  # only the good row survived


def test_load_skips_row_with_nonstring_title_or_lesson(tmp_path: Path):
    """Regression: a non-string ``title`` / ``lesson`` previously loaded,
    then the planner's ``_neutralize_prompt_markdown`` called ``.splitlines()``
    on a non-str and crashed the run. ``load()`` must skip+warn such rows."""
    path = tmp_path / "store.jsonl"
    rows = [
        _valid_row(title=123),
        _valid_row(lesson=["not", "a", "string"]),
        _valid_row(title="ok", lesson="ok"),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    store = MemoryStore(path)
    store.load()
    assert len(store.all()) == 1


def test_lazy_mkdir_on_first_write(tmp_path: Path):
    p = tmp_path / "nested" / "deep" / "store.jsonl"
    assert not p.parent.exists()
    s = MemoryStore(p)
    s.add(_exp())
    assert p.exists()
    assert p.parent.is_dir()
