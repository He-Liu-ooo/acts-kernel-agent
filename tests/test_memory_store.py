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


def test_add_writes_one_line_per_distinct_key(tmp_path: Path):
    # Distinct dedup keys (distinct conditions) each get their own line; the
    # store is consolidated-by-key on write (no longer pure-append).
    p = tmp_path / "store.jsonl"
    s = MemoryStore(p)
    s.add(_exp(row_id="r1", condition="compute_bound"))
    s.add(_exp(row_id="r2", condition="memory_bound"))
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert {json.loads(line)["row_id"] for line in lines} == {"r1", "r2"}


def test_add_many_distinct_keys_each_written(tmp_path: Path):
    p = tmp_path / "store.jsonl"
    s = MemoryStore(p)
    s.add_many([
        _exp(row_id="r1", condition="compute_bound"),
        _exp(row_id="r2", condition="memory_bound"),
        _exp(row_id="r3", condition="latency_bound"),
    ])
    lines = p.read_text().splitlines()
    assert {json.loads(line)["row_id"] for line in lines} == {"r1", "r2", "r3"}


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

    def good(**overrides):
        row = {
            "row_id": "r1", "schema_version": 1, "kernel_type": "matmul",
            "hardware_arch": "RTX6000Ada", "scope": "edge", "speedup": 1.5,
            "action_applied": {"action_id": "a", "tier": 1, "name": "n", "parameters": {}},
            "title": "t", "lesson": "l", "snippet_before": "a", "snippet_after": "b",
            "provenance": {}, "created_at": "",
        }
        row.update(overrides)
        return json.dumps(row)

    # Two valid rows with DISTINCT dedup keys (distinct conditions) so the
    # load-time dedup does not collapse them — the point of the test is that
    # the malformed line between them is skipped+warned, not the dedup.
    p.write_text(
        good(row_id="r1", condition="compute_bound") + "\n"
        + "{not json\n"
        + good(row_id="r2", condition="memory_bound") + "\n"
    )
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


def _e(row_id, *, action_id="t1_grid_shape", condition="compute_bound",
       speedup=1.2, created="2026-06-02T00:00:00+00:00", scope="edge"):
    action = None if scope == "run" else ActionRecord(action_id, 1, action_id, {})
    return Experience(
        row_id=row_id, schema_version=1, kernel_type="matmul", hardware_arch="RTX6000Ada",
        scope=scope, speedup=speedup, action_applied=action, title="t", lesson="l",
        snippet_before="b", snippet_after="a", provenance={}, created_at=created,
        condition=condition)


def test_add_many_merges_same_key_keeping_best_speedup(tmp_path):
    s = MemoryStore(tmp_path / "s.jsonl")
    s.add_many([_e("lo", speedup=1.2), _e("hi", speedup=1.6)])
    rows = s.all()
    assert len(rows) == 1 and rows[0].row_id == "hi"


def test_add_many_preserves_distinct_conditions(tmp_path):
    s = MemoryStore(tmp_path / "s.jsonl")
    s.add_many([_e("a", condition="compute_bound"), _e("b", condition="memory_bound")])
    assert {r.row_id for r in s.all()} == {"a", "b"}


def test_condition_round_trips_through_disk(tmp_path):
    p = tmp_path / "s.jsonl"
    MemoryStore(p).add(_e("x", condition="compute_bound | BLOCK_N=32"))
    s2 = MemoryStore(p)
    s2.load()
    assert s2.all()[0].condition == "compute_bound | BLOCK_N=32"


def test_load_dedups_existing_duplicates(tmp_path):
    p = tmp_path / "s.jsonl"
    # Two same-key rows written directly (simulating a legacy un-compacted file).
    s = MemoryStore(p)
    s.add(_e("lo", speedup=1.2))
    s._experiences = []           # force a fresh load from disk
    s.add(_e("hi", speedup=1.6))  # rewrites; both same key -> keep hi
    s2 = MemoryStore(p)
    s2.load()
    assert len(s2.all()) == 1 and s2.all()[0].row_id == "hi"


def test_legacy_row_without_condition_loads_as_empty(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        '{"row_id":"old","schema_version":1,"kernel_type":"matmul",'
        '"hardware_arch":"RTX6000Ada","scope":"edge","speedup":1.3,'
        '"action_applied":{"action_id":"t1_grid_shape","tier":1,"name":"x","parameters":{}},'
        '"title":"t","lesson":"l","snippet_before":"","snippet_after":"",'
        '"provenance":{},"created_at":"2026-05-31T00:00:00+00:00"}\n')
    s = MemoryStore(p); s.load()
    # Legacy row has no "condition" key AND empty params, so the #3
    # params-only backfill yields "" — still loads as empty.
    assert s.all()[0].condition == ""


def test_write_only_mode_does_not_truncate_existing_store(tmp_path):
    """#1 regression: a fresh store that never called ``load()``
    (write-only mode) must NOT truncate pre-existing on-disk lessons when
    it writes. The write path re-reads + merges disk before rewriting."""
    p = tmp_path / "s.jsonl"
    # Seed the store FILE with a pre-existing row (distinct dedup key).
    seed = MemoryStore(p)
    seed.add(_e("pre", action_id="t1_grid_shape", condition="compute_bound"))
    # Fresh store, NO load() — simulating write-only mode.
    writer = MemoryStore(p)
    writer.add(_e("new", action_id="t2_vectorize", condition="memory_bound"))
    # Third store: load and assert BOTH rows survived.
    reader = MemoryStore(p)
    reader.load()
    assert {r.row_id for r in reader.all()} == {"pre", "new"}


def test_non_string_condition_loads_as_empty(tmp_path, caplog):
    """#2 load-guard: a non-string ``condition`` (list/dict) drops to "" with
    a warning and the row is KEPT — condition is not correctness-load-bearing,
    and ``dedup_key`` must only ever see a hashable string."""
    p = tmp_path / "s.jsonl"
    # Distinct action_ids so the two rows keep distinct dedup keys once both
    # conditions drop to "" — the point of the test is the load-guard, not
    # dedup collapse.
    rows = [
        _valid_row(
            row_id="r_list", condition=["a", "b"],
            action_applied={"action_id": "a1", "tier": 1, "name": "n", "parameters": {}},
        ),
        _valid_row(
            row_id="r_dict", condition={"k": "v"},
            action_applied={"action_id": "a2", "tier": 1, "name": "n", "parameters": {}},
        ),
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = MemoryStore(p)
    with caplog.at_level(logging.WARNING):
        s.load()  # MUST NOT raise
    loaded = s.all()
    assert len(loaded) == 2
    assert all(e.condition == "" for e in loaded)
    assert any("condition" in r.message for r in caplog.records)


def test_legacy_distinct_params_not_collapsed(tmp_path):
    """#3 regression: two legacy edge rows (no ``condition`` key), same
    kernel/arch/scope/action_id but distinct params, must NOT collapse on
    load — the params-only condition backfill keeps their dedup keys
    distinct."""
    p = tmp_path / "s.jsonl"

    def legacy(row_id, block_n):
        return json.dumps({
            "row_id": row_id, "schema_version": 1, "kernel_type": "matmul",
            "hardware_arch": "RTX6000Ada", "scope": "edge", "speedup": 1.3,
            "action_applied": {
                "action_id": "t1_grid_shape", "tier": 1, "name": "x",
                "parameters": {"BLOCK_N": block_n},
            },
            "title": "t", "lesson": "l", "snippet_before": "",
            "snippet_after": "", "provenance": {},
            "created_at": "2026-05-31T00:00:00+00:00",
        })

    p.write_text(legacy("r32", "32") + "\n" + legacy("r64", "64") + "\n")
    s = MemoryStore(p)
    s.load()
    assert {r.row_id for r in s.all()} == {"r32", "r64"}


def test_future_version_row_preserved_through_rewrite(tmp_path):
    """P1 regression: the read-merge-rewrite must carry forward-compat rows
    (``schema_version > KNOWN_VERSION``) through the compaction verbatim, so
    an older binary cannot silently delete a newer binary's lessons. The
    future row is preserved in the FILE but not surfaced by ``load()``."""
    p = tmp_path / "s.jsonl"
    v1 = json.dumps({
        "row_id": "r1", "schema_version": 1, "kernel_type": "matmul",
        "hardware_arch": "RTX6000Ada", "scope": "edge", "speedup": 1.5,
        "action_applied": {"action_id": "a1", "tier": 1, "name": "n", "parameters": {}},
        "title": "t", "lesson": "l", "snippet_before": "a", "snippet_after": "b",
        "provenance": {}, "created_at": "2026-06-01T00:00:00Z", "condition": "compute_bound",
    })
    v2 = json.dumps({
        "row_id": "r_future", "schema_version": 2, "kernel_type": "matmul",
        "hardware_arch": "RTX6000Ada", "scope": "edge", "speedup": 1.9,
        "action_applied": {"action_id": "a1", "tier": 1, "name": "n", "parameters": {}},
        "title": "t", "lesson": "l", "snippet_before": "a", "snippet_after": "b",
        "provenance": {}, "created_at": "2026-06-01T00:00:00Z", "condition": "memory_bound",
        "new_v2_field": "stuff",
    })
    p.write_text(v1 + "\n" + v2 + "\n")
    # Fresh store, no load() — write-only mode rewrites the whole file.
    writer = MemoryStore(p)
    writer.add(_e("new", action_id="t2_vectorize", condition="latency_bound"))
    # The schema_version:2 line must still be present in the file.
    schema_versions = [
        json.loads(line)["schema_version"]
        for line in p.read_text().splitlines() if line.strip()
    ]
    assert schema_versions.count(2) == 1
    # The v2 row's distinguishing field is preserved verbatim.
    assert any(
        json.loads(line).get("new_v2_field") == "stuff"
        for line in p.read_text().splitlines() if line.strip()
    )
    # A fresh load surfaces the v1 rows (preexisting + new) but NOT the future row.
    reader = MemoryStore(p)
    reader.load()
    assert {r.row_id for r in reader.all()} == {"r1", "new"}


def test_non_string_identity_field_row_skipped(tmp_path, caplog):
    """P2a regression: a v1 row with a non-string dedup-key identity field
    (``kernel_type`` as a list, or ``action_applied.action_id`` as a list)
    would make ``dedup_key`` an unhashable tuple and abort the entire
    load/merge. The parser must SKIP such rows (junk) and keep valid ones."""
    p = tmp_path / "s.jsonl"
    rows = [
        _valid_row(row_id="r_bad_kernel", kernel_type=["matmul"]),
        _valid_row(
            row_id="r_bad_action",
            action_applied={"action_id": ["a"], "tier": 1, "name": "n", "parameters": {}},
        ),
        _valid_row(row_id="r_ok", condition="compute_bound"),
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = MemoryStore(p)
    with caplog.at_level(logging.WARNING):
        s.load()  # MUST NOT raise
    assert {r.row_id for r in s.all()} == {"r_ok"}


def test_non_string_created_at_does_not_crash_tiebreak(tmp_path, caplog):
    """P2b regression: ``dedup_best`` tie-breaks on ``(speedup, created_at)``.
    A non-string ``created_at`` (dict) on a same-key/equal-speedup row makes
    str-vs-dict comparison raise ``TypeError`` and aborts the whole merge.
    The parser must coerce it to "" — keeping the row but losing the tie to
    the valid-timestamp row."""
    p = tmp_path / "s.jsonl"
    good = _valid_row(
        row_id="r_good", speedup=1.5, created_at="2026-06-01T00:00:00Z",
        condition="compute_bound",
    )
    bad = _valid_row(
        row_id="r_bad", speedup=1.5, created_at={"corrupt": True},
        condition="compute_bound",
    )
    p.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    s = MemoryStore(p)
    with caplog.at_level(logging.WARNING):
        s.load()  # MUST NOT raise
    rows = s.all()
    # Same dedup key, equal speedup → tie broken by created_at. The dict
    # coerces to "" which sorts before the real timestamp, so r_good wins.
    assert len(rows) == 1
    assert rows[0].row_id == "r_good"
