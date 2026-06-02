# tests/test_correctness_worker.py
"""Tier-1 tests for correctness_worker.run_request (torch/SOL mocked)."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import src.eval.correctness_worker as cw


class _FakeStage:
    def __init__(self, value): self.value = value


@contextmanager
def _noop_anti_cheat(_names):
    # The real _run_per_iter_anti_cheat lazily imports src.eval.anti_cheat,
    # which imports torch at module top — unavailable on the Tier-1 venv.
    # Stub it so run_request's `with` block stays torchless under mock.
    yield None


def _patch_common(monkeypatch):
    # Rehydration helpers → trivial stand-ins.
    monkeypatch.setattr(cw, "_rehydrate_kernel_spec", lambda d: SimpleNamespace(entrypoint="kernel_fn"))
    monkeypatch.setattr(cw, "_rehydrate_workloads", lambda raw: [object() for _ in raw])
    monkeypatch.setattr(cw, "_load_definition", lambda p: SimpleNamespace(reference="def run(*a): return a"))
    monkeypatch.setattr(cw, "_build_input_generators", lambda req, wls, definition=None: [lambda s: () for _ in wls])
    monkeypatch.setattr(cw, "_build_reference_fn", lambda src: (lambda *a: a))
    # Anti-cheat context manager → torchless no-op (real impl imports torch).
    monkeypatch.setattr(cw, "_run_per_iter_anti_cheat", _noop_anti_cheat)
    # Kernel + compile.
    monkeypatch.setattr(cw, "_make_kernel", lambda spec, src, dps: SimpleNamespace())
    monkeypatch.setattr(cw, "_compile", lambda kernel: SimpleNamespace(success=True, compiled_fn=lambda *a: a, error_message=None))


def test_gate_mode_all_pass(monkeypatch):
    import src.eval.correctness as correctness_mod
    _patch_common(monkeypatch)
    monkeypatch.setattr(correctness_mod, "verify_correctness", lambda **kw: SimpleNamespace(
        passed=True, failed_stage=None, error_message=None, max_abs_error=2e-3))
    req = {"mode": "gate", "kernel_spec": {}, "source_code": "x", "dps": False,
           "definition_path": "/p", "workloads": [{}, {}, {}], "blob_roots": [],
           "input_seed": 0, "anti_cheat_critical_names": []}
    out = cw.run_request(req)
    assert out["passed"] is True
    assert out["total_workloads"] == 3
    assert out["max_err"] == pytest.approx(2e-3)


def test_gate_mode_fails_on_second_workload(monkeypatch):
    import src.eval.correctness as correctness_mod
    _patch_common(monkeypatch)
    calls = {"n": 0}
    def _verify(**kw):
        calls["n"] += 1
        if calls["n"] == 2:
            return SimpleNamespace(passed=False, failed_stage=_FakeStage("numerical"),
                                   error_message="mismatch at [0]", max_abs_error=9.9)
        return SimpleNamespace(passed=True, failed_stage=None, error_message=None, max_abs_error=1e-4)
    monkeypatch.setattr(correctness_mod, "verify_correctness", _verify)
    req = {"mode": "gate", "kernel_spec": {}, "source_code": "x", "dps": False,
           "definition_path": "/p", "workloads": [{}, {}, {}], "blob_roots": [],
           "input_seed": 0, "anti_cheat_critical_names": []}
    out = cw.run_request(req)
    assert out["passed"] is False
    assert out["failed_stage"] == "numerical"
    assert out["failed_workload_idx"] == 2
    assert "mismatch" in out["error_message"]


def test_build_input_generators_reads_top_level_blob_roots(monkeypatch):
    from pathlib import Path
    import src.eval.correctness_worker as cw
    captured = {}
    def _fake_big(definition, wl, blob_roots=None):
        captured["blob_roots"] = blob_roots
        return lambda s: ()
    monkeypatch.setattr("src.eval.inputs.build_input_generator", _fake_big)
    monkeypatch.setattr(cw, "_load_definition", lambda p: object())
    req = {"definition_path": "/p", "blob_roots": ["/root/a", "/root/b"]}
    cw._build_input_generators(req, [object()])
    assert captured["blob_roots"] == [Path("/root/a"), Path("/root/b")]


def test_build_input_generators_empty_blob_roots_yields_none(monkeypatch):
    import src.eval.correctness_worker as cw
    captured = {}
    def _fake_big(definition, wl, blob_roots=None):
        captured["blob_roots"] = blob_roots
        return lambda s: ()
    monkeypatch.setattr("src.eval.inputs.build_input_generator", _fake_big)
    monkeypatch.setattr(cw, "_load_definition", lambda p: object())
    # Absent blob_roots entirely.
    cw._build_input_generators({"definition_path": "/p"}, [object()])
    assert captured["blob_roots"] is None
    # Explicit empty list.
    cw._build_input_generators({"definition_path": "/p", "blob_roots": []}, [object()])
    assert captured["blob_roots"] is None


def test_compile_failure_short_circuits(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(cw, "_compile", lambda kernel: SimpleNamespace(
        success=False, compiled_fn=None, error_message="ptxas: bad"))
    req = {"mode": "gate", "kernel_spec": {}, "source_code": "x", "dps": False,
           "definition_path": "/p", "workloads": [{}], "blob_roots": [],
           "input_seed": 0, "anti_cheat_critical_names": []}
    out = cw.run_request(req)
    assert out["passed"] is False
    assert out["failed_stage"] == "compile"
    assert "ptxas" in out["error_message"]


def test_gate_uses_trusted_verify_despite_candidate_rebind(monkeypatch):
    """Candidate rebinding src.eval.correctness.verify_correctness at compile
    time must NOT forge a pass — run_request captures the trusted ref first."""
    import contextlib
    import types
    import src.eval.correctness as correctness_mod
    import src.eval.correctness_worker as cw

    def trusted_verify(**kwargs):  # the value present at run_request entry
        return types.SimpleNamespace(
            passed=False,
            failed_stage=types.SimpleNamespace(value="numeric"),
            error_message="real mismatch", max_abs_error=1.0,
        )
    monkeypatch.setattr(correctness_mod, "verify_correctness", trusted_verify)

    def forged_verify(**kwargs):  # what the candidate installs
        return types.SimpleNamespace(
            passed=True, failed_stage=None, error_message=None, max_abs_error=0.0,
        )

    def fake_compile(kernel):
        correctness_mod.verify_correctness = forged_verify  # candidate attack at module scope
        return types.SimpleNamespace(
            success=True, compiled_fn=(lambda *a, **k: None), error_message=None,
        )
    monkeypatch.setattr(cw, "_compile", fake_compile)
    monkeypatch.setattr(cw, "_rehydrate_kernel_spec", lambda d: object())
    monkeypatch.setattr(cw, "_make_kernel", lambda spec, src, dps: object())
    monkeypatch.setattr(cw, "_load_definition", lambda p: types.SimpleNamespace(reference=object()))
    monkeypatch.setattr(cw, "_rehydrate_workloads", lambda w: [object()])
    monkeypatch.setattr(cw, "_build_input_generators", lambda req, wls, defn: [lambda seed: ()])
    monkeypatch.setattr(cw, "_build_reference_fn", lambda ref: (lambda *a, **k: None))
    monkeypatch.setattr(cw, "_run_per_iter_anti_cheat", lambda names: contextlib.nullcontext())

    resp = cw.run_request({
        "mode": "gate", "kernel_spec": {}, "source_code": "x = 1", "dps": False,
        "definition_path": "/tmp/none", "workloads": [{}], "input_seed": 0,
        "anti_cheat_critical_names": [],
    })
    assert resp["passed"] is False  # trusted FAIL wins, not the forge


def test_strict_uses_trusted_compare_despite_candidate_rebind(monkeypatch):
    """Same property for strict_recheck mode + strict_compare_one_workload."""
    import contextlib
    import types
    import src.eval.correctness as correctness_mod
    import src.eval.correctness_worker as cw

    monkeypatch.setattr(correctness_mod, "strict_compare_one_workload",
                        lambda **kwargs: False)  # trusted: mismatch

    def fake_compile(kernel):
        correctness_mod.strict_compare_one_workload = lambda **kwargs: True  # forge: match
        return types.SimpleNamespace(
            success=True, compiled_fn=(lambda *a, **k: None), error_message=None,
        )
    monkeypatch.setattr(cw, "_compile", fake_compile)
    monkeypatch.setattr(cw, "_rehydrate_kernel_spec", lambda d: object())
    monkeypatch.setattr(cw, "_make_kernel", lambda spec, src, dps: object())
    monkeypatch.setattr(cw, "_load_definition", lambda p: types.SimpleNamespace(reference=object()))
    monkeypatch.setattr(cw, "_rehydrate_workloads", lambda w: [object()])
    monkeypatch.setattr(cw, "_build_input_generators", lambda req, wls, defn: [lambda seed: ()])
    monkeypatch.setattr(cw, "_build_reference_fn", lambda ref: (lambda *a, **k: None))
    monkeypatch.setattr(cw, "_run_per_iter_anti_cheat", lambda names: contextlib.nullcontext())

    resp = cw.run_request({
        "mode": "strict_recheck", "kernel_spec": {}, "source_code": "x = 1", "dps": False,
        "definition_path": "/tmp/none", "workloads": [{}], "input_seed": 0,
        "anti_cheat_critical_names": [], "strict_atol": 1e-5, "strict_rtol": 1e-4,
    })
    assert resp["passed"] is False  # trusted mismatch wins, not the forge
