"""Tests for eval/inputs.py — reference-function loader + input-generator factory.

The torch- and sol_execbench-backed factories can only run end-to-end with
torch + sol_execbench installed (and a GPU for real use). These tests
cover the torch-free helpers: the reference loader, its error modes, and
the structural shape of the factory outputs.
"""

from __future__ import annotations

import pytest

from src.eval.inputs import (
    ReferenceLoadError,
    build_reference_fn,
)


# ── build_reference_fn — happy path ────────────────────────────────────


def test_build_reference_fn_returns_callable_from_source():
    source = "def run(x, y):\n    return x + y\n"
    fn = build_reference_fn(source)
    assert callable(fn)
    assert fn(2, 3) == 5


def test_build_reference_fn_respects_custom_entrypoint():
    source = "def reference(x):\n    return x * 3\n"
    fn = build_reference_fn(source, entrypoint="reference")
    assert fn(4) == 12


def test_build_reference_fn_preserves_module_scope_helpers():
    """Helpers + module-level constants in the reference source stay reachable."""
    source = (
        "SCALE = 5\n"
        "def _inner(x):\n    return x + SCALE\n"
        "def run(x):\n    return _inner(x)\n"
    )
    fn = build_reference_fn(source)
    assert fn(1) == 6


# ── build_reference_fn — failure modes ─────────────────────────────────


def test_build_reference_fn_rejects_missing_entrypoint():
    source = "def not_run(): pass\n"
    with pytest.raises(ReferenceLoadError, match="run"):
        build_reference_fn(source)


def test_build_reference_fn_rejects_non_callable_entrypoint():
    source = "run = 42\n"
    with pytest.raises(ReferenceLoadError, match="callable"):
        build_reference_fn(source)


def test_build_reference_fn_propagates_syntax_error():
    source = "def run(: invalid\n"
    with pytest.raises(SyntaxError):
        build_reference_fn(source)
