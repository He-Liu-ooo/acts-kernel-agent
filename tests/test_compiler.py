"""Tests for kernels/compiler.py — file-backed importlib load + entrypoint resolve."""

from __future__ import annotations

from pathlib import Path

from src.kernels.compiler import compile_kernel
from src.kernels.kernel import Kernel, KernelSpec, KernelType


def _make_kernel(source: str, entrypoint: str = "kernel_fn") -> Kernel:
    spec = KernelSpec(
        name="test_kernel",
        kernel_type=KernelType.ELEMENTWISE,
        entrypoint=entrypoint,
    )
    return Kernel(spec=spec, source_code=source)


# ── KernelSpec.entrypoint field ────────────────────────────────────────────


def test_kernel_spec_entrypoint_defaults_to_kernel_fn():
    spec = KernelSpec(name="k", kernel_type=KernelType.MATMUL)
    assert spec.entrypoint == "kernel_fn"


def test_kernel_spec_entrypoint_configurable():
    spec = KernelSpec(name="k", kernel_type=KernelType.MATMUL, entrypoint="run")
    assert spec.entrypoint == "run"


# ── happy path ─────────────────────────────────────────────────────────────


def test_compile_kernel_returns_callable_on_success(tmp_path: Path):
    source = "def kernel_fn(x):\n    return x + 1\n"
    result = compile_kernel(_make_kernel(source), cache_dir=tmp_path)
    assert result.success is True
    assert result.error_message == ""
    assert callable(result.compiled_fn)
    assert result.compiled_fn(10) == 11


def test_compile_kernel_writes_source_to_cache_dir(tmp_path: Path):
    source = "def kernel_fn(): return 42\n"
    result = compile_kernel(_make_kernel(source), cache_dir=tmp_path)
    assert result.source_path is not None
    assert result.source_path.is_file()
    assert result.source_path.parent == tmp_path
    assert result.source_path.read_text() == source


def test_compile_kernel_respects_custom_entrypoint(tmp_path: Path):
    source = "def run(x):\n    return x * 2\n"
    result = compile_kernel(_make_kernel(source, entrypoint="run"), cache_dir=tmp_path)
    assert result.success is True
    assert result.compiled_fn(5) == 10


def test_compile_kernel_preserves_module_globals(tmp_path: Path):
    """Module-level helpers stay reachable from the entrypoint's closure."""
    source = "CONST = 7\ndef helper(x):\n    return x + CONST\ndef kernel_fn(x):\n    return helper(x)\n"
    result = compile_kernel(_make_kernel(source), cache_dir=tmp_path)
    assert result.success is True
    assert result.compiled_fn(3) == 10


# ── failure modes ──────────────────────────────────────────────────────────


def test_compile_kernel_returns_error_on_syntax_error(tmp_path: Path):
    source = "def kernel_fn(:\n    return 1\n"
    result = compile_kernel(_make_kernel(source), cache_dir=tmp_path)
    assert result.success is False
    assert result.compiled_fn is None
    assert "SyntaxError" in result.error_message


def test_compile_kernel_returns_error_on_import_error(tmp_path: Path):
    source = "import nonexistent_module_xyz\n\ndef kernel_fn(): pass\n"
    result = compile_kernel(_make_kernel(source), cache_dir=tmp_path)
    assert result.success is False
    assert result.compiled_fn is None
    assert "nonexistent_module_xyz" in result.error_message


def test_compile_kernel_error_traceback_references_source_file(tmp_path: Path):
    """Errors carry the on-disk path, not <string>, so the Coder can locate them."""
    source = "raise ValueError('boom at module load')\n"
    result = compile_kernel(_make_kernel(source), cache_dir=tmp_path)
    assert result.success is False
    assert result.source_path is not None
    assert str(result.source_path) in result.error_message


def test_compile_kernel_returns_error_on_missing_entrypoint(tmp_path: Path):
    source = "def some_other_name(): pass\n"
    result = compile_kernel(_make_kernel(source), cache_dir=tmp_path)
    assert result.success is False
    assert result.compiled_fn is None
    assert "kernel_fn" in result.error_message


def test_compile_kernel_returns_error_when_entrypoint_not_callable(tmp_path: Path):
    source = "kernel_fn = 42\n"
    result = compile_kernel(_make_kernel(source), cache_dir=tmp_path)
    assert result.success is False
    assert "callable" in result.error_message.lower()


# ── isolation & caching ────────────────────────────────────────────────────


def test_compile_kernel_two_versions_do_not_collide(tmp_path: Path):
    r1 = compile_kernel(_make_kernel("def kernel_fn(): return 1\n"), cache_dir=tmp_path)
    r2 = compile_kernel(_make_kernel("def kernel_fn(): return 2\n"), cache_dir=tmp_path)
    assert r1.success and r2.success
    assert r1.compiled_fn() == 1
    assert r2.compiled_fn() == 2
    assert r1.source_path != r2.source_path


def test_compile_kernel_identical_source_writes_same_path(tmp_path: Path):
    """Same source → same cache path (hash-keyed)."""
    source = "def kernel_fn(): return 7\n"
    r1 = compile_kernel(_make_kernel(source), cache_dir=tmp_path)
    r2 = compile_kernel(_make_kernel(source), cache_dir=tmp_path)
    assert r1.source_path == r2.source_path
    assert r1.success and r2.success
