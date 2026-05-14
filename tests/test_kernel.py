"""Tests for ``src/kernels/kernel.py`` — Kernel + KernelSpec dataclasses.

The legacy fields are exercised indirectly across the suite (compiler,
profiler, search). This file specifically pins the small contract bits
that need to round-trip: defaults, the ``dps`` flag for destination-
passing-style kernels, the Coder-output → Kernel propagation, and the
autotune fields parsed from source via AST (A1 PR 1).
"""

from __future__ import annotations

import textwrap

from src.agents.coder import KernelCodeOutput
from src.kernels.kernel import Kernel, KernelSpec, KernelType


def _spec() -> KernelSpec:
    return KernelSpec(name="k", kernel_type=KernelType.ELEMENTWISE)


def _custom_spec() -> KernelSpec:
    return KernelSpec(name="t", kernel_type=KernelType.CUSTOM)


# ── default field values ────────────────────────────────────────────────


def test_kernel_dps_defaults_to_false():
    """Hand-written starters and pre-DPS checkpoints must round-trip without
    setting ``dps``; the default has to stay False to preserve back-compat
    with every Kernel constructed before this field existed."""
    k = Kernel(spec=_spec(), source_code="def kernel_fn(x): return x")
    assert k.dps is False


def test_kernel_dps_can_be_set_true():
    """When the Coder declares destination-passing-style, the flag flows
    to the Kernel verbatim — the benchmark loop branches on it."""
    k = Kernel(spec=_spec(), source_code="def kernel_fn(x, out): pass", dps=True)
    assert k.dps is True


# ── KernelCodeOutput → Kernel propagation ──────────────────────────────

# A1 PR 1: KernelCodeOutput now validates @triton.autotune presence + >=4
# configs + non-empty key=. These dps-flag tests want the model to accept,
# so they need an autotune-bearing source.
_VALID_AUTOTUNE_KERNEL_SRC = """\
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=4),
    ],
    key=["N"],
)
@triton.jit
def k(X, N, BLOCK): pass
"""


def test_kernel_code_output_dps_defaults_to_false():
    out = KernelCodeOutput(
        source_code=_VALID_AUTOTUNE_KERNEL_SRC,
        triton_kernel_name="k",
    )
    assert out.dps is False


def test_kernel_code_output_dps_round_trips_true():
    out = KernelCodeOutput(
        source_code=_VALID_AUTOTUNE_KERNEL_SRC,
        triton_kernel_name="k",
        dps=True,
    )
    assert out.dps is True


# ── A1 PR 1: autotune fields, AST parser, legacy migration ─────────────


def test_kernel_default_autotune_fields_empty():
    """A Kernel built from non-autotune source has empty autotune fields."""
    k = Kernel(spec=_custom_spec(), source_code="# placeholder")
    assert k.autotune_configs == []
    assert k.autotune_keys == []
    assert k.autotune_winner == {}


def test_kernel_parses_autotune_from_source():
    """AST parser extracts configs + keys from a well-formed @triton.autotune."""
    src = textwrap.dedent(
        '''
        import triton
        import triton.language as tl

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
                triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
                triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
                triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
            ],
            key=["M", "N"],
        )
        @triton.jit
        def my_kernel(A, B, C, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
            pass
        '''
    )
    k = Kernel(spec=_custom_spec(), source_code=src, triton_kernel_name="my_kernel")
    assert len(k.autotune_configs) == 4
    assert k.autotune_configs[0] == {
        "kwargs": {"BLOCK_M": 64, "BLOCK_N": 64},
        "num_warps": 4,
        "num_stages": 2,
    }
    assert k.autotune_keys == ["M", "N"]


def test_kernel_from_legacy_dict_wraps_single_config():
    """Legacy checkpoint dict (with num_warps/num_stages/block_size) → single-entry autotune list."""
    legacy = {
        "spec": {
            "name": "legacy", "kernel_type": "matmul",
            "flop_count": 0, "memory_bytes": 0, "input_shapes": [],
            "definition_path": None, "pytorch_reference": "", "t_sol_us": None,
        },
        "source_code": "# old kernel",
        "num_warps": 8,
        "num_stages": 3,
        "block_size": {"BLOCK_M": 128, "BLOCK_N": 64},
        "triton_kernel_name": "",
        "dps": False,
    }
    k = Kernel.from_legacy_dict(legacy)
    assert k.autotune_configs == [{
        "kwargs": {"BLOCK_M": 128, "BLOCK_N": 64},
        "num_warps": 8,
        "num_stages": 3,
    }]
    assert k.autotune_keys == []
    assert k.autotune_winner == {}


def test_kernel_parser_no_autotune_silent():
    """Source without @triton.autotune leaves autotune_configs empty.

    Kernel construction stays lenient — the validator in
    src.agents.coder.KernelCodeOutput enforces presence for Coder-emitted
    source; hand-written starters and test fixtures don't autotune.
    """
    k = Kernel(spec=_custom_spec(), source_code="def kernel_fn(x): return x")
    assert k.autotune_configs == []
    assert k.autotune_keys == []


def test_kernel_parser_malformed_autotune_silent():
    """Source with non-literal configs= parses what it can without raising."""
    src = textwrap.dedent(
        '''
        @triton.autotune(configs=some_dynamic_list, key=["M"])
        @triton.jit
        def k(x): pass
        '''
    )
    k = Kernel(spec=_custom_spec(), source_code=src)
    # configs= isn't a list literal → empty.
    assert k.autotune_configs == []
    # key= IS a literal list of strings → parsed.
    assert k.autotune_keys == ["M"]


# ── A3: _find_autotune_decorator_span helper ───────────────────────────


def test_find_autotune_decorator_span_basic():
    """Returns (start_lineno, end_lineno) bracketing the @triton.autotune
    decorator immediately above the named @triton.jit def."""
    from src.kernels.kernel import _find_autotune_decorator_span

    src = textwrap.dedent(
        '''
        import triton
        import triton.language as tl

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
                triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
                triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
                triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=4),
            ],
            key=["M"],
        )
        @triton.jit
        def my_kernel(X, M, BLOCK_M: tl.constexpr): pass
        '''
    ).lstrip()

    span = _find_autotune_decorator_span(src, "my_kernel")
    assert span is not None
    start, end = span
    # 1-indexed inclusive lines: the @triton.autotune decorator opens on
    # the line with "@triton.autotune(" and closes on the ")" line.
    lines = src.splitlines()
    decorator_text = "\n".join(lines[start - 1 : end])
    assert decorator_text.startswith("@triton.autotune(")
    assert decorator_text.rstrip().endswith(")")


def test_find_autotune_decorator_span_returns_none_for_missing_decorator():
    """Bare @triton.jit def with no autotune wrapper → None."""
    from src.kernels.kernel import _find_autotune_decorator_span

    src = textwrap.dedent(
        '''
        @triton.jit
        def my_kernel(X): pass
        '''
    ).lstrip()

    assert _find_autotune_decorator_span(src, "my_kernel") is None


def test_find_autotune_decorator_span_returns_none_for_missing_target():
    """Source has the autotune+jit pair but on a different-named function."""
    from src.kernels.kernel import _find_autotune_decorator_span

    src = textwrap.dedent(
        '''
        @triton.autotune(
            configs=[triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2)],
            key=["N"],
        )
        @triton.jit
        def other_kernel(X, N, BLOCK): pass
        '''
    ).lstrip()

    assert _find_autotune_decorator_span(src, "my_kernel") is None


def test_find_autotune_decorator_span_returns_none_on_syntax_error():
    """Malformed Python returns None (degrades cleanly)."""
    from src.kernels.kernel import _find_autotune_decorator_span

    assert _find_autotune_decorator_span("not valid: python @@@", "x") is None


# ── A3: Kernel.render_condensed_source method ──────────────────────────


def _matmul_autotune_src(name: str = "my_kernel") -> str:
    """Reusable 4-config autotune source for render_condensed_source tests."""
    return textwrap.dedent(
        f'''
        import triton
        import triton.language as tl

        @triton.autotune(
            configs=[
                triton.Config({{"BLOCK_M": 64, "BLOCK_N": 64}}, num_warps=4, num_stages=2),
                triton.Config({{"BLOCK_M": 128, "BLOCK_N": 64}}, num_warps=4, num_stages=3),
                triton.Config({{"BLOCK_M": 128, "BLOCK_N": 128}}, num_warps=8, num_stages=3),
                triton.Config({{"BLOCK_M": 64, "BLOCK_N": 128}}, num_warps=4, num_stages=4),
            ],
            key=["M", "N"],
        )
        @triton.jit
        def {name}(A, B, C, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
            pass
        '''
    ).lstrip()


def test_render_condensed_source_basic():
    """Well-formed 4-config autotune, no winner data → one comment line
    replacing the decorator block. Union-per-axis form correct."""
    k = Kernel(
        spec=_custom_spec(),
        source_code=_matmul_autotune_src(),
        triton_kernel_name="my_kernel",
    )
    rendered = k.render_condensed_source()

    # The verbatim @triton.autotune block is gone.
    assert "@triton.autotune(" not in rendered
    assert "triton.Config(" not in rendered
    # The summary comment is present.
    assert "# autotune:" in rendered
    # Union-per-axis: BLOCK_M ∈ {64,128}, BLOCK_N ∈ {64,128},
    # num_warps ∈ {4,8}, num_stages ∈ {2,3,4}, key=[M,N].
    assert "BLOCK_M ∈ {64,128}" in rendered
    assert "BLOCK_N ∈ {64,128}" in rendered
    assert "num_warps ∈ {4,8}" in rendered
    assert "num_stages ∈ {2,3,4}" in rendered
    assert "key=[M,N]" in rendered
    # The kernel body is untouched.
    assert "@triton.jit" in rendered
    assert "def my_kernel(" in rendered


def test_render_condensed_source_with_winner():
    """When autotune_winner has an entry for the representative uuid,
    a second '# winner ...' comment line is appended."""
    k = Kernel(
        spec=_custom_spec(),
        source_code=_matmul_autotune_src(),
        triton_kernel_name="my_kernel",
    )
    k.autotune_winner = {
        "wl-1": {
            "kwargs": {"BLOCK_M": 128, "BLOCK_N": 64},
            "num_warps": 4,
            "num_stages": 3,
        },
    }

    rendered = k.render_condensed_source(representative_workload_uuid="wl-1")

    assert "# autotune:" in rendered
    assert "# winner" in rendered
    assert "BLOCK_M=128" in rendered
    assert "BLOCK_N=64" in rendered
    assert "num_warps=4" in rendered
    assert "num_stages=3" in rendered


def test_render_condensed_source_winner_uuid_missing():
    """When the representative uuid is not in autotune_winner, the summary
    line renders but the winner line is omitted."""
    k = Kernel(
        spec=_custom_spec(),
        source_code=_matmul_autotune_src(),
        triton_kernel_name="my_kernel",
    )
    k.autotune_winner = {"wl-other": {"kwargs": {}, "num_warps": 4, "num_stages": 3}}

    rendered = k.render_condensed_source(representative_workload_uuid="wl-1")
    assert "# autotune:" in rendered
    assert "# winner" not in rendered


def test_render_condensed_source_no_autotune():
    """Source without @triton.autotune → verbatim return."""
    src = "# placeholder kernel\n"
    k = Kernel(spec=_custom_spec(), source_code=src)
    assert k.render_condensed_source() == src


def test_render_condensed_source_malformed_decorator():
    """Source with non-literal configs= argument → empty autotune_configs
    at parse time → render falls back to verbatim source."""
    src = textwrap.dedent(
        '''
        @triton.autotune(configs=some_dynamic_list, key=["M"])
        @triton.jit
        def my_kernel(X): pass
        '''
    ).lstrip()
    k = Kernel(spec=_custom_spec(), source_code=src, triton_kernel_name="my_kernel")
    # autotune_configs is empty because the AST parser couldn't extract entries.
    assert k.autotune_configs == []
    assert k.render_condensed_source() == src


def test_render_condensed_source_helper_kernels():
    """Fused source with a bare @triton.jit def helper + the named kernel's
    autotune-bearing decorator: only the named kernel's decorator is
    condensed. Helper untouched."""
    src = textwrap.dedent(
        '''
        @triton.jit
        def _helper(x): pass

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
                triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
                triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
                triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=4),
            ],
            key=["M"],
        )
        @triton.jit
        def main_kernel(X, M, BLOCK_M: tl.constexpr): pass
        '''
    ).lstrip()
    k = Kernel(spec=_custom_spec(), source_code=src, triton_kernel_name="main_kernel")
    rendered = k.render_condensed_source()

    # Helper untouched.
    assert "def _helper(x): pass" in rendered
    # Main kernel's decorator condensed.
    assert "@triton.autotune(" not in rendered
    assert "# autotune:" in rendered
    assert "def main_kernel(" in rendered


# ── Codex adversarial review (2026-05-14, finding #2): target-aware AST ─


def test_parse_autotune_targets_named_kernel_with_autotuned_helper_first():
    """Codex adversarial-review finding (medium): when source carries a
    helper @triton.jit that's also wrapped in @triton.autotune AND the
    helper precedes the primary kernel, the parser must attribute the
    autotune metadata to the *named* primary kernel — not to the first
    decorator-bearing function it walks past.

    Pre-fix bug: ``_parse_autotune_from_source`` walks ``ast.walk(tree)``
    and returns the FIRST ``@triton.autotune`` regardless of which
    FunctionDef it sits on. ``Kernel.__post_init__`` passes no target
    name, so ``Kernel(triton_kernel_name='main_kernel', ...)`` would
    inherit the helper's configs/keys silently, corrupting both
    ``_record_autotune_winner`` cache lookups and A3's condensed
    rendering.
    """
    src = textwrap.dedent(
        '''
        @triton.autotune(
            configs=[
                triton.Config({"HELPER_BLOCK": 16}, num_warps=1, num_stages=1),
                triton.Config({"HELPER_BLOCK": 32}, num_warps=1, num_stages=1),
                triton.Config({"HELPER_BLOCK": 64}, num_warps=2, num_stages=2),
                triton.Config({"HELPER_BLOCK": 128}, num_warps=2, num_stages=2),
            ],
            key=["HK"],
        )
        @triton.jit
        def _helper(X, HK, HELPER_BLOCK: tl.constexpr): pass

        @triton.autotune(
            configs=[
                triton.Config({"MAIN_BLOCK_M": 64}, num_warps=4, num_stages=2),
                triton.Config({"MAIN_BLOCK_M": 128}, num_warps=4, num_stages=3),
                triton.Config({"MAIN_BLOCK_M": 64}, num_warps=8, num_stages=3),
                triton.Config({"MAIN_BLOCK_M": 128}, num_warps=8, num_stages=4),
            ],
            key=["M", "N"],
        )
        @triton.jit
        def main_kernel(A, M, N, MAIN_BLOCK_M: tl.constexpr): pass
        '''
    ).lstrip()
    k = Kernel(spec=_custom_spec(), source_code=src, triton_kernel_name="main_kernel")

    # Kernel.autotune_keys should be MAIN_kernel's keys, not the helper's.
    assert k.autotune_keys == ["M", "N"]
    # All recorded configs must come from main_kernel — i.e. carry the
    # ``MAIN_BLOCK_M`` kwarg and NOT the helper's ``HELPER_BLOCK`` kwarg.
    for cfg in k.autotune_configs:
        kwargs = cfg.get("kwargs", {})
        assert "MAIN_BLOCK_M" in kwargs, f"config came from helper: {cfg}"
        assert "HELPER_BLOCK" not in kwargs, f"config came from helper: {cfg}"
