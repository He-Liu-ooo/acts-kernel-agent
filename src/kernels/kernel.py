"""Kernel abstraction — code + metadata for a single kernel version."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class KernelType(Enum):
    """Known kernel archetypes for memory retrieval matching.

    Core types cover common kernel patterns.  SOL-ExecBench op_type values
    (gemm, rmsnorm, gqa, moe, …) are mapped to these via
    ``_OP_TYPE_TO_KERNEL_TYPE`` / ``_definition_to_kernel_spec`` in
    ``src.pipeline.optimize``.
    """

    MATMUL = "matmul"
    GEMM = "gemm"
    SOFTMAX = "softmax"
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    ATTENTION = "attention"
    GQA = "gqa"
    MOE = "moe"
    EMBEDDING = "embedding"
    LINEAR = "linear"
    FUSED_BLOCK = "fused_block"
    MLP = "mlp"
    CONV = "conv"
    SSM = "ssm"
    REDUCTION = "reduction"
    ELEMENTWISE = "elementwise"
    CUSTOM = "custom"


@dataclass
class KernelSpec:
    """Static metadata about a kernel problem (does not change across versions).

    For SOL-ExecBench problems, ``definition_path`` points to the source
    ``definition.json``, ``pytorch_reference`` holds the PyTorch ``run()``
    source that serves as the correctness oracle, and ``t_sol_us`` is the
    SOLAR-derived hardware bound (populated at problem-load time).
    """

    name: str
    kernel_type: KernelType
    # Computational profile for roofline (may be 0 when SOLAR provides T_SOL)
    flop_count: int = 0
    memory_bytes: int = 0
    # Reference input shapes for correctness testing
    input_shapes: list[dict] = field(default_factory=list)
    # SOL-ExecBench integration
    definition_path: Path | None = None
    pytorch_reference: str = ""
    t_sol_us: float | None = None
    # Name of the callable the compiler should resolve from the loaded module.
    # Matches AutoKernel's convention; overridable for fused ops where the
    # launchable symbol is a host wrapper around one or more @triton.jit fns.
    entrypoint: str = "kernel_fn"

    @classmethod
    def from_dict(cls, data: dict) -> "KernelSpec":
        """Reconstruct a KernelSpec from a checkpoint dict. Shared by
        ``Kernel.from_legacy_dict`` and ``tree._deserialize_node`` so
        spec deserialization lives in one place.
        """
        def_path = Path(data["definition_path"]) if data["definition_path"] else None
        return cls(
            name=data["name"],
            kernel_type=KernelType(data["kernel_type"]),
            flop_count=data["flop_count"],
            memory_bytes=data["memory_bytes"],
            input_shapes=data["input_shapes"],
            definition_path=def_path,
            pytorch_reference=data["pytorch_reference"],
            t_sol_us=data["t_sol_us"],
        )


@dataclass
class Kernel:
    """A single kernel version: source code + metadata.

    Autotune fields (A1 PR 1):
      - ``autotune_configs``: parsed at construction time from the
        ``@triton.autotune(configs=[...])`` decorator in ``source_code``.
        Each entry: ``{"kwargs": dict, "num_warps": int, "num_stages": int}``.
        Empty list when the source has no @triton.autotune (starters,
        legacy kernels, test fixtures). The validator in
        ``src.agents.coder.KernelCodeOutput`` enforces presence for
        Coder-emitted source; Kernel construction itself is lenient.
      - ``autotune_keys``: parsed from the same decorator's ``key=[...]``
        arg in the same AST pass. Retained as metadata and validator
        surface; winner attribution reads Triton's cache deltas directly
        instead of resolving these names against SOL axes.
      - ``autotune_winner``: populated post-bench by the orchestrator
        from ``BenchmarkResult.autotune_winner_per_workload``, keyed by
        ``workload.uuid``. Empty dict until the first benchmark with
        successful winner attribution.
    """

    spec: KernelSpec
    source_code: str
    # Bare name of the ``@triton.jit`` device function the profiler should
    # filter NCU on. Declared by the Coder via ``KernelCodeOutput`` so the
    # source-of-truth lives with the kernel that owns it; empty for hand-
    # written starters / test fixtures, where ``profile_kernel`` falls
    # back to source-regex extraction.
    triton_kernel_name: str = ""
    # Destination-passing-style flag. When True the host wrapper signature
    # accepts pre-allocated output buffers as positional args after the
    # inputs (e.g., ``def kernel_fn(x, y, out)``); the benchmark loop
    # allocates outputs via ``allocate_outputs`` and threads them through.
    # When False the kernel returns its outputs as the function's return
    # value. Default False for back-compat with hand-written starters and
    # checkpoint round-trips that pre-date this field.
    dps: bool = False
    autotune_configs: list[dict] = field(default_factory=list)
    autotune_keys: list[str] = field(default_factory=list)
    # Per-workload Triton-cache winner (Codex /simplify Q10): always a
    # dict — empty until the first benchmark with successful per-workload
    # winner attribution. Empty-dict semantics avoid the previous
    # ``None``-vs-``{}`` ambiguity that compressed three states (never
    # benched / benched-no-winner / benched-with-winners) into two.
    autotune_winner: dict[str, dict] = field(default_factory=dict)
    # Cached decorator line span (1-indexed inclusive) so
    # ``render_condensed_source`` doesn't re-parse on every call.
    # Populated alongside autotune_configs/keys by ``__post_init__``.
    _autotune_span: tuple[int, int] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Populate autotune_configs / autotune_keys / _autotune_span from
        # source unless the caller already supplied configs+keys (legacy
        # deserialization, explicit test fixtures). ``triton_kernel_name``
        # is threaded into the parser so fused sources with autotuned
        # helpers don't mis-attribute (Codex adversarial-review finding
        # #2, 2026-05-14).
        if not self.autotune_configs and not self.autotune_keys:
            call = _find_autotune_decorator_call(
                self.source_code, self.triton_kernel_name
            )
            if call is not None:
                cfgs, keys = _extract_autotune_kwargs(call)
                self.autotune_configs = cfgs
                self.autotune_keys = keys
                self._autotune_span = (
                    call.lineno,
                    getattr(call, "end_lineno", None) or call.lineno,
                )

    @classmethod
    def from_legacy_dict(cls, data: dict) -> "Kernel":
        """Reconstruct a Kernel from an old-format checkpoint dict.

        Pre-A1 checkpoints carried ``num_warps`` / ``num_stages`` /
        ``block_size`` as Kernel fields. Wrap that triple into a
        single-entry ``autotune_configs`` list so post-A1 code can read
        them uniformly. Keys not recoverable from the legacy schema →
        empty list. Winner stays empty dict (was never persisted on
        legacy nodes).
        """
        spec = KernelSpec.from_dict(data["spec"])
        # Legacy ``block_size`` was loosely typed — usually a dict (e.g.
        # ``{"BLOCK_M": 128}``) but some early checkpoints stored a bare
        # int (e.g. ``128``). The original code path accepted both because
        # the field was never consumed by name; only round-tripped. We
        # preserve dicts verbatim and drop non-dict scalars (their bare
        # values weren't load-bearing — no code branched on them).
        raw_block = data.get("block_size")
        block_kwargs = dict(raw_block) if isinstance(raw_block, dict) else {}
        legacy_config = {
            "kwargs": block_kwargs,
            "num_warps": int(data.get("num_warps", 4)),
            "num_stages": int(data.get("num_stages", 2)),
        }
        return cls(
            spec=spec,
            source_code=data["source_code"],
            triton_kernel_name=data.get("triton_kernel_name", ""),
            dps=data.get("dps", False),
            autotune_configs=[legacy_config],
            autotune_keys=[],
        )

    def render_condensed_source(
        self,
        representative_workload_uuid: str | None = None,
    ) -> str:
        """Return ``source_code`` with the ``@triton.autotune`` decorator
        replaced by a single-line ``# autotune: ...`` comment summarizing
        swept ranges, plus an optional second ``# winner ...`` comment line
        carrying the representative workload's winning config when
        ``autotune_winner`` has an entry for *representative_workload_uuid*.

        Falls back to verbatim ``source_code`` (the entire return) when:
          - source has no ``@triton.autotune`` decorator;
          - AST parse fails;
          - ``autotune_configs`` is empty (decorator was syntactically
            malformed at construction time);
          - the decorator's source-line span can't be located.

        Used by the orchestrator (``src/search/orchestrator.py``) to render
        condensed parent source for the Planner and Reviewer prompts.
        The Coder uses ``source_code`` verbatim because it must edit the
        decorator block (e.g. the reframed ``t1_block_size_tuning`` action
        widens the autotune sweep).
        """
        if not self.autotune_configs:
            return self.source_code
        if self._autotune_span is None:
            return self.source_code

        start, end = self._autotune_span
        lines = self.source_code.splitlines(keepends=True)
        # Convert 1-indexed inclusive lines to 0-indexed slice indices.
        before = lines[: start - 1]
        after = lines[end:]

        summary_line = _render_autotune_summary(
            self.autotune_configs, self.autotune_keys
        )
        replacement = [summary_line + "\n"]

        if (
            representative_workload_uuid is not None
            and representative_workload_uuid in self.autotune_winner
        ):
            winner_line = _render_autotune_winner(
                self.autotune_winner[representative_workload_uuid]
            )
            replacement.append(winner_line + "\n")

        return "".join(before + replacement + after)


def _parse_autotune_from_source(
    source: str,
    triton_kernel_name: str = "",
) -> tuple[list[dict], list[str]]:
    """Extract @triton.autotune(configs=[...], key=[...]) via stdlib ast.

    When *triton_kernel_name* is non-empty, only the autotune decorator
    attached to the ``FunctionDef`` with that exact name is considered.
    This is the correct behavior for Coder-emitted source where fused
    kernels can have an autotuned helper preceding the primary kernel —
    without target-awareness, the parser would silently attribute the
    helper's configs/keys to the named primary (Codex adversarial-review
    finding #2, 2026-05-14).

    Empty *triton_kernel_name* preserves legacy "first @autotune
    anywhere" behavior for hand-written starters and test fixtures that
    typically have only one ``@triton.jit def`` total.

    Returns ``(configs, keys)``. Both empty when no decorator present.
    Non-literal expressions (e.g. ``configs=some_var``) yield empty for
    that arg but the other arg still parses. No Triton import — works
    in the torchless Tier-1 test venv.
    """
    call = _find_autotune_decorator_call(source, triton_kernel_name)
    if call is None:
        return [], []
    return _extract_autotune_kwargs(call)


def _find_autotune_decorator_call(
    source: str,
    triton_kernel_name: str = "",
) -> ast.Call | None:
    """Locate the ``@triton.autotune(...)`` ``ast.Call`` node above the
    named ``@triton.jit def`` (or anywhere when *triton_kernel_name* is
    empty). Shared by ``_parse_autotune_from_source`` (extracts kwargs)
    and ``_find_autotune_decorator_span`` (extracts line range), so both
    return the same target deterministically.

    Returns ``None`` on SyntaxError, missing FunctionDef, or absent
    decorator. Stdlib ``ast`` only.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if triton_kernel_name and node.name != triton_kernel_name:
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            attr = ""
            if isinstance(dec.func, ast.Attribute):
                attr = dec.func.attr
            elif isinstance(dec.func, ast.Name):
                attr = dec.func.id
            if attr == "autotune":
                return dec
        if triton_kernel_name:
            return None  # named function found, no autotune on it
    return None


def _extract_autotune_kwargs(call: ast.Call) -> tuple[list[dict], list[str]]:
    """Pull configs= + key= out of an @triton.autotune call node."""
    configs: list[dict] = []
    keys: list[str] = []
    for kw in call.keywords:
        if kw.arg == "configs" and isinstance(kw.value, ast.List):
            for item in kw.value.elts:
                cfg = _parse_triton_config_call(item)
                if cfg is not None:
                    configs.append(cfg)
        elif kw.arg == "key" and isinstance(kw.value, ast.List):
            for item in kw.value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    keys.append(item.value)
    return configs, keys


def _parse_triton_config_call(node: ast.expr) -> dict | None:
    """Parse ``triton.Config({"BLOCK_M": 64, ...}, num_warps=4, num_stages=2)``.

    Returns ``None`` when the node isn't a Config call we can statically
    parse (no kwargs dict literal, non-Call elements like ``*spread``).
    """
    if not isinstance(node, ast.Call):
        return None
    kwargs: dict[str, Any] = {}
    if node.args and isinstance(node.args[0], ast.Dict):
        d = node.args[0]
        for key_node, val_node in zip(d.keys, d.values):
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                if isinstance(val_node, ast.Constant):
                    kwargs[key_node.value] = val_node.value
    num_warps = 4
    num_stages = 2
    for kw in node.keywords:
        if kw.arg == "num_warps" and isinstance(kw.value, ast.Constant):
            num_warps = int(kw.value.value)
        elif kw.arg == "num_stages" and isinstance(kw.value, ast.Constant):
            num_stages = int(kw.value.value)
    if not kwargs:
        return None
    return {"kwargs": kwargs, "num_warps": num_warps, "num_stages": num_stages}


def _find_autotune_decorator_span(
    source: str,
    triton_kernel_name: str,
) -> tuple[int, int] | None:
    """Return ``(start_lineno, end_lineno)`` (1-indexed, inclusive) of the
    ``@triton.autotune`` decorator immediately above
    ``def <triton_kernel_name>`` in *source*. ``None`` when the decorator
    is absent or parse fails. Used by ``Kernel.render_condensed_source``.
    """
    call = _find_autotune_decorator_call(source, triton_kernel_name)
    if call is None:
        return None
    return (call.lineno, getattr(call, "end_lineno", None) or call.lineno)


def _fmt_axis(name: str, values: list) -> str:
    """Render one axis's swept range: ``NAME=v`` for single-value sweeps,
    ``NAME ∈ {v1,v2,...}`` for multi-value. Used by
    ``_render_autotune_summary`` for kwarg axes plus num_warps / num_stages.
    """
    uniq = sorted(set(values))
    if len(uniq) == 1:
        return f"{name}={uniq[0]}"
    return f"{name} ∈ {{{','.join(str(v) for v in uniq)}}}"


def _render_autotune_summary(
    configs: list[dict],
    keys: list[str],
) -> str:
    """Build the ``# autotune: ...`` summary line from parsed configs + keys.

    Union per-axis: for each kwarg name, list the sorted distinct values
    across all configs. num_warps and num_stages follow the same rule.
    The ``key=[...]`` list mirrors the source's ``@triton.autotune.key=`` arg.
    """
    kwarg_values: dict[str, list] = {}
    for cfg in configs:
        for k, v in cfg.get("kwargs", {}).items():
            kwarg_values.setdefault(k, []).append(v)
    parts = [_fmt_axis(k, vs) for k, vs in kwarg_values.items()]
    parts.append(_fmt_axis("num_warps", [c.get("num_warps", 0) for c in configs]))
    parts.append(_fmt_axis("num_stages", [c.get("num_stages", 0) for c in configs]))
    parts.append(f"key=[{','.join(keys)}]")
    return "# autotune: " + ", ".join(parts)


def _flatten_autotune_config(cfg: dict) -> dict:
    """Lift the parser's nested ``{"kwargs": {...}, "num_warps": int,
    "num_stages": int}`` shape to a flat dict with ``num_warps`` /
    ``num_stages`` alongside the kwargs. Shared by ``_render_autotune_winner``
    and the Coder's ``autotune_exclude`` validator so both agree on what
    "flat" means."""
    return {
        **cfg.get("kwargs", {}),
        "num_warps": cfg.get("num_warps", 0),
        "num_stages": cfg.get("num_stages", 0),
    }


def _render_autotune_winner(winner: dict) -> str:
    """Build the ``# winner (representative wl): ...`` line from a single
    winner-config dict (kwargs + num_warps + num_stages)."""
    parts = [f"{k}={v}" for k, v in _flatten_autotune_config(winner).items()]
    return "# winner (representative wl): " + ", ".join(parts)
