"""Hybrid kernel profiler: analytical roofline + curated NCU subprocess.

Called by the orchestrator once per iteration after the Coder returns a
compiled, correct kernel. The analytical path is always required and
fail-closed; the NCU path is best-effort and degrades without killing
the branch.
"""

from __future__ import annotations

import csv
import getpass
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.config import HardwareSpec
from src.kernels.compiler import compile_kernel

if TYPE_CHECKING:
    from src.kernels.kernel import Kernel

# Cache-bust token. Bump when the curated metric map, stall reasons,
# parser contract, or *cache-key shape* changes. Embedded in every
# cache key so stale entries are unreachable from the new version and
# naturally evicted. ``v2`` (2026-04-22): added ``kernel_name`` to the
# key blob so two Kernels with identical source but different declared
# ``triton_kernel_name`` values can no longer alias to one entry (Codex
# P2 fix — was returning helper-kernel metrics when the run requested
# the dominant-kernel filter).
_METRIC_SET_VERSION: str = "v2"

# ``_UNSET`` sentinel distinguishes "not yet probed" from "probed → missing".
_UNSET: Any = object()
_NCU_BINARY_CACHE: Any = _UNSET

# Required metrics must appear in the CSV (parse degrades otherwise);
# optional metrics default to 0.0 when absent. Names are raw
# (``--print-metric-name=name``) since ``label`` varies across NCU releases.
_CURATED_REQUIRED = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "sm_occupancy_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate_pct",
}
_CURATED_OPTIONAL = {
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor_core_util_pct",
}

_STALL_PREFIX = "smsp__average_warp_latency_issue_stalled_"
_STALL_SUFFIX = ".pct"

# Wildcards (``_*.pct``) do NOT expand on ``ncu --metrics`` — every
# stall reason must be enumerated explicitly.
_STALL_REASONS = (
    "barrier",
    "branch_resolving",
    "dispatch_stall",
    "drain",
    "imc_miss",
    "lg_throttle",
    "long_scoreboard",
    "math_pipe_throttle",
    "membar",
    "mio_throttle",
    "misc",
    "no_instruction",
    "not_selected",
    "selected",
    "short_scoreboard",
    "sleeping",
    "tex_throttle",
    "wait",
)

# The four curated NCU sections (spec §4). Stall metrics are NOT in any
# section — they're requested via explicit ``--metrics``.
_CURATED_SECTIONS = (
    "Occupancy",
    "WarpStateStats",
    "MemoryWorkloadAnalysis",
    "ComputeWorkloadAnalysis",
)

_CSV_KERNEL_COL = "Kernel Name"
_CSV_METRIC_COL = "Metric Name"
_CSV_VALUE_COL = "Metric Value"

# Matches ``@triton.jit`` (optionally with decorator args) followed by
# ``def <name>``. Used by ``profile_kernel`` to discover the GPU-symbol
# name that NCU's ``--kernel-name regex:`` filter and the CSV parser's
# substring match both need. The host-wrapper ``spec.entrypoint`` is a
# Python function, not the mangled Triton kernel symbol.
_TRITON_JIT_DEF_RE = re.compile(
    r"@triton\.jit\s*(?:\([^)]*\))?\s*\n\s*def\s+(\w+)",
    re.DOTALL,
)


def _extract_triton_kernel_name(source: str) -> str | None:
    """First ``@triton.jit def <name>`` in ``source``, or ``None`` if
    there is no Triton-JIT'd kernel."""
    match = _TRITON_JIT_DEF_RE.search(source)
    return match.group(1) if match else None


def triton_kernel_names_in(source: str) -> list[str]:
    """All ``@triton.jit def <name>`` matches in source order.

    Public so the Coder's ``KernelCodeOutput`` validator can cross-check
    a declared ``triton_kernel_name`` against the actual jit'd functions
    in the emitted source. Returns an empty list when the source contains
    no Triton-JIT'd kernel — caller decides whether that's a failure
    (Coder validation) or a fallback signal (profiler regex extraction).
    """
    return _TRITON_JIT_DEF_RE.findall(source)


class ProfilerError(Exception):
    """Raised when the analytical path cannot produce a classification.

    Branch-killing. NCU failures do NOT raise — they degrade the result.
    """


@dataclass(frozen=True)
class AnalyticalMetrics:
    """Zero-overhead roofline-derived metrics.

    Pure runtime measurements — classification lives at the run level
    (``classify_run`` in ``src/eval/roofline.py``) since it's invariant
    per ``(problem, representative_workload, hardware)``.
    """

    arithmetic_intensity: float
    ridge_point: float
    achieved_tflops: float
    achieved_bandwidth_gb_s: float
    pct_peak_compute: float
    pct_peak_bandwidth: float


@dataclass(frozen=True)
class NCUMetrics:
    """Curated subset of NCU's metric output. Populated only when
    ``_run_ncu`` succeeds; ``ProfilingResult.ncu`` is ``None`` otherwise."""

    sm_occupancy_pct: float
    l2_hit_rate_pct: float
    tensor_core_util_pct: float
    warp_stall_dominant: str
    warp_stall_dominant_pct: float
    warp_stall_runner_up: str
    warp_stall_runner_up_pct: float


@dataclass(frozen=True)
class ProfilingResult:
    """Merged analytical + NCU view of one iteration's kernel."""

    analytical: AnalyticalMetrics
    ncu: NCUMetrics | None = None
    raw_metrics: dict[str, float] = field(default_factory=dict)
    degraded_reason: str | None = None

    @property
    def degraded(self) -> bool:
        return self.degraded_reason is not None

    @property
    def has_ncu(self) -> bool:
        return self.ncu is not None


def _compute_analytical(
    *,
    flops: int,
    nbytes: int,
    latency_s: float,
    hardware_spec: HardwareSpec,
) -> AnalyticalMetrics:
    """Derive roofline runtime metrics from measured latency.

    Raises ``ProfilerError`` when inputs make analysis meaningless:
    non-positive latency / nbytes, or hardware with zero peak compute /
    bandwidth (the ``detect_hardware()`` zeroed-spec fallback).
    """
    if latency_s <= 0:
        raise ProfilerError(f"latency_s must be positive, got {latency_s}")
    if nbytes <= 0:
        raise ProfilerError(f"nbytes must be positive, got {nbytes}")
    if flops < 0:
        raise ProfilerError(f"flops must be non-negative, got {flops}")

    peak_tflops = hardware_spec.peak_flops_fp32
    peak_bw_gb_s = hardware_spec.peak_memory_bandwidth_gb_s
    if peak_tflops <= 0 or peak_bw_gb_s <= 0:
        raise ProfilerError(
            "hardware peaks are zero — profiler needs a populated HardwareSpec "
            "(load via SOLAR arch YAML or implement detect_hardware)"
        )

    arithmetic_intensity = flops / nbytes
    ridge_point = (peak_tflops * 1e12) / (peak_bw_gb_s * 1e9)
    achieved_tflops = flops / latency_s / 1e12
    achieved_bandwidth_gb_s = nbytes / latency_s / 1e9
    pct_peak_compute = achieved_tflops / peak_tflops
    pct_peak_bandwidth = achieved_bandwidth_gb_s / peak_bw_gb_s

    return AnalyticalMetrics(
        arithmetic_intensity=arithmetic_intensity,
        ridge_point=ridge_point,
        achieved_tflops=achieved_tflops,
        achieved_bandwidth_gb_s=achieved_bandwidth_gb_s,
        pct_peak_compute=pct_peak_compute,
        pct_peak_bandwidth=pct_peak_bandwidth,
    )


def _parse_ncu_csv(
    stdout: str,
    entrypoint: str,
) -> tuple[NCUMetrics | None, dict[str, float], bool, str | None]:
    """Reduce ``ncu --csv --print-metric-name=name`` stdout to NCUMetrics.

    Returns ``(ncu, raw_metrics, degraded, reason)``:
      * On success — ``(NCUMetrics, {<name>: <value>, ...}, False, None)``.
      * On NCU-side failure — ``(None, raw, True, <reason_slug>)``. ``raw``
        may still be populated when the failure is "missing curated
        metric", preserving the escape-hatch surface for prompt engineers.
      * On CSV-level failure — ``(None, {}, True, "csv_parse:<kind>")``.

    Parser contract:
      * Skips ``==PROF==`` lines and any non-CSV noise the subprocess
        interleaves.
      * ``entrypoint`` is matched as a case-sensitive substring of the
        ``Kernel Name`` column (Triton/torch JIT mangles names; full
        equality is too strict, regex is overkill).
      * Values like ``"5,000.00"`` are stripped of thousands-separators
        before ``float()``; ``"n/a"`` is skipped.
      * On duplicate metric rows (defensive — ``--launch-count 1``
        prevents this normally), first-write-wins.
    """
    lines = [ln for ln in stdout.splitlines() if ln and not ln.startswith("==") and not ln.startswith("//")]
    try:
        rows = list(csv.reader(io.StringIO("\n".join(lines))))
    except Exception as exc:  # noqa: BLE001 — any parser exception is degradation
        return None, {}, True, f"csv_parse:{type(exc).__name__}"

    header_idx = next(
        (i for i, r in enumerate(rows) if _CSV_KERNEL_COL in r and _CSV_METRIC_COL in r),
        None,
    )
    if header_idx is None:
        return None, {}, True, "csv_parse:no_header"

    header = rows[header_idx]
    try:
        k_idx = header.index(_CSV_KERNEL_COL)
        m_idx = header.index(_CSV_METRIC_COL)
        v_idx = header.index(_CSV_VALUE_COL)
    except ValueError as exc:
        return None, {}, True, f"csv_parse:missing_column:{exc}"

    raw: dict[str, float] = {}
    max_idx = max(k_idx, m_idx, v_idx)
    for row in rows[header_idx + 1 :]:
        if len(row) <= max_idx:
            continue
        kernel_name = row[k_idx]
        if entrypoint not in kernel_name:
            continue
        metric = row[m_idx]
        value_str = row[v_idx]
        if not metric or value_str.strip().lower() in ("", "n/a"):
            continue
        try:
            value = float(value_str.replace(",", ""))
        except ValueError:
            continue
        raw.setdefault(metric, value)  # first-write-wins

    if not raw:
        return None, {}, True, "no_matching_kernel"

    curated_fields: dict[str, float] = {}
    for raw_name, field_name in _CURATED_REQUIRED.items():
        if raw_name not in raw:
            return None, raw, True, f"missing_metric:{raw_name}"
        curated_fields[field_name] = raw[raw_name]
    for raw_name, field_name in _CURATED_OPTIONAL.items():
        curated_fields[field_name] = raw.get(raw_name, 0.0)

    stalls = sorted(
        (
            (name[len(_STALL_PREFIX) : -len(_STALL_SUFFIX)], value)
            for name, value in raw.items()
            if name.startswith(_STALL_PREFIX) and name.endswith(_STALL_SUFFIX)
        ),
        # Sort by value desc, then reason asc for deterministic ties.
        key=lambda kv: (-kv[1], kv[0]),
    )
    if len(stalls) < 2:
        return None, raw, True, "stalls_incomplete"

    ncu = NCUMetrics(
        sm_occupancy_pct=curated_fields["sm_occupancy_pct"],
        l2_hit_rate_pct=curated_fields["l2_hit_rate_pct"],
        tensor_core_util_pct=curated_fields["tensor_core_util_pct"],
        warp_stall_dominant=stalls[0][0],
        warp_stall_dominant_pct=stalls[0][1],
        warp_stall_runner_up=stalls[1][0],
        warp_stall_runner_up_pct=stalls[1][1],
    )
    return ncu, raw, False, None


def _ncu_tmpdir() -> str:
    """User-scoped TMPDIR so ``/tmp/nsight-compute-lock`` with sticky-bit
    ownership by another user doesn't block the subprocess."""
    path = Path(tempfile.gettempdir()) / f"{getpass.getuser()}_ncu"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _build_ncu_argv(
    kernel: "Kernel",
    spec_json_path: Path,
    *,
    mode: str,
    kernel_name: str | None = None,
) -> list[str]:
    """Build the ``ncu`` command line.

    ``kernel_name`` is the GPU symbol (extracted from ``@triton.jit def``)
    NCU filters on. Falls back to ``kernel.spec.entrypoint`` when no
    Triton kernel is found in the source.
    """
    regex_name = kernel_name or kernel.spec.entrypoint
    argv: list[str] = [
        "ncu",
        "--csv",
        "--print-metric-name=name",
        "--target-processes",
        "application-only",
        "--replay-mode",
        "kernel",
        "--launch-skip-before-match",
        "0",
        "--launch-count",
        "1",
        "--kernel-name",
        f"regex:{regex_name}",
    ]

    if mode == "full":
        # Debug escape hatch — captures everything NCU knows; the curated
        # metric set is still what the parser pulls out.
        argv += ["--set", "full"]
    else:
        for section in _CURATED_SECTIONS:
            argv += ["--section", section]
        # Stalls live outside every section — enumerate them explicitly.
        stall_metrics = ",".join(
            f"{_STALL_PREFIX}{reason}{_STALL_SUFFIX}" for reason in _STALL_REASONS
        )
        argv += ["--metrics", stall_metrics]

    argv += [
        "--",
        # ``sys.executable`` (not bare ``python``) — PATH's ``python`` is
        # often system ``/usr/bin/python`` without torch / triton.
        sys.executable,
        "-m",
        "src.eval._profiler_driver",
        str(spec_json_path),
    ]
    return argv


def _run_ncu(
    kernel: "Kernel",
    workload: dict,
    input_generator: Callable[..., Any],
    *,
    timeout_s: float,
    mode: str,
    kernel_source_path: Path | None = None,
    kernel_name: str | None = None,
    problem_definition_path: Path | None = None,
) -> tuple[str, int, bool, str | None]:
    """Invoke ``ncu`` as a subprocess around ``_profiler_driver``.

    Returns ``(stdout, returncode, degraded, reason)``:

    * ``degraded=False`` and ``reason=None`` when the subprocess exits
      cleanly. ``stdout`` is handed to ``_parse_ncu_csv`` by the caller —
      CSV-level degradation is the parser's job, not the driver's.
    * ``degraded=True`` with a ``ncu_*`` reason slug when the subprocess
      itself failed (binary missing, non-zero exit, or timeout).

    NCU failures never raise. Callers interpret the degraded result via
    the failure taxonomy in the spec §4.2.

    ``kernel_source_path`` is the compiled-kernel path the driver imports.
    ``kernel_name`` is the GPU symbol for NCU's ``--kernel-name regex:``;
    ``None`` falls back to ``kernel.spec.entrypoint``.
    ``problem_definition_path`` is the SOL-ExecBench ``definition.json``;
    its parent directory is serialized as ``problem_dir`` so the driver
    can call ``load_problem(<dir>)``. ``None`` omits the key and the
    driver falls back to ``module.make_inputs`` or ``spec['args']``.
    """
    if _discover_ncu_binary() is None:
        return "", -1, True, "ncu_binary_not_found"

    spec_payload: dict[str, Any] = {
        "kernel_source_path": str(kernel_source_path) if kernel_source_path else "",
        "entrypoint": kernel.spec.entrypoint,
        "workload": workload,
        "mode": mode,
    }
    if problem_definition_path is not None:
        # ``load_problem`` wants the directory (``definition.json`` +
        # sibling ``workload.jsonl``), not the definition file itself.
        spec_payload["problem_dir"] = str(Path(problem_definition_path).parent)
    # ``input_generator`` can't cross the subprocess boundary; the driver
    # rebuilds it from the serialized problem.
    _ = input_generator

    env = os.environ.copy()
    env["TMPDIR"] = _ncu_tmpdir()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=env["TMPDIR"]
    ) as f:
        json.dump(spec_payload, f)
        spec_json_path = Path(f.name)

    argv = _build_ncu_argv(kernel, spec_json_path, mode=mode, kernel_name=kernel_name)

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
    except FileNotFoundError:
        # Race: shutil.which found ncu but it vanished before exec.
        return "", -1, True, "ncu_binary_not_found"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        return stdout, -1, True, "ncu_timeout"
    finally:
        spec_json_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        return completed.stdout, completed.returncode, True, f"ncu_nonzero_exit:{completed.returncode}"

    return completed.stdout, 0, False, None


def _discover_ncu_binary() -> str | None:
    """Return the absolute path of ``ncu`` on ``$PATH``, or ``None`` if
    missing. Result is cached at module level so long-lived orchestrators
    pay ``shutil.which`` only once per process."""
    global _NCU_BINARY_CACHE
    if _NCU_BINARY_CACHE is _UNSET:
        _NCU_BINARY_CACHE = shutil.which("ncu")
    return _NCU_BINARY_CACHE


def _cache_key(
    kernel_source: str, workload: dict, mode: str, kernel_name: str
) -> str:
    """Deterministic 16-hex-char key mixing source hash + workload + mode
    + resolved ``kernel_name`` + ``_METRIC_SET_VERSION``. ``repr()`` of a
    dict is insertion-order-stable on Python 3.7+ — sufficient within one
    process. Including ``kernel_name`` keeps multi-jit fused outputs from
    aliasing — the resolved name (Coder-declared, regex fallback, or
    entrypoint last-ditch) is what NCU's ``--kernel-name regex:`` filter
    actually targets, so two runs with the same source but different
    targets must produce distinct cache entries."""
    source_hash = hashlib.sha256(kernel_source.encode("utf-8")).hexdigest()
    blob = (
        source_hash + repr(workload) + mode + kernel_name + _METRIC_SET_VERSION
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _load_ncu_cache(cache_dir: Path, key: str) -> NCUMetrics | None:
    """Read and rehydrate a cached ``NCUMetrics``. Returns ``None`` on any
    error (missing file, corrupt JSON, missing field, unknown field) — a
    corrupt cache entry is treated as a silent miss, not a crash."""
    try:
        payload = json.loads(_cache_path(cache_dir, key).read_text())
        return NCUMetrics(**payload["ncu"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_ncu_cache(
    cache_dir: Path, key: str, ncu: NCUMetrics, raw: dict[str, float]
) -> None:
    """Persist ``ncu`` + ``raw`` atomically: write to a unique temp file
    in ``cache_dir``, then ``os.replace`` onto the final path. The temp
    file is cleaned up if rename fails, so a failed write leaves no
    partial ``<key>.json`` behind. Any OSError is swallowed — caching is
    best-effort, never branch-killing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = _cache_path(cache_dir, key)
    payload = {"ncu": dict(ncu.__dict__), "raw": dict(raw)}

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{key}.", suffix=".json.tmp", dir=str(cache_dir)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(payload, f)
        os.replace(str(tmp_path), str(final))
    except OSError:
        tmp_path.unlink(missing_ok=True)


def profile_kernel(
    kernel: "Kernel",
    workload: dict,
    input_generator: Callable[..., Any],
    *,
    hardware_spec: HardwareSpec,
    flops: int,
    nbytes: int,
    latency_s: float,
    mode: str = "curated",
    timeout_s: float = 60.0,
    cache_dir: Path | None = None,
    problem_definition_path: Path | None = None,
) -> ProfilingResult:
    """Hybrid analytical + NCU profile (spec §3.2).

    Flow:

    1. Always compute ``_compute_analytical``. Raises ``ProfilerError`` on
       impossible inputs — the branch dies.
    2. If ``cache_dir`` is given and a cached NCUMetrics exists under the
       (source-hash, workload, mode, metric-set-version) key, rehydrate
       it and skip the subprocess.
    3. Otherwise: discover ``ncu`` on PATH. Missing → degraded result
       with no cache write; branch survives.
    4. Run the NCU subprocess; driver-side failure → degraded, no cache.
    5. Parse CSV; parser-side failure → degraded, no cache.
    6. Both green → persist NCUMetrics, return full ProfilingResult.

    NCU failures never raise. Analytical failures always raise.
    """
    analytical = _compute_analytical(
        flops=flops,
        nbytes=nbytes,
        latency_s=latency_s,
        hardware_spec=hardware_spec,
    )

    # Resolve the NCU-target kernel name BEFORE the cache check so it
    # participates in the cache key — otherwise two Kernels with identical
    # source but different declared ``triton_kernel_name`` values alias
    # to one cache entry (Codex P2 fix). Priority: Coder-declared name
    # (validated upstream) → source-regex fallback (hand-written starters
    # / test fixtures with an empty declared name) → entrypoint last-ditch
    # (so we degrade to ``no_matching_kernel`` rather than crash when
    # neither source has a ``@triton.jit`` def at all). Pure-Python
    # resolution; no I/O cost added to the cache-hit path.
    kernel_name = (
        kernel.triton_kernel_name
        or _extract_triton_kernel_name(kernel.source_code)
        or kernel.spec.entrypoint
    )

    if cache_dir is not None:
        key = _cache_key(kernel.source_code, workload, mode, kernel_name)
        cached = _load_ncu_cache(cache_dir, key)
        if cached is not None:
            # Cache stores only the NCU piece — ``raw_metrics`` is
            # populated only on the freshly-parsed path.
            return ProfilingResult(
                analytical=analytical,
                ncu=cached,
                raw_metrics={},
            )
    else:
        key = None

    if _discover_ncu_binary() is None:
        return ProfilingResult(
            analytical=analytical,
            ncu=None,
            raw_metrics={},
            degraded_reason="ncu_binary_not_found",
        )

    # Materialise the kernel on disk so the subprocess driver has a
    # stable import target. ``compile_kernel`` is source-hash-keyed, so
    # repeated compiles for the same source reuse the file.
    compile_result = compile_kernel(kernel)
    if not compile_result.success or compile_result.source_path is None:
        raise ProfilerError(
            f"compile_kernel failed before NCU invocation: {compile_result.error_message}"
        )

    stdout, _rc, driver_degraded, driver_reason = _run_ncu(
        kernel,
        workload,
        input_generator,
        timeout_s=timeout_s,
        mode=mode,
        kernel_source_path=compile_result.source_path,
        kernel_name=kernel_name,
        problem_definition_path=problem_definition_path,
    )
    if driver_degraded:
        return ProfilingResult(
            analytical=analytical,
            ncu=None,
            raw_metrics={},
            degraded_reason=driver_reason,
        )

    ncu, raw, parser_degraded, parser_reason = _parse_ncu_csv(stdout, kernel_name)
    if parser_degraded:
        return ProfilingResult(
            analytical=analytical,
            ncu=None,
            raw_metrics=raw,
            degraded_reason=parser_reason,
        )

    if cache_dir is not None and key is not None:
        _save_ncu_cache(cache_dir, key, ncu, raw)

    return ProfilingResult(
        analytical=analytical,
        ncu=ncu,
        raw_metrics=raw,
    )
