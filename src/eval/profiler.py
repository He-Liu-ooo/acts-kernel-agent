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
import logging
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

# WARNING per degraded ``ProfilingResult`` so silent NCU degradations
# are greppable in ``run.log``.
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.kernels.kernel import Kernel

# Cache-bust token. Bump when the curated metric map, stall reasons,
# parser contract, or *cache-key shape* changes. Embedded in every
# cache key so stale entries are unreachable from the new version and
# naturally evicted. ``v2`` includes ``kernel_name`` in the key blob so
# two Kernels with identical source but different declared
# ``triton_kernel_name`` values cannot alias to one entry (otherwise the
# helper-kernel metrics would be returned when the run requested the
# dominant-kernel filter).
_METRIC_SET_VERSION: str = "v2"

# ``_UNSET`` sentinel distinguishes "not yet probed" from "probed → missing".
_UNSET: Any = object()
_NCU_BINARY_CACHE: Any = _UNSET

# Hardcoded fallback for ``_discover_ncu_binary`` when ``shutil.which("ncu")``
# misses — typically a freshly-built ``~/.venvs/acts_run_venv`` whose
# ``activate`` script doesn't prepend cuda's bin to ``PATH``. Matches the
# host-cuda invariant in ``configs/venvs/3.12.md`` (driver 570.x + CUDA
# 12.8). When the host cuda version bumps, update this path AND the
# corresponding step in that recipe in lockstep.
_NCU_FALLBACK_PATH: str = "/usr/local/cuda-12.8/bin/ncu"

# Once-per-process cache for NCU failures the OS makes permanent — most
# notably ``NVreg_RestrictProfilingToAdminUsers=1`` on the host kernel
# module, which makes every non-root NCU invocation fail with
# ``ERR_NVGPUCTRPERM``. After the first observed permission failure,
# subsequent ``_run_ncu`` calls return ``("", -1, True, <reason>)`` without
# forking a subprocess. Reset only by ``_reset_ncu_permission_cache()``
# (test hook); production callers should let the cache live the
# orchestrator's lifetime.
#
# Held value is the diagnostic reason slug (str), e.g.
# ``"ncu_skipped:permanently_unavailable:nvgpuctrperm"``. ``None`` means
# "no permanent failure observed yet". A boolean flag would lose the
# signature, so we stash the slug directly.
_NCU_PERMANENTLY_UNAVAILABLE: str | None = None

# Stderr signatures we consider "permanent for this process". Order
# matters: more specific first so the resulting reason slug is the
# highest-fidelity one. Pairs are ``(needle_lowercase, slug_suffix)``.
# Matching is case-insensitive substring on stderr+stdout combined (NCU
# sometimes writes the actionable error to stdout when --csv is set).
_NCU_PERMANENT_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("err_nvgpuctrperm", "nvgpuctrperm"),
    ("does not have permission to access the gpu performance counter", "counter_perm"),
    ("the user does not have permission", "user_perm"),
    ("nvidia-smi nvlink", "driver_init_failed"),
)

# Environment variable that suppresses NCU entirely — operator escape
# hatch for hosts where NCU is known-broken. ``"1"`` / ``"true"`` /
# ``"yes"`` (case-insensitive) all disable. Anything else, including
# unset, leaves NCU enabled and the auto-detect path takes over.
_NCU_DISABLE_ENV = "ACTS_DISABLE_NCU"


def _reset_ncu_permission_cache() -> None:
    """Test hook: clear the once-per-process permission-cache flag so the
    next ``_run_ncu`` call hits the subprocess path again.

    Production code never calls this — the whole point of the cache is
    to avoid re-paying the subprocess fork cost for an OS-level
    permission failure that won't change mid-run. Tests need to reset
    between scenarios because the module is imported once."""
    global _NCU_PERMANENTLY_UNAVAILABLE
    _NCU_PERMANENTLY_UNAVAILABLE = None


def _ncu_disabled_via_env() -> bool:
    """Return True iff ``ACTS_DISABLE_NCU`` is set to a truthy value."""
    raw = os.environ.get(_NCU_DISABLE_ENV, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _classify_ncu_permanent_failure(stderr: str, stdout: str) -> str | None:
    """Inspect NCU's combined stderr+stdout for a known permanent-failure
    signature. Returns the slug suffix on match (e.g. ``"nvgpuctrperm"``),
    or ``None`` when the failure looks transient / unknown.

    Matching is case-insensitive substring; NCU phrasings have shifted
    across releases (the ``ERR_NVGPUCTRPERM`` token is stable, but the
    free-text suffix is not), so we anchor on tokens NVIDIA documents in
    their counter-access KB article."""
    haystack = (stderr + "\n" + stdout).lower()
    for needle, slug in _NCU_PERMANENT_SIGNATURES:
        if needle in haystack:
            return slug
    return None


# Reason-slug suffixes are embedded into degraded tags read by humans in
# the run log, so we keep them ASCII-safe and short. ``[^\w-]`` would
# also strip the colons we use as field separators, which we want.
_FINGERPRINT_SAFE_RE = re.compile(r"[^\w-]+")


def _stderr_fingerprint(stderr: str, max_len: int = 60) -> str:
    """Distill NCU's stderr into a short, log-safe slug for the degraded
    reason tag. Picks the first non-empty line, slugifies non-word chars,
    and truncates to ``max_len``.

    Example: ``"==ERROR== ProfilerReply error: Not enough memory"``
    becomes ``"ERROR_ProfilerReply_error_Not_enough_memory"``.

    Returns ``""`` when stderr is empty / whitespace — caller falls back
    to the bare ``ncu_nonzero_exit:<rc>`` tag."""
    if not stderr:
        return ""
    first = next(
        (ln.strip() for ln in stderr.splitlines() if ln.strip()),
        "",
    )
    if not first:
        return ""
    # Strip NCU's ``==…==`` prefix decoration if present; the inner text
    # is the load-bearing part.
    first = first.replace("==", " ")
    slug = _FINGERPRINT_SAFE_RE.sub("_", first).strip("_")
    if not slug:
        return ""
    return slug[:max_len]

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
    """Per-iteration dynamic measurements derived from measured latency.

    Pure runtime metrics. Classification + the run-level invariants
    ``arithmetic_intensity`` and ``ridge_point`` live on
    ``RooflineResult`` (see ``src/eval/roofline.py``) since they are
    invariant per ``(problem, representative_workload, hardware)`` —
    re-storing them on every per-iter ``AnalyticalMetrics`` was just
    duplication. Consumers that need AI / ridge_point read them off
    the run-level ``RooflineResult`` instead.
    """

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
    # Path to the binary NCU report (.ncu-rep) when ncu produced one.
    # ``None`` on degraded runs (binary not found, permission denied, etc.)
    # and on cache hits where the report wasn't preserved. Excluded from
    # checkpoint round-trip — callers re-run the profiler if the path is
    # stale.
    ncu_rep_path: Path | None = None

    @property
    def degraded(self) -> bool:
        return self.degraded_reason is not None

    @property
    def has_ncu(self) -> bool:
        return self.ncu is not None

    @classmethod
    def make_degraded(
        cls, analytical: "AnalyticalMetrics", reason: str
    ) -> "ProfilingResult":
        """Construct a degraded ProfilingResult: no NCU data, empty raw
        metrics, with the given reason explaining why. Logs the reason at
        WARNING so the slug is greppable in ``run.log``.

        Named ``make_degraded`` (not ``degraded``) to avoid clashing with
        the ``degraded`` ``@property`` above — Python class attributes
        share a single namespace.
        """
        logger.warning("ncu degraded: %s", reason)
        return cls(
            analytical=analytical,
            ncu=None,
            raw_metrics={},
            degraded_reason=reason,
        )


def _compute_analytical(
    *,
    flops: int,
    nbytes: int,
    latency_s: float,
    hardware_spec: HardwareSpec,
) -> AnalyticalMetrics:
    """Derive per-iteration achieved-throughput metrics from measured latency.

    Returns achieved TFLOPS / bandwidth and their percent-of-peak ratios.
    The kernel's arithmetic_intensity and the hardware's ridge_point are
    run-level invariants and live on ``RooflineResult`` instead — this
    function does not recompute them per iteration.

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

    achieved_tflops = flops / latency_s / 1e12
    achieved_bandwidth_gb_s = nbytes / latency_s / 1e9
    pct_peak_compute = achieved_tflops / peak_tflops
    pct_peak_bandwidth = achieved_bandwidth_gb_s / peak_bw_gb_s

    return AnalyticalMetrics(
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


def _ncu_env() -> dict[str, str]:
    """TMPDIR override shared by capture and import subprocesses
    (workaround for shared-``/tmp`` lock-ownership)."""
    env = os.environ.copy()
    env["TMPDIR"] = _ncu_tmpdir()
    return env


def _extract_ncu_csv(rep_path: Path, *, timeout_s: float = 30.0) -> tuple[str, int, bool, str | None]:
    """Run ``ncu --import <rep_path> --csv --page details`` and return its
    stdout as a CSV stream.

    NCU 2025.x suppresses the CSV stream from stdout whenever ``-o`` is
    set on the profile call (the binary report is written, but stdout
    only carries ``==PROF==`` banners; there is no ``--csv-file`` flag and
    ``--log-file`` only redirects what would have been printed). The
    fix is a second subprocess that re-extracts the CSV from the binary
    after the profile runs — no GPU work, just binary→CSV serialization.

    Returns ``(stdout, returncode, degraded, reason)`` mirroring
    ``_run_ncu``'s shape:
      * ``("", -1, True, "ncu_binary_not_found")`` if ncu vanished from
        PATH between the profile call and this one.
      * ``("", -1, True, "ncu_import_failed:timeout")`` on timeout.
      * ``(stdout, rc, True, "ncu_import_failed:<rc>")`` on non-zero exit.
      * ``(stdout, 0, False, None)`` on success — caller hands stdout
        straight to ``_parse_ncu_csv``.
    """
    binary = _discover_ncu_binary()
    if binary is None:
        return "", -1, True, "ncu_binary_not_found"
    # ``--print-metric-name=name`` is required on the import call too —
    # without it, section data renders metrics under their human-readable
    # *labels* (``Achieved Occupancy``, ``Warp Cycles Per Issued
    # Instruction``) rather than raw dotted names. The flag on the
    # capture call only governs the suppressed-stdout stream; section
    # rendering is re-decided at import time.
    argv = [
        binary,
        "--import",
        str(rep_path),
        "--csv",
        "--page",
        "details",
        "--print-metric-name=name",
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_ncu_env(),
            check=False,
        )
    except FileNotFoundError:
        return "", -1, True, "ncu_binary_not_found"
    except subprocess.TimeoutExpired:
        return "", -1, True, "ncu_import_failed:timeout"
    if completed.returncode != 0:
        return (
            completed.stdout or "",
            completed.returncode,
            True,
            f"ncu_import_failed:{completed.returncode}",
        )
    return completed.stdout or "", 0, False, None


def _build_ncu_argv(
    kernel: "Kernel",
    spec_json_path: Path,
    *,
    mode: str,
    kernel_name: str | None = None,
    out_path: Path | None = None,
) -> list[str]:
    """Build the ``ncu`` command line.

    ``kernel_name`` is the GPU symbol (extracted from ``@triton.jit def``)
    NCU filters on. Falls back to ``kernel.spec.entrypoint`` when no
    Triton kernel is found in the source.

    ``out_path`` is the desired ``.ncu-rep`` report path. When set, the
    argv is extended with ``-o <out_path-without-suffix>`` (NCU appends
    the ``.ncu-rep`` suffix itself); when ``None``, no ``-o`` flag is
    emitted and NCU only writes the CSV stream to stdout.
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

    if out_path is not None:
        # NCU appends ``.ncu-rep`` to whatever ``-o`` value we pass; strip
        # the suffix here so we don't end up with ``foo.ncu-rep.ncu-rep``.
        # ``-f`` (force-overwrite) is required because ``_ncu_tmpdir()`` is
        # user-scoped and persistent across runs; without it, repeat profiles
        # of the same (source, workload, mode, kernel) cache key fail when
        # NCU refuses to overwrite the existing .ncu-rep from a prior run.
        #
        # NCU 2025.x routing quirk: when ``-o`` is set, NCU suppresses the
        # CSV stream from stdout entirely — there's no flag combination
        # that delivers both binary ``.ncu-rep`` and CSV stdout in one
        # invocation. ``_run_ncu`` runs a second subprocess
        # (``ncu --import <rep> --csv --page details``) post-profile to
        # re-extract the CSV from the binary.
        argv += ["-f", "-o", str(out_path.with_suffix(""))]

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
    blob_roots: list[Path] | None = None,
    ncu_rep_out: Path | None = None,
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
    can call ``src.benchmarks.sol_execbench.load(<dir>)``. ``None`` omits
    the key and the driver falls back to ``module.make_inputs`` or
    ``spec['args']``.

    ``blob_roots`` is the list of directories the driver's
    ``build_input_generator`` consults when the workload contains a
    ``SafetensorsInput``. Serialized as ``list[str]`` for JSON-safety;
    the driver rehydrates back to ``list[Path]``. ``None`` omits the
    field — the driver defaults to ``None`` so non-safetensors workloads
    and older cached specs keep working unchanged.

    ``ncu_rep_out`` is the desired ``.ncu-rep`` report path. When set, it
    is forwarded to ``_build_ncu_argv`` as the ``-o`` target so NCU emits
    a binary report alongside the CSV stream. The caller is responsible
    for checking ``ncu_rep_out.exists()`` after success and stashing it
    on the resulting ``ProfilingResult.ncu_rep_path``.

    Spec JSON contract (consumed by ``_profiler_driver``):

    * ``kernel_source_path`` (str): compiled .py the driver imports.
    * ``entrypoint`` (str): name of the host wrapper / kernel callable.
    * ``workload`` (dict): pydantic ``Workload.model_dump`` shape.
    * ``mode`` (str): ``"curated"`` or ``"full"``.
    * ``problem_dir`` (str, optional): SOL problem directory.
    * ``blob_roots`` (list[str], optional): forwarded to
      ``build_input_generator(blob_roots=…)`` in the driver.
    * ``dps`` (bool, default False): when True, the driver pre-allocates
      output buffers via ``sol_execbench.core.bench.io.allocate_outputs``
      and calls ``kernel_fn(*inputs, *outputs)``.
    * ``seed`` (int, optional): RNG seed for input generation; defaults to 0.
    """
    # ``global`` must come before any read of the name in the same
    # function — Python's compiler raises SyntaxError if we declare it
    # later, even on a path the early-return below would skip.
    global _NCU_PERMANENTLY_UNAVAILABLE

    # Operator escape hatch: ``ACTS_DISABLE_NCU=1`` skips NCU entirely.
    # Cheaper than the binary lookup and makes "I know NCU is broken on
    # this host" explicit in the run logs (distinct from the auto-detect
    # ``permanently_unavailable`` slug).
    if _ncu_disabled_via_env():
        return "", -1, True, "ncu_disabled_via_env"

    # Once-per-process cache: if a previous call observed a permanent
    # failure (host kernel NVreg_RestrictProfilingToAdminUsers=1, etc.),
    # short-circuit before forking another doomed subprocess. The cached
    # slug carries the original signature so operators can still tell
    # "permission" from "driver init failed" downstream.
    if _NCU_PERMANENTLY_UNAVAILABLE is not None:
        return "", -1, True, _NCU_PERMANENTLY_UNAVAILABLE

    binary = _discover_ncu_binary()
    if binary is None:
        return "", -1, True, "ncu_binary_not_found"

    spec_payload: dict[str, Any] = {
        "kernel_source_path": str(kernel_source_path) if kernel_source_path else "",
        "entrypoint": kernel.spec.entrypoint,
        "workload": workload,
        "mode": mode,
        # ``kernel.dps`` is the source of truth for whether the host
        # wrapper takes pre-allocated output buffers as positional args
        # after the inputs. Threading it into the spec mirrors the DPS
        # wiring already done in ``benchmark_kernel``,
        # ``verify_correctness``, and ``_reward_hack_re_eval`` so the NCU
        # profile path doesn't silently TypeError on DPS kernels.
        "dps": bool(kernel.dps),
    }
    if problem_definition_path is not None:
        # ``src.benchmarks.sol_execbench.load`` wants the directory
        # (``definition.json`` + sibling ``workload.jsonl``), not the
        # definition file itself.
        spec_payload["problem_dir"] = str(Path(problem_definition_path).parent)
    if blob_roots is not None:
        # JSON cannot serialize ``Path`` directly — coerce to str. The
        # driver rehydrates with ``[Path(p) for p in spec["blob_roots"]]``.
        spec_payload["blob_roots"] = [str(p) for p in blob_roots]
    # ``input_generator`` can't cross the subprocess boundary; the driver
    # rebuilds it from the serialized problem.
    _ = input_generator

    env = _ncu_env()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=env["TMPDIR"]
    ) as f:
        json.dump(spec_payload, f)
        spec_json_path = Path(f.name)

    argv = _build_ncu_argv(
        kernel,
        spec_json_path,
        mode=mode,
        kernel_name=kernel_name,
        out_path=ncu_rep_out,
    )
    # Substitute the discovered absolute path for the bare ``"ncu"``
    # ``_build_ncu_argv`` puts at argv[0]. On a clean venv where
    # ``shutil.which("ncu")`` misses but ``_NCU_FALLBACK_PATH`` resolves,
    # leaving argv[0] as bare ``"ncu"`` would make ``subprocess.run`` raise
    # ``FileNotFoundError`` and degrade as ``ncu_binary_not_found`` even
    # though discovery succeeded. Mirrors ``_extract_ncu_csv``'s pattern.
    argv[0] = binary

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
        # Classify before reporting: permission / driver-init failures
        # are permanent for this process and should flip the
        # short-circuit cache so iteration N+1 doesn't pay the fork
        # cost. Transient / unknown failures keep the legacy
        # ``ncu_nonzero_exit:<rc>`` reason so operators can still see the
        # exit code, but we surface a stderr fingerprint when one exists
        # to make "permission issue vs host crash vs section name typo"
        # distinguishable from the run log.
        stderr = completed.stderr or ""
        stdout = completed.stdout or ""
        permanent_slug = _classify_ncu_permanent_failure(stderr, stdout)
        if permanent_slug is not None:
            reason = f"ncu_skipped:permanently_unavailable:{permanent_slug}"
            _NCU_PERMANENTLY_UNAVAILABLE = reason
            return stdout, completed.returncode, True, reason

        fingerprint = _stderr_fingerprint(stderr)
        if fingerprint:
            return stdout, completed.returncode, True, (
                f"ncu_nonzero_exit:{completed.returncode}:{fingerprint}"
            )
        return stdout, completed.returncode, True, (
            f"ncu_nonzero_exit:{completed.returncode}"
        )

    # When ``-o`` is set, NCU 2025.x suppresses the CSV stream from
    # stdout — ``completed.stdout`` is just ``==PROF==`` banners and the
    # parser would degrade with ``csv_parse:no_header``. Run a second
    # subprocess (``ncu --import <rep> --csv --page details``) to
    # re-extract the CSV from the binary report. No GPU work — just a
    # binary→CSV serialization pass.
    if ncu_rep_out is not None:
        return _extract_ncu_csv(ncu_rep_out, timeout_s=timeout_s)

    return completed.stdout, 0, False, None


def _discover_ncu_binary() -> str | None:
    """Return the absolute path of ``ncu`` on ``$PATH``, or ``None`` if
    missing. Result is cached at module level so long-lived orchestrators
    pay ``shutil.which`` only once per process.

    Falls back to ``_NCU_FALLBACK_PATH`` (host cuda-12.8 install) when
    ``shutil.which`` returns None — survives venvs whose activate script
    doesn't prepend cuda's bin to PATH (clean rebuilds from
    ``configs/venvs/3.12.md``)."""
    global _NCU_BINARY_CACHE
    if _NCU_BINARY_CACHE is _UNSET:
        found = shutil.which("ncu")
        if found is None and Path(_NCU_FALLBACK_PATH).is_file():
            found = _NCU_FALLBACK_PATH
        _NCU_BINARY_CACHE = found
    return _NCU_BINARY_CACHE


def _cache_key(
    kernel_source: str, workload: dict, mode: str, kernel_name: str
) -> str:
    """Deterministic 16-hex-char key mixing source hash + workload + mode
    + resolved ``kernel_name`` + ``_METRIC_SET_VERSION``. The workload is
    serialized via ``json.dumps(..., sort_keys=True)`` so the cache key is
    invariant under dict insertion order — required because the cache
    persists across processes (``.acts_cache/``) and dict ordering on the
    JSON re-load path is not the same as on the original construction
    path. Including ``kernel_name`` keeps multi-jit fused outputs from
    aliasing — the resolved name (Coder-declared, regex fallback, or
    entrypoint last-ditch) is what NCU's ``--kernel-name regex:`` filter
    actually targets, so two runs with the same source but different
    targets must produce distinct cache entries."""
    source_hash = hashlib.sha256(kernel_source.encode("utf-8")).hexdigest()
    workload_repr = json.dumps(workload, sort_keys=True, default=str)
    blob = (
        source_hash + workload_repr + mode + kernel_name + _METRIC_SET_VERSION
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _load_ncu_cache(
    cache_dir: Path, key: str
) -> tuple[NCUMetrics, dict[str, float]] | None:
    """Read and rehydrate a cached ``(NCUMetrics, raw_metrics)`` pair.
    Returns ``None`` on any error (missing file, corrupt JSON, missing
    ``ncu`` field, unknown ``ncu`` field) — a corrupt cache entry is
    treated as a silent miss, not a crash.

    Pre-2026-05-11 cache entries written without a ``raw`` field load
    with ``raw_metrics={}`` — same shape as a degraded re-profile would
    produce on that path.
    """
    try:
        payload = json.loads(_cache_path(cache_dir, key).read_text())
        ncu = NCUMetrics(**payload["ncu"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    raw = payload.get("raw") or {}
    # Belt-and-braces: ensure raw is a mapping; if a legacy/corrupt entry
    # has it as a list/scalar, treat as missing.
    if not isinstance(raw, dict):
        raw = {}
    return ncu, raw


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
    blob_roots: list[Path] | None = None,
) -> ProfilingResult:
    """Hybrid analytical + NCU profile (spec §3.2).

    Flow:

    1. Always compute ``_compute_analytical``. Raises ``ProfilerError`` on
       impossible inputs — the branch dies.
    2. If ``cache_dir`` is given and a cached entry exists under the
       (source-hash, workload, mode, metric-set-version) key, rehydrate
       both the curated ``NCUMetrics`` and the full ``raw_metrics`` dict
       and skip the subprocess. The Reviewer's ``query_metric`` tool
       depends on ``raw_metrics`` being populated.
    3. Operator escape hatch — ``ACTS_DISABLE_NCU=1`` short-circuits with
       reason ``ncu_disabled_via_env``.
    4. Process-wide skip — once a previous call observed an OS-level
       permanent failure (host kernel restricted counters, etc.), every
       subsequent call returns the cached
       ``ncu_skipped:permanently_unavailable:<sig>`` reason without
       forking a subprocess.
    5. Otherwise: discover ``ncu`` on PATH. Missing → degraded result
       with no cache write; branch survives.
    6. Run the NCU subprocess; driver-side failure → degraded, no cache.
       Permission-class failures flip the process-wide skip flag so step
       4 catches the next call.
    7. Parse CSV; parser-side failure → degraded, no cache.
    8. Both green → persist NCUMetrics, return full ProfilingResult.

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
    # to one cache entry. Priority: Coder-declared name (validated
    # upstream) → source-regex fallback (hand-written starters / test
    # fixtures with an empty declared name) → entrypoint last-ditch (so
    # we degrade to ``no_matching_kernel`` rather than crash when neither
    # source has a ``@triton.jit`` def at all). Pure-Python resolution;
    # no I/O cost added to the cache-hit path.
    kernel_name = (
        kernel.triton_kernel_name
        or _extract_triton_kernel_name(kernel.source_code)
        or kernel.spec.entrypoint
    )

    # ``key`` is computed unconditionally — the JSON cache only consults
    # it when ``cache_dir`` is set, but the ``.ncu-rep`` filename uses it
    # in both cases (cache_dir present → next to JSON entry; cache_dir
    # absent → in the per-process NCU temp dir). Pure function, no I/O.
    key = _cache_key(kernel.source_code, workload, mode, kernel_name)

    # Derive the ``-o`` target for ``ncu`` so each (source, workload, mode,
    # kernel_name) tuple gets its own ``.ncu-rep``. When ``cache_dir`` is
    # set the report lands next to the JSON cache entry (dedup-friendly,
    # survives across processes); otherwise it lands in the per-process
    # NCU TMPDIR so the file is still on disk and readable by
    # ``tree_dump.dump_node`` — the orchestrator runs without a cache_dir
    # and would otherwise lose the binary report entirely.
    ncu_rep_out: Path = (cache_dir or Path(_ncu_tmpdir())) / f"{key}.ncu-rep"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is not None:
        cached = _load_ncu_cache(cache_dir, key)
        if cached is not None:
            # Cache stores both the curated NCUMetrics and the full raw
            # metric dict — the Reviewer's ``query_metric`` tool reads
            # ``raw_metrics`` and would otherwise see ``[no data]`` for
            # every query on a cache hit. ``ncu_rep_path`` surfaces the
            # previously-written ``.ncu-rep`` when it's still on disk;
            # if a sibling run pruned it, fall back to ``None`` (callers
            # treat that as "no report to copy").
            cached_ncu, cached_raw = cached
            return ProfilingResult(
                analytical=analytical,
                ncu=cached_ncu,
                raw_metrics=cached_raw,
                ncu_rep_path=ncu_rep_out if ncu_rep_out.exists() else None,
            )

    if _ncu_disabled_via_env():
        return ProfilingResult.make_degraded(analytical, "ncu_disabled_via_env")

    # Skip the (cheap but non-zero) compile + subprocess fork once we've
    # observed an OS-level permanent failure — the host kernel won't
    # un-restrict counters mid-run. The cached slug carries the original
    # signature so the run log keeps "permission" / "driver init" / etc.
    # distinguishable downstream.
    if _NCU_PERMANENTLY_UNAVAILABLE is not None:
        return ProfilingResult.make_degraded(analytical, _NCU_PERMANENTLY_UNAVAILABLE)

    if _discover_ncu_binary() is None:
        return ProfilingResult.make_degraded(analytical, "ncu_binary_not_found")

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
        blob_roots=blob_roots,
        ncu_rep_out=ncu_rep_out,
    )
    if driver_degraded:
        return ProfilingResult.make_degraded(
            analytical, driver_reason or "ncu_unknown_driver_failure"
        )

    ncu, raw, parser_degraded, parser_reason = _parse_ncu_csv(stdout, kernel_name)
    if parser_degraded:
        # Parser path constructs ProfilingResult directly because
        # ``raw_metrics=raw`` may be non-empty (escape hatch for prompt
        # engineers); ``make_degraded`` zeroes raw, so it can't be reused.
        # Log here to keep parity with every other degraded return.
        reason = parser_reason or "ncu_unknown_parser_failure"
        logger.warning("ncu degraded: %s", reason)
        return ProfilingResult(
            analytical=analytical,
            ncu=None,
            raw_metrics=raw,
            degraded_reason=reason,
        )

    if cache_dir is not None:
        _save_ncu_cache(cache_dir, key, ncu, raw)

    # Only surface the report path when ncu actually wrote it — the
    # subprocess could exit zero with the CSV stream intact but no
    # ``.ncu-rep`` on disk if the OS killed the writer between flush and
    # rename. Callers (orchestrator tree-dump) treat ``None`` as "no
    # report to copy".
    final_rep_path = ncu_rep_out if ncu_rep_out.exists() else None

    return ProfilingResult(
        analytical=analytical,
        ncu=ncu,
        raw_metrics=raw,
        ncu_rep_path=final_rep_path,
    )
