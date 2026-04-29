"""Report generation — Phase C."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload

    from src.config import HardwareSpec
    from src.eval.profiler import ProfilingResult
    from src.eval.types import BottleneckType
    from src.search.orchestrator import SearchResult


def _resolve_workload_roofline(
    definition: Definition | None,
    workload: Workload,
    kernel,
) -> tuple[int, int]:
    """Return ``(flops, nbytes)`` for a workload. SOL path derives from
    Definition + Workload via ``compute_roofline_inputs``; placeholder
    path falls back to the kernel spec's populated counts. Returns
    ``(0, 0)`` when the op_type has no formula — caller decides how to
    handle.
    """
    if definition is not None:
        from src.benchmark.roofline_shapes import compute_roofline_inputs
        return compute_roofline_inputs(definition, workload)
    return kernel.spec.flop_count, kernel.spec.memory_bytes


@dataclass
class OptimizationReport:
    """Final report of an ACTS optimization run.

    ``bottleneck`` is the once-per-run classification — sourced from
    SOLAR when available (``derive_t_sol_from_solar`` on the
    representative workload), otherwise from the analytical band
    classifier. Invariant across iterations because the problem +
    representative workload + hardware don't change.
    ``winner_per_workload_bottlenecks`` maps a workload UUID to its
    SOLAR-derived bottleneck (one ``derive_t_sol_from_solar`` call per
    selected workload), so individual workloads can disagree with the
    run-level summary above.
    ``winner_profiling_per_workload`` maps a workload UUID to the
    ``ProfilingResult`` captured by re-profiling the winning kernel on
    every selected workload.
    """

    baseline_latency_us: float = 0.0
    best_latency_us: float = 0.0
    sol_score: float = 0.0
    speedup: float = 0.0
    technique_trace: list[str] = field(default_factory=list)
    bottleneck: BottleneckType | None = None
    winner_per_workload_bottlenecks: dict[str, BottleneckType] = field(default_factory=dict)
    winner_profiling_per_workload: dict[str, ProfilingResult] = field(default_factory=dict)
    remaining_headroom_pct: float = 0.0
    total_iterations: int = 0
    termination_reason: str = ""
    reward_hack_suspect: bool = False
    calibration_warning: bool = False


def generate_report(
    result: SearchResult,
    *,
    workloads: list[Workload] | None = None,
    input_generators: list[Callable[..., Any]] | None = None,
    hardware_spec: HardwareSpec | None = None,
    cache_dir: Path | None = None,
    definition: Definition | None = None,
    definition_path: Path | None = None,
    blob_roots: list[Path] | None = None,
    arch_yaml_path: Path | None = None,
) -> OptimizationReport:
    """Generate an optimization report from a completed search result.

    ``bottleneck`` is taken verbatim from ``result.run_bottleneck`` —
    the once-per-run classification that drove retriever / planner /
    reviewer. ``winner_per_workload_bottlenecks`` is populated by
    calling ``derive_t_sol_from_solar`` on each selected workload when
    ``definition`` and ``hardware_spec`` are both provided; SOLAR is
    the authoritative bottleneck source. Workloads where SOLAR is
    unavailable or returns ``None`` get omitted from the dict rather
    than fall back to the analytical classifier.

    When ``workloads`` + ``input_generators`` + ``hardware_spec`` are
    provided, the winning kernel is re-profiled on *every* selected
    workload (spec §3.4 "Phase C full-suite rule") and the results are
    stored in ``winner_profiling_per_workload``. If any of those are
    ``None``, per-workload re-profiling is skipped and the field stays
    empty — callers in the placeholder pipeline pay no re-profile cost.

    *definition_path* is the source ``definition.json`` the profiler
    subprocess driver reloads to rebuild input generators (SOL no longer
    carries the path on the type itself).

    *blob_roots* is forwarded into ``profile_kernel`` so the NCU
    subprocess driver can resolve safetensors-backed workload inputs
    against the same root list the in-process generator used. ``None``
    falls back to ``[definition_path.parent]`` when ``definition_path``
    is provided (mirrors ``_load_problem``); if both are ``None`` the
    field is omitted from the spec JSON entirely.
    """
    best = result.best_node
    termination = result.termination_reason.value
    if best is None:
        # Degenerate run — search produced no scored node. Match the
        # ``score is None`` branch below: empty trace, no scoring fields,
        # bottleneck/run-level info still threaded through so the caller
        # at least sees the termination reason.
        return OptimizationReport(
            technique_trace=[],
            bottleneck=result.run_bottleneck,
            winner_per_workload_bottlenecks={},
            winner_profiling_per_workload={},
            total_iterations=result.total_iterations,
            termination_reason=termination,
        )
    path = result.tree.path_to_node(best.id)
    trace = [n.action_applied for n in path if n.action_applied]

    per_workload_bottlenecks: dict[str, BottleneckType] = {}
    per_workload_profiling: dict[str, ProfilingResult] = {}
    if workloads and hardware_spec is not None:
        do_reprofile = bool(input_generators)
        if do_reprofile:
            from src.eval.profiler import profile_kernel

            aggregate_latency_s = (
                best.score.candidate_latency_us / 1e6
                if best.score is not None and best.score.candidate_latency_us > 0
                else 1e-6
            )
            per_workload_latency_us = best.per_workload_latency_us or {}

            if len(input_generators) != len(workloads):
                raise ValueError(
                    "input_generators length must match workloads length "
                    f"(got {len(input_generators)} generators for "
                    f"{len(workloads)} workloads); the per-workload re-profile "
                    "pass would otherwise silently truncate via zip()."
                )

        if definition is not None:
            from src.eval.roofline import derive_t_sol_from_solar

        generators = input_generators if do_reprofile else [None] * len(workloads)
        for w, ig in zip(workloads, generators):
            flops, nbytes = _resolve_workload_roofline(definition, w, best.kernel)
            if flops <= 0 or nbytes <= 0:
                # No formula → skip both classification and re-profile for
                # this workload rather than poisoning the dicts.
                continue
            if definition is not None:
                solar = derive_t_sol_from_solar(
                    definition, w, hardware_spec, arch_yaml_path=arch_yaml_path,
                )
                if solar is not None:
                    per_workload_bottlenecks[w.uuid] = solar.bottleneck
            if not do_reprofile:
                continue

            # Per-workload latency first (threaded from BenchmarkResult),
            # falling back to the aggregate when the workload wasn't measured.
            latency_us = per_workload_latency_us.get(w.uuid)
            if latency_us is not None and latency_us > 0 and math.isfinite(latency_us):
                latency_s = latency_us / 1e6
            else:
                latency_s = aggregate_latency_s

            # Default blob_roots to the problem dir when caller didn't
            # supply an override — same precedence as ``_load_problem``.
            effective_blob_roots = blob_roots
            if effective_blob_roots is None and definition_path is not None:
                effective_blob_roots = [definition_path.parent]

            per_workload_profiling[w.uuid] = profile_kernel(
                best.kernel,
                w.model_dump(mode="json"),
                ig,
                hardware_spec=hardware_spec,
                flops=flops,
                nbytes=nbytes,
                latency_s=latency_s,
                cache_dir=cache_dir,
                problem_definition_path=definition_path,
                blob_roots=effective_blob_roots,
            )

    score = best.score
    if score is None:
        return OptimizationReport(
            technique_trace=trace,
            bottleneck=result.run_bottleneck,
            winner_per_workload_bottlenecks=per_workload_bottlenecks,
            winner_profiling_per_workload=per_workload_profiling,
            total_iterations=result.total_iterations,
            termination_reason=termination,
        )
    return OptimizationReport(
        baseline_latency_us=score.baseline_latency_us,
        best_latency_us=score.candidate_latency_us,
        sol_score=score.sol_score,
        speedup=score.speedup,
        technique_trace=trace,
        bottleneck=result.run_bottleneck,
        winner_per_workload_bottlenecks=per_workload_bottlenecks,
        winner_profiling_per_workload=per_workload_profiling,
        remaining_headroom_pct=(1.0 - score.sol_score) * 100,
        total_iterations=result.total_iterations,
        termination_reason=termination,
        reward_hack_suspect=score.reward_hack_suspect,
        calibration_warning=score.calibration_warning,
    )


def render_report(report: OptimizationReport) -> str:
    """Render a multi-line text summary of the optimization report.

    Skips the scoring block when ``baseline_latency_us == 0`` so a
    degenerate run (no scored best node) doesn't print misleading
    "0.00us / 0.00x" lines.

    When ``winner_profiling_per_workload`` is populated, emits an
    analytical + NCU profiling block. If every per-workload profile is
    degraded with ``ncu_binary_not_found``, the NCU block is suppressed
    (analytical only) — a common case on CI / machines without ncu.
    """
    lines = [
        f"Search completed: {report.termination_reason}",
        f"  Iterations: {report.total_iterations}",
    ]
    if report.baseline_latency_us > 0:
        lines.extend([
            f"  Baseline:  {report.baseline_latency_us:.2f} us",
            f"  Best:      {report.best_latency_us:.2f} us",
            f"  SOL score: {report.sol_score:.4f}  (headroom {report.remaining_headroom_pct:.1f}%)",
            f"  Speedup:   {report.speedup:.2f}x",
        ])
    if report.technique_trace:
        lines.append(f"  Trace: {' → '.join(report.technique_trace)}")
    if report.bottleneck is not None:
        lines.append(f"  Bottleneck (run): {report.bottleneck.value}")
    if report.winner_per_workload_bottlenecks:
        per_workload = ", ".join(
            f"{uuid}={label.value}"
            for uuid, label in report.winner_per_workload_bottlenecks.items()
        )
        lines.append(f"  Bottleneck (per workload): {per_workload}")
    if report.winner_profiling_per_workload:
        lines.extend(_render_profiling_block(report.winner_profiling_per_workload))
    if report.reward_hack_suspect:
        lines.append("  [AUDIT] reward_hack_suspect — candidate beats T_SOL (physics violation)")
    if report.calibration_warning:
        lines.append("  [AUDIT] calibration_warning — baseline already at/below T_SOL")
    return "\n".join(lines)


def _render_profiling_block(
    per_workload: dict[str, ProfilingResult],
) -> list[str]:
    """Format the per-workload analytical + NCU block for the rendered
    report. Suppresses the NCU section when every entry is degraded
    with ``ncu_binary_not_found`` (common on CI without the ncu binary)
    so the operator doesn't see a wall of DEGRADED notices."""
    all_ncu_missing = all(
        p.ncu is None and p.degraded_reason == "ncu_binary_not_found"
        for p in per_workload.values()
    )

    lines: list[str] = ["  Winner profile (per workload):"]
    for uuid, p in per_workload.items():
        a = p.analytical
        lines.append(
            f"    [{uuid}] "
            f"AI {a.arithmetic_intensity:.2f}, ridge {a.ridge_point:.2f}, "
            f"{a.achieved_tflops:.2f} TFLOPS / {a.achieved_bandwidth_gb_s:.2f} GB/s "
            f"(pct_peak: compute {a.pct_peak_compute * 100:.1f}% · "
            f"bw {a.pct_peak_bandwidth * 100:.1f}%)"
        )
        if all_ncu_missing:
            continue
        if p.ncu is not None:
            n = p.ncu
            lines.append(
                f"      NCU: occ {n.sm_occupancy_pct:.1f}% · "
                f"L2 {n.l2_hit_rate_pct:.1f}% · "
                f"TC {n.tensor_core_util_pct:.1f}% · "
                f"top stalls {n.warp_stall_dominant} ({n.warp_stall_dominant_pct:.1f}%), "
                f"{n.warp_stall_runner_up} ({n.warp_stall_runner_up_pct:.1f}%)"
            )
        elif p.degraded:
            lines.append(
                f"      [DEGRADED: {p.degraded_reason or 'unknown'}]"
            )
    return lines
