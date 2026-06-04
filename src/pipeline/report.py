"""Report generation — Phase C."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.runtime.usage import UsageBucket, UsageSnapshot, _fmt_tokens

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sol_execbench.core.data import Definition, Workload

    from src.config import HardwareSpec
    from src.eval.profiler import ProfilingResult
    from src.eval.roofline import RooflineResult
    from src.eval.types import BottleneckType
    from src.search.orchestrator import SearchResult


# Degraded-reason slug used when per-workload latency is missing /
# non-positive / non-finite. Routes through the renderer's degraded
# branch so analytical metrics are NOT fabricated from a mismatched
# (latency, nbytes) pair.
_DEGRADED_LATENCY_REASON = "per_workload_latency_missing"


def _degraded_for_missing_latency() -> ProfilingResult:
    """Build a ``ProfilingResult`` placeholder for a workload whose
    per-workload latency is missing / invalid. The ``analytical`` field
    is required by the dataclass; we zero it and rely on
    ``degraded_reason`` to suppress the analytical line at render time.
    Roofline data (AI / ridge / bottleneck) lives on a separate dict
    and is still rendered for these workloads.
    """
    from src.eval.profiler import AnalyticalMetrics, ProfilingResult
    return ProfilingResult.make_degraded(
        AnalyticalMetrics(
            achieved_tflops=0.0,
            achieved_bandwidth_gb_s=0.0,
            pct_peak_compute=0.0,
            pct_peak_bandwidth=0.0,
        ),
        _DEGRADED_LATENCY_REASON,
    )


def _resolve_workload_roofline(
    definition: Definition | None,
    workload: Workload,
    kernel,
    roofline=None,
) -> tuple[int, int]:
    """Return ``(flops, nbytes)`` for a workload. SOL path derives from
    Definition + Workload via ``compute_roofline_inputs``; placeholder
    path falls back to the kernel spec's populated counts.

    *roofline* (optional ``RooflineResult``) lets SOLAR counts outrank
    the shape-formula fallback — same precedence as the orchestrator.
    Returns ``(0, nbytes)`` when only flops can't be derived; ``(0, 0)``
    only when shape resolution also failed.
    """
    if definition is not None:
        from src.benchmark.roofline_shapes import compute_roofline_inputs
        return compute_roofline_inputs(definition, workload, roofline=roofline)
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
    ``winner_roofline_per_workload`` maps a workload UUID to its
    ``RooflineResult`` (run-level invariants — ``arithmetic_intensity``
    and ``ridge_point``, both in MACs/byte). Populated from the same
    SOLAR call that fills ``winner_per_workload_bottlenecks`` so the
    rendered report can surface AI / ridge_point alongside the
    achieved-throughput numbers from the per-workload profile.

    ``usage_stats`` carries the per-iter × per-agent LLM call counts
    and input/output token consumption. ``None`` and an empty snapshot
    both render the same ``(no LLM usage captured)`` fallback under the
    section header. The live pipeline always passes a populated
    snapshot — ``None`` only fires for legacy / direct-construction
    callers.
    """

    baseline_latency_us: float = 0.0
    best_latency_us: float = 0.0
    sol_score: float = 0.0
    speedup: float = 0.0
    reference_baseline_latency_us: float | None = None
    acts_root_latency_us: float | None = None
    technique_trace: list[str] = field(default_factory=list)
    bottleneck: BottleneckType | None = None
    winner_per_workload_bottlenecks: dict[str, BottleneckType] = field(default_factory=dict)
    winner_profiling_per_workload: dict[str, ProfilingResult] = field(default_factory=dict)
    winner_roofline_per_workload: dict[str, RooflineResult] = field(default_factory=dict)
    remaining_headroom_pct: float = 0.0
    total_iterations: int = 0
    termination_reason: str = ""
    reward_hack_suspect: bool = False
    calibration_warning: bool = False
    hardware_spec: HardwareSpec | None = None
    usage_stats: UsageSnapshot | None = None


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
            hardware_spec=hardware_spec,
        )
    path = result.tree.path_to_node(best.id)
    trace = [n.action_applied for n in path if n.action_applied]

    per_workload_bottlenecks: dict[str, BottleneckType] = {}
    per_workload_profiling: dict[str, ProfilingResult] = {}
    per_workload_roofline: dict[str, RooflineResult] = {}
    if workloads and hardware_spec is not None:
        do_reprofile = bool(input_generators)
        if do_reprofile:
            from src.eval.profiler import profile_kernel

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
            # SOLAR first so its counts feed compute_roofline_inputs via
            # roofline= below — otherwise shape formulas bail on op_type=None
            # and the report silently drops workloads SOLAR could classify.
            solar = None
            if definition is not None:
                solar = derive_t_sol_from_solar(
                    definition, w, hardware_spec, arch_yaml_path=arch_yaml_path,
                )
                if solar is not None:
                    per_workload_bottlenecks[w.uuid] = solar.bottleneck
                    # Capture the full RooflineResult so the renderer can
                    # surface AI / ridge_point (run-level invariants in
                    # MACs/byte) alongside the per-workload profile.
                    per_workload_roofline[w.uuid] = solar
            flops, nbytes = _resolve_workload_roofline(
                definition, w, best.kernel, roofline=solar,
            )
            # No (flops, nbytes) gate — symmetric with the per-iter loop.
            if not do_reprofile:
                continue

            # Per-workload latency is required for analytical metrics. If
            # missing / non-positive / non-finite, treat this workload as
            # **degraded** rather than fabricating throughput numbers from
            # the run-level aggregate latency (which belongs to a
            # different workload's nbytes — see report.py history for the
            # `bw 4839.4%` regression that motivated this).
            latency_us = per_workload_latency_us.get(w.uuid)
            if latency_us is None or latency_us <= 0 or not math.isfinite(latency_us):
                per_workload_profiling[w.uuid] = _degraded_for_missing_latency()
                continue
            latency_s = latency_us / 1e6

            # Default blob_roots to the problem dir when caller didn't
            # supply an override — same precedence as ``_load_problem``.
            effective_blob_roots = blob_roots
            if effective_blob_roots is None and definition_path is not None:
                effective_blob_roots = [definition_path.parent]

            # Materialize the inputs once to capture dtypes for the
            # pct_peak.compute denominator (see _pick_compute_peak). The
            # per-workload generator is called again inside profile_kernel,
            # so this is a small redundant call but keeps the dtype-gather
            # path uniform across the three profile_kernel sites.
            from src.eval.profiler import _collect_input_dtypes
            try:
                _repr_inputs = ig(0)
            except Exception as exc:
                logger.warning(
                    "report dtype-gather: input generator raised %s: %s — "
                    "pct_peak.compute will use fp32_fallback",
                    type(exc).__name__, exc,
                )
                _repr_inputs = ()
            _repr_dtypes = _collect_input_dtypes(_repr_inputs)

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
                input_dtypes=_repr_dtypes,
            )

    score = best.score
    if score is None:
        return OptimizationReport(
            technique_trace=trace,
            bottleneck=result.run_bottleneck,
            winner_per_workload_bottlenecks=per_workload_bottlenecks,
            winner_profiling_per_workload=per_workload_profiling,
            winner_roofline_per_workload=per_workload_roofline,
            total_iterations=result.total_iterations,
            termination_reason=termination,
            hardware_spec=hardware_spec,
        )
    return OptimizationReport(
        baseline_latency_us=score.baseline_latency_us,
        best_latency_us=score.candidate_latency_us,
        sol_score=score.sol_score,
        speedup=score.speedup,
        reference_baseline_latency_us=result.reference_baseline_latency_us,
        acts_root_latency_us=result.baseline_root_latency_us,
        technique_trace=trace,
        bottleneck=result.run_bottleneck,
        winner_per_workload_bottlenecks=per_workload_bottlenecks,
        winner_profiling_per_workload=per_workload_profiling,
        winner_roofline_per_workload=per_workload_roofline,
        remaining_headroom_pct=(1.0 - score.sol_score) * 100,
        total_iterations=result.total_iterations,
        termination_reason=termination,
        reward_hack_suspect=score.reward_hack_suspect,
        calibration_warning=score.calibration_warning,
        hardware_spec=hardware_spec,
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
        if report.reference_baseline_latency_us is not None:
            ref = report.reference_baseline_latency_us
            # In Option C, ``baseline_latency_us`` holds the scoring T_b, so
            # the Triton root's own median is carried separately. When the
            # root median is absent, render "n/a" rather than falling back
            # to ``baseline_latency_us`` (which holds the reference T_b here
            # and would mislabel it as the Triton root).
            root_str = (
                f"{report.acts_root_latency_us:.2f} us"
                if report.acts_root_latency_us is not None
                else "n/a"
            )
            lines.extend([
                f"  Scoring baseline: flashinfer reference   T_b = {ref:.2f} us",
                f"  ACTS Triton root: {root_str}   (SOL score vs reference below)",
                f"  Best:      {report.best_latency_us:.2f} us",
                f"  SOL score: {report.sol_score:.4f}  (vs reference; "
                f"headroom {report.remaining_headroom_pct:.1f}%)",
                # ``report.speedup`` already equals reference/candidate in
                # Option C (score.baseline_latency_us IS the reference T_b and
                # best_latency_us IS the candidate), and compute_sol_score
                # guards the zero-candidate division — so no inline recompute.
                f"  Speedup vs reference: {report.speedup:.2f}x",
            ])
        else:
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
        lines.extend(_render_profiling_block(
            report.winner_profiling_per_workload,
            report.winner_roofline_per_workload,
        ))
    if report.reward_hack_suspect:
        lines.append("  [AUDIT] reward_hack_suspect — candidate beats T_SOL (physics violation)")
    if report.calibration_warning:
        lines.append("  [AUDIT] calibration_warning — baseline already at/below T_SOL")
    lines.extend(_render_usage_block(report.usage_stats))
    if report.hardware_spec is not None:
        lines.extend(_render_hardware_spec_block(report.hardware_spec))
    return "\n".join(lines)


def _render_hardware_spec_block(hw: "HardwareSpec") -> list[str]:
    """Render the frozen merged HardwareSpec as a fixed-format block.

    Always emits all fields, even when zero — degraded-detection runs
    show zeros so the absence-of-detection case is visible at a glance
    rather than silently omitted.
    """
    return [
        "Hardware spec",
        f"  Name:               {hw.name or '(unknown)'}",
        f"  Frequency:          {hw.freq_GHz} GHz",
        f"  SRAM capacity:      {hw.SRAM_capacity} B",
        f"  SRAM B/cycle:       {hw.SRAM_byte_per_cycle}",
        f"  DRAM capacity:      {hw.DRAM_capacity} B",
        f"  DRAM B/cycle:       {hw.DRAM_byte_per_cycle}",
        f"  MAC/cycle FP32 SM:  {hw.MAC_per_cycle_fp32_sm}",
        f"  MAC/cycle TF32 TC:  {hw.MAC_per_cycle_tf32_tc}",
        f"  MAC/cycle FP16 TC:  {hw.MAC_per_cycle_fp16_tc}",
        f"  MAC/cycle BF16 TC:  {hw.MAC_per_cycle_bf16_tc}",
        f"  MAC/cycle FP8  TC:  {hw.MAC_per_cycle_fp8_tc}",
        f"  MAC/cycle INT8 TC:  {hw.MAC_per_cycle_int8_tc}",
        f"  MAC/cycle NVFP4 TC: {hw.MAC_per_cycle_nvfp4_tc}",
        f"  Peak DRAM BW:       {hw.peak_memory_bandwidth_gb_s:.2f} GB/s",
        f"  Peak SRAM BW:       {hw.peak_sram_bandwidth_gb_s:.2f} GB/s",
        f"  Peak FP32:          {hw.peak_flops_fp32:.2f} TFLOPS",
        # FP16 and BF16 reported on separate lines: ``MAC_per_cycle_fp16_tc``
        # and ``MAC_per_cycle_bf16_tc`` are distinct fields, and on hardware
        # where they differ (or one is zero with the other populated), a
        # combined label would misstate one of the throughputs.
        f"  Peak FP16:          {hw.peak_flops_fp16:.2f} TFLOPS",
        f"  Peak BF16:          {hw.peak_flops_bf16:.2f} TFLOPS",
    ]


def _render_profiling_block(
    per_workload: dict[str, ProfilingResult],
    per_workload_roofline: dict[str, RooflineResult] | None = None,
) -> list[str]:
    """Format the per-workload analytical + NCU block for the rendered
    report. Suppresses the NCU section when every entry is degraded
    with ``ncu_binary_not_found`` (common on CI without the ncu binary)
    so the operator doesn't see a wall of DEGRADED notices.

    AI / ridge_point are read from ``per_workload_roofline`` (run-level
    invariants in MACs/byte) and rendered alongside the achieved-
    throughput numbers from ``per_workload``. When no roofline is
    available for a given workload UUID, the AI / ridge prefix is
    omitted from that line.

    Per-workload entries flagged with ``degraded_reason ==
    _DEGRADED_LATENCY_REASON`` skip the analytical throughput line
    entirely (TFLOPS / GB/s / pct_peak require a valid latency to be
    meaningful) and render a ``[DEGRADED: ...]`` marker beneath the
    roofline summary. Operators still see the workload + its roofline
    classification, but no fabricated throughput numbers.
    """
    per_workload_roofline = per_workload_roofline or {}
    all_ncu_missing = all(
        p.ncu is None and p.degraded_reason == "ncu_binary_not_found"
        for p in per_workload.values()
    )

    lines: list[str] = ["  Winner profile (per workload):"]
    for uuid, p in per_workload.items():
        a = p.analytical
        rr = per_workload_roofline.get(uuid)
        latency_missing = p.degraded_reason == _DEGRADED_LATENCY_REASON

        if latency_missing:
            # Per-workload latency was missing / invalid upstream — render
            # roofline summary (independent of latency) but suppress the
            # analytical throughput line. Fabricating TFLOPS / GB/s from a
            # mismatched (latency, nbytes) pair was the root cause of the
            # `bw 4839.4%` regression on workload 841b0afa.
            if rr is not None:
                lines.append(
                    f"    [{uuid}] "
                    f"AI {rr.arithmetic_intensity:.2f}, "
                    f"ridge {rr.ridge_point:.2f}, "
                    f"bottleneck={rr.bottleneck.value}"
                )
            else:
                lines.append(f"    [{uuid}]")
            lines.append(
                "      [DEGRADED: missing per-workload latency — "
                "analytical metrics suppressed]"
            )
        else:
            # Happy path: render the analytical throughput line. Roofline
            # prefix (AI / ridge) keeps the same shape it had before this
            # rendering split — bottleneck is rendered run-level
            # elsewhere, not inlined here.
            roofline_prefix = (
                f"AI {rr.arithmetic_intensity:.2f}, ridge {rr.ridge_point:.2f}, "
                if rr is not None
                else ""
            )
            if not p.has_analytical:
                lines.append(
                    f"    [{uuid}] {roofline_prefix}"
                    "[analytical unavailable — no byte count]"
                )
            else:
                lines.append(
                    f"    [{uuid}] "
                    f"{roofline_prefix}"
                    f"{a.achieved_tflops:.2f} TFLOPS / {a.achieved_bandwidth_gb_s:.2f} GB/s "
                    f"(pct_peak: compute {a.pct_peak_compute * 100:.1f}% "
                    f"[{a.compute_peak_dtype}] · "
                    f"bw {a.pct_peak_bandwidth * 100:.1f}%)"
                )
        if all_ncu_missing:
            continue
        if p.ncu is not None:
            n = p.ncu
            # Stall values come from NCU's
            # ``smsp__average_warp_latency_issue_stalled_<reason>.pct``
            # metric. Despite the ``.pct`` suffix and NCU's ``%`` unit
            # tag, the value is **not** a bounded percentage — it's the
            # average warp-cycles-stalled per warp instruction issued,
            # multiplied by 100. For deeply-stalled kernels values in
            # the thousands or tens-of-thousands are normal (the
            # golden fixture's ``imc_miss`` is 57,700; a real GPU run
            # of rmsnorm produced 534,386 for ``long_scoreboard``).
            # Rendering with a trailing ``%`` would mislead the
            # operator into treating these as fractions of total
            # stalls; show them as unitless ``cyc/inst×100`` instead.
            # The relative ranking (dominant vs runner-up) is still
            # the actionable signal.
            tc = (
                "n/a" if n.tensor_core_util_pct is None
                else f"{n.tensor_core_util_pct:.1f}%"
            )
            lines.append(
                f"      NCU: occ {n.sm_occupancy_pct:.1f}% · "
                f"L2 {n.l2_hit_rate_pct:.1f}% · "
                f"TC {tc} · "
                f"top stalls {n.warp_stall_dominant} "
                f"({n.warp_stall_dominant_pct:.1f} cyc/inst×100), "
                f"{n.warp_stall_runner_up} "
                f"({n.warp_stall_runner_up_pct:.1f} cyc/inst×100)"
            )
        elif p.degraded:
            lines.append(
                f"      [DEGRADED: {p.degraded_reason or 'unknown'}]"
            )
    return lines


def _render_usage_block(snapshot: UsageSnapshot | None) -> list[str]:
    """Render the per-iter × per-agent LLM usage table.

    `None` and `snapshot.is_empty` collapse to the same fallback line.
    Populated snapshots render a wide table with em-dashes for empty
    cells, followed by conditional cached/reasoning footer lines (only
    when the sub-bucket is non-zero).
    """
    if snapshot is None or snapshot.is_empty:
        return ["Resource usage (LLM): (no LLM usage captured)"]

    columns = snapshot.columns
    iters = sorted(snapshot.by_iter.keys())

    def _cell(b: UsageBucket | None) -> str:
        if b is None or b.is_zero:
            return "—"
        return (
            f"{b.invocations} ({b.turns}) / "
            f"{_fmt_tokens(b.input_tokens)}→{_fmt_tokens(b.output_tokens)}"
        )

    # Column widths: header width vs. widest cell in that column.
    col_widths: dict[str, int] = {}
    for agent in columns:
        widest = len(agent)
        for it in iters:
            cell = _cell(snapshot.by_iter_agent.get((it, agent)))
            widest = max(widest, len(cell))
        # row-total column considered separately below
        col_widths[agent] = widest
    total_widest = len("total")
    for it in iters:
        row_total = _cell(snapshot.by_iter.get(it))
        total_widest = max(total_widest, len(row_total))
    iter_col_width = max(len("Iter"), len(str(iters[-1])))

    header = (
        f"{'Iter'.rjust(iter_col_width)} | "
        + " | ".join(a.ljust(col_widths[a]) for a in columns)
        + f" | {'total'.ljust(total_widest)}"
    )
    lines: list[str] = ["Resource usage (LLM)", header]

    for it in iters:
        row_cells = [
            _cell(snapshot.by_iter_agent.get((it, agent))).ljust(col_widths[agent])
            for agent in columns
        ]
        # Skip rows where every agent cell is empty.
        if all(c.strip() == "—" for c in row_cells):
            continue
        lines.append(
            f"{str(it).rjust(iter_col_width)} | "
            + " | ".join(row_cells)
            + f" | {_cell(snapshot.by_iter.get(it)).ljust(total_widest)}"
        )

    # Run-total row.
    total_cells = [
        _cell(snapshot.by_agent.get(agent)).ljust(col_widths[agent])
        for agent in columns
    ]
    lines.append(
        f"{'total'.rjust(iter_col_width)} | "
        + " | ".join(total_cells)
        + f" | {_cell(snapshot.total).ljust(total_widest)}"
    )

    # Conditional sub-bucket footer.
    total = snapshot.total
    if total.cached_input_tokens > 0 and total.input_tokens > 0:
        pct = total.cached_input_tokens / total.input_tokens * 100
        lines.append(
            f"  of which cached input: "
            f"{_fmt_tokens(total.cached_input_tokens)} ({pct:.1f}%)"
        )
    if total.reasoning_output_tokens > 0 and total.output_tokens > 0:
        pct = total.reasoning_output_tokens / total.output_tokens * 100
        lines.append(
            f"  of which reasoning output: "
            f"{_fmt_tokens(total.reasoning_output_tokens)} ({pct:.1f}%)"
        )

    return lines
