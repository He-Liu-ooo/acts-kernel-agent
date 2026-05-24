# LLM-call Pareto experiment — design spec (2026-05-19)

> **Status:** uncommitted scratch artifact per CLAUDE.md (specs/plans never get
> committed). Will be deleted from the working tree once results land back
> into RESEARCH.md / JOURNAL.md.

## 1. Goal & methodology claim

Back RESEARCH.md's primary research claim with empirical evidence for the
**first** instantiation of the budget-allocation methodology:

> Given a total LLM-call budget B (the system-level constraint), how should
> the budget be allocated across Planner / Coder / Reviewer (the per-agent
> tradeoff space) to maximize kernel-generation quality?

This experiment produces a **Pareto frontier** in (realized total LLM calls,
end-of-run quality) space across hand-curated agent-allocation regimes. The
artifact answers: at any given budget level, which allocation regime
dominates on quality.

**Scoping decisions consciously made.** This is a single-workload pilot.
Generalization claims across kernel classes (memory-bound vs compute-bound
vs mixed) are explicitly out of scope and queued for a follow-up experiment
once this first pass validates the framing.

## 2. System-level constraint & quality metrics

### Constraint axis (x-axis)

- **Realized total LLM calls** per run, read post-hoc from
  `<run_dir>/usage.json` (sum across all agents).
- **No hard cap on calls**: framing (B) per the brainstorming exchange.
  Cost is bounded by knob choices and a wallclock safety cap, not by a
  calls-budget kill-switch.

### Quality axes (y-axes)

- **Primary:** end-of-run SOL score of the best-of-tree kernel
  (`OptimizationReport.best_sol_score` in `report.txt`).
- **Secondary:** real runtime (µs) of best-of-tree, averaged across the
  selected workloads.
- Both metrics already populated by `pipeline/report.py`; no orchestrator
  changes required.

### Why these and not AUC-over-budget

End-state metrics suffice for a Pareto frontier characterization. AUC over
budget would require continuous best-so-far tracking across calls, which the
orchestrator does not emit today and which adds instrumentation cost the
first pilot does not need.

## 3. Workload

**L1/048 `fused_gate_up_projection_with_swiglu`** (Gemma3 fused MLP gate +
up projection + SwiGLU epilogue).

Rationale (from RESEARCH.md):

- RESEARCH.md's #1 suggested first-pass — "textbook SwiGLU + tiling win."
- Compute-bound, two parallel BF16 matmuls 3072→24576.
- Reliable signal, modest SOL gap, fewer iter-0 baseline-generation
  failures than memory-bound flash-attention or full decoder blocks.
- Representative workload: `bs=4, seq=2048` → ~1.2 TFLOPs MM.

**Class conclusion is therefore "compute-bound only."** A follow-up
experiment will add a memory-bound workload (L1/067 flash-attn) and a
mixed workload (L2/002 full decoder) to test class-dependence of the
optimal allocation. This pilot does not claim cross-class generalization.

## 4. Regime table

Calibrated against rough per-iter call cost (turns): Planner ≈ 2,
Coder ≈ 5·K (with retries=3) or 6·K (with retries=4), Reviewer ≈ 2 (off)
or 3 (metric_queries on). Plus ~10–15 baseline-generation calls
(`coder-translate`; baseline review adds ~3). `~exp_calls` is
order-of-magnitude only — measuring the realized distribution is the
experiment's whole point.

**Knob bounds reflect the meaningful ranges documented in
`doc/agents.md` → "Turn-budget floors per agent":** Planner ceiling 4,
Reviewer ceiling 4/6 (off/on), Coder via `max_debug_retries` is the
only unbounded turn-budget lever. Cells stay within these ceilings
unless explicitly probing the absolute-floor failure mode.

| # | Regime           | max_depth | K | debug_retries | P_turns | R_turns | metric_q | ~exp_calls | What it tests |
|---|------------------|-----------|---|---------------|---------|---------|----------|-----------:|---------------|
| 1 | Minimal          |  3 | 1 | 1 | 2 | 2 | off | ~35  | Practical floor (all knobs at min that completes workflows) |
| 2 | Lean balanced    |  5 | 2 | 2 | 2 | 2 | off | ~60  | Small-budget operating point |
| 3 | **Default**      | 10 | 4 | 3 | 4 | 6 | **on** | ~260 | Current ACTS defaults (matches `configs/example.cfg`) |
| 4 | Coder-wide       |  5 | 8 | 3 | 2 | 2 | off | ~230 | K=8 → Coder share peaks |
| 5 | Reviewer-heavy   | 10 | 1 | 2 | 2 | 6 | **on** | ~105 | K=1 + metric_q=on → Reviewer share peaks |
| 6 | Patient-Coder    |  5 | 4 | **4** | 2 | 2 | off | ~150 | retries=4 → 10 turns/candidate; replaces unreachable "Planner-deep" |
| 7 | Maxed            | 15 | 8 | **4** | 4 | 6 | on  | ~800 | Ceiling within meaningful range |
| 8 | Default-depth-3  |  3 | 4 | 3 | 4 | 6 | **on** | ~90 | Depth-axis point (sweep at Default knobs) |
| 9 | Default-depth-6  |  6 | 4 | 3 | 4 | 6 | **on** | ~165 | Depth-axis point |
|10 | Default-depth-15 | 15 | 4 | 3 | 4 | 6 | **on** | ~390 | Depth-axis point |

Cells 3 + 8 + 9 + 10 trace the **depth-only sweep curve** at Default knobs
(all with `metric_queries=on` to match the actual `ACTSConfig` default);
cells 1–7 are the named allocation regimes. The depth-only curve is the
control: it shows how quality scales with max_depth alone, against which
the regime variation can be measured.

**Allocation lever asymmetry.** The 6 cfg knobs are not symmetric levers
across agents (see `doc/agents.md` → "Turn-budget floors"):

- **Coder share** scales linearly in K and `max_debug_retries`. Both are
  genuine "Coder-heavy" levers.
- **Reviewer share** scales via `metric_queries` (binary; ~+2 turns per
  iter) and `reviewer_max_turns` up to ceiling 6. Bounded but real.
- **Planner share** has NO per-iter deliberation knob above ceiling 4.
  The only way to push Planner share at fixed total budget is to
  minimize Coder (K=1) + minimize Reviewer (metric_q=off, R=2) + crank
  `max_depth` — which is essentially the depth-axis sweep with K=1, a
  variant we don't include as a named cell to keep the experiment to
  10 regimes. Captured implicitly by comparing regime 5 (K=1) with
  regimes 8/9/10 (K=4) at the same max_depth.

**Total: 10 cells.**

### Per-agent share interpretation

The methodology question asks about *per-agent allocation*, not knob settings.
The knob → agent mapping is asymmetric:

- **Planner share** scales with `max_depth × planner_max_turns_avg` — call
  count is fixed at 1/iter, deliberation depth varies.
- **Coder share** scales with `max_depth × K × coder_avg_turns` — K linearly
  multiplies call count; debug_retries adds variable tail.
- **Reviewer share** scales with `max_depth × (reviewer_max_turns_avg +
  2·metric_queries)` — deliberation + tool calls.

Each run's `usage.json` records realized per-agent call counts directly,
so the post-hoc analysis can plot stacked-bar allocation shares per regime
without needing to derive them from knobs (see § 7).

## 5. Replication, failure handling, reproducibility

- **Replication:** 3 runs per cell → **30 runs total**. Estimated wallclock
  ~5h sequential on the dev GPU (one overnight run).
- **Hard wallclock cap:** 45 min per run via shell `timeout`. Feral runs
  are killed and logged; not retried; excluded from Pareto plot; counted
  toward "completion rate" per regime.
- **No call-count cap.** Realized calls are the constraint axis — capping
  them would re-introduce the (A) kill-switch framing that was already
  declined in brainstorming.
- **Failure handling:** if a run aborts (reward-hack trip, kernel error
  storm, timeout, infra error), it is logged to `failures.tsv` and NOT
  auto-retried. A regime that completes 0/3 reps is itself a research
  finding (the budget is too small to produce *any* viable kernel for this
  workload).
- **Reproducibility:** each run's `report.txt` already dumps the resolved
  `ACTSConfig` JSON + hardware spec. The sweep script additionally records
  the git SHA + `nvidia-smi -q` output once per sweep at startup.
- **LLM nondeterminism:** Planner/Reviewer run at temperature 0.3,
  Coder at 0.0. Three reps at the same cfg are three random draws from
  the agent's behavior distribution; this variance is what we want to
  characterize, not eliminate.

## 6. Implementation surface

Framing (B) keeps the orchestrator's *control flow* untouched — every
post-hoc artifact we need (`usage.json`, `report.txt`) is already
instrumented. The single src change required is exposing two previously
hardcoded turn budgets to `.cfg`.

### Pre-experiment src changes (landed 2026-05-19)

`planner_max_turns: int | None = None` and `reviewer_max_turns: int | None = None`
added to `ACTSConfig` (`src/config.py`); both routed through
`_section_map["search"]` so `.cfg` files override them via the
`[search]` block. `None` preserves the hardcoded defaults (planner=4;
reviewer=4-or-6 conditional on `reviewer_metric_queries`); non-None
overrides uniformly. The orchestrator threads both into
`plan(max_turns=...)` and `review(max_turns=...)` at all three call
sites (baseline review, per-iter planner, per-iter reviewer). Tests in
`tests/test_config.py`, `tests/test_planner.py`, `tests/test_reviewer.py`.

### Artifacts to create

1. **10 cfg files** under `configs/experiments/llm_call_pareto_l1_048/`:
   - `regime_01_minimal.cfg` through `regime_10_default_depth_15.cfg`.
   - Each cfg copies `configs/example.cfg` and overrides only the relevant
     knob block. Workload path pinned to L1/048. Model config pinned to
     the same backend used for all reps (set once at sweep start; recorded
     in the sweep manifest).
   - **Committed**: yes. Unlike spec/plan documents, the cfg fixtures need
     to live in-tree for the experiment to be re-runnable from any commit.
     They are reproducibility infrastructure, not scratch artifacts.

2. **`scripts/run_llm_call_pareto_sweep.sh`**:
   - Iterates 10 cfgs × 3 reps × calls
     `python -m src.pipeline.optimize --config <cfg> --run-dir <run_dir>`.
   - Sequential (one GPU).
   - **Resumable**: skips a `(regime, rep)` pair whose `run_dir` already
     contains a completed `report.txt`.
   - Wallclock-capped at 45 min per run via `timeout 2700 …` wrapper.
   - Logs the git SHA and `nvidia-smi -q` once at start to a
     `sweep_manifest.txt`.
   - Aborted/timed-out runs append a row to `failures.tsv`
     (regime, rep_idx, reason, wallclock_s) and the sweep continues.

3. **`scripts/analyze_llm_call_pareto.py`**:
   - Walks `runs/sweep_l1_048/`, parses `usage.json` + `report.txt` for
     each rep, builds a pandas dataframe matching the schema in § 7.
   - Emits the 3 plots + 1 table in § 7.
   - Output to `runs/sweep_l1_048/analysis/`.
   - Tier-1 dependencies only (pandas + matplotlib); no torch needed.

### Run-directory layout

```
runs/sweep_l1_048/
├── sweep_manifest.txt          # git SHA, GPU info, sweep start time, cfg list
├── failures.tsv                # one row per aborted/timed-out run
├── regime_01_minimal/
│   ├── rep_0/                  # one full ACTS run_dir per rep
│   │   ├── report.txt
│   │   ├── usage.json
│   │   ├── events.jsonl
│   │   ├── run.log
│   │   ├── traces/
│   │   └── tree/
│   ├── rep_1/
│   └── rep_2/
├── regime_02_lean_balanced/
│   └── …
├── regime_10_default_depth_15/
│   └── …
└── analysis/
    ├── df.csv                  # one row per rep, schema in § 7
    ├── plot_1_sol_pareto.png
    ├── plot_2_runtime_pareto.png
    ├── plot_3_allocation_shares.png
    └── summary_table.md        # per-regime median ± IQR + completion rate
```

## 7. Analysis pipeline output

### Dataframe schema (`analysis/df.csv`)

One row per (regime × rep_idx). Columns:

```
regime_name, regime_idx,
max_depth, K, debug_retries, P_turns, R_turns, metric_q,
rep_idx,
realized_calls_total, realized_calls_planner, realized_calls_coder, realized_calls_reviewer,
realized_tokens_total, realized_tokens_planner, realized_tokens_coder, realized_tokens_reviewer,
sol_score_best, runtime_us_best,
completed,                       # bool: report.txt readable + score populated
wallclock_s,
reward_hack_tripped,             # bool from report.txt audit flags
dead_branch_count,               # from tree/index.json
completion_rate_for_regime,      # rolled up across reps in summary table
git_sha
```

### Plots

- **Plot 1 (primary).** Scatter: x = `realized_calls_total`,
  y = `sol_score_best`. Color = regime. Marker = rep_idx. Pareto frontier
  outlined as a connected line over the dominating points. Failed runs
  shown as hollow markers at the x-axis.
- **Plot 2 (secondary).** Same scatter shape, y = `runtime_us_best`.
- **Plot 3 (allocation breakdown).** Stacked bar per regime: median of
  realized Planner / Coder / Reviewer call shares (% of total). Lets a
  reader see "Coder-wide actually means 85% of calls go to Coder," etc.

### Summary table (`analysis/summary_table.md`)

Per-regime row: median SOL ± IQR, median runtime ± IQR, completion rate
(reps completed / reps attempted), median realized total calls. Lets the
reader scan the headline numbers without staring at the scatter.

## 8. What this experiment does NOT do

Conscious scope cuts, all of which are queued for follow-up experiments:

- **No cross-class generalization claim.** Single workload only.
- **No hard call-count cap.** The constraint axis is realized calls,
  not capped calls.
- **No statistical-significance testing.** 3 reps is enough for a Pareto
  visual but not for p-values; if a methodology paper needs error bars
  the follow-up experiment bumps reps to 7+.
- **No interaction-effect analysis.** Hand-curated regimes don't form a
  factorial grid; we can't disentangle "K=8 helped" from "depth=5 hurt."
  A follow-up factorial sweep on the 2 most sensitive knobs identified
  here is the natural next step.
- **No AUC-over-budget metric.** End-state metrics only.
- **No model-config sweep.** One LLM backend per sweep (pinned at start).
  Different backends are a separate axis for a separate experiment.
- **No isolated "Planner-deep" regime.** The Planner has no per-iter
  deliberation knob above the validation-retry budget (`max_turns`
  ceiling = 4 per `doc/agents.md`). Pushing Planner share is achievable
  only via `max_depth` (more iters → more Planner calls), which is
  what the depth-axis sweep (regimes 8/9/10) already exercises in
  combination with regime 5's K=1. A dedicated factorial would be a
  follow-up if the depth-axis result motivates one.
- **No SDK-level absolute-floor probe.** Regime 1 sits at the *practical*
  floor (`max_turns=2/2`, retries=1), not the SDK degenerate floor
  (`max_turns=1/1`, retries=0). A floor probe is a separate experiment
  with a different question ("how often does captured-output recovery
  save a starved run") and a separate analysis pipeline.

## 9. Open questions for follow-up rounds

Recorded here so they don't get lost; each is the seed of its own
brainstorming round:

1. **Class-dependence.** Does the dominant allocation regime shift on
   memory-bound (L1/067) or mixed (L2/002) workloads? Follow-up
   experiment runs the same regime table on those two workloads.
2. **Statistical-grade rerun.** If a regime's Pareto position is
   visually borderline, rerun that regime at 7+ reps to get tight
   error bars.
3. **Factorial follow-up.** After identifying the 2 most sensitive
   knobs from this pilot, run a clean 2D factorial sweep on those two
   to characterize curvature.
4. **AUC-over-budget instrumentation.** If the methodology paper wants
   "per-call quality return" (not just end-state Pareto), the
   orchestrator needs per-iter best-so-far + per-iter realized-calls
   logging. That instrumentation is the gate for an AUC-shaped follow-up.
5. **Variance source decomposition.** Is rep-to-rep variance driven
   primarily by LLM nondeterminism, by GPU bench noise, or by
   reward-hack trips? A small ablation (rerun one regime with the
   OpenAI/DeepSeek `seed` request parameter pinned and clock-locked GPU)
   would tease these apart.

---

**Next step:** user review of this spec. If approved, the writing-plans
skill produces a task-by-task implementation plan covering the 3 artifacts
in § 6.
