# Runtime — `src/runtime/`

Per-run observability substrate: one run directory, three streaming sinks plus a per-node search-tree dump, structured events.

Every ACTS run produces exactly one `runs/run_<UTC>/` directory holding a human-readable log, a structured JSONL event stream, SDK trace records, a per-node dump of the search tree, and a rendered end-of-run report. The runtime module owns setup, file handles, and teardown for all five.

## Run directory layout

`RunContext.create()` creates `./runs/run_<YYYYMMDDTHHMMSS_ffffffZ>/` and populates:

| File | Purpose | Consumer |
|------|---------|----------|
| `run.log` | Human-readable stdlib log (DEBUG to file, INFO to stderr). | `tail -f` during a run; post-mortem text search. |
| `events.jsonl` | Structured ACTS-narrative events, one JSON object per line, RFC-8259 valid. | Tooling via `jq`; scoring / progress dashboards. |
| `traces/acts_trace_<UTC>.jsonl` | SDK-level records per LLM call and per tool call, written by `JSONLTraceProcessor`. | Ground-truth replay of agent conversations. |
| `report.txt` | Rendered end-of-run `OptimizationReport` plus an `=== ACTSConfig (resolved at run start) ===` JSON block. Written best-effort by `optimize.main()`; `OSError` during write degrades to a WARNING and the run continues. | Human-readable end-of-run summary; post-mortem reproduction of the exact resolved config. |
| `tree/` | Per-committed-node `node_<id>/{kernel.py, ncu.json, ncu.ncu-rep, meta.json}` (streamed per-iter) + end-of-run `index.json` / `tree.{txt,dot,mmd,preview.md}` (visualizations). Written by `tree_dump`. | Postmortem inspection of every kernel + score + status the search produced; cross-referenced into `traces/*.jsonl` via the SDK `workflow_name="acts_iter"` metadata. |

The five artifact families are independent: `events.jsonl` records the orchestrator's narrative (what the search did); `traces/*.jsonl` records what the SDK actually dispatched; `tree/` records every committed node's full state; `report.txt` carries the rendered end-of-run summary + resolved config. Cross-referencing them is how you verify claims that any single stream cannot make on its own — see the truthfulness note on `coder_submitted` below, and the `tree/` section's trace cross-reference recipe.

## `timefmt.py`

Two UTC timestamp formatters, deliberately distinct so run-directory names and JSONL payloads never mix formats.

- `filename_ts() -> str` — `YYYYMMDDTHHMMSS_ffffffZ` (microsecond precision, filename-safe, no colons). Used for the `run_<...>` directory name and the `acts_trace_<...>.jsonl` filename.
- `iso_ts() -> str` — `YYYY-MM-DDTHH:MM:SSZ` (second precision, ISO 8601). Used for the `ts` field on every `events.jsonl` record.

## `events.py`

Single module-level JSONL sink bound to the current run. No class — one process writes one event stream.

### `emit(kind, *, iter=None, **fields) -> None`

Fans out each call to two sinks:

1. Stdlib logger (`logger.info`) — renders a human string into `run.log`.
2. Bound JSONL file handle — writes `{"ts": iso_ts(), "kind": kind, "iter": iter, **fields}\n`.

Contract:

- `kind` must be in `CORE_EVENT_KINDS`; other kinds log a warning and are still written (schema drift stays visible, never silent).
- `iter` is an explicit keyword. It appears on per-iteration events (`iter_start`, `planner_selected`, `planner_failed`, `coder_submitted`, `coder_failed`, `bench_done`, `profile_done`, `score_computed`, `reviewer_feedback`, `reviewer_metric_query`, `branch_dead_end`, `iter_end`, `trace_emitted`, `reward_hack_detected`, `reward_hack_confirmed`, `reward_hack_cleared`, `calibration_warning`, `sibling_context_rendered`, `repeated_pathway_dead_end`) and is `None` on run-scope events (`run_start`, `baseline_*`, `verify_*`, `run_end`, `clock_lock_unavailable`, `clock_drift_detected`).
- **Never raises.** Serialization failures are caught and logged; file-handle errors during write do not propagate.
- Skips serialization entirely when `logger.isEnabledFor(INFO)` is false — cheap to leave in hot paths.
- All additional `**fields` are merged flat into the JSON object. Use `finite_or_none(x)` on any float that could be `inf`/`nan` (e.g. latency after a failed bench) so JSON stays valid.

### `finite_or_none(x) -> float | None`

Maps `math.inf`, `-math.inf`, `math.nan` → `None`; passes finite floats through. Required because `json.dumps(float('nan'))` produces `NaN`, which is not valid JSON.

### `bind(fh) / unbind()`

Module-level handle registration, guarded by `_lock`. `RunContext.create` calls `bind(fh)` after opening `events.jsonl` line-buffered; `RunContext.close` calls `unbind()` before closing the handle. A second `bind()` on top of an already-bound handle raises — one run, one sink.

### Event catalog — `CORE_EVENT_KINDS`

Frozenset of 29 kinds:

**Run scope** — `run_start`, `baseline_attempt`, `baseline_success`, `baseline_failure`, `baseline_ready`, `verify_start`, `verify_done`, `run_end`, `clock_lock_unavailable`, `clock_drift_detected`.

**Per-iteration** — `iter_start`, `planner_selected`, `planner_failed`, `coder_submitted`, `coder_failed`, `bench_done`, `profile_done`, `score_computed`, `reviewer_feedback`, `reviewer_metric_query`, `branch_dead_end`, `iter_end`, `trace_emitted`, `reward_hack_detected`, `reward_hack_confirmed`, `reward_hack_cleared`, `calibration_warning`, `sibling_context_rendered`, `repeated_pathway_dead_end`.

**SOL integration sub-grouping** (added 2026-04-27, distributed across the two scopes above):

- `trace_emitted` (per-iter) — fired once per evaluation with the SOL `Trace` payload (Tier 1).
- `clock_lock_unavailable` (run-scope) — clock-lock setup failed at run start; emitted once.
- `clock_drift_detected` (run-scope) — clock drift observed during the run; emitted once.
- `reward_hack_detected` (per-iter) — channel A: process-level detector raised inside the eval block → branch DEAD_END.
- `reward_hack_confirmed` (per-iter) — channel B confirm: sol_score's `reward_hack_suspect` re-eval reproduced the anomaly → branch DEAD_END.
- `reward_hack_cleared` (per-iter) — channel B clear: sol_score's `reward_hack_suspect` re-eval did not reproduce → score accepted.
- `calibration_warning` (per-iter) — fires when sol_score's `calibration_warning` bit is set (T_k near or below the ~T_SOL margin).

**Sibling-aware agent contracts sub-grouping** (added 2026-05-13, both per-iter; see [`doc/specs/2026-05-13-sibling-aware-agent-contracts-design.md`](specs/2026-05-13-sibling-aware-agent-contracts-design.md)):

- `sibling_context_rendered` (per-iter) — fires once per Planner call AND once per Reviewer call whenever the parent has been expanded before (`sibling_context != ""`). Payload: `iter`, `parent_node_id`, `sibling_count: int`, `regressed_actions: list[str]`, `consumer: Literal["planner", "reviewer"]`. On iter ≥2 with a sibling, both events fire (one per consumer) for the same parent.
- `repeated_pathway_dead_end` (per-iter) — fires when Reviewer verdict is DEAD_END AND `child.action_applied` matches a regressed sibling — the existing reviewer/system.md rule ("regression + same pathway on sibling = dead_end") actually fired with sibling evidence. Payload: `iter`, `action: str`, `sibling_iter: int`. Leading-indicator companion to the `branch_dead_end(reason="reviewer_judged")` that follows; lets postmortems count successful sibling-driven prunes without re-walking the tree.

Notable semantics:

- `coder_submitted` carries **no pass/fail claim**. The orchestrator cannot verify compile or correctness gates from `CoderAgent.implement()`'s return value alone. Ground-truth per-tool-call records live in `traces/*.jsonl`; cross-reference both streams when auditing.
- `coder_failed` covers any `ImplementationError` (compile failure, correctness failure, exhausted retries).
- `score_computed` carries `iter`, `score` (sol_score), `is_new_best`, `reward_hack_suspect`, `calibration_warning`, `t_k_us` (kernel median latency), `t_b_us` (baseline median latency), `t_sol_us` (T_SOL used in scoring), and `t_sol_source` — `"solar"` when SOLAR's pipeline produced T_SOL successfully, `"builtin"` when the in-process `compute_roofline()` fallback was used. The source field lets downstream consumers (telemetry, analysis, future memory retrieval) distinguish SOLAR-grounded sol_score numbers from fallback-grounded ones when auditing across runs that may have used different T_SOL backends.
- `planner_failed`: any `PlanningError` cause — turn-budget exhaustion, missing `submit_plan` call, transient retry exhaustion, or the available-actions guard rejecting an unknown technique. Carries `iter` and `reason` (truncated exception string ≤ 200 chars). Always followed by `iter_end(outcome="skipped")`; no tree mutation occurs on this path.
- `reviewer_metric_query`: emitted by the Reviewer's `query_metric` tool body each time the LLM invokes it during a multi-turn review. Carries `iter` (orchestrator iteration index), `count` (number of names in the query), and `names` (list of the first 8 names from the query, capped to keep `events.jsonl` lines bounded). Emission is gated on `ACTSConfig.reviewer_metric_queries=True`. Records what the Reviewer LLM asked for via the `query_metric` tool — useful post-run for analyzing whether the multi-turn capability is being exercised and on what metrics.
- `iter_end.outcome` is exactly one of three constants: `ITER_ADVANCED` (`"advanced"`), `ITER_DEAD_END` (`"dead_end"`), `ITER_SKIPPED` (`"skipped"`). `skipped` fires only after either `coder_failed` or `planner_failed` and implies no tree mutation.

### Dead-end reasons

The `branch_dead_end` event's `reason` field carries a `DeadReason` enum member's string value. The enum (`class DeadReason(str, Enum)`, defined in `events.py`) inherits from `str` so members JSON-serialize cleanly as their string value — telemetry consumers (log parsers, regression tests) key on these stable strings. Dynamic detail (which CUDA error message, which exception text) goes into the separate `detail` payload field, not concatenated into `reason`.

`DeadReason` also doubles as the type of `TreeNode.dead_reason` (see [`search.md`](search.md)), so the same enum drives both the telemetry payload and the on-tree distinction between promotable vs. unpromotable DEAD_END nodes. Two of the members (`BEAM_PRUNED`, `REVIEWER_JUDGED`) are not emitted via `branch_dead_end` today — beam pruning is a frontier-management decision, not a "branch died" event, and Reviewer verdicts flow through `reviewer_feedback`. They exist as enum members because the node field is the unified record of why a `DEAD_END` flag is set, regardless of which code path set it.

Members:

- `REWARD_HACK` (`"reward_hack"`) — channel A process-level reward-hack detector tripped.
- `REWARD_HACK_CONFIRMED` (`"reward_hack_confirmed"`) — channel B re-eval confirmed sol_score's `reward_hack_suspect`.
- `CUDA_ERROR` (`"cuda_error"`) — CUDA runtime error during evaluation.
- `PROFILER_ERROR` (`"profiler_error"`) — profiler subprocess or NCU failure.
- `BENCH_FAILURE` (`"bench_failure"`) — benchmark run did not produce a usable measurement.
- `REPR_LATENCY_UNAVAILABLE` (`"repr_workload_latency_unavailable"`) — representative-workload latency missing/non-finite (sol_score input incomplete).
- `AGENT_FAILURE` (`"agent_failure"`) — agent-side error not covered by the more specific buckets above.
- `BEAM_PRUNED` (`"beam_pruned"`) — node lost the beam competition. On-tree only; not emitted as `branch_dead_end`.
- `REVIEWER_JUDGED` (`"reviewer_judged"`) — Reviewer's verdict for the iter was `DEAD_END`. On-tree only; the verdict flows through `reviewer_feedback` for telemetry.

`DEAD_REASONS` is now `frozenset(DeadReason)`. The orchestrator's `_emit_dead_end` helper takes a typed `DeadReason` argument; the type system enforces validity, so the legacy `if reason not in DEAD_REASONS` warning has been removed.

## `run_context.py`

### `RunContext` (dataclass)

Public fields:

| Field | Type | Description |
|-------|------|-------------|
| `run_dir` | `Path` | The `runs/run_<UTC>/` directory. |
| `events_path` | `Path` | `run_dir / "events.jsonl"`. |
| `log_path` | `Path` | `run_dir / "run.log"`. |
| `traces_dir` | `Path` | `run_dir / "traces"` (or the explicit override). |
| `started_at` | `datetime` | `datetime.now(timezone.utc)` captured at create-time (timezone-aware UTC). |
| `trace_processor` | property | The wired `JSONLTraceProcessor`, or `None` if `capture_traces=False` or setup fell back. |

### `RunContext.create(root=None, *, trace_dir=None, capture_traces=True)`

One-shot setup, idempotent only via `close()`:

1. Create `<root-or-cwd>/runs/run_<filename_ts()>/` and `traces/` (or the `trace_dir` override).
2. Configure the root stdlib logger: `FileHandler(log_path, level=DEBUG)` + `StreamHandler(stderr, level=INFO)`, format `"%(asctime)s %(name)s %(levelname)s %(message)s"`. Silences noisy libraries (`httpx`, `openai`, `agents`) to `WARNING`.
3. Open `events.jsonl` line-buffered and call `events.bind(fh)`.
4. Call `_wire_trace_capture(target)` to register the SDK trace processor with the resolved `traces_dir` (skipped entirely when `capture_traces=False`).
5. Return the populated `RunContext`.

Setting `ACTS_OPENAI_DEBUG` to a truthy value (`"1"`, `"true"`, `"yes"`, case-insensitive) drops `openai` and `httpx` from the silenced set so the SDK's DEBUG-level request/response bodies — including `finish_reason`, raw `choices[0]`, and full request/response payloads — land in `<run_dir>/run.log`. Intended as an opt-in escape hatch for diagnosing thinking-model failures; `agents` stays silenced. **Warning**: persisted request bodies and response payloads may include API keys, system prompts, or other sensitive content — do not share `run.log` from a debug-mode run without redaction.

On any `OSError` during setup, `_cleanup_partial_setup(...)` tears down whatever was created, falls back to a null-paths `RunContext` with a stderr-only `basicConfig`, and returns it. The caller sees no exception — a partial disk failure must not kill the run.

### `close()`

Idempotent teardown, safe to call twice. Sequence:

1. `events.unbind()`.
2. Close the `events.jsonl` file handle.
3. Shut down the registered trace processor (if any).
4. Remove and close every handler this `RunContext` added to the root logger.

Calling `emit()` after `close()` no-ops on the JSONL side (sink unbound) and still writes to stderr through any surviving handler.

### Operator-visible failure modes

| Failure mode | Operator symptom | Run continues? |
|---|---|---|
| `emit()` write fails (disk full, closed FD) | silent for that event | yes |
| Unknown `kind` passed to `emit()` | `run.log` WARNING line; record still written | yes |
| `tree_dump.dump_node` / `finalize_tree` write fails (disk full, closed FD, hardlink-cross-device error) | `run.log` WARNING line, that node / those summary files missing from `tree/` | yes |
| `mkdir runs/run_<UTC>/` denied | stderr WARNING, null-paths `RunContext`, `emit()` degrades to logger-only | yes |
| SDK not installed | stderr WARNING, no `traces/*.jsonl`, everything else works | yes |
| Crash mid-run (uncaught exception) | `atexit` flushes `events.jsonl` + closes `run.log` | n/a |
| SIGKILL | line-buffered writes preserve the last complete line; partial last line may be lost | n/a |

Every path here is "continue, not abort" by design — the logger observes the run, it does not gate it.

## Tree dump (`<run_dir>/tree/`)

Every committed node is streamed to disk under `tree/node_<id>/` as the
search progresses; end-of-run files are written by
`tree_dump.finalize_tree`.

| Path | Content |
|------|---------|
| `tree/node_<id>/kernel.py` | Triton source verbatim. |
| `tree/node_<id>/ncu.json`  | NCU `raw_metrics` dict. Absent on degraded runs. |
| `tree/node_<id>/ncu.ncu-rep` | Binary NCU report. Open in Nsight Compute. Absent on degraded runs. |
| `tree/node_<id>/meta.json` | Structural + scoring + status fields. |
| `tree/index.json` | All-nodes summary + edges. |
| `tree/tree.txt`   | ASCII visualization. |
| `tree/tree.dot`   | Graphviz source — render: `dot -Tpng tree.dot > tree.png`. |
| `tree/tree.mmd`   | Mermaid source — preview in GitHub / VS Code. |
| `tree/tree.preview.md` | Markdown wrapper around `tree.mmd` — opens directly in VS Code's built-in Markdown preview (`Ctrl+Shift+V`) when the *Markdown Preview Mermaid Support* extension is installed. |

Module surface mirrors `events.py`: `bind(tree_root)` / `unbind()` /
`is_bound()` manage the single bound root, `dump_node(node, *, iter_no,
ncu_rep_src, failure_detail=None)` streams one committed node,
`finalize_tree(tree)` writes the five top-level files. Both write paths
are no-ops when unbound and swallow `OSError` (logged at `WARNING`) so
a tree-dump hiccup cannot kill a running search. `RunContext.create`
calls `bind(run_dir / "tree")` after `events.bind(...)`;
`RunContext.close` calls `unbind()`.

DEAD_END cause schema in `meta.json`:

- `dead_reason` (top-level, string from `DeadReason.value`) — categorical
  cause, set on every DEAD_END node regardless of path (infra-error
  kill, beam-pruned, Reviewer-judged). Single source of truth; comes
  from `node.dead_reason` via `_late_bound_fields` so it round-trips
  through `finalize_tree`.
- `failure_detail` (top-level, string) — kill-site prose, populated by
  `_kill_branch` only when the kill site carried a dynamic message
  (exception text, workload-errors string). Absent on beam-pruned and
  Reviewer-judged paths and on the advance path.

The previous nested `failure: {reason, detail}` block was retired
because `failure.reason` always duplicated `dead_reason`; only
`failure.detail` carried unique information.

`finalize_tree` also rewrites each per-node `meta.json`'s late-bound
fields — `branch_quality`, `dead_reason`, `score`,
`per_workload_latency_us`, `children_ids`, and `last_review` (6 fields
total) — from the final tree state, so any node whose state mutated
after its streamed dump reflects the truth on disk. The root's
`meta.json` is streamed at baseline-completion (orchestrator
`dump_node(root, ...)` after the baseline benchmark), so the rewrite
applies to it just as much as to children. Canonical reconciliation
cases: beam-evicted nodes whose `branch_quality` + `dead_reason`
mutate post-dump, Reviewer-judged DEAD_END nodes (kernel ran fine but
Reviewer's verdict for the iter was DEAD_END), nodes whose benchmark
or score landed after the initial dump (late `score` /
`per_workload_latency_us`), and parents that gained `children_ids` as
later iters attached. The rewrite preserves every other key (including
`failure_detail`) and skips nodes that never streamed a `meta.json`
(e.g., a node added before a crash aborted `_kill_branch`).

### Trace cross-reference

Per-iteration LLM-call detail lives in `traces/acts_trace_<ts>.jsonl`,
keyed by SDK trace metadata. The orchestrator wraps each agent
invocation in `with trace(workflow_name="acts_iter", metadata={"iter":
N, "agent": "planner|coder|reviewer"})`. To pull the planner / coder /
reviewer records for node 5 (committed on iter 3):

```bash
jq 'select(.metadata.iter == 3 and .metadata.agent == "reviewer")' \
  <run_dir>/traces/*.jsonl
```

`meta.json.trace_workflow` always equals `"acts_iter"` so the filter
recipe is documented in the file itself.

### Trace record schema

Each line in `traces/acts_trace_<ts>.jsonl` is one of two events emitted
by the OpenAI Agents SDK: `span_end` (a unit of work closed) or
`trace_end` (a top-level trace closed). Spans are written in
close-order, so children appear in the file **before** their parents.

**`event: "span_end"` envelope** — common to every span line:

| Field | Meaning |
|---|---|
| `span_id` | Unique id for this span. |
| `trace_id` | Trace this span belongs to; spans of one agent invocation share it. |
| `parent_id` | Span this one nests under, or `null` for the trace root. Forms the tree `agent ⊃ custom "turn" ⊃ generation` / `function`. |
| `started_at` / `ended_at` | ISO-8601 UTC; subtract for duration. |
| `span_data` | Polymorphic payload — shape selected by `span_data.type`. |
| `error` | `null` on success, or `{message, data}` on failure (e.g. agent span carries `{"message": "Max turns exceeded", "data": {"max_turns": 8}}` when the Coder loop hits its cap). |

**`event: "trace_end"` envelope**:

| Field | Meaning |
|---|---|
| `trace_id` | Matches the trace's spans. |
| `name` | SDK default `"Agent workflow"`. |
| `started_at` / `ended_at` | `null` unless explicitly enabled; span timestamps are authoritative. |
| `metadata` | Free-form caller tags (the orchestrator sets `{iter, agent}` here per the cross-reference recipe above). |

**`span_data` shapes**, dispatched on `type`:

- `type: "agent"` — the agent loop. Fields: `name` (agent identity,
  e.g. `"Coder-Translator"`), `handoffs` (other agents reachable), `tools`
  (list of exposed tool names, e.g.
  `["compile_kernel_tool", "check_correctness_tool", "submit_kernel"]`),
  `output_type` (structured-output schema; `"str"` when the agent emits
  via a tool call rather than a typed return).
- `type: "generation"` — one LLM round-trip. Fields: `input` (list of
  `{role, content}` messages sent), `output` (list of assistant
  messages — each carries `content`, `refusal`, `role`, `annotations`,
  `audio`, `function_call`, `tool_calls`, `reasoning_content`; reasoning
  CoT lands in `reasoning_content` separately from `content`), `model`,
  `model_config` (full request config: sampling knobs, `reasoning.effort`,
  Anthropic-style `extra_body.thinking`, `base_url`, retry/cache flags),
  `usage` (`requests`, `input_tokens`, `output_tokens`, `total_tokens`,
  `input_tokens_details.cached_tokens`,
  `output_tokens_details.reasoning_tokens`).
- `type: "function"` — one tool call. Fields: `name` (tool name),
  `input` (JSON-stringified args), `output` (tool return as string),
  `mcp_data` (`null` for in-process Python tools; populated when the
  tool came from an MCP server).
- `type: "custom"` — project-defined marker emitted by ACTS, not the
  SDK. Fields: `name` (`"turn"` or `"task"`), `data` (free-form dict
  using `sdk_span_type` as the discriminator; turn markers carry
  `turn`, `agent_name`, and a per-turn `usage` aggregate of
  `input_tokens` / `output_tokens` / `cached_input_tokens`). Used by
  the orchestrator to roll up token usage per agent call without
  re-walking every child `generation`.

**Counting turns**: one Agents-SDK turn = one model call + zero-or-more
tool calls the model requested in that response. A typical Coder turn
that calls one tool produces three `span_end` lines — the
`generation`, the `function`, and the enclosing `custom "turn"`. A
reasoning-only turn produces two (no `function` child).

### `★ best` convention

ASCII / DOT / Mermaid all mark the run's best-scoring node with `★`.
`index.json` sets `is_best=true` on that node and `best_node_id` at the
top level.

## Live-watch one-liners

Human narrative:

```bash
tail -f runs/run_<UTC>/run.log
```

Structured summary — iteration starts, scores, and milestones only:

```bash
tail -f runs/run_<UTC>/events.jsonl \
  | jq -c 'select(.kind | IN("iter_start","score_computed","run_end","baseline_ready","branch_dead_end"))'
```

Every `score_computed`:

```bash
jq -c 'select(.kind == "score_computed")' runs/run_<UTC>/events.jsonl
```

Dead-end reasons across a run:

```bash
jq -c 'select(.kind == "branch_dead_end") | {iter, reason}' runs/run_<UTC>/events.jsonl
```
