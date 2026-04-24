# Runtime — `src/runtime/`

Per-run observability substrate: one run directory, three sinks, structured events.

Every ACTS run produces exactly one `runs/run_<UTC>/` directory holding a human-readable log, a structured JSONL event stream, and SDK trace records. The runtime module owns setup, file handles, and teardown for all three.

## Run directory layout

`RunContext.create()` creates `./runs/run_<YYYYMMDDTHHMMSS_ffffffZ>/` and populates:

| File | Purpose | Consumer |
|------|---------|----------|
| `run.log` | Human-readable stdlib log (DEBUG to file, INFO to stderr). | `tail -f` during a run; post-mortem text search. |
| `events.jsonl` | Structured ACTS-narrative events, one JSON object per line, RFC-8259 valid. | Tooling via `jq`; scoring / progress dashboards. |
| `traces/acts_trace_<UTC>.jsonl` | SDK-level records per LLM call and per tool call, written by `JSONLTraceProcessor`. | Ground-truth replay of agent conversations. |

The three sinks are independent: `events.jsonl` records the orchestrator's narrative (what the search did); `traces/*.jsonl` records what the SDK actually dispatched. Cross-referencing both is how you verify claims that the narrative cannot make on its own — see the truthfulness note on `coder_submitted` below.

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
- `iter` is an explicit keyword. It appears on per-iteration events (`iter_start`, `planner_selected`, `coder_submitted`, `coder_failed`, `bench_done`, `profile_done`, `score_computed`, `reviewer_feedback`, `branch_dead_end`, `iter_end`) and is `None` on run-scope events (`run_start`, `baseline_*`, `verify_*`, `run_end`).
- **Never raises.** Serialization failures are caught and logged; file-handle errors during write do not propagate.
- Skips serialization entirely when `logger.isEnabledFor(INFO)` is false — cheap to leave in hot paths.
- All additional `**fields` are merged flat into the JSON object. Use `finite_or_none(x)` on any float that could be `inf`/`nan` (e.g. latency after a failed bench) so JSON stays valid.

### `finite_or_none(x) -> float | None`

Maps `math.inf`, `-math.inf`, `math.nan` → `None`; passes finite floats through. Required because `json.dumps(float('nan'))` produces `NaN`, which is not valid JSON.

### `bind(fh) / unbind()`

Module-level handle registration, guarded by `_lock`. `RunContext.create` calls `bind(fh)` after opening `events.jsonl` line-buffered; `RunContext.close` calls `unbind()` before closing the handle. A second `bind()` on top of an already-bound handle raises — one run, one sink.

### Event catalog — `CORE_EVENT_KINDS`

Frozenset of 18 kinds:

**Run scope** — `run_start`, `baseline_attempt`, `baseline_success`, `baseline_failure`, `baseline_ready`, `verify_start`, `verify_done`, `run_end`.

**Per-iteration** — `iter_start`, `planner_selected`, `coder_submitted`, `coder_failed`, `bench_done`, `profile_done`, `score_computed`, `reviewer_feedback`, `branch_dead_end`, `iter_end`.

Notable semantics:

- `coder_submitted` carries **no pass/fail claim**. The orchestrator cannot verify compile or correctness gates from `CoderAgent.implement()`'s return value alone. Ground-truth per-tool-call records live in `traces/*.jsonl`; cross-reference both streams when auditing.
- `coder_failed` covers any `ImplementationError` (compile failure, correctness failure, exhausted retries).
- `iter_end.outcome` is exactly one of three constants: `ITER_ADVANCED` (`"advanced"`), `ITER_DEAD_END` (`"dead_end"`), `ITER_SKIPPED` (`"skipped"`). `skipped` fires only after `coder_failed` and implies no tree mutation.

## `run_context.py`

### `RunContext` (dataclass)

Public fields:

| Field | Type | Description |
|-------|------|-------------|
| `run_dir` | `Path` | The `runs/run_<UTC>/` directory. |
| `events_path` | `Path` | `run_dir / "events.jsonl"`. |
| `log_path` | `Path` | `run_dir / "run.log"`. |
| `traces_dir` | `Path` | `run_dir / "traces"` (or the explicit override). |
| `started_at` | `str` | `iso_ts()` captured at create-time. |
| `trace_processor` | property | The wired `JSONLTraceProcessor`, or `None` if `capture_traces=False` or setup fell back. |

### `RunContext.create(root=None, *, trace_dir=None, capture_traces=True)`

One-shot setup, idempotent only via `close()`:

1. Create `<root-or-cwd>/runs/run_<filename_ts()>/` and `traces/` (or the `trace_dir` override).
2. Configure the root stdlib logger: `FileHandler(log_path, level=DEBUG)` + `StreamHandler(stderr, level=INFO)`, format `"%(asctime)s %(name)s %(levelname)s %(message)s"`. Silences noisy libraries (`httpx`, `openai`, `agents`) to `WARNING`.
3. Open `events.jsonl` line-buffered and call `events.bind(fh)`.
4. Call `_wire_trace_capture(target)` to register the SDK trace processor with the resolved `traces_dir` (skipped entirely when `capture_traces=False`).
5. Return the populated `RunContext`.

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
| `mkdir runs/run_<UTC>/` denied | stderr WARNING, null-paths `RunContext`, `emit()` degrades to logger-only | yes |
| SDK not installed | stderr WARNING, no `traces/*.jsonl`, everything else works | yes |
| Crash mid-run (uncaught exception) | `atexit` flushes `events.jsonl` + closes `run.log` | n/a |
| SIGKILL | line-buffered writes preserve the last complete line; partial last line may be lost | n/a |

Every path here is "continue, not abort" by design — the logger observes the run, it does not gate it.

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
