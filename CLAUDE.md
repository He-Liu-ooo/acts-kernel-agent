# ACTS Kernel Agent — Claude Code Instructions

## Session Start

1. [PROCESS.md](PROCESS.md) — current status and next feature
2. Read doc/ files only for the specific module you're about to work on
3. [PRD.md](PRD.md) — only if the task touches architecture or requirements
4. [JOURNAL.md](JOURNAL.md) — only if you need rationale for a past decision

Confirm your understanding of the project state and tell me where we left off.

## Workflow

Each step is user-triggered — Claude does not auto-advance.

> **Harness note**: this workflow assumes the Claude Code harness. References to the `Skill` tool, the `Agent` tool, and the `superpowers:*` / `codex:review` / `codex:rescue` skills are Claude-Code-specific invocation surfaces. When working under Codex directly, the equivalents are: design-phase brainstorming → an inline options/tradeoffs discussion before code; parallel subagent dispatch → sequential or manually parallelized terminal sessions; `codex:review` → invoking Codex directly for review (which is exactly what the skill wraps). The governance rules — user-triggered steps, test-first, commit-split approval, no auto-advance — apply in either harness.

1. **Pick feature** — next item from the dependency-ordered list in PROCESS.md.
2. **Design discussion** (if non-trivial) — align on approach before writing code. For any feature that is not mechanical (new module, new data surface, cross-module refactor, anything touching GPU / eval / search / agent contracts), **invoke the `superpowers:brainstorming` skill** via the `Skill` tool before proposing an approach. Default to brainstorming; skip it only for trivially mechanical changes (typo fix, single-call-site rename, one-line bug fix). The settled design + rationale gets recorded in JOURNAL.md before any code is written.
3. **Write tests** — test-first; define expected behavior before implementation.
4. **Write code** — implement to pass the tests. **Start by decomposing the work into subtasks** and identifying any with disjoint file sets and no data dependencies; dispatch those to subagents in parallel via the Agent tool (see "Parallel execution" subsection below for the gate, prompt discipline, and test-collision rule). Sequential subtasks (shared files or import dependencies) stay inline in the main conversation. Parallel dispatch uses general-purpose Agent calls with explicit prompts — *not* the `superpowers:subagent-driven-development` skill chain (see "Superpowers skill scope" subsection).
5. **Review** — user triggers the `codex:review` agent for an automated first pass, then asks Codex directly, then reviews themselves. Iterate on steps 3–4 until review passes.
6. **Simplify** — user triggers `simplify`.
7. **Update docs** — after review passes, verify consistency between src/ and:
   - **doc/** — record the feature's details in the relevant component file
   - **PROCESS.md** — mark feature complete
   - **JOURNAL.md** — record design rationale (if applicable)
   - **PRD.md**

   **At the start of this step, default to parallel subagent dispatch.** The governance files above (PRD / JOURNAL / PROCESS / doc/*) are almost always disjoint from each other — each agent touches one file with the same delta brief, no data dependencies. One agent per file, dispatched in a single message. Reserve inline edits for the case where only one file truly changes, or for a small touch-up after the main fan-out. If the consistency sweep itself is still needed (the read side — scanning for stale references), run that as a single read-only subagent first, then fan out the write side based on its punch list. See "Parallel execution" below for the disjoint-file-set gate.

   **Specs and plans are never committed.** Files under `doc/specs/` or `doc/plans/` (the authoring artifacts from `superpowers:brainstorming` / `superpowers:writing-plans`) live uncommitted in the working tree throughout the feature's lifecycle. They are scratch artifacts — process exhaust by design — and never get staged. The information that matters retires into JOURNAL / PRD / PROCESS / doc/* via direct edits as it stabilizes, *during* the brainstorm and writing-plans flows, not at the end. By the time code lands, the governance docs already carry every load-bearing decision; the spec and plan are deleted from the working tree without ever having been committed. Skill instructions to "commit the design document" are overridden by this rule. Exception: if the feature is multi-phase and later phases will extend the spec, keep the spec uncommitted in the working tree until the series completes, then delete.
8. **Commit** — propose the commit split, discuss with the user, wait for approval, then commit and update **PROCESS.md** for the next round.

### Rules

- Keep PRs small — each change should be small enough to be reviewed efficiently.
- **Do not commit until both Codex review and user review are complete.** This is non-negotiable. Even when work is "obviously good" (tests green, scope clean), even mid-session, even when a previous commit in the same session was approved — every new commit needs a fresh user review. The only exception is when the user has explicitly granted a session-window blanket authorization (e.g. "I permit everything" before going AFK); that authorization expires the moment the user is back in the loop. When in doubt, propose the commit split and wait.
- Do not skip the test-first step for deterministic modules (eval, search, memory, config). For LLM agent modules, mocked tests are acceptable.

### Commit splits

Before running `git commit`, propose the split to the user and wait for approval. A session's work almost always wants more than one commit; the wrong-sized commit is usually "too big," not "too small."

- **Propose, don't execute.** State the planned commits as a list — for each: a title, the file set, a one-line rationale, and the approximate line delta. Do not stage or commit until the user approves the plan.
- **Bias toward finer grain.** If a commit's diff crosses two concerns (e.g., "add module X + refactor X's callers"), split it. If a commit touches more than ~10 files or ~500 lines and the user hasn't already signed off on the size, flag it explicitly and offer a split.
- **Each commit should stand on its own.** A reader scanning `git log --oneline` should recognize what shipped from the title. Pre-session work (landed uncommitted by a prior session) goes in its own commit, separate from session work.
- **Doc-only commits are fine.** Consistency sweeps across PRD / JOURNAL / PROCESS / doc/* belong in their own commit.
- **Fixes surfaced during review are often their own commit.** If a bug fix touches files the refactor also touched, accept the hunk-level staging cost to keep the fix revertable on its own.

### Parallel execution

When a feature decomposes into tasks with **disjoint file sets and no data dependencies**, dispatch them to subagents in parallel instead of doing them sequentially. The cost of a cold-start agent is amortized quickly once the work is non-trivial, and wallclock matters more than per-task token count on multi-task PRs.

- **Gate**: map each task's touched files (source + tests) and confirm zero overlap with any other in-flight task. If two tasks edit the same file or one task's public API is the other's import, serialize them.
- **Test-run collisions**: if two parallel agents would both run the full `pytest tests/` suite, scope each to its own module's tests (`pytest tests/test_profiler_*.py`, etc.) and run the full suite once after all agents land. Concurrent full-suite runs against a shared checkout risk `__pycache__` races and mid-refactor import errors.
- **Prompt discipline**: each agent starts cold. Include in the prompt: the task's scope, explicit "do NOT touch" list for files owned by other in-flight agents, the exact verification command, and "do NOT commit."
- **Dependency-ordered batches**: dispatch the independent set in parallel, wait for all, then dispatch the next dependency tier. Don't serialize inside a tier just because the previous tier was serial.

### Design decisions

When multiple approaches exist, present options with tradeoffs. Wait for user to pick before implementing. Record the decision + rationale in JOURNAL.md.

For non-trivial features the `superpowers:brainstorming` skill is the required entry point — it structures the options / tradeoffs / open-questions dialogue so the user can steer before any code is written. If you catch yourself thinking "I'll just sketch the design inline," that's the signal to invoke the skill instead.

### Superpowers skill scope

Superpowers skills are invoked **only** for the design phase of the workflow:

- `superpowers:brainstorming` — workflow step 2 (design discussion). Produces a spec under `doc/specs/`.
- `superpowers:writing-plans` — runs after brainstorming. Produces an implementation plan under `doc/plans/`.

**Once the spec and plan are written, superpowers is disabled for the remainder of the feature.** Implementation (workflow steps 3–4) follows the plan directly using Read / Edit / Write / Bash and inline TDD; no other superpowers skills are invoked. **This applies equally to small bug fixes and one-line changes that skipped steps 1–2** — "no spec/plan exists yet" is not a loophole; the implementation-flavored skills are off at every coding entry point, full stop. Explicitly:

- Do **not** invoke `superpowers:subagent-driven-development`. Implementation is coordinated from the main conversation context, not via the skill's implementer/reviewer subagent loops. Parallel general-purpose Agent dispatches (per workflow step 4 + "Parallel execution") are still allowed and encouraged — the rule bans skill-chain control of implementation, not subagent dispatch itself.
- Do **not** invoke `superpowers:test-driven-development` or any other implementation-flavored superpowers skill.
- Do **not** invoke any skill-internal review flow. Review (workflow step 5) is **user-triggered only** — the user runs `codex:review` and conducts their own review; Claude Code does not auto-invoke any review skill.

This rule overrides any "next step" instruction inside a skill that suggests invoking another skill, including the writing-plans skill's terminal step (which historically chains into implementation skills) — that chain stops at the plan, and Claude Code waits for the user to trigger workflow step 3.

### After any architectural change

Run a consistency check across src/, doc/, PRD.md, JOURNAL.md, and PROCESS.md. Verify that terminology, function signatures, and data flow descriptions match the actual code. Fix stale references before committing.

### Test Environment

Two venvs, split by tier. **Both live under `~/.venvs/`** so they survive reboots — Ubuntu's `systemd-tmpfiles-setup` runs `D /tmp 1777 root root -` from `/usr/lib/tmpfiles.d/tmp.conf` on every boot, which empties `/tmp` regardless of filesystem type. Do not put venvs (or anything you want to keep) under `/tmp`. If a venv is missing, rebuild from the canonical recipe in [`configs/venvs/3.12.md`](configs/venvs/3.12.md) verbatim — do not subset to "what my current task imports."

- **Tier 1 — torchless unit tests** (`~/.venvs/acts_test_venv`, Python 3.10): pytest + pyyaml + pydantic, no torch. Default for deterministic / mocked tests.
  ```
  source ~/.venvs/acts_test_venv/bin/activate && python -m pytest tests/ -v
  ```
- **Tier 2 — SOL/torch integration + real-GPU tests** (`~/.venvs/acts_run_venv`, Python 3.12 + cu128 torch + editable `sol_execbench` + editable SOLAR + openai-agents + torchview): required for `@pytest.mark.gpu` suites, any test that imports SOL types or runs on GPU, and every live `python -m src.pipeline.optimize` run. Setup recipe + smoke test live in [`configs/venvs/3.12.md`](configs/venvs/3.12.md).
  ```
  source ~/.venvs/acts_run_venv/bin/activate && python -m pytest tests/ -v
  ```

Add new deps to `pyproject.toml` AND to whichever venv(s) the new tests target.

### Test delegation

- **TDD iteration stays inline.** The write-test → run → fail → write-code → run → pass loop needs fast turnaround on the same context; do not dispatch individual pytest runs to subagents during active iteration on a single module.
- **Delegate full-suite runs to a subagent.** Once a module's focused tests pass, the post-change `pytest tests/ -v` sweep goes to a subagent so the failure log + tracebacks don't flood the main context. The agent returns a short pass/fail summary with the specific failures.
- **Delegate Tier 2 GPU runs to a subagent.** `@pytest.mark.gpu` suites are long and log-heavy; always run via subagent with an explicit scope and a short-report instruction.

### Other delegation candidates

Beyond tests, three recurring subtasks in this workflow are read-heavy, have no user-interaction need, and return a short structured output — dispatch them to a subagent with an explicit scope + short-report instruction, same pattern as test delegation.

- **Upstream-repo reconnaissance** (pre-design, per the "Upstream reference repos" rule). Hand the subagent the target surface + the repo list from `reference_upstream_repos.md` auto-memory; it returns a punch list of the relevant files/patterns. Keeps the main context free of the full skim.
- **Post-change consistency sweep** (the "After any architectural change" rule). Grep/read fan-out across src/ ↔ doc/ ↔ PRD ↔ JOURNAL ↔ PROCESS. Brief the agent with the delta (renamed symbols, changed signatures, new/removed modules); it returns a list of stale references to fix.
- **Step 7 doc updates** when a change touches multiple `doc/*.md` files. Each doc update is an independent edit sharing one delta brief — dispatch in parallel per the "Parallel execution" gate (disjoint file sets, per-doc scope). Skip for single-doc changes; the cold-start cost isn't amortized.

Keep inline: step 1 (pick feature), step 2 (design discussion / brainstorming), step 8 (commit split), TDD iteration in steps 3–4, and `simplify` — all need either the user in the loop or the main conversation context.

### Doc mapping

See [`doc/README.md`](doc/README.md) for the per-component file index (config, kernels, eval, search, agents, llm_backend, actions, memory, pipeline, runtime). Updated as each feature lands; treat that file as canonical instead of duplicating the map here.

### Upstream reference repos

Before designing any new compile, correctness, benchmark, or search surface, skim the equivalent in at least one local upstream repo. The full list of paths + per-repo file pointers is in auto-memory at `reference_upstream_repos.md` (indexed in `MEMORY.md`). SOL-ExecBench is the canonical reference for timing/isolation; AccelOpt/Astra/autokernel are the pattern sources for compile/correctness/search.

### Don'ts

- Don't read PRD.md or JOURNAL.md in full unless the task requires it.
- Don't implement beyond the current skeleton interface without discussion.
- Don't change function signatures of (done) modules without consistency check.
- Don't add GPU-dependent logic to modules marked "Pure Python, no GPU."
