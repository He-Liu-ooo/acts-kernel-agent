# ACTS Kernel Agent — Claude Code Instructions

## Session Start

1. [PROCESS.md](PROCESS.md) — current status and next feature
2. Read doc/ files only for the specific module you're about to work on
3. [PRD.md](PRD.md) — only if the task touches architecture or requirements
4. [JOURNAL.md](JOURNAL.md) — only if you need rationale for a past decision

Confirm your understanding of the project state and tell me where we left off.

## Workflow

Each step is user-triggered — Claude does not auto-advance.

1. **Pick feature** — next item from the dependency-ordered list in PROCESS.md.
2. **Design discussion** (if non-trivial) — align on approach before writing code. For any feature that is not mechanical (new module, new data surface, cross-module refactor, anything touching GPU / eval / search / agent contracts), **invoke the `superpowers:brainstorming` skill** via the `Skill` tool before proposing an approach. Default to brainstorming; skip it only for trivially mechanical changes (typo fix, single-call-site rename, one-line bug fix). The settled design + rationale gets recorded in JOURNAL.md before any code is written.
3. **Write tests** — test-first; define expected behavior before implementation.
4. **Write code** — implement to pass the tests.
5. **Review** — user triggers the `codex:review` agent for an automated first pass, then asks Codex directly, then reviews themselves. Iterate on steps 3–4 until review passes.
6. **Simplify** — user triggers `simplify`.
7. **Update docs** — after review passes, verify consistency between src/ and:
   - **doc/** — record the feature's details in the relevant component file
   - **PROCESS.md** — mark feature complete
   - **JOURNAL.md** — record design rationale (if applicable)
   - **PRD.md**

   **At the start of this step, default to parallel subagent dispatch.** The governance files above (PRD / JOURNAL / PROCESS / doc/*) are almost always disjoint from each other — each agent touches one file with the same delta brief, no data dependencies. One agent per file, dispatched in a single message. Reserve inline edits for the case where only one file truly changes, or for a small touch-up after the main fan-out. If the consistency sweep itself is still needed (the read side — scanning for stale references), run that as a single read-only subagent first, then fan out the write side based on its punch list. See "Parallel execution" below for the disjoint-file-set gate.

   **Retire design specs and implementation plans during this step.** If the feature produced files under `doc/specs/` or `doc/plans/` (the authoring artifacts from `superpowers:brainstorming` / `superpowers:writing-plans`), they're process exhaust once the code lands — git log + the governance docs tell the ongoing story better. Before committing: section-by-section, fold any content that isn't already in JOURNAL / PRD / PROCESS / doc/* into the appropriate file (typical candidates: explicit non-goals list → JOURNAL; operator-visible failure matrix → doc/<module>.md; unique architectural rationale → JOURNAL). Then delete the spec + plan, strip any cross-references to their paths from the remaining docs, and remove the now-empty `doc/specs/` / `doc/plans/` directories if nothing else lives there. Exception: if the feature is multi-phase and later phases will extend the spec, keep the spec until the series completes.
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

### After any architectural change

Run a consistency check across src/, doc/, PRD.md, JOURNAL.md, and PROCESS.md. Verify that terminology, function signatures, and data flow descriptions match the actual code. Fix stale references before committing.

### Test Environment

Tests run via: `source /tmp/acts_test_venv/bin/activate && python -m pytest tests/ -v`

Venv has: pytest, pyyaml. Add new deps to both pyproject.toml AND the venv.

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

- `doc/eval.md` — eval harness (correctness, benchmark, profiler, roofline, scorer)
- `doc/config.md` — HardwareSpec, ACTSConfig, load paths
- `doc/search.md` — tree, beam, orchestrator
- `doc/memory.md` — experience, store, retriever
- `doc/pipeline.md` — optimize, verify, report
- `doc/runtime.md` — run context, events stream, timestamp helpers (src/runtime/)

### Upstream reference repos

Before designing any new compile, correctness, benchmark, or search surface, skim the equivalent in at least one local upstream repo. The full list of paths + per-repo file pointers is in auto-memory at `reference_upstream_repos.md` (indexed in `MEMORY.md`). SOL-ExecBench is the canonical reference for timing/isolation; AccelOpt/Astra/autokernel are the pattern sources for compile/correctness/search.

### Don'ts

- Don't read PRD.md or JOURNAL.md in full unless the task requires it.
- Don't implement beyond the current skeleton interface without discussion.
- Don't change function signatures of (done) modules without consistency check.
- Don't add GPU-dependent logic to modules marked "Pure Python, no GPU."
