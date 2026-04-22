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
8. **Commit** — propose the commit split, discuss with the user, wait for approval, then commit and update **PROCESS.md** for the next round.

### Rules

- Keep PRs small — each change should be small enough to be reviewed efficiently.
- Do not commit until both Codex review and user review are complete.
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

### Doc mapping

- `doc/eval.md` — eval harness (correctness, benchmark, profiler, roofline, scorer)
- `doc/config.md` — HardwareSpec, ACTSConfig, load paths
- `doc/search.md` — tree, beam, orchestrator
- `doc/memory.md` — experience, store, retriever
- `doc/pipeline.md` — optimize, verify, report

### Upstream reference repos

Before designing any new compile, correctness, benchmark, or search surface, skim the equivalent in at least one local upstream repo. The full list of paths + per-repo file pointers is in auto-memory at `reference_upstream_repos.md` (indexed in `MEMORY.md`). SOL-ExecBench is the canonical reference for timing/isolation; AccelOpt/Astra/autokernel are the pattern sources for compile/correctness/search.

### Don'ts

- Don't read PRD.md or JOURNAL.md in full unless the task requires it.
- Don't implement beyond the current skeleton interface without discussion.
- Don't change function signatures of (done) modules without consistency check.
- Don't add GPU-dependent logic to modules marked "Pure Python, no GPU."
