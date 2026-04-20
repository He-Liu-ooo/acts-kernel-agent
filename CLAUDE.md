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
2. **Design discussion** (if non-trivial) — align on approach before writing code. Use the brainstorming skill when multiple viable approaches exist.
3. **Write tests** — test-first; define expected behavior before implementation.
4. **Write code** — implement to pass the tests.
5. **Review** — user triggers the `codex:review` agent for an automated first pass, then asks Codex directly, then reviews themselves. Iterate on steps 3–4 until review passes.
6. **Simplify** — user triggers `simplify`.
7. **Update docs** — after review passes, verify consistency between src/ and:
   - **doc/** — record the feature's details in the relevant component file
   - **PROCESS.md** — mark feature complete
   - **JOURNAL.md** — record design rationale (if applicable)
   - **PRD.md**
8. **Commit** — user confirms, then commit and update **PROCESS.md** for the next round.

### Rules

- Keep PRs small — each change should be small enough to be reviewed efficiently.
- Do not commit until both Codex review and user review are complete.
- Do not skip the test-first step for deterministic modules (eval, search, memory, config). For LLM agent modules, mocked tests are acceptable.

### Design decisions

When multiple approaches exist, present options with tradeoffs. Wait for user to pick before implementing. Record the decision + rationale in JOURNAL.md.

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
