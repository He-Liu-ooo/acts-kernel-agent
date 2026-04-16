# ACTS Kernel Agent — Claude Code Instructions

## Session Start

New session. Read the following files in order:

1. [PRD.md](PRD.md) for project requirements, architecture, and constraints
2. [PROCESS.md](PROCESS.md) — specifically the Implementation Status section
3. If you want to review rationale behind certain design choices, refer to [JOURNAL.md](JOURNAL.md); if you want to know details about already implemented code, refer to [doc/](doc/README.md)

Confirm your understanding of the project state and tell me where we left off.

## Workflow

Each step is user-triggered — Claude does not auto-advance to the next step.

1. **Pick feature** — next item from the dependency-ordered list in PROCESS.md
2. **Design discussion** (if non-trivial) — align on approach before writing code. Use brainstorming skill if the feature has multiple viable approaches.
3. **Write tests** — test-first. Define expected behavior before implementation.
4. **Write code** — implement to pass the tests.
5. **Review** — user triggers `code-reviewer` agent for automated first-pass, then asks Codex to review, then reviews themselves. If changes are needed, iterate on steps 3-4 until review passes.
6. **Update docs** — after review passes:
   - **doc/** — record the feature's details in the relevant component file
   - **PROCESS.md** — mark feature complete
   - **JOURNAL.md** — record design rationale (if applicable)
7. **Commit** — user confirms, then commit.

### Rules

- Keep PRs small — each change should be small enough to be reviewed efficiently.
- Do not commit until both Codex review and user review are complete.
- Do not skip the test-first step for deterministic modules (eval, search, memory, config). For LLM agent modules, mocked tests are acceptable.
