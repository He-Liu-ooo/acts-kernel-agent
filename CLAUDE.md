# ACTS Kernel Agent — Claude Code Instructions

## Session Start

New session. Read the following files in order:

1. [PRD.md](PRD.md) for project requirements, architecture, and constraints
2. [PROCESS.md](PROCESS.md) — specifically the Implementation Status section
3. If you want to review rationale behind certain design choices, refer to [JOURNAL.md](JOURNAL.md); if you want to know details about already implemented code, refer to [doc/](doc/README.md)

Confirm your understanding of the project state and tell me where we left off.

## Workflow

- Keep PRs small — each change should be small enough to be reviewed efficiently.
- After implementing every new feature, update:
  - **doc/** — record the feature's details in the relevant component file
  - **PROCESS.md** — record implementation status
  - **JOURNAL.md** — record design rationale (if applicable)
