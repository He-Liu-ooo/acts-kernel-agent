# ACTS — Research Goals

## Primary research target

**Derive an agent-system design methodology**: given a fixed resource budget,
how should the tradeoffs across the agents in a multi-agent system be made to
achieve optimal outputs? The goal is prescriptive — to teach practitioners how
to build an agent system once they know their budget.

The kernel-optimization setting (ACTS: Planner → Coder → Reviewer over a search
tree) is the empirical vehicle. The methodology should generalize beyond
kernels: the contribution is "given budget B, here is how to think about
allocating it across roles in your agent system," not "here is the best
ACTS configuration."

## Why this framing

Existing agent-system papers typically report a single operating point
("our system uses N LLM calls and achieves X"). They do not characterize:

- How performance degrades as the budget shrinks
- How performance shifts when budget is reallocated between roles at fixed total
- Whether the optimal allocation depends on the input class (e.g.,
  memory-bound vs. compute-bound kernels)
- Whether there is a transferable rule of thumb a practitioner can apply
  before running their own sweep

A methodology that answers these — even partially — is what the field is
missing.

## Resource axes under consideration

The methodology teaches practitioners how to split budget *across agents*. The
right primary distinction is therefore between **per-agent allocatable**
resources (the actual tradeoff levers) and **system-level constraints** (the
total under which the allocation is optimized). The first experiment fixes
one system-level constraint and sweeps a per-agent allocation.

### Per-agent allocatable — the tradeoff space

These are the knobs a practitioner turns to split budget across roles. They
are the methodology's primary design surface.

1. **LLM calls per agent** — how many invocations Planner / Coder / Reviewer
   each receive within a run. ACTS today: `max_depth` × per-agent participation
   per iteration; reallocation is straightforward.
2. **Context tokens per agent** — how much each agent sees in its prompt.
   A core agent-system design choice ("what does this role need to know?"),
   not a slippery parameter — the implementation friction in ACTS is an
   experiment concern, not a methodology concern.
3. **Turns per agent** (`max_turns`) — how long each agent can deliberate
   per invocation. Distinct from total calls: a Reviewer with 6 turns
   thinks differently than one with 2, even at the same call count.
4. **Retries per agent** — when an agent has a verifiable submission
   (Coder's compile / correctness gate). Captures "how forgiving is this
   role's success criterion?"
5. **Retrieval depth per agent** — if the agent does RAG / memory lookup
   (`optimization_memory_top_k` in ACTS). Narrow but real lever.

### System-level constraints — the total to allocate under

These are the totals a practitioner is given (or chooses as the deployment
target). The methodology's job is to allocate the per-agent budgets above
optimally *given* one of these constraints.

1. **Total LLM calls** — scalar, cleanest experimental constraint.
2. **Total cumulative tokens (input + output)** — closer to actual $$ than
   call count; matters when call sizes vary dramatically across roles.
3. **Total verifier / oracle calls** — count of external-verification cycles.
   ACTS's GPU evaluation budget is one instance; other agent systems
   instantiate this as human-in-the-loop checks, external graders, or API
   verifiers. Framing it generally is what lets the methodology travel
   beyond kernels.
4. **Total exploration budget** — tree-node / candidate count. Conditional:
   only applies when the agent system is search-shaped. ACTS is; many agent
   systems aren't. The methodology should flag this as system-shape-dependent
   rather than universal.
5. **Total wall-clock** — operator-facing, deployment-realistic, but a
   *consequence* of the others rather than a clean design lever.

### First-experiment shape

Fix a system-level constraint (start with **total LLM calls**) and sweep a
per-agent allocation across Planner / Coder / Reviewer. This is the simplest
possible instantiation of the methodology question.

## Open design questions

Recorded for the brainstorming pass that produces the first experiment spec:

- Which experiment shape best serves the methodology goal?
  - (A) ACTS-vs-baseline at fixed small budgets
  - (B) Fix total, vary split across Planner / Coder / Reviewer (Pareto)
  - (C) Factorial: problem class × allocation, look for interaction effects
  - (D) Diminishing-returns curve on one problem
- Which baseline(s) does the methodology need to beat or contextualize against?
  (e.g., one-shot Coder, no-Reviewer, no-Retriever ablations)
- What does "optimal output" mean for the methodology paper —
  best SOL-score? best speedup? area under the budget–score curve?
- Which problem set covers enough diversity to support a generalization
  claim without exploding cost? (SOL-ExecBench subset, balanced across
  memory-bound / compute-bound / mixed?)

## Status

Brainstorming in progress. Next step: pick the experiment shape, then write
the experiment spec under `doc/specs/`.
