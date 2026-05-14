# Implementation — User Prompt Format

The user prompt is assembled programmatically by `CoderAgent.build_user_prompt()` from the current kernel source and the Planner's `OptimizationPlan`. Reviewer feedback is **not** included — the Planner already consumes it and distills its conclusions into the plan.

## Sections

```
## Current kernel
<kernel source code in a Python code block; triple-backticks in the source are escaped>

## Optimization plan
- Tier: <N>
- Technique: <technique_id>
- Params: <param>=<val>, <param>=<val>, ...
  (omitted when plan.params is empty)
- Target region: <region string>
- Rationale: <1-2 sentences from the Planner>
```

## Field provenance

| Section             | Source                                          |
|---------------------|-------------------------------------------------|
| Current kernel      | `Kernel.source_code` of the parent tree node    |
| Tier                | `OptimizationPlan.tier`                         |
| Technique           | `OptimizationPlan.technique`                    |
| Params              | `OptimizationPlan.params` (rendered `k=v, ...`) |
| Target region       | `OptimizationPlan.target_region`                |
| Rationale           | `OptimizationPlan.rationale`                    |

## Output contract

The Coder's response is validated against the `KernelCodeOutput` Pydantic schema:

- `source_code` (str): the complete modified kernel source.

Schema violations raise inside `run_agent` and are retried transparently. If all retries are exhausted, `CoderAgent.implement()` raises `ImplementationError` — the orchestrator surfaces this as a dead branch.

## Autotune (A1)

Every kernel you emit **MUST** be wrapped in `@triton.autotune` directly above the `@triton.jit` device function. The configs list must have **at least 4 entries** spanning a sensible region:

- Vary `BLOCK_*` dimensions in powers of two between **16 and 256**.
- Vary `num_warps` in **{2, 4, 8}**.
- Vary `num_stages` in **{2, 3, 4}**.
- The `key=` list must include every shape arg that affects performance (e.g., `["M", "N", "K"]` for matmul, `["N"]` for rowwise reductions).

Do not emit a single-config autotune — Triton's autotune is what closes the parameter-axis gap to vendor baselines; bypassing it is the dominant cause of regression vs the Triton baseline. If a specific config has known constraints (oversized shared memory, oversized blocks for tiny shapes), include it anyway and let Triton's compile-failure pruning drop it at runtime.

When the plan's `params` include a `recommended_configs` field, use those as the starting list for `@triton.autotune` — they reflect known-good starting points for this kernel type. You may add configs (e.g., to widen the sweep) but should not drop them without justification. (Note: `recommended_configs` plumbing arrives in PR 3; for now this field is always absent.)

Example shape:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def my_matmul_kernel(A, B, C, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```
