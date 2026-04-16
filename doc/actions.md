# Actions — `src/actions/`

Structured action library — 6-tier optimization actions.

## ActionRegistry

Complete catalog of all optimization actions, built once at startup by `build_default_registry()`. Contains 25 actions across 6 tiers.

The Planner does not search the registry directly. The orchestrator calls `list_applicable(kernel_type, bottleneck)` to filter by kernel type and bottleneck, then injects the filtered subset into the Planner's prompt context.

```
ActionRegistry (all 25 actions, static)
    → list_applicable() filters by kernel_type + bottleneck
        → filtered actions go into Planner's prompt
            → Planner picks one → OptimizationPlan
```

### Methods

- `register(action)`: Add an action.
- `get(action_id) -> Action`: Look up by ID.
- `list_by_tier(tier) -> list[Action]`: All actions in a tier.
- `list_applicable(kernel_type, bottleneck) -> list[Action]`: Filtered + sorted by tier.

## Action Record

Each action is a structured record:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (e.g., `t2_shared_memory_tiling`) |
| `tier` | Risk/reward tier (1-6) |
| `name` | Human-readable name |
| `description` | What the optimization does |
| `applicable_to` | Kernel types this applies to (empty = all) |
| `preconditions` | Conditions required (e.g., `memory_bound`) |
| `parameters` | Tunable knobs (e.g., `tile_size: "32-128"`) |
| `guidance` | Step-by-step recipe for the Coder (not code templates) |
| `anti_patterns` | Known mistakes to avoid |
| `expected_impact` | Expected improvement range |

## Tiers

| Tier | Focus | Risk | Count | Examples |
|------|-------|------|-------|----------|
| 1: Sizing | Block/grid tuning | Low | 3 | block_size_tuning, grid_shape, occupancy |
| 2: Memory | Memory optimization | Low-Med | 5 | shared_memory_tiling, coalescing, prefetching |
| 3: Compute | Compute optimization | Medium | 5 | tf32, mixed_precision, fused_ops, vectorized_loads |
| 4: Advanced | Structural changes | High | 4 | split_k, persistent_kernel, warp_spec, stream_k |
| 5: Arch-specific | Hardware intrinsics | High | 4 | h100_tma, h100_wgmma, a100_cp_async |
| 6: Kernel-specific | Algorithm tricks | High | 4 | welford, online_softmax, flash_attention |

Tiers are not strictly sequential — Planner can pick any tier, but ordering encodes risk/reward.
