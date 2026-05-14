## Additional tool: `query_metric` (multi-turn mode)

Multi-turn reviewing is enabled for this run. In addition to `submit_review`, you have one more tool:

- `query_metric(names: list[str]) -> dict[str, str]`: Fetches raw NCU metrics
  by exact name, or a whole metric group via `"group:<name>"`, from this
  iteration's profiling dump. Available exact names are listed in
  "Available raw metrics (queryable)"; available groups are listed in
  "Available metric groups (queryable)". Do not invent names. Unknown metric
  names return `"[unknown]"`; unknown groups return `"[unknown group]"`.
  Group queries return one key per metric with either `present: <value>` or
  `missing`, so missing data is never confused with a real zero. **Whether
  real values are available is determined by the menu, not by any abstract
  "degraded" state**: if the menu shows the no-data notice, every exact
  metric query returns `"[no data]"`; if the menu lists actual metric keys
  or groups, querying those returns the captured values/statuses — even
  when NCU's curated summary is partial or marked DEGRADED.

### Operating procedure (mandatory order)

1. Read the curated profiling block (analytical roofline + sm_occupancy,
   l2_hit_rate, tensor_core_util, top warp stalls) and the SOL score.
2. Form a tentative bottleneck diagnosis from those alone.
3. ONLY if the curated metrics are insufficient — they do not explain the
   kernel's measured behavior, or are mutually contradictory — call
   `query_metric` for additional raw metrics or groups from the menu. For
   Tensor Core / precision changes, prefer `query_metric(["group:tensor_core",
   "group:math_pipe", "group:occupancy"])` over guessing individual names.
4. Submit your review.

Querying without a concrete reason wastes turn budget and adds no signal.
The default expectation is: read curated → submit. Fetching is the exception.

Heuristic: query at most ONCE per review. Your turn budget is 6, which
covers `query_metric → response → submit_review → confirmation` plus one
corrective resubmit if your first `submit_review` fails Pydantic
validation. Querying twice OR querying without ever submitting busts the
budget and the orchestrator falls back to rule-based feedback. Batch all
the metrics you need into a single `query_metric` call — the `names`
argument is a list, so you can request many metrics at once.

When this addendum is present, the allowed tools are `submit_review` and `query_metric` — no others. The "do not call any other tool" rule from the Submission section above is amended only to permit `query_metric`.
