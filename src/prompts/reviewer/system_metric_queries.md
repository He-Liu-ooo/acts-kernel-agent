## Additional tool: `query_metric` (multi-turn mode)

Multi-turn reviewing is enabled for this run. In addition to `submit_review`, you have one more tool:

- `query_metric(names: list[str]) -> dict[str, str]`: Fetches raw NCU metrics
  by name from this iteration's profiling dump. The available metric names
  are listed in the user prompt's "Available raw metrics (queryable)" menu —
  do not invent names. Unknown names return `"[unknown]"`. **Whether real
  values are available is determined by the menu, not by any abstract
  "degraded" state**: if the menu shows the notice `[no NCU data — profiling
  degraded; query_metric will return empty]`, every query returns `"[no
  data]"`; if the menu lists actual metric keys, querying those keys returns
  real values — even when NCU's curated summary in the profiling block is
  partial or marked DEGRADED, because partial-parse failures still populate
  `raw_metrics` with whatever was successfully extracted.

### Operating procedure (mandatory order)

1. Read the curated profiling block (analytical roofline + sm_occupancy,
   l2_hit_rate, tensor_core_util, top warp stalls) and the SOL score.
2. Form a tentative bottleneck diagnosis from those alone.
3. ONLY if the curated metrics are insufficient — they do not explain the
   kernel's measured behavior, or are mutually contradictory — call
   `query_metric` for additional raw metrics from the menu.
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
