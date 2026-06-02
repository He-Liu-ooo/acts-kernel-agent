"""Producer — turns improving (parent → child) edges into stored Experiences.

Owns the per-run pending-write buffer and the session-cap accounting.
G1 per-improving-edge + G3 one-extra at run end (baseline → best-of-run).
Cap reserves one slot for the G3 row; edges contend for ``cap - 1``.

See ``doc/specs/2026-05-24-optimization-memory-design.md`` §8 + §9.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from src.memory.experience import ActionRecord, Experience, _format_condition
from src.memory.store import KNOWN_VERSION, MemoryStore
from src.memory.summarizer import SummarizerAgent, SummarizerResult

logger = logging.getLogger(__name__)


class Producer:
    """Per-run experience producer; flushes a top-N buffer to the store at run end."""

    def __init__(
        self,
        store: MemoryStore,
        summarizer: SummarizerAgent,
        config,
        run_id: str,
        kernel_type: str,
    ) -> None:
        self._store = store
        self._summarizer = summarizer
        self._config = config
        self._run_id = run_id
        self._kernel_type = kernel_type
        # (ratio, experience) tuples — cap eviction sorts by ratio.
        self._edge_buffer: list[tuple[float, Experience]] = []
        self._g3_row: Experience | None = None

    async def consider(
        self,
        parent_node,
        child_node,
        action: ActionRecord,
        *,
        iter_no: int = 0,
        bottleneck=None,
    ) -> None:
        if not self._config.opt_mem_write_enabled:
            return
        if self._edge_cap() == 0 and not self._edge_buffer:
            return
        # ``compiled`` / ``correct`` defaults are True: real ``TreeNode``s
        # don't carry these attributes — but by the time the orchestrator
        # calls ``consider()`` the child has already been added to the
        # tree, which itself gates on successful compile + correctness.
        # Test stubs set the attrs explicitly to exercise the gates.
        if not getattr(child_node, "compiled", True):
            return
        if not getattr(child_node, "correct", True):
            return
        if parent_node.runtime_ms is None or child_node.runtime_ms is None:
            return
        if child_node.runtime_ms <= 0:
            return
        ratio = parent_node.runtime_ms / child_node.runtime_ms
        if ratio < self._config.opt_mem_min_improvement_ratio:
            return
        # Pre-check eviction: if the edge buffer is full and this ratio
        # would not beat the worst buffered ratio, the row is going to be
        # evicted by ``_buffer_append`` immediately after the LLM call
        # produces it — burn the LLM token cost for nothing. Reject
        # before the summarize call. The +1e-9 tolerance avoids a noisy
        # float-equality short-circuit on identical ratios.
        cap = self._edge_cap()
        if (
            cap > 0
            and len(self._edge_buffer) >= cap
            and ratio <= min(r for r, _ in self._edge_buffer) + 1e-9
        ):
            return
        result = await self._summarizer.summarize(
            parent_src=parent_node.kernel.source_code,
            child_src=child_node.kernel.source_code,
            speedup=ratio,
            action=action,
            iter_no=iter_no,
        )
        if result is None:
            return
        condition = _format_condition(bottleneck, action)
        exp = self._build_experience(
            parent_node_id=str(parent_node.id),
            child_node_id=str(child_node.id),
            scope="edge",
            speedup=ratio,
            action=action,
            summary=result,
            condition=condition,
        )
        self._buffer_append(ratio, exp)

    async def finalize(self, baseline_node, best_of_run_node, *, bottleneck=None) -> None:
        if not self._config.opt_mem_write_enabled:
            return
        # G3 reserves 1 slot from the total cap; ``cap == 0`` means no slot
        # is available for the run-scope row either. Without this gate the
        # docs' contract ("cap=0 disables all writes") would silently leak
        # one G3 row per run that achieved any cumulative improvement.
        if self._config.opt_mem_writes_per_session_cap < 1:
            return
        if baseline_node.runtime_ms is None or best_of_run_node.runtime_ms is None:
            return
        if best_of_run_node.runtime_ms <= 0:
            return
        ratio = baseline_node.runtime_ms / best_of_run_node.runtime_ms
        if ratio < self._config.opt_mem_min_improvement_ratio:
            return
        # Single-edge run: if the baseline → best edge was actually captured in
        # the buffer, the run-scope G3 would duplicate it — skip it. Keying on
        # buffer presence (not ``best.parent_id``) means a single-edge win that
        # produced no buffered edge (cap=1 → ``_edge_cap()==0``; or that edge's
        # summarize returned None / was cap-evicted) still writes its G3 row,
        # so the only lesson of the run is never silently dropped.
        edge_captured = any(
            e.provenance.get("parent_node_id") == str(baseline_node.id)
            and e.provenance.get("child_node_id") == str(best_of_run_node.id)
            for _, e in self._edge_buffer
        )
        if edge_captured:
            return
        result = await self._summarizer.summarize_run(
            baseline_src=baseline_node.kernel.source_code,
            best_src=best_of_run_node.kernel.source_code,
            cumulative_speedup=ratio,
            # G3 is the end-of-run flush — no live iter context. Use
            # iter_no=0, the convention baseline/translate use for
            # out-of-loop work, so the trace still buckets in usage.json.
            iter_no=0,
        )
        if result is None:
            return
        condition = _format_condition(bottleneck, None)
        self._g3_row = self._build_experience(
            parent_node_id=str(baseline_node.id),
            child_node_id=str(best_of_run_node.id),
            scope="run",
            speedup=ratio,
            # G3 rows have no single applied action — the trajectory is
            # cumulative. ``None`` matches the schema invariant.
            action=None,
            summary=result,
            condition=condition,
        )

    async def flush(self) -> int:
        rows: list[Experience] = [e for _, e in self._edge_buffer]
        if self._g3_row is not None:
            rows.append(self._g3_row)
        if not rows:
            return 0
        self._store.add_many(rows)
        n = len(rows)
        self._edge_buffer.clear()
        self._g3_row = None
        return n

    # --- internals -----------------------------------------------------

    def _edge_cap(self) -> int:
        """Slots available for edge rows (cap minus 1 reserved for G3)."""
        return max(0, self._config.opt_mem_writes_per_session_cap - 1)

    def _buffer_append(self, ratio: float, exp: Experience) -> None:
        self._edge_buffer.append((ratio, exp))
        cap = self._edge_cap()
        if len(self._edge_buffer) > cap:
            self._edge_buffer.sort(key=lambda t: t[0], reverse=True)
            self._edge_buffer = self._edge_buffer[:cap]

    def _build_experience(
        self,
        *,
        parent_node_id: str,
        child_node_id: str,
        scope: str,
        speedup: float,
        action: ActionRecord | None,
        summary: SummarizerResult,
        condition: str,
    ) -> Experience:
        # ``scope`` is part of the digest so G1 (per-edge) and G3
        # (baseline → best-of-run) rows do not collide when the best-of-
        # run node is itself a direct child of the baseline — the most
        # common shape for clean 1-iter wins. Without ``scope`` here,
        # both rows would land in the JSONL with identical row_ids,
        # silently breaking the documented row-identity contract under
        # any future dedup pass.
        digest = hashlib.sha256(
            f"{self._run_id}||{parent_node_id}||{child_node_id}||{scope}".encode()
        ).hexdigest()[:16]
        return Experience(
            row_id=f"r_{digest}",
            schema_version=KNOWN_VERSION,
            kernel_type=self._kernel_type,
            hardware_arch=getattr(self._config.hardware, "name", ""),
            scope=scope,  # type: ignore[arg-type]
            speedup=speedup,
            action_applied=action,
            title=summary.title,
            lesson=summary.lesson,
            snippet_before=summary.snippet_before,
            snippet_after=summary.snippet_after,
            provenance={
                "run_id": self._run_id,
                "parent_node_id": parent_node_id,
                "child_node_id": child_node_id,
                "summarizer_model": self._summarizer.model_name,
            },
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            condition=condition,
        )
