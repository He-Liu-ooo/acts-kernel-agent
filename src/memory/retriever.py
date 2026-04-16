"""Experience retrieval — kernel-type filtering + bottleneck matching."""

from __future__ import annotations

from src.memory.experience import Experience
from src.memory.store import MemoryStore

# Scoring weights
_BOTTLENECK_MATCH = 10.0
_SUCCESS_BONUS = 3.0
_SPEEDUP_CAP = 5.0


def _score(exp: Experience, current_bottleneck: str) -> float:
    """Compute relevance score for a single experience."""
    s = 0.0
    if exp.bottleneck_before == current_bottleneck:
        s += _BOTTLENECK_MATCH
    if exp.success:
        s += _SUCCESS_BONUS
    s += min(exp.speedup, _SPEEDUP_CAP)
    return s


class MemoryRetriever:
    """Retrieves relevant past experiences for the Planner.

    Retrieval strategy:
        1. Filter by kernel type
        2. Filter by hardware (prefer same, fall back to cross-hardware)
        3. Score by bottleneck match + success + speedup
        4. Select top-K with reserved failure slots
    """

    def __init__(self, store: MemoryStore, top_k: int = 5) -> None:
        self._store = store
        self._top_k = top_k

    def retrieve(
        self,
        kernel_type: str,
        current_bottleneck: str,
        hardware: str = "",
    ) -> list[Experience]:
        """Retrieve the most relevant experiences for a planning step."""
        candidates = [e for e in self._store.all() if e.kernel_type == kernel_type]
        if not candidates:
            return []

        candidates = self._apply_hardware_filter(candidates, hardware)

        successes = [e for e in candidates if e.success]
        failures = [e for e in candidates if not e.success]

        key = lambda e: (_score(e, current_bottleneck), e.speedup)
        successes.sort(key=key, reverse=True)
        failures.sort(key=key, reverse=True)

        # Reserve failure slots only when top_k is large enough for both pools
        if self._top_k >= 3 and successes and failures:
            failure_slots = max(1, self._top_k // 3)
        elif self._top_k == 2 and successes and failures:
            failure_slots = 1
        else:
            failure_slots = 0
        picked_failures = failures[:failure_slots]
        remaining = self._top_k - len(picked_failures)
        picked_successes = successes[:remaining]

        # Backfill: if one pool was short, give slots to the other
        total = len(picked_successes) + len(picked_failures)
        if total < self._top_k:
            if len(picked_failures) < failure_slots:
                extra = self._top_k - total
                picked_successes = successes[: len(picked_successes) + extra]
            else:
                extra = self._top_k - total
                picked_failures = failures[: len(picked_failures) + extra]

        merged = picked_successes + picked_failures
        merged.sort(key=key, reverse=True)
        return merged[: self._top_k]

    def _apply_hardware_filter(
        self, candidates: list[Experience], hardware: str
    ) -> list[Experience]:
        """Prefer same-hardware experiences, fall back if too few."""
        if not hardware:
            return candidates
        same_hw = [e for e in candidates if e.hardware == hardware]
        if len(same_hw) >= self._top_k:
            return same_hw
        other = [e for e in candidates if e.hardware != hardware]
        return same_hw + other
