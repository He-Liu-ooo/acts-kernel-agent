"""Experience retrieval — kernel-type filtering + bottleneck matching."""

from __future__ import annotations

from src.memory.experience import Experience
from src.memory.store import MemoryStore


class MemoryRetriever:
    """Retrieves relevant past experiences for the Planner.

    Retrieval strategy:
        1. Filter by kernel type
        2. Rank by bottleneck relevance
        3. Select top-K (configurable, default 3-5)
    """

    def __init__(self, store: MemoryStore, top_k: int = 5) -> None:
        self._store = store
        self._top_k = top_k

    def retrieve(
        self,
        kernel_type: str,
        current_bottleneck: str,
    ) -> list[Experience]:
        """Retrieve the most relevant experiences for a planning step."""
        # Filter by kernel type
        typed = [e for e in self._store.all() if e.kernel_type == kernel_type]
        # Rank: exact bottleneck match first, then rest
        exact = [e for e in typed if e.bottleneck_before == current_bottleneck]
        rest = [e for e in typed if e.bottleneck_before != current_bottleneck]
        ranked = exact + rest
        return ranked[: self._top_k]
