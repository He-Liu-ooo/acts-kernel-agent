"""Experience retriever — kernel-type + hardware-preferred + speedup-weighted sample.

See ``doc/specs/2026-05-24-optimization-memory-design.md`` §6 + §10.
"""

from __future__ import annotations

import random
from typing import Protocol

from src.memory.experience import Experience


class _StoreLike(Protocol):
    def all(self) -> list[Experience]: ...


class MemoryRetriever:
    """Samples relevant past experiences for the Planner.

    Algorithm:
        1. ``read_enabled`` is False → return ``[]``.
        2. Filter by ``kernel_type``.
        3. Hardware-preferred sampling: prefer ``hardware_arch == current``.
           - same-arch count >= ``top_k`` → weight-sample ``top_k`` from same-arch.
           - 0 < same-arch count < ``top_k`` → keep ALL same-arch (guaranteed
             inclusion) and weight-sample the remaining ``top_k - len(same)``
             slots from the FULL cross-arch pool.
           - no cross-arch fill available → return whatever fits.
        4. Weighting: ``random.choices(pool, weights=[s.speedup ** alpha], k=...)``.
           Sampling is with replacement.
    """

    def __init__(
        self,
        store: _StoreLike,
        top_k: int,
        alpha: float,
        read_enabled: bool,
        rng: random.Random | None = None,
    ) -> None:
        self._store = store
        self._top_k = top_k
        self._alpha = alpha
        self._read_enabled = read_enabled
        self._rng = rng or random.Random()

    def sample(self, kernel_type: str, hardware_arch: str) -> list[Experience]:
        if not self._read_enabled:
            return []
        candidates = [e for e in self._store.all() if e.kernel_type == kernel_type]
        if not candidates:
            return []
        if hardware_arch:
            same = [e for e in candidates if e.hardware_arch == hardware_arch]
            other = [e for e in candidates if e.hardware_arch != hardware_arch]
        else:
            same, other = candidates, []
        if len(same) >= self._top_k:
            return self._weighted_sample(same, self._top_k)
        # Fallback: keep ALL same-arch (guaranteed inclusion — the whole
        # point of the preference), then weight-sample the remaining slots
        # from the FULL cross-arch pool. Truncating ``other`` to first-N by
        # storage order before sampling would defeat speedup-weighting on
        # the fill — the regression this guards against.
        remaining = self._top_k - len(same)
        if len(other) <= remaining:
            return same + other
        return same + self._weighted_sample(other, remaining)

    def _weighted_sample(self, pool: list[Experience], k: int) -> list[Experience]:
        """Speedup-weighted random sample with replacement (``random.choices``).

        Returns the whole pool unsampled when it does not exceed ``k``.
        """
        if len(pool) <= k:
            return pool
        weights = [e.speedup ** self._alpha for e in pool]
        return self._rng.choices(pool, weights=weights, k=k)
