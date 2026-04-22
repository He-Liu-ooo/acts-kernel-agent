"""Shared evaluation types.

Hosts primitive types (enums, lightweight dataclasses) that are imported
across multiple eval/memory/search modules. Keeping them in a leaf module
prevents the circular-import headaches that would arise from their
original home in ``src/eval/roofline.py`` once ``eval/profiler.py`` and
``memory/experience.py`` both need to type-check against them.
"""

from __future__ import annotations

from enum import Enum


class BottleneckType(Enum):
    """Kernel bottleneck classification from roofline model."""

    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    BALANCED = "balanced"
