"""JSON-file storage backend for optimization memory."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.eval.types import BottleneckType
from src.memory.experience import ActionRecord, Experience

logger = logging.getLogger(__name__)

# Default when a legacy JSON record is missing a bottleneck field — matches
# ``Experience``'s dataclass default.
_DEFAULT_BOTTLENECK = BottleneckType.BALANCED.value


def _parse_bottleneck(value: str) -> BottleneckType:
    """Tolerant parse for legacy / malformed bottleneck strings.

    Pre-profiler-PR records persisted ``bottleneck_*`` as ``""`` (the old
    Experience dataclass default) — those fall through silently. Unknown
    tokens are schema drift (hand-edited file, version skew) and get
    logged before falling back to ``BALANCED`` so the signal doesn't
    disappear into an opaque default.
    """
    if not value:
        return BottleneckType.BALANCED
    try:
        return BottleneckType(value)
    except ValueError:
        logger.warning(
            "unknown bottleneck token %r in memory store — defaulting to BALANCED. "
            "Likely schema drift or a hand-edited record.",
            value,
        )
        return BottleneckType.BALANCED


class MemoryStore:
    """Persistent JSON storage for optimization experiences."""

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._experiences: list[Experience] = []

    def load(self) -> None:
        """Load experiences from disk."""
        if self._store_path.exists():
            data = json.loads(self._store_path.read_text())
            self._experiences = [
                Experience(
                    kernel_type=e["kernel_type"],
                    action_applied=ActionRecord(**e["action_applied"]),
                    metrics=e.get("metrics", {}),
                    speedup=e.get("speedup", 0.0),
                    reviewer_summary=e.get("reviewer_summary", ""),
                    bottleneck_before=_parse_bottleneck(
                        e.get("bottleneck_before", _DEFAULT_BOTTLENECK)
                    ),
                    hardware=e.get("hardware", ""),
                    success=e.get("success", False),
                )
                for e in data
            ]

    def save(self) -> None:
        """Persist all experiences to disk."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(
            json.dumps(
                [_experience_to_dict(e) for e in self._experiences], indent=2
            )
        )

    def add(self, experience: Experience) -> None:
        """Add a new experience and persist."""
        self._experiences.append(experience)
        self.save()

    def all(self) -> list[Experience]:
        """Return all stored experiences."""
        return list(self._experiences)


def _experience_to_dict(exp: Experience) -> dict:
    """Serialize an Experience to a JSON-compatible dict.

    ``dataclasses.asdict`` preserves the ``BottleneckType`` enum object,
    which is not JSON-encodable. We flatten the enum field to its
    string ``.value`` so the file stays human-readable and round-trips
    through ``BottleneckType(...)`` on load.
    """
    d = asdict(exp)
    d["bottleneck_before"] = exp.bottleneck_before.value
    return d
