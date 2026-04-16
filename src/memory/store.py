"""JSON-file storage backend for optimization memory."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.memory.experience import ActionRecord, Experience


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
                    bottleneck_before=e.get("bottleneck_before", ""),
                    bottleneck_after=e.get("bottleneck_after", ""),
                    hardware=e.get("hardware", ""),
                    success=e.get("success", False),
                )
                for e in data
            ]

    def save(self) -> None:
        """Persist all experiences to disk."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(
            json.dumps([asdict(e) for e in self._experiences], indent=2)
        )

    def add(self, experience: Experience) -> None:
        """Add a new experience and persist."""
        self._experiences.append(experience)
        self.save()

    def all(self) -> list[Experience]:
        """Return all stored experiences."""
        return list(self._experiences)
