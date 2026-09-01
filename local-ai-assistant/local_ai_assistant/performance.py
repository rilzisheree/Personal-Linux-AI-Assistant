"""Small latency traces for the native voice conversation pipeline."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any


LOGGER = logging.getLogger("lura.performance")


class LatencyTrace:
    """Record first-occurrence pipeline markers and emit one structured summary."""

    def __init__(self, label: str = "voice_conversation") -> None:
        self.label = label
        self.started_at = time.monotonic()
        self._marks: dict[str, float] = {}
        self._lock = threading.Lock()

    def mark(self, name: str) -> None:
        with self._lock:
            self._marks.setdefault(name, time.monotonic())

    def finish(self) -> dict[str, Any]:
        with self._lock:
            marks = dict(self._marks)
        elapsed = {
            name: round(timestamp - self.started_at, 3)
            for name, timestamp in marks.items()
        }
        summary: dict[str, Any] = {"label": self.label, "elapsed_seconds": elapsed}
        LOGGER.info("VOICE_LATENCY %s", json.dumps(summary, sort_keys=True))
        return summary