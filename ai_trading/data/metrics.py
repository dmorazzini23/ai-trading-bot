from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    """Simple in-memory counters for data fetch events."""

    rate_limit: int = 0
    timeout: int = 0
    unauthorized: int = 0
    empty_payload: int = 0


metrics = Metrics()

__all__ = ["Metrics", "metrics"]
