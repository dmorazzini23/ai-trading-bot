from __future__ import annotations

from dataclasses import dataclass

from ai_trading.metrics import get_counter


@dataclass
class Metrics:
    """Simple in-memory counters for data fetch events."""

    rate_limit: int = 0
    timeout: int = 0
    unauthorized: int = 0
    empty_payload: int = 0


metrics = Metrics()

# Prometheus counter tracking backup provider usage
backup_provider_used = get_counter(
    "backup_provider_used_total",
    "Times backup data provider served data",
    ["provider", "symbol"],
)

__all__ = ["Metrics", "metrics", "backup_provider_used"]
