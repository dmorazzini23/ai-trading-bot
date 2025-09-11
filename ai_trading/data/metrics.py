from __future__ import annotations

from dataclasses import dataclass

from ai_trading.metrics import get_counter, get_gauge


@dataclass
class Metrics:
    """Simple in-memory counters for data fetch events."""

    rate_limit: int = 0
    timeout: int = 0
    unauthorized: int = 0
    empty_payload: int = 0
    feed_switch: int = 0
    empty_fallback: int = 0


metrics = Metrics()

# Prometheus counter tracking backup provider usage
backup_provider_used = get_counter(
    "backup_provider_used_total",
    "Times backup data provider served data",
    ["provider", "symbol"],
)

# Prometheus counter tracking provider fallback events
provider_fallback = get_counter(
    "data_provider_fallback_total",
    "Count of data provider fallbacks",
    ["from_provider", "to_provider"],
)

# Gauge indicating whether a primary provider is currently disabled
provider_disabled = get_gauge(
    "data_provider_disabled",
    "Flag set to 1 when a data provider is disabled",
    ["provider"],
)

__all__ = [
    "Metrics",
    "metrics",
    "backup_provider_used",
    "provider_fallback",
    "provider_disabled",
]
