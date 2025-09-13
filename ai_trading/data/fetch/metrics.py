"""Lightweight in-memory counters for data-fetch operations."""
from __future__ import annotations

from collections import defaultdict

from ai_trading.data.metrics import (
    backup_provider_used as _backup_provider_used_counter,
    metrics as _metrics,
    provider_disable_total as _provider_disable_total_counter,
    provider_fallback as _provider_fallback_counter,
)

_SKIPPED_SYMBOLS: dict[tuple[str, str], int] = {}
_RATE_LIMITS = defaultdict(int)
_TIMEOUTS = defaultdict(int)
_UNAUTH_SIP = defaultdict(int)
_EMPTY = defaultdict(int)


def mark_skipped(symbol: str, timeframe: str) -> None:
    key = (symbol, timeframe)
    _SKIPPED_SYMBOLS[key] = _SKIPPED_SYMBOLS.get(key, 0) + 1


def rate_limit(host: str) -> None:
    _RATE_LIMITS[host] += 1


def timeout(host: str) -> None:
    _TIMEOUTS[host] += 1


def unauthorized_sip(host: str) -> None:
    _UNAUTH_SIP[host] += 1


def empty_payload(symbol: str, timeframe: str) -> None:
    _EMPTY[(symbol, timeframe)] += 1


def _current_value(metric: object) -> int:
    """Return the current value of a prometheus metric."""
    value = getattr(metric, "_value", None)
    if value is None or not hasattr(value, "get"):
        return 0
    try:
        return int(value.get())
    except Exception:  # pragma: no cover - defensive
        return 0


def provider_fallback(from_provider: str, to_provider: str) -> int:
    """Increment fallback counter and return the current value."""
    metric = _provider_fallback_counter.labels(
        from_provider=from_provider, to_provider=to_provider
    )
    metric.inc()
    return _current_value(metric)


def backup_provider_used(provider: str, symbol: str) -> int:
    """Increment backup-provider counter and return the current value."""
    metric = _backup_provider_used_counter.labels(provider=provider, symbol=symbol)
    metric.inc()
    return _current_value(metric)


def provider_disable_total(provider: str) -> int:
    """Increment provider-disable counter and return the current value."""
    metric = _provider_disable_total_counter.labels(provider=provider)
    metric.inc()
    return _current_value(metric)


def snapshot(metrics_state: object = _metrics) -> dict[str, int]:
    """Return a snapshot of high level data-fetch metrics."""
    return {
        "rate_limit": getattr(metrics_state, "rate_limit", 0),
        "timeout": getattr(metrics_state, "timeout", 0),
        "unauthorized": getattr(metrics_state, "unauthorized", 0),
        "empty_payload": getattr(metrics_state, "empty_payload", 0),
        "feed_switch": getattr(metrics_state, "feed_switch", 0),
    }


def reset() -> None:
    """Reset all counters (test helper)."""
    _SKIPPED_SYMBOLS.clear()
    _RATE_LIMITS.clear()
    _TIMEOUTS.clear()
    _UNAUTH_SIP.clear()
    _EMPTY.clear()


__all__ = [
    "mark_skipped",
    "rate_limit",
    "timeout",
    "unauthorized_sip",
    "empty_payload",
    "provider_fallback",
    "backup_provider_used",
    "provider_disable_total",
    "snapshot",
    "reset",
    "_SKIPPED_SYMBOLS",
    "_RATE_LIMITS",
    "_TIMEOUTS",
    "_UNAUTH_SIP",
    "_EMPTY",
]

