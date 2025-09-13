"""Lightweight in-memory counters for data-fetch operations."""
from __future__ import annotations

from collections import Counter
from threading import Lock

from ai_trading.data.metrics import (
    backup_provider_used as _backup_provider_used_counter,
    metrics as _metrics,
    provider_disable_total as _provider_disable_total_counter,
    provider_fallback as _provider_fallback_counter,
)

# In-memory counters.  We use ``Counter`` for automatic zero initialization and
# guard updates with simple locks to ensure thread safety.
_SKIPPED_SYMBOLS: Counter[tuple[str, str]] = Counter()
_RATE_LIMITS: Counter[str] = Counter()
_TIMEOUTS: Counter[str] = Counter()
_UNAUTH_SIP: Counter[str] = Counter()
_EMPTY: Counter[tuple[str, str]] = Counter()
_FETCH_ATTEMPTS: Counter[str] = Counter()
_ALPACA_FAILED: int = 0

_SKIPPED_LOCK = Lock()
_RATE_LIMIT_LOCK = Lock()
_TIMEOUT_LOCK = Lock()
_UNAUTH_LOCK = Lock()
_EMPTY_LOCK = Lock()
_FETCH_ATTEMPT_LOCK = Lock()
_ALPACA_FAILED_LOCK = Lock()


def mark_skipped(symbol: str, timeframe: str) -> int:
    """Record that ``symbol``/``timeframe`` was skipped and return the count."""
    key = (symbol, timeframe)
    with _SKIPPED_LOCK:
        _SKIPPED_SYMBOLS[key] += 1
        return _SKIPPED_SYMBOLS[key]


def rate_limit(host: str) -> int:
    """Increment rate-limit counter for ``host`` and return the total."""
    with _RATE_LIMIT_LOCK:
        _RATE_LIMITS[host] += 1
        return _RATE_LIMITS[host]


def timeout(host: str) -> int:
    """Increment timeout counter for ``host`` and return the total."""
    with _TIMEOUT_LOCK:
        _TIMEOUTS[host] += 1
        return _TIMEOUTS[host]


def unauthorized_sip(host: str) -> int:
    """Increment unauthorized SIP counter for ``host`` and return the total."""
    with _UNAUTH_LOCK:
        _UNAUTH_SIP[host] += 1
        return _UNAUTH_SIP[host]


def empty_payload(symbol: str, timeframe: str) -> int:
    """Increment empty-payload counter for ``symbol``/``timeframe``."""
    key = (symbol, timeframe)
    with _EMPTY_LOCK:
        _EMPTY[key] += 1
        return _EMPTY[key]


def fetch_attempt(provider: str) -> int:
    """Record a fetch attempt for ``provider`` and return the running total."""
    with _FETCH_ATTEMPT_LOCK:
        _FETCH_ATTEMPTS[provider] += 1
        return _FETCH_ATTEMPTS[provider]


def alpaca_failed() -> int:
    """Increment and return the Alpaca failure count."""
    global _ALPACA_FAILED
    with _ALPACA_FAILED_LOCK:
        _ALPACA_FAILED += 1
        return _ALPACA_FAILED


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
    "_FETCH_ATTEMPTS",
    "_ALPACA_FAILED",
    "fetch_attempt",
    "alpaca_failed",
]

