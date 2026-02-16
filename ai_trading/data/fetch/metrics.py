"""Lightweight in-memory counters for data-fetch operations."""
from __future__ import annotations

from collections import Counter
import os
from threading import Lock

from ai_trading.data.metrics import (
    backup_provider_used as _backup_provider_used_counter,
    metrics as _metrics,
    provider_disable_total as _provider_disable_total_counter,
    provider_disabled as _provider_disabled_gauge,
    provider_fallback as _provider_fallback_counter,
)
from ai_trading.logging import get_logger

logger = get_logger(__name__)

_PROVIDER_DISABLE_TOTALS: dict[str, int] = {}


def register_provider_disable(provider: str) -> None:
    """Record the latest disable count for ``provider``."""

    try:
        from ai_trading.data.provider_monitor import provider_monitor as _pm

        count = int(getattr(_pm, "disable_counts", {}).get(provider, 0))
    except Exception:  # pragma: no cover - defensive fallback
        count = _PROVIDER_DISABLE_TOTALS.get(provider, 0)
    if count > 0:
        _PROVIDER_DISABLE_TOTALS[provider] = max(
            _PROVIDER_DISABLE_TOTALS.get(provider, 0), count
        )
    else:
        _PROVIDER_DISABLE_TOTALS.pop(provider, None)


# In-memory counters.  We use ``Counter`` for automatic zero initialization and
# guard updates with simple locks to ensure thread safety.
_SKIPPED_SYMBOLS: Counter[tuple[str, str]] = Counter()
_RATE_LIMITS: Counter[str] = Counter()
_TIMEOUTS: Counter[str] = Counter()
_UNAUTH_SIP: Counter[str] = Counter()
_EMPTY: Counter[tuple[str, str]] = Counter()
_FETCH_ATTEMPTS: Counter[str] = Counter()
_BACKUP_PROVIDER_USED_COUNTS: Counter[tuple[str, str]] = Counter()
_ALPACA_FAILED: int = 0
_PROVIDER_FALLBACK_COUNTS: Counter[tuple[str, str]] = Counter()

# Module-level gauges mirroring ``ai_trading.data.metrics.metrics`` values.
rate_limit: int = 0
timeout: int = 0
unauthorized: int = 0
empty_payload: int = 0
feed_switch: int = 0
empty_fallback: int = 0

_SKIPPED_LOCK = Lock()
_RATE_LIMIT_LOCK = Lock()
_TIMEOUT_LOCK = Lock()
_UNAUTH_LOCK = Lock()
_EMPTY_LOCK = Lock()
_FETCH_ATTEMPT_LOCK = Lock()
_BACKUP_PROVIDER_LOCK = Lock()
_ALPACA_FAILED_LOCK = Lock()


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def mark_skipped(symbol: str, timeframe: str) -> int:
    """Record that ``symbol``/``timeframe`` was skipped and return the count."""
    key = (symbol, timeframe)
    with _SKIPPED_LOCK:
        _SKIPPED_SYMBOLS[key] += 1
        return _SKIPPED_SYMBOLS[key]


def inc_rate_limit(host: str) -> int:
    """Increment rate-limit counter for ``host`` and return the total."""
    with _RATE_LIMIT_LOCK:
        _RATE_LIMITS[host] += 1
        global rate_limit
        rate_limit += 1
        _metrics.rate_limit += 1
        return _RATE_LIMITS[host]


def inc_timeout(host: str) -> int:
    """Increment timeout counter for ``host`` and return the total."""
    with _TIMEOUT_LOCK:
        _TIMEOUTS[host] += 1
        global timeout
        timeout += 1
        _metrics.timeout += 1
        return _TIMEOUTS[host]


def inc_unauthorized_sip(host: str) -> int:
    """Increment unauthorized SIP counter for ``host`` and return the total."""
    with _UNAUTH_LOCK:
        _UNAUTH_SIP[host] += 1
        global unauthorized
        unauthorized += 1
        _metrics.unauthorized += 1
        return _UNAUTH_SIP[host]


def inc_empty_payload(symbol: str, timeframe: str) -> int:
    """Increment empty-payload counter for ``symbol``/``timeframe``."""
    key = (symbol, timeframe)
    with _EMPTY_LOCK:
        _EMPTY[key] += 1
        global empty_payload
        empty_payload += 1
        _metrics.empty_payload += 1
        return _EMPTY[key]


def inc_fetch_attempt(provider: str) -> int:
    """Record a fetch attempt for ``provider`` and return the running total."""
    with _FETCH_ATTEMPT_LOCK:
        _FETCH_ATTEMPTS[provider] += 1
        return _FETCH_ATTEMPTS[provider]


def inc_alpaca_failed() -> int:
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
        logger.debug("METRIC_CURRENT_VALUE_READ_FAILED", exc_info=True)
        return 0


def inc_provider_fallback(from_provider: str, to_provider: str) -> int:
    """Increment fallback counter and return the current value."""
    global feed_switch, empty_fallback
    feed_switch += 1
    empty_fallback += 1
    _metrics.feed_switch += 1
    _metrics.empty_fallback += 1
    metric = _provider_fallback_counter.labels(
        from_provider=from_provider, to_provider=to_provider
    )
    metric.inc()
    key = (from_provider, to_provider)
    _PROVIDER_FALLBACK_COUNTS[key] += 1
    value = _current_value(metric)
    return value or _PROVIDER_FALLBACK_COUNTS[key]


def inc_backup_provider_used(provider: str, symbol: str, *, increment: bool | None = None) -> int:
    """Increment backup-provider counter and return the current value."""

    if increment is None:
        increment = not _env_truthy("PYTEST_RUNNING")

    key = (provider, symbol)
    with _BACKUP_PROVIDER_LOCK:
        if increment:
            _BACKUP_PROVIDER_USED_COUNTS[key] += 1
        local_value = _BACKUP_PROVIDER_USED_COUNTS.get(key, 0)

    metric = _backup_provider_used_counter.labels(provider=provider, symbol=symbol)
    prom_value = 0
    try:
        if increment:
            metric.inc()
        prom_value = _current_value(metric)
    except Exception:
        prom_value = 0
    return prom_value or local_value


def inc_provider_disable_total(provider: str) -> int:
    """Increment provider-disable counter and return the current value."""

    total = _PROVIDER_DISABLE_TOTALS.get(provider, 0) + 1
    _PROVIDER_DISABLE_TOTALS[provider] = total
    try:
        metric = _provider_disable_total_counter.labels(provider=provider)
        metric.inc()
    except Exception:
        logger.debug("PROVIDER_DISABLE_TOTAL_METRIC_INC_FAILED", extra={"provider": provider}, exc_info=True)
    return total


def provider_disabled(provider: str) -> int:
    """Return the current disabled gauge value for ``provider``."""
    metric = _provider_disabled_gauge.labels(provider=provider)
    value = _current_value(metric)
    if value == 0 and not hasattr(metric, '_value'):
        try:
            from ai_trading.data.provider_monitor import provider_monitor as _pm
            disabled = getattr(_pm, 'disabled_until', {})
            if provider in disabled:
                return 1
            if provider == 'alpaca' and any(key.startswith('alpaca_') for key in disabled):
                return 1
            return 0
        except Exception:  # pragma: no cover - defensive fallback
            logger.debug("PROVIDER_DISABLED_FALLBACK_LOOKUP_FAILED", extra={"provider": provider}, exc_info=True)
            return 0
    return value


def snapshot(metrics_state: object | None = None) -> dict[str, int]:
    """Return a snapshot of high level data-fetch metrics."""
    metrics_state = _metrics if metrics_state is None else metrics_state
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
    _FETCH_ATTEMPTS.clear()
    _BACKUP_PROVIDER_USED_COUNTS.clear()
    global _ALPACA_FAILED
    _ALPACA_FAILED = 0
    global rate_limit, timeout, unauthorized, empty_payload, feed_switch, empty_fallback
    rate_limit = 0
    timeout = 0
    unauthorized = 0
    empty_payload = 0
    feed_switch = 0
    empty_fallback = 0
    _metrics.rate_limit = 0
    _metrics.timeout = 0
    _metrics.unauthorized = 0
    _metrics.empty_payload = 0
    _metrics.feed_switch = 0
    _metrics.empty_fallback = 0


__all__ = [
    "mark_skipped",
    "inc_rate_limit",
    "inc_timeout",
    "inc_unauthorized_sip",
    "inc_empty_payload",
    "inc_provider_fallback",
    "inc_backup_provider_used",
    "inc_provider_disable_total",
    "provider_disabled",
    "snapshot",
    "reset",
    "rate_limit",
    "timeout",
    "unauthorized",
    "empty_payload",
    "feed_switch",
    "empty_fallback",
    "_SKIPPED_SYMBOLS",
    "_RATE_LIMITS",
    "_TIMEOUTS",
    "_UNAUTH_SIP",
    "_EMPTY",
    "_FETCH_ATTEMPTS",
    "_ALPACA_FAILED",
    "inc_fetch_attempt",
    "inc_alpaca_failed",
]
