"""Lightweight in-memory counters for data-fetch operations."""
from __future__ import annotations

from collections import defaultdict

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
    "reset",
    "_SKIPPED_SYMBOLS",
    "_RATE_LIMITS",
    "_TIMEOUTS",
    "_UNAUTH_SIP",
    "_EMPTY",
]

