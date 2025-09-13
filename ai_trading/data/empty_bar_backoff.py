"""Helpers for tracking empty-bar fetch backoffs.

This module centralizes bookkeeping around symbols that should be skipped
because prior fetch attempts resulted in empty bar responses.  It exposes a
shared ``_SKIPPED_SYMBOLS`` set along with convenience helpers for recording
attempts and marking successful fetches.  Retry counts are tracked per
``(symbol, timeframe)`` pair and an :class:`EmptyBarsError` is raised once the
configured limit is exceeded.
"""
from __future__ import annotations

from ai_trading.config.management import MAX_EMPTY_RETRIES

# Symbols that have recently triggered empty-bar skips. Each entry is keyed by
# ``(symbol, timeframe)``.
_SKIPPED_SYMBOLS: set[tuple[str, str]] = set()
# Consecutive empty-bar attempt counters keyed by ``(symbol, timeframe)``.
_EMPTY_BAR_COUNTS: dict[tuple[str, str], int] = {}


def record_attempt(symbol: str, timeframe: str) -> int:
    """Record a fetch attempt for ``symbol``/``timeframe``.

    The pair is added to ``_SKIPPED_SYMBOLS`` so subsequent calls can detect
    in-flight or failed attempts.  The retry counter is incremented and returned.
    Once the counter exceeds :data:`MAX_EMPTY_RETRIES`, an
    :class:`~ai_trading.data.fetch.EmptyBarsError` is raised.
    """

    key = (symbol, timeframe)
    cnt = _EMPTY_BAR_COUNTS.get(key, 0) + 1
    _EMPTY_BAR_COUNTS[key] = cnt
    _SKIPPED_SYMBOLS.add(key)
    if cnt > MAX_EMPTY_RETRIES:
        from ai_trading.data.fetch import EmptyBarsError  # avoid circular import

        raise EmptyBarsError(
            f"empty_bars: symbol={symbol}, timeframe={timeframe}, max_retries={cnt}"
        )
    return cnt


def mark_success(symbol: str, timeframe: str) -> None:
    """Mark a previously attempted fetch as succeeded.

    Removes the pair from ``_SKIPPED_SYMBOLS`` and clears retry counters to
    allow future fetches.
    """

    key = (symbol, timeframe)
    _SKIPPED_SYMBOLS.discard(key)
    _EMPTY_BAR_COUNTS.pop(key, None)


__all__ = [
    "_SKIPPED_SYMBOLS",
    "_EMPTY_BAR_COUNTS",
    "record_attempt",
    "mark_success",
    "MAX_EMPTY_RETRIES",
]
