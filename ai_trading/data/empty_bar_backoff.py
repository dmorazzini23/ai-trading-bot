"""Helpers for tracking empty-bar fetch backoffs.

This module centralizes bookkeeping around symbols that should be skipped
because prior fetch attempts resulted in empty bar responses.  It exposes a
shared ``_SKIPPED_SYMBOLS`` set along with convenience helpers for recording
attempts and marking successful fetches.
"""
from __future__ import annotations

# Symbols that have recently triggered empty-bar skips.
# Each entry is keyed by ``(symbol, timeframe)``.
_SKIPPED_SYMBOLS: set[tuple[str, str]] = set()


def record_attempt(symbol: str, timeframe: str) -> None:
    """Record that a symbol/timeframe fetch attempt was made.

    The pair is added to ``_SKIPPED_SYMBOLS`` so subsequent calls can detect
    in-flight or failed attempts and avoid redundant network requests.
    """
    _SKIPPED_SYMBOLS.add((symbol, timeframe))


def mark_success(symbol: str, timeframe: str) -> None:
    """Mark a previously attempted fetch as succeeded.

    Removes the pair from ``_SKIPPED_SYMBOLS`` allowing future fetches.
    """
    _SKIPPED_SYMBOLS.discard((symbol, timeframe))
