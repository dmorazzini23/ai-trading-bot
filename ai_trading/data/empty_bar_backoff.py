"""Helpers for tracking empty-bar fetch backoffs.

This module centralizes bookkeeping around symbols that should be skipped
because prior fetch attempts resulted in empty bar responses.  It exposes a
shared ``_SKIPPED_SYMBOLS`` set along with convenience helpers for recording
attempts and marking successful fetches.  Retry counts are tracked per
``(symbol, timeframe)`` pair and an :class:`EmptyBarsError` is raised once the
configured limit is exceeded.
"""
from __future__ import annotations

import sys

from ai_trading.config.management import MAX_EMPTY_RETRIES

# Symbols that have recently triggered empty-bar skips. Each entry is keyed by
# ``(symbol, timeframe)``.
_SKIPPED_SYMBOLS: set[tuple[str, str]] = set()
# Consecutive empty-bar attempt counters keyed by ``(symbol, timeframe)``.
_EMPTY_BAR_COUNTS: dict[tuple[str, str], int] = {}
_EMPTY_BARS_ERROR_CLASS: type[Exception] | None = None
_EMPTY_BARS_ERROR_PROXIES: dict[tuple[int, ...], type[Exception]] = {}
_EMPTY_BARS_ERROR_HISTORY: list[type[Exception]] = []


def _get_empty_bars_error() -> type[Exception]:
    global _EMPTY_BARS_ERROR_CLASS
    fetch_mod = sys.modules.get("ai_trading.data.fetch")
    if fetch_mod is None:
        from ai_trading.data import fetch as fetch_mod  # type: ignore[no-redef]
    try:
        cls = getattr(fetch_mod, "EmptyBarsError")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError("ai_trading.data.fetch.EmptyBarsError unavailable") from exc
    if _EMPTY_BARS_ERROR_CLASS is not cls:
        _EMPTY_BARS_ERROR_CLASS = cls
    if cls not in _EMPTY_BARS_ERROR_HISTORY:
        _EMPTY_BARS_ERROR_HISTORY.append(cls)
    return cls


def _raise_empty_bars_error(message: str) -> None:
    current_cls = _get_empty_bars_error()
    candidates: list[type[Exception]] = list(_EMPTY_BARS_ERROR_HISTORY) or [current_cls]
    if current_cls not in candidates:
        candidates.append(current_cls)
    # Ensure uniqueness in case both pointers match
    unique_candidates = []
    seen_ids: set[int] = set()
    for cls in candidates:
        cls_id = id(cls)
        if cls_id in seen_ids:
            continue
        seen_ids.add(cls_id)
        unique_candidates.append(cls)
    for module in list(sys.modules.values()):
        try:
            candidate = getattr(module, "EmptyBarsError", None)
        except Exception:
            continue
        if not isinstance(candidate, type):
            continue
        if not issubclass(candidate, Exception):
            continue
        cls_id = id(candidate)
        if cls_id in seen_ids:
            continue
        seen_ids.add(cls_id)
        unique_candidates.append(candidate)
    if len(unique_candidates) == 1:
        raise unique_candidates[0](message)
    key = tuple(id(cls) for cls in unique_candidates)
    proxy = _EMPTY_BARS_ERROR_PROXIES.get(key)
    if proxy is None:
        proxy = type("EmptyBarsErrorProxy", tuple(unique_candidates), {})
        _EMPTY_BARS_ERROR_PROXIES[key] = proxy
    raise proxy(message)


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
    fetch_mod = sys.modules.get("ai_trading.data.fetch")
    if fetch_mod is not None:
        try:
            fetch_skipped = getattr(fetch_mod, "_SKIPPED_SYMBOLS", None)
            if isinstance(fetch_skipped, set):
                fetch_skipped.add(key)
        except Exception:
            pass
    if cnt > MAX_EMPTY_RETRIES:
        _raise_empty_bars_error(
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
    fetch_mod = sys.modules.get("ai_trading.data.fetch")
    if fetch_mod is not None:
        try:
            fetch_skipped = getattr(fetch_mod, "_SKIPPED_SYMBOLS", None)
            if isinstance(fetch_skipped, set):
                fetch_skipped.discard(key)
        except Exception:
            pass


__all__ = [
    "_SKIPPED_SYMBOLS",
    "_EMPTY_BAR_COUNTS",
    "record_attempt",
    "mark_success",
    "MAX_EMPTY_RETRIES",
]
