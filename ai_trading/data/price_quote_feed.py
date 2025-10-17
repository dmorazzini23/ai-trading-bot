"""Utilities for normalizing and caching Alpaca price quote feeds."""

from __future__ import annotations

from typing import Dict

_VALID_FEEDS = {"iex", "sip"}
_FEED_CACHE: Dict[str, str] = {}


def _normalize_feed(value: str | None) -> str | None:
    """Return a canonical Alpaca feed value when *value* is recognized."""

    if value is None:
        return None
    try:
        lowered = str(value).strip().lower()
    except Exception:  # pragma: no cover - defensive
        return None
    if not lowered:
        return None
    if lowered in _VALID_FEEDS:
        return lowered
    if lowered.startswith("alpaca_"):
        suffix = lowered.split("_", 1)[1]
        if suffix in _VALID_FEEDS:
            return suffix
    return None


def ensure_entitled_feed(requested: str | None, cached: str | None) -> str | None:
    """Return a usable feed based on the requested and cached values."""

    requested_norm = _normalize_feed(requested)
    if requested_norm in _VALID_FEEDS:
        return requested_norm
    cached_norm = _normalize_feed(cached)
    if cached_norm in _VALID_FEEDS:
        return cached_norm
    return None


def resolve(symbol: str, requested: str | None) -> str | None:
    """Resolve *requested* for *symbol* while updating the cache."""

    cached = _FEED_CACHE.get(symbol)
    resolved = ensure_entitled_feed(requested, cached)
    if resolved is None:
        _FEED_CACHE.pop(symbol, None)
        return None
    _FEED_CACHE[symbol] = resolved
    return resolved


def cache(symbol: str, feed: str | None) -> str | None:
    """Persist a sanitized feed for *symbol* and return the stored value."""

    resolved = ensure_entitled_feed(feed, None)
    if resolved is None:
        _FEED_CACHE.pop(symbol, None)
        return None
    _FEED_CACHE[symbol] = resolved
    return resolved


def get_cached(symbol: str) -> str | None:
    """Return the cached feed for *symbol* if present."""

    return _normalize_feed(_FEED_CACHE.get(symbol))


def clear(symbol: str | None = None) -> None:
    """Clear the feed cache for *symbol* or entirely when omitted."""

    if symbol is None:
        _FEED_CACHE.clear()
    else:
        _FEED_CACHE.pop(symbol, None)


__all__ = [
    "cache",
    "clear",
    "ensure_entitled_feed",
    "get_cached",
    "resolve",
]

