"""Utilities for normalizing and caching Alpaca price quote feeds."""

from __future__ import annotations

import os
import sys
from typing import Dict

from ai_trading.config.management import get_env

_VALID_FEEDS = {"iex", "sip"}
_FEED_CACHE: Dict[str, str] = {}
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


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


def _pytest_mode() -> bool:
    return "pytest" in sys.modules or str(os.getenv("PYTEST_RUNNING", "")).strip().lower() in _TRUTHY


def _env_false(key: str) -> bool:
    try:
        value = get_env(key, None)
    except Exception:
        value = None
    if value is None:
        value = os.getenv(key)
    if value is None:
        return False
    try:
        lowered = str(value).strip().lower()
    except Exception:  # pragma: no cover - defensive
        return False
    return lowered in _FALSY


def _sip_disabled_env(py_mode: bool) -> bool:
    """Return ``True`` when environment flags explicitly disable SIP access."""

    if _env_false("ALPACA_ALLOW_SIP") or _env_false("ALPACA_SIP_ENTITLED"):
        return True
    if py_mode:
        return False
    return False


def ensure_entitled_feed(requested: str | None, cached: str | None) -> str | None:
    """Return a usable feed based on the requested and cached values."""

    requested_norm = _normalize_feed(requested)
    cached_norm = _normalize_feed(cached)
    candidates: list[str] = []
    if requested_norm in _VALID_FEEDS:
        candidates.append(requested_norm)
    if cached_norm in _VALID_FEEDS and cached_norm not in candidates:
        candidates.append(cached_norm)

    py_mode = _pytest_mode()
    sip_env_blocked = _sip_disabled_env(py_mode)
    sip_unauthorized = _sip_unauthorized()
    sip_blocked = sip_env_blocked or sip_unauthorized

    if requested_norm == "sip":
        return "sip" if not sip_blocked else "iex"
    if requested_norm == "iex":
        return "iex"

    if not sip_blocked and "sip" in candidates:
        return "sip"

    if "iex" in candidates:
        return "iex"

    if not candidates:
        return "iex" if sip_blocked else "sip"

    for candidate in candidates:
        if candidate == "sip":
            return "sip" if not sip_blocked else "iex"
        if candidate == "iex":
            return "iex"

    return "iex" if sip_blocked else "sip"


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


def _sip_unauthorized() -> bool:
    try:
        from ai_trading.data import fetch as data_fetcher  # local import to avoid cycles
    except Exception:
        return False

    state = getattr(data_fetcher, "_state", {})
    unauthorized_state = False
    if isinstance(state, dict):
        unauthorized_state = bool(state.get("sip_unauthorized"))
    if not unauthorized_state:
        unauthorized_state = bool(getattr(data_fetcher, "_SIP_UNAUTHORIZED", False))
    return unauthorized_state


def _apply_sip_guard(feed: str) -> str | None:
    if feed != "sip":
        return feed
    if _sip_unauthorized() or _sip_disabled_env(_pytest_mode()):
        return "iex"
    return "sip"


def _fallback_feed() -> str:
    return "iex" if (_sip_unauthorized() or _sip_disabled_env(_pytest_mode())) else "sip"


__all__ = [
    "cache",
    "clear",
    "ensure_entitled_feed",
    "get_cached",
    "resolve",
]
