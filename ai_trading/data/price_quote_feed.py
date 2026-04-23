"""Utilities for normalizing and caching Alpaca price quote feeds."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import sys
from typing import Dict, Literal

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.utils.env import resolve_alpaca_reference_feed
from ai_trading.utils.env import resolve_alpaca_feed

FeedRole = Literal["execution", "reference"]

_VALID_EXEC_FEEDS = {"iex", "sip"}
_VALID_REFERENCE_FEEDS = {"iex", "sip", "delayed_sip"}
_FEED_CACHE_BY_ROLE: Dict[str, Dict[str, str]] = {
    "execution": {},
    "reference": {},
}
_TRUTHY = {"1", "true", "yes", "on"}
logger = get_logger(__name__)


def _normalize_role(role: FeedRole | str) -> FeedRole:
    return "reference" if str(role).strip().lower() == "reference" else "execution"


def _normalize_feed(value: str | None, *, role: FeedRole = "execution") -> str | None:
    """Return a canonical Alpaca feed value when *value* is recognized."""

    if value is None:
        return None
    valid_feeds = _VALID_REFERENCE_FEEDS if role == "reference" else _VALID_EXEC_FEEDS
    try:
        lowered = str(value).strip().lower()
    except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - defensive
        logger.debug("FEED_VALUE_NORMALIZE_FAILED", extra={"value": value}, exc_info=True)
        return None
    if not lowered:
        return None
    if lowered in valid_feeds:
        return lowered
    if lowered.startswith("alpaca_"):
        suffix = lowered.split("_", 1)[1]
        if suffix in valid_feeds:
            return suffix
    if role == "reference" and lowered in {"delayed", "delayed-sip", "dsip"}:
        return "delayed_sip"
    return None


def _pytest_mode() -> bool:
    return "pytest" in sys.modules or str(
        get_env("PYTEST_RUNNING", "", cast=str, resolve_aliases=False)
    ).strip().lower() in _TRUTHY


def _sip_disabled_env(py_mode: bool) -> bool:
    """Return ``True`` when environment flags disallow SIP access."""

    _ = py_mode
    try:
        resolved = resolve_alpaca_feed("sip")
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("SIP_DISABLED_ENV_RESOLVE_FAILED", exc_info=True)
        return True
    return str(resolved).strip().lower() != "sip"


def ensure_entitled_feed(
    requested: str | None,
    cached: str | None,
    *,
    role: FeedRole | str = "execution",
) -> str | None:
    """Return a usable feed based on the requested and cached values."""
    normalized_role = _normalize_role(role)
    requested_norm = _normalize_feed(requested, role=normalized_role)
    cached_norm = _normalize_feed(cached, role=normalized_role)
    if normalized_role == "reference":
        if requested_norm == "delayed_sip":
            return "delayed_sip"
        if requested_norm == "sip":
            resolved = str(resolve_alpaca_feed("sip") or "").strip().lower()
            return "sip" if resolved == "sip" else "delayed_sip"
        if requested_norm == "iex":
            return "iex"
        if cached_norm in _VALID_REFERENCE_FEEDS:
            return cached_norm
        return resolve_alpaca_reference_feed(None)

    py_mode = _pytest_mode()
    sip_env_blocked = _sip_disabled_env(py_mode)
    sip_unauthorized = False if py_mode else _sip_unauthorized()
    sip_blocked = sip_env_blocked or sip_unauthorized

    def _sip_allowed() -> bool:
        return not sip_blocked

    if requested_norm == "sip":
        return "sip" if _sip_allowed() else "iex"
    if requested_norm == "iex":
        return "iex"
    if requested_norm in _VALID_EXEC_FEEDS:
        return requested_norm

    if cached_norm == "sip":
        return "sip" if _sip_allowed() else "iex"
    if cached_norm == "iex":
        return "iex"

    return "sip" if _sip_allowed() else "iex"


def resolve(
    symbol: str,
    requested: str | None,
    *,
    role: FeedRole | str = "execution",
) -> str | None:
    """Resolve *requested* for *symbol* while updating the cache."""

    normalized_role = _normalize_role(role)
    role_cache = _FEED_CACHE_BY_ROLE[normalized_role]
    cached = role_cache.get(symbol)
    resolved = ensure_entitled_feed(requested, cached, role=normalized_role)
    if resolved is None:
        role_cache.pop(symbol, None)
        return None
    role_cache[symbol] = resolved
    return resolved


def cache(
    symbol: str,
    feed: str | None,
    *,
    role: FeedRole | str = "execution",
) -> str | None:
    """Persist a sanitized feed for *symbol* and return the stored value."""

    normalized_role = _normalize_role(role)
    role_cache = _FEED_CACHE_BY_ROLE[normalized_role]
    resolved = ensure_entitled_feed(feed, None, role=normalized_role)
    if resolved is None:
        role_cache.pop(symbol, None)
        return None
    role_cache[symbol] = resolved
    return resolved


def get_cached(symbol: str, *, role: FeedRole | str = "execution") -> str | None:
    """Return the cached feed for *symbol* if present."""

    normalized_role = _normalize_role(role)
    role_cache = _FEED_CACHE_BY_ROLE[normalized_role]
    return _normalize_feed(role_cache.get(symbol), role=normalized_role)


def clear(symbol: str | None = None, *, role: FeedRole | str | None = None) -> None:
    """Clear feed caches for a role and symbol, or all caches when omitted."""

    if role is None:
        roles: tuple[FeedRole, ...] = ("execution", "reference")
    else:
        roles = (_normalize_role(role),)
    for role_name in roles:
        role_cache = _FEED_CACHE_BY_ROLE[role_name]
        if symbol is None:
            role_cache.clear()
        else:
            role_cache.pop(symbol, None)


def _sip_unauthorized() -> bool:
    if _pytest_mode():
        return False
    try:
        from ai_trading.data import fetch as data_fetcher  # local import to avoid cycles
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("SIP_UNAUTHORIZED_FETCH_IMPORT_FAILED", exc_info=True)
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
