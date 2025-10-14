"""Alpaca feed entitlement helpers used by tests and data fetchers."""

from __future__ import annotations

from typing import Any, Set

_ENTITLE_CACHE: dict[int, dict[str, Any]] = {}


def _get_entitled_feeds(client: Any) -> Set[str]:
    """Return normalized entitled feeds advertised by *client*."""

    if hasattr(client, "get_entitlements"):
        feeds = set(client.get_entitlements())
    elif hasattr(client, "entitlements"):
        feeds = set(client.entitlements)
    elif hasattr(client, "feeds"):
        feeds = set(client.feeds)
    else:
        feeds = {"iex"}
    normalized = {"sip" if str(feed).lower() == "sip" else "iex" for feed in feeds if feed}
    return normalized or {"iex"}


def _get_entitled_feeds_cached(client: Any) -> Set[str]:
    generation = getattr(client, "generation", None)
    key = id(client)
    cached = _ENTITLE_CACHE.get(key)
    if not cached or cached.get("generation") != generation:
        feeds = _get_entitled_feeds(client)
        _ENTITLE_CACHE[key] = {"feeds": feeds, "generation": generation}
        return feeds
    return cached["feeds"]


def _ensure_entitled_feed(client: Any, preferred: str) -> str:
    """Return an entitled feed matching *preferred* when available."""

    preferred_norm = "sip" if str(preferred).lower() == "sip" else "iex"
    entitled = _get_entitled_feeds_cached(client)
    if preferred_norm in entitled:
        return preferred_norm
    if "iex" in entitled:
        return "iex"
    if "sip" in entitled:
        return "sip"
    return next(iter(entitled)) if entitled else "iex"


__all__ = [
    "_ENTITLE_CACHE",
    "_get_entitled_feeds",
    "_get_entitled_feeds_cached",
    "_ensure_entitled_feed",
]

