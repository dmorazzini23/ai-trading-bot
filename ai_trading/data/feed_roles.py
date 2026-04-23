"""Helpers for separating execution and reference Alpaca feed selection."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from typing import Literal

from ai_trading.config.management import get_env
from ai_trading.utils.env import resolve_alpaca_feed

FeedRole = Literal["execution", "reference"]
ExecutionFeed = Literal["iex", "sip"]
ReferenceFeed = Literal["iex", "sip", "delayed_sip"]

_DELAYED_TOKENS = {"delayed", "delayed_sip", "delayed-sip", "dsip"}


def normalize_feed_role(value: str | None) -> FeedRole:
    token = str(value or "").strip().lower()
    if token == "reference":
        return "reference"
    return "execution"


def is_delayed_feed(feed: str | None) -> bool:
    token = str(feed or "").strip().lower()
    return token in _DELAYED_TOKENS


def _settings_feed(attr: str) -> str | None:
    try:
        from ai_trading.settings import get_settings

        settings = get_settings()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return None
    try:
        value = getattr(settings, attr, None)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return None
    if value in (None, ""):
        return None
    return str(value)


def _normalize_reference_feed(value: str | None) -> ReferenceFeed:
    token = str(value or "").strip().lower()
    if token in {"", "auto"}:
        return "delayed_sip"
    if token in _DELAYED_TOKENS:
        return "delayed_sip"
    if token == "sip":
        # Promote to delayed SIP when SIP entitlement is not available.
        return "sip" if str(resolve_alpaca_feed("sip")).strip().lower() == "sip" else "delayed_sip"
    if token == "iex":
        return "iex"
    return "delayed_sip"


def get_execution_feed(requested: str | None = None) -> ExecutionFeed:
    candidate = requested
    if candidate in (None, ""):
        candidate = get_env("ALPACA_EXECUTION_FEED", None, cast=str, resolve_aliases=False)
    if candidate in (None, ""):
        # Legacy fallback only after explicit execution-feed knobs.
        candidate = get_env("ALPACA_DATA_FEED", None, cast=str, resolve_aliases=False)
    if candidate in (None, ""):
        candidate = _settings_feed("alpaca_execution_feed")
    if candidate in (None, ""):
        candidate = _settings_feed("alpaca_data_feed")
    resolved = resolve_alpaca_feed(candidate)
    if str(resolved or "").strip().lower() == "sip":
        return "sip"
    return "iex"


def get_reference_feed(requested: str | None = None) -> ReferenceFeed:
    candidate = requested
    if candidate in (None, ""):
        candidate = get_env("ALPACA_REFERENCE_FEED", None, cast=str, resolve_aliases=False)
    if candidate in (None, ""):
        candidate = _settings_feed("alpaca_reference_feed")
    return _normalize_reference_feed(candidate)


def get_reference_bars_feed(requested: str | None = None) -> ExecutionFeed:
    """Return a bars-endpoint compatible feed for reference-role bar fetches.

    Alpaca bars endpoints accept real-time feeds (``iex``/``sip``) and reject
    ``delayed_sip``. When delayed reference is requested, prefer ``sip`` when
    entitled; otherwise degrade to ``iex``.
    """

    reference_feed = get_reference_feed(requested)
    if reference_feed == "iex":
        return "iex"
    sip_resolved = str(resolve_alpaca_feed("sip") or "").strip().lower()
    if sip_resolved == "sip":
        return "sip"
    return "iex"


def resolve_feed_for_role(
    requested: str | None = None,
    *,
    role: FeedRole | str = "execution",
) -> str:
    normalized_role = normalize_feed_role(str(role))
    if normalized_role == "reference":
        return get_reference_feed(requested)
    return get_execution_feed(requested)


__all__ = [
    "FeedRole",
    "ExecutionFeed",
    "ReferenceFeed",
    "normalize_feed_role",
    "is_delayed_feed",
    "get_execution_feed",
    "get_reference_feed",
    "get_reference_bars_feed",
    "resolve_feed_for_role",
]
