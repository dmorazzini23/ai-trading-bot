"""Environment variable helpers for broker configuration."""
from __future__ import annotations

import os
from typing import Mapping

from ai_trading.broker.alpaca_credentials import (
    AlpacaCredentials,
    alpaca_auth_headers,
    resolve_alpaca_credentials,
    resolve_alpaca_credentials_with_base,
    reset_alpaca_credential_state,
)

_DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"

_DATA_FEED_OVERRIDE_CACHE: tuple[str | None, str | None] | None = None


def _resolve_data_feed_override() -> tuple[str | None, str | None]:
    """Return cached (override, reason) tuple for Alpaca data feed."""

    global _DATA_FEED_OVERRIDE_CACHE
    if _DATA_FEED_OVERRIDE_CACHE is not None:
        return _DATA_FEED_OVERRIDE_CACHE
    creds = resolve_alpaca_credentials()
    override: str | None = None
    reason: str | None = None
    if not creds.key or not creds.secret:
        override = "yahoo"
        reason = "missing_credentials"
    _DATA_FEED_OVERRIDE_CACHE = (override, reason)
    return _DATA_FEED_OVERRIDE_CACHE


def refresh_alpaca_credentials_cache() -> None:
    """Reset cached Alpaca credential-derived helpers."""

    global _DATA_FEED_OVERRIDE_CACHE
    _DATA_FEED_OVERRIDE_CACHE = None
    reset_alpaca_credential_state()


def get_resolved_alpaca_credentials() -> AlpacaCredentials:
    """Return the resolved Alpaca credentials with source metadata."""

    return resolve_alpaca_credentials_with_base()


def resolve_alpaca_creds(env: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    """Return Alpaca credentials preferring canonical env vars with AP""" "CA_* fallback."""

    creds = resolve_alpaca_credentials(env)
    return creds.key, creds.secret


def get_alpaca_creds() -> tuple[str, str]:
    """Return Alpaca API credentials using canonical precedence."""

    creds = resolve_alpaca_credentials()
    if not creds.key or not creds.secret:
        missing: list[str] = []
        if not creds.key:
            missing.append("ALPACA_API_KEY")
        if not creds.secret:
            missing.append("ALPACA_SECRET_KEY")
        raise RuntimeError(f"Missing Alpaca credentials: {', '.join(missing)}")
    return creds.key, creds.secret


def get_alpaca_base_url() -> str:
    """Return the configured Alpaca base URL with sensible defaults."""

    creds = resolve_alpaca_credentials_with_base()
    return creds.base_url


def get_alpaca_data_base_url() -> str:
    """Return the Alpaca market data base URL with optional override."""

    override = os.getenv("ALPACA_DATA_BASE_URL", "").strip()
    if override:
        normalized = override.rstrip("/")
        if normalized.lower().startswith(("http://", "https://")):
            return normalized
    return _DEFAULT_DATA_BASE_URL


def get_alpaca_http_headers() -> dict[str, str]:
    """Return HTTP headers including Alpaca credentials when available."""

    return alpaca_auth_headers()


def resolve_alpaca_feed(requested: str | None) -> str | None:
    """Return a valid Alpaca feed name for API requests.

    ``None`` indicates the caller should not contact Alpaca and instead route to
    a non-Alpaca provider such as Yahoo Finance.
    """

    override, _ = _resolve_data_feed_override()
    if override:
        requested = override
    elif not requested:
        requested = (
            os.getenv("ALPACA_DEFAULT_FEED")
            or os.getenv("ALPACA_DATA_FEED")
            or "iex"
        )
    normalized = requested.lower()
    if normalized in {"sip", "iex"}:
        return normalized
    if normalized == "yahoo":
        return None
    return "iex"


def alpaca_credential_status() -> tuple[bool, bool]:
    """Return boolean flags for Alpaca credential presence."""

    creds = resolve_alpaca_credentials()
    return bool(creds.key), bool(creds.secret)


def is_data_feed_downgraded() -> bool:
    """Return ``True`` when the Alpaca data feed has been downgraded."""

    override, _ = _resolve_data_feed_override()
    return bool(override)


def get_data_feed_override() -> str | None:
    """Return the forced data feed override if one is active."""

    override, _ = _resolve_data_feed_override()
    return override


def get_data_feed_downgrade_reason() -> str | None:
    """Return the downgrade reason when Alpaca data feed is forced off."""

    _, reason = _resolve_data_feed_override()
    return reason


__all__ = [
    "get_alpaca_creds",
    "get_alpaca_base_url",
    "get_alpaca_data_base_url",
    "get_alpaca_http_headers",
    "resolve_alpaca_feed",
    "alpaca_credential_status",
    "get_resolved_alpaca_credentials",
    "resolve_alpaca_creds",
    "is_data_feed_downgraded",
    "get_data_feed_override",
    "get_data_feed_downgrade_reason",
    "refresh_alpaca_credentials_cache",
]
