"""Environment variable helpers for broker configuration."""
from __future__ import annotations

import os

from ai_trading.broker.alpaca_credentials import (
    AlpacaCredentials,
    resolve_alpaca_credentials,
    reset_alpaca_credential_state,
)

_DATA_FEED_OVERRIDE_CACHE: tuple[str | None, str | None] | None = None


def _resolve_data_feed_override() -> tuple[str | None, str | None]:
    """Return cached (override, reason) tuple for Alpaca data feed."""

    global _DATA_FEED_OVERRIDE_CACHE
    if _DATA_FEED_OVERRIDE_CACHE is not None:
        return _DATA_FEED_OVERRIDE_CACHE
    creds = resolve_alpaca_credentials()
    override: str | None = None
    reason: str | None = None
    if creds.has_execution_credentials() and not creds.has_data_credentials():
        override = "yahoo"
        reason = "missing_data_keys"
    _DATA_FEED_OVERRIDE_CACHE = (override, reason)
    return _DATA_FEED_OVERRIDE_CACHE


def refresh_alpaca_credentials_cache() -> None:
    """Reset cached Alpaca credential-derived helpers."""

    global _DATA_FEED_OVERRIDE_CACHE
    _DATA_FEED_OVERRIDE_CACHE = None
    reset_alpaca_credential_state()


def get_resolved_alpaca_credentials() -> AlpacaCredentials:
    """Return the resolved Alpaca credentials with source metadata."""

    return resolve_alpaca_credentials()


def get_alpaca_creds() -> tuple[str, str]:
    """Return Alpaca API credentials using canonical precedence."""

    creds = resolve_alpaca_credentials()
    if not creds.api_key or not creds.secret_key:
        missing: list[str] = []
        if not creds.api_key:
            missing.append(
                "ALPACA_API_KEY (or ALPACA_DATA_API_KEY/APCA_API_KEY_ID)"
            )
        if not creds.secret_key:
            missing.append(
                "ALPACA_SECRET_KEY (or ALPACA_DATA_SECRET_KEY/APCA_API_SECRET_KEY)"
            )
        raise RuntimeError(f"Missing Alpaca credentials: {', '.join(missing)}")
    return creds.api_key, creds.secret_key


def get_alpaca_base_url() -> str:
    """Return the configured Alpaca base URL with sensible defaults."""

    creds = resolve_alpaca_credentials()
    return creds.base_url


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
    return bool(creds.api_key), bool(creds.secret_key)


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
    "resolve_alpaca_feed",
    "alpaca_credential_status",
    "get_resolved_alpaca_credentials",
    "is_data_feed_downgraded",
    "get_data_feed_override",
    "get_data_feed_downgrade_reason",
    "refresh_alpaca_credentials_cache",
]
