"""Environment variable helpers for broker configuration."""
from __future__ import annotations

import os


_ALPACA_KEY_ENV_KEYS: tuple[str, ...] = (
    "ALPACA_API_KEY_ID",
    "APCA_API_KEY_ID",
    "ALPACA_API_KEY",
)

_ALPACA_SECRET_ENV_KEYS: tuple[str, ...] = (
    "ALPACA_API_SECRET_KEY",
    "APCA_API_SECRET_KEY",
    "ALPACA_SECRET_KEY",
)


def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _any_env(*keys: str) -> bool:
    return any(os.getenv(k) for k in keys)


def get_alpaca_creds() -> tuple[str, str]:
    """Return Alpaca API credentials using common environment aliases."""

    key = _first_env(*_ALPACA_KEY_ENV_KEYS)
    secret = _first_env(*_ALPACA_SECRET_ENV_KEYS)
    if not key or not secret:
        missing: list[str] = []
        if not key:
            missing.append("ALPACA_API_KEY_ID (or APCA_API_KEY_ID/ALPACA_API_KEY)")
        if not secret:
            missing.append("ALPACA_API_SECRET_KEY (or APCA_API_SECRET_KEY/ALPACA_SECRET_KEY)")
        raise RuntimeError(f"Missing Alpaca credentials: {', '.join(missing)}")
    return key, secret


def get_alpaca_base_url() -> str:
    """Return the configured Alpaca base URL with sensible defaults."""

    return (
        os.getenv("ALPACA_API_URL")
        or os.getenv("ALPACA_BASE_URL")
        or "https://paper-api.alpaca.markets"
    )


def resolve_alpaca_feed(requested: str | None) -> str | None:
    """Return a valid Alpaca feed name for API requests.

    ``None`` indicates the caller should not contact Alpaca and instead route to
    a non-Alpaca provider such as Yahoo Finance.
    """

    if not requested:
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

    return _any_env(*_ALPACA_KEY_ENV_KEYS), _any_env(*_ALPACA_SECRET_ENV_KEYS)


__all__ = [
    "get_alpaca_creds",
    "get_alpaca_base_url",
    "resolve_alpaca_feed",
    "alpaca_credential_status",
]
