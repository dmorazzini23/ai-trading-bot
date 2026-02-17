"""Environment variable helpers for broker configuration."""
from __future__ import annotations

import os
from typing import Mapping
from urllib.parse import urlparse

from ai_trading.broker.alpaca_credentials import (
    AlpacaCredentials,
    alpaca_auth_headers,
    resolve_alpaca_credentials,
    resolve_alpaca_credentials_with_base,
    reset_alpaca_credential_state,
)

_DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"
_FORBIDDEN_TRADING_HOSTS = {
    "api.alpaca.markets",
    "paper-api.alpaca.markets",
}

_DATA_FEED_OVERRIDE_CACHE: tuple[str | None, str | None, str | None, str | None] | None = None


def _bool_env(env: Mapping[str, str | None], key: str) -> bool:
    raw = env.get(key)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_data_feed_override() -> tuple[str | None, str | None]:
    """Return cached (override, reason) tuple for Alpaca data feed."""

    global _DATA_FEED_OVERRIDE_CACHE

    creds = resolve_alpaca_credentials()
    current_fingerprint = (creds.key or None, creds.secret or None)

    if _DATA_FEED_OVERRIDE_CACHE is not None:
        cached_override, cached_reason, cached_key, cached_secret = _DATA_FEED_OVERRIDE_CACHE
        if (cached_key, cached_secret) == current_fingerprint:
            return cached_override, cached_reason

    override: str | None = None
    reason: str | None = None
    key, secret = current_fingerprint
    if not key or not secret:
        override = "yahoo"
        reason = "missing_credentials"

    _DATA_FEED_OVERRIDE_CACHE = (override, reason, key, secret)
    return override, reason


def refresh_alpaca_credentials_cache() -> None:
    """Reset cached Alpaca credential-derived helpers."""

    global _DATA_FEED_OVERRIDE_CACHE
    _DATA_FEED_OVERRIDE_CACHE = None
    reset_alpaca_credential_state()


def get_resolved_alpaca_credentials() -> AlpacaCredentials:
    """Return the resolved Alpaca credentials with source metadata."""

    return resolve_alpaca_credentials_with_base()


def resolve_alpaca_creds(env: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    """Return Alpaca credentials preferring canonical env vars."""

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
    """Return the Alpaca market data base URL with optional override.

    The canonical data host is ``https://data.alpaca.markets``. When an
    override is provided, ensure it does **not** point at the trading API
    hosts (``api.alpaca.markets`` or ``paper-api.alpaca.markets``) to avoid
    accidental credential leakage.
    """

    def _sanitize(candidate: str | None) -> str:
        raw = (candidate or "").strip()
        if not raw:
            raw = _DEFAULT_DATA_BASE_URL
        if "://" not in raw:
            raw = f"https://{raw}"
        parsed = urlparse(raw)
        hostname = (parsed.hostname or "").strip().lower()
        if not hostname:
            hostname = urlparse(_DEFAULT_DATA_BASE_URL).hostname or "data.alpaca.markets"
        if hostname in _FORBIDDEN_TRADING_HOSTS:
            return _DEFAULT_DATA_BASE_URL
        scheme = parsed.scheme.lower() if parsed.scheme else "https"
        if scheme not in {"http", "https"}:
            scheme = "https"
        path = (parsed.path or "").strip()
        path = path.rstrip("/")
        if path.lower() == "/v2":
            path = ""
        if path and not path.startswith("/"):
            path = f"/{path}"
        port = parsed.port
        netloc = hostname
        if port:
            netloc = f"{netloc}:{port}"
        normalized = f"{scheme}://{netloc}"
        if path and path != "/":
            normalized = f"{normalized}{path}"
        return normalized.rstrip("/") or _DEFAULT_DATA_BASE_URL

    override = os.getenv("ALPACA_DATA_BASE_URL") or os.getenv("ALPACA_DATA_URL")
    return _sanitize(override)


def get_alpaca_data_v2_base() -> str:
    """Return the Alpaca data base URL with an ensured ``/v2`` suffix."""

    base = get_alpaca_data_base_url().rstrip("/")
    if base.lower().endswith("/v2"):
        return base
    return f"{base}/v2"


def get_alpaca_http_headers() -> dict[str, str]:
    """Return HTTP headers including Alpaca credentials when available."""

    return alpaca_auth_headers()


def resolve_alpaca_feed(requested: str | None) -> str | None:
    """Return a valid Alpaca feed name for API requests.

    ``None`` indicates the caller should not contact Alpaca and instead route to
    a non-Alpaca provider such as Yahoo Finance.
    """

    override, _ = _resolve_data_feed_override()
    if override == "yahoo":
        return None
    env = os.environ
    allow_sip = _bool_env(env, "ALPACA_ALLOW_SIP")
    has_sip = _bool_env(env, "ALPACA_HAS_SIP") or _bool_env(env, "ALPACA_SIP_ENTITLED")
    sip_unauth = _bool_env(env, "ALPACA_SIP_UNAUTHORIZED")

    if override:
        requested = override
    elif not requested:
        requested = (
            os.getenv("ALPACA_DEFAULT_FEED")
            or os.getenv("ALPACA_DATA_FEED")
            or os.getenv("DATA_FEED_INTRADAY")
            or "iex"
        )

    normalized = str(requested).strip().lower()
    if normalized in {"", "auto"}:
        normalized = "auto"

    if normalized == "yahoo":
        return None

    if normalized in {"auto", "sip"}:
        if allow_sip and has_sip and not sip_unauth:
            return "sip"
        return "iex"

    if normalized == "iex":
        return "iex"

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
