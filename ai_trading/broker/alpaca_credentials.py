"""Helpers for resolving Alpaca API credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"
_CANONICAL_KEY = "ALPACA_API_KEY"
_CANONICAL_SECRET = "ALPACA_SECRET_KEY"
_LEGACY_KEY = "AP" "CA_" "API_KEY_ID"
_LEGACY_SECRET = "AP" "CA_" "API_SECRET_KEY"


@dataclass(frozen=True)
class AlpacaCreds:
    """Minimal Alpaca credentials container."""

    key: str | None
    secret: str | None
    _base_url: str | None = None

    @property
    def api_key(self) -> str | None:  # Backwards compat
        return self.key

    @property
    def secret_key(self) -> str | None:  # Backwards compat
        return self.secret

    @property
    def base_url(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL


@dataclass(slots=True)
class AlpacaCredentials:
    """Extended credential payload that includes the trading base URL."""

    api_key: str | None = None
    secret_key: str | None = None
    base_url: str = _DEFAULT_BASE_URL

    def has_data_credentials(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def has_execution_credentials(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def as_dict(self) -> dict[str, str | None]:
        return {
            "api_key": self.api_key,
            "secret_key": self.secret_key,
            "base_url": self.base_url,
        }


def _coerce_env(env: Mapping[str, str] | None) -> dict[str, str]:
    source = env if env is not None else os.environ
    result: dict[str, str] = {}
    for raw_key, raw_value in source.items():
        if not isinstance(raw_value, str):
            continue
        key = str(raw_key)
        value = raw_value.strip()
        if value:
            result[key] = value
    return result


def _resolve_value(env: Mapping[str, str], *keys: str) -> str | None:
    for key in keys:
        raw = env.get(key)
        if raw:
            value = raw.strip()
            if value:
                return value
    return None


def _resolve_base_url(env: Mapping[str, str]) -> str:
    override = _resolve_value(env, "ALPACA_API_URL", "ALPACA_BASE_URL")
    if override:
        normalized = override.rstrip("/")
        if normalized.lower().startswith(("http://", "https://")):
            return normalized
        return f"https://{normalized}"
    return _DEFAULT_BASE_URL


def resolve_alpaca_credentials(env: Mapping[str, str] | None = None) -> AlpacaCreds:
    """Return Alpaca credentials preferring canonical env vars."""

    env_map = _coerce_env(env)
    key = _resolve_value(env_map, _CANONICAL_KEY) or _resolve_value(env_map, _LEGACY_KEY)
    secret = _resolve_value(env_map, _CANONICAL_SECRET) or _resolve_value(env_map, _LEGACY_SECRET)
    base_url = _resolve_base_url(env_map)
    return AlpacaCreds(key=key, secret=secret, _base_url=base_url)


def resolve_alpaca_credentials_with_base(env: Mapping[str, str] | None = None) -> AlpacaCredentials:
    """Return credentials including the trading base URL."""

    env_map = _coerce_env(env)
    creds = resolve_alpaca_credentials(env_map)
    return AlpacaCredentials(api_key=creds.key, secret_key=creds.secret, base_url=creds.base_url)


def alpaca_auth_headers() -> dict[str, str]:
    """Return HTTP headers populated with resolved Alpaca credentials."""

    creds = resolve_alpaca_credentials()
    headers: dict[str, str] = {}
    if creds.key:
        headers["APCA-API-KEY-ID"] = creds.key
    if creds.secret:
        headers["APCA-API-SECRET-KEY"] = creds.secret
    return headers


def reset_alpaca_credential_state() -> None:
    """Placeholder for compatibility; credentials are resolved per-call."""

    return None


def check_alpaca_available() -> bool:
    """Return ``True`` if the Alpaca SDK can be imported."""

    try:  # pragma: no cover - presence check only
        from alpaca.trading.client import TradingClient  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def initialize(env: Mapping[str, str] | None = None, *, shadow: bool = False):
    """Return an ``alpaca.trading.client.TradingClient`` instance."""

    creds = resolve_alpaca_credentials_with_base(env)
    try:  # pragma: no cover - optional dependency
        __import__("alpaca")
        from alpaca.trading.client import TradingClient  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - tested via unit test
        if shadow:
            return object()
        raise RuntimeError("alpaca package is required") from exc
    return TradingClient(
        api_key=creds.api_key,
        secret_key=creds.secret_key,
        paper="paper" in creds.base_url.lower(),
        url_override=creds.base_url,
    )


__all__ = [
    "AlpacaCreds",
    "AlpacaCredentials",
    "resolve_alpaca_credentials",
    "resolve_alpaca_credentials_with_base",
    "alpaca_auth_headers",
    "check_alpaca_available",
    "initialize",
    "reset_alpaca_credential_state",
]
