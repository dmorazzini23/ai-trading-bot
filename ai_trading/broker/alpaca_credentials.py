"""Helpers for working with Alpaca API credentials."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"

_CANONICAL_KEY = "ALPACA_API_KEY"
_CANONICAL_SECRET = "ALPACA_SECRET_KEY"
_LEGACY_KEY = "APCA_API_KEY_ID"
_LEGACY_SECRET = "APCA_API_SECRET_KEY"


@dataclass(slots=True)
class AlpacaCredentials:
    """Typed container for Alpaca API credentials."""

    api_key: str | None = None
    secret_key: str | None = None
    base_url: str = _DEFAULT_BASE_URL

    def has_data_credentials(self) -> bool:
        """Return ``True`` when canonical credentials are present."""

        return bool(self.api_key and self.secret_key)

    def has_execution_credentials(self) -> bool:
        """Return ``True`` when canonical credentials are present."""

        return bool(self.api_key and self.secret_key)

    def as_dict(self) -> dict[str, str | None]:
        """Return a mapping representation for structured logging/tests."""

        return {
            "api_key": self.api_key,
            "secret_key": self.secret_key,
            "base_url": self.base_url,
        }


def resolve_alpaca_credentials(env: Mapping[str, str] | None = None) -> AlpacaCredentials:
    """Return Alpaca credentials resolved from the environment."""

    env_map = {str(k): v for k, v in dict(env or os.environ).items() if isinstance(v, str)}

    def _first(*keys: str) -> str | None:
        for key in keys:
            value = env_map.get(key)
            if value:
                stripped = value.strip()
                if stripped:
                    return stripped
        return None

    api_key = _first(_CANONICAL_KEY, _LEGACY_KEY)
    secret_key = _first(_CANONICAL_SECRET, _LEGACY_SECRET)

    if not api_key or not secret_key:
        api_key = api_key or None
        secret_key = secret_key or None

    base_url = _first("ALPACA_API_URL", "ALPACA_BASE_URL") or _DEFAULT_BASE_URL
    return AlpacaCredentials(api_key, secret_key, base_url)


def reset_alpaca_credential_state() -> None:
    """No-op retained for compatibility."""

    return None


def check_alpaca_available() -> bool:
    """Return ``True`` if the :mod:`alpaca` SDK is importable."""

    try:  # pragma: no cover - purely a presence check
        from alpaca.trading.client import TradingClient  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def initialize(env: Mapping[str, str] | None = None, *, shadow: bool = False):
    """Return an :class:`alpaca.trading.client.TradingClient` instance.

    If the SDK is missing and *shadow* is ``True``, a simple ``object`` stub is
    returned. Otherwise, a :class:`RuntimeError` is raised when the ``alpaca``
    package cannot be imported.
    """

    creds = resolve_alpaca_credentials(env)
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
    "AlpacaCredentials",
    "resolve_alpaca_credentials",
    "check_alpaca_available",
    "initialize",
    "reset_alpaca_credential_state",
]

