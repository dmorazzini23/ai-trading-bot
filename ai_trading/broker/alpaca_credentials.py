"""Helpers for working with Alpaca API credentials."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


_CANONICAL_KEY = "ALPACA_API_KEY"
_CANONICAL_SECRET = "ALPACA_SECRET_KEY"
_DATA_KEY_FALLBACKS: tuple[str, ...] = ("ALPACA_DATA_API_KEY",)
_DATA_SECRET_FALLBACKS: tuple[str, ...] = ("ALPACA_DATA_SECRET_KEY",)
_EXEC_KEY_FALLBACKS: tuple[str, ...] = ("APCA_API_KEY_ID", "ALPACA_API_KEY_ID")
_EXEC_SECRET_FALLBACKS: tuple[str, ...] = ("APCA_API_SECRET_KEY", "ALPACA_API_SECRET_KEY")
_FALLBACK_SOURCE: dict[str, str | None] = {"api": None, "secret": None}
_FALLBACK_ACTIVE: dict[str, bool] = {"api": False, "secret": False}


@dataclass(slots=True)
class AlpacaCredentials:
    """Typed container for Alpaca API credentials."""

    api_key: str | None = None
    secret_key: str | None = None
    base_url: str = _DEFAULT_BASE_URL
    api_source: str | None = None
    secret_source: str | None = None

    def has_data_credentials(self) -> bool:
        """Return ``True`` when credentials originate from data-capable keys."""

        return (
            self.api_source in (_CANONICAL_KEY,) + _DATA_KEY_FALLBACKS
            and self.secret_source in (_CANONICAL_SECRET,) + _DATA_SECRET_FALLBACKS
        )

    def has_execution_credentials(self) -> bool:
        """Return ``True`` when credentials can authenticate trading."""

        return (
            self.api_source in (_CANONICAL_KEY,) + _EXEC_KEY_FALLBACKS
            and self.secret_source in (_CANONICAL_SECRET,) + _EXEC_SECRET_FALLBACKS
        )

    def as_dict(self) -> dict[str, str | None]:
        """Return a mapping representation for structured logging/tests."""

        return {
            "api_key": self.api_key,
            "secret_key": self.secret_key,
            "base_url": self.base_url,
            "api_source": self.api_source,
            "secret_source": self.secret_source,
        }


def _select_env_value(env: Mapping[str, str], canonical: str, aliases: tuple[str, ...]) -> tuple[str | None, str | None]:
    for key in (canonical, *aliases):
        value = env.get(key)
        if value:
            return value, key
    return None, None


def resolve_alpaca_credentials(env: Mapping[str, str] | None = None) -> AlpacaCredentials:
    """Return Alpaca credentials resolved from the environment."""

    env_map = dict(env or os.environ)
    canonical_key_present = bool(env_map.get(_CANONICAL_KEY))
    canonical_secret_present = bool(env_map.get(_CANONICAL_SECRET))

    api_key, api_source = _select_env_value(env_map, _CANONICAL_KEY, _DATA_KEY_FALLBACKS + _EXEC_KEY_FALLBACKS)
    secret_key, secret_source = _select_env_value(
        env_map,
        _CANONICAL_SECRET,
        _DATA_SECRET_FALLBACKS + _EXEC_SECRET_FALLBACKS,
    )

    effective_api_source = api_source
    effective_secret_source = secret_source

    if api_key:
        if api_source == _CANONICAL_KEY:
            stored = _FALLBACK_SOURCE.get("api")
            if _FALLBACK_ACTIVE.get("api") and stored and env_map.get(stored) == api_key:
                effective_api_source = stored
            else:
                _FALLBACK_SOURCE["api"] = None
                _FALLBACK_ACTIVE["api"] = False
        else:
            _FALLBACK_SOURCE["api"] = api_source
            _FALLBACK_ACTIVE["api"] = not canonical_key_present
    else:
        _FALLBACK_SOURCE["api"] = None
        _FALLBACK_ACTIVE["api"] = False

    if secret_key:
        if secret_source == _CANONICAL_SECRET:
            stored_secret = _FALLBACK_SOURCE.get("secret")
            if _FALLBACK_ACTIVE.get("secret") and stored_secret and env_map.get(stored_secret) == secret_key:
                effective_secret_source = stored_secret
            else:
                _FALLBACK_SOURCE["secret"] = None
                _FALLBACK_ACTIVE["secret"] = False
        else:
            _FALLBACK_SOURCE["secret"] = secret_source
            _FALLBACK_ACTIVE["secret"] = not canonical_secret_present
    else:
        _FALLBACK_SOURCE["secret"] = None
        _FALLBACK_ACTIVE["secret"] = False

    if env is None:
        if api_key and not os.getenv(_CANONICAL_KEY):
            os.environ[_CANONICAL_KEY] = api_key
        if secret_key and not os.getenv(_CANONICAL_SECRET):
            os.environ[_CANONICAL_SECRET] = secret_key

    base_url = env_map.get("ALPACA_BASE_URL") or env_map.get("ALPACA_API_URL") or _DEFAULT_BASE_URL
    return AlpacaCredentials(api_key, secret_key, base_url, effective_api_source, effective_secret_source)


def reset_alpaca_credential_state() -> None:
    """Clear cached fallback bookkeeping used for source tracking."""

    _FALLBACK_SOURCE["api"] = None
    _FALLBACK_SOURCE["secret"] = None
    _FALLBACK_ACTIVE["api"] = False
    _FALLBACK_ACTIVE["secret"] = False


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

