"""Helpers for working with Alpaca API credentials."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


@dataclass(slots=True)
class AlpacaCredentials:
    """Typed container for Alpaca API credentials."""

    api_key: str | None = None
    secret_key: str | None = None
    base_url: str = _DEFAULT_BASE_URL


def resolve_alpaca_credentials(env: Mapping[str, str] | None = None) -> AlpacaCredentials:
    """Return Alpaca credentials from *env* as an :class:`AlpacaCredentials` dataclass."""

    env = dict(env or os.environ)
    return AlpacaCredentials(
        env.get("ALPACA_API_KEY"),
        env.get("ALPACA_SECRET_KEY"),
        env.get("ALPACA_BASE_URL") or _DEFAULT_BASE_URL,
    )


def check_alpaca_available() -> bool:
    """Return ``True`` if the Alpaca SDK is importable."""

    try:  # pragma: no cover - purely a presence check
        import alpaca_trade_api  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def initialize(env: Mapping[str, str] | None = None, *, shadow: bool = False):
    """Return an ``alpaca_trade_api.REST`` instance or a benign placeholder."""

    creds = resolve_alpaca_credentials(env)
    if shadow:
        return object()
    try:  # pragma: no cover - optional dependency
        from alpaca_trade_api import REST  # type: ignore
    except ModuleNotFoundError:
        return object()
    return REST(key_id=creds.api_key, secret_key=creds.secret_key, base_url=creds.base_url)


__all__ = [
    "AlpacaCredentials",
    "resolve_alpaca_credentials",
    "check_alpaca_available",
    "initialize",
]

