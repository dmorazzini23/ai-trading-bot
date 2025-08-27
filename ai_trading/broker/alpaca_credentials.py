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


def initialize(env: Mapping[str, str] | None = None, *, shadow: bool = False):
    """Return an ``alpaca.trading.client.TradingClient`` instance.

    If *shadow* is ``True``, a simple ``object`` stub is returned.
    """

    creds = resolve_alpaca_credentials(env)
    if shadow:
        return object()
    from alpaca.trading.client import TradingClient

    return TradingClient(
        creds.api_key,
        creds.secret_key,
        base_url=creds.base_url,
    )


__all__ = ["AlpacaCredentials", "resolve_alpaca_credentials", "initialize"]

