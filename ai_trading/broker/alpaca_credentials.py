from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from ai_trading.logging import get_logger

log = get_logger(__name__)


@dataclass
class Credentials:
    API_KEY: str | None = None
    SECRET_KEY: str | None = None
    BASE_URL: str = "https://paper-api.alpaca.markets"


def resolve_alpaca_credentials(env: Mapping[str, str] | None = None, *, prefer: str = "ALPACA") -> Credentials:
    env = dict(env or os.environ)
    prefer = prefer.upper()
    def _pick(key_a: str, key_b: str) -> str | None:
        a = env.get(key_a)
        b = env.get(key_b)
        if a and b and a != b:
            log.warning("Conflicting credentials for %s vs %s; using %s", key_a, key_b, key_a if prefer == "ALPACA" else key_b)
        return (a if prefer == "ALPACA" else b) or (b if prefer == "ALPACA" else a)
    api_key = _pick("ALPACA_API_KEY", "APCA_API_KEY_ID")
    secret = _pick("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY")
    base = _pick("ALPACA_BASE_URL", "APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
    return Credentials(api_key, secret, base)


def check_alpaca_available() -> bool:
    try:
        import alpaca_trade_api  # type: ignore  # noqa: F401
        return True
    except Exception:  # pragma: no cover - optional dep
        return False


def initialize(env: Mapping[str, str] | None = None, *, shadow: bool = False):
    creds = resolve_alpaca_credentials(env)
    if shadow or not check_alpaca_available():
        return object()
    from alpaca_trade_api import REST as TradeApiREST  # type: ignore
    return TradeApiREST(creds.API_KEY, creds.SECRET_KEY, creds.BASE_URL)


__all__ = ["Credentials", "resolve_alpaca_credentials", "check_alpaca_available", "initialize"]
