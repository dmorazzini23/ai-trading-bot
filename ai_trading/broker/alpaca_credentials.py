from __future__ import annotations
import os
from collections.abc import Mapping
from dataclasses import dataclass

@dataclass
class Credentials:
    API_KEY: str | None = None
    SECRET_KEY: str | None = None
    BASE_URL: str = 'https://paper-api.alpaca.markets'

def resolve_alpaca_credentials(env: Mapping[str, str] | None=None) -> Credentials:
    """Resolve Alpaca credentials from environment variables."""
    env = dict(env or os.environ)
    return Credentials(env.get('ALPACA_API_KEY'), env.get('ALPACA_SECRET_KEY'), env.get('ALPACA_BASE_URL') or 'https://paper-api.alpaca.markets')

def check_alpaca_available() -> bool:
    try:
        import alpaca_trade_api  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True

def initialize(env: Mapping[str, str] | None=None, *, shadow: bool=False):
    creds = resolve_alpaca_credentials(env)
    if shadow:
        return object()
    try:
        from alpaca_trade_api import REST  # type: ignore
    except ModuleNotFoundError:
        return object()
    return REST(key_id=creds.API_KEY, secret_key=creds.SECRET_KEY, base_url=creds.BASE_URL)
__all__ = ['Credentials', 'resolve_alpaca_credentials', 'check_alpaca_available', 'initialize']