from __future__ import annotations
import os
from collections.abc import Mapping
from dataclasses import dataclass
from ai_trading.utils.optional_import import optional_import

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
    return optional_import('alpaca_trade_api') is not None

def initialize(env: Mapping[str, str] | None=None, *, shadow: bool=False):
    creds = resolve_alpaca_credentials(env)
    if shadow or not check_alpaca_available():
        return object()
    TradeApiREST = optional_import('alpaca_trade_api', 'REST')
    if TradeApiREST is None:
        return object()
    return TradeApiREST(key_id=creds.API_KEY, secret_key=creds.SECRET_KEY, base_url=creds.BASE_URL)
__all__ = ['Credentials', 'resolve_alpaca_credentials', 'check_alpaca_available', 'initialize']