from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from .settings import broker_keys, get_settings

@dataclass(frozen=True)
class AlpacaConfig:
    base_url: str
    key_id: str
    secret_key: str
    use_paper: bool
    rate_limit_per_min: int | None = None

def get_alpaca_config() -> AlpacaConfig:
    s = get_settings()
    keys: Any = broker_keys()
    if isinstance(keys, Mapping):
        key_id = keys.get('ALPACA_KEY_ID') or keys.get('ALPACA_API_KEY') or ''
        secret = keys.get('ALPACA_SECRET_KEY', '')
    else:
        key_id, secret = keys
    use_paper = getattr(s, 'env', 'dev') != 'prod'
    base_url = getattr(s, 'alpaca_base_url', None)
    if not base_url:
        base_url = 'https://paper-api.alpaca.markets' if use_paper else 'https://api.alpaca.markets'
    rate_limit = getattr(s, 'alpaca_rate_limit_per_min', None)
    if not key_id or not secret:
        raise RuntimeError('Missing Alpaca credentials in broker_keys() (ALPACA_KEY_ID/ALPACA_SECRET_KEY)')
    return AlpacaConfig(base_url=base_url, key_id=key_id, secret_key=secret, use_paper=use_paper, rate_limit_per_min=rate_limit)