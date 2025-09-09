from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import logging
import os
from ai_trading.alpaca_api import ALPACA_AVAILABLE, get_trading_client_cls
from .settings import broker_keys, get_settings

@dataclass(frozen=True)
class AlpacaConfig:
    base_url: str
    key_id: str
    secret_key: str
    use_paper: bool
    rate_limit_per_min: int | None = None
    data_feed: str = "iex"

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
    feed = (os.getenv('ALPACA_DATA_FEED') or getattr(s, 'alpaca_data_feed', 'iex')).lower()
    if not key_id or not secret:
        raise RuntimeError('Missing Alpaca credentials in broker_keys() (ALPACA_KEY_ID/ALPACA_SECRET_KEY)')
    if ALPACA_AVAILABLE:
        try:
            TradingClient = get_trading_client_cls()
            client = TradingClient(api_key=key_id, secret_key=secret, paper=use_paper)
            if hasattr(client, 'get_account'):
                acct = client.get_account()
                sub = getattr(acct, 'market_data_subscription', None) or getattr(acct, 'data_feed', None)
                if isinstance(sub, str):
                    entitled = {sub.lower()}
                elif isinstance(sub, (set, list, tuple)):
                    entitled = {str(x).lower() for x in sub}
                else:
                    entitled = set()
                if entitled and feed not in entitled:
                    alt = next(iter(entitled))
                    logging.getLogger(__name__).warning(
                        'ALPACA_FEED_UNENTITLED_SWITCH', extra={'requested': feed, 'using': alt}
                    )
                    feed = alt
            else:
                logging.getLogger(__name__).warning('ALPACA_CLIENT_NO_GET_ACCOUNT')
        except Exception:
            pass
    return AlpacaConfig(
        base_url=base_url,
        key_id=key_id,
        secret_key=secret,
        use_paper=use_paper,
        rate_limit_per_min=rate_limit,
        data_feed=feed,
    )
