import os
from functools import lru_cache
from ai_trading.settings import Settings, _secret_to_str
from ai_trading.settings import get_settings as _base_get_settings

TICKERS_FILE = os.getenv('AI_TRADING_TICKERS_FILE', 'tickers.csv')
MODEL_PATH = os.getenv('AI_TRADING_MODEL_PATH')

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return _base_get_settings()

def broker_keys(s: Settings | None=None) -> dict[str, str]:
    """Return broker credential mapping."""
    s = s or get_settings()
    keys = {
        'ALPACA_API_KEY': getattr(s, 'alpaca_api_key', ''),
        'ALPACA_SECRET_KEY': _secret_to_str(getattr(s, 'alpaca_secret_key', None)) or '',
    }
    if getattr(s, 'finnhub_api_key', None):
        keys['finnhub'] = s.finnhub_api_key
    return keys

def provider_priority(s: Settings | None = None) -> tuple[str, ...]:
    """Return configured data provider priority order."""
    s = s or get_settings()
    return tuple(getattr(s, 'data_provider_priority', ())) or (
        'alpaca_iex', 'alpaca_sip', 'yahoo'
    )

def max_data_fallbacks(s: Settings | None = None) -> int:
    """Return maximum allowed provider fallbacks."""
    s = s or get_settings()
    return int(getattr(s, 'max_data_fallbacks', 2))

__all__ = ['Settings', 'get_settings', 'broker_keys', 'provider_priority', 'max_data_fallbacks']
