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


_original_cache_clear = get_settings.cache_clear


def _cache_clear() -> None:
    """Clear cached settings, including the base Settings singleton."""

    _original_cache_clear()
    try:
        _base_get_settings.cache_clear()  # type: ignore[attr-defined]
    except AttributeError:
        pass


get_settings.cache_clear = _cache_clear  # type: ignore[assignment]

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


def minute_data_freshness_tolerance(s: Settings | None = None) -> int:
    """Return maximum tolerated staleness for minute data in seconds."""

    s = s or get_settings()
    raw_value = getattr(s, 'minute_data_freshness_tolerance_seconds', 900)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 900
    return value if value > 0 else 900


def alpaca_feed_failover(s: Settings | None = None) -> tuple[str, ...]:
    """Return preferred Alpaca feed fallback order."""
    s = s or get_settings()
    feeds = getattr(s, 'alpaca_feed_failover', ())
    if isinstance(feeds, tuple):
        return feeds
    return tuple(feeds or ())


def alpaca_empty_to_backup(s: Settings | None = None) -> bool:
    """Return whether to route empty payloads to the backup provider."""
    s = s or get_settings()
    return bool(getattr(s, 'alpaca_empty_to_backup', True))


def sentiment_retry_max(s: Settings | None = None) -> int:
    """Return maximum sentiment fetch retry count (defaults to 5 attempts)."""
    s = s or get_settings()
    return int(getattr(s, 'sentiment_max_retries', 5))


def sentiment_backoff_base(s: Settings | None = None) -> float:
    """Return base delay for sentiment fetch backoff (defaults to 5 seconds)."""
    s = s or get_settings()
    return float(getattr(s, 'sentiment_backoff_base', 5.0))


def sentiment_backoff_strategy(s: Settings | None = None) -> str:
    """Return strategy for sentiment fetch backoff."""
    s = s or get_settings()
    return str(getattr(s, 'sentiment_backoff_strategy', 'exponential'))

__all__ = [
    'Settings',
    'get_settings',
    'broker_keys',
    'provider_priority',
    'max_data_fallbacks',
    'minute_data_freshness_tolerance',
    'alpaca_feed_failover',
    'alpaca_empty_to_backup',
    'sentiment_retry_max',
    'sentiment_backoff_base',
    'sentiment_backoff_strategy',
]
