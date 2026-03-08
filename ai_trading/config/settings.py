import sys
from functools import lru_cache
from typing import Any, cast
from ai_trading.settings import Settings, _secret_to_str, _to_bool, _to_float, _to_int
from ai_trading.settings import get_settings as _base_get_settings

_SETTINGS_FALLBACK_FIELDS: dict[str, str] = {
    "AI_TRADING_TICKERS_FILE": "tickers_file",
    "AI_TRADING_MODEL_PATH": "model_path",
}


def _managed_env(name: str, default: str | None = None) -> str | None:
    management_module = sys.modules.get("ai_trading.config.management")
    getter = (
        getattr(management_module, "get_env", None)
        if management_module is not None
        else None
    )
    if callable(getter):
        try:
            value = getter(name, default, cast=str, resolve_aliases=False)
        except Exception:
            value = default
        return None if value is None else str(value)

    fallback_field = _SETTINGS_FALLBACK_FIELDS.get(name)
    if fallback_field:
        try:
            settings_obj = _base_get_settings()
        except Exception:
            return default
        candidate = getattr(settings_obj, fallback_field, default)
        return None if candidate is None else str(candidate)
    return default


TICKERS_FILE = str(_managed_env("AI_TRADING_TICKERS_FILE", "tickers.csv") or "tickers.csv")
MODEL_PATH = _managed_env("AI_TRADING_MODEL_PATH")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return _base_get_settings()


_original_cache_clear = get_settings.cache_clear


def _cache_clear() -> None:
    """Clear cached settings, including the base Settings singleton."""

    _original_cache_clear()
    base_cache_clear = getattr(_base_get_settings, "cache_clear", None)
    if callable(base_cache_clear):
        base_cache_clear()


cast(Any, get_settings).cache_clear = _cache_clear

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
    priority = tuple(getattr(s, 'data_provider_priority', ())) or ()
    if priority:
        return priority
    try:
        from ai_trading.utils.env import resolve_alpaca_feed
        sip_allowed = str(resolve_alpaca_feed("sip")).strip().lower() == "sip"
    except Exception:
        sip_allowed = False
    if sip_allowed:
        return ('alpaca_iex', 'alpaca_sip', 'yahoo')
    return ('alpaca_iex', 'yahoo')

def max_data_fallbacks(s: Settings | None = None) -> int:
    """Return maximum allowed provider fallbacks."""
    s = s or get_settings()
    return _to_int(getattr(s, 'max_data_fallbacks', 2), 2)


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
    return _to_bool(getattr(s, 'alpaca_empty_to_backup', True), True)


def sentiment_retry_max(s: Settings | None = None) -> int:
    """Return maximum sentiment fetch retry count (defaults to 5 attempts)."""
    s = s or get_settings()
    return _to_int(getattr(s, 'sentiment_max_retries', 5), 5)


def sentiment_backoff_base(s: Settings | None = None) -> float:
    """Return base delay for sentiment fetch backoff (defaults to 5 seconds)."""
    s = s or get_settings()
    return _to_float(getattr(s, 'sentiment_backoff_base', 5.0), 5.0)


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
