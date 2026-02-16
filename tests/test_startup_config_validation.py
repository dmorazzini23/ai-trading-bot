import pytest

from ai_trading.__main__ import _validate_startup_config
from ai_trading.settings import get_settings


def _clear_settings_cache():
    try:
        get_settings.cache_clear()
    except AttributeError:
        pass


def _clear_alias_env(monkeypatch) -> None:
    keys = (
        "MAX_ORDER_DOLLARS",
        "AI_TRADING_MAX_ORDER_DOLLARS",
        "MAX_ORDER_SHARES",
        "AI_TRADING_MAX_ORDER_SHARES",
        "PRICE_COLLAR_PCT",
        "AI_TRADING_PRICE_COLLAR_PCT",
        "MAX_ORDERS_PER_MINUTE_GLOBAL",
        "AI_TRADING_ORDERS_PER_MIN_GLOBAL",
        "MAX_ORDERS_PER_MINUTE_PER_SYMBOL",
        "AI_TRADING_ORDERS_PER_MIN_SYMBOL",
        "MAX_CANCELS_PER_MINUTE_GLOBAL",
        "AI_TRADING_CANCELS_PER_MIN",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_invalid_timeframe_raises(monkeypatch):
    _clear_alias_env(monkeypatch)
    monkeypatch.setenv("TIMEFRAME", "10Min")
    monkeypatch.setenv("DATA_FEED", "iex")
    _clear_settings_cache()
    with pytest.raises(SystemExit):
        _validate_startup_config()


def test_invalid_data_feed_raises(monkeypatch):
    _clear_alias_env(monkeypatch)
    monkeypatch.setenv("TIMEFRAME", "1Min")
    monkeypatch.setenv("DATA_FEED", "badfeed")
    _clear_settings_cache()
    with pytest.raises(SystemExit):
        _validate_startup_config()


def test_env_alias_mismatch_raises(monkeypatch):
    _clear_alias_env(monkeypatch)
    monkeypatch.setenv("TIMEFRAME", "1Min")
    monkeypatch.setenv("DATA_FEED", "iex")
    monkeypatch.setenv("MAX_ORDERS_PER_MINUTE_GLOBAL", "20")
    monkeypatch.setenv("AI_TRADING_ORDERS_PER_MIN_GLOBAL", "21")
    _clear_settings_cache()

    with pytest.raises(SystemExit, match="conflicting duplicated env keys"):
        _validate_startup_config()


def test_env_alias_matching_values_pass(monkeypatch):
    _clear_alias_env(monkeypatch)
    monkeypatch.setenv("TIMEFRAME", "1Min")
    monkeypatch.setenv("DATA_FEED", "iex")
    monkeypatch.setenv("MAX_ORDERS_PER_MINUTE_GLOBAL", "020")
    monkeypatch.setenv("AI_TRADING_ORDERS_PER_MIN_GLOBAL", "20")
    _clear_settings_cache()

    cfg = _validate_startup_config()
    assert cfg.timeframe == "1Min"
    assert cfg.data_feed == "iex"
