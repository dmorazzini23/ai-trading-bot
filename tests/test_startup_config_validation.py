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


@pytest.fixture(autouse=True)
def _startup_env_defaults(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_ENABLED", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)


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


def test_live_mode_requires_oms_intent_store_enabled(monkeypatch):
    _clear_alias_env(monkeypatch)
    monkeypatch.setenv("TIMEFRAME", "1Min")
    monkeypatch.setenv("DATA_FEED", "iex")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_ENABLED", "0")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://user:pass@db.example.com:5432/postgres",
    )
    _clear_settings_cache()

    with pytest.raises(SystemExit, match="AI_TRADING_OMS_INTENT_STORE_ENABLED=1"):
        _validate_startup_config()


def test_live_mode_requires_non_sqlite_database_url(monkeypatch):
    _clear_alias_env(monkeypatch)
    monkeypatch.setenv("TIMEFRAME", "1Min")
    monkeypatch.setenv("DATA_FEED", "iex")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_ENABLED", "1")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///runtime/oms_intents.db")
    _clear_settings_cache()

    with pytest.raises(SystemExit, match="requires DATABASE_URL to a non-sqlite database"):
        _validate_startup_config()


def test_live_mode_accepts_postgres_database_url(monkeypatch):
    _clear_alias_env(monkeypatch)
    monkeypatch.setenv("TIMEFRAME", "1Min")
    monkeypatch.setenv("DATA_FEED", "iex")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_ENABLED", "1")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgres://user:pass@db.example.com:5432/postgres",
    )
    _clear_settings_cache()

    cfg = _validate_startup_config()
    assert cfg.timeframe == "1Min"
    assert cfg.data_feed == "iex"
