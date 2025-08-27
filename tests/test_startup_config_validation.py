import pytest

from ai_trading.__main__ import _validate_startup_config
from ai_trading.settings import get_settings


def _clear_settings_cache():
    try:
        get_settings.cache_clear()
    except AttributeError:
        pass


def test_invalid_timeframe_raises(monkeypatch):
    monkeypatch.setenv("TIMEFRAME", "10Min")
    monkeypatch.setenv("DATA_FEED", "iex")
    _clear_settings_cache()
    with pytest.raises(SystemExit):
        _validate_startup_config()


def test_invalid_data_feed_raises(monkeypatch):
    monkeypatch.setenv("TIMEFRAME", "1Min")
    monkeypatch.setenv("DATA_FEED", "badfeed")
    _clear_settings_cache()
    with pytest.raises(SystemExit):
        _validate_startup_config()
