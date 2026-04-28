import pytest

from ai_trading.config import management as config
from ai_trading.config import settings as settings_mod


def test_get_env_entrypoint(monkeypatch):
    """Canonical get_env returns casted values and handles required flags."""
    monkeypatch.setenv("FOO", "123")
    assert config.get_env("FOO") == "123"
    assert config.get_env("FOO", cast=int) == 123
    assert config.get_env("BAR", "baz") == "baz"
    with pytest.raises(RuntimeError):
        config.get_env("MISSING", required=True)


def test_runtime_override_wins_in_merged_snapshot(monkeypatch):
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    config.set_runtime_env_override(
        "ALPACA_TRADING_BASE_URL",
        "https://paper-api.alpaca.markets",
    )

    try:
        snapshot = config.merged_env_snapshot()
        base_url, source, errors = config._select_alpaca_base_url()
    finally:
        config.clear_runtime_env_override("ALPACA_TRADING_BASE_URL")

    assert snapshot["ALPACA_TRADING_BASE_URL"] == "https://paper-api.alpaca.markets"
    assert base_url == "https://paper-api.alpaca.markets"
    assert source == "ALPACA_TRADING_BASE_URL"
    assert errors == []


def test_get_settings_singleton():
    """Management and settings helpers share the same Settings instance."""
    assert config.get_settings() is settings_mod.get_settings()
