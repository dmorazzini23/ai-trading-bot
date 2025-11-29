"""Ensure Alpaca credentials load from ALPACA_* env vars without APCA_* fallbacks."""

from ai_trading.config import settings as config_settings
from ai_trading.settings import _secret_to_str


def test_alpaca_settings_prioritize_alpaca_env(monkeypatch):
    config_settings.get_settings.cache_clear()
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("AI_TRADING_ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("AI_TRADING_ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("AP" "CA_API_KEY_ID", raising=False)
    monkeypatch.delenv("AP" "CA_API_SECRET_KEY", raising=False)

    monkeypatch.setenv("ALPACA_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "alpaca-secret")

    settings_obj = config_settings.get_settings()
    try:
        assert settings_obj.alpaca_api_key == "alpaca-key"
        assert _secret_to_str(settings_obj.alpaca_secret_key) == "alpaca-secret"
    finally:
        config_settings.get_settings.cache_clear()
