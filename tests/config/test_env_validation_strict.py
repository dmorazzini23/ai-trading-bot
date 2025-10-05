import pytest

from ai_trading.config.runtime import TradingConfig


def test_health_tick_seconds_enforces_minimum(monkeypatch):
    monkeypatch.delenv("HEALTH_TICK_SECONDS", raising=False)
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.1")

    with pytest.raises(ValueError, match=r"health_tick_seconds must be >= 30"):
        TradingConfig.from_env(
            {
                "MAX_DRAWDOWN_THRESHOLD": "0.1",
                "HEALTH_TICK_SECONDS": "29",
            }
        )


def test_unknown_trading_env_var_raises(monkeypatch):
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.1")
    monkeypatch.delenv("AI_TRADING_FAKE_FLAG", raising=False)

    with pytest.raises(RuntimeError, match=r"AI_TRADING_FAKE_FLAG"):
        TradingConfig.from_env(
            {
                "MAX_DRAWDOWN_THRESHOLD": "0.1",
                "AI_TRADING_FAKE_FLAG": "1",
            },
            allow_missing_drawdown=True,
        )
