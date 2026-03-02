import importlib

import pytest


def test_import_succeeds_with_required_threshold(monkeypatch):
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.1")
    monkeypatch.delenv("AI_TRADING_MAX_DRAWDOWN_THRESHOLD", raising=False)
    import ai_trading.risk.kelly as kelly
    importlib.reload(kelly)


def test_alias_for_drawdown_threshold(monkeypatch):
    monkeypatch.delenv("MAX_DRAWDOWN_THRESHOLD", raising=False)
    monkeypatch.setenv("AI_TRADING_MAX_DRAWDOWN_THRESHOLD", "0.08")
    from ai_trading.config.management import TradingConfig

    with pytest.raises(RuntimeError, match="AI_TRADING_MAX_DRAWDOWN_THRESHOLD"):
        TradingConfig.from_env()
