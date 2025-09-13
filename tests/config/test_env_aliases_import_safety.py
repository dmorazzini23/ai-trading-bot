import importlib


def test_import_succeeds_with_required_threshold(monkeypatch):
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.1")
    monkeypatch.delenv("AI_TRADING_MAX_DRAWDOWN_THRESHOLD", raising=False)
    import ai_trading.risk.kelly as kelly
    importlib.reload(kelly)


def test_alias_for_drawdown_threshold(monkeypatch):
    monkeypatch.delenv("MAX_DRAWDOWN_THRESHOLD", raising=False)
    monkeypatch.setenv("AI_TRADING_MAX_DRAWDOWN_THRESHOLD", "0.08")
    from ai_trading.config.management import TradingConfig

    cfg = TradingConfig.from_env()
    assert cfg.max_drawdown_threshold == 0.08

