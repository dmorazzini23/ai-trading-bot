from ai_trading.config.management import TradingConfig


def test_modern_env_keys_satisfy_tradingconfig(monkeypatch):
    monkeypatch.setenv("AI_TRADING_BUY_THRESHOLD", "0.4")
    monkeypatch.setenv("AI_TRADING_CONF_THRESHOLD", "0.8")
    monkeypatch.setenv("AI_TRADING_MAX_DRAWDOWN_THRESHOLD", "0.08")
    cfg = TradingConfig.from_env({})
    assert cfg.buy_threshold == 0.4
    assert cfg.conf_threshold == 0.8
    assert cfg.max_drawdown_threshold == 0.08
