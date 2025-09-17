from ai_trading.config.management import TradingConfig


def test_modern_env_keys_satisfy_tradingconfig(monkeypatch):
    monkeypatch.setenv("BUY_THRESHOLD", "0.1")
    monkeypatch.setenv("CONF_THRESHOLD", "0.6")
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.2")
    monkeypatch.setenv("MAX_POSITION_SIZE", "5000")
    monkeypatch.setenv("POSITION_SIZE_MIN_USD", "150")
    monkeypatch.setenv("CONFIDENCE_LEVEL", "0.5")
    monkeypatch.setenv("KELLY_FRACTION_MAX", "0.15")
    monkeypatch.setenv("MIN_SAMPLE_SIZE", "3")
    monkeypatch.setenv("AI_TRADING_BUY_THRESHOLD", "0.4")
    monkeypatch.setenv("AI_TRADING_CONF_THRESHOLD", "0.8")
    monkeypatch.setenv("AI_TRADING_MAX_DRAWDOWN_THRESHOLD", "0.08")
    monkeypatch.setenv("AI_TRADING_MAX_POSITION_SIZE", "9000")
    monkeypatch.setenv("AI_TRADING_POSITION_SIZE_MIN_USD", "25")
    monkeypatch.setenv("AI_TRADING_CONFIDENCE_LEVEL", "0.85")
    monkeypatch.setenv("AI_TRADING_KELLY_FRACTION_MAX", "0.20")
    monkeypatch.setenv("AI_TRADING_MIN_SAMPLE_SIZE", "12")
    cfg = TradingConfig.from_env({})
    assert cfg.buy_threshold == 0.4
    assert cfg.conf_threshold == 0.8
    assert cfg.max_drawdown_threshold == 0.08
    assert cfg.max_position_size == 9000
    assert cfg.position_size_min_usd == 25
    assert cfg.confidence_level == 0.85
    assert cfg.kelly_fraction_max == 0.20
    assert cfg.min_sample_size == 12
