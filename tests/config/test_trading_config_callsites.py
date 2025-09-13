from ai_trading.config.management import TradingConfig


def test_trading_config_accepts_known_callsites():
    cfg = TradingConfig(buy_threshold=0.5, capital_cap=0.2)
    assert cfg.buy_threshold == 0.5
    assert cfg.capital_cap == 0.2
