"""Ensure TradingConfig supports kwargs used at call sites."""

from ai_trading.config.management import TradingConfig


def test_trading_config_accepts_buy_threshold_and_capital_cap():
    cfg = TradingConfig(buy_threshold=0.5, capital_cap=0.2)
    assert cfg.buy_threshold == 0.5
    assert cfg.capital_cap == 0.2

