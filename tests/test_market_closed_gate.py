import types

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine


def test_market_closed_gate(monkeypatch):
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(rth_only=True, allow_extended=False)
    monkeypatch.setattr(bot_engine, "_resolve_trading_config", lambda ctx: cfg)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda now=None: False)
    order = bot_engine.submit_order(types.SimpleNamespace(), "AAPL", 1, "buy", price=100.0)
    assert order is None
