from ai_trading.core import bot_engine  # replace old bot import
from tests.test_bot import _DummyTradingClient


def test_runner_starts():
    ctx = bot_engine.ctx
    ctx.api = _DummyTradingClient()
    symbols = ["AAPL", "MSFT"]
    summary = bot_engine.pre_trade_health_check(ctx, symbols)
    assert "checked" in summary
