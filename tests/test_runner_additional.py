from ai_trading.core import bot_engine  # replace old bot import
from typing import Any, cast

from tests.test_bot import _DummyTradingClient


def test_runner_starts():
    ctx = bot_engine.ctx
    cast(Any, ctx).api = _DummyTradingClient()
    symbols = ["AAPL", "MSFT"]
    summary = bot_engine.pre_trade_health_check(ctx, symbols)
    assert "checked" in summary
