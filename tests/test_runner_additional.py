from ai_trading.core import bot_engine  # replace old bot import
from typing import Any, cast


class _DummyTradingClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def test_runner_starts():
    ctx = bot_engine.ctx
    cast(Any, ctx).api = _DummyTradingClient()
    symbols = ["AAPL", "MSFT"]
    summary = bot_engine.pre_trade_health_check(ctx, symbols)
    assert "checked" in summary
