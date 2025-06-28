import bot_engine  # replace old bot import


def test_runner_starts():
    ctx = bot_engine.ctx
    symbols = ["AAPL", "MSFT"]
    summary = bot_engine.pre_trade_health_check(ctx, symbols)
    assert isinstance(summary, dict)
