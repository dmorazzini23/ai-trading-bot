import logging
from types import SimpleNamespace

import pandas as pd

import ai_trading.data.bars as data_bars
import ai_trading.data_fetcher as data_fetcher
from ai_trading.core import bot_engine


def test_minute_fallback_uses_http_yahoo(monkeypatch, caplog):
    ctx = SimpleNamespace()
    ctx.data_fetcher = SimpleNamespace(get_daily_df=lambda ctx, sym: pd.DataFrame())
    ctx.data_client = object()

    session = SimpleNamespace(
        open=pd.Timestamp("2024-01-01 09:30", tz="UTC"),
        close=pd.Timestamp("2024-01-01 16:00", tz="UTC"),
    )
    monkeypatch.setattr(bot_engine, "last_market_session", lambda _: session)

    def boom(*args, **kwargs):
        raise AttributeError("'NoneType' object has no attribute 'get'")

    monkeypatch.setattr(data_bars, "safe_get_stock_bars", boom)

    called = {}

    def fake_get_minute_df(symbol, start, end, feed=None):
        called["called"] = True
        rng = pd.date_range("2024-01-01 09:30", periods=390, freq="T", tz="UTC")
        return pd.DataFrame({"timestamp": rng, "close": 1.0})

    monkeypatch.setattr(data_fetcher, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(bot_engine, "get_minute_df", data_fetcher.get_minute_df)

    with caplog.at_level(logging.INFO):
        bot_engine.data_source_health_check(ctx, ["AAPL"])

    assert called.get("called")
    record = next(r for r in caplog.records if "minute fallback ok" in r.message)
    assert record.rows >= 300
