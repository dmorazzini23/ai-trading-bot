import logging
from types import SimpleNamespace

import ai_trading.data.bars as data_bars
import pytest
pd = pytest.importorskip("pandas")
from ai_trading import data_fetcher
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

    # AI-AGENT-REF: ensure no function object leakage in logs
    assert not any("<function" in (r.message or "") for r in caplog.records)

    # AI-AGENT-REF: if availability/fallback events emitted, extras are canonical
    for r in caplog.records:
        if r.getMessage() in {"DATA_SOURCE_AVAILABLE", "DATA_SOURCE_FALLBACK_ATTEMPT"}:
            d = r.__dict__
            feed = d.get("feed") or (d.get("extra") or {}).get("feed")
            timeframe = d.get("timeframe") or (d.get("extra") or {}).get("timeframe")
            if feed is not None:
                assert feed in {"iex", "sip"}
            if timeframe is not None:
                assert timeframe in {"1Min", "1Day"}
