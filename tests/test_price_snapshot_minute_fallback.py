import datetime as dt
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.portfolio import core as portfolio_core
from ai_trading.data import fetch as data_fetcher
from ai_trading import utils as _utils  # type: ignore


def test_price_snapshot_minute_fallback(monkeypatch):
    ctx = SimpleNamespace(
        data_fetcher=SimpleNamespace(get_daily_df=lambda ctx, s: pd.DataFrame()),
        data_client=object(),
    )

    def fake_safe_get_stock_bars(client, request, symbol, context=""):
        ts = _utils.time.utcnow()  # AI-AGENT-REF: avoid datetime.utcnow
        return pd.DataFrame({"timestamp": [pd.Timestamp(ts)], "close": [123.0]})

    class DummyRequest:
        def __init__(self, *args, **kwargs):
            self.feed = kwargs.get("feed")
            self.timeframe = kwargs.get("timeframe")

    monkeypatch.setattr(portfolio_core, "safe_get_stock_bars", fake_safe_get_stock_bars)
    monkeypatch.setattr(portfolio_core, "StockBarsRequest", DummyRequest)
    monkeypatch.setattr(portfolio_core, "TimeFrame", lambda *a, **k: None)
    monkeypatch.setattr(portfolio_core, "TimeFrameUnit", SimpleNamespace(Minute=None))

    price = portfolio_core.get_latest_price(ctx, "SPY")
    assert price == 123.0


def test_yahoo_minute_split_long_range(monkeypatch, caplog):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setattr(data_fetcher, "_fetch_bars", lambda *a, **k: pd.DataFrame())

    calls: list[tuple[dt.datetime, dt.datetime]] = []

    def fake_yahoo(symbol, start, end, interval):
        calls.append((start, end))
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(start)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [float(len(calls))],
                "volume": [0],
            }
        )

    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", fake_yahoo)

    start = dt.datetime(2024, 1, 1)
    end = dt.datetime(2024, 1, 20)
    with caplog.at_level("WARNING"):
        df = data_fetcher.get_minute_df("AAPL", start, end)

    assert list(df["close"]) == [1.0, 2.0, 3.0]
    assert len(calls) == 3
    for s, e in calls:
        assert e - s <= dt.timedelta(days=8)
    assert any("YF_1" in str(r.msg) for r in caplog.records)
