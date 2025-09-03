import logging
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core.bot_engine import DataFetcher, _init_metrics


def _stub_daily_df():
    index = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [100, 100],
        },
        index=index,
    )


def test_daily_fetcher_uses_cache(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="ai_trading.core.bot_engine")
    _init_metrics()

    monkeypatch.setattr(
        "ai_trading.core.bot_engine.get_settings",
        lambda: SimpleNamespace(alpaca_api_key="k", alpaca_secret_key_plain="s"),
    )

    class DummyClient:  # noqa: D401 - simple stub
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(
        "ai_trading.core.bot_engine.StockHistoricalDataClient", DummyClient
    )

    calls = {"count": 0}

    def fake_safe_get_stock_bars(client, req, symbol, tag):  # noqa: D401
        calls["count"] += 1
        return _stub_daily_df()

    monkeypatch.setattr(
        "ai_trading.core.bot_engine.bars.safe_get_stock_bars", fake_safe_get_stock_bars
    )

    fetcher = DataFetcher()
    ctx = SimpleNamespace()

    df1 = fetcher.get_daily_df(ctx, "AAPL")
    df2 = fetcher.get_daily_df(ctx, "AAPL")

    assert calls["count"] == 1
    assert df1 is df2
    messages = [rec.message for rec in caplog.records]
    assert "DAILY_FETCH_CACHE_HIT" in messages
