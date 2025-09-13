import datetime as dt
import logging
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import fetch as data_fetcher
from ai_trading.config import settings as config_settings
from ai_trading.data.fetch.metrics import inc_backup_provider_used


def test_switches_to_backup_provider(monkeypatch, caplog):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setenv("BACKUP_DATA_PROVIDER", "yahoo")
    config_settings.get_settings.cache_clear()

    start = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    end = start + dt.timedelta(minutes=1)

    def empty_fetch(*a, **k):
        return pd.DataFrame()

    called: dict[str, bool] = {}

    def fake_backup(symbol, start, end, interval):
        called["used"] = True
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(start)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [0],
            }
        )

    monkeypatch.setattr(data_fetcher, "_fetch_bars", empty_fetch)
    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", fake_backup)
    before = inc_backup_provider_used("yahoo", "AAPL")
    with caplog.at_level(logging.INFO):
        df = data_fetcher.get_minute_df("AAPL", start, end)

    after = inc_backup_provider_used("yahoo", "AAPL")
    assert called.get("used")
    assert not df.empty
    assert after == before + 1
    assert any(r.message == "BACKUP_PROVIDER_USED" for r in caplog.records)
