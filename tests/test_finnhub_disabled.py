import datetime as dt
import pytest

pd = pytest.importorskip("pandas")

from ai_trading import data_fetcher


def test_get_minute_df_returns_empty_when_finnhub_disabled(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")

    def fail_fetch(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("Finnhub fetch should be skipped")

    monkeypatch.setattr(data_fetcher.fh_fetcher, "fetch", fail_fetch)
    monkeypatch.setattr(data_fetcher, "_fetch_bars", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", lambda *a, **k: pd.DataFrame())

    df = data_fetcher.get_minute_df("AAPL", dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 2))
    assert df.empty
