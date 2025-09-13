import datetime as dt
import pytest

pd = pytest.importorskip("pandas")
from ai_trading.data import fetch as data_fetcher


@pytest.fixture(autouse=True)
def _force_window(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)


def test_finnhub_called_when_alpaca_none(monkeypatch):
    monkeypatch.setenv("ENABLE_FINNHUB", "1")
    monkeypatch.setenv("FINNHUB_API_KEY", "test")

    # Alpaca returns None
    monkeypatch.setattr(data_fetcher, "_fetch_bars", lambda *a, **k: None)
    monkeypatch.setattr(data_fetcher.fh_fetcher, "is_stub", False)

    def fail_backup(*args, **kwargs):
        raise AssertionError("backup provider should not be called")

    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fail_backup)

    called = {}

    def fake_fetch(symbol, start, end, resolution="1"):
        called["used"] = True
        return pd.DataFrame({"timestamp": [pd.Timestamp(start)], "close": [1.0]})

    monkeypatch.setattr(data_fetcher.fh_fetcher, "fetch", fake_fetch)

    mark_called = {}
    monkeypatch.setattr(data_fetcher, "_mark_fallback", lambda *a, **k: mark_called.setdefault("called", True))

    start = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2023, 1, 2, tzinfo=dt.UTC)
    df = data_fetcher.get_minute_df("AAPL", start, end)

    assert called.get("used")
    assert mark_called.get("called")
    assert not df.empty
