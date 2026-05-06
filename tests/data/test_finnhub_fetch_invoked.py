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
    fetcher = type("DummyFinnhubFetcher", (), {})()

    def fail_backup(*args, **kwargs):
        raise AssertionError("backup provider should not be called")

    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fail_backup)

    called: dict[str, bool] = {}

    def fake_fetch(symbol, start, end, resolution="1"):
        called["used"] = True
        return pd.DataFrame({"timestamp": [pd.Timestamp(start)], "close": [1.0]})

    setattr(fetcher, "fetch", fake_fetch)
    monkeypatch.setattr(data_fetcher, "fh_fetcher", fetcher, raising=False)

    mark_called: dict[str, bool] = {}
    monkeypatch.setattr(data_fetcher, "_mark_fallback", lambda *a, **k: mark_called.setdefault("called", True))

    start = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2023, 1, 2, tzinfo=dt.UTC)
    df = data_fetcher.get_minute_df("AAPL", start, end)

    assert called.get("used")
    assert mark_called.get("called")
    assert not df.empty


def test_early_finnhub_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setenv("FINNHUB_API_KEY", "test")
    monkeypatch.setenv("BACKUP_DATA_PROVIDER", "yahoo")

    monkeypatch.setattr(data_fetcher, "_fetch_bars", lambda *a, **k: pd.DataFrame())
    fetcher = type("DummyFinnhubFetcher", (), {})()

    def fail_fetch(*_args, **_kwargs):  # pragma: no cover - should never be called
        raise AssertionError("Finnhub fetch should be skipped when disabled")

    setattr(fetcher, "fetch", fail_fetch)
    monkeypatch.setattr(data_fetcher, "fh_fetcher", fetcher, raising=False)
    monkeypatch.setattr(
        data_fetcher,
        "_safe_backup_get_bars",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2023-01-01T00:00:00Z")],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1],
            }
        ),
    )

    start = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2023, 1, 2, tzinfo=dt.UTC)
    df = data_fetcher.get_minute_df("AAPL", start, end)

    assert not df.empty
    assert df["close"].iloc[-1] == 1.0
