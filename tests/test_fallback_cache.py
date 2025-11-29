from types import SimpleNamespace
from datetime import datetime, UTC, timedelta

import pytest

import ai_trading.data.fetch as data_fetcher
from tests.helpers.dummy_http import DummyResp

pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _force_window(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)


def test_alpaca_skipped_after_yahoo_fallback(monkeypatch):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    monkeypatch.setenv("ENABLE_HTTP_FALLBACK", "1")
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True)

    calls = {"alpaca": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["alpaca"] += 1
        return DummyResp({"bars": []})

    monkeypatch.setattr(data_fetcher, "requests", SimpleNamespace(get=fake_get))
    monkeypatch.setattr(data_fetcher._HTTP_SESSION, "get", fake_get, raising=False)

    yahoo_calls = {"n": 0}
    df_fallback = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(start)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
        }
    )

    def fake_yahoo(symbol, s, e, interval):  # noqa: ARG001 - test stub
        yahoo_calls["n"] += 1
        return df_fallback

    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", fake_yahoo)
    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fake_yahoo)

    out1 = data_fetcher.get_minute_df("AAPL", start, end)
    assert not out1.empty
    assert yahoo_calls["n"] == 1
    tf_key = ("AAPL", "1Min")
    skip_until = data_fetcher._BACKUP_SKIP_UNTIL.get(tf_key)
    assert isinstance(skip_until, datetime)
    remaining = skip_until - datetime.now(UTC)
    assert remaining >= timedelta(minutes=9, seconds=50)
    assert tf_key in data_fetcher._SKIPPED_SYMBOLS
    first_calls = calls["alpaca"]

    out2 = data_fetcher.get_minute_df("AAPL", start, end)
    assert not out2.empty
    assert yahoo_calls["n"] == 2
    assert calls["alpaca"] == first_calls

    out3 = data_fetcher.get_minute_df("AAPL", start, end)
    assert not out3.empty
    assert yahoo_calls["n"] == 3
    assert calls["alpaca"] == first_calls
    refreshed_until = data_fetcher._BACKUP_SKIP_UNTIL.get(tf_key)
    assert isinstance(refreshed_until, datetime)
    assert refreshed_until > datetime.now(UTC)
    assert tf_key in data_fetcher._SKIPPED_SYMBOLS
