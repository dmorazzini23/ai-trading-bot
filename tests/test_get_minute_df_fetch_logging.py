import logging
from datetime import UTC, datetime, timedelta

import pytest

import ai_trading.data.fetch as data_fetcher

pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _force_window(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def _reset_state(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_MINUTE_CACHE", {})
    monkeypatch.setattr(data_fetcher, "_EMPTY_BAR_COUNTS", {})
    monkeypatch.setattr(data_fetcher, "_SKIPPED_SYMBOLS", set())


def test_fetch_success_no_error_logged(monkeypatch, caplog):
    start, end = _dt_range()
    df = pd.DataFrame({"t": [start], "o": [1], "h": [1], "l": [1], "c": [1], "v": [1]})
    _reset_state(monkeypatch)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True)
    monkeypatch.setattr(data_fetcher, "_fetch_bars", lambda *a, **k: df)

    called = {"yahoo": False}

    def _yahoo(*a, **k):
        called["yahoo"] = True
        return pd.DataFrame()

    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", _yahoo)

    with caplog.at_level(logging.WARNING):
        out = data_fetcher.get_minute_df("AAPL", start, end)

    assert out is df
    assert not called["yahoo"]
    assert all(r.message != "ALPACA_FETCH_FAILED" for r in caplog.records)


@pytest.mark.parametrize("exc_type", [ValueError, RuntimeError])
def test_fetch_error_logs_and_sets_none(monkeypatch, caplog, exc_type):
    start, end = _dt_range()
    _reset_state(monkeypatch)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True)

    def _fail(*a, **k):
        raise exc_type("boom")

    monkeypatch.setattr(data_fetcher, "_fetch_bars", _fail)

    fallback_df = pd.DataFrame({"t": [start], "o": [1], "h": [1], "l": [1], "c": [1], "v": [1]})
    yahoo_called = {"called": False}

    def _yahoo(*a, **k):
        yahoo_called["called"] = True
        return fallback_df

    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", _yahoo)

    with caplog.at_level(logging.WARNING):
        out = data_fetcher.get_minute_df("AAPL", start, end)

    assert yahoo_called["called"]
    assert list(out.loc[:, ["open", "high", "low", "close", "volume"]].iloc[0]) == [1, 1, 1, 1, 1]
    assert list(out.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert out.index.name == "timestamp"
    assert any(r.message == "ALPACA_FETCH_FAILED" for r in caplog.records)
