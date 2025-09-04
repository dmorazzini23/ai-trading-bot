import logging
import types
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from ai_trading.data import fetch


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def test_backoff_uses_alternate_provider(monkeypatch, caplog):
    start, end = _dt_range()
    symbol = "AAPL"
    fetch._EMPTY_BAR_COUNTS.clear()
    fetch._SKIPPED_SYMBOLS.clear()
    key = (symbol, "1Min")
    fetch._EMPTY_BAR_COUNTS[key] = fetch._EMPTY_BAR_THRESHOLD - 1

    def _raise_empty(*_a, **_k):
        raise fetch.EmptyBarsError("empty_bars")

    monkeypatch.setattr(fetch, "_fetch_bars", _raise_empty)
    monkeypatch.setattr(fetch, "fh_fetcher", None)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")

    called: dict[str, object] = {}

    def _sleep(sec):
        called["sleep"] = sec

    monkeypatch.setattr(fetch, "time", types.SimpleNamespace(sleep=_sleep))

    df = pd.DataFrame(
        {
            "timestamp": [start],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
        }
    )

    def _alt_fetch(sym, s, e, interval="1m"):
        called["alt"] = True
        return df

    monkeypatch.setattr(fetch, "_yahoo_get_bars", _alt_fetch)

    with caplog.at_level(logging.WARNING):
        out = fetch.get_minute_df(symbol, start, end)

    assert not out.empty
    assert called.get("alt")
    assert called.get("sleep")
    assert key not in fetch._SKIPPED_SYMBOLS
    assert any(r.message == "ALPACA_EMPTY_BAR_BACKOFF" for r in caplog.records)


def test_backoff_skips_when_alternate_empty(monkeypatch, caplog):
    start, end = _dt_range()
    symbol = "MSFT"
    fetch._EMPTY_BAR_COUNTS.clear()
    fetch._SKIPPED_SYMBOLS.clear()
    key = (symbol, "1Min")
    fetch._EMPTY_BAR_COUNTS[key] = fetch._EMPTY_BAR_THRESHOLD - 1

    def _raise_empty(*_a, **_k):
        raise fetch.EmptyBarsError("empty_bars")

    monkeypatch.setattr(fetch, "_fetch_bars", _raise_empty)
    monkeypatch.setattr(fetch, "fh_fetcher", None)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setattr(fetch, "time", types.SimpleNamespace(sleep=lambda _s: None))

    monkeypatch.setattr(fetch, "_yahoo_get_bars", lambda *a, **k: pd.DataFrame())

    with caplog.at_level(logging.WARNING):
        out = fetch.get_minute_df(symbol, start, end)

    assert out.empty
    assert key in fetch._SKIPPED_SYMBOLS
    assert any(r.message == "ALPACA_EMPTY_BAR_BACKOFF" for r in caplog.records)


def test_retry_limit_raises(monkeypatch, caplog):
    start, end = _dt_range()
    symbol = "TSLA"
    fetch._EMPTY_BAR_COUNTS.clear()
    fetch._SKIPPED_SYMBOLS.clear()
    key = (symbol, "1Min")
    fetch._EMPTY_BAR_COUNTS[key] = fetch._EMPTY_BAR_MAX_RETRIES

    def _raise_empty(*_a, **_k):
        raise fetch.EmptyBarsError("empty_bars")

    monkeypatch.setattr(fetch, "_fetch_bars", _raise_empty)
    monkeypatch.setattr(fetch, "fh_fetcher", None)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(fetch.EmptyBarsError):
            fetch.get_minute_df(symbol, start, end)

    assert any(r.message == "ALPACA_EMPTY_BAR_MAX_RETRIES" for r in caplog.records)


def test_skip_retry_when_market_closed(monkeypatch, caplog):
    start, end = _dt_range()
    symbol = "IBM"
    fetch._EMPTY_BAR_COUNTS.clear()
    fetch._SKIPPED_SYMBOLS.clear()

    def _raise_empty(*_a, **_k):
        raise fetch.EmptyBarsError("empty_bars")

    monkeypatch.setattr(fetch, "_fetch_bars", _raise_empty)
    monkeypatch.setattr(fetch, "fh_fetcher", None)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setattr(fetch, "is_market_open", lambda: False)

    with caplog.at_level(logging.INFO):
        out = fetch.get_minute_df(symbol, start, end)

    assert out.empty
    assert any(
        r.message == "ALPACA_EMPTY_BAR_MARKET_CLOSED" for r in caplog.records
    )


def test_feed_switch_and_window_shrink(monkeypatch):
    symbol = "ZZZZ"
    fetch._EMPTY_BAR_COUNTS.clear()
    fetch._SKIPPED_SYMBOLS.clear()
    key = (symbol, "1Min")
    fetch._EMPTY_BAR_COUNTS[key] = fetch._EMPTY_BAR_THRESHOLD - 1

    calls: list[tuple[str, datetime, datetime]] = []

    def _raise_empty(_symbol, start, end, timeframe, *, feed=fetch._DEFAULT_FEED, adjustment="raw"):
        calls.append((feed, start, end))
        raise fetch.EmptyBarsError("empty")

    monkeypatch.setattr(fetch, "_fetch_bars", _raise_empty)
    monkeypatch.setattr(fetch, "fh_fetcher", None)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(fetch, "time", types.SimpleNamespace(sleep=lambda _s: None))
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)

    df = pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
        "open": [1],
        "high": [1],
        "low": [1],
        "close": [1],
    })
    monkeypatch.setattr(fetch, "_yahoo_get_bars", lambda *a, **k: df)

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=3)
    fetch.get_minute_df(symbol, start, end)

    feeds = [f for f, _, _ in calls]
    assert "sip" in feeds
    starts = [s for _, s, _ in calls]
    assert any(s > start for s in starts)
