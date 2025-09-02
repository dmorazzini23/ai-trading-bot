import logging
import types
from datetime import UTC, datetime, timedelta

import pandas as pd

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
        raise ValueError("empty_bars")

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
        raise ValueError("empty_bars")

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
