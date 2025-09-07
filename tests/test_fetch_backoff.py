from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from ai_trading.data.fetch import backoff as fb


def _dt_range(days: int = 3):
    end = datetime(2024, 1, 4, tzinfo=UTC)
    start = end - timedelta(days=days)
    return start, end


def test_fetch_feed_switches_to_alternate_and_shrinks_window(monkeypatch):
    start, end = _dt_range(3)
    calls: list[tuple[str, datetime, datetime]] = []

    def fake_fetch(symbol, s, e, timeframe, *, feed):
        calls.append((feed, s, e))
        if feed == "iex":
            raise fb.EmptyBarsError("empty")
        return pd.DataFrame({"timestamp": [s]})

    monkeypatch.setattr(fb, "_fetch_bars", fake_fetch)
    fb._EMPTY_BAR_COUNTS.clear()
    fb._SKIPPED_SYMBOLS.clear()

    out = fb._fetch_feed("TEST", start, end, "1Min", feed="iex")
    assert not out.empty
    assert calls[0][0] == "iex"
    assert calls[1][0] == "sip"
    assert calls[1][1] > start  # window shrunk
    assert ("TEST", "1Min") not in fb._SKIPPED_SYMBOLS


def test_fetch_feed_adds_skip_list_after_max_retries(monkeypatch):
    start, end = _dt_range(1)

    def always_empty(*a, **k):
        raise fb.EmptyBarsError("empty")

    monkeypatch.setattr(fb, "_fetch_bars", always_empty)
    fb._SKIPPED_SYMBOLS.clear()
    key = ("FAIL", "1Min")
    fb._EMPTY_BAR_COUNTS[key] = fb._EMPTY_BAR_MAX_RETRIES - 1

    with pytest.raises(fb.EmptyBarsError):
        fb._fetch_feed("FAIL", start, end, "1Min", feed="iex")

    assert key in fb._SKIPPED_SYMBOLS
