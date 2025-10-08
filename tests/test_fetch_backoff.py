from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from ai_trading.data import fetch
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
    fetch._SKIPPED_SYMBOLS.clear()

    out = fb._fetch_feed("TEST", start, end, "1Min", feed="iex")
    assert not out.empty
    assert calls[0][0] == "iex"
    assert calls[1][0] == "sip"
    assert calls[1][1] > start  # window shrunk
    assert ("TEST", "1Min") not in fb._SKIPPED_SYMBOLS
    assert ("TEST", "1Min") not in fetch._SKIPPED_SYMBOLS


def test_fetch_feed_adds_skip_list_after_max_retries(monkeypatch):
    start, end = _dt_range(1)

    def always_empty(*a, **k):
        raise fb.EmptyBarsError("empty")

    monkeypatch.setattr(fb, "_fetch_bars", always_empty)
    fb._SKIPPED_SYMBOLS.clear()
    fetch._SKIPPED_SYMBOLS.clear()
    key = ("FAIL", "1Min")
    fb._EMPTY_BAR_COUNTS[key] = fb._EMPTY_BAR_MAX_RETRIES - 1

    with pytest.raises(fb.EmptyBarsError):
        fb._fetch_feed("FAIL", start, end, "1Min", feed="iex")

    assert key in fb._SKIPPED_SYMBOLS
    assert key in fetch._SKIPPED_SYMBOLS


def test_fetch_feed_uses_http_fallback_when_enabled(monkeypatch):
    pd = pytest.importorskip("pandas")
    start, end = _dt_range(1)

    def always_empty(symbol, s, e, timeframe, *, feed):
        raise fb.EmptyBarsError(f"empty:{feed}")

    fb._SKIPPED_SYMBOLS.clear()
    fb._EMPTY_BAR_COUNTS.clear()
    fetch._SKIPPED_SYMBOLS.clear()

    monkeypatch.setattr(fb, "_fetch_bars", always_empty)
    monkeypatch.setattr(fb, "provider_priority", lambda: ["alpaca_iex", "alpaca_sip"])
    monkeypatch.setattr(fb, "max_data_fallbacks", lambda: 2)
    monkeypatch.setattr(fetch, "_ENABLE_HTTP_FALLBACK", True, raising=False)

    fallback_calls: list[tuple[str, str]] = []

    def fake_backup(symbol, s, e, interval):
        fallback_calls.append((symbol, interval))
        return pd.DataFrame({"timestamp": [s]})

    monkeypatch.setattr(fb, "_backup_get_bars", fake_backup)
    monkeypatch.setattr(fetch, "_backup_get_bars", fake_backup)
    monkeypatch.setattr(fb.provider_monitor, "record_switchover", lambda *a, **k: None)

    out = fb._fetch_feed("TEST", start, end, "1Min", feed="iex")
    assert isinstance(out, pd.DataFrame)
    assert not out.empty
    assert fallback_calls == [("TEST", "1m")]


def test_fetch_feed_cooldown_skips_primary_after_fallback(monkeypatch):
    start, end = _dt_range(1)
    call_feeds: list[str] = []

    def always_empty(symbol, s, e, timeframe, *, feed):
        call_feeds.append(feed)
        raise fb.EmptyBarsError("empty")

    def yahoo_success(*_a, **_k):
        return pd.DataFrame({"timestamp": [start]})

    fb._EMPTY_BAR_COUNTS.clear()
    fb._SKIPPED_SYMBOLS.clear()
    fetch._SKIPPED_SYMBOLS.clear()
    fb._PROVIDER_COOLDOWNS.clear()
    fb._PROVIDER_DECISION_CACHE = (0.0, 0.0)

    monkeypatch.setattr(fb, "_fetch_bars", always_empty)
    monkeypatch.setattr(fb, "_yahoo_get_bars", yahoo_success)
    monkeypatch.setattr(fb, "_provider_decision_window", lambda: 120.0)
    times = iter([0.0] * 12)
    monkeypatch.setattr(fb, "monotonic_time", lambda: next(times))

    first = fb._fetch_feed("COOL", start, end, "1Min", feed="iex")
    assert isinstance(first, pd.DataFrame)
    assert not first.empty

    second = fb._fetch_feed("COOL", start, end, "1Min", feed="iex")
    assert isinstance(second, pd.DataFrame)
    assert not second.empty

    assert call_feeds == ["iex"], "primary feed should not be retried during cooldown"
    key = ("COOL", "1Min")
    assert key in fb._PROVIDER_COOLDOWNS
