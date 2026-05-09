"""Yahoo fallback should split oversized intraday windows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import types

import pytest

from ai_trading.data import fetch as fetch_mod


@pytest.mark.parametrize("days", [10])
def test_backup_split_large_intraday_window(monkeypatch, days):
    pd = pytest.importorskip("pandas")

    monkeypatch.setattr(
        fetch_mod,
        "_current_settings",
        lambda: types.SimpleNamespace(backup_data_provider="yahoo", alpaca_adjustment="raw"),
    )
    monkeypatch.setattr(fetch_mod, "_has_alpaca_keys", lambda: False)
    monkeypatch.setattr(fetch_mod, "fh_fetcher", None)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setenv("FINNHUB_API_KEY", "")
    monkeypatch.setattr(fetch_mod, "warn_finnhub_disabled_no_data", lambda *a, **k: None)
    monkeypatch.setattr(fetch_mod, "log_finnhub_disabled", lambda *a, **k: None)
    monkeypatch.setattr(fetch_mod.provider_monitor, "active_provider", lambda primary, backup: primary)
    monkeypatch.setattr(fetch_mod.provider_monitor, "record_switchover", lambda *a, **k: None)
    monkeypatch.setattr(fetch_mod, "_post_process", lambda df, **_: df)
    monkeypatch.setattr(fetch_mod, "_verify_minute_continuity", lambda df, **_: df)
    monkeypatch.setattr(fetch_mod, "_repair_rth_minute_gaps", lambda df, **_: (df, {"expected": 0, "missing_after": 0, "gap_ratio": 0.0}, False))
    monkeypatch.setattr(fetch_mod, "_http_fallback_permitted", lambda *a, **k: True)
    monkeypatch.setattr(fetch_mod, "mark_success", lambda *a, **k: None)
    monkeypatch.setattr(fetch_mod, "_mark_fallback", lambda *a, **k: None)
    monkeypatch.setattr(fetch_mod, "_pytest_active", lambda: False)

    calls: list[tuple[datetime, datetime]] = []

    def fake_backup(symbol, start, end, interval):
        calls.append((start, end))
        frame = pd.DataFrame({"timestamp": [start], "close": [1.0]})
        frame.attrs["data_provider"] = "yahoo"
        frame.attrs["data_feed"] = "yahoo"
        return frame

    monkeypatch.setattr(fetch_mod, "_backup_get_bars", fake_backup)

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=days)

    df = fetch_mod.get_bars("AAPL", "1Min", start, end, feed=None)

    assert df is not None
    assert calls  # backup provider should be exercised
    max_span = max((end - start for start, end in calls), default=timedelta())
    assert max_span <= timedelta(days=10)


def test_chunked_yahoo_sparse_coverage_meta_marks_gap() -> None:
    pd = pytest.importorskip("pandas")

    start = datetime(2025, 8, 20, 13, 30, tzinfo=UTC)
    end = datetime(2025, 8, 20, 20, 0, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [start, start + timedelta(minutes=1)],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
    )
    frame.attrs["yf_1m_range_split"] = True

    meta = fetch_mod._yahoo_chunked_coverage_meta(
        frame,
        start=start,
        end=end,
        tz=fetch_mod.ZoneInfo("America/New_York"),
    )

    assert meta["expected"] == 390
    assert meta["missing_after"] == 388
    assert meta["gap_ratio"] > 0.99
