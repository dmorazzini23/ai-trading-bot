from datetime import UTC, datetime, timedelta

import pandas as pd

from ai_trading.data import fetch


def test_fallback_frame_is_usable_rejects_nan_close():
    now = datetime.now(UTC)
    start = now - timedelta(minutes=2)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start - timedelta(minutes=1), periods=2, freq="1min", tz="UTC"),
            "close": [float("nan"), float("nan")],
        }
    )

    assert fetch._fallback_frame_is_usable(frame, start, now) is False


def test_fallback_frame_is_usable_accepts_recent_bars():
    now = datetime.now(UTC)
    start = now - timedelta(minutes=2)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=3, freq="1min", tz="UTC"),
            "close": [101.0, 101.5, 102.0],
        }
    )

    assert fetch._fallback_frame_is_usable(frame, start, now) is True


def test_fallback_frame_is_usable_accepts_recent_datetime_index():
    now = datetime.now(UTC)
    start = now - timedelta(minutes=2)
    index = pd.date_range(start, periods=3, freq="1min", tz="UTC")
    frame = pd.DataFrame({"close": [101.0, 101.5, 102.0]}, index=index)

    assert fetch._fallback_frame_is_usable(frame, start, now) is True


def test_fallback_frame_is_usable_rejects_stale_datetime_index():
    now = datetime.now(UTC)
    start = now - timedelta(minutes=2)
    index = pd.date_range(start - timedelta(hours=1), periods=2, freq="1min", tz="UTC")
    frame = pd.DataFrame({"close": [101.0, 101.5]}, index=index)

    assert fetch._fallback_frame_is_usable(frame, start, now) is False


def test_fallback_frame_is_usable_rejects_intraday_without_timestamp_or_index():
    now = datetime.now(UTC)
    start = now - timedelta(minutes=2)
    frame = pd.DataFrame({"close": [101.0, 101.5]})

    assert fetch._fallback_frame_is_usable(frame, start, now) is False


def test_safe_backup_get_bars_logs_and_marks_failures(monkeypatch, caplog):
    def fail_backup(*_args):
        raise RuntimeError("backup offline")

    monkeypatch.setattr(fetch, "_backup_get_bars", fail_backup)
    caplog.set_level("WARNING")

    frame = fetch._safe_backup_get_bars("AAPL", "2026-01-01", "2026-01-02", "1m")

    assert frame.empty
    assert frame.attrs["fallback_failure_reason"] == "RuntimeError"
    assert "backup offline" in frame.attrs["fallback_failure_error"]
    assert any(record.message == "BACKUP_GET_BARS_FAILED" for record in caplog.records)
