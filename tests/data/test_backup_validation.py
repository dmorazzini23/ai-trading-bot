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


def test_no_session_http_fallback_rejects_unusable_frame(monkeypatch, caplog):
    start = datetime(2024, 1, 7, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    stale = pd.DataFrame(
        {
            "timestamp": [start - timedelta(days=3)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
        }
    )

    class DummySession:
        def get(self, *_args, **_kwargs):
            class Response:
                status_code = 200
                headers = {"Content-Type": "application/json"}
                text = "{\"bars\":[]}"

                def json(self):
                    return {"bars": []}

            return Response()

    monkeypatch.setenv("ENABLE_HTTP_FALLBACK", "1")
    monkeypatch.setattr(fetch, "_HTTP_SESSION", DummySession())
    monkeypatch.setattr(fetch, "_ENABLE_HTTP_FALLBACK", True, raising=False)
    monkeypatch.setattr(fetch, "DATA_HTTP_FALLBACK_INTRADAY", True, raising=False)
    monkeypatch.setattr(fetch, "DATA_HTTP_FALLBACK_OUT_OF_SESSION", True, raising=False)
    monkeypatch.setattr(fetch, "is_trading_day", lambda _day: False)
    monkeypatch.setattr(fetch, "_yahoo_get_bars", lambda *_args, **_kwargs: stale)
    fetch._clear_minute_fallback_state("AAPL", "1Min", start, end)

    caplog.set_level("WARNING")

    frame = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert frame is not None
    assert frame.empty
    assert fetch._used_fallback("AAPL", "1Min", start, end) is False
    assert any(record.message == "BACKUP_DATA_REJECTED" for record in caplog.records)
