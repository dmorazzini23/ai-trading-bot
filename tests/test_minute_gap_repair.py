from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import ai_trading.data.fetch as fetch_module
from ai_trading.data.market_calendar import rth_session_utc


def _build_base_frame(start_local: datetime, end_local: datetime, missing: set[pd.Timestamp]) -> pd.DataFrame:
    tz = ZoneInfo("America/New_York")
    index_local = pd.date_range(start_local, end_local, freq="min", tz=tz, inclusive="left")
    index_utc = index_local.tz_convert("UTC")
    rows: list[dict[str, object]] = []
    for ts in index_utc:
        if ts in missing:
            continue
        rows.append(
            {
                "timestamp": ts,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["symbol"] = "AAPL"
    return frame


def test_repair_rth_gap_uses_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    tz = ZoneInfo("America/New_York")
    start_local = datetime(2024, 1, 2, 9, 30, tzinfo=tz)
    end_local = datetime(2024, 1, 2, 16, 0, tzinfo=tz)
    expected_local = pd.date_range(start_local, end_local, freq="min", tz=tz, inclusive="left")
    missing = {expected_local[i].tz_convert("UTC") for i in range(0, 30)}
    base_df = _build_base_frame(start_local, end_local, missing)

    captured: dict[str, pd.Timestamp] = {}

    def fake_backup(symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        captured["start"] = start
        captured["end"] = end
        rows = []
        for ts in sorted(missing):
            rows.append(
                {
                    "timestamp": ts,
                    "open": 2.0,
                    "high": 2.0,
                    "low": 2.0,
                    "close": 2.0,
                    "volume": 50.0,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(fetch_module, "_backup_get_bars", fake_backup)
    monkeypatch.setenv("AI_TRADING_GAP_RATIO_LIMIT", "0.05")
    repaired, meta, used_backup = fetch_module._repair_rth_minute_gaps(  # type: ignore[attr-defined]
        base_df,
        symbol="AAPL",
        start=start_local.astimezone(UTC),
        end=end_local.astimezone(UTC),
        tz=tz,
    )

    assert used_backup is True
    assert meta["missing_after"] == 0
    assert captured["start"] == min(missing)
    assert captured["end"] == max(missing) + timedelta(minutes=1)
    repaired_index = pd.to_datetime(repaired["timestamp"], utc=True)
    for ts in missing:
        assert ts in set(repaired_index)

def test_repair_rth_gap_local_backfill_under_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    tz = ZoneInfo("America/New_York")
    start_local = datetime(2024, 1, 2, 9, 30, tzinfo=tz)
    end_local = datetime(2024, 1, 2, 10, 30, tzinfo=tz)
    expected_local = pd.date_range(start_local, end_local, freq="min", tz=tz, inclusive="left")
    missing = {expected_local[i].tz_convert("UTC") for i in (5, 25)}
    base_df = _build_base_frame(start_local, end_local, missing)

    backup_calls = {"count": 0}

    def fake_backup(symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        backup_calls["count"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(fetch_module, "_backup_get_bars", fake_backup)
    monkeypatch.setenv("AI_TRADING_GAP_RATIO_LIMIT", "0.05")

    repaired, meta, used_backup = fetch_module._repair_rth_minute_gaps(  # type: ignore[attr-defined]
        base_df,
        symbol="AAPL",
        start=start_local.astimezone(UTC),
        end=end_local.astimezone(UTC),
        tz=tz,
    )

    assert used_backup is False
    assert backup_calls["count"] == 0
    assert meta["local_backfill"] is True
    assert meta["backup_fill_suppressed"] is True
    assert meta["missing_after"] == 0
    repaired_index = pd.to_datetime(repaired["timestamp"], utc=True)
    for ts in missing:
        assert ts in set(repaired_index)


def test_should_skip_symbol_full_gap_triggers_skip(caplog: pytest.LogCaptureFixture) -> None:
    tz = ZoneInfo("America/New_York")
    start_local = datetime(2024, 1, 3, 9, 30, tzinfo=tz)
    end_local = datetime(2024, 1, 3, 16, 0, tzinfo=tz)
    expected_local = pd.date_range(start_local, end_local, freq="min", tz=tz, inclusive="left")
    full_df = _build_base_frame(start_local, end_local, set())
    df = full_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]) - pd.Timedelta(days=1)
    df.attrs["symbol"] = "SKIPX"
    fetch_module._SKIP_LOGGED.clear()  # type: ignore[attr-defined]
    caplog.set_level("WARNING")
    should_skip = fetch_module.should_skip_symbol(
        df,
        window=(start_local.astimezone(UTC), end_local.astimezone(UTC)),
        tz=tz,
        max_gap_ratio=0.0,
    )
    assert should_skip is True
    assert any(
        "SKIP_SYMBOL_INSUFFICIENT_INTRADAY_COVERAGE" in record.message
        for record in caplog.records
    )
    assert df.attrs["_coverage_meta"]["skip_flagged"] is True


def test_should_skip_symbol_partial_gap_sets_metadata() -> None:
    tz = ZoneInfo("America/New_York")
    start_local = datetime(2024, 1, 4, 9, 30, tzinfo=tz)
    end_local = datetime(2024, 1, 4, 16, 0, tzinfo=tz)
    expected_local = pd.date_range(start_local, end_local, freq="min", tz=tz, inclusive="left")
    missing = {expected_local[i].tz_convert("UTC") for i in range(0, 30)}
    partial_df = _build_base_frame(start_local, end_local, missing)
    partial_df.attrs["symbol"] = "KEEPX"
    fetch_module._SKIP_LOGGED.clear()  # type: ignore[attr-defined]

    should_skip = fetch_module.should_skip_symbol(
        partial_df,
        window=(start_local.astimezone(UTC), end_local.astimezone(UTC)),
        tz=tz,
        max_gap_ratio=0.2,
    )

    assert should_skip is False
    meta = partial_df.attrs.get("_coverage_meta")
    assert isinstance(meta, dict)
    assert meta["expected"] == expected_local.size
    assert meta["missing_after"] == len(missing)
    assert meta["gap_ratio"] == pytest.approx(len(missing) / expected_local.size)
    assert "skip_flagged" not in meta


def test_should_skip_symbol_partial_gap_exceeds_limit(caplog: pytest.LogCaptureFixture) -> None:
    tz = ZoneInfo("America/New_York")
    start_local = datetime(2024, 1, 4, 9, 30, tzinfo=tz)
    end_local = datetime(2024, 1, 4, 16, 0, tzinfo=tz)
    expected_local = pd.date_range(start_local, end_local, freq="min", tz=tz, inclusive="left")
    missing = {expected_local[i].tz_convert("UTC") for i in range(0, 30)}
    partial_df = _build_base_frame(start_local, end_local, missing)
    partial_df.attrs["symbol"] = "SKIPLIM"
    fetch_module._SKIP_LOGGED.clear()  # type: ignore[attr-defined]
    caplog.set_level("INFO")

    should_skip = fetch_module.should_skip_symbol(
        partial_df,
        window=(start_local.astimezone(UTC), end_local.astimezone(UTC)),
        tz=tz,
        max_gap_ratio=0.01,
    )

    assert should_skip is True
    meta = partial_df.attrs.get("_coverage_meta")
    assert isinstance(meta, dict)
    assert meta.get("skip_flagged") is True
    assert "SKIP_SYMBOL_GAP_RATIO_LIMIT" in " ".join(record.msg for record in caplog.records)


def test_should_skip_symbol_zero_gap_records_metadata() -> None:
    tz = ZoneInfo("America/New_York")
    start_local = datetime(2024, 1, 5, 9, 30, tzinfo=tz)
    end_local = datetime(2024, 1, 5, 16, 0, tzinfo=tz)
    df = _build_base_frame(start_local, end_local, set())
    df.attrs["symbol"] = "CLEAN"
    fetch_module._SKIP_LOGGED.clear()  # type: ignore[attr-defined]

    should_skip = fetch_module.should_skip_symbol(
        df,
        window=(start_local.astimezone(UTC), end_local.astimezone(UTC)),
        tz=tz,
        max_gap_ratio=0.0,
    )

    assert should_skip is False
    meta = df.attrs.get("_coverage_meta")
    assert isinstance(meta, dict)
    assert meta["expected"] > 0
    assert meta["missing_after"] == 0
    assert meta["gap_ratio"] == pytest.approx(0.0)
    assert "skip_flagged" not in meta


def test_normalize_window_bounds_limits_to_rth_minutes() -> None:
    tz = ZoneInfo("America/New_York")
    window_start = datetime(2024, 1, 9, 0, 0, tzinfo=UTC)
    window_end = datetime(2024, 1, 9, 23, 59, tzinfo=UTC)
    session_start_utc, session_end_utc = rth_session_utc(window_start.date())
    expected_minutes = int((session_end_utc - session_start_utc).total_seconds() / 60)

    expected_local, start_utc, end_utc = fetch_module._normalize_window_bounds(  # type: ignore[attr-defined]
        window_start,
        window_end,
        tz,
    )

    assert start_utc == window_start
    assert end_utc == window_end
    assert len(expected_local) == expected_minutes
    assert expected_local[0].hour == 9 and expected_local[0].minute == 30
    assert expected_local[-1].hour == 15 and expected_local[-1].minute == 59


def test_should_skip_symbol_ignores_offsession_gap_window() -> None:
    tz = ZoneInfo("America/New_York")
    session_start = datetime(2024, 1, 10, 9, 30, tzinfo=tz)
    session_end = datetime(2024, 1, 10, 16, 0, tzinfo=tz)
    df = _build_base_frame(session_start, session_end, set())
    df.attrs["symbol"] = "RTHONLY"
    fetch_module._SKIP_LOGGED.clear()  # type: ignore[attr-defined]

    window_start = (session_start - timedelta(hours=12)).astimezone(UTC)
    window_end = (session_end + timedelta(hours=12)).astimezone(UTC)
    should_skip = fetch_module.should_skip_symbol(
        df,
        window=(window_start, window_end),
        tz=tz,
        max_gap_ratio=0.05,
    )

    assert should_skip is False
    meta = df.attrs.get("_coverage_meta")
    assert isinstance(meta, dict)
    assert meta["expected"] == int((session_end - session_start).total_seconds() / 60)
    assert meta["missing_after"] == 0
    assert meta["gap_ratio"] == pytest.approx(0.0)


def test_yahoo_gap_interpolation_restores_contiguity() -> None:
    tz = ZoneInfo("America/New_York")
    start_local = datetime(2024, 1, 8, 9, 30, tzinfo=tz)
    end_local = datetime(2024, 1, 8, 9, 36, tzinfo=tz)
    expected_local = pd.date_range(start_local, end_local, freq="min", tz=tz, inclusive="left")
    missing = {expected_local[2].tz_convert("UTC")}
    yahoo_frame = _build_base_frame(start_local, end_local, missing)
    yahoo_frame.attrs["data_provider"] = "yahoo"

    repaired, meta, used_backup = fetch_module._repair_rth_minute_gaps(  # type: ignore[attr-defined]
        yahoo_frame,
        symbol="AAPL",
        start=start_local.astimezone(UTC),
        end=end_local.astimezone(UTC),
        tz=tz,
    )

    assert used_backup is False
    assert meta["missing_after"] == 0
    assert meta["fallback_repaired"] is True
    assert meta["fallback_contiguous"] is True
    assert meta["primary_feed_gap"] is False
    repaired_index = pd.to_datetime(repaired["timestamp"], utc=True)
    assert len(repaired_index) == expected_local.size
    assert repaired_index.is_monotonic_increasing
