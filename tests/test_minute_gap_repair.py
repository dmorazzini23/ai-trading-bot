from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import ai_trading.data.fetch as fetch_module


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
    missing = {expected_local[5].tz_convert("UTC"), expected_local[25].tz_convert("UTC")}
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
