from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai_trading.data.historical_bars import (
    filter_historical_bars_window,
    load_historical_bars,
)


def test_load_historical_bars_handles_mixed_timestamps_and_pathologies(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "event_time": [
                "2025-01-03T00:00:00Z",
                "2025-01-01 00:00:00+00:00",
                "not-a-timestamp",
                "2025-01-02T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-04T00:00:00Z",
            ],
            "open": [102.0, 100.0, 99.0, 0.0, 101.0, 104.0],
            "high": [102.5, 100.5, 99.5, 0.0, 101.5, 104.5],
            "low": [101.5, 99.5, 98.5, 0.0, 100.5, 103.5],
            "close": [102.0, 100.0, 99.0, 0.0, 101.0, 104.0],
        }
    ).to_csv(csv_path, index=False)

    frame, report = load_historical_bars(csv_path, timestamp_col="event_time")

    assert list(frame.index.astype(str)) == [
        "2025-01-01 00:00:00+00:00",
        "2025-01-02 00:00:00+00:00",
        "2025-01-03 00:00:00+00:00",
        "2025-01-04 00:00:00+00:00",
    ]
    assert report.timestamp_column == "event_time"
    assert report.missing_volume_filled is True
    assert report.invalid_timestamp_rows == 1
    assert report.duplicate_timestamp_rows == 2
    assert report.non_positive_rows_dropped == 0


def test_load_historical_bars_drops_non_positive_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad_prices.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-03T00:00:00Z",
            ],
            "open": [100.0, -1.0, 102.0],
            "high": [101.0, -0.5, 103.0],
            "low": [99.0, -2.0, 101.0],
            "close": [100.5, -1.5, 102.5],
            "volume": [10.0, 20.0, 30.0],
        }
    ).to_csv(csv_path, index=False)

    frame, report = load_historical_bars(csv_path)

    assert len(frame) == 2
    assert report.non_positive_rows_dropped == 1


def test_load_historical_bars_uses_range_index_when_no_datetime_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "range.csv"
    pd.DataFrame(
        {
            "seq": [0, 1, 2],
            "open": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "close": [100.2, 101.2, 102.2],
        }
    ).to_csv(csv_path, index=False)

    frame, report = load_historical_bars(csv_path, timestamp_col="event_time")

    assert list(frame.index) == [0, 1, 2]
    assert report.inferred_range_index is True
    assert report.timestamp_column is None
    assert report.missing_volume_filled is True


def test_filter_historical_bars_window_can_return_empty_frame(tmp_path: Path) -> None:
    csv_path = tmp_path / "window.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
            "open": [100.0, 101.0],
            "high": [100.5, 101.5],
            "low": [99.5, 100.5],
            "close": [100.2, 101.2],
            "volume": [10.0, 10.0],
        }
    ).to_csv(csv_path, index=False)

    frame, _report = load_historical_bars(csv_path)
    filtered = filter_historical_bars_window(
        frame,
        start="2026-01-01",
        end="2026-01-02",
    )
    assert filtered.empty


def test_filter_historical_bars_window_rejects_range_index_with_date_filters(tmp_path: Path) -> None:
    csv_path = tmp_path / "range.csv"
    pd.DataFrame(
        {
            "seq": [0, 1, 2],
            "open": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "close": [100.2, 101.2, 102.2],
        }
    ).to_csv(csv_path, index=False)

    frame, _report = load_historical_bars(csv_path, timestamp_col="event_time")
    with pytest.raises(ValueError, match="date filters require timestamped historical bars"):
        filter_historical_bars_window(frame, start="2025-01-01")


def test_load_historical_bars_rejects_missing_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "open": [100.0],
            "high": [101.0],
            "close": [100.5],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_historical_bars(csv_path)
