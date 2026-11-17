from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import cast
from zoneinfo import ZoneInfo

import pytest




def test_repair_rth_minute_gaps_zero_volume_not_gap(monkeypatch):
    pd = pytest.importorskip("pandas")

    from ai_trading.data import fetch

    tz = ZoneInfo("America/New_York")
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=5)
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": [100, 0, 120, 0, 140],
        }
    )
    frame.attrs["data_provider"] = "alpaca_iex"
    gap_events: list[dict] = []

    monkeypatch.setattr(
        fetch,
        "record_minute_gap_event",
        lambda payload: gap_events.append(payload),
    )

    repaired, metadata, used_backup = fetch._repair_rth_minute_gaps(
        frame,
        symbol="AAPL",
        start=start,
        end=end,
        tz=tz,
    )

    assert used_backup is False
    assert metadata.get("gap_ratio") == 0.0
    assert metadata.get("missing_after") == 0
    assert metadata.get("gap_over_limit") is False
    assert gap_events == []
    assert repaired is frame
