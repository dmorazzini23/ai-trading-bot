from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest


def test_repair_rth_minute_gaps_backfill_suppresses_safe_mode(monkeypatch):
    pd = pytest.importorskip("pandas")

    from ai_trading.data import fetch as fetch_mod

    start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    end = start + timedelta(minutes=3)
    base_index = pd.date_range(start, periods=3, freq="min", tz="UTC")

    df = pd.DataFrame(
        {
            "timestamp": [base_index[0].isoformat(), base_index[2].isoformat()],
            "open": [100.0, 101.0],
            "high": [100.5, 101.5],
            "low": [99.5, 100.5],
            "close": [100.2, 101.2],
            "volume": [1000, 1200],
        }
    )
    df.attrs["data_provider"] = "alpaca"

    fallback_df = pd.DataFrame(
        {
            "timestamp": [base_index[1].isoformat()],
            "open": [100.4],
            "high": [100.6],
            "low": [100.1],
            "close": [100.5],
            "volume": [1100],
        }
    )

    events: list[dict] = []

    monkeypatch.setattr(fetch_mod, "_safe_backup_get_bars", lambda *_, **__: fallback_df)
    monkeypatch.setattr(fetch_mod, "record_minute_gap_event", lambda payload: events.append(payload))

    repaired, metadata, used_backup = fetch_mod._repair_rth_minute_gaps(
        df,
        symbol="AAPL",
        start=start,
        end=end,
        tz=ZoneInfo("America/New_York"),
    )

    assert used_backup is True
    assert metadata["missing_after"] == 0
    assert metadata["residual_gap"] is False
    assert repaired is not None
    assert len(repaired) == 3
    assert events == []
