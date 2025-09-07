"""Test minute-to-daily resampling fallback."""

from datetime import UTC, datetime

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.data import bars
from tests.support.assert_df_like import assert_df_like


def test_get_daily_bars_resamples_minutes(monkeypatch):
    """When daily bars are empty, minute bars are resampled."""  # AI-AGENT-REF
    empty = pd.DataFrame()
    monkeypatch.setattr(
        bars,
        "_fetch_daily_bars",
        lambda client, symbol, start, end, feed=None, **kwargs: empty,
    )

    idx = pd.date_range(
        "2024-01-02 14:30",
        periods=5,
        freq="1min",
        tz="UTC",
    )
    data = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [10, 10, 10, 10, 10],
        },
        index=idx,
    )
    monkeypatch.setattr(
        bars,
        "_get_minute_bars",
        lambda symbol, start_dt, end_dt, feed, adjustment=None: data,
    )

    out = bars.get_daily_bars(
        "SPY", None, datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)
    )
    assert_df_like(out, columns=["open", "high", "low", "close", "volume"])
    assert not out.empty
    assert len(out) == 1
    assert float(out.iloc[0]["open"]) == 1


def test_resample_passes_adjustment(monkeypatch):
    """Minute fallback should forward the adjustment parameter."""
    empty = pd.DataFrame()
    monkeypatch.setattr(
        bars,
        "_fetch_daily_bars",
        lambda client, symbol, start, end, feed=None, **kwargs: empty,
    )

    idx = pd.date_range(
        "2024-01-02 14:30",
        periods=1,
        freq="1min",
        tz="UTC",
    )
    data = pd.DataFrame(
        {
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [10],
        },
        index=idx,
    )

    captured: dict[str, str | None] = {}

    def fake_get_minute_bars(symbol, start_dt, end_dt, feed, adjustment=None):
        captured["adjustment"] = adjustment
        return data

    monkeypatch.setattr(bars, "_get_minute_bars", fake_get_minute_bars)

    bars.get_daily_bars(
        "SPY", None, datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)
    )

    assert captured["adjustment"] == bars.get_settings().alpaca_adjustment
