"""Ensure Finnhub disabled logs are deduped per window."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.logging import warn_finnhub_disabled_no_data


def test_warn_finnhub_disabled_no_data_dedupes(caplog):
    caplog.set_level("INFO")

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=1)

    warn_finnhub_disabled_no_data("AAPL", timeframe="1Min", start=start, end=end)
    warn_finnhub_disabled_no_data("AAPL", timeframe="1Min", start=start, end=end)

    records = [rec for rec in caplog.records if rec.message == "FINNHUB_DISABLED_NO_DATA"]
    assert len(records) == 1

    caplog.clear()
    later_end = end + timedelta(days=1)
    warn_finnhub_disabled_no_data("AAPL", timeframe="1Min", start=start, end=later_end)
    records = [rec for rec in caplog.records if rec.message == "FINNHUB_DISABLED_NO_DATA"]
    assert len(records) == 1
