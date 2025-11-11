from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest


def test_gap_failover_promotes_high_res_and_clears_safe_mode(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    from ai_trading.data import fetch
    from ai_trading.data.fetch import fallback_order
    from ai_trading.data.fetch import validators
    from ai_trading.data import provider_monitor
    from ai_trading.execution import live_trading

    caplog.set_level(logging.DEBUG)

    fallback_order.reset()
    validators.reset_gap_statistics()

    monkeypatch.setenv("FINNHUB_API_KEY", "test-token")

    class DummySettings:
        backup_data_provider = "yahoo"

    monkeypatch.setattr(fetch, "get_settings", lambda: DummySettings(), raising=False)

    def fake_finnhub(symbol: str, start: datetime, end: datetime, interval: str):
        index = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
        frame = pd.DataFrame(
            {
                "timestamp": index,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100,
            }
        )
        frame.attrs["data_provider"] = "finnhub"
        return frame

    monkeypatch.setattr(fetch, "_finnhub_get_bars", fake_finnhub)

    start = datetime(2024, 5, 1, 14, 0, tzinfo=UTC)
    gap_times = [start, start + timedelta(minutes=2)]
    primary = pd.DataFrame(
        {
            "timestamp": gap_times,
            "open": [1.0, 1.2],
            "high": [1.1, 1.3],
            "low": [0.9, 1.1],
            "close": [1.0, 1.2],
            "volume": [100, 120],
        }
    )
    primary.attrs["data_provider"] = "alpaca"

    tz = ZoneInfo("UTC")
    repaired, metadata, used_backup = fetch._repair_rth_minute_gaps(
        primary,
        symbol="AVGO",
        start=start,
        end=start + timedelta(minutes=3),
        tz=tz,
    )

    assert used_backup is True
    assert metadata["fallback_provider"].startswith("finnhub")
    assert repaired is not None
    assert not getattr(repaired, "empty", True)

    stats = validators.get_gap_statistics("AVGO")["AVGO"]
    assert stats["fallback_provider"].startswith("finnhub")

    provider_monitor._gap_events.clear()
    provider_monitor._SAFE_MODE_ACTIVE = False
    provider_monitor._SAFE_MODE_REASON = None
    provider_monitor._SAFE_MODE_HEALTHY_PASSES = 0

    for _ in range(provider_monitor._GAP_EVENT_THRESHOLD):
        provider_monitor.record_minute_gap_event(metadata)

    assert metadata.get("fallback_contiguous") is True
    assert not provider_monitor.is_safe_mode_active()

    annotations = {
        "fallback_quote_age": 0.25,
        "fallback_quote_ok": True,
        "fallback_quote_timestamp": datetime.now(UTC) - timedelta(seconds=0.25),
        "gap_limit": 0.5,
    }
    allowed, details = live_trading._maybe_accept_backup_quote(
        annotations,
        provider_hint="finnhub",
        gap_ratio_value=metadata.get("gap_ratio", 0.0),
        min_quote_fresh_ms=1500.0,
        quote_age_ms=5000.0,
        quote_timestamp_present=False,
    )

    assert allowed is True
    assert isinstance(details.get("timestamp"), datetime)
    assert details["provider"] == "finnhub"

    promoted = fallback_order.resolve_promoted_provider("AVGO")
    assert promoted == "finnhub"

    assert any(record.message == "GAP_EVENT_SUPPRESSED" for record in caplog.records)
