"""Tests for execution guard helpers."""

from datetime import UTC, datetime, timedelta

from ai_trading.execution import guards


def test_can_execute_accepts_iso_quote_timestamp(monkeypatch) -> None:
    now = datetime(2026, 4, 24, 15, 0, tzinfo=UTC)
    monkeypatch.setattr(guards, "_require_bid_ask", lambda: True)

    allowed, reason = guards.can_execute(
        {
            "bid": 100.0,
            "ask": 100.1,
            "timestamp": (now - timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
        },
        now=now,
        max_age_sec=10,
    )

    assert allowed is True
    assert reason is None


def test_can_execute_rejects_future_iso_quote_timestamp(monkeypatch) -> None:
    now = datetime(2026, 4, 24, 15, 0, tzinfo=UTC)
    monkeypatch.setattr(guards, "_require_bid_ask", lambda: True)

    allowed, reason = guards.can_execute(
        {
            "bid": 100.0,
            "ask": 100.1,
            "timestamp": (now + timedelta(seconds=30)).isoformat(),
        },
        now=now,
        max_age_sec=10,
    )

    assert allowed is False
    assert reason == "future_quote_timestamp"
