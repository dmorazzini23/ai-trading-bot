"""Tests for execution guard helpers."""

from datetime import UTC, datetime, timedelta

from ai_trading.execution import guards


def _reset_state() -> None:
    guards.STATE.shadow_cycle = False
    guards.STATE.stale_symbols = 0
    guards.STATE.universe_size = 0


def test_shadow_cycle_persists_after_stale_ratio_trip() -> None:
    _reset_state()
    guards.begin_cycle(universe_size=10, degraded=False)
    for _ in range(4):
        guards.mark_symbol_stale()
    guards.end_cycle(stale_threshold_ratio=0.30)
    assert not guards.shadow_active()

    guards.begin_cycle(universe_size=8, degraded=False)
    assert not guards.shadow_active()

    guards.end_cycle(stale_threshold_ratio=0.50)
    guards.begin_cycle(universe_size=5, degraded=False)
    assert not guards.shadow_active()


def test_shadow_cycle_respects_degraded_flag() -> None:
    _reset_state()
    guards.begin_cycle(universe_size=12, degraded=True)
    assert not guards.shadow_active()

    guards.end_cycle(stale_threshold_ratio=0.30)
    guards.begin_cycle(universe_size=12, degraded=False)
    assert not guards.shadow_active()


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
