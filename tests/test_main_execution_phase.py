from __future__ import annotations

from datetime import UTC, datetime
import logging

from ai_trading import main
from ai_trading.telemetry import runtime_state


def test_set_execution_phase_updates_runtime_state() -> None:
    runtime_state.reset_all_states()

    main._set_execution_phase(
        "bootstrap",
        status="warming_up",
        reason="startup",
        cycle_index=0,
    )
    snapshot = runtime_state.observe_service_status()
    assert snapshot.get("status") == "warming_up"
    assert snapshot.get("phase") == "bootstrap"
    assert snapshot.get("cycle_index") == 0
    assert isinstance(snapshot.get("phase_since"), str)

    main._set_execution_phase(
        "active",
        status="ready",
        reason="startup_complete",
        cycle_index=1,
    )
    updated = runtime_state.observe_service_status()
    assert updated.get("status") == "ready"
    assert updated.get("phase") == "active"
    assert updated.get("cycle_index") == 1


def test_emit_cycle_slo_alerts_emits_compute_and_skips_stale_provider_when_primary_steady(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_CYCLE_SLO_ALERTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SLO_CYCLE_COMPUTE_WARN_MS", "1000")
    monkeypatch.setenv("AI_TRADING_SLO_CYCLE_COMPUTE_CRIT_MS", "2000")
    monkeypatch.setenv("AI_TRADING_SLO_PROVIDER_TELEMETRY_STALE_WARN_SEC", "5")
    monkeypatch.setattr(
        main.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "updated": "2000-01-01T00:00:00+00:00",
            "active": "alpaca",
            "primary": "alpaca-iex",
            "status": "healthy",
        },
    )

    events: list[tuple[str, str]] = []

    def _capture_alert(name: str, **kwargs):
        events.append((name, str(kwargs.get("severity", ""))))

    monkeypatch.setattr(main, "emit_runtime_alert", _capture_alert)

    main._emit_cycle_slo_alerts(cycle_index=3, compute_ms=2500.0, closed=False)

    assert ("ALERT_CYCLE_COMPUTE_CRITICAL", "critical") in events
    assert not any(name == "ALERT_PROVIDER_TELEMETRY_STALE" for name, _ in events)


def test_emit_cycle_market_snapshot_respects_cadence(monkeypatch, caplog) -> None:
    monkeypatch.setenv("AI_TRADING_MARKET_SNAPSHOT_EVERY_N_CYCLES", "2")
    monkeypatch.setattr(
        main.runtime_state,
        "observe_data_provider_state",
        lambda: {"updated": "2026-02-24T15:00:00+00:00", "status": "healthy", "active": "alpaca", "reason": None, "safe_mode": False},
    )
    monkeypatch.setattr(
        main.runtime_state,
        "observe_quote_status",
        lambda: {"status": "fresh", "allowed": True},
    )
    monkeypatch.setattr(
        main.runtime_state,
        "observe_broker_status",
        lambda: {"status": "reachable", "connected": True, "last_error": None},
    )
    monkeypatch.setattr(
        main.runtime_state,
        "observe_service_status",
        lambda: {"status": "ready", "phase": "active"},
    )

    caplog.set_level(logging.INFO)
    main._emit_cycle_market_snapshot(cycle_index=1, closed=False, interval_s=60)
    assert not any(record.message == "CYCLE_MARKET_SNAPSHOT" for record in caplog.records)

    main._emit_cycle_market_snapshot(cycle_index=2, closed=False, interval_s=60)
    assert any(record.message == "CYCLE_MARKET_SNAPSHOT" for record in caplog.records)


def test_emit_cycle_market_snapshot_uses_closed_cadence_override(monkeypatch, caplog) -> None:
    monkeypatch.setenv("AI_TRADING_MARKET_SNAPSHOT_EVERY_N_CYCLES", "1")
    monkeypatch.setenv("AI_TRADING_MARKET_SNAPSHOT_EVERY_N_CYCLES_CLOSED", "3")
    monkeypatch.setattr(
        main.runtime_state,
        "observe_data_provider_state",
        lambda: {"updated": "2026-02-24T15:00:00+00:00", "status": "healthy", "active": "alpaca", "reason": None, "safe_mode": False},
    )
    monkeypatch.setattr(
        main.runtime_state,
        "observe_quote_status",
        lambda: {"status": "fresh", "allowed": True},
    )
    monkeypatch.setattr(
        main.runtime_state,
        "observe_broker_status",
        lambda: {"status": "reachable", "connected": True, "last_error": None},
    )
    monkeypatch.setattr(
        main.runtime_state,
        "observe_service_status",
        lambda: {"status": "ready", "phase": "active"},
    )

    caplog.set_level(logging.INFO)
    main._emit_cycle_market_snapshot(cycle_index=1, closed=True, interval_s=60)
    main._emit_cycle_market_snapshot(cycle_index=2, closed=True, interval_s=60)
    assert not any(record.message == "CYCLE_MARKET_SNAPSHOT" for record in caplog.records)

    main._emit_cycle_market_snapshot(cycle_index=3, closed=True, interval_s=60)
    assert any(record.message == "CYCLE_MARKET_SNAPSHOT" for record in caplog.records)


def test_emit_cycle_slo_alerts_uses_closed_thresholds(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_CYCLE_SLO_ALERTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SLO_CYCLE_COMPUTE_WARN_MS", "1000")
    monkeypatch.setenv("AI_TRADING_SLO_CYCLE_COMPUTE_CRIT_MS", "2000")
    monkeypatch.setenv("AI_TRADING_SLO_CYCLE_COMPUTE_WARN_MS_CLOSED", "30000")
    monkeypatch.setenv("AI_TRADING_SLO_CYCLE_COMPUTE_CRIT_MS_CLOSED", "60000")
    monkeypatch.setattr(
        main.runtime_state,
        "observe_data_provider_state",
        lambda: {"updated": None, "active": "alpaca", "status": "healthy"},
    )
    events: list[tuple[str, str]] = []

    def _capture_alert(name: str, **kwargs):
        events.append((name, str(kwargs.get("severity", ""))))

    monkeypatch.setattr(main, "emit_runtime_alert", _capture_alert)

    main._emit_cycle_slo_alerts(cycle_index=4, compute_ms=2500.0, closed=True)

    assert not any(name.startswith("ALERT_CYCLE_COMPUTE_") for name, _ in events)


def test_emit_cycle_slo_alerts_emits_prolonged_primary_fallback_alert(monkeypatch) -> None:
    main._PRIMARY_FALLBACK_STREAK_SINCE_TS = None
    main._PRIMARY_FALLBACK_LAST_ALERT_TS = 0.0
    monkeypatch.setenv("AI_TRADING_CYCLE_SLO_ALERTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SLO_PROVIDER_TELEMETRY_STALE_WARN_SEC", "999999")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_ALERT_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_WARN_SEC", "5")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_CRIT_SEC", "10")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_ALERT_COOLDOWN_SEC", "0")
    monkeypatch.setattr(
        main.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "updated": datetime.now(UTC).isoformat(),
            "active": "yahoo",
            "reason": "upstream_unavailable",
            "status": "degraded",
            "timeframes": {"1Min": True},
        },
    )
    clock = {"value": 100.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["value"])

    events: list[tuple[str, str]] = []

    def _capture_alert(name: str, **kwargs):
        events.append((name, str(kwargs.get("severity", ""))))

    monkeypatch.setattr(main, "emit_runtime_alert", _capture_alert)

    main._emit_cycle_slo_alerts(cycle_index=1, compute_ms=50.0, closed=False)
    clock["value"] = 108.0
    main._emit_cycle_slo_alerts(cycle_index=2, compute_ms=50.0, closed=False)
    clock["value"] = 112.0
    main._emit_cycle_slo_alerts(cycle_index=3, compute_ms=50.0, closed=False)

    assert ("ALERT_PRIMARY_FEED_FALLBACK_PROLONGED", "warning") in events
    assert ("ALERT_PRIMARY_FEED_FALLBACK_PROLONGED", "critical") in events


def test_emit_cycle_slo_alerts_can_include_daily_fallback(monkeypatch) -> None:
    main._PRIMARY_FALLBACK_STREAK_SINCE_TS = None
    main._PRIMARY_FALLBACK_LAST_ALERT_TS = 0.0
    monkeypatch.setenv("AI_TRADING_CYCLE_SLO_ALERTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SLO_PROVIDER_TELEMETRY_STALE_WARN_SEC", "999999")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_ALERT_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_WARN_SEC", "1")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_CRIT_SEC", "2")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_ALERT_COOLDOWN_SEC", "0")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_INCLUDE_DAILY", "1")
    monkeypatch.setattr(
        main.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "updated": datetime.now(UTC).isoformat(),
            "active": "yahoo",
            "reason": "upstream_unavailable",
            "status": "degraded",
            "timeframes": {"1Day": True},
        },
    )
    clock = {"value": 200.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["value"])
    emitted: list[str] = []
    monkeypatch.setattr(main, "emit_runtime_alert", lambda name, **_kwargs: emitted.append(name))

    main._emit_cycle_slo_alerts(cycle_index=1, compute_ms=10.0, closed=False)
    clock["value"] = 202.0
    main._emit_cycle_slo_alerts(cycle_index=2, compute_ms=10.0, closed=False)

    assert "ALERT_PRIMARY_FEED_FALLBACK_PROLONGED" in emitted


def test_emit_cycle_slo_alerts_suppresses_fallback_alert_when_provider_telemetry_stale(
    monkeypatch,
) -> None:
    main._PRIMARY_FALLBACK_STREAK_SINCE_TS = None
    main._PRIMARY_FALLBACK_LAST_ALERT_TS = 0.0
    main._PROVIDER_TELEMETRY_STALE_LAST_ALERT_TS = 0.0
    monkeypatch.setenv("AI_TRADING_CYCLE_SLO_ALERTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SLO_PROVIDER_TELEMETRY_STALE_WARN_SEC", "5")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_ALERT_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_WARN_SEC", "1")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_CRIT_SEC", "2")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_ALERT_COOLDOWN_SEC", "0")
    monkeypatch.setattr(
        main.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "updated": "2000-01-01T00:00:00+00:00",
            "active": "yahoo",
            "reason": "upstream_unavailable",
            "status": "degraded",
            "timeframes": {"1Min": True},
        },
    )
    monkeypatch.setattr(main.time, "time", lambda: 100.0)
    emitted: list[str] = []
    monkeypatch.setattr(main, "emit_runtime_alert", lambda name, **_kwargs: emitted.append(name))

    main._emit_cycle_slo_alerts(cycle_index=1, compute_ms=10.0, closed=False)

    assert "ALERT_PROVIDER_TELEMETRY_STALE" in emitted
    assert "ALERT_PRIMARY_FEED_FALLBACK_PROLONGED" not in emitted


def test_emit_cycle_slo_alerts_applies_provider_telemetry_stale_alert_cooldown(
    monkeypatch,
) -> None:
    main._PROVIDER_TELEMETRY_STALE_LAST_ALERT_TS = 0.0
    monkeypatch.setenv("AI_TRADING_CYCLE_SLO_ALERTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SLO_PROVIDER_TELEMETRY_STALE_WARN_SEC", "5")
    monkeypatch.setenv("AI_TRADING_SLO_PROVIDER_TELEMETRY_STALE_ALERT_COOLDOWN_SEC", "300")
    monkeypatch.setenv("AI_TRADING_SLO_PRIMARY_FALLBACK_ALERT_ENABLED", "0")
    monkeypatch.setattr(
        main.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "updated": "2000-01-01T00:00:00+00:00",
            "active": "yahoo",
            "status": "degraded",
        },
    )
    clock = {"value": 100.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["value"])
    emitted: list[str] = []
    monkeypatch.setattr(main, "emit_runtime_alert", lambda name, **_kwargs: emitted.append(name))

    main._emit_cycle_slo_alerts(cycle_index=1, compute_ms=10.0, closed=False)
    clock["value"] = 120.0
    main._emit_cycle_slo_alerts(cycle_index=2, compute_ms=10.0, closed=False)

    assert emitted.count("ALERT_PROVIDER_TELEMETRY_STALE") == 1
