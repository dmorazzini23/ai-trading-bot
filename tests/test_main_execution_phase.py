from __future__ import annotations

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


def test_emit_cycle_slo_alerts_emits_compute_and_stale_provider(monkeypatch) -> None:
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
            "status": "healthy",
        },
    )

    events: list[tuple[str, str]] = []

    def _capture_alert(name: str, **kwargs):
        events.append((name, str(kwargs.get("severity", ""))))

    monkeypatch.setattr(main, "emit_runtime_alert", _capture_alert)

    main._emit_cycle_slo_alerts(cycle_index=3, compute_ms=2500.0, closed=False)

    assert ("ALERT_CYCLE_COMPUTE_CRITICAL", "critical") in events
    assert any(name == "ALERT_PROVIDER_TELEMETRY_STALE" for name, _ in events)


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

