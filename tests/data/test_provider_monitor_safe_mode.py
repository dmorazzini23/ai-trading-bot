from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import pytest

import ai_trading.data.provider_monitor as pm
from ai_trading.monitoring.alerts import AlertType, AlertSeverity


class _DummyAlerts:
    def __init__(self) -> None:
        self.events: list[tuple[AlertType, AlertSeverity, str, dict | None]] = []

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        *,
        metadata: dict | None = None,
    ) -> None:
        self.events.append((alert_type, severity, message, metadata))


@pytest.mark.parametrize(
    ("record_name", "expected_reason"),
    [
        ("record_unauthorized_sip_event", "unauthorized_sip"),
        ("record_minute_gap_event", "minute_gap"),
    ],
)
def test_provider_monitor_safe_mode_triggers_and_resets(
    tmp_path, monkeypatch, record_name: str, expected_reason: str
) -> None:
    dummy_alerts = _DummyAlerts()
    monitor = pm.ProviderMonitor(alert_manager=dummy_alerts, cooldown=1, threshold=3)
    disable_calls: list[tuple[str, float]] = []

    def _capture(provider: str, duration) -> None:
        seconds = float(duration.total_seconds()) if duration else 0.0
        disable_calls.append((provider, seconds))

    monitor.register_disable_callback("alpaca", lambda duration: _capture("alpaca", duration))
    monitor.register_disable_callback(
        "alpaca_sip", lambda duration: _capture("alpaca_sip", duration)
    )
    monkeypatch.setattr(pm, "provider_monitor", monitor)
    monkeypatch.setattr(pm, "_sip_auth_events", deque(), raising=False)
    monkeypatch.setattr(pm, "_gap_events", deque(), raising=False)
    monkeypatch.setattr(pm, "_SAFE_MODE_ACTIVE", False, raising=False)
    monkeypatch.setattr(pm, "_SAFE_MODE_REASON", None, raising=False)
    monkeypatch.setattr(pm, "_last_halt_reason", None, raising=False)
    monkeypatch.setattr(pm, "_last_halt_ts", 0.0, raising=False)
    monkeypatch.setattr(pm, "_HALT_SUPPRESS_SECONDS", 0.0, raising=False)

    halt_path = tmp_path / "halt.flag"
    monkeypatch.setattr(
        pm,
        "get_settings",
        lambda: SimpleNamespace(halt_flag_path=str(halt_path)),
    )

    record_func = getattr(pm, record_name)
    if record_name == "record_unauthorized_sip_event":
        monkeypatch.setenv("DATA_FEED_INTRADAY", "sip")
    else:
        monkeypatch.delenv("DATA_FEED_INTRADAY", raising=False)
    for _ in range(3):
        record_func({"symbol": "AAPL"})

    assert halt_path.exists()
    assert pm.is_safe_mode_active() is True
    assert pm.safe_mode_reason() == expected_reason
    assert any(event[0] == AlertType.PROVIDER_OUTAGE for event in dummy_alerts.events)
    providers = {name for name, _ in disable_calls}
    assert {"alpaca", "alpaca_sip"}.issubset(providers)
    assert monitor.is_disabled("alpaca") is True

    monitor.record_success("alpaca")
    assert pm.is_safe_mode_active() is False


def test_minute_gap_events_ignore_fallback_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pm, "_gap_events", deque(), raising=False)
    monkeypatch.setattr(pm, "_gap_event_diagnostics", {}, raising=False)
    triggered: list[tuple[str, int, Mapping[str, Any] | None]] = []
    diagnostics_snapshots: list[dict[str, Dict[str, Any]]] = []

    original_trigger = pm._trigger_provider_safe_mode

    def _capture_trigger(reason: str, *, count: int, metadata: Mapping[str, Any] | None = None) -> None:
        snapshot: dict[str, Dict[str, Any]] = {}
        for key, value in pm._gap_event_diagnostics.items():
            if isinstance(value, dict):
                snapshot[key] = dict(value)
        if snapshot:
            diagnostics_snapshots.append(snapshot)
        triggered.append((reason, count, metadata))
        original_trigger(reason, count=count, metadata=metadata)

    monkeypatch.setattr(pm, "_trigger_provider_safe_mode", _capture_trigger)
    monkeypatch.setattr(pm, "_write_halt_flag", lambda *_, **__: None)

    fallback_payload = {
        "provider": "yahoo",
        "provider_canonical": "yahoo",
        "primary_feed_gap": False,
        "using_fallback_provider": True,
        "residual_gap": True,
    }
    for _ in range(pm._GAP_EVENT_THRESHOLD * 2):
        pm.record_minute_gap_event(fallback_payload)

    assert triggered == []
    assert len(pm._gap_events) == 0

    primary_payload = {
        "provider": "alpaca",
        "provider_canonical": "alpaca",
        "primary_feed_gap": True,
        "residual_gap": True,
        "gap_ratio": 0.05,
        "missing_after": 5,
        "expected": 100,
    }
    for _ in range(pm._GAP_EVENT_THRESHOLD):
        pm.record_minute_gap_event(primary_payload)

    assert triggered
    reason, count, metadata = triggered[-1]
    assert reason == "minute_gap"
    assert count == pm._GAP_EVENT_THRESHOLD
    assert isinstance(metadata, Mapping)
    assert diagnostics_snapshots
    last_diag = diagnostics_snapshots[-1].get("alpaca")
    assert isinstance(last_diag, dict)
    assert last_diag.get("events") == pm._GAP_EVENT_THRESHOLD
    assert last_diag.get("max_gap_ratio") == pytest.approx(0.05)
    assert pm._gap_event_diagnostics == {}
