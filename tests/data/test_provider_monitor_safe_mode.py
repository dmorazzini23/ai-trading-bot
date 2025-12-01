from __future__ import annotations

import importlib
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
        payload: dict[str, Any] = {"symbol": "AAPL"}
    else:
        monkeypatch.delenv("DATA_FEED_INTRADAY", raising=False)
        payload = {
            "symbol": "AAPL",
            "provider": "alpaca",
            "primary_feed_gap": True,
            "gap_ratio": 0.5,
            "initial_gap_ratio": 0.5,
            "missing_after": 10,
            "initial_missing": 10,
            "expected": 100,
            "residual_gap": True,
        }
    for _ in range(3):
        record_func(payload)

    assert halt_path.exists()
    assert pm.is_safe_mode_active() is True
    assert pm.safe_mode_reason() == expected_reason
    assert any(event[0] == AlertType.PROVIDER_OUTAGE for event in dummy_alerts.events)
    providers = {name for name, _ in disable_calls}
    assert {"alpaca", "alpaca_sip"}.issubset(providers)
    assert monitor.is_disabled("alpaca") is True

    monitor.record_success("alpaca")
    assert pm.is_safe_mode_active() is False


def test_gap_safe_mode_uses_failsoft_when_backup_available(tmp_path, monkeypatch) -> None:
    dummy_alerts = _DummyAlerts()
    monitor = pm.ProviderMonitor(alert_manager=dummy_alerts, cooldown=1, threshold=3)
    disable_calls: list[str] = []

    monitor.register_disable_callback("alpaca", lambda duration: disable_calls.append("alpaca"))
    monitor.register_disable_callback("alpaca_sip", lambda duration: disable_calls.append("alpaca_sip"))

    monkeypatch.setattr(pm, "provider_monitor", monitor)
    monkeypatch.setattr(pm, "_gap_events", deque(), raising=False)
    monkeypatch.setattr(pm, "_gap_event_diagnostics", {}, raising=False)
    monkeypatch.setattr(pm, "_gap_trigger_cooldown_until", 0.0, raising=False)
    monkeypatch.setattr(pm, "_GAP_EVENT_THRESHOLD", 1, raising=False)
    monkeypatch.setattr(pm, "_SAFE_MODE_ACTIVE", False, raising=False)
    monkeypatch.setattr(pm, "_SAFE_MODE_REASON", None, raising=False)
    monkeypatch.setattr(pm, "_SAFE_MODE_HEALTHY_PASSES", 0, raising=False)
    monkeypatch.setattr(pm, "_SAFE_MODE_DEGRADED_ONLY", False, raising=False)

    halt_path = tmp_path / "halt.flag"
    monkeypatch.setattr(
        pm,
        "get_settings",
        lambda: SimpleNamespace(halt_flag_path=str(halt_path)),
    )

    payload = {
        "symbol": "AAPL",
        "provider": "alpaca_iex",
        "provider_canonical": "alpaca_iex",
        "primary_feed_gap": True,
        "gap_ratio": 0.32,
        "initial_gap_ratio": 0.32,
        "missing_after": 32,
        "initial_missing": 32,
        "expected": 100,
        "residual_gap": True,
        "used_backup": True,
        "using_fallback_provider": True,
        "fallback_provider": "yahoo",
    }
    for _ in range(pm._GAP_EVENT_THRESHOLD):
        pm.record_minute_gap_event(payload)

    assert pm.is_safe_mode_active() is True
    assert pm.safe_mode_reason() == "minute_gap"
    assert pm.safe_mode_degraded_only() is True
    assert halt_path.exists() is False
    assert disable_calls == []
    assert monitor.is_disabled("alpaca") is False
    assert monitor.is_disabled("alpaca_sip") is False


def test_gap_failsoft_handles_percent_ratio(tmp_path, monkeypatch) -> None:
    pm_local = importlib.reload(pm)
    dummy_alerts = _DummyAlerts()
    monitor = pm_local.ProviderMonitor(alert_manager=dummy_alerts, cooldown=1, threshold=3)

    monkeypatch.setattr(pm_local, "provider_monitor", monitor)
    monkeypatch.setattr(pm_local, "_gap_events", deque(), raising=False)
    monkeypatch.setattr(pm_local, "_gap_event_diagnostics", {}, raising=False)
    monkeypatch.setattr(pm_local, "_gap_trigger_cooldown_until", 0.0, raising=False)
    monkeypatch.setattr(pm_local, "_GAP_EVENT_THRESHOLD", 1, raising=False)
    monkeypatch.setattr(pm_local, "_SAFE_MODE_ACTIVE", False, raising=False)
    monkeypatch.setattr(pm_local, "_SAFE_MODE_REASON", None, raising=False)
    monkeypatch.setattr(pm_local, "_SAFE_MODE_HEALTHY_PASSES", 0, raising=False)
    monkeypatch.setattr(pm_local, "_SAFE_MODE_DEGRADED_ONLY", False, raising=False)
    times = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(pm_local, "monotonic_time", lambda: next(times), raising=False)

    halt_path = tmp_path / "halt.flag"
    monkeypatch.setattr(
        pm_local,
        "get_settings",
        lambda: SimpleNamespace(halt_flag_path=str(halt_path)),
    )

    payload = {
        "symbol": "AAPL",
        "provider": "alpaca_iex",
        "provider_canonical": "alpaca_iex",
        "primary_feed_gap": True,
        "gap_ratio": None,
        "gap_ratio_pct": 30.32,  # percent representation
        "initial_gap_ratio": 30.32,
        "missing_after": 30,
        "initial_missing": 30,
        "expected": 99,
        "residual_gap": True,
        "used_backup": True,
        "using_fallback_provider": True,
        "fallback_provider": "yahoo",
    }
    pm_local._trigger_provider_safe_mode("minute_gap", count=3, metadata=payload)

    assert pm_local.is_safe_mode_active() is True
    assert pm_local.safe_mode_degraded_only() is True
    assert halt_path.exists() is False


def test_minute_gap_events_ignore_fallback_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pm, "_gap_events", deque(), raising=False)
    monkeypatch.setattr(pm, "_gap_event_diagnostics", {}, raising=False)
    monkeypatch.setattr(pm, "_gap_trigger_cooldown_until", 0.0, raising=False)
    pm._prime_gap_ratio_cache(pm._GAP_RATIO_TRIGGER)
    triggered: list[tuple[str, int, Mapping[str, Any] | None]] = []
    diagnostics_snapshots: list[dict[str, Dict[str, Any]]] = []

    def _capture_trigger(reason: str, *, count: int, metadata: Mapping[str, Any] | None = None) -> None:
        snapshot: dict[str, Dict[str, Any]] = {}
        for key, value in pm._gap_event_diagnostics.items():
            if isinstance(value, dict):
                snapshot[key] = dict(value)
        if snapshot:
            diagnostics_snapshots.append(snapshot)
        triggered.append((reason, count, metadata))

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


def test_gap_event_threshold_elevated_for_iex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    pm._prime_gap_ratio_cache(pm._GAP_RATIO_TRIGGER)
    monkeypatch.setattr(pm, "_gap_events", deque(), raising=False)
    monkeypatch.setattr(pm, "_gap_event_diagnostics", {}, raising=False)
    monkeypatch.setattr(pm, "_gap_trigger_cooldown_until", 0.0, raising=False)
    monkeypatch.setattr(pm, "_gap_trigger_cooldown_until", 0.0, raising=False)
    payload = {
        "provider": "alpaca_iex",
        "primary_feed_gap": True,
        "used_backup": False,
        "gap_ratio": 0.25,
        "initial_gap_ratio": 0.25,
        "missing_after": 25,
        "initial_missing": 25,
        "expected": 100,
        "residual_gap": True,
        "active_feed": "iex",
    }

    assert pm._gap_event_is_severe(payload) is False


def test_gap_event_threshold_strict_for_sip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    pm._prime_gap_ratio_cache(pm._GAP_RATIO_TRIGGER)
    monkeypatch.setattr(pm, "_gap_events", deque(), raising=False)
    monkeypatch.setattr(pm, "_gap_event_diagnostics", {}, raising=False)
    payload = {
        "provider": "alpaca_sip",
        "primary_feed_gap": True,
        "used_backup": False,
        "gap_ratio": 0.25,
        "initial_gap_ratio": 0.25,
        "missing_after": 30,
        "initial_missing": 30,
        "expected": 120,
        "residual_gap": True,
        "active_feed": "sip",
    }

    assert pm._gap_event_is_severe(payload) is True
