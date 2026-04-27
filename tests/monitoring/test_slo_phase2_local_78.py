from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.monitoring import slo


def test_slo_threshold_transitions_alerts_and_health_summary() -> None:
    monitor = slo.SLOMonitor()
    monitor._slo_thresholds.clear()  # noqa: SLF001
    monitor._slo_status.clear()  # noqa: SLF001

    threshold = slo.SLOThreshold(
        name="latency_ms",
        warning_threshold=10.0,
        critical_threshold=20.0,
        breach_threshold=30.0,
        window_minutes=5,
        min_samples=2,
        description="latency",
    )
    monitor.add_slo_threshold(threshold)
    old = datetime.now(UTC) - timedelta(minutes=10)
    now = datetime.now(UTC)

    monitor.record_metric("latency_ms", 100.0, timestamp=old)
    monitor.record_metric("latency_ms", 100.0, timestamp=old)
    assert monitor.get_slo_status("latency_ms")["status"] == "healthy"

    monitor.record_metric("latency_ms", 15.0, tags={"op": "submit"}, timestamp=now)
    monitor.record_metric("latency_ms", 17.0, timestamp=now)
    warning_status = monitor.get_slo_status("latency_ms")
    assert warning_status["status"] == "warning"
    assert warning_status["sample_count"] == 2
    assert warning_status["current_value"] == 16.0

    monitor.record_metric("latency_ms", 45.0, timestamp=now)
    monitor.record_metric("latency_ms", 45.0, timestamp=now)
    assert monitor.get_slo_status("latency_ms")["status"] == "breached"
    assert monitor.get_alerts(limit=2)[-1]["level"] == "emergency"
    assert monitor.get_health_summary()["overall_health"] == "critical"
    assert monitor.get_slo_status("missing") == {"error": "SLO missing not found"}


def test_live_sharpe_uses_lower_values_as_worse_and_callbacks_are_guarded() -> None:
    monitor = slo.SLOMonitor()
    seen: list[tuple[str, float]] = []
    monitor.register_circuit_breaker("live_sharpe_ratio", lambda name, value: seen.append((name, value)))

    def broken_callback(_name: str, _value: float) -> None:
        raise ValueError("callback failed")

    monitor.register_circuit_breaker("live_sharpe_ratio", broken_callback)
    now = datetime.now(UTC)

    for value in [-0.75] * 10:
        monitor.record_metric("live_sharpe_ratio", value, timestamp=now)

    status = monitor.get_slo_status("live_sharpe_ratio")
    assert status["status"] == "breached"
    assert seen == [("live_sharpe_ratio", -0.75)]

    for value in [1.25] * 20:
        monitor.record_metric("live_sharpe_ratio", value, timestamp=now)

    assert monitor.get_slo_status("live_sharpe_ratio")["status"] == "healthy"
    assert monitor.get_health_summary()["overall_health"] == "healthy"


def test_config_loading_and_module_level_recorders(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "slos.json"
    config_path.write_text(
        json.dumps(
            {
                "slos": [
                    {
                        "name": "custom_metric",
                        "warning_threshold": 1.0,
                        "critical_threshold": 2.0,
                        "breach_threshold": 3.0,
                        "window_minutes": 2,
                        "min_samples": 1,
                        "description": "custom",
                    }
                ]
            }
        )
    )

    monitor = slo.SLOMonitor(str(config_path))
    assert monitor.get_slo_status("custom_metric")["description"] == "custom"

    broken_config = tmp_path / "broken.json"
    broken_config.write_text("{")
    broken_monitor = slo.SLOMonitor(str(broken_config))
    assert "order_latency_ms" in broken_monitor.get_slo_status()

    monkeypatch.setattr(slo, "_global_slo_monitor", None)
    created = slo.get_slo_monitor()
    assert slo.get_slo_monitor() is created

    recorded: list[tuple[str, float, dict[str, str] | None]] = []
    monkeypatch.setattr(
        slo,
        "get_slo_monitor",
        lambda: type(
            "Recorder",
            (),
            {"record_metric": lambda self, name, value, tags=None: recorded.append((name, value, tags))},
        )(),
    )
    slo.record_latency("order", 42.0)
    slo.record_error_rate("broker", 1.5)
    slo.record_performance_metric("turnover_ratio", 2.5, {"cycle": "open"})
    slo.record_order_reject_rate(3)
    slo.record_execution_drift(12)
    slo.record_realized_slippage(14)
    slo.record_live_calibration_ece(0.2)
    slo.record_live_calibration_brier(0.4)
    slo.record_feature_drift_psi(0.3)
    slo.record_label_drift_psi(0.31)
    slo.record_residual_drift_psi(0.32)
    slo.record_pending_orders_count(9)
    slo.record_pending_oldest_age(600)
    slo.record_order_pacing_cap_hit_rate(50)

    assert ("order_latency_ms", 42.0, None) in recorded
    assert ("turnover_ratio", 2.5, {"cycle": "open"}) in recorded
    assert recorded[-1] == ("order_pacing_cap_hit_rate_pct", 50.0, None)
