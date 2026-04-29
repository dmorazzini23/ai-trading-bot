from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.monitoring import slo
from ai_trading.monitoring.slo import SLOMonitor, SLOStatus, SLOThreshold


def test_get_all_statuses_returns_snapshot_without_recursive_lock() -> None:
    monitor = SLOMonitor()

    for _ in range(5):
        monitor.record_metric("order_latency_ms", 125.0)

    all_statuses = monitor.get_slo_status()

    assert all_statuses["order_latency_ms"]["status"] == SLOStatus.WARNING.value
    assert all_statuses["order_latency_ms"]["sample_count"] == 5
    assert monitor.get_slo_status("not_configured") == {"error": "SLO not_configured not found"}


def test_breach_alerts_and_circuit_breakers_are_tolerant() -> None:
    monitor = SLOMonitor()
    monitor.add_slo_threshold(
        SLOThreshold(
            name="custom_latency_ms",
            warning_threshold=10.0,
            critical_threshold=20.0,
            breach_threshold=30.0,
            min_samples=3,
            description="custom latency",
        )
    )
    triggered: list[tuple[str, float]] = []

    def record_callback(metric_name: str, value: float) -> None:
        triggered.append((metric_name, value))

    def noisy_callback(metric_name: str, value: float) -> None:
        raise ValueError(f"{metric_name}:{value}")

    monitor.register_circuit_breaker("custom_latency_ms", record_callback)
    monitor.register_circuit_breaker("custom_latency_ms", noisy_callback)

    for _ in range(3):
        monitor.record_metric("custom_latency_ms", 35.0)

    status = monitor.get_slo_status("custom_latency_ms")
    alert = monitor.get_alerts(limit=1)[0]
    health = monitor.get_health_summary()

    assert status["status"] == SLOStatus.BREACHED.value
    assert triggered == [("custom_latency_ms", 35.0)]
    assert alert["level"] == "emergency"
    assert alert["threshold"]["breach"] == 30.0
    assert health["overall_health"] == "critical"
    assert health["recent_alerts"] >= 1


def test_inverse_sharpe_and_windowed_samples() -> None:
    monitor = SLOMonitor()
    old_timestamp = datetime.now(UTC) - timedelta(hours=2)

    for _ in range(10):
        monitor.record_metric("live_sharpe_ratio", -1.0, timestamp=old_timestamp)

    stale_status = monitor.get_slo_status("live_sharpe_ratio")
    assert stale_status["status"] == SLOStatus.HEALTHY.value
    assert stale_status["current_value"] is None
    assert stale_status["sample_count"] == 0

    for _ in range(10):
        monitor.record_metric("live_sharpe_ratio", -1.0)

    assert monitor.get_slo_status("live_sharpe_ratio")["status"] == SLOStatus.BREACHED.value


def test_config_loading_and_module_recording_wrappers(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "slo_config.json"
    config_path.write_text(
        """
        {
          "slos": [
            {
              "name": "custom_error_rate_pct",
              "warning_threshold": 1.0,
              "critical_threshold": 5.0,
              "breach_threshold": 10.0,
              "window_minutes": 2,
              "min_samples": 2,
              "description": "custom errors"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    monitor = SLOMonitor(config_path=str(config_path))
    monkeypatch.setattr(slo, "_global_slo_monitor", monitor)

    slo.record_error_rate("custom", 2.0)
    slo.record_error_rate("custom", 2.0)
    slo.record_latency("order", 50.0)
    slo.record_performance_metric("turnover_ratio", 2.5, tags={"scope": "unit"})
    slo.setup_default_circuit_breakers()
    slo.pause_trading_circuit_breaker("custom_error_rate_pct", 11.0)
    slo.reduce_position_size_circuit_breaker("live_sharpe_ratio", -1.0)

    status = monitor.get_slo_status("custom_error_rate_pct")

    assert status["status"] == SLOStatus.WARNING.value
    assert monitor._breach_callbacks["order_reject_rate_pct"]  # noqa: SLF001
    assert monitor._metrics["order_latency_ms"]  # noqa: SLF001
    assert monitor._metrics["turnover_ratio"]  # noqa: SLF001

    invalid_path = tmp_path / "bad_slo_config.json"
    invalid_path.write_text("{not-json", encoding="utf-8")
    SLOMonitor(config_path=str(invalid_path))


def test_unthresholded_component_error_rate_updates_canonical_metric(monkeypatch) -> None:
    monitor = SLOMonitor()
    monkeypatch.setattr(slo, "_global_slo_monitor", monitor)
    monitor._metrics["broker_error_rate_pct"].clear()  # noqa: SLF001
    monitor._metrics["error_rate_pct"].clear()  # noqa: SLF001

    slo.record_error_rate("broker", 50.0)

    assert monitor._metrics["broker_error_rate_pct"][-1].value == 50.0  # noqa: SLF001
    assert monitor._metrics["error_rate_pct"][-1].value == 50.0  # noqa: SLF001
