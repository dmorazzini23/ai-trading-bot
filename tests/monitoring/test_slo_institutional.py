from __future__ import annotations

from ai_trading.monitoring.slo import (
    SLOStatus,
    get_slo_monitor,
    record_execution_drift,
    record_order_reject_rate,
)


def test_default_slo_contains_reject_and_execution_drift() -> None:
    monitor = get_slo_monitor()
    reject_status = monitor.get_slo_status("order_reject_rate_pct")
    drift_status = monitor.get_slo_status("execution_drift_bps")

    assert reject_status["metric"] == "order_reject_rate_pct"
    assert drift_status["metric"] == "execution_drift_bps"


def test_reject_rate_metric_can_breach() -> None:
    monitor = get_slo_monitor()
    # Keep a tiny local window of samples to trigger status update quickly.
    monitor._metrics["order_reject_rate_pct"].clear()  # noqa: SLF001 - test-only reset
    monitor._slo_status["order_reject_rate_pct"] = SLOStatus.HEALTHY  # noqa: SLF001
    for _ in range(6):
        record_order_reject_rate(6.0)
    status = monitor.get_slo_status("order_reject_rate_pct")

    assert status["status"] in {"critical", "breached"}


def test_execution_drift_metric_records() -> None:
    monitor = get_slo_monitor()
    monitor._metrics["execution_drift_bps"].clear()  # noqa: SLF001 - test-only reset
    for _ in range(5):
        record_execution_drift(12.0)
    status = monitor.get_slo_status("execution_drift_bps")

    assert status["metric"] == "execution_drift_bps"
