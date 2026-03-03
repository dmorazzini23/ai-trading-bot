from __future__ import annotations

from ai_trading.monitoring.slo import (
    SLOStatus,
    get_slo_monitor,
    record_execution_drift,
    record_order_pacing_cap_hit_rate,
    record_order_reject_rate,
    record_pending_oldest_age,
    record_pending_orders_count,
    record_realized_slippage,
)


def test_default_slo_contains_reject_and_execution_drift() -> None:
    monitor = get_slo_monitor()
    reject_status = monitor.get_slo_status("order_reject_rate_pct")
    drift_status = monitor.get_slo_status("execution_drift_bps")
    slippage_status = monitor.get_slo_status("realized_slippage_bps")
    pending_count_status = monitor.get_slo_status("pending_orders_count")
    pending_age_status = monitor.get_slo_status("pending_oldest_age_sec")
    pacing_cap_status = monitor.get_slo_status("order_pacing_cap_hit_rate_pct")

    assert reject_status["metric"] == "order_reject_rate_pct"
    assert drift_status["metric"] == "execution_drift_bps"
    assert slippage_status["metric"] == "realized_slippage_bps"
    assert pending_count_status["metric"] == "pending_orders_count"
    assert pending_age_status["metric"] == "pending_oldest_age_sec"
    assert pacing_cap_status["metric"] == "order_pacing_cap_hit_rate_pct"


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
    monitor._metrics["realized_slippage_bps"].clear()  # noqa: SLF001 - test-only reset
    for _ in range(5):
        record_execution_drift(12.0)
        record_realized_slippage(9.0)
    status = monitor.get_slo_status("execution_drift_bps")
    slippage_status = monitor.get_slo_status("realized_slippage_bps")

    assert status["metric"] == "execution_drift_bps"
    assert slippage_status["metric"] == "realized_slippage_bps"


def test_pending_and_pacing_metrics_record() -> None:
    monitor = get_slo_monitor()
    monitor._metrics["pending_orders_count"].clear()  # noqa: SLF001 - test-only reset
    monitor._metrics["pending_oldest_age_sec"].clear()  # noqa: SLF001 - test-only reset
    monitor._metrics["order_pacing_cap_hit_rate_pct"].clear()  # noqa: SLF001 - test-only reset

    for _ in range(3):
        record_pending_orders_count(5.0)
        record_pending_oldest_age(180.0)
        record_order_pacing_cap_hit_rate(25.0)

    assert monitor.get_slo_status("pending_orders_count")["sample_count"] >= 1
    assert monitor.get_slo_status("pending_oldest_age_sec")["sample_count"] >= 1
    assert monitor.get_slo_status("order_pacing_cap_hit_rate_pct")["sample_count"] >= 1
