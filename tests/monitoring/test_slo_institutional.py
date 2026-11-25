from __future__ import annotations

import pytest

from ai_trading.monitoring import slo as slo_module
from ai_trading.monitoring.slo import SLOStatus, get_slo_monitor, record_execution_drift, record_order_pacing_cap_hit_rate, record_order_reject_rate, record_realized_slippage


@pytest.fixture(autouse=True)
def _fresh_global_slo_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests record samples on the process-wide monitor. Give each test its
    # own and restore the original afterwards so later tests (for example the
    # netting gate tests, which read reject rate and drift) see no leaked samples.
    monkeypatch.setattr(slo_module, "_global_slo_monitor", None)


def test_default_slo_contains_reject_and_execution_drift() -> None:
    monitor = get_slo_monitor()
    reject_status = monitor.get_slo_status("order_reject_rate_pct")
    drift_status = monitor.get_slo_status("execution_drift_bps")
    slippage_status = monitor.get_slo_status("realized_slippage_bps")
    pending_count_status = monitor.get_slo_status("pending_orders_count")
    pending_age_status = monitor.get_slo_status("pending_oldest_age_sec")
    pacing_cap_status = monitor.get_slo_status("order_pacing_cap_hit_rate_pct")
    calibration_ece_status = monitor.get_slo_status("live_calibration_ece")
    calibration_brier_status = monitor.get_slo_status("live_calibration_brier")
    drift_psi_status = monitor.get_slo_status("drift_psi")
    label_drift_status = monitor.get_slo_status("label_drift_psi")
    residual_drift_status = monitor.get_slo_status("residual_drift_psi")

    assert reject_status["metric"] == "order_reject_rate_pct"
    assert drift_status["metric"] == "execution_drift_bps"
    assert slippage_status["metric"] == "realized_slippage_bps"
    assert pending_count_status["metric"] == "pending_orders_count"
    assert pending_age_status["metric"] == "pending_oldest_age_sec"
    assert pacing_cap_status["metric"] == "order_pacing_cap_hit_rate_pct"
    assert calibration_ece_status["metric"] == "live_calibration_ece"
    assert calibration_brier_status["metric"] == "live_calibration_brier"
    assert drift_psi_status["metric"] == "drift_psi"
    assert label_drift_status["metric"] == "label_drift_psi"
    assert residual_drift_status["metric"] == "residual_drift_psi"


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




