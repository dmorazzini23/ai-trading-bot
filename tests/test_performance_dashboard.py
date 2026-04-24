from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from ai_trading.monitoring import performance_dashboard as pd


def test_performance_metrics_calculate_ratios_drawdown_and_win_rate():
    metrics = pd.PerformanceMetrics(lookback_days=40)
    start = datetime(2026, 4, 1, tzinfo=UTC)

    for idx in range(30):
        metrics.add_return(0.01 if idx % 2 == 0 else -0.004 - idx * 0.0001, 100_000 + idx * 100)
    metrics.add_trade("AAPL", start, start + timedelta(hours=2), 100.0, 105.0, 10, 50.0, 1.0)
    metrics.add_trade("MSFT", start, start + timedelta(hours=1), 50.0, 48.0, 5, -10.0, 0.5)

    sharpe = metrics.calculate_sharpe_ratio()
    sortino = metrics.calculate_sortino_ratio()
    drawdown, duration = metrics.calculate_max_drawdown([100.0, 110.0, 90.0, 95.0, 120.0])
    win_stats = metrics.calculate_win_rate()
    current = metrics.get_current_metrics()

    assert sharpe != 0.0
    assert sortino != 0.0
    assert drawdown == (110.0 - 90.0) / 110.0
    assert duration == 2
    assert win_stats["total_trades"] == 2
    assert win_stats["winning_trades"] == 1
    assert win_stats["losing_trades"] == 1
    assert current["total_trades"] == 0


def test_performance_metrics_edge_cases_return_defaults():
    metrics = pd.PerformanceMetrics()

    assert metrics.calculate_sharpe_ratio([0.01] * 29) == 0.0
    assert metrics.calculate_sharpe_ratio([0.01] * 30) == 0.0
    assert metrics.calculate_sortino_ratio([0.01] * 30) == math.inf
    assert metrics.calculate_max_drawdown([100.0]) == (0.0, 0)
    assert metrics.calculate_win_rate()["profit_factor"] == 0.0


def test_real_time_pnl_tracker_positions_trades_and_summary():
    tracker = pd.RealTimePnLTracker()

    tracker.start_new_session(100_000.0)
    tracker.update_equity(102_000.0)
    tracker.update_equity(99_000.0)
    tracker.update_position("AAPL", 10, 100.0, 110.0, commission=2.0)
    tracker.record_trade("AAPL", -5, 112.0, 1.0, trade_type="sell")
    summary = tracker.get_pnl_summary()
    positions = tracker.get_position_details()
    tracker.update_position("AAPL", 0, 0.0, 0.0)

    assert summary["unrealized_pnl"] == 98.0
    assert summary["realized_pnl"] == 59.0
    assert summary["total_pnl"] == 157.0
    assert summary["session_high"] == 102_000.0
    assert summary["session_low"] == 99_000.0
    assert len(positions) == 1
    assert tracker.get_position_details() == []


def test_anomaly_detector_updates_thresholds_detects_and_filters_recent():
    detector = pd.AnomalyDetector(sensitivity=1.0)
    old = datetime.now(UTC) - timedelta(days=2)

    for idx in range(30):
        detector.update_data(0.001 * (idx % 3), 100.0 + idx, 0.10 + (idx % 2) * 0.01)
    detector.recent_anomalies.append({"timestamp": old, "type": "old"})
    anomalies = detector.detect_anomalies(0.20, 20_000.0, 0.75)
    recent = detector.get_recent_anomalies(hours=24)

    assert detector.return_threshold > 0
    assert detector.pnl_threshold > 0
    assert detector.volatility_threshold > 0
    assert {anomaly["type"] for anomaly in anomalies} == {
        "unusual_return",
        "unusual_pnl",
        "unusual_volatility",
    }
    assert all(anomaly["type"] != "old" for anomaly in recent)


def test_dashboard_update_routes_anomalies_and_threshold_alerts():
    class AlertManager:
        def __init__(self):
            self.alerts: list[tuple[object, ...]] = []

        def send_performance_alert(self, *args):
            self.alerts.append(args)

    alert_manager = AlertManager()
    dashboard = pd.PerformanceDashboard(alert_manager)
    dashboard.metrics.current_metrics = {
        "sharpe_ratio": -1.0,
        "max_drawdown": 0.25,
        "win_rate": 0.10,
    }
    for idx in range(30):
        dashboard.anomaly_detector.update_data(0.001 * (idx % 3), 100.0 + idx, 0.10 + (idx % 2) * 0.01)

    dashboard.update_performance(101_000.0, daily_return=0.20, volatility=0.75)

    alert_names = [alert[0] for alert in alert_manager.alerts]
    assert "unusual_return" in alert_names
    assert "unusual_pnl" in alert_names
    assert "unusual_volatility" in alert_names
    assert "Sharpe Ratio" in alert_names
    assert "Maximum Drawdown" in alert_names
    assert "Win Rate" in alert_names
    assert dashboard.last_update is not None


def test_dashboard_summary_positions_and_trade_forwarding():
    dashboard = pd.PerformanceDashboard()
    start = datetime(2026, 4, 1, tzinfo=UTC)

    dashboard.update_position("AAPL", 10, 100.0, 102.0)
    dashboard.add_trade("AAPL", start, start + timedelta(hours=1), 100.0, 102.0, -5, 10.0, 1.0)
    summary = dashboard.get_dashboard_summary()

    assert summary["position_count"] == 1
    assert summary["pnl_summary"]["open_positions"] == 1
    assert summary["pnl_summary"]["total_trades"] == 1
    assert summary["system_status"]["alerts_configured"] is False
    assert dashboard.get_detailed_positions()[0]["symbol"] == "AAPL"
