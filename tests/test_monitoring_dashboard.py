from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

from ai_trading.monitoring import dashboard


def _collector(now: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        trade_metrics=[
            {
                "timestamp": now - timedelta(minutes=15),
                "pnl": 50.0,
                "notional_value": 1_000.0,
                "symbol": "AAPL",
                "strategy_id": "mean",
            },
            {
                "timestamp": now - timedelta(minutes=30),
                "pnl": -10.0,
                "notional_value": 500.0,
                "symbol": "MSFT",
                "strategy_id": "trend",
            },
            {"timestamp": now - timedelta(days=2), "pnl": 99.0, "notional_value": 2_000.0},
        ],
        portfolio_metrics=[
            {
                "timestamp": now - timedelta(minutes=10),
                "total_value": 101_000.0,
                "day_change": 250.0,
                "day_change_pct": 0.0025,
                "unrealized_pnl": 100.0,
                "realized_pnl": 40.0,
                "cash": 20_000.0,
                "positions_count": 3,
            }
        ],
        risk_metrics=[
            {
                "timestamp": now - timedelta(minutes=5),
                "var_95": 0.02,
                "max_drawdown": 0.05,
                "current_drawdown": 0.01,
                "sharpe_ratio": 1.2,
                "volatility": 0.18,
            }
        ],
        execution_metrics=[
            {
                "timestamp": now - timedelta(minutes=5),
                "orders_submitted": 10,
                "orders_filled": 8,
                "fill_rate": 0.8,
                "average_fill_time": 1.5,
                "total_volume": 1_500.0,
            }
        ],
        system_metrics=[
            {
                "timestamp": now - timedelta(minutes=1),
                "cpu_usage": 81.0,
                "memory_usage": 70.0,
                "latency": 100.0,
            }
        ],
    )


def test_realtime_metrics_summaries_and_cache():
    now = datetime.now(UTC)
    collector = _collector(now)
    realtime = dashboard.RealtimeMetrics(cast(Any, collector))

    pnl = realtime.get_current_pnl()
    assert pnl["realized_pnl"] == 40.0
    assert pnl["trade_count"] == 2
    assert pnl["win_rate"] == 0.5
    assert realtime.get_current_pnl() is pnl

    portfolio = realtime.get_portfolio_summary()
    risk = realtime.get_risk_summary()
    execution = realtime.get_execution_summary()

    assert portfolio["total_value"] == 101_000.0
    assert risk["sharpe_ratio"] == 1.2
    assert execution["orders_filled"] == 8
    assert realtime._is_cached("execution_summary")
    realtime._cache_timestamps["execution_summary"] = now - timedelta(minutes=5)
    assert not realtime._is_cached("execution_summary")


def test_realtime_metrics_empty_collectors_return_defaults():
    realtime = dashboard.RealtimeMetrics(
        cast(Any, SimpleNamespace(
            trade_metrics=[],
            portfolio_metrics=[],
            risk_metrics=[],
            execution_metrics=[],
        ))
    )

    assert realtime.get_current_pnl() == {"realized_pnl": 0.0, "trade_count": 0, "win_rate": 0.0}
    assert realtime.get_portfolio_summary()["total_value"] == 0.0
    assert realtime.get_risk_summary()["var_95"] == 0.0
    assert realtime.get_execution_summary()["orders_submitted"] == 0


def test_dashboard_data_provider_aggregates_activity_alerts_health_and_charts():
    now = datetime.now(UTC)
    collector = _collector(now)

    class PerfMonitor:
        def get_performance_report(self, time_range):
            return {"hours": time_range.total_seconds() / 3600}

    class Alert:
        def __init__(self, severity, timestamp):
            self.severity = severity
            self.timestamp = timestamp

        def to_dict(self):
            return {"severity": self.severity.value}

    critical = Alert(dashboard.AlertSeverity.CRITICAL, now - timedelta(minutes=10))
    info = Alert(dashboard.AlertSeverity.INFO, now - timedelta(hours=2))
    alert_manager = SimpleNamespace(
        alerts=[critical, info],
        get_active_alerts=lambda: [critical, info],
    )
    provider = dashboard.DashboardDataProvider(
        cast(Any, collector),
        cast(Any, PerfMonitor()),
        cast(Any, alert_manager),
    )

    data = provider.get_dashboard_data(time_range=timedelta(hours=2))
    activity = provider.get_trading_activity_summary(time_range=timedelta(hours=1))
    alerts = provider._get_alert_summary()
    health = provider._get_system_health()
    charts = provider._get_chart_data(timedelta(hours=2))

    assert data["time_range_hours"] == 2.0
    assert data["performance"] == {"hours": 2.0}
    assert activity["total_trades"] == 2
    assert activity["symbols_traded"] == 2
    assert activity["win_rate"] == 0.5
    assert alerts["active_count"] == 2
    assert alerts["severity_breakdown"]["critical"] == 1
    assert alerts["critical_alerts"] == [{"severity": "critical"}]
    assert health["status"] == "warning"
    assert len(charts["portfolio_value"]) == 1
    assert len(charts["risk_metrics"]) == 1
    assert charts["trading_volume"][0]["trade_count"] >= 1


def test_dashboard_data_provider_uses_performance_metrics_fallback_and_defaults():
    collector = SimpleNamespace(
        trade_metrics=[],
        portfolio_metrics=[],
        risk_metrics=[],
        execution_metrics=[],
        system_metrics=[],
    )
    perf_monitor = SimpleNamespace(get_performance_metrics=lambda: {"fallback": True})
    alert_manager = SimpleNamespace(alerts=[], get_active_alerts=lambda: [])
    provider = dashboard.DashboardDataProvider(
        cast(Any, collector),
        cast(Any, perf_monitor),
        cast(Any, alert_manager),
    )

    data = provider.get_dashboard_data()
    activity = provider.get_trading_activity_summary()
    health = provider._get_system_health()

    assert data["performance"] == {"fallback": True}
    assert activity == {
        "total_trades": 0,
        "total_volume": 0.0,
        "symbols_traded": 0,
        "strategies_active": 0,
    }
    assert health["status"] == "unknown"


def test_dashboard_error_paths_degrade_to_empty_payloads():
    broken_collector = SimpleNamespace(
        trade_metrics=[{"timestamp": "not-a-date"}],
        portfolio_metrics=[{"timestamp": "not-a-date"}],
        risk_metrics=[{"timestamp": "not-a-date"}],
        execution_metrics=[],
        system_metrics=[{"timestamp": "not-a-date"}],
    )
    broken_alerts = SimpleNamespace(
        alerts=[SimpleNamespace(timestamp="not-a-date", severity=dashboard.AlertSeverity.INFO)],
        get_active_alerts=lambda: (_ for _ in ()).throw(TypeError("alerts unavailable")),
    )
    provider = dashboard.DashboardDataProvider(
        cast(Any, broken_collector),
        cast(Any, object()),
        cast(Any, broken_alerts),
    )

    data = provider.get_dashboard_data()
    activity = provider.get_trading_activity_summary()

    assert data["charts"] == {"portfolio_value": [], "risk_metrics": [], "trading_volume": []}
    assert data["alerts"]["active_count"] == 0
    assert data["system_health"]["status"] == "error"
    assert activity == {"total_trades": 0, "total_volume": 0.0}
