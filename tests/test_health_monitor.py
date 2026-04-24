from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

from ai_trading import health_monitor as hm


def test_health_checker_handles_dict_unexpected_exception_and_timeout():
    async def run_checks():
        dict_checker = hm.HealthChecker(
            "dict",
            hm.ComponentType.API_SERVICE,
            lambda: {
                "status": "warning",
                "message": "degraded",
                "details": {"latency": 12},
            },
        )
        unexpected_checker = hm.HealthChecker(
            "raw",
            hm.ComponentType.API_SERVICE,
            cast(Any, lambda: object()),
        )

        def broken_check():
            raise ValueError("bad check")

        broken_checker = hm.HealthChecker("broken", hm.ComponentType.API_SERVICE, broken_check)

        async def slow_check():
            await asyncio.sleep(0.05)
            return True

        timeout_checker = hm.HealthChecker(
            "slow",
            hm.ComponentType.API_SERVICE,
            slow_check,
            timeout_seconds=0.001,
        )

        dict_result = await dict_checker.run_check()
        unexpected_result = await unexpected_checker.run_check()
        broken_result = await broken_checker.run_check()
        timeout_result = await timeout_checker.run_check()

        assert dict_result.status is hm.HealthStatus.WARNING
        assert dict_result.message == "degraded"
        assert dict_result.details == {"latency": 12}
        assert unexpected_result.status is hm.HealthStatus.UNKNOWN
        assert "raw_result" in unexpected_result.details
        assert broken_result.status is hm.HealthStatus.CRITICAL
        assert broken_result.details["error_type"] == "ValueError"
        assert timeout_result.status is hm.HealthStatus.CRITICAL
        assert timeout_result.details == {"timeout": True}
        assert timeout_checker.consecutive_failures == 1

    asyncio.run(run_checks())


def test_health_monitor_register_unregister_and_run_all_checks():
    async def run_monitor():
        monitor = hm.HealthMonitor(check_interval=0.01)
        monitor.checkers.clear()
        monitor.register_check("ok", hm.ComponentType.API_SERVICE, lambda: True, interval_seconds=0)
        monitor.register_check("skip", hm.ComponentType.API_SERVICE, lambda: False, interval_seconds=0)
        monitor.checkers["skip"].enabled = False

        results = await monitor.run_all_checks()

        assert [result.component for result in results] == ["ok"]
        assert monitor.health_history == results
        assert monitor.unregister_check("skip")
        assert not monitor.unregister_check("missing")

    asyncio.run(run_monitor())


def test_monitoring_loop_runs_one_iteration_and_stops(monkeypatch):
    monitor = hm.HealthMonitor(check_interval=0.01)
    calls: list[str] = []

    async def run_all_checks():
        calls.append("checks")
        monitor.running = False
        return []

    async def sleep(_seconds):
        calls.append("sleep")

    monkeypatch.setattr(monitor, "run_all_checks", run_all_checks)
    monkeypatch.setattr(monitor, "_collect_system_metrics", lambda: calls.append("metrics"))
    monkeypatch.setattr(monitor, "_process_alerts", lambda: calls.append("alerts"))
    monkeypatch.setattr(monitor, "_cleanup_history", lambda: calls.append("cleanup"))
    monkeypatch.setattr(hm.asyncio, "sleep", sleep)

    monitor.running = True
    asyncio.run(monitor._monitoring_loop())

    assert calls == ["checks", "metrics", "alerts", "cleanup", "sleep"]


def test_start_and_stop_monitoring_cancel_task(monkeypatch):
    async def run_monitor():
        monitor = hm.HealthMonitor(check_interval=0.01)
        canceled = {"value": False}

        async def loop():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                canceled["value"] = True
                raise

        monkeypatch.setattr(monitor, "_monitoring_loop", loop)

        await monitor.start_monitoring()
        assert monitor.running
        await asyncio.sleep(0)
        await monitor.start_monitoring()
        await monitor.stop_monitoring()
        assert not monitor.running
        assert canceled["value"]

    asyncio.run(run_monitor())


def test_collect_system_metrics_uses_snapshot_when_psutil_unavailable(monkeypatch):
    monitor = hm.HealthMonitor()
    monkeypatch.setattr(hm, "_HAS_PSUTIL", False)
    monkeypatch.setattr(
        hm,
        "snapshot_basic",
        lambda: {"cpu_percent": 12.5, "mem_percent": 34.5},
    )

    metrics = monitor._collect_system_metrics()

    assert metrics.cpu_percent == 12.5
    assert metrics.memory_percent == 34.5
    assert monitor.system_metrics_history[-1] is metrics


def test_process_alerts_emits_critical_failure_and_latency_alerts():
    monitor = hm.HealthMonitor()
    monitor.checkers.clear()
    checker = hm.HealthChecker("api", hm.ComponentType.API_SERVICE, lambda: False)
    checker.consecutive_failures = 3
    monitor.checkers["api"] = checker
    result = hm.HealthCheckResult(
        component="api",
        component_type=hm.ComponentType.API_SERVICE,
        status=hm.HealthStatus.CRITICAL,
        message="down",
        response_time_ms=20_000,
        timestamp=datetime.now(UTC),
        details={"reason": "timeout"},
    )
    monitor.health_history = [result]
    alerts: list[str] = []
    setattr(cast(Any, monitor), "_send_alert", lambda message, result: alerts.append(message))

    monitor._process_alerts()

    assert alerts == [
        "CRITICAL: api health check failed",
        "CRITICAL: api has 3 consecutive failures",
        "CRITICAL: api response time 20000ms",
    ]
    monitor.alerts_enabled = False
    monitor._process_alerts()
    assert len(alerts) == 3


def test_cleanup_history_and_overall_health_statuses():
    monitor = hm.HealthMonitor()
    monitor.checkers.clear()
    assert monitor.get_overall_health()["status"] == "unknown"

    old_result = hm.HealthCheckResult(
        component="old",
        component_type=hm.ComponentType.API_SERVICE,
        status=hm.HealthStatus.HEALTHY,
        message="old",
        response_time_ms=1,
        timestamp=datetime.now(UTC) - timedelta(days=2),
    )
    new_result = hm.HealthCheckResult(
        component="api",
        component_type=hm.ComponentType.API_SERVICE,
        status=hm.HealthStatus.WARNING,
        message="warn",
        response_time_ms=1,
        timestamp=datetime.now(UTC),
    )
    monitor.health_history = [old_result, new_result]
    monitor.system_metrics_history = [
        hm.SystemMetrics(0, 0, 0, 0, 0, 0, 0, (0, 0, 0), 0, 0, 0, old_result.timestamp),
        hm.SystemMetrics(1, 2, 3, 4, 5, 6, 7, (1, 2, 3), 4, 5, 6, new_result.timestamp),
    ]
    monitor.register_check("api", hm.ComponentType.API_SERVICE, lambda: True)
    monitor.checkers["api"].last_result = new_result

    monitor._cleanup_history()
    overall = monitor.get_overall_health()

    assert monitor.health_history == [new_result]
    assert len(monitor.system_metrics_history) == 1
    assert overall["status"] == "warning"
    assert overall["components"] == {"api": "warning"}
    assert overall["metrics"]["cpu_percent"] == 1


def test_default_checks_market_data_and_global_wrappers(monkeypatch):
    monitor = hm.HealthMonitor()
    monkeypatch.setattr(
        hm,
        "get_settings",
        lambda: SimpleNamespace(alpaca_data_feed="sip"),
    )
    assert monitor._check_trading_engine()["status"] == "healthy"
    assert monitor._check_market_data()["status"] == "warning"

    monkeypatch.setattr(hm, "_health_monitor", monitor)
    assert hm.get_health_monitor() is monitor
    assert hm.get_system_health()["status"] in {"unknown", "warning"}
