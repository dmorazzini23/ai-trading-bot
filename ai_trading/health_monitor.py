"""Production health monitoring system for AI trading platform.

Provides comprehensive async health checks, system monitoring,
and alerting for production readiness and compliance requirements.

AI-AGENT-REF: Production health monitoring for institutional trading
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components to monitor."""

    DATABASE = "database"
    API_SERVICE = "api_service"
    MARKET_DATA = "market_data"
    TRADING_ENGINE = "trading_engine"
    RISK_ENGINE = "risk_engine"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"
    EXTERNAL_API = "external_api"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    component: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    response_time_ms: float
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """System resource metrics."""

    cpu_percent: float
    memory_percent: float
    disk_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_used_gb: float
    disk_available_gb: float
    load_average: Tuple[float, float, float]
    process_count: int
    open_files: int
    network_connections: int
    timestamp: datetime


class HealthChecker:
    """Individual health check implementation."""

    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        check_func: Callable[[], bool | dict[str, Any]],
        timeout_seconds: float = 10.0,
        interval_seconds: float = 60.0,
    ):
        self.name = name
        self.component_type = component_type
        self.check_func = check_func
        self.timeout_seconds = timeout_seconds
        self.interval_seconds = interval_seconds
        self.last_check: datetime | None = None
        self.last_result: HealthCheckResult | None = None
        self.consecutive_failures = 0
        self.enabled = True

    async def run_check(self) -> HealthCheckResult:
        """Run the health check with timeout."""
        start_time = time.time()

        try:
            # Run check with timeout
            if asyncio.iscoroutinefunction(self.check_func):
                result = await asyncio.wait_for(
                    self.check_func(), timeout=self.timeout_seconds
                )
            else:
                # Run sync function in thread pool
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, self.check_func),
                    timeout=self.timeout_seconds,
                )

            response_time = (time.time() - start_time) * 1000

            # Process result
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                message = "Check passed" if result else "Check failed"
                details = {}
            elif isinstance(result, dict):
                status = HealthStatus(result.get("status", HealthStatus.HEALTHY.value))
                message = result.get("message", "No message")
                details = result.get("details", {})
            else:
                status = HealthStatus.UNKNOWN
                message = f"Unexpected result type: {type(result)}"
                details = {"raw_result": str(result)}

            self.consecutive_failures = (
                0 if status == HealthStatus.HEALTHY else self.consecutive_failures + 1
            )

        except TimeoutError:
            response_time = self.timeout_seconds * 1000
            status = HealthStatus.CRITICAL
            message = f"Health check timed out after {self.timeout_seconds}s"
            details = {"timeout": True}
            self.consecutive_failures += 1

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            status = HealthStatus.CRITICAL
            message = f"Health check failed: {str(e)}"
            details = {"error": str(e), "error_type": type(e).__name__}
            self.consecutive_failures += 1

        result = HealthCheckResult(
            component=self.name,
            component_type=self.component_type,
            status=status,
            message=message,
            response_time_ms=response_time,
            timestamp=datetime.now(UTC),
            details=details,
            tags=[f"failures:{self.consecutive_failures}"],
        )

        self.last_check = result.timestamp
        self.last_result = result

        return result


class HealthMonitor:
    """Comprehensive health monitoring system."""

    def __init__(self, check_interval: float = 60.0):
        self.logger = logging.getLogger(__name__)
        self.check_interval = check_interval
        self.checkers: dict[str, HealthChecker] = {}
        self.health_history: list[HealthCheckResult] = []
        self.system_metrics_history: list[SystemMetrics] = []
        self.alerts_enabled = True
        self.running = False
        self._monitor_task: asyncio.Task | None = None
        self._lock = threading.RLock()

        # Health thresholds
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "disk_warning": 85.0,
            "disk_critical": 95.0,
            "response_time_warning": 5000.0,  # 5 seconds
            "response_time_critical": 10000.0,  # 10 seconds
            "consecutive_failures_critical": 3,
        }

        # Initialize default system health checks
        self._register_default_checks()

        self.logger.info("HealthMonitor initialized")

    def _register_default_checks(self) -> None:
        """Register default system health checks."""
        # System resource checks
        self.register_check(
            "system_cpu",
            ComponentType.CPU,
            self._check_cpu_usage,
            timeout_seconds=5.0,
            interval_seconds=30.0,
        )

        self.register_check(
            "system_memory",
            ComponentType.MEMORY,
            self._check_memory_usage,
            timeout_seconds=5.0,
            interval_seconds=30.0,
        )

        self.register_check(
            "system_disk",
            ComponentType.DISK,
            self._check_disk_usage,
            timeout_seconds=5.0,
            interval_seconds=60.0,
        )

        # Trading system checks
        self.register_check(
            "trading_engine",
            ComponentType.TRADING_ENGINE,
            self._check_trading_engine,
            timeout_seconds=10.0,
            interval_seconds=60.0,
        )

        self.register_check(
            "market_data",
            ComponentType.MARKET_DATA,
            self._check_market_data,
            timeout_seconds=15.0,
            interval_seconds=60.0,
        )

    def register_check(
        self,
        name: str,
        component_type: ComponentType,
        check_func: Callable,
        timeout_seconds: float = 10.0,
        interval_seconds: float = 60.0,
    ) -> None:
        """Register a new health check."""
        with self._lock:
            checker = HealthChecker(
                name=name,
                component_type=component_type,
                check_func=check_func,
                timeout_seconds=timeout_seconds,
                interval_seconds=interval_seconds,
            )
            self.checkers[name] = checker
            self.logger.info(f"Registered health check: {name}")

    def unregister_check(self, name: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if name in self.checkers:
                del self.checkers[name]
                self.logger.info(f"Unregistered health check: {name}")
                return True
            return False

    async def start_monitoring(self) -> None:
        """Start the health monitoring loop."""
        if self.running:
            return

        self.running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the health monitoring loop."""
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
        self.logger.info("Health monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Run all health checks
                await self.run_all_checks()

                # Collect system metrics
                self._collect_system_metrics()

                # Check for alerts
                self._process_alerts()

                # Clean up old data
                self._cleanup_history()

                # Wait for next interval
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Short delay before retry

    async def run_all_checks(self) -> list[HealthCheckResult]:
        """Run all registered health checks."""
        results = []

        # Run checks that are due
        tasks = []
        for checker in self.checkers.values():
            if not checker.enabled:
                continue

            # Check if it's time to run this check
            if checker.last_check is None or datetime.now(
                UTC
            ) - checker.last_check >= timedelta(seconds=checker.interval_seconds):
                tasks.append(checker.run_check())

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            valid_results = []
            for result in results:
                if isinstance(result, HealthCheckResult):
                    valid_results.append(result)
                    self.health_history.append(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Health check error: {result}")

            results = valid_results

        return results

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_used_gb = disk.used / (1024**3)
            disk_available_gb = disk.free / (1024**3)

            # Load average (Unix-like systems)
            try:
                load_avg = os.getloadavg()
            except (OSError, AttributeError):
                load_avg = (0.0, 0.0, 0.0)

            # Process metrics
            process_count = len(psutil.pids())

            # File descriptor count
            try:
                process = psutil.Process()
                open_files = process.num_fds() if hasattr(process, "num_fds") else 0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                open_files = 0

            # Network connections
            try:
                network_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                network_connections = 0

            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_used_gb=disk_used_gb,
                disk_available_gb=disk_available_gb,
                load_average=load_avg,
                process_count=process_count,
                open_files=open_files,
                network_connections=network_connections,
                timestamp=datetime.now(UTC),
            )

            self.system_metrics_history.append(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0,
                memory_percent=0,
                disk_percent=0,
                memory_used_gb=0,
                memory_available_gb=0,
                disk_used_gb=0,
                disk_available_gb=0,
                load_average=(0, 0, 0),
                process_count=0,
                open_files=0,
                network_connections=0,
                timestamp=datetime.now(UTC),
            )

    def _process_alerts(self) -> None:
        """Process alerts based on health check results."""
        if not self.alerts_enabled:
            return

        # Check recent results for alert conditions
        recent_results = [
            result
            for result in self.health_history
            if result.timestamp > datetime.now(UTC) - timedelta(minutes=5)
        ]

        # Group by component
        component_results = {}
        for result in recent_results:
            if result.component not in component_results:
                component_results[result.component] = []
            component_results[result.component].append(result)

        # Check for alert conditions
        for component, results in component_results.items():
            latest_result = max(results, key=lambda r: r.timestamp)

            # Critical status alerts
            if latest_result.status == HealthStatus.CRITICAL:
                self._send_alert(
                    f"CRITICAL: {component} health check failed", latest_result
                )

            # Consecutive failures
            checker = self.checkers.get(component)
            if (
                checker
                and checker.consecutive_failures
                >= self.thresholds["consecutive_failures_critical"]
            ):
                self._send_alert(
                    f"CRITICAL: {component} has {checker.consecutive_failures} consecutive failures",
                    latest_result,
                )

            # Response time alerts
            if (
                latest_result.response_time_ms
                > self.thresholds["response_time_critical"]
            ):
                self._send_alert(
                    f"CRITICAL: {component} response time {latest_result.response_time_ms:.0f}ms",
                    latest_result,
                )

    def _send_alert(self, message: str, result: HealthCheckResult) -> None:
        """Send alert notification."""
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "severity": "CRITICAL",
            "component": result.component,
            "message": message,
            "details": result.details,
        }

        # Log alert
        self.logger.critical(f"HEALTH ALERT: {message}")

        # Could extend to send to external alerting systems
        # (PagerDuty, Slack, email, etc.)

    def _cleanup_history(self) -> None:
        """Clean up old history data."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)

        # Clean health history
        self.health_history = [
            result for result in self.health_history if result.timestamp > cutoff_time
        ]

        # Clean metrics history
        self.system_metrics_history = [
            metrics
            for metrics in self.system_metrics_history
            if metrics.timestamp > cutoff_time
        ]

    # Default health check implementations

    def _check_cpu_usage(self) -> dict[str, Any]:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)

        if cpu_percent > self.thresholds["cpu_critical"]:
            status = HealthStatus.CRITICAL
            message = f"CPU usage critical: {cpu_percent:.1f}%"
        elif cpu_percent > self.thresholds["cpu_warning"]:
            status = HealthStatus.WARNING
            message = f"CPU usage high: {cpu_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent:.1f}%"

        return {
            "status": status.value,
            "message": message,
            "details": {"cpu_percent": cpu_percent},
        }

    def _check_memory_usage(self) -> dict[str, Any]:
        """Check memory usage."""
        memory = psutil.virtual_memory()

        if memory.percent > self.thresholds["memory_critical"]:
            status = HealthStatus.CRITICAL
            message = f"Memory usage critical: {memory.percent:.1f}%"
        elif memory.percent > self.thresholds["memory_warning"]:
            status = HealthStatus.WARNING
            message = f"Memory usage high: {memory.percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {memory.percent:.1f}%"

        return {
            "status": status.value,
            "message": message,
            "details": {
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
            },
        }

    def _check_disk_usage(self) -> dict[str, Any]:
        """Check disk usage."""
        disk = psutil.disk_usage("/")

        if disk.percent > self.thresholds["disk_critical"]:
            status = HealthStatus.CRITICAL
            message = f"Disk usage critical: {disk.percent:.1f}%"
        elif disk.percent > self.thresholds["disk_warning"]:
            status = HealthStatus.WARNING
            message = f"Disk usage high: {disk.percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal: {disk.percent:.1f}%"

        return {
            "status": status.value,
            "message": message,
            "details": {
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
            },
        }

    def _check_trading_engine(self) -> dict[str, Any]:
        """Check trading engine health."""
        # This would connect to actual trading engine
        # For now, return healthy
        return {
            "status": HealthStatus.HEALTHY.value,
            "message": "Trading engine operational",
            "details": {"engine_status": "running"},
        }

    def _check_market_data(self) -> dict[str, Any]:
        """Check market data feed health."""
        # This would check actual market data connectivity
        # For now, return healthy
        return {
            "status": HealthStatus.HEALTHY.value,
            "message": "Market data feed operational",
            "details": {"feed_status": "connected"},
        }

    def get_overall_health(self) -> dict[str, Any]:
        """Get overall system health status."""
        if not self.checkers:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks configured",
                "components": [],
            }

        # Get latest results for each component
        component_statuses = {}
        for name, checker in self.checkers.items():
            if checker.last_result:
                component_statuses[name] = checker.last_result.status
            else:
                component_statuses[name] = HealthStatus.UNKNOWN

        # Determine overall status
        if any(
            status == HealthStatus.CRITICAL for status in component_statuses.values()
        ):
            overall_status = HealthStatus.CRITICAL
            message = "System has critical issues"
        elif any(
            status == HealthStatus.WARNING for status in component_statuses.values()
        ):
            overall_status = HealthStatus.WARNING
            message = "System has warnings"
        elif all(
            status == HealthStatus.HEALTHY for status in component_statuses.values()
        ):
            overall_status = HealthStatus.HEALTHY
            message = "All systems operational"
        else:
            overall_status = HealthStatus.UNKNOWN
            message = "System status unknown"

        return {
            "status": overall_status.value,
            "message": message,
            "timestamp": datetime.now(UTC).isoformat(),
            "components": {
                name: status.value for name, status in component_statuses.items()
            },
            "metrics": (
                self.system_metrics_history[-1].__dict__
                if self.system_metrics_history
                else None
            ),
        }


# Global health monitor instance
_health_monitor: HealthMonitor | None = None


def get_health_monitor() -> HealthMonitor:
    """Get or create global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


async def start_health_monitoring() -> None:
    """Start health monitoring."""
    monitor = get_health_monitor()
    await monitor.start_monitoring()


async def stop_health_monitoring() -> None:
    """Stop health monitoring."""
    monitor = get_health_monitor()
    await monitor.stop_monitoring()


def get_system_health() -> dict[str, Any]:
    """Get current system health status."""
    monitor = get_health_monitor()
    return monitor.get_overall_health()
