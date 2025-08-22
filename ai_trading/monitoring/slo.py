"""
SLO monitoring and circuit breakers for performance thresholds.

Provides service level objective monitoring with automatic circuit
breaking when performance degrades beyond acceptable thresholds.
"""

import json
import logging
import threading
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SLOStatus(Enum):
    """SLO status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACHED = "breached"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SLOThreshold:
    """SLO threshold definition."""

    name: str
    warning_threshold: float
    critical_threshold: float
    breach_threshold: float
    window_minutes: int = 5
    min_samples: int = 3
    description: str = ""


@dataclass
class SLOMetric:
    """Individual SLO metric measurement."""

    timestamp: datetime
    value: float
    tags: dict[str, str] = field(default_factory=dict)


class SLOMonitor:
    """
    SLO monitor with circuit breakers and alerting.

    Tracks performance metrics against defined thresholds and
    triggers alerts and circuit breakers when SLOs are breached.
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize SLO monitor.

        Args:
            config_path: Optional path to SLO configuration file
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Metric storage
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # SLO definitions
        self._slo_thresholds: dict[str, SLOThreshold] = {}

        # Current status
        self._slo_status: dict[str, SLOStatus] = {}

        # Circuit breaker callbacks
        self._breach_callbacks: dict[str, list[Callable]] = defaultdict(list)

        # Alert history
        self._alert_history: deque = deque(maxlen=100)

        # Thread safety
        self._lock = threading.Lock()

        # Setup default SLOs
        self._setup_default_slos()

        # Load configuration if provided
        if config_path:
            self._load_config(config_path)

    def _setup_default_slos(self) -> None:
        """Setup default SLO thresholds."""
        # AI-AGENT-REF: Default SLO thresholds for trading system
        default_slos = {
            "order_latency_ms": SLOThreshold(
                name="order_latency_ms",
                warning_threshold=100.0,  # 100ms warning
                critical_threshold=500.0,  # 500ms critical
                breach_threshold=1000.0,  # 1s breach
                window_minutes=5,
                min_samples=5,
                description="Order execution latency",
            ),
            "position_skew_pct": SLOThreshold(
                name="position_skew_pct",
                warning_threshold=2.0,  # 2% skew warning
                critical_threshold=5.0,  # 5% skew critical
                breach_threshold=10.0,  # 10% skew breach
                window_minutes=1,
                min_samples=3,
                description="Position skew from target",
            ),
            "turnover_ratio": SLOThreshold(
                name="turnover_ratio",
                warning_threshold=1.5,  # 1.5x expected turnover
                critical_threshold=2.0,  # 2x expected turnover
                breach_threshold=3.0,  # 3x expected turnover
                window_minutes=10,
                min_samples=3,
                description="Trading turnover ratio",
            ),
            "live_sharpe_ratio": SLOThreshold(
                name="live_sharpe_ratio",
                warning_threshold=0.3,  # Below 0.3 warning
                critical_threshold=0.0,  # Below 0.0 critical
                breach_threshold=-0.5,  # Below -0.5 breach
                window_minutes=60,
                min_samples=10,
                description="Live trading Sharpe ratio",
            ),
            "error_rate_pct": SLOThreshold(
                name="error_rate_pct",
                warning_threshold=1.0,  # 1% error rate warning
                critical_threshold=5.0,  # 5% error rate critical
                breach_threshold=10.0,  # 10% error rate breach
                window_minutes=5,
                min_samples=10,
                description="System error rate",
            ),
            "data_staleness_minutes": SLOThreshold(
                name="data_staleness_minutes",
                warning_threshold=2.0,  # 2 minutes stale warning
                critical_threshold=5.0,  # 5 minutes stale critical
                breach_threshold=10.0,  # 10 minutes stale breach
                window_minutes=1,
                min_samples=1,
                description="Market data staleness",
            ),
            "pnl_drift_bps": SLOThreshold(
                name="pnl_drift_bps",
                warning_threshold=10.0,  # 10bps drift warning
                critical_threshold=25.0,  # 25bps drift critical
                breach_threshold=50.0,  # 50bps drift breach
                window_minutes=15,
                min_samples=5,
                description="P&L attribution drift",
            ),
        }

        for slo in default_slos.values():
            self.add_slo_threshold(slo)

    def add_slo_threshold(self, threshold: SLOThreshold) -> None:
        """
        Add SLO threshold definition.

        Args:
            threshold: SLO threshold to add
        """
        with self._lock:
            self._slo_thresholds[threshold.name] = threshold
            self._slo_status[threshold.name] = SLOStatus.HEALTHY

        self.logger.info(f"Added SLO threshold: {threshold.name}")

    def record_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Record a metric measurement.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional metric tags
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        metric = SLOMetric(timestamp=timestamp, value=value, tags=tags or {})

        with self._lock:
            self._metrics[name].append(metric)

            # Check SLO if threshold exists
            if name in self._slo_thresholds:
                self._check_slo(name)

    def _check_slo(self, name: str) -> None:
        """Check SLO status for a metric (called with lock held)."""
        threshold = self._slo_thresholds[name]
        metrics = self._metrics[name]

        if len(metrics) < threshold.min_samples:
            return

        # Filter to window
        cutoff_time = datetime.now(UTC) - timedelta(minutes=threshold.window_minutes)
        recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        if len(recent_metrics) < threshold.min_samples:
            return

        # Calculate average value in window
        avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)

        # Determine status
        old_status = self._slo_status[name]

        if avg_value >= threshold.breach_threshold:
            new_status = SLOStatus.BREACHED
        elif avg_value >= threshold.critical_threshold:
            new_status = SLOStatus.CRITICAL
        elif avg_value >= threshold.warning_threshold:
            new_status = SLOStatus.WARNING
        else:
            new_status = SLOStatus.HEALTHY

        # Handle inverted thresholds (lower is worse, like Sharpe ratio)
        if name in ["live_sharpe_ratio"]:
            if avg_value <= threshold.breach_threshold:
                new_status = SLOStatus.BREACHED
            elif avg_value <= threshold.critical_threshold:
                new_status = SLOStatus.CRITICAL
            elif avg_value <= threshold.warning_threshold:
                new_status = SLOStatus.WARNING
            else:
                new_status = SLOStatus.HEALTHY

        # Update status and alert if changed
        if new_status != old_status:
            self._slo_status[name] = new_status
            self._emit_alert(name, old_status, new_status, avg_value, threshold)

            # Trigger circuit breakers for breaches
            if new_status == SLOStatus.BREACHED:
                self._trigger_circuit_breakers(name, avg_value)

    def _emit_alert(
        self,
        metric_name: str,
        old_status: SLOStatus,
        new_status: SLOStatus,
        current_value: float,
        threshold: SLOThreshold,
    ) -> None:
        """Emit SLO status change alert."""
        alert_level = {
            SLOStatus.HEALTHY: AlertLevel.INFO,
            SLOStatus.WARNING: AlertLevel.WARNING,
            SLOStatus.CRITICAL: AlertLevel.CRITICAL,
            SLOStatus.BREACHED: AlertLevel.EMERGENCY,
        }[new_status]

        alert = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "SLO_STATUS_CHANGE",
            "level": alert_level.value,
            "metric": metric_name,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "current_value": current_value,
            "threshold": {
                "warning": threshold.warning_threshold,
                "critical": threshold.critical_threshold,
                "breach": threshold.breach_threshold,
            },
            "description": threshold.description,
        }

        self._alert_history.append(alert)

        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.ERROR,
            AlertLevel.EMERGENCY: logging.CRITICAL,
        }[alert_level]

        self.logger.log(
            log_level,
            f"SLO ALERT: {metric_name} changed from {old_status.value} to {new_status.value} "
            f"(value: {current_value:.3f})",
        )

    def _trigger_circuit_breakers(self, metric_name: str, current_value: float) -> None:
        """Trigger circuit breaker callbacks for SLO breach."""
        callbacks = self._breach_callbacks.get(metric_name, [])

        for callback in callbacks:
            try:
                callback(metric_name, current_value)
            except (ValueError, TypeError) as e:
                self.logger.error(
                    f"Error in circuit breaker callback for {metric_name}: {e}"
                )

    def register_circuit_breaker(
        self, metric_name: str, callback: Callable[[str, float], None]
    ) -> None:
        """
        Register circuit breaker callback for metric.

        Args:
            metric_name: Metric to monitor
            callback: Function to call when SLO is breached
        """
        with self._lock:
            self._breach_callbacks[metric_name].append(callback)

        self.logger.info(f"Registered circuit breaker for {metric_name}")

    def get_slo_status(self, metric_name: str | None = None) -> dict[str, Any]:
        """
        Get SLO status for metric(s).

        Args:
            metric_name: Specific metric name or None for all

        Returns:
            Dictionary with SLO status information
        """
        with self._lock:
            if metric_name is not None:
                if metric_name not in self._slo_thresholds:
                    return {"error": f"SLO {metric_name} not found"}

                threshold = self._slo_thresholds[metric_name]
                status = self._slo_status[metric_name]
                metrics = list(self._metrics[metric_name])

                # Get recent metrics for current value
                cutoff_time = datetime.now(UTC) - timedelta(
                    minutes=threshold.window_minutes
                )
                recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                current_value = (
                    sum(m.value for m in recent_metrics) / len(recent_metrics)
                    if recent_metrics
                    else None
                )

                return {
                    "metric": metric_name,
                    "status": status.value,
                    "current_value": current_value,
                    "threshold": {
                        "warning": threshold.warning_threshold,
                        "critical": threshold.critical_threshold,
                        "breach": threshold.breach_threshold,
                    },
                    "window_minutes": threshold.window_minutes,
                    "sample_count": len(recent_metrics),
                    "description": threshold.description,
                }

            # Return all SLO statuses
            result = {}
            for name in self._slo_thresholds:
                status_info = self.get_slo_status(name)
                if "error" not in status_info:
                    result[name] = status_info

            return result

    def get_alerts(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        with self._lock:
            return list(self._alert_history)[-limit:]

    def get_health_summary(self) -> dict[str, Any]:
        """Get overall system health summary."""
        with self._lock:
            status_counts = defaultdict(int)
            for status in self._slo_status.values():
                status_counts[status.value] += 1

            # Determine overall health
            if status_counts["breached"] > 0:
                overall_health = "critical"
            elif status_counts["critical"] > 0:
                overall_health = "degraded"
            elif status_counts["warning"] > 0:
                overall_health = "warning"
            else:
                overall_health = "healthy"

            return {
                "overall_health": overall_health,
                "status_counts": dict(status_counts),
                "total_slos": len(self._slo_thresholds),
                "recent_alerts": len(
                    [
                        a
                        for a in self._alert_history
                        if datetime.fromisoformat(a["timestamp"].replace("Z", "+00:00"))
                        > datetime.now(UTC) - timedelta(hours=1)
                    ]
                ),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def _load_config(self, config_path: str) -> None:
        """Load SLO configuration from file."""
        try:
            with open(config_path) as f:
                config = json.load(f)

            for slo_config in config.get("slos", []):
                threshold = SLOThreshold(
                    name=slo_config["name"],
                    warning_threshold=slo_config["warning_threshold"],
                    critical_threshold=slo_config["critical_threshold"],
                    breach_threshold=slo_config["breach_threshold"],
                    window_minutes=slo_config.get("window_minutes", 5),
                    min_samples=slo_config.get("min_samples", 3),
                    description=slo_config.get("description", ""),
                )
                self.add_slo_threshold(threshold)

            self.logger.info(f"Loaded SLO configuration from {config_path}")

        except (ValueError, TypeError) as e:
            self.logger.error(f"Error loading SLO configuration: {e}")


# Global SLO monitor instance
_global_slo_monitor: SLOMonitor | None = None


def get_slo_monitor() -> SLOMonitor:
    """Get or create global SLO monitor."""
    global _global_slo_monitor
    if _global_slo_monitor is None:
        _global_slo_monitor = SLOMonitor()
    return _global_slo_monitor


# Convenience functions
def record_latency(operation: str, latency_ms: float) -> None:
    """Record operation latency."""
    monitor = get_slo_monitor()
    monitor.record_metric(f"{operation}_latency_ms", latency_ms)


def record_error_rate(component: str, error_rate_pct: float) -> None:
    """Record component error rate."""
    monitor = get_slo_monitor()
    monitor.record_metric(f"{component}_error_rate_pct", error_rate_pct)


def record_performance_metric(
    name: str, value: float, tags: dict[str, str] | None = None
) -> None:
    """Record performance metric."""
    monitor = get_slo_monitor()
    monitor.record_metric(name, value, tags)


# AI-AGENT-REF: Example circuit breaker functions
def pause_trading_circuit_breaker(metric_name: str, value: float) -> None:
    """Circuit breaker to pause trading on critical SLO breach."""
    logger.critical(
        f"CIRCUIT BREAKER: Pausing trading due to {metric_name} breach (value: {value})"
    )
    # In real implementation, would trigger trading pause


def reduce_position_size_circuit_breaker(metric_name: str, value: float) -> None:
    """Circuit breaker to reduce position sizes on SLO breach."""
    logger.warning(
        f"CIRCUIT BREAKER: Reducing position sizes due to {metric_name} breach (value: {value})"
    )
    # In real implementation, would reduce risk budgets


def setup_default_circuit_breakers() -> None:
    """Setup default circuit breakers for critical SLOs."""
    monitor = get_slo_monitor()

    # Pause trading on critical latency or error rate
    monitor.register_circuit_breaker("order_latency_ms", pause_trading_circuit_breaker)
    monitor.register_circuit_breaker("error_rate_pct", pause_trading_circuit_breaker)

    # Reduce position size on performance degradation
    monitor.register_circuit_breaker(
        "live_sharpe_ratio", reduce_position_size_circuit_breaker
    )
    monitor.register_circuit_breaker(
        "position_skew_pct", reduce_position_size_circuit_breaker
    )
