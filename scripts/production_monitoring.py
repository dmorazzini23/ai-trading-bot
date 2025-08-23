"""Production-grade monitoring and performance tracking for AI trading bot.

This module provides comprehensive monitoring capabilities including:
- Real-time performance metrics and KPIs
- Health checks and readiness probes
- Circuit breakers for external dependencies
- Advanced alerting and anomaly detection
- Resource usage monitoring and optimization
"""
from __future__ import annotations
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any
import psutil

class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    UNHEALTHY = 'unhealthy'
    CRITICAL = 'critical'

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    order_latency_ms: float
    data_processing_latency_ms: float
    active_positions: int
    pending_orders: int
    daily_pnl: float
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None

@dataclass
class HealthCheckResult:
    """Health check result structure."""
    service: str
    status: HealthStatus
    latency_ms: float
    message: str
    details: dict[str, Any]
    timestamp: datetime

class CircuitBreaker:
    """Circuit breaker for external service protection."""

    def __init__(self, failure_threshold: int=5, recovery_timeout: int=60, expected_exception: type=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.RLock()

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to function."""

        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == CircuitBreakerState.OPEN:
                    if self._should_attempt_reset():
                        self.state = CircuitBreakerState.HALF_OPEN
                    else:
                        raise Exception(f'Circuit breaker OPEN for {func.__name__}')
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except self.expected_exception as e:
                    self._on_failure()
                    raise e
        return wrapper

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return self.last_failure_time and time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class ProductionMonitor:
    """Production-grade monitoring system for trading bot."""

    def __init__(self, alert_callback: Callable | None=None):
        self.logger = logging.getLogger(__name__)
        self.alert_callback = alert_callback
        self.metrics_history: deque = deque(maxlen=10000)
        self.latency_tracker: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.health_checks: dict[str, Callable] = {}
        self.last_health_results: dict[str, HealthCheckResult] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.baseline_metrics: dict[str, float] = {}
        self.anomaly_thresholds: dict[str, float] = {'cpu_percent': 80.0, 'memory_percent': 85.0, 'order_latency_ms': 50.0, 'data_processing_latency_ms': 20.0}
        self.start_time = time.time()
        self._monitoring_active = False
        self._monitor_thread = None
        self.performance_targets = {'order_execution_latency_ms': 10.0, 'data_processing_latency_ms': 5.0, 'cpu_utilization_percent': 70.0, 'memory_growth_percent_per_day': 1.0, 'uptime_percent': 99.9}

    def start_monitoring(self, interval_seconds: int=30):
        """Start continuous monitoring."""
        if self._monitoring_active:
            return
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval_seconds,), daemon=True)
        self._monitor_thread.start()
        self.logger.info('Production monitoring started')

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info('Production monitoring stopped')

    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                self._check_anomalies(metrics)
                self._run_health_checks()
                self._log_performance_summary(metrics)
            except (ValueError, TypeError) as e:
                self.logger.error(f'Error in monitoring loop: {e}')
            time.sleep(interval_seconds)

    def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system and trading metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
            order_latency = self._get_average_latency('order_execution')
            data_latency = self._get_average_latency('data_processing')
            return PerformanceMetrics(timestamp=datetime.now(UTC), cpu_percent=cpu_percent, memory_percent=memory_percent, memory_mb=memory_mb, order_latency_ms=order_latency, data_processing_latency_ms=data_latency, active_positions=0, pending_orders=0, daily_pnl=0.0)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error collecting metrics: {e}')
            return PerformanceMetrics(timestamp=datetime.now(UTC), cpu_percent=0.0, memory_percent=0.0, memory_mb=0.0, order_latency_ms=0.0, data_processing_latency_ms=0.0, active_positions=0, pending_orders=0, daily_pnl=0.0)

    def _get_average_latency(self, operation: str) -> float:
        """Get average latency for an operation."""
        if operation not in self.latency_tracker:
            return 0.0
        latencies = list(self.latency_tracker[operation])
        return statistics.mean(latencies) if latencies else 0.0

    def track_latency(self, operation: str, latency_ms: float):
        """Track latency for an operation."""
        self.latency_tracker[operation].append(latency_ms)

    @contextmanager
    def track_operation(self, operation: str):
        """Context manager to track operation latency."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            self.track_latency(operation, elapsed)

    def _check_anomalies(self, metrics: PerformanceMetrics):
        """Check for performance anomalies."""
        anomalies = []
        if metrics.cpu_percent > self.anomaly_thresholds['cpu_percent']:
            anomalies.append(f'High CPU usage: {metrics.cpu_percent:.1f}%')
        if metrics.memory_percent > self.anomaly_thresholds['memory_percent']:
            anomalies.append(f'High memory usage: {metrics.memory_percent:.1f}%')
        if metrics.order_latency_ms > self.anomaly_thresholds['order_latency_ms']:
            anomalies.append(f'High order latency: {metrics.order_latency_ms:.1f}ms')
        if metrics.data_processing_latency_ms > self.anomaly_thresholds['data_processing_latency_ms']:
            anomalies.append(f'High data processing latency: {metrics.data_processing_latency_ms:.1f}ms')
        if anomalies and self.alert_callback:
            self.alert_callback('PERFORMANCE_ANOMALY', anomalies)

    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Register a health check function."""
        self.health_checks[name] = check_func

    def _run_health_checks(self):
        """Run all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                self.last_health_results[name] = result
                if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    if self.alert_callback:
                        self.alert_callback('HEALTH_CHECK_FAILED', {'service': name, 'status': result.status.value, 'message': result.message})
            except (ValueError, TypeError) as e:
                self.logger.error(f'Health check {name} failed: {e}')
                self.last_health_results[name] = HealthCheckResult(service=name, status=HealthStatus.CRITICAL, latency_ms=0.0, message=f'Health check failed: {e}', details={}, timestamp=datetime.now(UTC))

    def get_health_status(self) -> dict[str, HealthCheckResult]:
        """Get latest health check results."""
        return self.last_health_results.copy()

    def register_circuit_breaker(self, name: str, circuit_breaker: CircuitBreaker):
        """Register a circuit breaker."""
        self.circuit_breakers[name] = circuit_breaker

    def get_circuit_breaker(self, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)

    def _log_performance_summary(self, metrics: PerformanceMetrics):
        """Log performance summary."""
        self.logger.info(f'Performance: CPU={metrics.cpu_percent:.1f}% Memory={metrics.memory_percent:.1f}% OrderLatency={metrics.order_latency_ms:.1f}ms DataLatency={metrics.data_processing_latency_ms:.1f}ms')

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        recent_metrics = list(self.metrics_history)[-100:]
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        order_latency_values = [m.order_latency_ms for m in recent_metrics if m.order_latency_ms > 0]
        report = {'timestamp': datetime.now(UTC).isoformat(), 'uptime_seconds': time.time() - self.start_time, 'metrics_collected': len(self.metrics_history), 'performance': {'cpu_percent': {'current': cpu_values[-1] if cpu_values else 0, 'average': statistics.mean(cpu_values) if cpu_values else 0, 'max': max(cpu_values) if cpu_values else 0, 'target': self.performance_targets['cpu_utilization_percent']}, 'memory_percent': {'current': memory_values[-1] if memory_values else 0, 'average': statistics.mean(memory_values) if memory_values else 0, 'max': max(memory_values) if memory_values else 0}, 'order_latency_ms': {'current': order_latency_values[-1] if order_latency_values else 0, 'average': statistics.mean(order_latency_values) if order_latency_values else 0, 'p95': statistics.quantiles(order_latency_values, n=20)[18] if len(order_latency_values) > 20 else 0, 'target': self.performance_targets['order_execution_latency_ms']}}, 'health_checks': {name: asdict(result) for name, result in self.last_health_results.items()}, 'circuit_breakers': {name: cb.state.value for name, cb in self.circuit_breakers.items()}}
        return report

    def alert(self, level: str, message: str, details: dict | None=None):
        """Send alert through configured callback."""
        if self.alert_callback:
            self.alert_callback(level, message, details or {})
        else:
            self.logger.error(f'ALERT [{level}]: {message}')
_production_monitor: ProductionMonitor | None = None

def get_production_monitor() -> ProductionMonitor:
    """Get global production monitor instance."""
    global _production_monitor
    if _production_monitor is None:
        _production_monitor = ProductionMonitor()
    return _production_monitor

def initialize_production_monitoring(alert_callback: Callable | None=None) -> ProductionMonitor:
    """Initialize production monitoring system."""
    global _production_monitor
    _production_monitor = ProductionMonitor(alert_callback)
    return _production_monitor