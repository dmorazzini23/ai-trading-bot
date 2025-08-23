"""
System Health Checker - Comprehensive monitoring for all trading bot components.

Provides centralized health monitoring for:
- Sentiment analysis success rates
- Meta-learning system status  
- Order execution performance
- Liquidity management effectiveness
- Overall system health metrics
"""
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from ai_trading.config import management as config
from ai_trading.config.management import TradingConfig
CONFIG = TradingConfig()
logger = logging.getLogger(__name__)

@dataclass
class ComponentHealth:
    """Health status for a system component."""
    name: str
    status: str
    success_rate: float
    last_check: datetime
    error_count: int = 0
    warning_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealthStatus:
    """Overall system health status."""
    overall_status: str
    components: dict[str, ComponentHealth]
    alerts: list[str]
    metrics: dict[str, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

class SystemHealthChecker:
    """
    Comprehensive system health monitoring for the trading bot.
    
    Monitors all critical components and provides centralized health reporting
    with automated alerting for degraded performance.
    """

    def __init__(self):
        """Initialize the system health checker."""
        self.logger = logging.getLogger(__name__ + '.SystemHealthChecker')
        self._component_metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._health_history: deque = deque(maxlen=1000)
        self.health_thresholds = {'sentiment': {'success_rate_warning': 0.8, 'success_rate_critical': 0.5, 'max_failures_per_hour': 20}, 'meta_learning': {'min_trades_for_health': 5, 'activation_time_warning': 1800, 'bootstrap_success_rate': 0.8}, 'order_execution': {'success_rate_warning': 0.7, 'success_rate_critical': 0.5, 'avg_fill_time_warning': 180, 'avg_fill_time_critical': 600}, 'liquidity_management': {'excessive_reduction_rate': 0.3, 'avg_reduction_threshold': 0.2}}
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        self._component_status: dict[str, ComponentHealth] = {}
        self.logger.info('SystemHealthChecker initialized')

    def start_monitoring(self, check_interval: int=60) -> None:
        """Start the system health monitoring."""
        if self._monitoring_active:
            self.logger.warning('System health monitoring already active')
            return
        self._monitoring_active = True
        self._check_interval = check_interval
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info('System health monitoring started (interval: %ds)', check_interval)

    def stop_monitoring(self) -> None:
        """Stop the system health monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        self.logger.info('System health monitoring stopped')

    def _monitor_loop(self) -> None:
        """Main health monitoring loop."""
        while self._monitoring_active:
            try:
                health_status = self._check_all_components()
                with self._lock:
                    self._health_history.append(health_status)
                if health_status.alerts:
                    self._process_health_alerts(health_status)
                self._log_health_summary(health_status)
                time.sleep(self._check_interval)
            except (ValueError, TypeError) as e:
                self.logger.error('Error in health monitoring loop: %s', e, exc_info=True)
                time.sleep(30)

    def _check_all_components(self) -> SystemHealthStatus:
        """Check health of all system components."""
        components = {}
        alerts = []
        metrics = {}
        sentiment_health = self._check_sentiment_health()
        components['sentiment'] = sentiment_health
        metrics['sentiment_success_rate'] = sentiment_health.success_rate
        meta_health = self._check_meta_learning_health()
        components['meta_learning'] = meta_health
        metrics['meta_learning_trades'] = meta_health.details.get('trade_count', 0)
        order_health = self._check_order_execution_health()
        components['order_execution'] = order_health
        metrics['order_success_rate'] = order_health.success_rate
        liquidity_health = self._check_liquidity_management_health()
        components['liquidity'] = liquidity_health
        metrics['liquidity_reduction_rate'] = liquidity_health.details.get('reduction_rate', 0)
        overall_status = self._determine_overall_status(components)
        for comp_name, comp_health in components.items():
            if comp_health.status == 'critical':
                alerts.append(f"{comp_name.title()}: CRITICAL - {comp_health.details.get('issue', 'Unknown issue')}")
            elif comp_health.status == 'warning':
                alerts.append(f"{comp_name.title()}: WARNING - {comp_health.details.get('issue', 'Performance degraded')}")
        return SystemHealthStatus(overall_status=overall_status, components=components, alerts=alerts, metrics=metrics)

    def _check_sentiment_health(self) -> ComponentHealth:
        """Check sentiment analysis component health."""
        try:
            import sentiment
            cache_size = len(sentiment._SENTIMENT_CACHE)
            circuit_breaker = sentiment._SENTIMENT_CIRCUIT_BREAKER
            if circuit_breaker['state'] == 'open':
                success_rate = 0.0
                status = 'critical'
                issue = f"Circuit breaker open, {circuit_breaker['failures']} failures"
            elif circuit_breaker['state'] == 'half-open':
                success_rate = 0.5
                status = 'warning'
                issue = 'Circuit breaker in recovery mode'
            else:
                max_failures = sentiment.SENTIMENT_FAILURE_THRESHOLD
                current_failures = circuit_breaker['failures']
                success_rate = max(0.0, 1.0 - current_failures / max_failures)
                if success_rate >= self.health_thresholds['sentiment']['success_rate_warning']:
                    status = 'healthy'
                    issue = 'Operating normally'
                elif success_rate >= self.health_thresholds['sentiment']['success_rate_critical']:
                    status = 'warning'
                    issue = f'Success rate {success_rate:.1%}, {current_failures} failures'
                else:
                    status = 'critical'
                    issue = f'Success rate {success_rate:.1%}, {current_failures} failures'
            return ComponentHealth(name='sentiment', status=status, success_rate=success_rate, last_check=datetime.now(UTC), error_count=circuit_breaker['failures'], details={'issue': issue, 'cache_size': cache_size, 'circuit_breaker_state': circuit_breaker['state'], 'last_failure': circuit_breaker['last_failure']})
        except (ValueError, TypeError) as e:
            self.logger.error('Failed to check sentiment health: %s', e)
            return ComponentHealth(name='sentiment', status='unknown', success_rate=0.0, last_check=datetime.now(UTC), details={'error': str(e)})

    def _check_meta_learning_health(self) -> ComponentHealth:
        """Check meta-learning system health."""
        try:
            from ai_trading import meta_learning
            trade_log_path = getattr(config, 'TRADE_LOG_FILE', 'trades.csv')
            if trade_log_path:
                quality_report = meta_learning.validate_trade_data_quality(trade_log_path)
                trade_count = quality_report.get('valid_price_rows', 0)
                min_trades = getattr(config, 'META_LEARNING_MIN_TRADES_REDUCED', 10)
                if trade_count >= min_trades:
                    success_rate = min(1.0, trade_count / (min_trades * 2))
                    status = 'healthy'
                    issue = f'Active with {trade_count} trades'
                elif trade_count >= 3:
                    success_rate = 0.6
                    status = 'warning'
                    issue = f'Bootstrap ready with {trade_count} trades'
                else:
                    success_rate = 0.0
                    status = 'critical'
                    issue = f'Insufficient data: {trade_count} trades (need {min_trades})'
                return ComponentHealth(name='meta_learning', status=status, success_rate=success_rate, last_check=datetime.now(UTC), details={'issue': issue, 'trade_count': trade_count, 'min_required': min_trades, 'data_quality_score': quality_report.get('data_quality_score', 0), 'bootstrap_enabled': getattr(config, 'META_LEARNING_BOOTSTRAP_ENABLED', True)})
            else:
                return ComponentHealth(name='meta_learning', status='warning', success_rate=0.0, last_check=datetime.now(UTC), details={'issue': 'Trade log path not configured'})
        except (ValueError, TypeError) as e:
            self.logger.error('Failed to check meta-learning health: %s', e)
            return ComponentHealth(name='meta_learning', status='unknown', success_rate=0.0, last_check=datetime.now(UTC), details={'error': str(e)})

    def _check_order_execution_health(self) -> ComponentHealth:
        """Check order execution system health."""
        try:
            try:
                from ai_trading.monitoring.order_health_monitor import get_order_health_monitor
                monitor = get_order_health_monitor()
                health_summary = monitor.get_health_summary()
                current_metrics = health_summary.get('current_metrics', {})
                success_rate = current_metrics.get('success_rate', 0.0)
                avg_fill_time = current_metrics.get('avg_fill_time', 0.0)
                stuck_orders = current_metrics.get('stuck_orders', 0)
                if success_rate >= self.health_thresholds['order_execution']['success_rate_warning'] and avg_fill_time <= self.health_thresholds['order_execution']['avg_fill_time_warning']:
                    status = 'healthy'
                    issue = 'Operating normally'
                elif success_rate >= self.health_thresholds['order_execution']['success_rate_critical'] and avg_fill_time <= self.health_thresholds['order_execution']['avg_fill_time_critical']:
                    status = 'warning'
                    issue = f'Performance degraded: {success_rate:.1%} success, {avg_fill_time:.1f}s avg fill'
                else:
                    status = 'critical'
                    issue = f'Poor performance: {success_rate:.1%} success, {avg_fill_time:.1f}s avg fill'
                return ComponentHealth(name='order_execution', status=status, success_rate=success_rate, last_check=datetime.now(UTC), details={'issue': issue, 'avg_fill_time': avg_fill_time, 'stuck_orders': stuck_orders, 'monitoring_active': health_summary.get('monitoring_active', False)})
            except ImportError:
                from trade_execution import _active_orders, _order_tracking_lock
                with _order_tracking_lock:
                    active_count = len(_active_orders)
                if active_count < 10:
                    status = 'healthy'
                    success_rate = 0.8
                    issue = f'{active_count} active orders'
                else:
                    status = 'warning'
                    success_rate = 0.6
                    issue = f'Many active orders: {active_count}'
                return ComponentHealth(name='order_execution', status=status, success_rate=success_rate, last_check=datetime.now(UTC), details={'issue': issue, 'active_orders': active_count, 'monitoring_available': False})
        except (ValueError, TypeError) as e:
            self.logger.error('Failed to check order execution health: %s', e)
            return ComponentHealth(name='order_execution', status='unknown', success_rate=0.0, last_check=datetime.now(UTC), details={'error': str(e)})

    def _check_liquidity_management_health(self) -> ComponentHealth:
        """Check liquidity management system health."""
        try:
            spread_threshold = getattr(config, 'LIQUIDITY_SPREAD_THRESHOLD', 0.1)
            vol_threshold = getattr(config, 'LIQUIDITY_VOL_THRESHOLD', 0.5)
            aggressive_reduction = getattr(config, 'LIQUIDITY_REDUCTION_AGGRESSIVE', 0.75)
            if spread_threshold <= 0.2 and aggressive_reduction >= 0.7:
                status = 'healthy'
                success_rate = 0.9
                issue = 'Thresholds configured appropriately'
            elif spread_threshold <= 0.5 and aggressive_reduction >= 0.5:
                status = 'warning'
                success_rate = 0.7
                issue = 'Thresholds moderately aggressive'
            else:
                status = 'critical'
                success_rate = 0.3
                issue = 'Thresholds too aggressive'
            return ComponentHealth(name='liquidity', status=status, success_rate=success_rate, last_check=datetime.now(UTC), details={'issue': issue, 'spread_threshold': spread_threshold, 'vol_threshold': vol_threshold, 'aggressive_reduction': aggressive_reduction, 'reduction_rate': 0.0})
        except (ValueError, TypeError) as e:
            self.logger.error('Failed to check liquidity management health: %s', e)
            return ComponentHealth(name='liquidity', status='unknown', success_rate=0.0, last_check=datetime.now(UTC), details={'error': str(e)})

    def _determine_overall_status(self, components: dict[str, ComponentHealth]) -> str:
        """Determine overall system status from component health."""
        if not components:
            return 'unknown'
        statuses = [comp.status for comp in components.values()]
        if 'critical' in statuses:
            return 'critical'
        elif 'warning' in statuses:
            return 'warning'
        elif all((status == 'healthy' for status in statuses)):
            return 'healthy'
        else:
            return 'unknown'

    def _process_health_alerts(self, health_status: SystemHealthStatus) -> None:
        """Process and log health alerts."""
        for alert in health_status.alerts:
            self.logger.error('SYSTEM_HEALTH_ALERT: %s', alert)

    def _log_health_summary(self, health_status: SystemHealthStatus) -> None:
        """Log periodic health summary."""
        summary = {'overall_status': health_status.overall_status, 'component_count': len(health_status.components), 'healthy_components': sum((1 for c in health_status.components.values() if c.status == 'healthy')), 'warning_components': sum((1 for c in health_status.components.values() if c.status == 'warning')), 'critical_components': sum((1 for c in health_status.components.values() if c.status == 'critical')), 'alert_count': len(health_status.alerts)}
        if health_status.overall_status == 'healthy':
            self.logger.info('SYSTEM_HEALTH_OK', extra=summary)
        else:
            self.logger.warning('SYSTEM_HEALTH_DEGRADED', extra=summary)

    def get_current_health(self) -> dict[str, Any]:
        """Get current system health status."""
        health_status = self._check_all_components()
        return {'overall_status': health_status.overall_status, 'components': {name: {'status': comp.status, 'success_rate': round(comp.success_rate, 3), 'last_check': comp.last_check.isoformat(), 'details': comp.details} for name, comp in health_status.components.items()}, 'alerts': health_status.alerts, 'metrics': health_status.metrics, 'timestamp': health_status.timestamp.isoformat()}

    def export_health_report(self, filepath: str) -> None:
        """Export comprehensive health report."""
        try:
            health_data = self.get_current_health()
            with self._lock:
                health_data['history'] = [{'timestamp': h.timestamp.isoformat(), 'overall_status': h.overall_status, 'alert_count': len(h.alerts), 'metrics': h.metrics} for h in list(self._health_history)[-50:]]
            with open(filepath, 'w') as f:
                json.dump(health_data, f, indent=2)
            self.logger.info('Health report exported to %s', filepath)
        except (ValueError, TypeError) as e:
            self.logger.error('Failed to export health report: %s', e)
_system_health_checker = None

def get_system_health_checker() -> SystemHealthChecker:
    """Get or create the global system health checker."""
    global _system_health_checker
    if _system_health_checker is None:
        _system_health_checker = SystemHealthChecker()
    return _system_health_checker

def start_system_health_monitoring(check_interval: int=60) -> None:
    """Start system health monitoring."""
    checker = get_system_health_checker()
    checker.start_monitoring(check_interval)

def stop_system_health_monitoring() -> None:
    """Stop system health monitoring."""
    if _system_health_checker:
        _system_health_checker.stop_monitoring()

def get_system_health() -> dict[str, Any]:
    """Get current system health status."""
    checker = get_system_health_checker()
    return checker.get_current_health()