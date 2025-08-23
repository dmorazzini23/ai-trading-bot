"""
Production safety and monitoring system for live trading.

This module provides comprehensive production safety features including:
- Real-time monitoring and alerting
- Emergency stop mechanisms
- Circuit breakers and kill switches
- Performance monitoring
- Risk threshold enforcement
"""
import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from ai_trading.logging import logger

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 'info'
    WARNING = 'warning'
    CRITICAL = 'critical'
    EMERGENCY = 'emergency'

class TradingState(Enum):
    """Trading system states."""
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    EMERGENCY_STOP = 'emergency_stop'
    MAINTENANCE = 'maintenance'

class SafetyMonitor:
    """
    Comprehensive safety monitoring system for live trading.

    Monitors key metrics and triggers alerts/actions when thresholds are exceeded.
    Provides emergency stop capabilities and real-time system health tracking.
    """

    def __init__(self):
        """Initialize safety monitoring system."""
        self.state = TradingState.STOPPED
        self.is_monitoring = False
        self.emergency_stop_triggered = False
        self.thresholds = {'max_daily_loss': 0.05, 'max_position_risk': 0.02, 'max_drawdown': 0.1, 'min_available_cash': 1000, 'max_orders_per_minute': 50, 'max_failed_orders': 10, 'circuit_breaker_threshold': 5}
        self.metrics = {'daily_pnl': 0.0, 'current_drawdown': 0.0, 'orders_this_minute': 0, 'failed_orders_count': 0, 'last_order_time': None, 'available_cash': 0.0, 'total_portfolio_value': 0.0, 'active_positions': 0, 'system_start_time': datetime.now(UTC)}
        self.alert_callbacks: list[Callable] = []
        self._monitor_thread = None
        self.emergency_actions: list[Callable] = []
        logger.info('SafetyMonitor initialized')

    def start_monitoring(self):
        """Start the safety monitoring system."""
        if self.is_monitoring:
            logger.warning('Safety monitoring already running')
            return
        self.is_monitoring = True
        self.state = TradingState.RUNNING
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info('Safety monitoring started')
        self._send_alert(AlertSeverity.INFO, 'Safety monitoring system started')

    def stop_monitoring(self):
        """Stop the safety monitoring system."""
        if not self.is_monitoring:
            return
        self.is_monitoring = False
        self.state = TradingState.STOPPED
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info('Safety monitoring stopped')
        self._send_alert(AlertSeverity.INFO, 'Safety monitoring system stopped')

    def emergency_stop(self, reason: str='Manual emergency stop'):
        """Trigger emergency stop of trading system."""
        logger.critical(f'EMERGENCY STOP TRIGGERED: {reason}')
        self.emergency_stop_triggered = True
        self.state = TradingState.EMERGENCY_STOP
        for action in self.emergency_actions:
            try:
                action(reason)
            except (ValueError, TypeError) as e:
                logger.error(f'Error executing emergency action: {e}')
        self._send_alert(AlertSeverity.EMERGENCY, f'Emergency stop triggered: {reason}')
        self.pause_trading('Emergency stop')

    def pause_trading(self, reason: str='Manual pause'):
        """Pause trading activities."""
        if self.state == TradingState.EMERGENCY_STOP:
            logger.warning('Cannot pause - system in emergency stop')
            return
        self.state = TradingState.PAUSED
        logger.warning(f'Trading paused: {reason}')
        self._send_alert(AlertSeverity.WARNING, f'Trading paused: {reason}')

    def resume_trading(self, reason: str='Manual resume'):
        """Resume trading activities."""
        if self.emergency_stop_triggered:
            logger.error('Cannot resume - emergency stop active. Manual reset required.')
            return False
        self.state = TradingState.RUNNING
        logger.info(f'Trading resumed: {reason}')
        self._send_alert(AlertSeverity.INFO, f'Trading resumed: {reason}')
        return True

    def reset_emergency_stop(self, authorization_code: str=None):
        """Reset emergency stop (requires authorization)."""
        if authorization_code != 'RESET_AUTHORIZED':
            logger.error('Emergency stop reset denied - invalid authorization')
            return False
        self.emergency_stop_triggered = False
        self.state = TradingState.STOPPED
        self.metrics['failed_orders_count'] = 0
        logger.warning('Emergency stop reset - system ready for restart')
        self._send_alert(AlertSeverity.WARNING, 'Emergency stop reset - manual restart required')
        return True

    def update_metrics(self, **kwargs):
        """Update monitoring metrics."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key] = value
        if 'new_order' in kwargs:
            self.metrics['last_order_time'] = datetime.now(UTC)
            self.metrics['orders_this_minute'] += 1

    def check_safety_thresholds(self) -> list[dict]:
        """Check all safety thresholds and return any violations."""
        violations = []
        if abs(self.metrics['daily_pnl']) > self.thresholds['max_daily_loss'] * self.metrics['total_portfolio_value']:
            violations.append({'type': 'daily_loss_limit', 'severity': AlertSeverity.CRITICAL, 'message': f"Daily loss limit exceeded: {self.metrics['daily_pnl']:.2f}", 'action': 'emergency_stop'})
        if self.metrics['current_drawdown'] > self.thresholds['max_drawdown']:
            violations.append({'type': 'max_drawdown', 'severity': AlertSeverity.CRITICAL, 'message': f"Maximum drawdown exceeded: {self.metrics['current_drawdown']:.2%}", 'action': 'pause_trading'})
        if self.metrics['available_cash'] < self.thresholds['min_available_cash']:
            violations.append({'type': 'low_cash', 'severity': AlertSeverity.WARNING, 'message': f"Low available cash: ${self.metrics['available_cash']:.2f}", 'action': 'alert_only'})
        if self.metrics['orders_this_minute'] > self.thresholds['max_orders_per_minute']:
            violations.append({'type': 'order_rate_limit', 'severity': AlertSeverity.WARNING, 'message': f"Order rate limit exceeded: {self.metrics['orders_this_minute']} orders/minute", 'action': 'pause_trading'})
        if self.metrics['failed_orders_count'] > self.thresholds['max_failed_orders']:
            violations.append({'type': 'failed_orders', 'severity': AlertSeverity.CRITICAL, 'message': f"Too many failed orders: {self.metrics['failed_orders_count']}", 'action': 'emergency_stop'})
        return violations

    def get_system_health(self) -> dict[str, Any]:
        """Get comprehensive system health report."""
        uptime = datetime.now(UTC) - self.metrics['system_start_time']
        health = {'status': self.state.value, 'emergency_stop_active': self.emergency_stop_triggered, 'monitoring_active': self.is_monitoring, 'uptime_seconds': uptime.total_seconds(), 'metrics': self.metrics.copy(), 'thresholds': self.thresholds.copy(), 'violations': self.check_safety_thresholds(), 'timestamp': datetime.now(UTC).isoformat()}
        return health

    def add_alert_callback(self, callback: Callable):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def add_emergency_action(self, action: Callable):
        """Add emergency action to execute during emergency stop."""
        self.emergency_actions.append(action)

    def _monitoring_loop(self):
        """Main monitoring loop."""
        minute_reset_time = datetime.now(UTC)
        while self.is_monitoring:
            try:
                current_time = datetime.now(UTC)
                if (current_time - minute_reset_time).total_seconds() >= 60:
                    self.metrics['orders_this_minute'] = 0
                    minute_reset_time = current_time
                violations = self.check_safety_thresholds()
                for violation in violations:
                    logger.warning(f"Safety violation: {violation['message']}")
                    self._send_alert(violation['severity'], violation['message'])
                    if violation['action'] == 'emergency_stop':
                        self.emergency_stop(f"Safety violation: {violation['type']}")
                        break
                    elif violation['action'] == 'pause_trading':
                        self.pause_trading(f"Safety violation: {violation['type']}")
                time.sleep(1)
            except (ValueError, TypeError) as e:
                logger.error(f'Error in monitoring loop: {e}')
                time.sleep(5)

    def _send_alert(self, severity: AlertSeverity, message: str):
        """Send alert to all registered callbacks."""
        alert = {'severity': severity.value, 'message': message, 'timestamp': datetime.now(UTC).isoformat(), 'system_state': self.state.value}
        if severity == AlertSeverity.EMERGENCY:
            logger.critical(f'EMERGENCY ALERT: {message}')
        elif severity == AlertSeverity.CRITICAL:
            logger.critical(f'CRITICAL ALERT: {message}')
        elif severity == AlertSeverity.WARNING:
            logger.warning(f'WARNING ALERT: {message}')
        else:
            logger.info(f'INFO ALERT: {message}')
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except (ValueError, TypeError) as e:
                logger.error(f'Error in alert callback: {e}')

class KillSwitch:
    """
    Emergency kill switch for immediate trading halt.

    Provides multiple trigger mechanisms for emergency stops
    including file-based, time-based, and programmatic triggers.
    """

    def __init__(self, safety_monitor: SafetyMonitor):
        """Initialize kill switch."""
        self.safety_monitor = safety_monitor
        self.kill_file_path = 'KILL_SWITCH.flag'
        self.auto_kill_time = None
        self.is_monitoring = False
        self._monitor_thread = None
        logger.info('Kill switch initialized')

    def start_monitoring(self):
        """Start kill switch monitoring."""
        if self.is_monitoring:
            return
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._kill_switch_monitor, daemon=True)
        self._monitor_thread.start()
        logger.info('Kill switch monitoring started')

    def stop_monitoring(self):
        """Stop kill switch monitoring."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

    def set_auto_kill_time(self, kill_time: datetime):
        """Set automatic kill time."""
        self.auto_kill_time = kill_time
        logger.warning(f'Auto-kill time set for: {kill_time.isoformat()}')

    def trigger_kill_switch(self, reason: str='Manual kill switch'):
        """Manually trigger kill switch."""
        logger.critical(f'KILL SWITCH TRIGGERED: {reason}')
        self.safety_monitor.emergency_stop(f'Kill switch: {reason}')

    def _kill_switch_monitor(self):
        """Monitor for kill switch triggers."""
        while self.is_monitoring:
            try:
                if self._check_kill_file():
                    self.trigger_kill_switch('Kill file detected')
                    break
                if self.auto_kill_time and datetime.now(UTC) >= self.auto_kill_time:
                    self.trigger_kill_switch('Auto-kill time reached')
                    self.auto_kill_time = None
                time.sleep(1)
            except (ValueError, TypeError) as e:
                logger.error(f'Error in kill switch monitor: {e}')
                time.sleep(5)

    def _check_kill_file(self) -> bool:
        """Check if kill file exists."""
        import os
        if os.path.exists(self.kill_file_path):
            logger.critical(f'Kill file detected: {self.kill_file_path}')
            try:
                os.remove(self.kill_file_path)
            except (ValueError, TypeError) as e:
                logger.error(f'Could not remove kill file: {e}')
            return True
        return False

class PerformanceMonitor:
    """
    Performance monitoring and optimization system.

    Tracks execution metrics, identifies bottlenecks,
    and provides performance optimization recommendations.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {'order_latency': [], 'execution_times': [], 'memory_usage': [], 'cpu_usage': [], 'api_response_times': [], 'error_rates': {}, 'throughput': 0}
        self.start_time = datetime.now(UTC)
        logger.info('Performance monitor initialized')

    def record_order_latency(self, latency_ms: float):
        """Record order execution latency."""
        self.metrics['order_latency'].append({'timestamp': datetime.now(UTC), 'latency_ms': latency_ms})
        if len(self.metrics['order_latency']) > 1000:
            self.metrics['order_latency'] = self.metrics['order_latency'][-500:]

    def record_api_response_time(self, endpoint: str, response_time_ms: float):
        """Record API response time."""
        self.metrics['api_response_times'].append({'timestamp': datetime.now(UTC), 'endpoint': endpoint, 'response_time_ms': response_time_ms})

    def record_error(self, error_type: str):
        """Record error occurrence."""
        if error_type not in self.metrics['error_rates']:
            self.metrics['error_rates'][error_type] = 0
        self.metrics['error_rates'][error_type] += 1

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        uptime = datetime.now(UTC) - self.start_time
        latencies = [l['latency_ms'] for l in self.metrics['order_latency']]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        api_times = [a['response_time_ms'] for a in self.metrics['api_response_times']]
        avg_api_time = sum(api_times) / len(api_times) if api_times else 0
        total_errors = sum(self.metrics['error_rates'].values())
        report = {'uptime_seconds': uptime.total_seconds(), 'average_order_latency_ms': avg_latency, 'max_order_latency_ms': max_latency, 'average_api_response_ms': avg_api_time, 'total_orders_processed': len(latencies), 'total_errors': total_errors, 'error_rate': total_errors / len(latencies) if latencies else 0, 'throughput_orders_per_hour': len(latencies) / (uptime.total_seconds() / 3600) if uptime.total_seconds() > 0 else 0, 'error_breakdown': self.metrics['error_rates'].copy(), 'performance_grade': self._calculate_performance_grade(avg_latency, total_errors, len(latencies))}
        return report

    def _calculate_performance_grade(self, avg_latency: float, total_errors: int, total_orders: int) -> str:
        """Calculate performance grade based on metrics."""
        score = 100
        if avg_latency > 100:
            score -= 20
        elif avg_latency > 50:
            score -= 10
        if total_orders > 0:
            error_rate = total_errors / total_orders
            if error_rate > 0.05:
                score -= 30
            elif error_rate > 0.01:
                score -= 15
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

def console_alert_callback(alert: dict[str, Any]):
    """Simple console alert callback."""
    severity = alert['severity'].upper()
    message = alert['message']
    timestamp = alert['timestamp']
    logger.info(f'[{timestamp}] {severity}: {message}')

def file_alert_callback(alert: dict[str, Any]):
    """File-based alert logging callback."""
    try:
        with open('trading_alerts.log', 'a') as f:
            f.write(f'{json.dumps(alert)}\n')
    except (ValueError, TypeError) as e:
        logger.error(f'Failed to write alert to file: {e}')

def emergency_close_all_positions(reason: str):
    """Emergency action to close all positions."""
    logger.critical(f'EMERGENCY ACTION: Closing all positions - {reason}')

def emergency_cancel_all_orders(reason: str):
    """Emergency action to cancel all pending orders."""
    logger.critical(f'EMERGENCY ACTION: Cancelling all orders - {reason}')