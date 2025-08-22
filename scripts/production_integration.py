#!/usr/bin/env python3
"""Production integration module for AI trading bot.

This module integrates all production-grade systems:
- Production monitoring with the trading engine
- Performance optimization for critical trading functions
- Security management for API and data protection
- Real-time monitoring dashboard
- Enhanced health checks and circuit breakers

AI-AGENT-REF: Production integration for institutional-grade trading
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

# AI-AGENT-REF: Import all production systems
try:
    from production_monitoring import (
        CircuitBreaker,
        ProductionMonitor,
        get_production_monitor,
        initialize_production_monitoring,
    )
    PRODUCTION_MONITORING_AVAILABLE = True
except ImportError:
    PRODUCTION_MONITORING_AVAILABLE = False

try:
    from performance_optimizer import (
        cached,
        get_performance_optimizer,
        initialize_performance_optimizer,
        profile_performance,
    )
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False

try:
    from security_manager import get_security_manager, initialize_security_manager
    SECURITY_MANAGER_AVAILABLE = True
except ImportError:
    SECURITY_MANAGER_AVAILABLE = False

try:
    from monitoring_dashboard import (
        get_monitoring_dashboard,
        initialize_monitoring_dashboard,
    )
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

try:
    from health_check import health_monitor
    HEALTH_CHECK_AVAILABLE = True
except ImportError:
    HEALTH_CHECK_AVAILABLE = False


class ProductionIntegrator:
    """Main integration class for production systems."""

    def __init__(self, enable_all: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_all = enable_all

        # Initialize systems
        self.production_monitor: ProductionMonitor | None = None
        self.performance_optimizer = None
        self.security_manager = None
        self.monitoring_dashboard = None

        # Integration flags
        self.systems_initialized = False
        self.monitoring_active = False

        # Circuit breakers for key services
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        self.logger.info("Production integrator initialized")

    def initialize_all_systems(self, alert_callback: Callable | None = None) -> bool:
        """Initialize all production systems."""
        try:
            success_count = 0
            total_systems = 0

            # Initialize production monitoring
            if PRODUCTION_MONITORING_AVAILABLE and self.enable_all:
                total_systems += 1
                try:
                    self.production_monitor = initialize_production_monitoring(alert_callback)
                    self.logger.info("✓ Production monitoring initialized")
                    success_count += 1
                except (ValueError, TypeError) as e:
                    self.logger.error(f"✗ Failed to initialize production monitoring: {e}")

            # Initialize performance optimizer
            if PERFORMANCE_OPTIMIZER_AVAILABLE and self.enable_all:
                total_systems += 1
                try:
                    self.performance_optimizer = initialize_performance_optimizer(True)
                    self.logger.info("✓ Performance optimizer initialized")
                    success_count += 1
                except (ValueError, TypeError) as e:
                    self.logger.error(f"✗ Failed to initialize performance optimizer: {e}")

            # Initialize security manager
            if SECURITY_MANAGER_AVAILABLE and self.enable_all:
                total_systems += 1
                try:
                    self.security_manager = initialize_security_manager(True)
                    self.logger.info("✓ Security manager initialized")
                    success_count += 1
                except (ValueError, TypeError) as e:
                    self.logger.error(f"✗ Failed to initialize security manager: {e}")

            # Initialize monitoring dashboard
            if DASHBOARD_AVAILABLE and self.enable_all:
                total_systems += 1
                try:
                    self.monitoring_dashboard = initialize_monitoring_dashboard(5000)
                    self.logger.info("✓ Monitoring dashboard initialized")
                    success_count += 1
                except (ValueError, TypeError) as e:
                    self.logger.error(f"✗ Failed to initialize monitoring dashboard: {e}")

            # Setup circuit breakers
            self._setup_circuit_breakers()

            # Setup health checks integration
            self._setup_health_check_integration()

            self.systems_initialized = (success_count == total_systems and total_systems > 0)

            if self.systems_initialized:
                self.logger.info(f"✓ All {total_systems} production systems initialized successfully")
            else:
                self.logger.warning(f"⚠ Only {success_count}/{total_systems} production systems initialized")

            return self.systems_initialized

        except (ValueError, TypeError) as e:
            self.logger.error(f"Failed to initialize production systems: {e}")
            return False

    def _setup_circuit_breakers(self):
        """Setup circuit breakers for critical services."""
        if not self.production_monitor:
            return

        # Alpaca API circuit breaker
        alpaca_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        self.circuit_breakers['alpaca_api'] = alpaca_breaker
        # AI-AGENT-REF: Add defensive null checks for production systems
        if self.production_monitor is not None:
            self.production_monitor.register_circuit_breaker('alpaca_api', alpaca_breaker)

        # Data feed circuit breaker
        data_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=Exception
        )
        self.circuit_breakers['data_feed'] = data_breaker
        # AI-AGENT-REF: Add defensive null checks for production systems
        if self.production_monitor is not None:
            self.production_monitor.register_circuit_breaker('data_feed', data_breaker)

        self.logger.info("Circuit breakers configured for critical services")

    def _setup_health_check_integration(self):
        """Setup health check integration with production monitoring."""
        if not (self.production_monitor and HEALTH_CHECK_AVAILABLE):
            return

        try:
            # Register health checks with production monitor
            def trading_system_health():
                from health_check import HealthCheckResult, HealthStatus
                # Simplified health check
                return HealthCheckResult(
                    service="trading_system",
                    status=HealthStatus.HEALTHY,
                    latency_ms=1.0,
                    message="Trading system operational",
                    details={},
                    timestamp=datetime.now(UTC)
                )

            # AI-AGENT-REF: Add defensive null checks for production systems
            if self.production_monitor is not None:
                self.production_monitor.register_health_check(
                    "trading_system", trading_system_health
                )

            self.logger.info("Health check integration configured")

        except (ValueError, TypeError) as e:
            self.logger.error(f"Failed to setup health check integration: {e}")

    def start_monitoring(self):
        """Start all monitoring systems."""
        if not self.systems_initialized:
            self.logger.warning("Systems not initialized, cannot start monitoring")
            return False

        try:
            # Start production monitoring
            if self.production_monitor:
                self.production_monitor.start_monitoring(interval_seconds=30)

            # Start dashboard monitoring
            if self.monitoring_dashboard:
                self.monitoring_dashboard.start_monitoring(interval_seconds=60)

            self.monitoring_active = True
            self.logger.info("✓ All monitoring systems started")
            return True

        except (ValueError, TypeError) as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return False

    def stop_monitoring(self):
        """Stop all monitoring systems."""
        try:
            if self.production_monitor:
                self.production_monitor.stop_monitoring()

            if self.monitoring_dashboard:
                self.monitoring_dashboard.stop_monitoring()

            self.monitoring_active = False
            self.logger.info("✓ All monitoring systems stopped")

        except (ValueError, TypeError) as e:
            self.logger.error(f"Error stopping monitoring: {e}")

    def wrap_trading_function(self, func: Callable, operation_name: str = None) -> Callable:
        """Wrap trading function with production monitoring and optimization."""
        operation_name = operation_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Security check for sensitive operations
            # AI-AGENT-REF: Add defensive null checks for production systems
            if self.security_manager is not None and 'order' in operation_name.lower():
                # Add security logging for order operations
                self.security_manager.audit_logger.log_event(
                    "FUNCTION_CALL",
                    {"function": operation_name, "args_count": len(args)},
                    user_id="system"
                )

            # Performance monitoring
            start_time = time.perf_counter()

            try:
                # Execute with circuit breaker if applicable
                if 'alpaca' in operation_name.lower() and 'alpaca_api' in self.circuit_breakers:
                    circuit_breaker = self.circuit_breakers['alpaca_api']
                    return circuit_breaker(func)(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Track successful execution
                execution_time = (time.perf_counter() - start_time) * 1000
                # AI-AGENT-REF: Add defensive null checks for production systems
                if self.production_monitor is not None:
                    self.production_monitor.track_latency(operation_name, execution_time)

                return result

            except (ValueError, TypeError) as e:
                # Track failed execution
                execution_time = (time.perf_counter() - start_time) * 1000
                # AI-AGENT-REF: Add defensive null checks for production systems
                if self.production_monitor is not None:
                    self.production_monitor.track_latency(f"{operation_name}_failed", execution_time)

                # Security logging for failures
                # AI-AGENT-REF: Add defensive null checks for production systems
                if self.security_manager is not None:
                    self.security_manager.audit_logger.log_event(
                        "FUNCTION_ERROR",
                        {"function": operation_name, "error": str(e)},
                        user_id="system"
                    )

                raise

        return wrapper

    def secure_api_endpoint(self, func: Callable) -> Callable:
        """Secure API endpoint with authentication and rate limiting."""
        if not self.security_manager:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract request information (this would be adapted based on framework)
            client_ip = kwargs.get('client_ip', 'unknown')
            api_key = kwargs.get('api_key', '')
            signature = kwargs.get('signature', '')
            timestamp = kwargs.get('timestamp', str(time.time()))
            body = kwargs.get('body', '')

            # Authenticate request
            # AI-AGENT-REF: Add defensive null checks for production systems
            if self.security_manager is not None:
                if not self.security_manager.authenticate_api_request(
                    api_key, signature, timestamp, body, client_ip
                ):
                    raise Exception("Authentication failed")
            else:
                self.logger.warning("Security manager not available, skipping authentication")

            return func(*args, **kwargs)

        return wrapper

    def monitor_trade_execution(self, symbol: str, side: str, quantity: float,
                              price: float, pnl: float = 0.0, order_id: str = None):
        """Monitor trade execution across all systems."""
        try:
            # Record in dashboard
            if self.monitoring_dashboard:
                self.monitoring_dashboard.record_trade(
                    symbol, side, quantity, price, pnl, order_id
                )

            # Security anomaly detection
            # AI-AGENT-REF: Add defensive null checks for production systems
            if self.security_manager is not None:
                anomaly = self.security_manager.analyze_trade_for_anomalies(
                    symbol, side, quantity, price
                )
                if anomaly:
                    self.logger.warning(f"Trade anomaly detected: {anomaly}")

            # Audit logging
            # AI-AGENT-REF: Add defensive null checks for production systems
            if self.security_manager is not None:
                self.security_manager.audit_logger.log_trade_execution(
                    symbol, side, quantity, price, order_id or "unknown", "system"
                )

        except (ValueError, TypeError) as e:
            self.logger.error(f"Error monitoring trade execution: {e}")

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'timestamp': datetime.now(UTC).isoformat(),
            'systems_initialized': self.systems_initialized,
            'monitoring_active': self.monitoring_active,
            'systems': {}
        }

        # Production monitoring status
        if self.production_monitor:
            try:
                status['systems']['production_monitoring'] = {
                    'status': 'active',
                    'performance_report': self.production_monitor.get_performance_report()
                }
            except (ValueError, TypeError) as e:
                status['systems']['production_monitoring'] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Security status
        # AI-AGENT-REF: Add defensive null checks for production systems
        if self.security_manager is not None:
            try:
                status['systems']['security'] = {
                    'status': 'active',
                    'security_report': self.security_manager.get_security_report()
                }
            except (ValueError, TypeError) as e:
                status['systems']['security'] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Performance optimizer status
        # AI-AGENT-REF: Add defensive null checks for production systems
        if self.performance_optimizer is not None:
            try:
                status['systems']['performance'] = {
                    'status': 'active',
                    'performance_report': self.performance_optimizer.get_performance_report()
                }
            except (ValueError, TypeError) as e:
                status['systems']['performance'] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Health check status
        if HEALTH_CHECK_AVAILABLE:
            try:
                from health_check import get_health_status
                status['systems']['health'] = get_health_status()
            except (ValueError, TypeError) as e:
                status['systems']['health'] = {
                    'status': 'error',
                    'error': str(e)
                }

        return status

    def run_comprehensive_audit(self) -> dict[str, Any]:
        """Run comprehensive production audit."""
        audit_results = {
            'timestamp': datetime.now(UTC).isoformat(),
            'audit_type': 'comprehensive_production_audit',
            'systems_audited': [],
            'overall_score': 0,
            'critical_issues': [],
            'recommendations': []
        }

        total_score = 0
        systems_count = 0

        # Security audit
        # AI-AGENT-REF: Add defensive null checks for production systems
        if self.security_manager is not None:
            try:
                security_audit = self.security_manager.run_security_audit()
                audit_results['security_audit'] = security_audit
                audit_results['systems_audited'].append('security')
                total_score += security_audit['security_score']
                systems_count += 1

                if security_audit['security_score'] < 70:
                    audit_results['critical_issues'].extend(
                        security_audit['recommendations']
                    )

            except (ValueError, TypeError) as e:
                audit_results['security_audit'] = {'error': str(e)}

        # Performance audit
        # AI-AGENT-REF: Add defensive null checks for production systems
        if self.performance_optimizer is not None:
            try:
                perf_report = self.performance_optimizer.get_performance_report()
                audit_results['performance_audit'] = perf_report
                audit_results['systems_audited'].append('performance')

                # Score performance (simplified)
                perf_score = 100
                violations = perf_report.get('performance_violations', [])
                if violations:
                    perf_score -= len(violations) * 10

                total_score += max(0, perf_score)
                systems_count += 1

                if perf_score < 70:
                    audit_results['critical_issues'].append(
                        f"Performance violations detected: {len(violations)}"
                    )

            except (ValueError, TypeError) as e:
                audit_results['performance_audit'] = {'error': str(e)}

        # Health audit
        if HEALTH_CHECK_AVAILABLE:
            try:
                from health_check import get_health_status
                health_status = get_health_status()
                audit_results['health_audit'] = health_status
                audit_results['systems_audited'].append('health')

                # Score health
                health_score = 100
                if health_status['overall_status'] == 'critical':
                    health_score = 0
                elif health_status['overall_status'] == 'warning':
                    health_score = 60

                total_score += health_score
                systems_count += 1

                if health_score < 70:
                    audit_results['critical_issues'].append(
                        f"Health status: {health_status['overall_status']}"
                    )

            except (ValueError, TypeError) as e:
                audit_results['health_audit'] = {'error': str(e)}

        # Calculate overall score
        if systems_count > 0:
            audit_results['overall_score'] = total_score / systems_count

        # Generate recommendations
        if audit_results['overall_score'] < 80:
            audit_results['recommendations'].extend([
                "Review and address critical issues immediately",
                "Implement additional monitoring and alerting",
                "Consider reducing trading activity until issues resolved"
            ])
        elif audit_results['overall_score'] < 90:
            audit_results['recommendations'].extend([
                "Address identified issues during next maintenance window",
                "Implement additional safeguards for identified risks"
            ])
        else:
            audit_results['recommendations'].append(
                "System operating at production standards"
            )

        return audit_results


# Global production integrator instance
_production_integrator: ProductionIntegrator | None = None


def get_production_integrator() -> ProductionIntegrator:
    """Get global production integrator instance."""
    global _production_integrator
    if _production_integrator is None:
        _production_integrator = ProductionIntegrator()
    return _production_integrator


def initialize_production_systems(alert_callback: Callable | None = None) -> ProductionIntegrator:
    """Initialize all production systems."""
    global _production_integrator
    _production_integrator = ProductionIntegrator()

    if _production_integrator.initialize_all_systems(alert_callback):
        _production_integrator.start_monitoring()

    return _production_integrator


# Convenience decorators for existing code integration
def production_monitor(operation_name: str = None):
    """Decorator to add production monitoring to functions."""
    def decorator(func: Callable) -> Callable:
        integrator = get_production_integrator()
        return integrator.wrap_trading_function(func, operation_name)
    return decorator


def secure_endpoint():
    """Decorator to secure API endpoints."""
    def decorator(func: Callable) -> Callable:
        integrator = get_production_integrator()
        return integrator.secure_api_endpoint(func)
    return decorator
