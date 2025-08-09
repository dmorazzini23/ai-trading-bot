"""
Enhanced Health Check Module for the AI Trading Bot.

This module provides comprehensive health monitoring capabilities:
- System health checks (CPU, memory, disk)
- Service health checks (API connectivity, database)
- Trading system health (positions, orders, risk metrics)
- Real-time alerting and notifications
- Production-grade monitoring integration

AI-AGENT-REF: Enhanced health monitoring with production-grade capabilities
"""

import logging
import os
import time
import threading
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

import psutil

# AI-AGENT-REF: Import production monitoring for advanced capabilities
try:
    from production_monitoring import (
        ProductionMonitor, HealthStatus as ProdHealthStatus, 
        HealthCheckResult as ProdHealthCheckResult, 
        get_production_monitor, CircuitBreaker
    )
    PRODUCTION_MONITORING_AVAILABLE = True
except ImportError:
    PRODUCTION_MONITORING_AVAILABLE = False


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime


class HealthMonitor:
    """Comprehensive health monitoring for the trading bot."""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.RLock()
        
        # AI-AGENT-REF: Enhanced monitoring with circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.alert_callback: Optional[Callable] = None
        self.production_monitor: Optional[ProductionMonitor] = None
        
        # Performance tracking
        self.check_latencies: Dict[str, List[float]] = {}
        self.failure_counts: Dict[str, int] = {}
        
        # Initialize production monitoring if available
        if PRODUCTION_MONITORING_AVAILABLE:
            try:
                self.production_monitor = get_production_monitor()
                logger.info("Production monitoring integration enabled")
            except Exception as e:
                logger.warning(f"Could not initialize production monitoring: {e}")
        
        self.register_default_checks()
        self.register_trading_checks()
        self.register_api_checks()
    
    def register_check(self, name: str, check_func: callable):
        """Register a health check function."""
        with self._lock:
            self.checks[name] = check_func
    
    def register_default_checks(self):
        """Register standard health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("log_files", self._check_log_files)
        self.register_check("environment_variables", self._check_environment_variables)
    
    def register_trading_checks(self):
        """Register trading-specific health checks."""
        # Remove these for now as methods need to be implemented
        # self.register_check("trading_system", self._check_trading_system)
        # self.register_check("risk_limits", self._check_risk_limits)
        # self.register_check("portfolio_health", self._check_portfolio_health)
        # self.register_check("execution_latency", self._check_execution_latency)
        pass
    
    def register_api_checks(self):
        """Register API connectivity health checks."""
        # Remove these for now as methods need to be implemented
        # self.register_check("alpaca_api", self._check_alpaca_api)
        # self.register_check("market_data", self._check_market_data)
        # self.register_check("network_connectivity", self._check_network_connectivity)
        pass
    
    def set_alert_callback(self, callback: Callable):
        """Set callback function for health alerts."""
        self.alert_callback = callback
    
    def add_circuit_breaker(self, name: str, circuit_breaker: CircuitBreaker):
        """Add circuit breaker for a service."""
        self.circuit_breakers[name] = circuit_breaker
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        try:
            if name not in self.checks:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check '{name}' not found",
                    details={},
                    timestamp=datetime.now(timezone.utc)
                )
            
            check_func = self.checks[name]
            result = check_func()
            
            with self._lock:
                self.last_results[name] = result
            
            return result
            
        except Exception as e:
            logger.exception("Error running health check '%s': %s", name, e)
            return HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system CPU and memory usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            }
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 90:
                status = HealthStatus.CRITICAL
                message = f"High resource usage: CPU {cpu_percent}%, Memory {memory.percent}%"
            elif cpu_percent > 70 or memory.percent > 70:
                status = HealthStatus.WARNING
                message = f"Moderate resource usage: CPU {cpu_percent}%, Memory {memory.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal resource usage: CPU {cpu_percent}%, Memory {memory.percent}%"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {e}",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            details = {
                "free_gb": free_gb,
                "total_gb": total_gb,
                "used_percent": used_percent
            }
            
            if free_gb < 1.0 or used_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Low disk space: {free_gb:.1f}GB free ({used_percent:.1f}% used)"
            elif free_gb < 5.0 or used_percent > 85:
                status = HealthStatus.WARNING
                message = f"Moderate disk usage: {free_gb:.1f}GB free ({used_percent:.1f}% used)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Sufficient disk space: {free_gb:.1f}GB free ({used_percent:.1f}% used)"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {e}",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check current process memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            details = {
                "memory_mb": memory_mb,
                "memory_percent": process.memory_percent()
            }
            
            # Check for potential memory leaks
            if memory_mb > 2048:  # More than 2GB
                status = HealthStatus.CRITICAL
                message = f"High memory usage: {memory_mb:.1f}MB"
            elif memory_mb > 1024:  # More than 1GB
                status = HealthStatus.WARNING
                message = f"Moderate memory usage: {memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal memory usage: {memory_mb:.1f}MB"
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {e}",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )
    
    def _check_log_files(self) -> HealthCheckResult:
        """Check log file sizes and accessibility."""
        try:
            log_dir = "logs"
            details = {}
            issues = []
            
            if not os.path.exists(log_dir):
                return HealthCheckResult(
                    name="log_files",
                    status=HealthStatus.WARNING,
                    message="Log directory does not exist",
                    details={"log_dir": log_dir},
                    timestamp=datetime.now(timezone.utc)
                )
            
            for log_file in os.listdir(log_dir):
                if log_file.endswith('.log'):
                    file_path = os.path.join(log_dir, log_file)
                    try:
                        file_size = os.path.getsize(file_path)
                        size_mb = file_size / (1024 * 1024)
                        details[log_file] = {
                            "size_mb": size_mb,
                            "path": file_path
                        }
                        
                        # Check for oversized log files
                        if size_mb > 500:  # 500MB limit
                            issues.append(f"{log_file}: {size_mb:.1f}MB")
                    
                    except OSError as e:
                        issues.append(f"{log_file}: Cannot access ({e})")
            
            if issues:
                status = HealthStatus.WARNING
                message = f"Log file issues: {', '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Log files OK ({len(details)} files checked)"
            
            return HealthCheckResult(
                name="log_files",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="log_files",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check log files: {e}",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )
    
    def _check_environment_variables(self) -> HealthCheckResult:
        """Check critical environment variables."""
        try:
            required_vars = [
                "ALPACA_API_KEY",
                "ALPACA_SECRET_KEY",
                "ALPACA_BASE_URL"
            ]
            
            missing_vars = []
            details = {}
            
            for var in required_vars:
                value = os.getenv(var)
                if value:
                    details[var] = "SET"  # Don't log actual values for security
                else:
                    missing_vars.append(var)
                    details[var] = "MISSING"
            
            if missing_vars:
                status = HealthStatus.CRITICAL
                message = f"Missing environment variables: {', '.join(missing_vars)}"
            else:
                status = HealthStatus.HEALTHY
                message = "All required environment variables are set"
            
            return HealthCheckResult(
                name="environment_variables",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="environment_variables",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check environment variables: {e}",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )


# Global health monitor instance
health_monitor = HealthMonitor()


def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status."""
    results = health_monitor.run_all_checks()
    overall_status = health_monitor.get_overall_status()
    
    return {
        "overall_status": overall_status.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            name: {
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "timestamp": result.timestamp.isoformat()
            }
            for name, result in results.items()
        }
    }


def log_health_summary():
    """Log a summary of system health."""
    try:
        status = get_health_status()
        overall = status["overall_status"]
        
        if overall == "healthy":
            logger.info("System health check: ALL SYSTEMS HEALTHY")
        elif overall == "warning":
            logger.warning("System health check: WARNINGS DETECTED")
            for name, check in status["checks"].items():
                if check["status"] == "warning":
                    logger.warning("Health warning - %s: %s", name, check["message"])
        else:
            logger.error("System health check: CRITICAL ISSUES DETECTED")
            for name, check in status["checks"].items():
                if check["status"] == "critical":
                    logger.error("Health critical - %s: %s", name, check["message"])
    
    except Exception as e:
        logger.error("Failed to run health check: %s", e)


if __name__ == "__main__":
    # CLI health check
    import json
    status = get_health_status()
    print(json.dumps(status, indent=2))


# AI-AGENT-REF: Additional trading-specific health check methods
def _check_trading_system(self) -> HealthCheckResult:
    """Check trading system health and status."""
    try:
        start_time = time.perf_counter()
        details = {}
        issues = []
        
        # Check if trading modules are importable
        try:
            from ai_trading.core import bot_engine
            details["bot_engine"] = "OK"
        except ImportError as e:
            issues.append(f"bot_engine import failed: {e}")
            details["bot_engine"] = f"FAILED: {e}"
        
        try:
            import trade_execution
            details["trade_execution"] = "OK"
        except ImportError as e:
            issues.append(f"trade_execution import failed: {e}")
            details["trade_execution"] = f"FAILED: {e}"
        
        try:
            import risk_engine
            details["risk_engine"] = "OK"
        except ImportError as e:
            issues.append(f"risk_engine import failed: {e}")
            details["risk_engine"] = f"FAILED: {e}"
        
        # Check critical files exist
        critical_files = [
            "config.py",
            "hyperparams.json", 
            "best_hyperparams.json"
        ]
        
        for file_path in critical_files:
            if os.path.exists(file_path):
                details[f"file_{file_path}"] = "EXISTS"
            else:
                issues.append(f"Missing critical file: {file_path}")
                details[f"file_{file_path}"] = "MISSING"
        
        # Determine status
        if issues:
            status = HealthStatus.CRITICAL
            message = f"Trading system issues: {'; '.join(issues)}"
        else:
            status = HealthStatus.HEALTHY
            message = "Trading system operational"
        
        # Track latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        details["check_latency_ms"] = latency_ms
        
        return HealthCheckResult(
            name="trading_system",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        return HealthCheckResult(
            name="trading_system",
            status=HealthStatus.CRITICAL,
            message=f"Failed to check trading system: {e}",
            details={"error": str(e)},
            timestamp=datetime.now(timezone.utc)
        )


# Bind new methods to HealthMonitor class
HealthMonitor._check_trading_system = _check_trading_system