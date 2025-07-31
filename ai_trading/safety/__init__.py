"""
Production safety and monitoring package for live trading.

This package provides comprehensive safety features including:
- Real-time monitoring and alerting
- Emergency stop mechanisms
- Performance monitoring
- Kill switches and circuit breakers
"""

from .monitoring import (
    SafetyMonitor,
    KillSwitch,
    PerformanceMonitor,
    AlertSeverity,
    TradingState,
    console_alert_callback,
    file_alert_callback,
    emergency_close_all_positions,
    emergency_cancel_all_orders
)

__all__ = [
    "SafetyMonitor",
    "KillSwitch", 
    "PerformanceMonitor",
    "AlertSeverity",
    "TradingState",
    "console_alert_callback",
    "file_alert_callback",
    "emergency_close_all_positions",
    "emergency_cancel_all_orders"
]