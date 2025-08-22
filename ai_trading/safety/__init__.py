"""
Production safety and monitoring package for live trading.

This package provides comprehensive safety features including:
- Real-time monitoring and alerting
- Emergency stop mechanisms
- Performance monitoring
- Kill switches and circuit breakers
"""

from .monitoring import (
    AlertSeverity as AlertSeverity,
    KillSwitch as KillSwitch,
    PerformanceMonitor as PerformanceMonitor,
    SafetyMonitor as SafetyMonitor,
    TradingState as TradingState,
    console_alert_callback as console_alert_callback,
    emergency_cancel_all_orders as emergency_cancel_all_orders,
    emergency_close_all_positions as emergency_close_all_positions,
    file_alert_callback as file_alert_callback,
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
    "emergency_cancel_all_orders",
]
