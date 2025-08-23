"""
Monitoring Module - Institutional Grade Performance Monitoring

This module provides comprehensive monitoring capabilities for
institutional trading operations including:

- Real-time metrics collection and performance analysis
- Advanced alerting and risk monitoring systems
- Dashboard data providers and real-time interfaces
- System health monitoring and compliance tracking
- Multi-channel alerting (email, Slack, SMS)
- Performance dashboard with anomaly detection
- P&L tracking and position monitoring

The module is designed for institutional-scale operations with proper
monitoring, alerting, and compliance capabilities.
"""

from .alerting import Alert, AlertChannel, AlertManager, AlertSeverity, EmailAlerter, SlackAlerter
from .alerts import AlertType
from .dashboard import RealtimeMetrics
from .metrics import MetricsCollector, PerformanceMonitor
from .order_health_monitor import (
    OrderHealthMonitor,
    OrderInfo,
    _active_orders,
    _order_health_monitor,
    _order_tracking_lock,
    get_order_health_monitor,
)
from .performance_dashboard import (
    AnomalyDetector,
    PerformanceDashboard,
    PerformanceMetrics,
    RealTimePnLTracker,
)
from .system_health import snapshot_basic

__all__ = [
    "AlertManager",
    "EmailAlerter",
    "SlackAlerter",
    "Alert",
    "AlertSeverity",
    "AlertChannel",
    "PerformanceDashboard",
    "PerformanceMetrics",
    "RealTimePnLTracker",
    "AnomalyDetector",
    "MetricsCollector",
    "PerformanceMonitor",
    "AlertType",
    "RealtimeMetrics",
    "snapshot_basic",
]
__all__ += [
    "OrderHealthMonitor",
    "OrderInfo",
    "get_order_health_monitor",
    "_active_orders",
    "_order_tracking_lock",
    "_order_health_monitor",
]
