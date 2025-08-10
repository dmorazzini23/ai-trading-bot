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

# Core monitoring components
from .alerting import (
    Alert,
    AlertChannel,
    AlertManager,
    AlertSeverity,
    EmailAlerter,
    SlackAlerter,
)
from .performance_dashboard import (
    AnomalyDetector,
    PerformanceDashboard,
    PerformanceMetrics,
    RealTimePnLTracker,
)

# Import internal monitoring components
from .metrics import MetricsCollector, PerformanceMonitor
from .alerts import AlertType
from .dashboard import RealtimeMetrics


# Export all monitoring classes
__all__ = [
    # New alerting system
    "AlertManager",
    "EmailAlerter",
    "SlackAlerter",
    "Alert",
    "AlertSeverity",
    "AlertChannel",
    # New performance monitoring
    "PerformanceDashboard",
    "PerformanceMetrics",
    "RealTimePnLTracker",
    "AnomalyDetector",
    # Existing components
    "MetricsCollector",
    "PerformanceMonitor",
    "AlertType",
    "RealtimeMetrics",
]
