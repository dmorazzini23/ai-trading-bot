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

# Import existing components if available
try:
    from .metrics import MetricsCollector, PerformanceMonitor
except ImportError:
    # Create placeholder classes if not available
    class MetricsCollector:
        pass

    class PerformanceMonitor:
        pass


try:
    from .alerts import AlertType
except ImportError:
    # Create placeholder if not available
    class AlertType:
        pass


try:
    from .dashboard import RealtimeMetrics
except ImportError:
    # Create placeholder if not available
    class RealtimeMetrics:
        pass


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
