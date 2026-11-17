"""
Advanced alerting and risk monitoring system.

Provides real-time alerts, risk monitoring, and notification
management for institutional trading operations.
"""
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import Enum
from ai_trading.logging import logger
from ai_trading.utils.time import safe_utcnow
from ai_trading.utils.timing import sleep as psleep
from ..core.constants import PERFORMANCE_THRESHOLDS, RISK_PARAMETERS

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 'info'
    WARNING = 'warning'
    CRITICAL = 'critical'
    EMERGENCY = 'emergency'

class AlertType(Enum):
    """Types of alerts."""
    RISK_LIMIT = 'risk_limit'
    PERFORMANCE = 'performance'
    EXECUTION = 'execution'
    SYSTEM = 'system'
    COMPLIANCE = 'compliance'
    PROVIDER_OUTAGE = 'provider_outage'

class Alert:
    """
    Trading alert representation.

    Encapsulates alert data including severity, type,
    and notification requirements.
    """

    def __init__(self, alert_type: AlertType, severity: AlertSeverity, message: str, **kwargs):
        """Initialize alert."""
        self.id = f'alert_{int(time.time() * 1000)}'
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.timestamp = safe_utcnow()
        self.symbol = kwargs.get('symbol')
        self.strategy_id = kwargs.get('strategy_id')
        self.value = kwargs.get('value')
        self.threshold = kwargs.get('threshold')
        self.metadata = kwargs.get('metadata', {})
        self.acknowledged = False
        self.resolved = False
        self.acknowledged_by: str | None = None
        self.acknowledged_at: datetime | None = None
        self.resolved_at: datetime | None = None

    def acknowledge(self, user: str='system'):
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = safe_utcnow()

    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = safe_utcnow()

    def to_dict(self) -> dict:
        """Convert alert to dictionary."""
        return {'id': self.id, 'type': self.alert_type.value, 'severity': self.severity.value, 'message': self.message, 'timestamp': self.timestamp.isoformat(), 'symbol': self.symbol, 'strategy_id': self.strategy_id, 'value': self.value, 'threshold': self.threshold, 'acknowledged': self.acknowledged, 'resolved': self.resolved, 'acknowledged_by': self.acknowledged_by, 'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None, 'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None, 'metadata': self.metadata}

class AlertManager:
    """
    Centralized alert management system.

    Manages alert generation, notification, escalation,
    and resolution tracking for institutional oversight.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.alerts: list[Alert] = []
        self.alert_handlers: list[Callable] = []
        self.notification_channels = {}
        self.max_alerts = 1000
        self.alert_retention_hours = 24
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._cleanup_running = False
        logger.info('AlertManager initialized')

    def create_alert(self, alert_type: AlertType, severity: AlertSeverity, message: str, **kwargs) -> Alert:
        """
        Create and register a new alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity level
            message: Alert message
            **kwargs: Additional alert data

        Returns:
            Created alert object
        """
        try:
            alert = Alert(alert_type, severity, message, **kwargs)
            with self._lock:
                self.alerts.append(alert)
                if len(self.alerts) > self.max_alerts:
                    self.alerts = self.alerts[-self.max_alerts:]
            self._notify_handlers(alert)
            log_message = f'Alert created: {alert.severity.value.upper()} - {message}'
            if severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY):
                logger.error(log_message)
            elif severity == AlertSeverity.WARNING:
                logger.warning(log_message)
            else:
                logger.info(log_message)
            return alert
        except (ValueError, TypeError) as e:
            logger.error(f'Error creating alert: {e}')
            return Alert(AlertType.SYSTEM, AlertSeverity.CRITICAL, f'Alert creation failed: {e}')

    def get_active_alerts(self, severity_filter: AlertSeverity=None) -> list[Alert]:
        """Get list of active (unresolved) alerts."""
        with self._lock:
            active_alerts = [a for a in self.alerts if not a.resolved]
            if severity_filter:
                active_alerts = [a for a in active_alerts if a.severity == severity_filter]
            return active_alerts.copy()

    def get_alerts_by_type(self, alert_type: AlertType) -> list[Alert]:
        """Get alerts by type."""
        with self._lock:
            return [a for a in self.alerts if a.alert_type == alert_type]

    def acknowledge_alert(self, alert_id: str, user: str='system') -> bool:
        """Acknowledge an alert."""
        try:
            with self._lock:
                for alert in self.alerts:
                    if alert.id == alert_id:
                        alert.acknowledge(user)
                        logger.info(f'Alert {alert_id} acknowledged by {user}')
                        return True
            logger.warning(f'Alert {alert_id} not found for acknowledgment')
            return False
        except (ValueError, TypeError) as e:
            logger.error(f'Error acknowledging alert {alert_id}: {e}')
            return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        try:
            with self._lock:
                for alert in self.alerts:
                    if alert.id == alert_id:
                        alert.resolve()
                        logger.info(f'Alert {alert_id} resolved')
                        return True
            logger.warning(f'Alert {alert_id} not found for resolution')
            return False
        except (ValueError, TypeError) as e:
            logger.error(f'Error resolving alert {alert_id}: {e}')
            return False

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler."""
        self.alert_handlers.append(handler)
        logger.debug(f'Alert handler added: {handler.__name__}')

    def _notify_handlers(self, alert: Alert):
        """Notify all registered alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except (ValueError, TypeError) as e:
                logger.error(f'Error in alert handler {handler.__name__}: {e}')

    def start_cleanup(self):
        """Start background cleanup thread."""
        if self._cleanup_running:
            return
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_alerts, daemon=True)
        self._cleanup_thread.start()
        logger.info('Alert cleanup thread started')

    def stop_cleanup(self):
        """Stop background cleanup thread."""
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        logger.info('Alert cleanup thread stopped')

    def _cleanup_old_alerts(self):
        """Background cleanup of old alerts."""
        while self._cleanup_running:
            try:
                cutoff_time = safe_utcnow() - timedelta(hours=self.alert_retention_hours)
                with self._lock:
                    self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
                psleep(3600)
            except (ValueError, TypeError) as e:
                logger.error(f'Error in alert cleanup: {e}')
                psleep(300)


def emit_runtime_alert(
    event: str,
    *,
    severity: str = "warning",
    details: dict | None = None,
    state: object | None = None,
    halt_reason: str | None = None,
) -> None:
    """Emit a structured ALERT_* runtime event and optionally set halt reason."""

    payload = dict(details or {})
    payload.setdefault("severity", severity)
    level = severity.strip().lower()
    if level in {"critical", "error"}:
        logger.error(event, extra=payload)
    elif level == "info":
        logger.info(event, extra=payload)
    else:
        logger.warning(event, extra=payload)

    if state is not None and halt_reason:
        try:
            setattr(state, "halt_reason", halt_reason)
        except (AttributeError, TypeError):
            pass


def _is_auth_halt_reason(halt_reason: str) -> bool:
    normalized = str(halt_reason or "").strip().upper()
    return normalized.startswith("AUTH_")


def evaluate_runtime_alert_thresholds(
    *,
    breaker_open: str | None = None,
    repeated_rejects: int = 0,
    reject_threshold: int = 3,
    data_stale_seconds: float | None = None,
    data_stale_threshold: float = 120.0,
    recon_mismatch: bool = False,
    halt_reason: str | None = None,
    state: object | None = None,
) -> None:
    """Evaluate thin threshold rules for runtime alerting."""

    if breaker_open:
        emit_runtime_alert(
            "ALERT_BREAKER_OPEN",
            severity="critical",
            details={"dependency": breaker_open},
            state=state,
            halt_reason=f"CIRCUIT_OPEN_{breaker_open}",
        )
    if repeated_rejects >= max(1, int(reject_threshold)):
        emit_runtime_alert(
            "ALERT_REPEATED_REJECTS",
            severity="warning",
            details={"count": repeated_rejects, "threshold": reject_threshold},
            state=state,
        )
    if data_stale_seconds is not None and data_stale_seconds > float(data_stale_threshold):
        emit_runtime_alert(
            "ALERT_DATA_STALE",
            severity="warning",
            details={"stale_seconds": data_stale_seconds, "threshold": data_stale_threshold},
            state=state,
        )
    if recon_mismatch:
        emit_runtime_alert(
            "ALERT_RECON_MISMATCH",
            severity="critical",
            details={"recon_mismatch": True},
            state=state,
            halt_reason="RECON_MISMATCH_HALT",
        )
    if halt_reason:
        emit_runtime_alert(
            "ALERT_HALT_REASON",
            severity="critical",
            details={"halt_reason": halt_reason},
            state=state,
            halt_reason=halt_reason,
        )
        if _is_auth_halt_reason(halt_reason):
            emit_runtime_alert(
                "ALERT_AUTH_HALT",
                severity="critical",
                details={"halt_reason": halt_reason},
                state=state,
                halt_reason=halt_reason,
            )
