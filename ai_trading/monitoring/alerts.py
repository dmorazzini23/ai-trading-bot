"""
Advanced alerting and risk monitoring system.

Provides real-time alerts, risk monitoring, and notification
management for institutional trading operations.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..core.constants import PERFORMANCE_THRESHOLDS, RISK_PARAMETERS


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts."""
    RISK_LIMIT = "risk_limit"
    PERFORMANCE = "performance"
    EXECUTION = "execution"
    SYSTEM = "system"
    COMPLIANCE = "compliance"


class Alert:
    """
    Trading alert representation.
    
    Encapsulates alert data including severity, type,
    and notification requirements.
    """
    
    def __init__(self, alert_type: AlertType, severity: AlertSeverity, 
                 message: str, **kwargs):
        """Initialize alert."""
        # AI-AGENT-REF: Trading alert representation
        self.id = f"alert_{int(time.time() * 1000)}"
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.timestamp = datetime.now()
        
        # Additional alert data
        self.symbol = kwargs.get('symbol')
        self.strategy_id = kwargs.get('strategy_id')
        self.value = kwargs.get('value')
        self.threshold = kwargs.get('threshold')
        self.metadata = kwargs.get('metadata', {})
        
        # Alert state
        self.acknowledged = False
        self.resolved = False
        self.acknowledged_by = None
        self.acknowledged_at = None
        self.resolved_at = None
    
    def acknowledge(self, user: str = "system"):
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.now()
        
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "value": self.value,
            "threshold": self.threshold,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }


class AlertManager:
    """
    Centralized alert management system.
    
    Manages alert generation, notification, escalation,
    and resolution tracking for institutional oversight.
    """
    
    def __init__(self):
        """Initialize alert manager."""
        # AI-AGENT-REF: Centralized alert management
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        self.notification_channels = {}
        
        # Alert configuration
        self.max_alerts = 1000
        self.alert_retention_hours = 24
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background cleanup
        self._cleanup_thread = None
        self._cleanup_running = False
        
        logger.info("AlertManager initialized")
    
    def create_alert(self, alert_type: AlertType, severity: AlertSeverity, 
                    message: str, **kwargs) -> Alert:
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
                
                # Maintain alert limit
                if len(self.alerts) > self.max_alerts:
                    self.alerts = self.alerts[-self.max_alerts:]
            
            # Notify handlers
            self._notify_handlers(alert)
            
            # Log based on severity
            log_message = f"Alert created: {alert.severity.value.upper()} - {message}"
            if severity == AlertSeverity.CRITICAL or severity == AlertSeverity.EMERGENCY:
                logger.error(log_message)
            elif severity == AlertSeverity.WARNING:
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            # Return dummy alert to prevent cascading errors
            return Alert(AlertType.SYSTEM, AlertSeverity.CRITICAL, f"Alert creation failed: {e}")
    
    def get_active_alerts(self, severity_filter: AlertSeverity = None) -> List[Alert]:
        """Get list of active (unresolved) alerts."""
        with self._lock:
            active_alerts = [a for a in self.alerts if not a.resolved]
            
            if severity_filter:
                active_alerts = [a for a in active_alerts if a.severity == severity_filter]
            
            return active_alerts.copy()
    
    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """Get alerts by type."""
        with self._lock:
            return [a for a in self.alerts if a.alert_type == alert_type]
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        try:
            with self._lock:
                for alert in self.alerts:
                    if alert.id == alert_id:
                        alert.acknowledge(user)
                        logger.info(f"Alert {alert_id} acknowledged by {user}")
                        return True
            
            logger.warning(f"Alert {alert_id} not found for acknowledgment")
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        try:
            with self._lock:
                for alert in self.alerts:
                    if alert.id == alert_id:
                        alert.resolve()
                        logger.info(f"Alert {alert_id} resolved")
                        return True
            
            logger.warning(f"Alert {alert_id} not found for resolution")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler."""
        self.alert_handlers.append(handler)
        logger.debug(f"Alert handler added: {handler.__name__}")
    
    def _notify_handlers(self, alert: Alert):
        """Notify all registered alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.__name__}: {e}")
    
    def start_cleanup(self):
        """Start background cleanup thread."""
        if self._cleanup_running:
            return
        
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_alerts, daemon=True)
        self._cleanup_thread.start()
        logger.info("Alert cleanup thread started")
    
    def stop_cleanup(self):
        """Stop background cleanup thread."""
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        logger.info("Alert cleanup thread stopped")
    
    def _cleanup_old_alerts(self):
        """Background cleanup of old alerts."""
        while self._cleanup_running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
                
                with self._lock:
                    # Keep only recent alerts
                    self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
                
                # Sleep for an hour
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in alert cleanup: {e}")
                time.sleep(300)  # Sleep 5 minutes on error


class RiskAlertEngine:
    """
    Specialized risk alert engine.
    
    Monitors risk metrics and generates alerts when
    thresholds are breached or risk limits are exceeded.
    """
    
    def __init__(self, alert_manager: AlertManager):
        """Initialize risk alert engine."""
        # AI-AGENT-REF: Risk alert monitoring engine
        self.alert_manager = alert_manager
        self.thresholds = PERFORMANCE_THRESHOLDS
        self.risk_params = RISK_PARAMETERS
        
        # Alert state tracking
        self.last_alert_times = {}
        self.alert_cooldown = 300  # 5 minutes between similar alerts
        
        logger.info("RiskAlertEngine initialized")
    
    def check_portfolio_risk(self, portfolio_metrics: Dict):
        """
        Check portfolio metrics against risk thresholds.
        
        Args:
            portfolio_metrics: Dictionary of portfolio risk metrics
        """
        try:
            current_time = datetime.now()
            
            # Check drawdown
            max_drawdown = portfolio_metrics.get('max_drawdown', 0)
            if max_drawdown > self.thresholds['MAX_DRAWDOWN']:
                self._create_risk_alert(
                    "max_drawdown",
                    AlertSeverity.CRITICAL,
                    f"Maximum drawdown {max_drawdown:.2%} exceeds threshold {self.thresholds['MAX_DRAWDOWN']:.2%}",
                    value=max_drawdown,
                    threshold=self.thresholds['MAX_DRAWDOWN']
                )
            
            # Check Sharpe ratio
            sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio < self.thresholds['MIN_SHARPE_RATIO']:
                self._create_risk_alert(
                    "sharpe_ratio",
                    AlertSeverity.WARNING,
                    f"Sharpe ratio {sharpe_ratio:.2f} below minimum {self.thresholds['MIN_SHARPE_RATIO']:.2f}",
                    value=sharpe_ratio,
                    threshold=self.thresholds['MIN_SHARPE_RATIO']
                )
            
            # Check VaR
            var_95 = portfolio_metrics.get('var_95', 0)
            if var_95 > self.thresholds['MAX_VAR_95']:
                self._create_risk_alert(
                    "var_95",
                    AlertSeverity.WARNING,
                    f"95% VaR {var_95:.2%} exceeds maximum {self.thresholds['MAX_VAR_95']:.2%}",
                    value=var_95,
                    threshold=self.thresholds['MAX_VAR_95']
                )
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
    
    def check_position_risk(self, symbol: str, position_metrics: Dict):
        """
        Check individual position risk.
        
        Args:
            symbol: Trading symbol
            position_metrics: Position-specific metrics
        """
        try:
            # Check position concentration
            position_pct = position_metrics.get('position_percentage', 0)
            if position_pct > self.risk_params['MAX_POSITION_SIZE']:
                self._create_risk_alert(
                    f"position_concentration_{symbol}",
                    AlertSeverity.WARNING,
                    f"Position in {symbol} ({position_pct:.2%}) exceeds maximum {self.risk_params['MAX_POSITION_SIZE']:.2%}",
                    symbol=symbol,
                    value=position_pct,
                    threshold=self.risk_params['MAX_POSITION_SIZE']
                )
            
            # Check position loss
            unrealized_pnl_pct = position_metrics.get('unrealized_pnl_pct', 0)
            if unrealized_pnl_pct < -0.10:  # 10% loss threshold
                self._create_risk_alert(
                    f"position_loss_{symbol}",
                    AlertSeverity.WARNING,
                    f"Position in {symbol} showing {unrealized_pnl_pct:.2%} unrealized loss",
                    symbol=symbol,
                    value=unrealized_pnl_pct,
                    threshold=-0.10
                )
            
        except Exception as e:
            logger.error(f"Error checking position risk for {symbol}: {e}")
    
    def check_execution_risk(self, execution_metrics: Dict):
        """
        Check execution-related risk metrics.
        
        Args:
            execution_metrics: Execution performance metrics
        """
        try:
            # Check fill rate
            fill_rate = execution_metrics.get('fill_rate', 1.0)
            if fill_rate < 0.8:  # 80% minimum fill rate
                self._create_risk_alert(
                    "low_fill_rate",
                    AlertSeverity.WARNING,
                    f"Order fill rate {fill_rate:.2%} below 80% threshold",
                    value=fill_rate,
                    threshold=0.8
                )
            
            # Check average slippage
            avg_slippage_bps = execution_metrics.get('average_slippage_bps', 0)
            max_slippage_bps = self.risk_params.get('MAX_SLIPPAGE_BPS', 20)
            if avg_slippage_bps > max_slippage_bps:
                self._create_risk_alert(
                    "high_slippage",
                    AlertSeverity.WARNING,
                    f"Average slippage {avg_slippage_bps:.1f} bps exceeds maximum {max_slippage_bps} bps",
                    value=avg_slippage_bps,
                    threshold=max_slippage_bps
                )
            
        except Exception as e:
            logger.error(f"Error checking execution risk: {e}")
    
    def _create_risk_alert(self, alert_key: str, severity: AlertSeverity, 
                          message: str, **kwargs):
        """Create risk alert with cooldown protection."""
        try:
            current_time = datetime.now()
            
            # Check cooldown
            last_alert_time = self.last_alert_times.get(alert_key)
            if (last_alert_time and 
                (current_time - last_alert_time).total_seconds() < self.alert_cooldown):
                return  # Skip alert due to cooldown
            
            # Create alert
            alert = self.alert_manager.create_alert(
                AlertType.RISK_LIMIT,
                severity,
                message,
                **kwargs
            )
            
            # Update last alert time
            self.last_alert_times[alert_key] = current_time
            
        except Exception as e:
            logger.error(f"Error creating risk alert: {e}")