"""Institutional-grade structured logging system."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4

import structlog
from structlog.stdlib import LoggerFactory
from structlog.types import Processor

from ..core.enums import RiskLevel


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, include_timestamp: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add extra fields
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = str(record.correlation_id)
        
        if hasattr(record, 'symbol'):
            log_entry["symbol"] = record.symbol
        
        if hasattr(record, 'strategy_id'):
            log_entry["strategy_id"] = record.strategy_id
        
        if hasattr(record, 'trade_id'):
            log_entry["trade_id"] = str(record.trade_id)
        
        if hasattr(record, 'execution_time_ms'):
            log_entry["execution_time_ms"] = record.execution_time_ms
        
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add any custom fields stored in 'extra'
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'exc_info', 'exc_text',
                'stack_info', 'correlation_id', 'symbol', 'strategy_id',
                'trade_id', 'execution_time_ms', 'user_id'
            }:
                if not key.startswith('_'):
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=self._json_default)
    
    def _json_default(self, obj: Any) -> str:
        """Handle non-serializable objects."""
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return repr(obj)


class TradingLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with trading-specific context."""
    
    def __init__(
        self,
        logger: logging.Logger,
        correlation_id: Optional[UUID] = None,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None
    ):
        self.correlation_id = correlation_id or uuid4()
        extra = {
            'correlation_id': self.correlation_id,
            'strategy_id': strategy_id,
            'symbol': symbol
        }
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Process log message with extra context."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Merge adapter extra with call-specific extra
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs
    
    def trade_executed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        trade_id: UUID,
        execution_time_ms: float,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> None:
        """Log trade execution with structured data."""
        self.info(
            f"Trade executed: {side} {quantity} {symbol} @ ${price:.4f}",
            extra={
                'event_type': 'trade_execution',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'trade_id': trade_id,
                'execution_time_ms': execution_time_ms,
                'commission': commission,
                'slippage': slippage
            }
        )
    
    def signal_generated(
        self,
        symbol: str,
        signal_type: str,
        side: str,
        strength: float,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log signal generation with structured data."""
        self.info(
            f"Signal generated: {signal_type} {side} {symbol} "
            f"(strength={strength:.2f}, confidence={confidence:.2f})",
            extra={
                'event_type': 'signal_generation',
                'symbol': symbol,
                'signal_type': signal_type,
                'side': side,
                'strength': strength,
                'confidence': confidence,
                'metadata': metadata or {}
            }
        )
    
    def risk_alert(
        self,
        alert_type: str,
        severity: RiskLevel,
        message: str,
        current_value: float,
        threshold_value: float,
        symbol: Optional[str] = None
    ) -> None:
        """Log risk alerts with structured data."""
        level_map = {
            RiskLevel.LOW: logging.INFO,
            RiskLevel.MEDIUM: logging.WARNING,
            RiskLevel.HIGH: logging.ERROR,
            RiskLevel.CRITICAL: logging.CRITICAL
        }
        
        log_level = level_map.get(severity, logging.WARNING)
        
        self.log(
            log_level,
            f"Risk alert: {alert_type} - {message}",
            extra={
                'event_type': 'risk_alert',
                'alert_type': alert_type,
                'severity': severity.name,
                'current_value': current_value,
                'threshold_value': threshold_value,
                'symbol': symbol
            }
        )
    
    def performance_metrics(
        self,
        strategy_id: str,
        total_pnl: float,
        daily_pnl: float,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        win_rate: Optional[float] = None
    ) -> None:
        """Log performance metrics with structured data."""
        self.info(
            f"Performance update for {strategy_id}: "
            f"PnL=${total_pnl:.2f}, Daily=${daily_pnl:.2f}",
            extra={
                'event_type': 'performance_metrics',
                'strategy_id': strategy_id,
                'total_pnl': total_pnl,
                'daily_pnl': daily_pnl,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
        )


class AuditLogger:
    """Separate audit trail logger for regulatory compliance."""
    
    def __init__(self, audit_log_path: Union[str, Path]):
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dedicated audit logger
        self.logger = logging.getLogger('audit_trail')
        self.logger.setLevel(logging.INFO)
        
        # Prevent audit logs from going to other handlers
        self.logger.propagate = False
        
        # Create file handler with rotation
        handler = logging.handlers.RotatingFileHandler(
            self.audit_log_path,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10,
            encoding='utf-8'
        )
        
        # Use JSON formatter for audit logs
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
    
    def log_trade_decision(
        self,
        correlation_id: UUID,
        strategy_id: str,
        symbol: str,
        decision: str,
        reasoning: str,
        market_data: Dict[str, Any],
        position_size: float,
        risk_metrics: Dict[str, Any]
    ) -> None:
        """Log trade decision for audit trail."""
        self.logger.info(
            f"Trade decision: {decision} for {symbol}",
            extra={
                'audit_type': 'trade_decision',
                'correlation_id': correlation_id,
                'strategy_id': strategy_id,
                'symbol': symbol,
                'decision': decision,
                'reasoning': reasoning,
                'market_data': market_data,
                'position_size': position_size,
                'risk_metrics': risk_metrics,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )
    
    def log_risk_override(
        self,
        correlation_id: UUID,
        user_id: str,
        override_type: str,
        original_limit: float,
        new_limit: float,
        justification: str
    ) -> None:
        """Log risk limit overrides for compliance."""
        self.logger.warning(
            f"Risk override: {override_type} changed from {original_limit} to {new_limit}",
            extra={
                'audit_type': 'risk_override',
                'correlation_id': correlation_id,
                'user_id': user_id,
                'override_type': override_type,
                'original_limit': original_limit,
                'new_limit': new_limit,
                'justification': justification,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log system events for audit trail."""
        self.logger.info(
            f"System event: {event_type} - {description}",
            extra={
                'audit_type': 'system_event',
                'event_type': event_type,
                'description': description,
                'metadata': metadata or {},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )


def setup_institutional_logging(
    log_level: str = "INFO",
    log_dir: Union[str, Path] = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_audit: bool = True,
    max_file_size: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 5
) -> tuple[logging.Logger, Optional[AuditLogger]]:
    """Setup institutional-grade logging system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_audit: Enable audit trail logging
        max_file_size: Maximum size per log file
        backup_count: Number of backup files to keep
        
    Returns:
        Tuple of (main logger, audit logger)
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "trading_system.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
        
        # Separate error log
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        logger.addHandler(error_handler)
    
    # Setup audit logger
    audit_logger = None
    if enable_audit:
        audit_logger = AuditLogger(log_dir / "audit_trail.log")
    
    logger.info(
        "Institutional logging system initialized",
        extra={
            'log_level': log_level,
            'log_dir': str(log_dir),
            'console_enabled': enable_console,
            'file_enabled': enable_file,
            'audit_enabled': enable_audit
        }
    )
    
    return logger, audit_logger


def get_trading_logger(
    name: str,
    correlation_id: Optional[UUID] = None,
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None
) -> TradingLoggerAdapter:
    """Get a trading-specific logger adapter.
    
    Args:
        name: Logger name
        correlation_id: Request correlation ID
        strategy_id: Strategy identifier
        symbol: Trading symbol
        
    Returns:
        Trading logger adapter
    """
    base_logger = logging.getLogger(name)
    return TradingLoggerAdapter(
        base_logger,
        correlation_id=correlation_id,
        strategy_id=strategy_id,
        symbol=symbol
    )