"""Custom exceptions for institutional trading system."""

from typing import Any, Dict, Optional
from uuid import UUID


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[UUID] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.correlation_id = correlation_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
        }


class RiskLimitExceededError(TradingSystemError):
    """Raised when risk limits are exceeded."""
    
    def __init__(
        self, 
        limit_type: str,
        current_value: float,
        limit_value: float,
        symbol: Optional[str] = None,
        **kwargs
    ):
        message = f"Risk limit exceeded: {limit_type} ({current_value:.4f} > {limit_value:.4f})"
        if symbol:
            message += f" for {symbol}"
        
        details = kwargs.get('details', {})
        details.update({
            "limit_type": limit_type,
            "current_value": current_value,
            "limit_value": limit_value,
            "symbol": symbol,
        })
        
        super().__init__(message, error_code="RISK_LIMIT_EXCEEDED", details=details, **kwargs)


class DataValidationError(TradingSystemError):
    """Raised when data validation fails."""
    
    def __init__(
        self, 
        field: str,
        value: Any,
        expected: str,
        **kwargs
    ):
        message = f"Data validation failed: {field}={value} (expected: {expected})"
        
        details = kwargs.get('details', {})
        details.update({
            "field": field,
            "value": str(value),
            "expected": expected,
        })
        
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", details=details, **kwargs)


class ExecutionError(TradingSystemError):
    """Raised when order execution fails."""
    
    def __init__(
        self, 
        order_id: UUID,
        symbol: str,
        reason: str,
        **kwargs
    ):
        message = f"Order execution failed: {order_id} for {symbol} - {reason}"
        
        details = kwargs.get('details', {})
        details.update({
            "order_id": str(order_id),
            "symbol": symbol,
            "reason": reason,
        })
        
        super().__init__(message, error_code="EXECUTION_ERROR", details=details, **kwargs)


class InsufficientLiquidityError(ExecutionError):
    """Raised when insufficient liquidity for execution."""
    
    def __init__(
        self, 
        symbol: str,
        requested_quantity: float,
        available_quantity: float,
        **kwargs
    ):
        reason = f"Insufficient liquidity: requested {requested_quantity}, available {available_quantity}"
        
        details = kwargs.get('details', {})
        details.update({
            "requested_quantity": requested_quantity,
            "available_quantity": available_quantity,
        })
        
        super().__init__(
            order_id=kwargs.get('order_id'), 
            symbol=symbol, 
            reason=reason,
            details=details,
            **kwargs
        )


class PositionNotFoundError(TradingSystemError):
    """Raised when position cannot be found."""
    
    def __init__(self, symbol: str, **kwargs):
        message = f"Position not found for symbol: {symbol}"
        
        details = kwargs.get('details', {})
        details.update({"symbol": symbol})
        
        super().__init__(message, error_code="POSITION_NOT_FOUND", details=details, **kwargs)


class StrategyError(TradingSystemError):
    """Raised when strategy execution fails."""
    
    def __init__(
        self, 
        strategy_id: str,
        strategy_type: str,
        reason: str,
        **kwargs
    ):
        message = f"Strategy error in {strategy_id} ({strategy_type}): {reason}"
        
        details = kwargs.get('details', {})
        details.update({
            "strategy_id": strategy_id,
            "strategy_type": strategy_type,
            "reason": reason,
        })
        
        super().__init__(message, error_code="STRATEGY_ERROR", details=details, **kwargs)


class MarketDataError(TradingSystemError):
    """Raised when market data issues occur."""
    
    def __init__(
        self, 
        symbol: str,
        data_type: str,
        reason: str,
        **kwargs
    ):
        message = f"Market data error for {symbol} ({data_type}): {reason}"
        
        details = kwargs.get('details', {})
        details.update({
            "symbol": symbol,
            "data_type": data_type,
            "reason": reason,
        })
        
        super().__init__(message, error_code="MARKET_DATA_ERROR", details=details, **kwargs)


class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self, 
        config_key: str,
        reason: str,
        **kwargs
    ):
        message = f"Configuration error in {config_key}: {reason}"
        
        details = kwargs.get('details', {})
        details.update({
            "config_key": config_key,
            "reason": reason,
        })
        
        super().__init__(message, error_code="CONFIGURATION_ERROR", details=details, **kwargs)


class DatabaseError(TradingSystemError):
    """Raised when database operations fail."""
    
    def __init__(
        self, 
        operation: str,
        table: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ):
        message = f"Database error during {operation}"
        if table:
            message += f" on table {table}"
        if reason:
            message += f": {reason}"
        
        details = kwargs.get('details', {})
        details.update({
            "operation": operation,
            "table": table,
            "reason": reason,
        })
        
        super().__init__(message, error_code="DATABASE_ERROR", details=details, **kwargs)


class AuthenticationError(TradingSystemError):
    """Raised when authentication fails."""
    
    def __init__(self, service: str, reason: str, **kwargs):
        message = f"Authentication failed for {service}: {reason}"
        
        details = kwargs.get('details', {})
        details.update({
            "service": service,
            "reason": reason,
        })
        
        super().__init__(message, error_code="AUTHENTICATION_ERROR", details=details, **kwargs)


class RateLimitError(TradingSystemError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(
        self, 
        service: str,
        limit: int,
        reset_time: Optional[int] = None,
        **kwargs
    ):
        message = f"Rate limit exceeded for {service} (limit: {limit}/min)"
        if reset_time:
            message += f", resets in {reset_time}s"
        
        details = kwargs.get('details', {})
        details.update({
            "service": service,
            "limit": limit,
            "reset_time": reset_time,
        })
        
        super().__init__(message, error_code="RATE_LIMIT_ERROR", details=details, **kwargs)