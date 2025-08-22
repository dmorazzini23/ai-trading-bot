"""
Enhanced error handling for trading bot resilience.
Provides robust error handling throughout the trading system.
"""

import logging
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

class TradingBotError(Exception):
    """Base exception for trading bot errors."""

class APIError(TradingBotError):
    """API-related errors."""

class DataError(TradingBotError):
    """Data quality or availability errors."""

class ExecutionError(TradingBotError):
    """Trade execution errors."""

def with_error_handling(
    error_type: type[Exception] = Exception,
    default_return: Any = None,
    log_level: str = "error",
    reraise: bool = False
):
    """
    Decorator for robust error handling in trading operations.
    
    Args:
        error_type: Type of exception to catch
        default_return: Value to return on error
        log_level: Logging level for errors
        reraise: Whether to re-raise the exception after logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                error_details = {
                    "function": func.__name__,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "traceback": traceback.format_exc()
                }

                if log_level == "error":
                    logger.error(error_msg, extra=error_details)
                elif log_level == "warning":
                    logger.warning(error_msg, extra=error_details)
                elif log_level == "critical":
                    logger.critical(error_msg, extra=error_details)

                if reraise:
                    raise

                return default_return
        return wrapper
    return decorator

def safe_api_call(func: Callable, retries: int = 3, delay: float = 1.0) -> Any:
    """
    Safely execute API calls with retry logic.
    
    Args:
        func: Function to execute
        retries: Number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Function result or None on failure
    """
    import time

    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt < retries:
                logger.warning(f"API call failed (attempt {attempt + 1}/{retries + 1}): {e}")
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"API call failed after {retries + 1} attempts: {e}")

    return None

def validate_trade_data(symbol: str, qty: int, side: str, price: float | None = None) -> bool:
    """
    Validate trade data before execution.
    
    Args:
        symbol: Trading symbol
        qty: Quantity to trade
        side: 'buy' or 'sell'
        price: Optional price for validation
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        # Basic validation
        if not symbol or not isinstance(symbol, str):
            logger.error("Invalid symbol: %s", symbol)
            return False

        if not isinstance(qty, int) or qty <= 0:
            logger.error("Invalid quantity: %s", qty)
            return False

        if side not in ['buy', 'sell']:
            logger.error("Invalid side: %s", side)
            return False

        if price is not None and (not isinstance(price, int | float) or price <= 0):
            logger.error("Invalid price: %s", price)
            return False

        return True

    except Exception as e:
        logger.error("Error validating trade data: %s", e)
        return False

def graceful_shutdown(reason: str = "Unknown"):
    """
    Perform graceful shutdown of trading operations.
    
    Args:
        reason: Reason for shutdown
    """
    logger.critical("TRADING_BOT_SHUTDOWN", extra={
        "reason": reason,
        "timestamp": datetime.now(UTC).isoformat()
    })

    # Log final statistics if monitor is available
    try:
        from trade_monitor import trade_monitor
        trade_monitor.log_periodic_summary()
    except ImportError:
        pass

# AI-AGENT-REF: Enhanced error handling decorators for critical trading functions
trade_execution_error_handler = with_error_handling(
    error_type=Exception,
    default_return=None,
    log_level="error",
    reraise=False
)

api_error_handler = with_error_handling(
    error_type=Exception,
    default_return=None,
    log_level="warning",
    reraise=False
)

critical_error_handler = with_error_handling(
    error_type=Exception,
    default_return=None,
    log_level="critical",
    reraise=True
)
