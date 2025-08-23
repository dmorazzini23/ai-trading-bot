"""
Enhanced monitoring for trade execution issues.
This module provides monitoring capabilities to track when trades are executed vs skipped.
"""
import logging
from datetime import UTC, datetime
from typing import Any
logger = logging.getLogger(__name__)

class TradeExecutionMonitor:
    """Monitor and track trade execution statistics."""

    def __init__(self):
        self.execution_stats = {'buy_orders_attempted': 0, 'buy_orders_executed': 0, 'buy_orders_skipped_insufficient_funds': 0, 'buy_orders_skipped_other': 0, 'sell_orders_attempted': 0, 'sell_orders_executed': 0, 'sell_orders_skipped_no_position': 0, 'sell_orders_skipped_other': 0, 'api_errors': 0, 'last_reset': datetime.now(UTC)}

    def log_buy_attempt(self, symbol: str, qty: int, reason: str=None):
        """Log a buy order attempt."""
        self.execution_stats['buy_orders_attempted'] += 1
        logger.info('BUY_ORDER_ATTEMPT', extra={'symbol': symbol, 'qty': qty, 'reason': reason or 'signal_generated'})

    def log_buy_executed(self, symbol: str, qty: int):
        """Log a successful buy order execution."""
        self.execution_stats['buy_orders_executed'] += 1
        logger.info('BUY_ORDER_EXECUTED', extra={'symbol': symbol, 'qty': qty})

    def log_buy_skipped(self, symbol: str, qty: int, reason: str):
        """Log a skipped buy order."""
        if 'insufficient' in reason.lower():
            self.execution_stats['buy_orders_skipped_insufficient_funds'] += 1
        else:
            self.execution_stats['buy_orders_skipped_other'] += 1
        logger.warning('BUY_ORDER_SKIPPED', extra={'symbol': symbol, 'qty': qty, 'reason': reason})

    def log_sell_attempt(self, symbol: str, qty: int):
        """Log a sell order attempt."""
        self.execution_stats['sell_orders_attempted'] += 1
        logger.info('SELL_ORDER_ATTEMPT', extra={'symbol': symbol, 'qty': qty})

    def log_sell_executed(self, symbol: str, qty: int):
        """Log a successful sell order execution."""
        self.execution_stats['sell_orders_executed'] += 1
        logger.info('SELL_ORDER_EXECUTED', extra={'symbol': symbol, 'qty': qty})

    def log_sell_skipped(self, symbol: str, qty: int, reason: str):
        """Log a skipped sell order."""
        if 'no position' in reason.lower():
            self.execution_stats['sell_orders_skipped_no_position'] += 1
        else:
            self.execution_stats['sell_orders_skipped_other'] += 1
        logger.warning('SELL_ORDER_SKIPPED', extra={'symbol': symbol, 'qty': qty, 'reason': reason})

    def log_api_error(self, error_type: str, details: str):
        """Log API-related errors."""
        self.execution_stats['api_errors'] += 1
        logger.error('API_ERROR', extra={'error_type': error_type, 'details': details})

    def get_execution_summary(self) -> dict[str, Any]:
        """Get a summary of execution statistics."""
        total_buy_attempts = self.execution_stats['buy_orders_attempted']
        total_sell_attempts = self.execution_stats['sell_orders_attempted']
        summary = {'period_start': self.execution_stats['last_reset'].isoformat(), 'current_time': datetime.now(UTC).isoformat(), 'buy_execution_rate': self.execution_stats['buy_orders_executed'] / total_buy_attempts * 100 if total_buy_attempts > 0 else 0, 'sell_execution_rate': self.execution_stats['sell_orders_executed'] / total_sell_attempts * 100 if total_sell_attempts > 0 else 0, 'stats': self.execution_stats.copy()}
        return summary

    def log_periodic_summary(self):
        """Log a periodic summary of execution statistics."""
        summary = self.get_execution_summary()
        logger.info('TRADE_EXECUTION_SUMMARY', extra=summary)
        if summary['buy_execution_rate'] < 50 and self.execution_stats['buy_orders_attempted'] >= 5:
            logger.critical('LOW_BUY_EXECUTION_RATE', extra={'rate': summary['buy_execution_rate'], 'attempted': self.execution_stats['buy_orders_attempted'], 'executed': self.execution_stats['buy_orders_executed']})

    def reset_stats(self):
        """Reset execution statistics."""
        self.execution_stats = {k: 0 if isinstance(v, int) else datetime.now(UTC) for k, v in self.execution_stats.items()}
        self.execution_stats['last_reset'] = datetime.now(UTC)
trade_monitor = TradeExecutionMonitor()