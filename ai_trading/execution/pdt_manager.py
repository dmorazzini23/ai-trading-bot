"""Pattern Day Trader (PDT) Management Module.

This module provides intelligent PDT-aware trading logic to prevent violations
while maximizing trading opportunities within regulatory constraints.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PDTStatus:
    """PDT status information."""
    
    is_pattern_day_trader: bool
    daytrade_count: int
    daytrade_limit: int
    can_daytrade: bool
    remaining_daytrades: int
    strategy_recommendation: str


class PDTManager:
    """Manages PDT compliance and provides trading strategy recommendations."""
    
    def __init__(self):
        self.last_check_time: Optional[datetime] = None
        self.cached_status: Optional[PDTStatus] = None
        self.cache_ttl_seconds = 60  # Cache for 1 minute
    
    def get_pdt_status(self, account: Any) -> PDTStatus:
        """Get current PDT status from account."""
        
        # Check cache
        if self._is_cache_valid():
            return self.cached_status
        
        # Extract PDT info from account
        is_pdt = self._extract_bool(account, "pattern_day_trader", "is_pattern_day_trader", "pdt")
        
        daytrade_limit = self._extract_int(
            account, 
            "daytrade_limit", 
            "day_trade_limit", 
            "pattern_day_trade_limit",
            default=3
        )
        
        daytrade_count = self._extract_int(
            account,
            "daytrade_count",
            "day_trade_count", 
            "pattern_day_trades",
            "pattern_day_trades_count",
            default=0
        )
        
        # Calculate status
        can_daytrade = daytrade_count < daytrade_limit
        remaining = max(0, daytrade_limit - daytrade_count)
        
        # Determine strategy recommendation
        if not is_pdt:
            strategy = "normal"
        elif daytrade_count >= daytrade_limit:
            strategy = "swing_only"  # Must hold overnight
        elif daytrade_count == daytrade_limit - 1:
            strategy = "conservative"  # One trade left, be careful
        else:
            strategy = "normal"
        
        status = PDTStatus(
            is_pattern_day_trader=is_pdt,
            daytrade_count=daytrade_count,
            daytrade_limit=daytrade_limit,
            can_daytrade=can_daytrade,
            remaining_daytrades=remaining,
            strategy_recommendation=strategy
        )
        
        # Update cache
        self.cached_status = status
        self.last_check_time = datetime.now()
        
        return status
    
    def should_allow_order(
        self, 
        account: Any, 
        symbol: str,
        side: str,
        current_position: int = 0,
        force_swing_mode: bool = False
    ) -> tuple[bool, str, dict]:
        """
        Determine if an order should be allowed based on PDT rules.
        
        Returns:
            (allow, reason, context) tuple
        """
        
        status = self.get_pdt_status(account)
        
        # Build context for logging
        context = {
            "symbol": symbol,
            "side": side,
            "pattern_day_trader": status.is_pattern_day_trader,
            "daytrade_count": status.daytrade_count,
            "daytrade_limit": status.daytrade_limit,
            "remaining_daytrades": status.remaining_daytrades,
            "strategy": status.strategy_recommendation
        }
        
        # Not a PDT account - allow all trades
        if not status.is_pattern_day_trader:
            return (True, "not_pdt", context)
        
        # Closing existing position - always allowed (doesn't count as day trade if held overnight)
        is_closing = (current_position > 0 and side.lower() in ["sell", "close"]) or \
                     (current_position < 0 and side.lower() in ["buy", "cover"])
        
        if is_closing:
            return (True, "closing_position", context)
        
        # Force swing mode (hold overnight) - only allow if no position exists
        if force_swing_mode:
            if current_position == 0:
                return (True, "swing_mode_entry", context)
            else:
                return (False, "swing_mode_has_position", context)
        
        # Check if we've hit the limit
        if not status.can_daytrade:
            return (False, "pdt_limit_reached", context)
        
        # Conservative mode - warn but allow
        if status.strategy_recommendation == "conservative":
            logger.warning(
                "PDT_CONSERVATIVE_MODE",
                extra={
                    "message": f"Last day trade available ({status.remaining_daytrades} remaining)",
                    **context
                }
            )
            return (True, "pdt_conservative", context)
        
        # Normal trading allowed
        return (True, "pdt_ok", context)
    
    def get_recommended_strategy(self, account: Any) -> str:
        """Get recommended trading strategy based on PDT status."""
        
        status = self.get_pdt_status(account)
        
        if not status.is_pattern_day_trader:
            return "day_trading"  # Can day trade freely
        
        if status.daytrade_count >= status.daytrade_limit:
            return "swing_trading"  # Must hold overnight
        
        if status.remaining_daytrades <= 1:
            return "conservative_swing"  # Prefer swing trades, day trade only if critical
        
        return "mixed"  # Can do both
    
    def _is_cache_valid(self) -> bool:
        """Check if cached status is still valid."""
        
        if self.cached_status is None or self.last_check_time is None:
            return False
        
        age = (datetime.now() - self.last_check_time).total_seconds()
        return age < self.cache_ttl_seconds
    
    def _extract_bool(self, obj: Any, *attrs: str) -> bool:
        """Extract boolean value from object attributes."""
        
        for attr in attrs:
            try:
                val = getattr(obj, attr, None)
                if val is not None:
                    if isinstance(val, bool):
                        return val
                    return str(val).lower() in ("true", "1", "yes")
            except (AttributeError, TypeError, ValueError) as exc:
                logger.debug(
                    "PDT_ATTR_BOOL_EXTRACT_FAILED",
                    extra={"attribute": attr, "error": str(exc)},
                )
                continue
        return False

    def _extract_int(self, obj: Any, *attrs: str, default: int = 0) -> int:
        """Extract integer value from object attributes."""

        for attr in attrs:
            try:
                val = getattr(obj, attr, None)
                if val is not None:
                    return int(val)
            except (AttributeError, TypeError, ValueError) as exc:
                logger.debug(
                    "PDT_ATTR_INT_EXTRACT_FAILED",
                    extra={"attribute": attr, "error": str(exc)},
                )
                continue
        return default

