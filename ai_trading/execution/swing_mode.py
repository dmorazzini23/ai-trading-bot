"""Swing Trading Mode - PDT-Safe Trading Strategy.

This module implements swing trading logic that avoids day trades
by holding positions overnight, ensuring PDT compliance.
"""

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class SwingTradingMode:
    """
    Swing trading mode that prevents day trades.
    
    Rules:
    - Only enter new positions
    - Never exit positions on the same day they were entered
    - Track entry times to prevent same-day exits
    - Allow exits only after market close of entry day
    """
    
    def __init__(self):
        self.position_entry_times = {}  # symbol -> entry datetime
        self.enabled = False
    
    def enable(self):
        """Enable swing trading mode."""
        self.enabled = True
        logger.info("SWING_MODE_ENABLED | PDT-safe trading activated")
    
    def disable(self):
        """Disable swing trading mode."""
        self.enabled = False
        logger.info("SWING_MODE_DISABLED | Normal trading resumed")
    
    def record_entry(self, symbol: str, entry_time: Optional[datetime] = None):
        """Record when a position was entered."""
        
        if entry_time is None:
            entry_time = datetime.now(ZoneInfo("America/New_York"))
        
        self.position_entry_times[symbol] = entry_time
        logger.info(
            "SWING_ENTRY_RECORDED",
            extra={
                "symbol": symbol,
                "entry_time": entry_time.isoformat(),
                "entry_date": entry_time.date().isoformat()
            }
        )
    
    def can_exit_position(self, symbol: str) -> tuple[bool, str]:
        """
        Check if a position can be exited without creating a day trade.
        
        Returns:
            (can_exit, reason) tuple
        """
        
        if not self.enabled:
            return (True, "swing_mode_disabled")
        
        if symbol not in self.position_entry_times:
            # No entry time recorded, assume it's old position - safe to exit
            return (True, "no_entry_time_recorded")
        
        entry_time = self.position_entry_times[symbol]
        now = datetime.now(ZoneInfo("America/New_York"))
        
        # Check if we're on a different calendar day
        if now.date() > entry_time.date():
            return (True, "different_day")
        
        # Same trading day - must hold overnight to avoid day trade classification
        return (False, "must_hold_overnight")
    
    def clear_entry(self, symbol: str):
        """Clear entry time after position is closed."""
        
        if symbol in self.position_entry_times:
            del self.position_entry_times[symbol]
            logger.info("SWING_ENTRY_CLEARED", extra={"symbol": symbol})
    
    def should_allow_new_position(self, symbol: str) -> tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        In swing mode, we only allow new positions if we don't already have one.
        """
        
        if not self.enabled:
            return (True, "swing_mode_disabled")
        
        if symbol in self.position_entry_times:
            return (False, "already_have_position")
        
        return (True, "can_open_new_position")
    
    def get_status(self) -> dict:
        """Get current swing mode status."""
        
        return {
            "enabled": self.enabled,
            "active_positions": len(self.position_entry_times),
            "symbols": list(self.position_entry_times.keys()),
            "entry_times": {
                sym: dt.isoformat() 
                for sym, dt in self.position_entry_times.items()
            }
        }


# Global swing mode instance
_swing_mode = SwingTradingMode()


def get_swing_mode() -> SwingTradingMode:
    """Get the global swing trading mode instance."""
    return _swing_mode


def enable_swing_mode():
    """Enable swing trading mode globally."""
    _swing_mode.enable()


def disable_swing_mode():
    """Disable swing trading mode globally."""
    _swing_mode.disable()

