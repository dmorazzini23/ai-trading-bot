"""Swing Trading Mode - PDT-Safe Trading Strategy."""

import logging
import os
from datetime import datetime, time, timezone
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def can_exit_today(position: Mapping[str, Any], now_utc: datetime) -> bool:
    """Return ``True`` when a position may exit on ``now_utc``."""

    allow_env = os.getenv("AI_TRADING_SWING_ALLOW_SAME_DAY_EXIT", "").strip()
    if allow_env == "1":
        return True

    opened_at = position.get("opened_at") if isinstance(position, Mapping) else None
    if opened_at is None:
        return True

    opened_dt: datetime | None
    if isinstance(opened_at, datetime):
        opened_dt = opened_at
    elif isinstance(opened_at, (int, float)):
        opened_dt = datetime.fromtimestamp(float(opened_at), tz=timezone.utc)
    elif isinstance(opened_at, str):
        try:
            opened_dt = datetime.fromisoformat(opened_at)
        except ValueError:
            return True
    else:
        return True

    if opened_dt.tzinfo is None:
        opened_dt = opened_dt.replace(tzinfo=timezone.utc)

    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    opened_date = opened_dt.astimezone(timezone.utc).date()
    now_date = now_utc.astimezone(timezone.utc).date()
    return opened_date < now_date


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
        now_utc = datetime.now(timezone.utc)
        env_allow = os.getenv("AI_TRADING_SWING_ALLOW_SAME_DAY_EXIT", "").strip() == "1"
        if not can_exit_today({"opened_at": entry_time}, now_utc):
            return (False, "same_day_trade_blocked")
        if env_allow:
            return (True, "same_day_exit_allowed")

        now = now_utc.astimezone(ZoneInfo("America/New_York"))

        # Check if we're on a different calendar day
        if now.date() > entry_time.date():
            return (True, "different_day")

        # Same day - check if market has closed since entry
        market_close = time(16, 0)  # 4:00 PM ET

        if entry_time.time() < market_close and now.time() >= market_close:
            # Entered before close, now after close - safe to exit
            return (True, "after_market_close")

        # Default allowance when the position predates current day
        return (True, "same_day_after_close")
    
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

