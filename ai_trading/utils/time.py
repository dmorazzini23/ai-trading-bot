from __future__ import annotations
from datetime import UTC, datetime, timedelta
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import pandas as pd

def utcnow() -> datetime:
    """Repository-standard UTC now (timezone-aware)."""
    return datetime.now(UTC)

now_utc = utcnow

@dataclass
class SessionWindow:
    open: pd.Timestamp
    close: pd.Timestamp

def last_market_session(now: pd.Timestamp) -> SessionWindow | None:
    """Return previous market session window for NYSE.

    Returns ``None`` if pandas or market calendars are unavailable.
    """
    pd = load_pandas()
    if pd is None:
        return None
    try:
        from ai_trading.market.calendars import get_calendar_registry
    except Exception:  # ImportError: calendars package missing
        return None
    cal = get_calendar_registry()
    current = now.tz_convert('UTC').date()
    for _ in range(10):
        start, end = cal.get_session_bounds('SPY', current)
        if start and end and (end <= now.to_pydatetime()):
            return SessionWindow(pd.Timestamp(start).tz_convert('UTC'), pd.Timestamp(end).tz_convert('UTC'))
        current -= timedelta(days=1)
    return None

__all__ = ['utcnow', 'now_utc', 'SessionWindow', 'last_market_session']
