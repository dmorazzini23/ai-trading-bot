from __future__ import annotations
from datetime import UTC, datetime, timedelta, tzinfo, date as _date
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_trading.utils.lazy_imports import load_pandas

# Simple day-scoped cache for last_market_session to avoid repeated calendar work
_LAST_SESSION_CACHE: dict[str, "SessionWindow | None"] = {}

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import pandas as pd

def utcnow(tz: tzinfo | None = UTC) -> datetime:
    """Repository-standard aware now with optional timezone.

    Args:
        tz: Desired timezone. Defaults to UTC.

    Returns:
        timezone-aware ``datetime`` in the requested zone.
    """
    now = datetime.now(UTC)
    return now if tz in (UTC, None) else now.astimezone(tz)

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
    except ImportError:  # calendars package missing
        return None
    cal = get_calendar_registry()
    current: _date = now.tz_convert('UTC').date()
    key = current.isoformat()
    cached = _LAST_SESSION_CACHE.get(key)
    if cached is not None:
        return cached
    for _ in range(10):
        start, end = cal.get_session_bounds('SPY', current)
        if start and end and (end <= now.to_pydatetime()):
            win = SessionWindow(pd.Timestamp(start).tz_convert('UTC'), pd.Timestamp(end).tz_convert('UTC'))
            _LAST_SESSION_CACHE[key] = win
            return win
        current -= timedelta(days=1)
    _LAST_SESSION_CACHE[key] = None
    return None

__all__ = ['utcnow', 'now_utc', 'SessionWindow', 'last_market_session']
