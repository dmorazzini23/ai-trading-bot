from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta, timezone
from zoneinfo import ZoneInfo

# AI-AGENT-REF: centralized time helpers
NY = ZoneInfo("America/New_York")
UTC = timezone.utc


def ensure_utc_datetime(x) -> datetime:
    """Return tz-aware UTC datetime. Accepts datetime/date/str; reject callables."""
    if callable(x):
        raise TypeError(f"datetime argument was callable: {x!r}")
    if isinstance(x, datetime):
        return x.astimezone(UTC) if x.tzinfo else x.replace(tzinfo=UTC)
    if isinstance(x, date):
        return datetime(x.year, x.month, x.day, tzinfo=UTC)
    if isinstance(x, str):
        dt = datetime.fromisoformat(x.replace("Z", "+00:00"))
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    raise TypeError(f"Unsupported datetime type: {type(x).__name__}")


def nyse_session_utc(for_day: date):
    """Return (start_utc, end_utc) for regular 09:30–16:00 NY session converted to UTC."""
    start_ny = datetime.combine(for_day, time(9, 30), tzinfo=NY)
    end_ny = datetime.combine(for_day, time(16, 0), tzinfo=NY)
    return start_ny.astimezone(UTC), end_ny.astimezone(UTC)


def previous_business_day(d: date) -> date:
    wd = d.weekday()
    if wd == 0:  # Monday -> Friday
        return d - timedelta(days=3)
    if wd == 6:  # Sunday -> Friday
        return d - timedelta(days=2)
    if wd == 5:  # Saturday -> Friday
        return d - timedelta(days=1)
    return d - timedelta(days=1)


def expected_regular_minutes() -> int:
    return 390  # 6.5 hours * 60


__all__ = [
    "ensure_utc_datetime",
    "nyse_session_utc",
    "previous_business_day",
    "expected_regular_minutes",
    "NY",
    "UTC",
]
