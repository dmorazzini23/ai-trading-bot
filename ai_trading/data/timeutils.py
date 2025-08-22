from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

# AI-AGENT-REF: centralized time helpers
NY = ZoneInfo("America/New_York")
UTC = UTC


def ensure_utc_datetime(
    value: Any,
    *,
    default: datetime | None = None,
    clamp_to: str | None = None,
    allow_callables: bool = False,
) -> datetime:
    """Normalize a variety of inputs to a timezone-aware UTC datetime.

    - If ``value`` is callable and ``allow_callables`` is ``True``, call it (no args) and
      re-run normalization on the result.
    - If ``value`` is callable and ``allow_callables`` is ``False``, raise ``TypeError``.
    - If normalization fails, return ``default`` if provided; otherwise raise ``ValueError``.
    """
    # Reject/handle callables early
    if callable(value):
        if allow_callables:
            try:
                value = value()
            except Exception as e:
                raise TypeError(f"datetime argument callable failed: {e}") from e
        else:
            raise TypeError("datetime argument was callable")

    try:
        if isinstance(value, datetime):
            dt = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        elif isinstance(value, date):
            dt = datetime(value.year, value.month, value.day, tzinfo=UTC)
        elif isinstance(value, str):
            tmp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            dt = tmp.astimezone(UTC) if tmp.tzinfo else tmp.replace(tzinfo=UTC)
        else:
            raise TypeError(f"Unsupported datetime type: {type(value).__name__}")
    except Exception as e:
        if default is not None:
            return ensure_utc_datetime(default, allow_callables=allow_callables, clamp_to=clamp_to)
        raise ValueError(f"Invalid datetime input: {value!r}") from e

    if clamp_to == "bod":
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif clamp_to == "eod":
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    return dt


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
