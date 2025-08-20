from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict

# AI-AGENT-REF: timezone helpers for Alpaca compatibility


def ensure_datetime(x: Any) -> datetime:
    """Return timezone-aware UTC datetime."""
    if isinstance(x, datetime):
        dt = x
    elif isinstance(x, date):
        dt = datetime.combine(x, datetime.min.time())
    elif isinstance(x, str):
        if not x:
            raise ValueError("empty_string")
        try:
            dt = datetime.fromisoformat(x.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("invalid_string") from exc
    else:
        raise TypeError(f"Unsupported type for ensure_datetime: {type(x)!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_rfc3339z(dt: datetime) -> str:
    dt_utc = ensure_datetime(dt).replace(microsecond=0)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_date_only(val: datetime | date | str) -> str:
    if isinstance(val, str):
        if len(val) == 10 and val.count("-") == 2:
            return val
        dt_obj = ensure_datetime(val)
    else:
        dt_obj = ensure_datetime(val)
    return dt_obj.date().isoformat()


def compose_intraday_params(start: Any, end: Any) -> Dict[str, str]:
    return {"start": to_rfc3339z(ensure_datetime(start)), "end": to_rfc3339z(ensure_datetime(end))}


def compose_daily_params(start: Any, end: Any) -> Dict[str, str]:
    return {"start": to_date_only(start), "end": to_date_only(end)}


__all__ = [
    "ensure_datetime",
    "to_rfc3339z",
    "to_date_only",
    "compose_intraday_params",
    "compose_daily_params",
]
