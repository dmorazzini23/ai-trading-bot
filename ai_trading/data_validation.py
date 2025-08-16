from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


def check_data_freshness(
    data: dict[str, Any], *, now: datetime | None = None, max_age_minutes: int = 30, **_: Any
) -> list[str]:
    """Return list of symbols whose 'asof' is older than max_age_minutes."""  # AI-AGENT-REF: permissive sig
    now = now or datetime.now(UTC)
    stale: list[str] = []
    for sym, rec in (data or {}).items():
        asof = rec.get("asof") if isinstance(rec, dict) else None
        if not isinstance(asof, datetime):
            stale.append(sym)
            continue
        if (now - asof) > timedelta(minutes=max_age_minutes):
            stale.append(sym)
    return stale


def get_stale_symbols(data: dict[str, Any], **kwargs: Any) -> list[str]:
    return check_data_freshness(data, **kwargs)


def validate_trading_data(data: dict[str, Any], **_: Any) -> bool:
    return len(get_stale_symbols(data)) == 0


def emergency_data_check(data: dict[str, Any], *, strict: bool = True, **kwargs: Any) -> bool:
    """Stricter gate used by tests; default to strict=True."""  # AI-AGENT-REF: emergency gate
    stale = get_stale_symbols(data, **kwargs)
    return len(stale) == 0 if strict else len(stale) < len(data)


__all__ = [
    "check_data_freshness",
    "get_stale_symbols",
    "validate_trading_data",
    "emergency_data_check",
]
