"""Lightweight execution guard rails for live trading hotfixes."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass


def _utcnow() -> _dt.datetime:
    """Return timezone-aware UTC now."""

    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)


def _trading_day(ts: _dt.datetime | None = None) -> _dt.date:
    """Return the NYSE trading day inferred from ``ts`` (UTC)."""

    ts = ts or _utcnow()
    return ts.date()


@dataclass
class PDTState:
    """In-memory pattern day trader lockout state."""

    locked_day: _dt.date | None = None
    limit: int = 0
    count: int = 0


class SafetyState:
    """Process-wide safety state (reset on process restart)."""

    pdt: PDTState = PDTState()
    shadow_cycle: bool = False
    stale_symbols: int = 0
    universe_size: int = 0


STATE = SafetyState()


def pdt_guard(pattern_day_trader: bool, daytrade_limit: int, daytrade_count: int) -> bool:
    """Return ``True`` when opening trades are allowed for the current day."""

    today = _trading_day()
    if STATE.pdt.locked_day == today:
        return False

    if pattern_day_trader and daytrade_count >= daytrade_limit > 0:
        STATE.pdt = PDTState(
            locked_day=today,
            limit=daytrade_limit,
            count=daytrade_count,
        )
        return False
    return True


def pdt_lockout_active() -> bool:
    """Return ``True`` when PDT lockout is active for today."""

    return STATE.pdt.locked_day == _trading_day()


def pdt_lockout_info() -> dict[str, int | bool | None]:
    """Return metadata describing the PDT lockout state."""

    return {
        "active": pdt_lockout_active(),
        "limit": STATE.pdt.limit,
        "count": STATE.pdt.count,
    }


def quote_fresh_enough(
    quote_ts_utc: _dt.datetime | None,
    max_age_sec: int,
) -> bool:
    """Return ``True`` when ``quote_ts_utc`` is within ``max_age_sec`` seconds."""

    if quote_ts_utc is None:
        return False
    if quote_ts_utc.tzinfo is None:
        quote_ts_utc = quote_ts_utc.replace(tzinfo=_dt.timezone.utc)
    age = (_utcnow() - quote_ts_utc).total_seconds()
    return age <= max_age_sec


def begin_cycle(universe_size: int, degraded: bool) -> None:
    """Reset per-cycle counters and shadow-mode hint."""

    STATE.universe_size = int(universe_size or 0)
    STATE.stale_symbols = 0
    STATE.shadow_cycle = bool(degraded)


def mark_symbol_stale() -> None:
    """Increment the per-cycle stale symbol counter."""

    STATE.stale_symbols += 1


def end_cycle(stale_threshold_ratio: float = 0.30) -> None:
    """Finalize cycle bookkeeping and promote shadow mode if required."""

    if STATE.universe_size > 0:
        ratio = STATE.stale_symbols / float(STATE.universe_size)
        if ratio >= stale_threshold_ratio:
            STATE.shadow_cycle = True


def shadow_active() -> bool:
    """Return ``True`` when the current cycle is shadow-only."""

    return STATE.shadow_cycle

