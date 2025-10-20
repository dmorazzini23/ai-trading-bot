"""Lightweight execution guard rails for live trading hotfixes."""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal
from dataclasses import dataclass
from typing import Any, Tuple

from ai_trading.config.management import get_trading_config
from ai_trading.logging import get_logger


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
    shadow_cycle_forced: bool = False
    stale_symbols: int = 0
    universe_size: int = 0


STATE = SafetyState()
logger = get_logger(__name__)


def _now() -> _dt.datetime:
    """Return timezone-aware ``datetime`` in UTC."""

    return _dt.datetime.now(tz=_dt.timezone.utc)


def _coerce_timestamp(value: Any) -> _dt.datetime | None:
    """Return a timezone-aware timestamp extracted from *value* if possible."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return _dt.datetime.fromtimestamp(float(value), tz=_dt.timezone.utc)
        except Exception:
            return None
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_dt.timezone.utc)
        try:
            return value.astimezone(_dt.timezone.utc)
        except Exception:
            return None
    return None


def _is_stale(quote: dict[str, Any], now: _dt.datetime, max_age_sec: int) -> Tuple[bool, str | None]:
    """Return ``(True, reason)`` when *quote* is stale or timestamp missing."""

    ts = (
        quote.get("timestamp")
        or quote.get("ts")
        or quote.get("t")
        or quote.get("time")
    )
    dt_val = _coerce_timestamp(ts)
    if dt_val is None:
        return True, "quote_timestamp_missing"
    try:
        age = (now - dt_val).total_seconds()
    except Exception:
        return True, "quote_timestamp_missing"
    if age > float(max_age_sec):
        return True, "stale_quote"
    return False, None


def _require_bid_ask() -> bool:
    try:
        cfg = get_trading_config()
    except Exception:
        return True
    return bool(getattr(cfg, "execution_require_bid_ask", True))


def _max_age_seconds() -> int:
    try:
        cfg = get_trading_config()
    except Exception:
        return 60
    value = getattr(cfg, "execution_max_staleness_sec", 60)
    try:
        return int(value)
    except Exception:
        return 60


def _safe_bool(value: Any) -> bool:
    """Best-effort boolean normalization for configuration payloads."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, Decimal)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return False


def can_execute(
    quote: dict[str, Any] | None,
    *,
    now: _dt.datetime | None = None,
    max_age_sec: int | None = None,
) -> Tuple[bool, str | None]:
    """Return gating decision for *quote* with optional overrides."""

    if quote is None:
        return False, "no_quote"
    now = now or _now()
    max_age = int(max_age_sec if max_age_sec is not None else _max_age_seconds())
    bid = quote.get("bid") or quote.get("bp") or quote.get("bid_price")
    ask = quote.get("ask") or quote.get("ap") or quote.get("ask_price")
    if _require_bid_ask():
        if bid is None or ask is None:
            return False, "missing_bid_ask"
        try:
            spread_ok = float(ask) >= float(bid) > 0
        except (TypeError, ValueError):
            spread_ok = False
        if not spread_ok:
            return False, "negative_spread"
    stale, reason = _is_stale(quote, now, max_age)
    if stale:
        return False, reason
    return True, None


def pdt_preflight(ctx: dict[str, Any]) -> Tuple[bool, str | None]:
    """Return PDT eligibility based on context mapping used in tests."""

    pattern_day_trader = _safe_bool(ctx.get("pattern_day_trader"))
    if not pattern_day_trader:
        return True, None
    count = int(ctx.get("daytrade_count", 0) or 0)
    limit = int(ctx.get("daytrade_limit", 0) or 0)
    if limit > 0 and count >= limit:
        return False, "pdt_limit_exceeded"
    return True, None


def pdt_guard(pattern_day_trader: bool, daytrade_limit: int, daytrade_count: int) -> bool:
    """Return ``True`` when opening trades are allowed for the current day."""

    today = _trading_day()
    if STATE.pdt.locked_day == today:
        STATE.shadow_cycle_forced = True
        STATE.shadow_cycle = True
        return False

    if pattern_day_trader and daytrade_count >= daytrade_limit > 0:
        previous_day = STATE.pdt.locked_day
        previously_forced = STATE.shadow_cycle_forced
        STATE.pdt = PDTState(
            locked_day=today,
            limit=daytrade_limit,
            count=daytrade_count,
        )
        STATE.shadow_cycle_forced = True
        STATE.shadow_cycle = True
        if previous_day != today or not previously_forced:
            logger.info(
                "PDT_SHADOW_MODE_ENABLED",
                extra={
                    "day": today.isoformat(),
                    "limit": daytrade_limit,
                    "count": daytrade_count,
                },
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
    if STATE.shadow_cycle_forced:
        STATE.shadow_cycle = True
    else:
        STATE.shadow_cycle = bool(degraded)


def mark_symbol_stale() -> None:
    """Increment the per-cycle stale symbol counter."""

    STATE.stale_symbols += 1


def end_cycle(stale_threshold_ratio: float = 0.30) -> None:
    """Finalize cycle bookkeeping and promote shadow mode if required."""

    if STATE.universe_size > 0:
        ratio = STATE.stale_symbols / float(STATE.universe_size)
        trigger_shadow = ratio >= stale_threshold_ratio
        STATE.shadow_cycle_forced = trigger_shadow
        if trigger_shadow:
            STATE.shadow_cycle = True


def shadow_active() -> bool:
    """Return ``True`` when the current cycle is shadow-only."""

    return STATE.shadow_cycle
