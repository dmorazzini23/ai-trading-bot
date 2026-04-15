"""Lightweight execution guard rails for live trading hotfixes."""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal
from typing import Any, Tuple

from ai_trading.config.management import get_trading_config
from ai_trading.logging import get_logger


def _utcnow() -> _dt.datetime:
    """Return timezone-aware UTC now."""

    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)


class SafetyState:
    """Process-wide safety state (reset on process restart)."""

    shadow_cycle: bool = False
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
            logger.debug("TIMESTAMP_FROM_EPOCH_FAILED", extra={"value": value}, exc_info=True)
            return None
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_dt.timezone.utc)
        try:
            return value.astimezone(_dt.timezone.utc)
        except Exception:
            logger.debug("TIMESTAMP_TZ_NORMALIZE_FAILED", extra={"value": value}, exc_info=True)
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
        logger.debug("QUOTE_STALENESS_AGE_COMPUTE_FAILED", exc_info=True)
        return True, "quote_timestamp_missing"
    if age > float(max_age_sec):
        return True, "stale_quote"
    return False, None


def _require_bid_ask() -> bool:
    try:
        cfg = get_trading_config()
    except Exception:
        logger.debug("REQUIRE_BID_ASK_CONFIG_UNAVAILABLE", exc_info=True)
        return True
    return bool(getattr(cfg, "execution_require_bid_ask", True))


def _max_age_seconds() -> int:
    try:
        cfg = get_trading_config()
    except Exception:
        logger.debug("MAX_AGE_SECONDS_CONFIG_UNAVAILABLE", exc_info=True)
        return 60
    value = getattr(cfg, "execution_max_staleness_sec", 60)
    try:
        return int(value)
    except Exception:
        logger.debug("MAX_AGE_SECONDS_PARSE_FAILED", extra={"value": value}, exc_info=True)
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
    # Degraded data should not implicitly enable shadow mode.
    STATE.shadow_cycle = False


def mark_symbol_stale() -> None:
    """Increment the per-cycle stale symbol counter."""

    STATE.stale_symbols += 1


def end_cycle(stale_threshold_ratio: float = 0.30) -> None:
    """Finalize cycle bookkeeping and promote shadow mode if required."""

    _ = stale_threshold_ratio  # Kept for API stability; stale-ratio promotion was removed.

    # Shadow cycles are not promoted based on stale ratios.


def shadow_active() -> bool:
    """Return ``True`` when the current cycle is shadow-only."""

    return STATE.shadow_cycle
