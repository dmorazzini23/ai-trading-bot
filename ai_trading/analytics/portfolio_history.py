"""Broker-reported portfolio history as a reported-only check on internal P&L.

Alpaca's ``get_portfolio_history`` returns the account's daily equity and
profit/loss as the broker computes it. That number includes unrealised moves on
open positions, fees and corporate actions; the internal figure is the realised
P&L of same-day round trips rebuilt from fill events. The two are different
evidence classes and are shown side by side, never merged.

Nothing here feeds a gate, threshold or trading decision. Fetches never raise:
failures come back as ``available: False`` with a reason.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError, RetryException

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text

logger = get_logger(__name__)

EVIDENCE_CLASS = "broker_reported_portfolio_history"
INTERNAL_EVIDENCE_CLASS = "internal_same_day_fill_round_trips"
SESSION_TZ = ZoneInfo("America/New_York")
# One month of daily rows covers the report date and the trailing week.
DEFAULT_PERIOD = "1M"
DEFAULT_TIMEFRAME = "1D"
CACHE_TTL_SECONDS = 300.0
BASIS_NOTE = (
    "Broker P&L is the account equity change: it includes unrealised moves, "
    "fees and corporate actions. Internal P&L is realised same-day round trips "
    "from fill events."
)

# Credential, secret-hydration (AWS CLI), network, SDK validation and broker errors.
_FETCH_ERRORS = (
    APIError,
    RetryException,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    ImportError,
    subprocess.SubprocessError,
)

_MEMORY_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _field(raw: Any, name: str) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(name)
    return getattr(raw, name, None)


def _list_field(raw: Any, name: str) -> list[Any]:
    value = _field(raw, name)
    return list(value) if isinstance(value, (list, tuple)) else []


def normalize_portfolio_history(
    raw: Any,
    *,
    period: str,
    timeframe: str,
    fetched_at: str,
) -> dict[str, Any]:
    """Return one row per New York session date from a broker history payload."""

    timestamps = _list_field(raw, "timestamp")
    equity = _list_field(raw, "equity")
    profit_loss = _list_field(raw, "profit_loss")
    profit_loss_pct = _list_field(raw, "profit_loss_pct")
    base_value = _float_or_none(_field(raw, "base_value"))
    by_date: dict[str, dict[str, Any]] = {}
    for index, stamp in enumerate(timestamps):
        epoch = _float_or_none(stamp)
        if epoch is None:
            continue
        instant = datetime.fromtimestamp(epoch, tz=UTC)
        session_date = instant.astimezone(SESSION_TZ).date().isoformat()
        by_date[session_date] = {
            "date": session_date,
            "broker_timestamp": int(epoch),
            "broker_timestamp_utc": instant.isoformat(),
            "equity": _float_or_none(equity[index]) if index < len(equity) else None,
            "profit_loss": _float_or_none(profit_loss[index]) if index < len(profit_loss) else None,
            # Broker reports this as a fraction rounded to four decimals.
            "profit_loss_pct": (
                _float_or_none(profit_loss_pct[index]) if index < len(profit_loss_pct) else None
            ),
            "base_value": base_value,
        }
    rows = [by_date[key] for key in sorted(by_date)]
    return {
        "available": bool(rows),
        "evidence_class": EVIDENCE_CLASS,
        "period": period,
        "timeframe": timeframe,
        "broker_timeframe": str(_field(raw, "timeframe") or "") or None,
        "base_value": base_value,
        "fetched_at": fetched_at,
        "rows": rows,
        **({} if rows else {"reason": "empty_broker_series"}),
    }


def _default_trading_client() -> Any:
    from ai_trading.config.managed_secrets import hydrate_managed_secrets

    hydrate_managed_secrets(required_keys=("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
    from ai_trading.alpaca_api import _get_rest

    return _get_rest(bars=False)


def _history_request(period: str, timeframe: str) -> Any:
    from alpaca.trading.requests import GetPortfolioHistoryRequest

    return GetPortfolioHistoryRequest(period=period, timeframe=timeframe)


def _cache_key_text(period: str, timeframe: str) -> str:
    return f"{period}|{timeframe}"


def _read_file_cache(
    cache_path: Path, period: str, timeframe: str, now: float, ttl_seconds: float
) -> tuple[float, dict[str, Any]] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = payload.get(_cache_key_text(period, timeframe)) if isinstance(payload, dict) else None
    if not isinstance(entry, dict):
        return None
    cached_at = _float_or_none(entry.get("cached_at_epoch"))
    history = entry.get("history")
    if cached_at is None or not isinstance(history, dict) or not 0 <= now - cached_at <= ttl_seconds:
        return None
    return cached_at, history


def _write_file_cache(
    cache_path: Path, period: str, timeframe: str, now: float, history: dict[str, Any]
) -> None:
    try:
        existing = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing[_cache_key_text(period, timeframe)] = {"cached_at_epoch": now, "history": history}
    try:
        atomic_write_text(cache_path, json.dumps(existing, sort_keys=True, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        logger.warning(
            "BROKER_PORTFOLIO_HISTORY_CACHE_WRITE_FAILED",
            extra={"path": str(cache_path), "error": str(exc)},
        )


def _with_cache_info(history: dict[str, Any], *, source: str, age_s: float) -> dict[str, Any]:
    result = dict(history)
    result["cache"] = {"source": source, "age_s": round(max(age_s, 0.0), 3)}
    return result


def fetch_portfolio_history(
    *,
    period: str = DEFAULT_PERIOD,
    timeframe: str = DEFAULT_TIMEFRAME,
    client: Any | None = None,
    cache_path: Path | None = None,
    ttl_seconds: float = CACHE_TTL_SECONDS,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Return the broker's normalised daily portfolio history; never raises.

    Successful fetches are cached by ``(period, timeframe)`` in memory and, when
    ``cache_path`` is given, on disk so that one-shot processes started every
    minute share the cache. Failures are not cached.
    """

    now = float(clock())
    key = (period, timeframe)
    cached = _MEMORY_CACHE.get(key)
    if cached is not None and 0 <= now - cached[0] <= ttl_seconds:
        return _with_cache_info(cached[1], source="memory", age_s=now - cached[0])
    if cache_path is not None:
        file_cached = _read_file_cache(cache_path, period, timeframe, now, ttl_seconds)
        if file_cached is not None:
            _MEMORY_CACHE[key] = file_cached
            return _with_cache_info(file_cached[1], source="file", age_s=now - file_cached[0])
    fetched_at = datetime.fromtimestamp(now, tz=UTC).isoformat()
    try:
        trading_client = client if client is not None else _default_trading_client()
        raw = trading_client.get_portfolio_history(_history_request(period, timeframe))
        history = normalize_portfolio_history(
            raw, period=period, timeframe=timeframe, fetched_at=fetched_at
        )
    except _FETCH_ERRORS as exc:  # report, never raise
        logger.warning(
            "BROKER_PORTFOLIO_HISTORY_UNAVAILABLE",
            extra={"period": period, "timeframe": timeframe, "error": str(exc)[:200]},
        )
        return {
            "available": False,
            "evidence_class": EVIDENCE_CLASS,
            "reason": "broker_fetch_failed",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "period": period,
            "timeframe": timeframe,
            "fetched_at": fetched_at,
            "rows": [],
        }
    if history.get("available"):
        _MEMORY_CACHE[key] = (now, history)
        if cache_path is not None:
            _write_file_cache(cache_path, period, timeframe, now, history)
    return _with_cache_info(history, source="broker", age_s=0.0)


def clear_portfolio_history_cache() -> None:
    _MEMORY_CACHE.clear()


def _session_rows(history: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row for row in history.get("rows") or [] if isinstance(row, dict) and row.get("date")]


def _session_change(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    row = rows[index]
    end = _float_or_none(row.get("equity"))
    pnl = _float_or_none(row.get("profit_loss"))
    start = _float_or_none(rows[index - 1].get("equity")) if index > 0 else None
    if start is None and end is not None and pnl is not None:
        start = end - pnl
    pct = _float_or_none(row.get("profit_loss_pct"))
    return {
        "date": row["date"],
        "broker_timestamp_utc": row.get("broker_timestamp_utc"),
        "broker_equity_start": start,
        "broker_equity_end": end,
        "broker_daily_pnl": pnl,
        # Percent (not a fraction); the broker rounds its fraction to four decimals.
        "broker_daily_pnl_pct": None if pct is None else pct * 100.0,
        # Unrounded, from the broker's own P&L and start equity.
        "broker_daily_pnl_bps_of_start_equity": (
            pnl / start * 10_000.0 if pnl is not None and start else None
        ),
    }


def _difference(
    broker_pnl: float | None, internal_pnl: float | None, start_equity: float | None
) -> tuple[float | None, float | None]:
    if broker_pnl is None or internal_pnl is None:
        return None, None
    difference = float(broker_pnl) - float(internal_pnl)
    bps = difference / float(start_equity) * 10_000.0 if start_equity else None
    return difference, bps


def daily_crosscheck(
    history: Mapping[str, Any],
    session_date: str,
    *,
    internal_pnl: float | None,
    internal_status: str,
) -> dict[str, Any]:
    """Compare one session's broker equity change with internal same-day-fill P&L."""

    base = {
        "evidence_class": EVIDENCE_CLASS,
        "internal_evidence_class": INTERNAL_EVIDENCE_CLASS,
        "reported_only": True,
        "report_date": session_date,
        "internal_same_day_fill_pnl": internal_pnl,
        "internal_status": internal_status,
        "basis_note": BASIS_NOTE,
        "fetched_at": history.get("fetched_at"),
        "cache": history.get("cache"),
    }
    if not history.get("available"):
        return {
            **base,
            "available": False,
            "reason": str(history.get("reason") or "unavailable"),
            "error": history.get("error"),
        }
    rows = _session_rows(history)
    index = next((i for i, row in enumerate(rows) if row["date"] == session_date), None)
    if index is None:
        return {
            **base,
            "available": False,
            "reason": "report_date_not_in_broker_series",
            "broker_latest_date": rows[-1]["date"] if rows else None,
        }
    change = _session_change(rows, index)
    difference, difference_bps = _difference(
        change["broker_daily_pnl"], internal_pnl, change["broker_equity_start"]
    )
    return {
        **base,
        "available": True,
        **{key: value for key, value in change.items() if key != "date"},
        "difference_usd": difference,
        "difference_bps_of_start_equity": difference_bps,
    }


def weekly_crosscheck(
    history: Mapping[str, Any],
    internal_by_session: Mapping[str, float],
    *,
    start_date: date,
    end_date: date,
    partial_sessions: Iterable[str] = (),
    internal_available: bool = True,
) -> dict[str, Any]:
    """Broker daily series and weekly equity change next to internal daily P&L.

    A session with a broker row but no internal round trips counts as zero
    internal P&L. Sessions in ``partial_sessions`` (cut by the window edge) and
    every session when the fill source is unavailable have no internal figure and
    are left out of the comparison.
    """

    partial = set(partial_sessions)
    base = {
        "evidence_class": EVIDENCE_CLASS,
        "internal_evidence_class": INTERNAL_EVIDENCE_CLASS,
        "reported_only": True,
        "basis_note": BASIS_NOTE,
        "window_sessions": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "fetched_at": history.get("fetched_at"),
        "internal_available": internal_available,
        "internal_weekly_same_day_fill_pnl": (
            float(sum(internal_by_session.values())) if internal_available else None
        ),
    }
    if not history.get("available"):
        return {
            **base,
            "available": False,
            "reason": str(history.get("reason") or "unavailable"),
            "error": history.get("error"),
            "daily": [
                {"date": day, "internal_same_day_fill_pnl": float(pnl)}
                for day, pnl in sorted(internal_by_session.items())
            ],
        }
    rows = _session_rows(history)
    daily: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not start_date.isoformat() <= row["date"] <= end_date.isoformat():
            continue
        change = _session_change(rows, index)
        internal_pnl: float | None = None
        if internal_available and row["date"] not in partial:
            internal_pnl = float(internal_by_session.get(row["date"], 0.0))
        difference, difference_bps = _difference(
            change["broker_daily_pnl"], internal_pnl, change["broker_equity_start"]
        )
        daily.append(
            {
                **change,
                "internal_same_day_fill_pnl": internal_pnl,
                "internal_partial_window": row["date"] in partial,
                "difference_usd": difference,
                "difference_bps_of_start_equity": difference_bps,
            }
        )
    if not daily:
        return {**base, "available": False, "reason": "no_broker_sessions_in_window", "daily": []}
    start_equity = daily[0]["broker_equity_start"]
    end_equity = daily[-1]["broker_equity_end"]
    weekly_change = (
        float(end_equity) - float(start_equity)
        if start_equity is not None and end_equity is not None
        else None
    )
    comparable = [row for row in daily if row["internal_same_day_fill_pnl"] is not None]
    broker_comparable = sum(float(row["broker_daily_pnl"] or 0.0) for row in comparable)
    internal_comparable = sum(float(row["internal_same_day_fill_pnl"]) for row in comparable)
    weekly_difference, weekly_difference_bps = _difference(
        broker_comparable if comparable else None,
        internal_comparable if comparable else None,
        start_equity,
    )
    return {
        **base,
        "available": True,
        "broker_equity_start": start_equity,
        "broker_equity_end": end_equity,
        "broker_weekly_equity_change": weekly_change,
        "comparable_sessions": len(comparable),
        "broker_comparable_pnl": broker_comparable if comparable else None,
        "internal_comparable_pnl": internal_comparable if comparable else None,
        "weekly_difference_usd": weekly_difference,
        "weekly_difference_bps_of_start_equity": weekly_difference_bps,
        "daily": daily,
    }


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def fill_event_from_row(row: Mapping[str, Any]) -> tuple[datetime, dict[str, Any]] | None:
    """Return ``(timestamp, event)`` for a usable fill-event row, else ``None``."""

    ts = _parse_ts(row.get("ts") or row.get("entry_time") or row.get("timestamp"))
    if ts is None:
        return None
    side = str(row.get("side") or "").strip().lower()
    if side not in {"buy", "sell", "sell_short"}:
        return None
    qty = _float_or_none(row.get("fill_qty") or row.get("qty") or row.get("quantity"))
    price = _float_or_none(row.get("fill_price") or row.get("entry_price") or row.get("price"))
    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol or qty is None or price is None or qty <= 0 or price <= 0:
        return None
    return ts, {
        "symbol": symbol,
        "side": "sell" if side == "sell_short" else side,
        "qty": float(qty),
        "price": float(price),
    }


def same_day_round_trips(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """FIFO-match time-ordered fills of one session into realised round trips."""

    books: dict[str, list[dict[str, float | str]]] = {}
    pnl_by_symbol: dict[str, float] = {}
    closed_trades = 0
    winning_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    for event in events:
        symbol = str(event["symbol"])
        side = str(event["side"])
        remaining = float(event["qty"])
        price = float(event["price"])
        book = books.setdefault(symbol, [])
        while remaining > 0 and book and str(book[0].get("side")) != side:
            lot = book[0]
            lot_qty = float(lot.get("qty") or 0.0)
            lot_price = float(lot.get("price") or 0.0)
            close_qty = min(remaining, lot_qty)
            if str(lot.get("side")) == "buy" and side == "sell":
                pnl = (price - lot_price) * close_qty
            else:
                pnl = (lot_price - price) * close_qty
            pnl_by_symbol[symbol] = pnl_by_symbol.get(symbol, 0.0) + pnl
            closed_trades += 1
            if pnl > 0:
                winning_trades += 1
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += -pnl
            remaining -= close_qty
            lot["qty"] = lot_qty - close_qty
            if float(lot["qty"]) <= 1e-9:
                book.pop(0)
        if remaining > 1e-9:
            book.append({"side": side, "qty": remaining, "price": price})
    open_qty_by_symbol: dict[str, float] = {}
    for symbol, lots in books.items():
        net_qty = 0.0
        for lot in lots:
            qty = float(lot.get("qty") or 0.0)
            net_qty += qty if str(lot.get("side")) == "buy" else -qty
        if abs(net_qty) > 1e-9:
            open_qty_by_symbol[symbol] = net_qty
    return {
        "pnl_by_symbol": pnl_by_symbol,
        "closed_trades": closed_trades,
        "winning_trades": winning_trades,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "open_qty_by_symbol": open_qty_by_symbol,
    }


def internal_same_day_fill_pnl_by_session(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Realised same-day round-trip P&L per New York session date."""

    by_session: dict[str, list[tuple[datetime, int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        parsed = fill_event_from_row(row)
        if parsed is None:
            continue
        ts, event = parsed
        session = ts.astimezone(SESSION_TZ).date().isoformat()
        by_session.setdefault(session, []).append((ts, index, event))
    result: dict[str, float] = {}
    for session, events in sorted(by_session.items()):
        events.sort(key=lambda item: (item[0], item[1]))
        trips = same_day_round_trips(event for _ts, _index, event in events)
        result[session] = float(sum(trips["pnl_by_symbol"].values()))
    return result


__all__ = [
    "BASIS_NOTE",
    "EVIDENCE_CLASS",
    "INTERNAL_EVIDENCE_CLASS",
    "clear_portfolio_history_cache",
    "daily_crosscheck",
    "fetch_portfolio_history",
    "fill_event_from_row",
    "internal_same_day_fill_pnl_by_session",
    "normalize_portfolio_history",
    "same_day_round_trips",
    "weekly_crosscheck",
]
