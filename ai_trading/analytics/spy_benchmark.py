"""Reported-only SPY benchmark for the weekly research reset scorecard.

Two comparisons answer "is the bot worth more than holding the market?":

* account vs SPY: broker equity return over the window's sessions next to SPY's
  close-to-close return over the same sessions;
* trades vs exposure-matched SPY: for each closed same-day round trip, the P&L
  SPY would have made with the same entry notional, held from the same entry
  minute to the same exit minute.

Both are observations only; they never feed a blocker, gate or decision.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_trading.analytics import portfolio_history
from ai_trading.logging import get_logger

logger = get_logger(__name__)

BENCHMARK_SYMBOL = "SPY"
BENCHMARK_FEED = "iex"
EVIDENCE_CLASS = "reported_only_spy_benchmark"
MINUTE = timedelta(minutes=1)
# A completed SPY minute bar older than this at a fill leaves the trip unpriced.
MAX_PRICE_STALENESS = timedelta(minutes=5)
DAILY_LOOKBACK = timedelta(days=10)


def _default_data_client() -> Any:
    from ai_trading.config.managed_secrets import hydrate_managed_secrets

    hydrate_managed_secrets(required_keys=("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
    from ai_trading.alpaca_api import _get_rest

    return _get_rest(bars=True)


def _bar_rows(response: Any) -> list[Any]:
    data = getattr(response, "data", response)
    if isinstance(data, Mapping):
        return list(data.get(BENCHMARK_SYMBOL) or [])
    return []


def _bar_time(bar: Any) -> datetime | None:
    raw = getattr(bar, "timestamp", None)
    if not isinstance(raw, datetime):
        return None
    return raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)


def _bar_close(bar: Any) -> float | None:
    close = getattr(bar, "close", None)
    try:
        value = float(close)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def fetch_spy_bars(start: datetime, end: datetime, *, client: Any | None = None) -> dict[str, Any]:
    """Return SPY daily and one-minute closes covering ``[start, end]``; never raises."""

    fetched_at = datetime.now(UTC).isoformat()
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        data_client = client if client is not None else _default_data_client()
        daily_response = data_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=BENCHMARK_SYMBOL,
                timeframe=TimeFrame.Day,
                start=start - DAILY_LOOKBACK,
                end=end,
                feed=DataFeed.IEX,
            )
        )
        minute_response = data_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=BENCHMARK_SYMBOL,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
        )
    except portfolio_history._FETCH_ERRORS as exc:  # report, never raise
        logger.warning("SPY_BENCHMARK_BARS_UNAVAILABLE", extra={"error": str(exc)[:200]})
        return {
            "available": False,
            "reason": "bar_fetch_failed",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "fetched_at": fetched_at,
        }
    daily = []
    for bar in _bar_rows(daily_response):
        stamp, close = _bar_time(bar), _bar_close(bar)
        if stamp is not None and close is not None:
            session = stamp.astimezone(portfolio_history.SESSION_TZ).date().isoformat()
            daily.append({"date": session, "close": close})
    minute = []
    for bar in _bar_rows(minute_response):
        stamp, close = _bar_time(bar), _bar_close(bar)
        if stamp is not None and close is not None:
            minute.append({"ts": stamp.isoformat(), "close": close})
    return {
        "available": bool(daily or minute),
        **({} if daily or minute else {"reason": "empty_bar_series"}),
        "symbol": BENCHMARK_SYMBOL,
        "feed": BENCHMARK_FEED,
        "fetched_at": fetched_at,
        "daily": sorted(daily, key=lambda row: row["date"]),
        "minute": sorted(minute, key=lambda row: row["ts"]),
    }


def timed_round_trips(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """FIFO-match fills of each New York session into round trips with entry/exit times."""

    by_session: dict[str, list[tuple[datetime, int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        parsed = portfolio_history.fill_event_from_row(row)
        if parsed is None:
            continue
        ts, event = parsed
        session = ts.astimezone(portfolio_history.SESSION_TZ).date().isoformat()
        by_session.setdefault(session, []).append((ts, index, event))
    trips: list[dict[str, Any]] = []
    for session, events in sorted(by_session.items()):
        events.sort(key=lambda item: (item[0], item[1]))
        books: dict[str, list[dict[str, Any]]] = {}
        for ts, _index, event in events:
            symbol, side = str(event["symbol"]), str(event["side"])
            remaining, price = float(event["qty"]), float(event["price"])
            book = books.setdefault(symbol, [])
            while remaining > 1e-9 and book and book[0]["side"] != side:
                lot = book[0]
                qty = min(remaining, float(lot["qty"]))
                long = lot["side"] == "buy"
                trips.append(
                    {
                        "session": session,
                        "symbol": symbol,
                        "direction": "long" if long else "short",
                        "qty": qty,
                        "entry_ts": lot["ts"],
                        "exit_ts": ts,
                        "entry_price": float(lot["price"]),
                        "exit_price": price,
                        "pnl": (price - float(lot["price"])) * qty * (1 if long else -1),
                    }
                )
                remaining -= qty
                lot["qty"] = float(lot["qty"]) - qty
                if lot["qty"] <= 1e-9:
                    book.pop(0)
            if remaining > 1e-9:
                book.append({"side": side, "qty": remaining, "price": price, "ts": ts})
    return trips


def _minute_index(minute: Iterable[Mapping[str, Any]]) -> tuple[list[datetime], list[float]]:
    pairs = []
    for row in minute:
        try:
            stamp = datetime.fromisoformat(str(row["ts"]))
            close = float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
        pairs.append((stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC), close))
    pairs.sort(key=lambda item: item[0])
    return [stamp + MINUTE for stamp, _close in pairs], [close for _stamp, close in pairs]


def _completed_close(ends: list[datetime], closes: list[float], instant: datetime) -> float | None:
    """Close of the latest SPY minute bar completed at ``instant`` (no lookahead)."""

    position = bisect.bisect_right(ends, instant) - 1
    if position < 0 or instant - ends[position] > MAX_PRICE_STALENESS:
        return None
    return closes[position]


def trades_vs_spy(fills: Iterable[Mapping[str, Any]], minute: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Bot round-trip P&L next to SPY held with the same notional over the same minutes."""

    trips = timed_round_trips(fills)
    ends, closes = _minute_index(minute)
    priced = []
    for trip in trips:
        spy_entry = _completed_close(ends, closes, trip["entry_ts"])
        spy_exit = _completed_close(ends, closes, trip["exit_ts"])
        if spy_entry is None or spy_exit is None:
            continue
        notional = trip["entry_price"] * trip["qty"]
        sign = 1.0 if trip["direction"] == "long" else -1.0
        priced.append({**trip, "notional": notional, "spy_pnl": sign * notional * (spy_exit / spy_entry - 1.0)})
    bot_pnl = sum(trip["pnl"] for trip in priced)
    spy_pnl = sum(trip["spy_pnl"] for trip in priced)
    notional = sum(trip["notional"] for trip in priced)
    return {
        "closed_round_trips": len(trips),
        "priced_round_trips": len(priced),
        "winning_round_trips": sum(1 for trip in trips if trip["pnl"] > 0),
        "bot_pnl_usd_all_trips": sum(trip["pnl"] for trip in trips),
        "bot_pnl_usd": bot_pnl if priced else None,
        "spy_equivalent_pnl_usd": spy_pnl if priced else None,
        "bot_minus_spy_usd": bot_pnl - spy_pnl if priced else None,
        "bot_minus_spy_bps_of_notional": (bot_pnl - spy_pnl) / notional * 10_000.0 if notional else None,
        "traded_notional_usd": notional if priced else None,
        "by_symbol": {
            symbol: {
                "round_trips": sum(1 for trip in priced if trip["symbol"] == symbol),
                "bot_pnl_usd": sum(trip["pnl"] for trip in priced if trip["symbol"] == symbol),
                "spy_equivalent_pnl_usd": sum(trip["spy_pnl"] for trip in priced if trip["symbol"] == symbol),
            }
            for symbol in sorted({trip["symbol"] for trip in priced})
        },
    }


def account_vs_spy(broker: Mapping[str, Any], daily: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Account equity return over the broker window next to SPY close-to-close."""

    sessions = [row for row in broker.get("daily") or [] if isinstance(row, Mapping)]
    if not broker.get("available") or not sessions:
        return {"available": False, "reason": "broker_portfolio_unavailable"}
    start_equity = sessions[0].get("broker_equity_start")
    end_equity = sessions[-1].get("broker_equity_end")
    closes = {str(row["date"]): float(row["close"]) for row in daily if row.get("date") and row.get("close")}
    first, last = str(sessions[0]["date"]), str(sessions[-1]["date"])
    prior = [day for day in closes if day < first]
    if not start_equity or end_equity is None or not prior or last not in closes:
        return {"available": False, "reason": "missing_equity_or_spy_close"}
    spy_start, spy_end = closes[max(prior)], closes[last]
    account_return = float(end_equity) / float(start_equity) - 1.0
    spy_return = spy_end / spy_start - 1.0
    return {
        "available": True,
        "sessions": {"first": first, "last": last, "count": len(sessions)},
        "spy_close_start": {"date": max(prior), "close": spy_start},
        "spy_close_end": {"date": last, "close": spy_end},
        "account_return_bps": account_return * 10_000.0,
        "spy_return_bps": spy_return * 10_000.0,
        "excess_return_bps": (account_return - spy_return) * 10_000.0,
        "account_pnl_usd": float(end_equity) - float(start_equity),
        "spy_equivalent_pnl_usd": float(start_equity) * spy_return,
    }


def weekly_benchmark(
    broker: Mapping[str, Any], fills: Iterable[Mapping[str, Any]], bars: Mapping[str, Any]
) -> dict[str, Any]:
    """Combine both SPY comparisons for one scorecard window; reported only."""

    base = {
        "evidence_class": EVIDENCE_CLASS,
        "reported_only": True,
        "symbol": BENCHMARK_SYMBOL,
        "feed": bars.get("feed", BENCHMARK_FEED),
        "fetched_at": bars.get("fetched_at"),
    }
    if not bars.get("available"):
        return {**base, "available": False, "reason": str(bars.get("reason") or "unavailable"), "error": bars.get("error")}
    return {
        **base,
        "available": True,
        "account_vs_spy": account_vs_spy(broker, bars.get("daily") or []),
        "trades_vs_spy": trades_vs_spy(fills, bars.get("minute") or []),
    }


__all__ = [
    "EVIDENCE_CLASS",
    "account_vs_spy",
    "fetch_spy_bars",
    "timed_round_trips",
    "trades_vs_spy",
    "weekly_benchmark",
]
