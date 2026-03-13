from __future__ import annotations

"""Summarize realized trade and decision-gate performance from runtime artifacts."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any, Mapping

_DEFAULT_TRADE_HISTORY_PATH = "runtime/trade_history.parquet"
_DEFAULT_TCA_PATH = "runtime/tca_records.jsonl"


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _as_int(value: Any) -> int | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except (TypeError, ValueError):
        return None


def _normalise_iso_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return None


def _select_recent_daily_rows(
    rows: Any,
    *,
    lookback_days: int,
) -> tuple[list[dict[str, Any]], int]:
    if lookback_days <= 0 or not isinstance(rows, list):
        return [], 0
    by_day: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        day = _normalise_iso_date(raw.get("date"))
        if not day:
            continue
        by_day[day] = dict(raw)
    days = sorted(by_day.keys())
    if not days:
        return [], 0
    selected_days = days[-int(lookback_days) :]
    return [by_day[day] for day in selected_days], len(days)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _normalise_fill_source(value: Any) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "unknown"
    if token in {"broker_reconcile", "reconcile", "backfill"}:
        return "reconcile_backfill"
    if token.startswith("broker_reconcile"):
        return "reconcile_backfill"
    if token in {"live", "initial", "poll", "final", "manual_probe"}:
        return "live"
    if "reconcile" in token:
        return "reconcile_backfill"
    return "live"


def _build_order_source_lookup(path: Path) -> dict[str, str]:
    latest: dict[str, tuple[str, str]] = {}
    for row in _load_json_lines(path):
        order_id = str(row.get("order_id") or "").strip()
        if not order_id:
            continue
        ts = str(row.get("ts") or "")
        source = _normalise_fill_source(row.get("source"))
        previous = latest.get(order_id)
        if previous is None or ts >= previous[0]:
            latest[order_id] = (ts, source)
    return {order_id: source for order_id, (_ts, source) in latest.items()}


def _load_trade_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_json_lines(path)
    if suffix == ".json":
        payload = _load_json(path)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            candidate = payload.get("trades")
            if isinstance(candidate, list):
                return [row for row in candidate if isinstance(row, dict)]
        return []
    try:
        import pandas as pd
    except ImportError:
        return []

    frame = None
    if suffix in {".parquet", ".pq"}:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            try:
                frame = pd.read_pickle(path)
            except Exception:
                frame = None
    elif suffix in {".pkl", ".pickle"}:
        try:
            frame = pd.read_pickle(path)
        except Exception:
            frame = None
    elif suffix == ".csv":
        try:
            frame = pd.read_csv(path)
        except Exception:
            frame = None
    if frame is None:
        return []
    return [row for row in frame.to_dict(orient="records") if isinstance(row, dict)]


def _normalise_side(value: Any) -> str | None:
    side = str(value or "").strip().lower()
    if side in {"buy", "long", "b"}:
        return "buy"
    if side in {"sell", "short", "s"}:
        return "sell"
    return None


def _normalise_status_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    token = token.replace("-", "_").replace(" ", "_")
    if "." in token:
        token = token.rsplit(".", 1)[-1]
    if token in {"partial_fill", "partiallyfilled"}:
        return "partially_filled"
    return token


def _resolve_qty(row: dict[str, Any]) -> float | None:
    for key in ("qty", "quantity", "filled_qty"):
        qty = _as_float(row.get(key))
        if qty is not None and qty > 0:
            return qty
    return None


def _resolve_price(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        price = _as_float(row.get(key))
        if price is not None and price > 0:
            return price
    return None


def _resolve_fee_amount_with_source(
    row: dict[str, Any],
    qty: float,
    price: float,
) -> tuple[float, str | None]:
    for key in ("fee_amount", "fee", "fees", "commission", "commission_amount"):
        fee = _as_float(row.get(key))
        if fee is not None:
            return abs(fee), key
    fee_bps = _as_float(row.get("fee_bps"))
    if fee_bps is not None and fee_bps > 0:
        return abs(qty * price * (fee_bps / 10000.0)), "fee_bps"
    return 0.0, None


def _resolve_fee_amount(row: dict[str, Any], qty: float, price: float) -> float:
    return _resolve_fee_amount_with_source(row, qty, price)[0]


def _resolve_slippage_bps_with_source(
    row: dict[str, Any],
    side: str,
    price: float,
) -> tuple[float, str | None]:
    for key in (
        "slippage_bps",
        "is_bps",
        "implementation_shortfall_bps",
        "spread_paid_bps",
    ):
        slippage_bps = _as_float(row.get(key))
        if slippage_bps is not None:
            return slippage_bps, key

    expected = _resolve_price(row, "expected_price")
    if expected is None or expected <= 0:
        return 0.0, None
    if side == "buy":
        return ((price - expected) / expected) * 10000.0, "expected_price"
    return ((expected - price) / expected) * 10000.0, "expected_price"


def _resolve_slippage_bps(row: dict[str, Any], side: str, price: float) -> float:
    return _resolve_slippage_bps_with_source(row, side, price)[0]


def _resolve_slippage_cost_with_source(
    row: dict[str, Any],
    *,
    qty: float,
    side: str,
    price: float,
) -> tuple[float, str | None]:
    for key in (
        "slippage_cost",
        "slippage",
        "slippage_amount",
        "implementation_shortfall_cost",
    ):
        value = _as_float(row.get(key))
        if value is not None:
            return abs(value), key
    slippage_bps, source = _resolve_slippage_bps_with_source(row, side, price)
    if source is None:
        return 0.0, None
    return abs(qty * price * (slippage_bps / 10000.0)), source


@dataclass(slots=True)
class _FillEvent:
    symbol: str
    side: str
    qty: float
    price: float
    timestamp: datetime | None
    strategy: str
    signal_tags: str
    fee_per_share: float
    slippage_per_share: float
    order_id: str | None
    fill_source: str


@dataclass(slots=True)
class _OpenLot:
    side: str
    qty: float
    price: float
    timestamp: datetime | None
    strategy: str
    signal_tags: str
    fee_per_share: float
    slippage_per_share: float
    fill_source: str


@dataclass(slots=True)
class _TCALegEvent:
    symbol: str
    side: str
    qty: float
    price: float
    timestamp: datetime
    fee_cost: float
    slippage_cost: float


def _as_fill_event(
    row: dict[str, Any],
    *,
    order_source_lookup: Mapping[str, str] | None = None,
) -> _FillEvent | None:
    symbol = str(row.get("symbol", "") or "").strip().upper()
    if not symbol:
        return None
    side = _normalise_side(row.get("side"))
    if side is None:
        return None
    qty = _resolve_qty(row)
    price = _resolve_price(
        row,
        "entry_price",
        "price",
        "fill_price",
        "filled_avg_price",
        "average_price",
    )
    if qty is None or price is None:
        return None
    order_id_raw = row.get("order_id")
    order_id = None if order_id_raw in (None, "") else str(order_id_raw)
    source_hint = row.get("source")
    if source_hint in (None, "") and order_id and order_source_lookup:
        source_hint = order_source_lookup.get(order_id)
    fill_source = _normalise_fill_source(source_hint)
    fee_amount = _resolve_fee_amount(row, qty, price)
    slippage_bps = _resolve_slippage_bps(row, side, price)
    return _FillEvent(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        timestamp=_parse_timestamp(
            row.get("entry_time")
            or row.get("timestamp")
            or row.get("filled_at")
            or row.get("executed_at")
            or row.get("updated_at")
            or row.get("ts")
        ),
        strategy=str(row.get("strategy", "") or row.get("strategy_id", "") or ""),
        signal_tags=str(row.get("signal_tags", "") or ""),
        fee_per_share=(fee_amount / qty) if qty > 0 else 0.0,
        slippage_per_share=price * (slippage_bps / 10000.0),
        order_id=order_id,
        fill_source=fill_source,
    )


def _extract_fill_events(
    records: list[dict[str, Any]],
    *,
    order_source_lookup: Mapping[str, str] | None = None,
) -> list[_FillEvent]:
    events: list[tuple[datetime, int, _FillEvent]] = []
    for index, row in enumerate(records):
        event = _as_fill_event(row, order_source_lookup=order_source_lookup)
        if event is None:
            continue
        event_ts = event.timestamp or datetime.max.replace(tzinfo=UTC)
        events.append((event_ts, index, event))
    events.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in events]


def _closed_trade_record(
    *,
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    entry_time: datetime | None,
    exit_time: datetime | None,
    strategy: str,
    signal_tags: str,
    gross_pnl: float,
    fee_cost: float,
    slippage_cost: float,
    fill_source: str | None = None,
) -> dict[str, Any]:
    entry_notional = abs(entry_price * qty)
    net_pnl = gross_pnl - fee_cost - slippage_cost
    net_edge_bps = (net_pnl / entry_notional * 10000.0) if entry_notional > 0 else None
    holding_seconds = None
    if entry_time is not None and exit_time is not None:
        holding_seconds = max(0.0, (exit_time - entry_time).total_seconds())
    return {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": entry_time.isoformat() if entry_time is not None else None,
        "exit_time": exit_time.isoformat() if exit_time is not None else None,
        "strategy": strategy,
        "signal_tags": signal_tags,
        "entry_notional": entry_notional,
        "gross_pnl": gross_pnl,
        "fee_cost": fee_cost,
        "slippage_cost": slippage_cost,
        "net_pnl": net_pnl,
        "net_edge_bps": net_edge_bps,
        "holding_seconds": holding_seconds,
        "fill_source": _normalise_fill_source(fill_source),
    }


def _direct_closed_trades(
    records: list[dict[str, Any]],
    *,
    order_source_lookup: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    for row in records:
        pnl = None
        for key in ("pnl", "net_pnl", "realized_pnl", "reward"):
            pnl = _as_float(row.get(key))
            if pnl is not None:
                break
        if pnl is None:
            continue
        side = _normalise_side(row.get("side"))
        if side is None:
            side = "buy"
        qty = _resolve_qty(row) or 1.0
        entry_price = _resolve_price(row, "entry_price", "price")
        exit_price = _resolve_price(row, "exit_price") or entry_price
        if entry_price is None:
            entry_price = 1.0
        if exit_price is None:
            exit_price = entry_price
        fee_cost, fee_source = _resolve_fee_amount_with_source(row, qty, entry_price)
        slippage_cost, slippage_source = _resolve_slippage_cost_with_source(
            row,
            qty=qty,
            side=side,
            price=entry_price,
        )
        order_id_raw = row.get("order_id")
        order_id = None if order_id_raw in (None, "") else str(order_id_raw)
        source_hint = row.get("source")
        if source_hint in (None, "") and order_id and order_source_lookup:
            source_hint = order_source_lookup.get(order_id)
        trade = _closed_trade_record(
            symbol=str(row.get("symbol", "") or "").strip().upper() or "UNKNOWN",
            side="long" if side == "buy" else "short",
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=_parse_timestamp(row.get("entry_time") or row.get("timestamp")),
            exit_time=_parse_timestamp(row.get("exit_time") or row.get("timestamp")),
            strategy=str(row.get("strategy", "") or ""),
            signal_tags=str(row.get("signal_tags", "") or ""),
            gross_pnl=pnl,
            fee_cost=fee_cost,
            slippage_cost=slippage_cost,
            fill_source=None if source_hint in (None, "") else str(source_hint),
        )
        trade["_fee_source"] = fee_source
        trade["_slippage_source"] = slippage_source
        trades.append(trade)
    return trades


def _as_tca_leg_event(row: dict[str, Any]) -> _TCALegEvent | None:
    symbol = str(row.get("symbol", "") or "").strip().upper()
    if not symbol:
        return None
    side = _normalise_side(row.get("side"))
    if side is None:
        return None
    qty = _resolve_qty(row)
    price = _resolve_price(
        row,
        "fill_price",
        "fill_vwap",
        "price",
        "entry_price",
        "submit_price_reference",
    )
    timestamp = _parse_timestamp(
        row.get("first_fill_ts")
        or row.get("filled_at")
        or row.get("executed_at")
        or row.get("ts")
        or row.get("submit_ts")
    )
    if qty is None or price is None or timestamp is None:
        return None
    status = _normalise_status_token(row.get("status"))
    if status in {"rejected", "canceled", "cancelled", "expired", "done_for_day"}:
        return None
    if status and status not in {"filled", "partially_filled"}:
        return None
    fee_cost, _ = _resolve_fee_amount_with_source(row, qty, price)
    slippage_cost, _ = _resolve_slippage_cost_with_source(
        row,
        qty=qty,
        side=side,
        price=price,
    )
    return _TCALegEvent(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        timestamp=timestamp,
        fee_cost=fee_cost,
        slippage_cost=slippage_cost,
    )


def _pick_tca_event(
    events: list[_TCALegEvent],
    *,
    target_ts: datetime | None,
    qty: float,
    price: float,
    max_delta_seconds: float = 300.0,
) -> _TCALegEvent | None:
    if target_ts is None or not events:
        return None
    best_idx: int | None = None
    best_score: tuple[float, float, float, int] | None = None
    for idx, event in enumerate(events):
        delta = abs((event.timestamp - target_ts).total_seconds())
        if delta > max_delta_seconds:
            continue
        score = (delta, abs(event.qty - qty), abs(event.price - price), idx)
        if best_score is None or score < best_score:
            best_idx = idx
            best_score = score
    if best_idx is None:
        return None
    return events.pop(best_idx)


def _enrich_direct_trades_with_tca_costs(
    direct_trades: list[dict[str, Any]],
    *,
    tca_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[tuple[str, str], list[_TCALegEvent]] = defaultdict(list)
    for row in tca_records:
        event = _as_tca_leg_event(row)
        if event is None:
            continue
        buckets[(event.symbol, event.side)].append(event)
    for events in buckets.values():
        events.sort(key=lambda item: item.timestamp)

    matched_entry_legs = 0
    matched_exit_legs = 0
    enriched_trades = 0
    trades_with_fee = 0
    trades_with_slippage = 0
    for trade in direct_trades:
        symbol = str(trade.get("symbol", "") or "").strip().upper()
        side = str(trade.get("side", "") or "").strip().lower()
        if not symbol or side not in {"long", "short"}:
            continue
        qty = abs(float(trade.get("qty", 0.0) or 0.0))
        entry_price = abs(float(trade.get("entry_price", 0.0) or 0.0))
        exit_price = abs(float(trade.get("exit_price", 0.0) or 0.0))
        entry_ts = _parse_timestamp(trade.get("entry_time"))
        exit_ts = _parse_timestamp(trade.get("exit_time"))
        if qty <= 0 or entry_price <= 0:
            continue

        entry_side = "buy" if side == "long" else "sell"
        exit_side = "sell" if entry_side == "buy" else "buy"
        entry_event = _pick_tca_event(
            buckets.get((symbol, entry_side), []),
            target_ts=entry_ts,
            qty=qty,
            price=entry_price,
        )
        exit_event = _pick_tca_event(
            buckets.get((symbol, exit_side), []),
            target_ts=exit_ts,
            qty=qty,
            price=exit_price if exit_price > 0 else entry_price,
        )

        fee_add = 0.0
        slippage_add = 0.0
        if entry_event is not None:
            matched_entry_legs += 1
            scale = min(1.0, qty / entry_event.qty) if entry_event.qty > 0 else 1.0
            fee_add += entry_event.fee_cost * scale
            slippage_add += entry_event.slippage_cost * scale
        if exit_event is not None:
            matched_exit_legs += 1
            scale = min(1.0, qty / exit_event.qty) if exit_event.qty > 0 else 1.0
            fee_add += exit_event.fee_cost * scale
            slippage_add += exit_event.slippage_cost * scale
        if entry_event is None and exit_event is None:
            continue

        trade["fee_cost"] = abs(float(trade.get("fee_cost", 0.0) or 0.0)) + abs(fee_add)
        trade["slippage_cost"] = abs(float(trade.get("slippage_cost", 0.0) or 0.0)) + abs(slippage_add)
        trade["_fee_source"] = "tca_matched"
        trade["_slippage_source"] = "tca_matched"
        gross_pnl = float(trade.get("gross_pnl", 0.0) or 0.0)
        net_pnl = gross_pnl - float(trade.get("fee_cost", 0.0) or 0.0) - float(
            trade.get("slippage_cost", 0.0) or 0.0
        )
        trade["net_pnl"] = net_pnl
        entry_notional = abs(float(trade.get("entry_notional", 0.0) or 0.0))
        trade["net_edge_bps"] = (net_pnl / entry_notional * 10000.0) if entry_notional > 0 else None
        enriched_trades += 1
        if float(trade.get("fee_cost", 0.0) or 0.0) > 0:
            trades_with_fee += 1
        if float(trade.get("slippage_cost", 0.0) or 0.0) > 0:
            trades_with_slippage += 1

    total_events = sum(len(events) for events in buckets.values()) + matched_entry_legs + matched_exit_legs
    return direct_trades, {
        "enabled": True,
        "source": "tca",
        "records": int(len(tca_records)),
        "events_considered": int(total_events),
        "matched_entry_legs": int(matched_entry_legs),
        "matched_exit_legs": int(matched_exit_legs),
        "matched_legs": int(matched_entry_legs + matched_exit_legs),
        "enriched_trades": int(enriched_trades),
        "trades_with_nonzero_fee": int(trades_with_fee),
        "trades_with_nonzero_slippage": int(trades_with_slippage),
    }


def _reconstruct_closed_trades(
    events: list[_FillEvent],
) -> tuple[list[dict[str, Any]], dict[str, float], int]:
    books: dict[str, list[_OpenLot]] = defaultdict(list)
    closed: list[dict[str, Any]] = []
    for event in events:
        remaining = event.qty
        book = books[event.symbol]
        while remaining > 0 and book and book[0].side != event.side:
            lot = book[0]
            close_qty = min(remaining, lot.qty)
            if lot.side == "buy" and event.side == "sell":
                gross_pnl = (event.price - lot.price) * close_qty
                trade_side = "long"
            else:
                gross_pnl = (lot.price - event.price) * close_qty
                trade_side = "short"
            fill_source = lot.fill_source
            if fill_source != event.fill_source:
                if "reconcile_backfill" in {fill_source, event.fill_source}:
                    fill_source = "reconcile_backfill"
                elif fill_source == "unknown":
                    fill_source = event.fill_source
                elif event.fill_source == "unknown":
                    fill_source = fill_source
                else:
                    fill_source = "mixed"
            entry_fee = lot.fee_per_share * close_qty
            exit_fee = event.fee_per_share * close_qty
            entry_slippage = lot.slippage_per_share * close_qty
            exit_slippage = event.slippage_per_share * close_qty
            closed.append(
                _closed_trade_record(
                    symbol=event.symbol,
                    side=trade_side,
                    qty=close_qty,
                    entry_price=lot.price,
                    exit_price=event.price,
                    entry_time=lot.timestamp,
                    exit_time=event.timestamp,
                    strategy=lot.strategy or event.strategy,
                    signal_tags=lot.signal_tags or event.signal_tags,
                    gross_pnl=gross_pnl,
                    fee_cost=entry_fee + exit_fee,
                    slippage_cost=entry_slippage + exit_slippage,
                    fill_source=fill_source,
                )
            )
            closed[-1]["_fee_source"] = "fifo_reconstructed"
            closed[-1]["_slippage_source"] = "fifo_reconstructed"
            lot.qty -= close_qty
            remaining -= close_qty
            if lot.qty <= 0:
                book.pop(0)

        if remaining > 0:
            book.append(
                _OpenLot(
                    side=event.side,
                    qty=remaining,
                    price=event.price,
                    timestamp=event.timestamp,
                    strategy=event.strategy,
                    signal_tags=event.signal_tags,
                    fee_per_share=event.fee_per_share,
                    slippage_per_share=event.slippage_per_share,
                    fill_source=event.fill_source,
                )
            )

    open_by_symbol: dict[str, float] = {}
    open_lot_count = 0
    for symbol, lots in books.items():
        net_qty = 0.0
        for lot in lots:
            open_lot_count += 1
            qty = lot.qty if lot.side == "buy" else -lot.qty
            net_qty += qty
        if net_qty != 0:
            open_by_symbol[symbol] = net_qty
    return closed, open_by_symbol, open_lot_count


def _aggregate_closed_trades(
    *,
    records_count: int,
    source: str,
    closed_trades: list[dict[str, Any]],
    open_positions: dict[str, float],
    open_lot_count: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "records": records_count,
        "pnl_source": source,
        "pnl_records": len(closed_trades),
        "closed_trades": len(closed_trades),
        "open_lot_count": int(open_lot_count),
        "open_positions": open_positions,
    }
    if not closed_trades:
        summary["pnl_available"] = False
        return summary

    pnl_values = [float(row.get("net_pnl", 0.0) or 0.0) for row in closed_trades]
    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    profit_factor = sum(wins) / abs(sum(losses)) if losses else None

    side_totals: dict[str, float] = {}
    symbol_totals: dict[str, float] = defaultdict(float)
    strategy_totals: dict[str, float] = defaultdict(float)
    daily: dict[str, dict[str, Any]] = {}
    daily_by_fill_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    closed_trades_by_fill_source: dict[str, int] = defaultdict(int)
    pnl_by_fill_source: dict[str, float] = defaultdict(float)
    total_fee_cost = 0.0
    total_slippage_cost = 0.0
    fee_attributed_trades = 0
    slippage_attributed_trades = 0
    nonzero_fee_trades = 0
    nonzero_slippage_trades = 0
    fee_sources: dict[str, int] = defaultdict(int)
    slippage_sources: dict[str, int] = defaultdict(int)
    for row in closed_trades:
        side = str(row.get("side", "unknown") or "unknown").strip().lower()
        symbol = str(row.get("symbol", "UNKNOWN") or "UNKNOWN").strip().upper()
        strategy = str(row.get("strategy", "") or "")
        net_pnl = float(row.get("net_pnl", 0.0) or 0.0)
        gross_pnl = float(row.get("gross_pnl", 0.0) or 0.0)
        fee_cost = abs(float(row.get("fee_cost", 0.0) or 0.0))
        slippage_cost = float(row.get("slippage_cost", 0.0) or 0.0)
        fee_source = row.get("_fee_source")
        slippage_source = row.get("_slippage_source")
        notional = abs(float(row.get("entry_notional", 0.0) or 0.0))
        fill_source = _normalise_fill_source(
            row.get("fill_source") or row.get("_fill_source")
        )
        side_totals[side] = side_totals.get(side, 0.0) + net_pnl
        symbol_totals[symbol] += net_pnl
        strategy_totals[strategy] += net_pnl
        closed_trades_by_fill_source[fill_source] += 1
        pnl_by_fill_source[fill_source] += net_pnl
        total_fee_cost += fee_cost
        total_slippage_cost += slippage_cost
        if fee_source:
            key = str(fee_source)
            fee_attributed_trades += 1
            fee_sources[key] += 1
        if slippage_source:
            key = str(slippage_source)
            slippage_attributed_trades += 1
            slippage_sources[key] += 1
        if fee_cost > 0:
            nonzero_fee_trades += 1
        if abs(slippage_cost) > 0:
            nonzero_slippage_trades += 1

        day = "unknown"
        exit_ts = _parse_timestamp(row.get("exit_time"))
        entry_ts = _parse_timestamp(row.get("entry_time"))
        if exit_ts is not None:
            day = exit_ts.date().isoformat()
        elif entry_ts is not None:
            day = entry_ts.date().isoformat()
        bucket = daily.setdefault(
            day,
            {
                "date": day,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "gross_win_pnl": 0.0,
                "gross_loss_pnl": 0.0,
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
                "fee_cost": 0.0,
                "slippage_cost": 0.0,
                "entry_notional": 0.0,
            },
        )
        bucket["trades"] += 1
        bucket["gross_pnl"] += gross_pnl
        bucket["net_pnl"] += net_pnl
        bucket["fee_cost"] += fee_cost
        bucket["slippage_cost"] += slippage_cost
        bucket["entry_notional"] += notional
        if net_pnl > 0:
            bucket["wins"] += 1
            bucket["gross_win_pnl"] += net_pnl
        elif net_pnl < 0:
            bucket["losses"] += 1
            bucket["gross_loss_pnl"] += abs(net_pnl)

        source_daily = daily_by_fill_source[fill_source]
        source_bucket = source_daily.setdefault(
            day,
            {
                "date": day,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "gross_win_pnl": 0.0,
                "gross_loss_pnl": 0.0,
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
                "fee_cost": 0.0,
                "slippage_cost": 0.0,
                "entry_notional": 0.0,
            },
        )
        source_bucket["trades"] += 1
        source_bucket["gross_pnl"] += gross_pnl
        source_bucket["net_pnl"] += net_pnl
        source_bucket["fee_cost"] += fee_cost
        source_bucket["slippage_cost"] += slippage_cost
        source_bucket["entry_notional"] += notional
        if net_pnl > 0:
            source_bucket["wins"] += 1
            source_bucket["gross_win_pnl"] += net_pnl
        elif net_pnl < 0:
            source_bucket["losses"] += 1
            source_bucket["gross_loss_pnl"] += abs(net_pnl)

    def _daily_expectancy_from_buckets(payload: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key in sorted(payload):
            bucket = payload[key]
            trades = int(bucket["trades"])
            avg_net = bucket["net_pnl"] / trades if trades > 0 else 0.0
            entry_notional = float(bucket["entry_notional"])
            net_edge_bps = (
                (bucket["net_pnl"] / entry_notional * 10000.0)
                if entry_notional > 0
                else None
            )
            out.append(
                {
                    "date": key,
                    "trades": trades,
                    "gross_pnl": bucket["gross_pnl"],
                    "net_pnl": bucket["net_pnl"],
                    "avg_net_pnl": avg_net,
                    "fee_cost": bucket["fee_cost"],
                    "slippage_cost": bucket["slippage_cost"],
                    "net_edge_bps": net_edge_bps,
                }
            )
        return out

    daily_expectancy = _daily_expectancy_from_buckets(daily)
    daily_expectancy_by_fill_source = {
        source_key: _daily_expectancy_from_buckets(buckets)
        for source_key, buckets in sorted(daily_by_fill_source.items())
    }

    def _daily_trade_stats_from_buckets(
        payload: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key in sorted(payload):
            bucket = payload[key]
            trades = int(bucket["trades"])
            wins = int(bucket.get("wins", 0))
            losses = int(bucket.get("losses", 0))
            gross_win_pnl = float(bucket.get("gross_win_pnl", 0.0) or 0.0)
            gross_loss_pnl = float(bucket.get("gross_loss_pnl", 0.0) or 0.0)
            win_rate = (wins / trades) if trades > 0 else 0.0
            profit_factor = (
                gross_win_pnl / gross_loss_pnl
                if gross_loss_pnl > 0
                else None
            )
            out.append(
                {
                    "date": key,
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "gross_win_pnl": gross_win_pnl,
                    "gross_loss_pnl": gross_loss_pnl,
                    "net_pnl": float(bucket.get("net_pnl", 0.0) or 0.0),
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                }
            )
        return out

    daily_trade_stats = _daily_trade_stats_from_buckets(daily)
    daily_trade_stats_by_fill_source = {
        source_key: _daily_trade_stats_from_buckets(buckets)
        for source_key, buckets in sorted(daily_by_fill_source.items())
    }

    def _top_losses(values: dict[str, float]) -> list[dict[str, Any]]:
        ranked = sorted(
            (
                (name, total)
                for name, total in values.items()
                if name and total < 0
            ),
            key=lambda item: (item[1], item[0]),
        )
        return [
            {"name": name, "net_pnl": total}
            for name, total in ranked[:5]
        ]

    summary.update(
        {
            "pnl_available": True,
            "pnl_sum": sum(pnl_values),
            "pnl_avg": sum(pnl_values) / len(pnl_values),
            "pnl_median": median(pnl_values),
            "win_rate": len(wins) / len(pnl_values),
            "profit_factor": profit_factor,
            "total_fee_cost": total_fee_cost,
            "total_slippage_cost": total_slippage_cost,
            "cost_attribution": {
                "fee_attributed_trades": int(fee_attributed_trades),
                "slippage_attributed_trades": int(slippage_attributed_trades),
                "nonzero_fee_trades": int(nonzero_fee_trades),
                "nonzero_slippage_trades": int(nonzero_slippage_trades),
                "fee_sources": dict(sorted(fee_sources.items())),
                "slippage_sources": dict(sorted(slippage_sources.items())),
            },
            "side_totals": side_totals,
            "daily_expectancy": daily_expectancy,
            "daily_expectancy_by_fill_source": daily_expectancy_by_fill_source,
            "daily_expectancy_live": daily_expectancy_by_fill_source.get("live", []),
            "daily_expectancy_reconcile_backfill": daily_expectancy_by_fill_source.get(
                "reconcile_backfill",
                [],
            ),
            "daily_trade_stats": daily_trade_stats,
            "daily_trade_stats_by_fill_source": daily_trade_stats_by_fill_source,
            "closed_trades_by_fill_source": dict(
                sorted(closed_trades_by_fill_source.items())
            ),
            "pnl_by_fill_source": dict(sorted(pnl_by_fill_source.items())),
            "top_loss_drivers": {
                "symbols": _top_losses(symbol_totals),
                "strategies": _top_losses(strategy_totals),
            },
        }
    )
    return summary


def summarize_trade_history(path: Path, *, tca_path: Path | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "records": 0,
    }
    if tca_path is not None:
        summary["tca_path"] = str(tca_path)
        summary["tca_exists"] = bool(tca_path.exists())
    if not path.exists():
        return summary

    records = _load_trade_rows(path)
    summary["records"] = len(records)
    if not records:
        summary["pnl_available"] = False
        return summary

    order_events_path = path.parent / "order_events.jsonl"
    order_source_lookup: dict[str, str] = {}
    if order_events_path.exists():
        order_source_lookup = _build_order_source_lookup(order_events_path)
    summary["order_events_path"] = str(order_events_path)
    summary["order_events_exists"] = bool(order_events_path.exists())
    summary["order_source_records"] = int(len(order_source_lookup))

    direct_trades = _direct_closed_trades(
        records,
        order_source_lookup=order_source_lookup,
    )
    cost_enrichment: dict[str, Any] | None = None
    if direct_trades and tca_path is not None and tca_path.exists():
        tca_records = _load_trade_rows(tca_path)
        if tca_records:
            direct_trades, cost_enrichment = _enrich_direct_trades_with_tca_costs(
                direct_trades,
                tca_records=tca_records,
            )
    if direct_trades:
        aggregated = _aggregate_closed_trades(
            records_count=len(records),
            source="direct_pnl_rows",
            closed_trades=direct_trades,
            open_positions={},
            open_lot_count=0,
        )
        if cost_enrichment is not None:
            aggregated["cost_enrichment"] = cost_enrichment
        summary.update(aggregated)
        return summary

    events = _extract_fill_events(records, order_source_lookup=order_source_lookup)
    if not events:
        summary["pnl_available"] = False
        return summary
    closed_trades, open_positions, open_lot_count = _reconstruct_closed_trades(events)
    summary.update(
        _aggregate_closed_trades(
            records_count=len(records),
            source="fifo_reconstructed_from_fills",
            closed_trades=closed_trades,
            open_positions=open_positions,
            open_lot_count=open_lot_count,
        )
    )
    return summary


def _top_negative_attr(
    payload: dict[str, Any],
    key: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    source = payload.get(key, {})
    if not isinstance(source, dict):
        return []
    ranked: list[tuple[str, float, int]] = []
    for name, value in source.items():
        if not isinstance(value, dict):
            continue
        score = _as_float(value.get("expected_net_edge_bps_sum"))
        if score is None or score >= 0:
            continue
        count = _as_int(value.get("count")) or 0
        ranked.append((str(name), score, count))
    ranked.sort(key=lambda item: (item[1], -item[2], item[0]))
    return [
        {"name": name, "expected_net_edge_bps_sum": score, "count": count}
        for name, score, count in ranked[:limit]
    ]


def _summarize_gate_effectiveness_daily(path: Path) -> list[dict[str, Any]]:
    rows = _load_json_lines(path)
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        ts = _parse_timestamp(row.get("ts"))
        if ts is None:
            continue
        day = ts.date().isoformat()
        total_records = max(0, _as_int(row.get("records_total")) or 0)
        accepted_records = max(0, _as_int(row.get("accepted_records")) or 0)
        rejected_raw = _as_int(row.get("rejected_records"))
        rejected_records = (
            max(0, rejected_raw)
            if rejected_raw is not None
            else max(0, total_records - accepted_records)
        )
        expected_edge = _as_float(row.get("total_expected_net_edge_bps")) or 0.0
        bucket = buckets.setdefault(
            day,
            {
                "date": day,
                "total_records": 0,
                "accepted_records": 0,
                "rejected_records": 0,
                "total_expected_net_edge_bps": 0.0,
            },
        )
        bucket["total_records"] += int(total_records)
        bucket["accepted_records"] += int(accepted_records)
        bucket["rejected_records"] += int(rejected_records)
        bucket["total_expected_net_edge_bps"] += float(expected_edge)
    out: list[dict[str, Any]] = []
    for day in sorted(buckets.keys()):
        bucket = buckets[day]
        total_records = int(bucket["total_records"])
        accepted_records = int(bucket["accepted_records"])
        acceptance_rate = (
            accepted_records / total_records if total_records > 0 else None
        )
        out.append(
            {
                "date": day,
                "total_records": total_records,
                "accepted_records": accepted_records,
                "rejected_records": int(bucket["rejected_records"]),
                "acceptance_rate": acceptance_rate,
                "total_expected_net_edge_bps": float(
                    bucket["total_expected_net_edge_bps"]
                ),
            }
        )
    return out


def summarize_gate_effectiveness(
    path: Path,
    *,
    gate_log_path: Path | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    resolved_gate_log = (
        Path(gate_log_path)
        if gate_log_path is not None
        else path.parent / "gate_effectiveness.jsonl"
    )
    summary["gate_log_path"] = str(resolved_gate_log)
    summary["gate_log_exists"] = bool(resolved_gate_log.exists())
    summary["daily_gate_stats"] = []
    if resolved_gate_log.exists():
        try:
            summary["daily_gate_stats"] = _summarize_gate_effectiveness_daily(
                resolved_gate_log
            )
        except Exception:
            summary["daily_gate_stats"] = []
    if not path.exists():
        return summary

    payload = _load_json(path)
    if not isinstance(payload, dict):
        summary["valid"] = False
        return summary

    total_records = int(payload.get("total_records", 0) or 0)
    accepted_records = int(payload.get("total_accepted_records", 0) or 0)
    rejected_records = int(payload.get("total_rejected_records", 0) or 0)
    gate_totals = payload.get("gate_totals", {})
    top_gates: list[dict[str, Any]] = []
    if isinstance(gate_totals, dict):
        ranked = sorted(
            ((str(name), int(count or 0)) for name, count in gate_totals.items()),
            key=lambda item: (-item[1], item[0]),
        )
        top_gates = [{"gate": name, "count": count} for name, count in ranked[:10]]

    acceptance_rate = 0.0
    if total_records > 0:
        acceptance_rate = accepted_records / total_records

    summary.update(
        {
            "valid": True,
            "total_records": total_records,
            "accepted_records": accepted_records,
            "rejected_records": rejected_records,
            "acceptance_rate": acceptance_rate,
            "total_expected_net_edge_bps": _as_float(
                payload.get("total_expected_net_edge_bps")
            ),
            "top_gates": top_gates,
            "top_negative_gates": _top_negative_attr(payload, "gate_attribution"),
            "top_negative_symbols": _top_negative_attr(payload, "symbol_attribution"),
            "top_negative_regimes": _top_negative_attr(payload, "regime_attribution"),
        }
    )
    return summary


def build_report(
    *,
    trade_history_path: Path,
    gate_summary_path: Path,
    tca_path: Path | None = None,
    gate_log_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "trade_history": summarize_trade_history(trade_history_path, tca_path=tca_path),
        "gate_effectiveness": summarize_gate_effectiveness(
            gate_summary_path,
            gate_log_path=gate_log_path,
        ),
    }


def evaluate_go_no_go(
    report: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_map = dict(thresholds or {})

    min_closed_trades = max(0, _as_int(threshold_map.get("min_closed_trades")) or 20)
    min_profit_factor = _as_float(threshold_map.get("min_profit_factor"))
    if min_profit_factor is None:
        min_profit_factor = 1.1
    min_win_rate = _as_float(threshold_map.get("min_win_rate"))
    if min_win_rate is None:
        min_win_rate = 0.5
    min_win_rate = max(0.0, min(1.0, float(min_win_rate)))
    min_net_pnl = _as_float(threshold_map.get("min_net_pnl"))
    if min_net_pnl is None:
        min_net_pnl = 0.0
    min_acceptance_rate = _as_float(threshold_map.get("min_acceptance_rate"))
    if min_acceptance_rate is None:
        min_acceptance_rate = 0.05
    min_acceptance_rate = max(0.0, min(1.0, float(min_acceptance_rate)))
    min_expected_net_edge_bps = _as_float(
        threshold_map.get("min_expected_net_edge_bps")
    )
    if min_expected_net_edge_bps is None:
        min_expected_net_edge_bps = -50.0
    min_used_days = _as_int(threshold_map.get("min_used_days"))
    if min_used_days is None:
        min_used_days = 0
    min_used_days = max(0, int(min_used_days))
    lookback_days = _as_int(threshold_map.get("lookback_days"))
    if lookback_days is None:
        lookback_days = 0
    lookback_days = max(0, int(lookback_days))
    require_pnl_available = bool(
        threshold_map.get("require_pnl_available", True)
    )
    require_gate_valid = bool(threshold_map.get("require_gate_valid", False))

    trade = report.get("trade_history", {})
    if not isinstance(trade, Mapping):
        trade = {}
    gate = report.get("gate_effectiveness", {})
    if not isinstance(gate, Mapping):
        gate = {}

    pnl_available = bool(trade.get("pnl_available"))
    closed_trades = _as_int(trade.get("closed_trades"))
    if closed_trades is None:
        closed_trades = _as_int(trade.get("pnl_records")) or 0
    profit_factor = _as_float(trade.get("profit_factor"))
    win_rate = _as_float(trade.get("win_rate")) or 0.0
    net_pnl = _as_float(trade.get("pnl_sum")) or 0.0
    gate_valid = bool(gate.get("valid"))
    acceptance_rate = _as_float(gate.get("acceptance_rate"))
    expected_net_edge_bps = _as_float(gate.get("total_expected_net_edge_bps"))
    trade_metric_scope: dict[str, Any] = {"mode": "full_history"}
    gate_metric_scope: dict[str, Any] = {"mode": "full_history"}
    trade_used_days = 0
    gate_used_days = 0

    if lookback_days > 0:
        recent_trade_rows, trade_available_days = _select_recent_daily_rows(
            trade.get("daily_trade_stats"),
            lookback_days=lookback_days,
        )
        if recent_trade_rows:
            trade_used_days = int(len(recent_trade_rows))
            closed_trades = sum(
                max(0, _as_int(row.get("trades")) or 0)
                for row in recent_trade_rows
            )
            wins = sum(
                max(0, _as_int(row.get("wins")) or 0)
                for row in recent_trade_rows
            )
            net_pnl = sum(_as_float(row.get("net_pnl")) or 0.0 for row in recent_trade_rows)
            gross_win_pnl = sum(
                max(0.0, _as_float(row.get("gross_win_pnl")) or 0.0)
                for row in recent_trade_rows
            )
            gross_loss_pnl = sum(
                max(0.0, _as_float(row.get("gross_loss_pnl")) or 0.0)
                for row in recent_trade_rows
            )
            win_rate = (wins / closed_trades) if closed_trades > 0 else 0.0
            profit_factor = (
                gross_win_pnl / gross_loss_pnl
                if gross_loss_pnl > 0
                else None
            )
            pnl_available = bool(trade.get("pnl_available")) and closed_trades > 0
            trade_metric_scope = {
                "mode": "rolling_days",
                "lookback_days": int(lookback_days),
                "available_days": int(trade_available_days),
                "used_days": trade_used_days,
                "start_date": str(recent_trade_rows[0].get("date")),
                "end_date": str(recent_trade_rows[-1].get("date")),
            }
        else:
            trade_metric_scope = {
                "mode": "full_history_fallback",
                "lookback_days": int(lookback_days),
                "reason": "daily_trade_stats_unavailable",
            }

        recent_gate_rows, gate_available_days = _select_recent_daily_rows(
            gate.get("daily_gate_stats"),
            lookback_days=lookback_days,
        )
        if recent_gate_rows:
            gate_used_days = int(len(recent_gate_rows))
            total_records_window = sum(
                max(0, _as_int(row.get("total_records")) or 0)
                for row in recent_gate_rows
            )
            accepted_records_window = sum(
                max(0, _as_int(row.get("accepted_records")) or 0)
                for row in recent_gate_rows
            )
            acceptance_rate = (
                (accepted_records_window / total_records_window)
                if total_records_window > 0
                else None
            )
            expected_net_edge_bps = sum(
                _as_float(row.get("total_expected_net_edge_bps")) or 0.0
                for row in recent_gate_rows
            )
            gate_metric_scope = {
                "mode": "rolling_days",
                "lookback_days": int(lookback_days),
                "available_days": int(gate_available_days),
                "used_days": gate_used_days,
                "start_date": str(recent_gate_rows[0].get("date")),
                "end_date": str(recent_gate_rows[-1].get("date")),
                "window_total_records": int(total_records_window),
                "window_accepted_records": int(accepted_records_window),
            }
        else:
            gate_metric_scope = {
                "mode": "full_history_fallback",
                "lookback_days": int(lookback_days),
                "reason": "daily_gate_stats_unavailable",
            }
    else:
        trade_rows = trade.get("daily_trade_stats")
        if isinstance(trade_rows, list):
            trade_used_days = len({str(row.get("date")) for row in trade_rows if isinstance(row, Mapping)})
        gate_rows = gate.get("daily_gate_stats")
        if isinstance(gate_rows, list):
            gate_used_days = len({str(row.get("date")) for row in gate_rows if isinstance(row, Mapping)})

    enforce_used_days = bool(lookback_days > 0 and min_used_days > 0)

    checks = {
        "pnl_available": (pnl_available if require_pnl_available else True),
        "trade_used_days": (
            int(trade_used_days) >= int(min_used_days)
            if enforce_used_days
            else True
        ),
        "gate_used_days": (
            int(gate_used_days) >= int(min_used_days)
            if enforce_used_days
            else True
        ),
        "closed_trades": int(closed_trades) >= int(min_closed_trades),
        "profit_factor": (
            (profit_factor is not None and profit_factor >= float(min_profit_factor))
            if require_pnl_available
            else True
        ),
        "win_rate": (
            (float(win_rate) >= float(min_win_rate))
            if require_pnl_available
            else True
        ),
        "net_pnl": (
            (float(net_pnl) >= float(min_net_pnl))
            if require_pnl_available
            else True
        ),
        "gate_valid": (gate_valid if require_gate_valid else True),
        "acceptance_rate": (
            (acceptance_rate is not None and acceptance_rate >= float(min_acceptance_rate))
            if gate_valid
            else (not require_gate_valid)
        ),
        "expected_net_edge_bps": (
            (
                expected_net_edge_bps is not None
                and expected_net_edge_bps >= float(min_expected_net_edge_bps)
            )
            if gate_valid
            else (not require_gate_valid)
        ),
    }

    failed_checks = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "gate_passed": all(bool(value) for value in checks.values()),
        "checks": checks,
        "failed_checks": failed_checks,
        "thresholds": {
            "min_closed_trades": int(min_closed_trades),
            "min_profit_factor": float(min_profit_factor),
            "min_win_rate": float(min_win_rate),
            "min_net_pnl": float(min_net_pnl),
            "min_acceptance_rate": float(min_acceptance_rate),
            "min_expected_net_edge_bps": float(min_expected_net_edge_bps),
            "min_used_days": int(min_used_days),
            "lookback_days": int(lookback_days),
            "require_pnl_available": bool(require_pnl_available),
            "require_gate_valid": bool(require_gate_valid),
        },
        "observed": {
            "pnl_available": pnl_available,
            "trade_used_days": int(trade_used_days),
            "gate_used_days": int(gate_used_days),
            "closed_trades": int(closed_trades),
            "profit_factor": profit_factor,
            "win_rate": float(win_rate),
            "net_pnl": float(net_pnl),
            "gate_valid": gate_valid,
            "acceptance_rate": acceptance_rate,
            "expected_net_edge_bps": expected_net_edge_bps,
            "trade_metric_scope": trade_metric_scope,
            "gate_metric_scope": gate_metric_scope,
        },
    }


def format_text_report(report: dict[str, Any]) -> str:
    trade = report.get("trade_history", {})
    gate = report.get("gate_effectiveness", {})
    lines = [
        "Runtime Performance Report",
        f"- Trade history file: {trade.get('path')} (exists={trade.get('exists')})",
        f"- Gate summary file: {gate.get('path')} (exists={gate.get('exists')})",
    ]

    if trade.get("pnl_available"):
        lines.extend(
            [
                f"- Trade records: {trade.get('records')} (realized={trade.get('pnl_records')} source={trade.get('pnl_source')})",
                f"- Realized net pnl sum: {trade.get('pnl_sum'):.4f}",
                f"- Win rate: {trade.get('win_rate'):.2%}",
                f"- Profit factor: {trade.get('profit_factor')}",
                f"- Total fee cost: {float(trade.get('total_fee_cost', 0.0) or 0.0):.4f}",
                f"- Total slippage cost: {float(trade.get('total_slippage_cost', 0.0) or 0.0):.4f}",
                f"- Open lots (unrealized): {trade.get('open_lot_count')}",
            ]
        )
        cost_attr = trade.get("cost_attribution")
        if isinstance(cost_attr, Mapping):
            lines.append(
                "- Cost attribution coverage: "
                f"fees={cost_attr.get('fee_attributed_trades')}/{trade.get('closed_trades')} "
                f"slippage={cost_attr.get('slippage_attributed_trades')}/{trade.get('closed_trades')}"
            )
        cost_enrichment = trade.get("cost_enrichment")
        if isinstance(cost_enrichment, Mapping):
            lines.append(
                "- TCA enrichment: "
                f"matched_legs={cost_enrichment.get('matched_legs')} "
                f"enriched_trades={cost_enrichment.get('enriched_trades')}"
            )
        fill_source_counts = trade.get("closed_trades_by_fill_source")
        if isinstance(fill_source_counts, Mapping) and fill_source_counts:
            lines.append(f"- Closed trades by fill source: {dict(fill_source_counts)}")
        top_symbols = (trade.get("top_loss_drivers") or {}).get("symbols", [])
        if top_symbols:
            lines.append("- Top loss symbols:")
            for item in top_symbols:
                lines.append(f"  - {item.get('name')}: {item.get('net_pnl'):.4f}")
        daily = trade.get("daily_expectancy", [])
        if daily:
            lines.append("- Daily expectancy (latest 5):")
            for item in daily[-5:]:
                lines.append(
                    "  - "
                    f"{item.get('date')}: trades={item.get('trades')} "
                    f"net_pnl={item.get('net_pnl'):.4f} "
                    f"net_edge_bps={item.get('net_edge_bps')}"
                )
        source_daily = trade.get("daily_expectancy_by_fill_source")
        if isinstance(source_daily, Mapping) and source_daily:
            live_rows = source_daily.get("live")
            if isinstance(live_rows, list) and live_rows:
                latest_live = live_rows[-1]
                lines.append(
                    "- Daily expectancy live (latest): "
                    f"{latest_live.get('date')} trades={latest_live.get('trades')} "
                    f"net_pnl={latest_live.get('net_pnl'):.4f} "
                    f"net_edge_bps={latest_live.get('net_edge_bps')}"
                )
            reconcile_rows = source_daily.get("reconcile_backfill")
            if isinstance(reconcile_rows, list) and reconcile_rows:
                latest_reconcile = reconcile_rows[-1]
                lines.append(
                    "- Daily expectancy reconcile (latest): "
                    f"{latest_reconcile.get('date')} trades={latest_reconcile.get('trades')} "
                    f"net_pnl={latest_reconcile.get('net_pnl'):.4f} "
                    f"net_edge_bps={latest_reconcile.get('net_edge_bps')}"
                )
    else:
        lines.append("- Realized pnl: unavailable (no usable closed-trade records found)")

    if gate.get("valid"):
        lines.extend(
            [
                f"- Decisions: total={gate.get('total_records')} accepted={gate.get('accepted_records')} rejected={gate.get('rejected_records')}",
                f"- Acceptance rate: {gate.get('acceptance_rate'):.2%}",
                f"- Expected net edge sum (bps): {gate.get('total_expected_net_edge_bps')}",
            ]
        )
        top_gates = gate.get("top_gates", [])
        if top_gates:
            lines.append("- Top gate counts:")
            for item in top_gates:
                lines.append(f"  - {item.get('gate')}: {item.get('count')}")
        top_negative_gates = gate.get("top_negative_gates", [])
        if top_negative_gates:
            lines.append("- Worst expected edge gates:")
            for item in top_negative_gates:
                lines.append(
                    "  - "
                    f"{item.get('name')}: "
                    f"{item.get('expected_net_edge_bps_sum')} bps"
                )

    go_no_go = report.get("go_no_go")
    if isinstance(go_no_go, Mapping):
        lines.append(f"- Go/No-Go gate passed: {bool(go_no_go.get('gate_passed'))}")
        failed = go_no_go.get("failed_checks", [])
        if isinstance(failed, list) and failed:
            lines.append(f"- Failed checks: {', '.join(str(item) for item in failed)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize runtime trading performance artifacts."
    )
    parser.add_argument(
        "--trade-history",
        default=_DEFAULT_TRADE_HISTORY_PATH,
        help="Path to canonical trade history (parquet/pickle/json/jsonl/csv).",
    )
    parser.add_argument(
        "--gate-summary",
        default="runtime/gate_effectiveness_summary.json",
        help="Path to gate effectiveness summary json.",
    )
    parser.add_argument(
        "--gate-log-path",
        default="",
        help="Optional path to per-cycle gate effectiveness jsonl (for rolling checks).",
    )
    parser.add_argument(
        "--tca-path",
        default=_DEFAULT_TCA_PATH,
        help="Optional path to TCA jsonl for fee/slippage enrichment.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of text.",
    )
    parser.add_argument(
        "--go-no-go",
        action="store_true",
        help="Evaluate go/no-go criteria and include decision payload.",
    )
    parser.add_argument(
        "--fail-on-no-go",
        action="store_true",
        help="Return exit code 2 when go/no-go criteria fail.",
    )
    parser.add_argument("--min-closed-trades", type=int, default=None)
    parser.add_argument("--min-profit-factor", type=float, default=None)
    parser.add_argument("--min-win-rate", type=float, default=None)
    parser.add_argument("--min-net-pnl", type=float, default=None)
    parser.add_argument("--min-acceptance-rate", type=float, default=None)
    parser.add_argument("--min-expected-net-edge-bps", type=float, default=None)
    parser.add_argument("--min-used-days", type=int, default=None)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Rolling trading-day window for go/no-go metrics; 0 or unset uses full history.",
    )
    parser.add_argument(
        "--require-gate-valid",
        action="store_true",
        help="Require gate summary validity for go/no-go.",
    )
    parser.add_argument(
        "--allow-missing-pnl",
        action="store_true",
        help="Do not require realized pnl availability for go/no-go.",
    )
    args = parser.parse_args(argv)

    report = build_report(
        trade_history_path=Path(args.trade_history),
        gate_summary_path=Path(args.gate_summary),
        tca_path=(Path(args.tca_path) if args.tca_path else None),
        gate_log_path=(Path(args.gate_log_path) if str(args.gate_log_path).strip() else None),
    )
    if args.go_no_go or args.fail_on_no_go:
        thresholds = {
            "require_pnl_available": not bool(args.allow_missing_pnl),
            "require_gate_valid": bool(args.require_gate_valid),
        }
        for key in (
            "min_closed_trades",
            "min_profit_factor",
            "min_win_rate",
            "min_net_pnl",
            "min_acceptance_rate",
            "min_expected_net_edge_bps",
            "min_used_days",
            "lookback_days",
        ):
            value = getattr(args, key)
            if value is not None:
                thresholds[key] = value
        report["go_no_go"] = evaluate_go_no_go(report, thresholds=thresholds)

    if args.json:
        sys.stdout.write(f"{json.dumps(report, indent=2, sort_keys=True)}\n")
    else:
        sys.stdout.write(f"{format_text_report(report)}\n")
    if args.fail_on_no_go:
        go_no_go = report.get("go_no_go", {})
        if isinstance(go_no_go, Mapping) and not bool(go_no_go.get("gate_passed")):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
