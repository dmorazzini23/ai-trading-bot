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

_DEFAULT_TRADE_HISTORY_PATH = "artifacts/trade_history.parquet"


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


def _resolve_fee_amount(row: dict[str, Any], qty: float, price: float) -> float:
    for key in ("fee_amount", "fee", "fees", "commission", "commission_amount"):
        fee = _as_float(row.get(key))
        if fee is not None:
            return abs(fee)
    fee_bps = _as_float(row.get("fee_bps"))
    if fee_bps is not None and fee_bps > 0:
        return abs(qty * price * (fee_bps / 10000.0))
    return 0.0


def _resolve_slippage_bps(row: dict[str, Any], side: str, price: float) -> float:
    slippage_bps = _as_float(row.get("slippage_bps"))
    if slippage_bps is not None:
        return slippage_bps
    expected = _resolve_price(row, "expected_price")
    if expected is None or expected <= 0:
        return 0.0
    if side == "buy":
        return ((price - expected) / expected) * 10000.0
    return ((expected - price) / expected) * 10000.0


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


def _as_fill_event(row: dict[str, Any]) -> _FillEvent | None:
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
    )


def _extract_fill_events(records: list[dict[str, Any]]) -> list[_FillEvent]:
    events: list[tuple[datetime, int, _FillEvent]] = []
    for index, row in enumerate(records):
        event = _as_fill_event(row)
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
    }


def _direct_closed_trades(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    for row in records:
        pnl = _as_float(row.get("pnl"))
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
        fee_cost = _resolve_fee_amount(row, qty, entry_price)
        slippage_cost = _as_float(row.get("slippage_cost"))
        if slippage_cost is None:
            slippage_bps = _resolve_slippage_bps(row, side, entry_price)
            slippage_cost = abs(qty * entry_price * (slippage_bps / 10000.0))
        trades.append(
            _closed_trade_record(
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
            )
        )
    return trades


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
                )
            )
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
    for row in closed_trades:
        side = str(row.get("side", "unknown") or "unknown").strip().lower()
        symbol = str(row.get("symbol", "UNKNOWN") or "UNKNOWN").strip().upper()
        strategy = str(row.get("strategy", "") or "")
        net_pnl = float(row.get("net_pnl", 0.0) or 0.0)
        gross_pnl = float(row.get("gross_pnl", 0.0) or 0.0)
        fee_cost = abs(float(row.get("fee_cost", 0.0) or 0.0))
        slippage_cost = float(row.get("slippage_cost", 0.0) or 0.0)
        notional = abs(float(row.get("entry_notional", 0.0) or 0.0))
        side_totals[side] = side_totals.get(side, 0.0) + net_pnl
        symbol_totals[symbol] += net_pnl
        strategy_totals[strategy] += net_pnl

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

    daily_expectancy: list[dict[str, Any]] = []
    for key in sorted(daily):
        bucket = daily[key]
        trades = int(bucket["trades"])
        avg_net = bucket["net_pnl"] / trades if trades > 0 else 0.0
        entry_notional = float(bucket["entry_notional"])
        net_edge_bps = (
            (bucket["net_pnl"] / entry_notional * 10000.0)
            if entry_notional > 0
            else None
        )
        daily_expectancy.append(
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
            "side_totals": side_totals,
            "daily_expectancy": daily_expectancy,
            "top_loss_drivers": {
                "symbols": _top_losses(symbol_totals),
                "strategies": _top_losses(strategy_totals),
            },
        }
    )
    return summary


def summarize_trade_history(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "records": 0,
    }
    if not path.exists():
        return summary

    records = _load_trade_rows(path)
    summary["records"] = len(records)
    if not records:
        summary["pnl_available"] = False
        return summary

    direct_trades = _direct_closed_trades(records)
    if direct_trades:
        summary.update(
            _aggregate_closed_trades(
                records_count=len(records),
                source="direct_pnl_rows",
                closed_trades=direct_trades,
                open_positions={},
                open_lot_count=0,
            )
        )
        return summary

    events = _extract_fill_events(records)
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


def summarize_gate_effectiveness(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
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
) -> dict[str, Any]:
    return {
        "trade_history": summarize_trade_history(trade_history_path),
        "gate_effectiveness": summarize_gate_effectiveness(gate_summary_path),
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

    checks = {
        "pnl_available": (pnl_available if require_pnl_available else True),
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
            "require_pnl_available": bool(require_pnl_available),
            "require_gate_valid": bool(require_gate_valid),
        },
        "observed": {
            "pnl_available": pnl_available,
            "closed_trades": int(closed_trades),
            "profit_factor": profit_factor,
            "win_rate": float(win_rate),
            "net_pnl": float(net_pnl),
            "gate_valid": gate_valid,
            "acceptance_rate": acceptance_rate,
            "expected_net_edge_bps": expected_net_edge_bps,
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
                f"- Open lots (unrealized): {trade.get('open_lot_count')}",
            ]
        )
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
