"""Read-only reconciliation of recorded decisions, orders, fills and FIFO lots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from decimal import Decimal, InvalidOperation
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.utils.market_calendar import session_info

logger = get_logger(__name__)


def _identities(row: Mapping[str, Any]) -> set[tuple[str, str]]:
    keys = set()
    containers = [row, *[row[key] for key in ("decision_journal", "order", "metrics", "tca", "metadata", "annotations", "context") if isinstance(row.get(key), Mapping)]]
    for container in containers:
        for key in ("order_id", "broker_order_id", "client_order_id", "correlation_id", "decision_trace_id", "decision_id"):
            value = container.get(key)
            if value not in (None, ""):
                keys.add(("order_id" if key == "broker_order_id" else key, str(value)))
    return keys


def _number(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _reconcile_positions(accepted: list[dict[str, Any]], opening: Mapping[str, Any] | None, closing: Mapping[str, Any] | None) -> dict[str, Any]:
    if opening is None or closing is None:
        return {"status": "unavailable", "reason": "time_aligned_opening_and_closing_broker_position_snapshots_required"}
    if opening.get("positions_complete") is False or closing.get("positions_complete") is False:
        return {"status": "unavailable", "reason": "incomplete_position_snapshot"}
    start = pd.to_datetime(opening.get("timestamp"), utc=True, errors="coerce")
    end = pd.to_datetime(closing.get("timestamp"), utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("position snapshots require increasing valid timestamps")
    if not opening.get("account_id") or opening.get("account_id") != closing.get("account_id") or opening.get("trading_mode") not in ("paper", "live") or opening.get("trading_mode") != closing.get("trading_mode"):
        raise ValueError("position snapshots must identify the same account and trading mode")
    if not isinstance(opening.get("positions"), Mapping) or not isinstance(closing.get("positions"), Mapping):
        raise ValueError("position snapshots require symbol-to-quantity mappings")
    expected = {str(symbol): _number(qty) for symbol, qty in opening["positions"].items()}
    observed = {str(symbol): _number(qty) for symbol, qty in closing["positions"].items()}
    if any(qty is None for qty in [*expected.values(), *observed.values()]):
        raise ValueError("position snapshot quantities must be finite")
    net: dict[str, Decimal] = {symbol: qty for symbol, qty in expected.items() if qty is not None}
    included = 0
    for row in accepted:
        if start < row["timestamp"] <= end:
            included += 1
            net[row["symbol"]] = net.get(row["symbol"], Decimal(0)) + (row["qty"] if row["side"] == "buy" else -row["qty"])
    differences = {symbol: str(net.get(symbol, Decimal(0)) - (observed.get(symbol) or Decimal(0))) for symbol in set(net) | set(observed) if net.get(symbol, Decimal(0)) != (observed.get(symbol) or Decimal(0))}
    return {"status": "mismatched" if differences else "matched", "differences": differences, "included_fills": included, "opening_timestamp": start.isoformat(), "closing_timestamp": end.isoformat(), "trading_mode": opening["trading_mode"], "scope": "quantity_arithmetic_only_not_execution_feed_completeness"}


def _order_quantity_reconciliation(orders: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[str, dict[str, Any]] = {}
    for row in orders:
        key = str(row.get("order_id") or row.get("client_order_id") or "")
        timestamp = pd.to_datetime(row.get("ts"), utc=True, errors="coerce")
        qty = _number(row.get("filled_qty"))
        if key and not pd.isna(timestamp) and qty is not None and qty >= 0:
            if key not in latest or timestamp >= latest[key]["timestamp"]:
                latest[key] = {"timestamp": timestamp, "qty": qty}
    totals: dict[str, Decimal] = defaultdict(Decimal)
    last_fill: dict[str, Any] = {}
    for row in accepted:
        key = row["quantity_key"]
        totals[key] += row["qty"]
        last_fill[key] = max(last_fill.get(key, row["timestamp"]), row["timestamp"])
    counts: Counter[str] = Counter()
    mismatches = []
    keys = set(totals) | {key for key, row in latest.items() if row["qty"] > 0}
    for key in sorted(keys):
        snapshot = latest.get(key)
        if snapshot is None:
            counts["cumulative_order_quantity_missing"] += 1
        elif key in last_fill and snapshot["timestamp"] < last_fill[key]:
            counts["order_snapshot_precedes_last_fill"] += 1
        elif snapshot["qty"] != totals[key]:
            counts["quantity_mismatch"] += 1
            mismatches.append({"order_identity": key, "recorded_cumulative_qty": str(snapshot["qty"]), "observed_fill_qty": str(totals[key])})
        else:
            counts["quantity_matched"] += 1
    return {"counts": dict(counts), "mismatches": mismatches, "scope": "latest_nonstale_recorded_cumulative_quantity_vs_unique_fill_sum"}


def reconcile_evidence(*, decisions: list[dict[str, Any]], orders: list[dict[str, Any]], fills: list[dict[str, Any]], tca: list[dict[str, Any]], opening_positions: Mapping[str, Any] | None = None, closing_positions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    decision_keys = set().union(*(_identities(row) for row in decisions))
    order_keys = set().union(*(_identities(row) for row in orders))
    tca_keys = set().union(*(_identities(row) for row in tca if not row.get("pending_event") and row.get("fill_price") is not None))
    counts: Counter[str] = Counter()
    unique: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for row in fills:
        fill_id = str(row.get("fill_id") or "")
        if not fill_id:
            counts["missing_fill_id"] += 1
            continue
        if fill_id in unique:
            fields = ("symbol", "side", "fill_qty", "fill_price", "ts", "fee_amount", "fee_bps", "fee_source", "order_id")
            if all(row.get(key) == unique[fill_id].get(key) for key in fields):
                counts["duplicate_fill_rows"] += 1
            else:
                ambiguous.add(fill_id)
                counts["conflicting_fill_rows"] += 1
            continue
        unique[fill_id] = row
    accepted = []
    for fill_id, row in unique.items():
        if fill_id in ambiguous:
            continue
        qty, price = _number(row.get("fill_qty")), _number(row.get("fill_price"))
        timestamp = pd.to_datetime(row.get("ts"), errors="coerce", utc=True)
        if qty is None or price is None or qty <= 0 or price <= 0 or pd.isna(timestamp) or not row.get("symbol") or row.get("side") not in ("buy", "sell"):
            counts["invalid_fill_rows"] += 1
            continue
        identities = _identities(row)
        for name, source_keys in (("decision", decision_keys), ("order", order_keys), ("tca", tca_keys)):
            counts[f"{name}_matched" if identities & source_keys else f"{name}_unmatched"] += 1
        fee = _number(row.get("fee_amount"))
        fee_bps = _number(row.get("fee_bps"))
        fee_source = str(row.get("fee_source") or "legacy_unverified")
        counts[f"fee_source_{fee_source}"] += 1
        if fee is None and fee_bps is not None:
            fee = qty * price * fee_bps / Decimal(10000)
        if fee_source not in {"broker_payload", "broker_activity"}:
            if fee_source == "configured_estimate":
                counts["fees_estimated"] += 1
            fee = None
        counts["fees_known" if fee is not None else "fees_missing"] += 1
        accepted.append({"fill_id": fill_id, "quantity_key": str(row.get("order_id") or row.get("client_order_id") or ""), "symbol": str(row["symbol"]), "side": row["side"], "qty": qty, "price": price, "fee_per_share": fee / qty if fee is not None else None, "timestamp": timestamp})
    lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    net_change: dict[str, Decimal] = defaultdict(Decimal)
    unmatched_sells: dict[str, Decimal] = defaultdict(Decimal)
    links = []
    for row in sorted(accepted, key=lambda item: (item["timestamp"], item["fill_id"])):
        symbol, qty = row["symbol"], row["qty"]
        net_change[symbol] += qty if row["side"] == "buy" else -qty
        if row["side"] == "buy":
            lots[symbol].append(dict(row))
            continue
        while qty > 0 and lots[symbol]:
            entry = lots[symbol][0]
            matched = min(qty, entry["qty"])
            gross = matched * (row["price"] - entry["price"])
            fees_known = entry["fee_per_share"] is not None and row["fee_per_share"] is not None
            net = gross - matched * (entry["fee_per_share"] + row["fee_per_share"]) if fees_known else None
            links.append({"symbol": symbol, "entry_fill_id": entry["fill_id"], "exit_fill_id": row["fill_id"], "entry_timestamp": entry["timestamp"].isoformat(), "exit_timestamp": row["timestamp"].isoformat(), "qty": str(matched), "gross_pnl": str(gross), "net_pnl": str(net) if net is not None else None, "link_method": "inferred_fifo_observed_fills_not_broker_lot_confirmation"})
            entry["qty"] -= matched
            qty -= matched
            if entry["qty"] == 0:
                lots[symbol].popleft()
        unmatched_sells[symbol] += qty
    return {
        "artifact_type": "execution_evidence_reconciliation", "rows": {"decisions": len(decisions), "orders": len(orders), "fills": len(fills), "tca": len(tca)},
        "accepted_unique_fills": len(accepted), "conflicting_fill_ids": len(ambiguous), "counts": dict(counts),
        "observed_position_changes": {symbol: str(qty) for symbol, qty in net_change.items()},
        "unmatched_sell_quantity": {symbol: str(qty) for symbol, qty in unmatched_sells.items() if qty},
        "unclosed_observed_buy_quantity": {symbol: str(sum((lot["qty"] for lot in queue), Decimal(0))) for symbol, queue in lots.items() if queue},
        "entry_exit_links": links,
        "position_reconciliation": _reconcile_positions(accepted, opening_positions, closing_positions),
        "order_quantity_reconciliation": _order_quantity_reconciliation(orders, accepted),
        "execution_environment": "historical_environment_not_independently_proven_by_source_live_field",
        "status": "evidence_gaps" if counts["decision_unmatched"] or counts["order_unmatched"] or counts["fees_missing"] or counts["invalid_fill_rows"] or counts["missing_fill_id"] or ambiguous else "recorded_fills_linked_position_confirmation_required",
        "limitations": ["FIFO links describe observed fills and are not certified broker tax lots or strategy exits.", "An unmatched sell can indicate initial inventory, short selling, or missing history.", "Missing fees remain unknown; gross P&L is not net profitability.", "Recorded cumulative quantities and position snapshots are reconciled separately; neither alone certifies execution-feed completeness."],
        "promotion_authority": False, "orders_sent": 0,
    }


def reconcile_session(*, session_date: str, account_id: str, boundaries: list[dict[str, Any]], decisions: list[dict[str, Any]], orders: list[dict[str, Any]], fills: list[dict[str, Any]], tca: list[dict[str, Any]]) -> dict[str, Any]:
    """Require a complete paper-session boundary pair; never infer absent snapshots."""
    session = session_info(date.fromisoformat(session_date))
    if session is None:
        raise ValueError("requested date is not a trading session")
    start, end = pd.Timestamp(session.start_utc), pd.Timestamp(session.end_utc)
    valid = []
    for row in boundaries:
        ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        if row.get("account_id") == account_id and row.get("trading_mode") == "paper" and row.get("positions_complete") is True and not pd.isna(ts):
            valid.append((ts, row))
    opening = max((item for item in valid if start - pd.Timedelta(minutes=15) <= item[0] <= start), key=lambda item: item[0], default=None)
    closing = min((item for item in valid if end <= item[0] <= end + pd.Timedelta(minutes=15)), key=lambda item: item[0], default=None)
    lower, upper = opening[0] if opening else start, closing[0] if closing else end
    selected_fills = []
    unknown_identity = 0
    invalid_timestamps = 0
    for row in fills:
        ts = pd.to_datetime(row.get("ts"), utc=True, errors="coerce")
        if pd.isna(ts):
            invalid_timestamps += 1
            continue
        if not lower < ts <= upper:
            continue
        if row.get("account_id") is None or row.get("trading_mode") not in {"paper", "live"}:
            unknown_identity += 1
        elif row.get("account_id") == account_id and row.get("trading_mode") == "paper":
            selected_fills.append(row)
    identities = set().union(*(_identities(row) for row in selected_fills))
    relevant_orders = [row for row in orders if _identities(row) & identities or lower < pd.to_datetime(row.get("ts"), utc=True, errors="coerce") <= upper]
    report = reconcile_evidence(decisions=decisions, orders=relevant_orders, fills=selected_fills, tca=tca, opening_positions=opening[1] if opening else None, closing_positions=closing[1] if closing else None)
    gaps = []
    if invalid_timestamps:
        gaps.append("unclassifiable_fill_timestamps")
    if not opening or not closing:
        gaps.append("complete_session_boundary_pair_missing")
    if unknown_identity:
        gaps.append("fill_account_or_mode_missing")
    if not report["accepted_unique_fills"]:
        gaps.append("no_execution_samples")
    if report["status"] == "evidence_gaps":
        gaps.append("fill_lineage_or_fee_gaps")
    if report["position_reconciliation"]["status"] != "matched":
        gaps.append("positions_not_reconciled")
    if any(name != "quantity_matched" and count for name, count in report["order_quantity_reconciliation"]["counts"].items()):
        gaps.append("order_quantities_not_reconciled")
    if report["counts"].get("tca_unmatched", 0):
        gaps.append("tca_missing")
    report.update(session_date=session_date, unknown_identity_fills=unknown_identity, unclassifiable_fill_timestamps=invalid_timestamps, session_gaps=gaps, status="evidence_gaps" if gaps else "paper_session_reconciled")
    report["boundary_evidence"] = {"opening_timestamp": opening[0].isoformat() if opening else None, "closing_timestamp": closing[0].isoformat() if closing else None, "fill_interval": "opening_exclusive_closing_inclusive", "opening_window_utc": [(start - pd.Timedelta(minutes=15)).isoformat(), start.isoformat()], "closing_window_utc": [end.isoformat(), (end + pd.Timedelta(minutes=15)).isoformat()]}
    return report


def _read(path: Path) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    rows, invalid = [], 0
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            try:
                value = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                invalid += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                invalid += 1
    after = path.stat()
    stable = (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    return rows, invalid, {"path": str(path), "sha256": digest.hexdigest(), "stable_during_read": stable, "rows": len(rows), "invalid_rows": invalid}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("decisions", "orders", "fills", "tca"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opening-positions", type=Path)
    parser.add_argument("--closing-positions", type=Path)
    parser.add_argument("--session", help="NYSE session date for complete paper-session validation")
    parser.add_argument("--account-id")
    parser.add_argument("--position-boundaries", type=Path)
    args = parser.parse_args()
    inputs, invalid, sources = {}, {}, {}
    for name in ("decisions", "orders", "fills", "tca"):
        inputs[name], invalid[name], sources[name] = _read(getattr(args, name))
    if args.session:
        if not args.account_id or not args.position_boundaries or args.opening_positions or args.closing_positions:
            parser.error("--session requires --account-id and --position-boundaries, without individual position files")
        boundaries, invalid["boundaries"], sources["boundaries"] = _read(args.position_boundaries)
        report = reconcile_session(**inputs, session_date=args.session, account_id=args.account_id, boundaries=boundaries)
    else:
        report = reconcile_evidence(**inputs, opening_positions=json.loads(args.opening_positions.read_text()) if args.opening_positions else None, closing_positions=json.loads(args.closing_positions.read_text()) if args.closing_positions else None)
    report["invalid_json_rows"] = invalid
    report["sources"] = sources
    report["generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    if any(not source["stable_during_read"] for source in sources.values()):
        report["status"] = "unstable_input_snapshot"
    elif any(invalid.values()):
        report["status"] = "invalid_input_rows"
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info("EXECUTION_EVIDENCE_RECONCILED", extra={"status": report["status"], "output": str(args.output)})


if __name__ == "__main__":
    main()
