"""Read-only paper account activity capture and execution accounting comparison."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from decimal import Decimal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ai_trading.config.management import get_env
from ai_trading.contracts import position_snapshot_from_position
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.execution_evidence_reconciliation import _number, _read


_POSITION_ACTION_TYPES = frozenset({
    "ACATS", "FOPT", "JNLS", "MA", "NC", "OPASN", "OPEXP", "OPXRC",
    "REORG", "SC", "SPLIT", "SSO", "SSP",
})
_DISTRIBUTION_ACTION_TYPES = frozenset({
    "CGD", "DIV", "DIVCGL", "DIVCGS", "DIVFEE", "DIVFT", "DIVNRA",
    "DIVROC", "DIVTW", "DIVTXEX",
})


def daily_fee_totals(fees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate booked charges by reported date/currency, never by guessed fill."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in fees:
        key = (str(row.get("date") or "unknown"), str(row.get("currency") or "unknown"))
        group = groups.setdefault(key, {"reported_date": key[0], "currency": key[1], "charges": Decimal(0), "credits": Decimal(0), "activity_count": 0, "invalid_amounts": 0})
        group["activity_count"] += 1
        amount = _number(row.get("net_amount"))
        if amount is None:
            group["invalid_amounts"] += 1
        elif amount < 0:
            group["charges"] -= amount
        else:
            group["credits"] += amount
    return [{**group, "charges": str(group["charges"]), "credits": str(group["credits"]), "net_charges": str(group["charges"] - group["credits"]), "scope": "reported_accounting_date_not_certified_trade_session", "per_fill_allocation": False} for _, group in sorted(groups.items())]


def capture(client: Any, *, after: str, max_pages: int = 100) -> dict[str, Any]:
    account = client.get_account()
    account_id = str(account.id)
    account_boundary = {
        "account_id": account_id,
        "trading_mode": "paper",
        "timestamp": datetime.now(UTC).isoformat(),
        "currency": getattr(account, "currency", None),
        "cash": str(getattr(account, "cash", "")),
        "equity": str(getattr(account, "equity", "")),
        "portfolio_value": str(getattr(account, "portfolio_value", "")),
        "source": "broker_get_account_observation",
    }
    rows: list[dict[str, Any]] = []
    tokens = set()
    token = None
    complete = False
    for _ in range(max_pages):
        params = {"after": after, "direction": "asc", "page_size": 100}
        if token:
            params["page_token"] = token
        page = client.get("/account/activities", data=params)
        if not isinstance(page, list) or any(not isinstance(row, dict) or not row.get("id") for row in page):
            raise ValueError("invalid account activities response")
        rows.extend(page)
        if len(page) < 100:
            complete = True
            break
        token = str(page[-1]["id"])
        if token in tokens:
            break
        tokens.add(token)
    return {"account_id": account_id, "trading_mode": "paper", "after": after, "fetched_at": datetime.now(UTC).isoformat(), "pagination_complete": complete, "activities": rows, "account_boundary": account_boundary}


def capture_current_position_boundary(client: Any) -> dict[str, Any]:
    """Record a complete paper position observation before activity capture."""
    account_id = str(client.get_account().id)
    positions: dict[str, str] = {}
    for row in client.get_all_positions():
        snapshot = position_snapshot_from_position(row, provider="alpaca")
        if snapshot is None or snapshot.symbol in positions:
            raise ValueError("invalid or duplicate broker position")
        positions[snapshot.symbol] = str(snapshot.qty)
    return {
        "account_id": account_id,
        "trading_mode": "paper",
        "positions_complete": True,
        "timestamp": datetime.now(UTC).isoformat(),
        "positions": positions,
        "source": "broker_get_all_positions_observation",
    }


def _aware_instant(value: Any) -> datetime | None:
    """Accept only an explicit timezone-bearing instant, never an accounting date."""
    if not isinstance(value, str) or "T" not in value:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return result.astimezone(UTC) if result.tzinfo is not None else None


def reconcile_account_equity(
    activities: dict[str, Any], opening: dict[str, Any], closing: dict[str, Any]
) -> dict[str, Any]:
    """Audit broker cash/equity boundaries without assigning charges to fills."""
    account_id = activities.get("account_id")
    mode = activities.get("trading_mode")
    start = _aware_instant(opening.get("timestamp"))
    end = _aware_instant(closing.get("timestamp"))
    after = _aware_instant(activities.get("after"))
    fetched = _aware_instant(activities.get("fetched_at"))
    if (
        not account_id or mode not in {"paper", "live"}
        or activities.get("pagination_complete") is not True
        or None in (start, end, after, fetched)
        or not after < start < end <= fetched
        or any(boundary.get("account_id") != account_id
               or boundary.get("trading_mode") != mode
               or boundary.get("currency") != "USD"
               for boundary in (opening, closing))
    ):
        raise ValueError("complete same-account USD boundaries and covering activities required")
    balances = [_number(boundary.get(field)) for boundary in (opening, closing)
                for field in ("cash", "equity")]
    if any(amount is None for amount in balances):
        raise ValueError("broker cash and equity boundaries must be finite")
    open_cash, open_equity, close_cash, close_equity = balances
    assert open_cash is not None and open_equity is not None
    assert close_cash is not None and close_equity is not None

    seen: dict[str, dict[str, Any]] = {}
    for row in activities.get("activities", []):
        if not isinstance(row, dict) or not row.get("id"):
            raise ValueError("activity identity missing")
        key = str(row["id"])
        if key in seen and seen[key] != row:
            raise ValueError("conflicting activity identity")
        seen[key] = row

    fill_cash = Decimal(0)
    other_cash = Decimal(0)
    fee_cash = Decimal(0)
    included: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    corporate_actions: list[dict[str, Any]] = []
    for key, row in sorted(seen.items()):
        if row.get("account_id") not in (None, account_id):
            raise ValueError("activity account conflict")
        if row.get("trading_mode") not in (None, mode):
            raise ValueError("activity trading-mode conflict")
        activity_type = str(row.get("activity_type") or "unknown")
        is_fill = activity_type == "FILL"
        # FEE.date is a booked day, not a causal execution timestamp.
        instant = _aware_instant(row.get("transaction_time") if is_fill else
                                 row.get("executed_at") or row.get("transaction_time"))
        if activity_type in _POSITION_ACTION_TYPES | _DISTRIBUTION_ACTION_TYPES:
            corporate_actions.append({
                "activity_id": key,
                "activity_type": activity_type,
                "kind": "position_change" if activity_type in _POSITION_ACTION_TYPES else "cash_distribution",
                "symbol": row.get("symbol"),
                "reported_date": row.get("date"),
                "effective_at": instant.isoformat() if instant is not None else None,
                "window_relation": (
                    "unplaced" if instant is None else
                    "inside" if start < instant <= end else "outside"
                ),
                "position_effect_reconciled": (
                    False if activity_type in _POSITION_ACTION_TYPES else None
                ),
            })
        if instant is None:
            unresolved.append({"activity_id": key, "reason": "effective_instant_unavailable"})
            continue
        if not start < instant <= end:
            continue
        if row.get("status") not in (None, "executed"):
            unresolved.append({"activity_id": key, "reason": "activity_not_executed"})
            continue
        if is_fill:
            quantity = _number(row.get("qty"))
            price = _number(row.get("price"))
            if quantity is None or price is None or quantity <= 0 or price <= 0 or row.get("side") not in {"buy", "sell"} or not row.get("order_id"):
                unresolved.append({"activity_id": key, "reason": "invalid_execution_cash_fields"})
                continue
            amount = quantity * price * (-1 if row["side"] == "buy" else 1)
            fill_cash += amount
        else:
            amount = _number(row.get("net_amount"))
            if amount is None or row.get("currency") != "USD":
                unresolved.append({"activity_id": key, "reason": "cash_amount_or_currency_unavailable"})
                continue
            other_cash += amount
            if activity_type in {"FEE", "CFEE", "PTC", "PTR"}:
                fee_cash += amount
        included.append({"activity_id": key, "activity_type": activity_type,
                         "effective_at": instant.isoformat(), "cash_effect_usd": str(amount)})

    expected_cash = open_cash + fill_cash + other_cash
    difference = close_cash - expected_cash
    return {
        "status": "cash_reconciled" if not unresolved and difference == 0 else "unverified",
        "account_id": account_id, "trading_mode": mode,
        "opening_at": start.isoformat(), "closing_at": end.isoformat(),
        "opening_cash_usd": str(open_cash), "closing_cash_usd": str(close_cash),
        "opening_equity_usd": str(open_equity), "closing_equity_usd": str(close_equity),
        "equity_change_usd": str(close_equity - open_equity),
        "position_value_change_usd": str((close_equity - close_cash) - (open_equity - open_cash)),
        "execution_cash_effect_usd": str(fill_cash),
        "other_booked_cash_effect_usd": str(other_cash),
        "booked_fee_cash_effect_usd": str(fee_cash),
        "expected_closing_cash_usd": str(expected_cash), "cash_difference_usd": str(difference),
        "included_activities": included, "unresolved_activities": unresolved,
        "corporate_action_observations": corporate_actions,
        "position_effect_reconciliation": "not_verified_without_position_boundaries",
        "fee_allocation": "unknown_per_execution", "performance_authority": False,
        "scope": "broker_boundary_and_cash_activity_audit_not_verified_net_strategy_return",
    }


def fee_record_coverage(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Expose what the actual activity schema can support without inventing fees."""
    fills = [row for row in snapshot.get("activities", []) if row.get("activity_type") == "FILL"]
    fees = [row for row in snapshot.get("activities", []) if row.get("activity_type") in {"FEE", "CFEE", "PTC", "PTR"}]
    references = sum(bool(row.get("fill_id") or row.get("order_id")) for row in fees)
    return {
        "status": "references_present_completeness_unverified" if references else "no_execution_linkage_in_fee_records",
        "scope": "raw_activity_rows_not_unique_executions",
        "fill_activity_rows": len(fills),
        "fee_activity_rows": len(fees),
        "fill_rows_with_fee_amount": sum(_number(row.get("fee_amount")) is not None for row in fills),
        "fee_rows_with_execution_reference": references,
        "fill_field_counts": dict(Counter(key for row in fills for key in row)),
        "fee_field_counts": dict(Counter(key for row in fees for key in row)),
        "snapshot_fetched_at": snapshot.get("fetched_at"),
        "pagination_complete": snapshot.get("pagination_complete") is True,
        "interpretation": "A reference or fee amount alone does not certify the complete total fee; missing records never establish zero fees.",
        "promotion_authority": False,
    }


def reconcile(snapshot: dict[str, Any], fills: list[dict[str, Any]]) -> dict[str, Any]:
    rejected: Counter[str] = Counter()
    unique: dict[str, dict[str, Any]] = {}
    conflicts = set()
    for row in snapshot.get("activities", []):
        key = str(row.get("id") or "")
        if not key:
            rejected["activity_id_missing"] += 1
        elif key in unique and row != unique[key]:
            conflicts.add(key)
        elif key in unique:
            rejected["duplicate_activity"] += 1
        else:
            unique[key] = row
    rejected["conflicting_activity_id"] = len(conflicts)
    account = snapshot.get("account_id")
    broker_order_ids = {str(row["order_id"]) for key, row in unique.items() if key not in conflicts and row.get("activity_type") == "FILL" and row.get("order_id")}
    observed: dict[str, Any] = defaultdict(lambda: 0)
    observed_execution_counts: Counter[str] = Counter()
    seen = set()
    fee_coverage: Counter[str] = Counter()
    recovered_order_links = 0
    after = pd.to_datetime(snapshot.get("after"), utc=True, errors="coerce")
    for row in fills:
        if row.get("account_id") != account or row.get("trading_mode") != "paper":
            # A broker-returned order ID proves order-account membership, not
            # the identity/completeness of individual local executions.
            if account and snapshot.get("trading_mode") == "paper" and row.get("account_id") in (None, "") and row.get("trading_mode") in (None, "", "paper") and str(row.get("order_id") or "") in broker_order_ids:
                recovered_order_links += 1
            else:
                rejected["fill_account_unverified"] += 1
                continue
        ts = pd.to_datetime(row.get("ts"), utc=True, errors="coerce")
        if pd.isna(ts):
            rejected["fill_timestamp_invalid"] += 1
            continue
        if not pd.isna(after) and ts < after:
            rejected["fill_before_accounting_window"] += 1
            continue
        key = row.get("fill_id")
        if not key or key in seen:
            rejected["fill_identity_missing_or_duplicate"] += 1
            continue
        seen.add(key)
        qty = _number(row.get("fill_qty"))
        if qty is None or qty <= 0 or not row.get("order_id"):
            rejected["fill_quantity_or_order_invalid"] += 1
            continue
        observed[str(row["order_id"])] += qty
        observed_execution_counts[str(row["order_id"])] += 1
        fee = _number(row.get("fee_amount"))
        complete_fee = row.get("fee_basis") == "per_fill_total" and row.get("fee_currency") == "USD" and row.get("fee_source") in {"broker_payload", "broker_activity"} and fee is not None and fee >= 0
        fee_coverage["explicit_total_fee" if complete_fee else "total_fee_unknown"] += 1
    broker: dict[str, Any] = defaultdict(lambda: 0)
    broker_execution_counts: Counter[str] = Counter()
    fees = []
    for key, row in unique.items():
        if key in conflicts:
            continue
        kind = row.get("activity_type")
        if kind == "FILL":
            qty = _number(row.get("qty"))
            if qty is None or qty <= 0 or not row.get("order_id"):
                rejected["broker_fill_fields_invalid"] += 1
                continue
            broker[str(row["order_id"])] += qty
            broker_execution_counts[str(row["order_id"])] += 1
        elif kind in {"FEE", "CFEE", "PTC", "PTR"}:
            amount = _number(row.get("net_amount"))
            fees.append({"activity_id": key, "activity_type": kind, "activity_sub_type": row.get("activity_sub_type"), "currency": row.get("currency"), "net_amount": str(amount) if amount is not None else None, "date": row.get("date"), "allocation_status": "unallocated_no_complete_per_fill_fee_contract"})
    orders = [
        {
            "order_id": key,
            "broker_qty": str(broker.get(key, 0)),
            "recorded_qty": str(observed.get(key, 0)),
            "status": "quantity_matched" if broker.get(key, 0) == observed.get(key, 0) else "quantity_mismatch",
            "broker_execution_count": broker_execution_counts[key],
            "recorded_fill_count": observed_execution_counts[key],
            "execution_count_status": (
                "count_matched_identity_unverified"
                if broker_execution_counts[key] == observed_execution_counts[key]
                else "count_mismatch"
            ),
        }
        for key in sorted(set(broker) | set(observed))
    ]
    return {"status": "accounting_compared" if snapshot.get("pagination_complete") and snapshot.get("trading_mode") == "paper" and account else "accounting_incomplete", "activity_count": len(unique), "activity_types": dict(Counter(row.get("activity_type") for row in unique.values())), "recovered_order_account_links": recovered_order_links, "fee_coverage": dict(fee_coverage), "fee_activities": fees, "order_quantity_comparison": orders, "order_quantity_counts": dict(Counter(row["status"] for row in orders)), "execution_count_counts": dict(Counter(row["execution_count_status"] for row in orders)), "rejection_counts": dict(rejected), "net_fee_validation": "unavailable_without_complete_per_fill_totals", "limitations": ["Matching order quantities or execution counts do not establish individual fill identity or fee completeness.", "No fee activity does not establish zero execution fees.", "Unallocated charges and rebates are never distributed across fills."], "promotion_authority": False, "orders_sent": 0}


def rebuild_quantity_ledger(snapshot: dict[str, Any], opening: dict[str, Any], closing: dict[str, Any]) -> dict[str, Any]:
    """Build a separate quantity audit; never replace historical execution evidence."""
    from ai_trading.tools.execution_evidence_reconciliation import _reconcile_positions

    account = snapshot.get("account_id")
    start = pd.to_datetime(opening.get("timestamp"), utc=True, errors="coerce")
    end = pd.to_datetime(closing.get("timestamp"), utc=True, errors="coerce")
    after = pd.to_datetime(snapshot.get("after"), utc=True, errors="coerce")
    fetched = pd.to_datetime(snapshot.get("fetched_at"), utc=True, errors="coerce")
    if (not account or snapshot.get("trading_mode") != "paper"
            or snapshot.get("pagination_complete") is not True
            or any(pd.isna(t) for t in (start, end, after, fetched))
            or not after < start < end <= fetched
            or any(b.get("account_id") != account or b.get("trading_mode") != "paper"
                   or b.get("positions_complete") is not True for b in (opening, closing))):
        raise ValueError("complete same-account paper snapshots and covering broker pagination required")
    unique: dict[str, dict[str, Any]] = {}
    for row in snapshot.get("activities", []):
        if row.get("activity_type") in _POSITION_ACTION_TYPES:
            action_time = _aware_instant(row.get("transaction_time"))
            if action_time is None or start < action_time <= end:
                raise ValueError("position-changing non-fill activity requires separate quantity evidence")
        if row.get("activity_type") != "FILL":
            continue
        key = str(row.get("id") or "")
        if not key or (key in unique and unique[key] != row):
            raise ValueError("missing or conflicting broker execution identity")
        unique[key] = row
    accepted = []
    for key, row in unique.items():
        ts = pd.to_datetime(row.get("transaction_time"), utc=True, errors="coerce")
        qty = _number(row.get("qty"))
        if (pd.isna(ts) or qty is None or qty <= 0 or not row.get("symbol")
                or not row.get("order_id") or row.get("side") not in ("buy", "sell")):
            raise ValueError("invalid broker execution fields")
        if start < ts <= end:
            accepted.append({"fill_id": key, "order_id": str(row["order_id"]),
                             "timestamp": ts, "symbol": str(row["symbol"]),
                             "side": row["side"], "qty": qty})
    accepted.sort(key=lambda row: (row["timestamp"], row["fill_id"]))
    comparison = _reconcile_positions(accepted, opening, closing)
    return {"status": comparison["status"], "position_reconciliation": comparison,
            "opening": opening, "closing": closing,
            "executions": [{**row, "timestamp": row["timestamp"].isoformat(), "qty": str(row["qty"])} for row in accepted],
            "historical_scope": "pre_opening_history_unverified_excluded_from_this_ledger",
            "scope": "broker_execution_quantity_audit_not_fee_or_performance_certification",
            "promotion_authority": False, "orders_sent": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fills", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-paper", action="store_true")
    parser.add_argument("--capture-closing-positions", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--opening-positions", type=Path)
    parser.add_argument("--closing-positions", type=Path)
    parser.add_argument("--ledger-output", type=Path)
    parser.add_argument("--position-evidence-output", type=Path)
    parser.add_argument("--opening-account", type=Path)
    parser.add_argument("--closing-account", type=Path)
    parser.add_argument("--equity-output", type=Path)
    args = parser.parse_args()
    if any((args.opening_positions, args.closing_positions, args.ledger_output)) and not all((args.opening_positions, args.closing_positions, args.ledger_output)):
        parser.error("opening-positions, closing-positions and ledger-output must be supplied together")
    if args.position_evidence_output and not args.ledger_output:
        parser.error("position-evidence-output requires opening-positions, closing-positions and ledger-output")
    if args.capture_closing_positions and not (args.fetch_paper and args.opening_positions and args.closing_positions):
        parser.error("capture-closing-positions requires fetch-paper and both position paths")
    if any((args.opening_account, args.closing_account, args.equity_output)) and not all((args.opening_account, args.closing_account, args.equity_output)):
        parser.error("opening-account, closing-account and equity-output must be supplied together")
    if args.fetch_paper:
        from alpaca.trading.client import TradingClient
        from ai_trading.config.managed_secrets import hydrate_managed_secrets
        from ai_trading.env import ensure_dotenv_loaded
        ensure_dotenv_loaded()
        hydrate_managed_secrets(required_keys=("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
        client = TradingClient(api_key=get_env("ALPACA_API_KEY"), secret_key=get_env("ALPACA_SECRET_KEY"), paper=True)
        if args.capture_closing_positions:
            opening = json.loads(args.opening_positions.read_text())
            opening_time = _aware_instant(opening.get("timestamp"))
            if opening_time is None:
                raise ValueError("opening position timestamp must be timezone-aware")
            closing = capture_current_position_boundary(client)
            atomic_write_text(args.closing_positions, json.dumps(closing, indent=2, sort_keys=True) + "\n")
            after = (opening_time - timedelta(seconds=1)).isoformat()
        else:
            after = (datetime.now(UTC) - timedelta(days=args.lookback_days)).isoformat()
        snapshot = capture(client, after=after)
        atomic_write_text(args.snapshot, json.dumps(snapshot, indent=2) + "\n")
    else:
        snapshot = json.loads(args.snapshot.read_text())
    fills, _, source = _read(args.fills)
    if args.ledger_output:
        opening_positions = json.loads(args.opening_positions.read_text())
        closing_positions = json.loads(args.closing_positions.read_text())
        ledger = rebuild_quantity_ledger(snapshot, opening_positions, closing_positions)
        atomic_write_text(args.ledger_output, json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        if args.position_evidence_output:
            atomic_write_text(
                args.position_evidence_output,
                json.dumps({
                    "snapshot": snapshot,
                    "opening": opening_positions,
                    "closing": closing_positions,
                }, indent=2, sort_keys=True) + "\n",
            )
    if args.equity_output:
        def _boundary(path: Path) -> dict[str, Any]:
            payload = json.loads(path.read_text())
            boundary = payload.get("account_boundary", payload)
            if not isinstance(boundary, dict):
                raise ValueError("account boundary must be an object")
            return dict(boundary)
        equity = reconcile_account_equity(snapshot, _boundary(args.opening_account), _boundary(args.closing_account))
        atomic_write_text(args.equity_output, json.dumps(equity, indent=2, sort_keys=True) + "\n")
    report = reconcile(snapshot, fills)
    report["fee_record_coverage"] = fee_record_coverage(snapshot)
    report["daily_account_fee_totals"] = daily_fee_totals(report["fee_activities"])
    report["document_evidence"] = {"status": "unavailable_in_configured_trading_client", "available_document_api": "Broker API /v1/accounts/{account_id}/documents", "configured_api": "paper Trading API", "transaction_fee_confirmation": "not_obtained", "source": "https://docs.alpaca.markets/us/reference/getdocsforaccount"}
    report.update(fill_source=source, snapshot_path=str(args.snapshot), generated_at=datetime.now(UTC).isoformat())
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    get_logger(__name__).info("BROKER_ACCOUNTING_REVIEW_COMPLETE", extra={"status": report["status"]})


if __name__ == "__main__":
    main()
