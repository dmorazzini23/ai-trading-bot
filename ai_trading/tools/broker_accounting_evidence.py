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
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.execution_evidence_reconciliation import _number, _read


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
    return {"account_id": account_id, "trading_mode": "paper", "after": after, "fetched_at": datetime.now(UTC).isoformat(), "pagination_complete": complete, "activities": rows}


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
        fee = _number(row.get("fee_amount"))
        complete_fee = row.get("fee_basis") == "per_fill_total" and row.get("fee_currency") == "USD" and row.get("fee_source") in {"broker_payload", "broker_activity"} and fee is not None and fee >= 0
        fee_coverage["explicit_total_fee" if complete_fee else "total_fee_unknown"] += 1
    broker: dict[str, Any] = defaultdict(lambda: 0)
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
        elif kind in {"FEE", "CFEE", "PTC", "PTR"}:
            amount = _number(row.get("net_amount"))
            fees.append({"activity_id": key, "activity_type": kind, "activity_sub_type": row.get("activity_sub_type"), "currency": row.get("currency"), "net_amount": str(amount) if amount is not None else None, "date": row.get("date"), "allocation_status": "unallocated_no_complete_per_fill_fee_contract"})
    orders = [{"order_id": key, "broker_qty": str(broker.get(key, 0)), "recorded_qty": str(observed.get(key, 0)), "status": "quantity_matched" if broker.get(key, 0) == observed.get(key, 0) else "quantity_mismatch"} for key in sorted(set(broker) | set(observed))]
    return {"status": "accounting_compared" if snapshot.get("pagination_complete") and snapshot.get("trading_mode") == "paper" and account else "accounting_incomplete", "activity_count": len(unique), "activity_types": dict(Counter(row.get("activity_type") for row in unique.values())), "recovered_order_account_links": recovered_order_links, "fee_coverage": dict(fee_coverage), "fee_activities": fees, "order_quantity_comparison": orders, "order_quantity_counts": dict(Counter(row["status"] for row in orders)), "rejection_counts": dict(rejected), "net_fee_validation": "unavailable_without_complete_per_fill_totals", "limitations": ["Accounting quantities alone do not establish fill identity or fee completeness.", "No fee activity does not establish zero execution fees.", "Unallocated charges and rebates are never distributed across fills."], "promotion_authority": False, "orders_sent": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fills", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-paper", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=90)
    args = parser.parse_args()
    if args.fetch_paper:
        from alpaca.trading.client import TradingClient
        from ai_trading.config.managed_secrets import hydrate_managed_secrets
        hydrate_managed_secrets(required_keys=("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
        client = TradingClient(api_key=get_env("ALPACA_API_KEY"), secret_key=get_env("ALPACA_SECRET_KEY"), paper=True)
        snapshot = capture(client, after=(datetime.now(UTC) - timedelta(days=args.lookback_days)).isoformat())
        atomic_write_text(args.snapshot, json.dumps(snapshot, indent=2) + "\n")
    else:
        snapshot = json.loads(args.snapshot.read_text())
    fills, _, source = _read(args.fills)
    report = reconcile(snapshot, fills)
    report["fee_record_coverage"] = fee_record_coverage(snapshot)
    report["daily_account_fee_totals"] = daily_fee_totals(report["fee_activities"])
    report["document_evidence"] = {"status": "unavailable_in_configured_trading_client", "available_document_api": "Broker API /v1/accounts/{account_id}/documents", "configured_api": "paper Trading API", "transaction_fee_confirmation": "not_obtained", "source": "https://docs.alpaca.markets/us/reference/getdocsforaccount"}
    report.update(fill_source=source, snapshot_path=str(args.snapshot), generated_at=datetime.now(UTC).isoformat())
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    get_logger(__name__).info("BROKER_ACCOUNTING_REVIEW_COMPLETE", extra={"status": report["status"]})


if __name__ == "__main__":
    main()
