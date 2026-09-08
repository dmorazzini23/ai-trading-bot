"""Read-only daily paper-session, cap-selection and execution-cost review."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.execution_evidence_reconciliation import _number, _read, reconcile_session
from ai_trading.utils.market_calendar import is_trading_day, session_info


def _fee_total(rows: list[dict[str, Any]], *, simulated: bool) -> float | None:
    """Require explicit total fees per execution in the USD price currency."""
    fees = []
    sources = {"simulated_notional_bps_v1"} if simulated else {"broker_payload", "broker_activity"}
    for row in rows:
        fee = _number(row.get("fee_amount"))
        if fee is None or fee < 0 or row.get("fee_currency") != "USD" or row.get("fee_basis") != "per_fill_total" or row.get("fee_source") not in sources:
            return None
        if simulated:
            rate = _number(row.get("fee_rate_bps"))
            if rate is None or rate < 0 or not np.isclose(float(fee), float(row["fill_qty"]) * float(row["fill_price"]) * float(rate) / 10000, rtol=1e-9, atol=1e-10):
                return None
        fees.append(float(fee))
    return sum(fees)


def cap_selection_review(replay: dict[str, Any]) -> dict[str, Any]:
    baseline = replay.get("baseline_summary", {}).get("markout_observations", [])
    candidate = replay.get("replay_summary", {}).get("markout_observations", [])
    if not baseline or not candidate:
        return {"status": "unavailable", "reason": "paired_markout_observations_missing"}
    retained_ids = {str(row["client_order_id"]) for row in candidate}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline:
        groups[str(row["symbol"])].append(row)
    rows = []
    for symbol, group in sorted(groups.items()):
        retained = [float(row["net_edge_bps"]) for row in group if str(row["client_order_id"]) in retained_ids]
        removed = [float(row["net_edge_bps"]) for row in group if str(row["client_order_id"]) not in retained_ids]
        rows.append({"symbol": symbol, "retained_markouts": len(retained), "removed_markouts": len(removed), "retained_mean_bps": float(np.mean(retained)) if retained else None, "removed_mean_bps": float(np.mean(removed)) if removed else None, "retained_minus_removed_bps": float(np.mean(retained) - np.mean(removed)) if retained and removed else None})
    adjustments = replay.get("cap_adjustments", [])
    return {"status": "diagnosed", "symbols": rows, "adjustments_by_cap_kind": dict(Counter(str(row.get("cap_kind")) for row in adjustments)), "loss_attribution": replay.get("loss_attribution"), "interpretation": "Retained means use baseline outcomes; this diagnoses selection, not a tradable ranking signal.", "allocation_change": "none_without_pretrade_scores_and_fixed_input_validation", "limits_changed": False}


def compare_execution_costs(replay: dict[str, Any], fills: list[dict[str, Any]], *, account_id: str) -> dict[str, Any]:
    """Pair by client order and identical arrival benchmark; do not pool unlike costs."""
    simulated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for row in replay.get("replay_summary", {}).get("markout_observations", []):
        simulated[str(row["client_order_id"])].append(row)
    seen: dict[str, dict[str, Any]] = {}
    ambiguous = set()
    for row in fills:
        identity = str(row.get("fill_id") or "")
        if not identity:
            rejected["fill_id_missing"] += 1
        elif identity in seen:
            if row != seen[identity]:
                ambiguous.add(identity)
            else:
                rejected["duplicate"] += 1
        else:
            seen[identity] = row
    for identity, row in seen.items():
        if identity in ambiguous:
            rejected["conflicting_fill_id"] += 1
            continue
        if row.get("trading_mode") != "paper" or row.get("account_id") != account_id:
            rejected["paper_account_identity_unverified"] += 1
            continue
        key = str(row.get("client_order_id") or "")
        if not key or key not in simulated:
            rejected["simulated_order_missing"] += 1
            continue
        observed[key].append(row)
    pairs = []
    for identity, actual in observed.items():
        sim = simulated[identity]
        references = [_number(row.get("reference_price")) for row in sim] + [_number(row.get("expected_price")) for row in actual]
        sides = {row.get("side") for row in [*sim, *actual]}
        symbols = {row.get("symbol") for row in [*sim, *actual]}
        if any(ref is None or ref <= 0 for ref in references) or len(set(references)) != 1 or len(sides) != 1 or not sides <= {"buy", "sell"} or len(symbols) != 1:
            rejected["arrival_benchmark_or_order_contract_mismatch"] += 1
            continue
        prices, quantities = [], []
        for rows in [sim, actual]:
            qty = [_number(row.get("fill_qty")) for row in rows]
            px = [_number(row.get("fill_price")) for row in rows]
            if any(value is None or value <= 0 for value in [*qty, *px]):
                break
            quantities.append(sum(float(value) for value in qty))
            prices.append(sum(float(q) * float(p) for q, p in zip(qty, px)) / quantities[-1])
        if len(prices) != 2:
            rejected["invalid_fill_fields"] += 1
            continue
        timestamps = [pd.to_datetime(row.get("ts"), utc=True, errors="coerce") for row in actual]
        if any(pd.isna(ts) for ts in timestamps):
            rejected["invalid_execution_timestamp"] += 1
            continue
        sign = 1 if next(iter(sides)) == "buy" else -1
        reference = float(references[0])
        sim_bps, actual_bps = [sign * (price / reference - 1) * 10000 for price in prices]
        fees = [_number(row.get("fee_amount")) if row.get("fee_source") in {"broker_payload", "broker_activity"} else None for row in actual]
        pairs.append({"client_order_id": identity, "symbol": next(iter(symbols)), "session": str(timestamps[0].tz_convert("America/New_York").date()), "simulated_slippage_bps": sim_bps, "observed_slippage_bps": actual_bps, "simulation_minus_observed_bps": sim_bps - actual_bps, "simulated_qty": quantities[0], "observed_qty": quantities[1], "observed_fee_bps": sum(float(fee) for fee in fees) / (prices[1] * quantities[1]) * 10000 if all(fee is not None for fee in fees) else None})
        sim_fee, actual_fee = _fee_total(sim, simulated=True), _fee_total(actual, simulated=False)
        if sim_fee is None:
            rejected["simulated_fee_contract_missing_or_invalid"] += 1
        if actual_fee is None:
            rejected["observed_fee_contract_missing_or_invalid"] += 1
        pair = pairs[-1]
        pair["net_cost_contract_valid"] = sim_fee is not None and actual_fee is not None
        if sim_fee is not None and actual_fee is not None:
            sim_net = sim_bps + sim_fee / (reference * quantities[0]) * 10000
            actual_net = actual_bps + actual_fee / (reference * quantities[1]) * 10000
            pair.update(simulated_total_cost_bps=sim_net, observed_total_cost_bps=actual_net, net_cost_error_bps=sim_net - actual_net)
    sessions = len({row["session"] for row in pairs})
    errors = [row["simulation_minus_observed_bps"] for row in pairs]
    net_pairs = [row for row in pairs if row["net_cost_contract_valid"]]
    supported = len(pairs) >= 30 and sessions >= 5
    net_status = "validated" if supported and len(net_pairs) == len(pairs) else "insufficient_execution_evidence" if len(net_pairs) == len(pairs) and pairs else "unavailable_without_simulated_and_observed_fee_contracts"
    return {"status": "comparison_available" if supported else "insufficient_execution_evidence", "minimum_orders": 30, "minimum_sessions": 5, "paired_orders": len(pairs), "sessions": sessions, "pairs": pairs, "rejection_counts": dict(rejected), "mean_slippage_error_bps": float(np.mean(errors)) if errors else None, "p90_absolute_slippage_error_bps": float(np.quantile(np.abs(errors), .9)) if errors else None, "net_cost_validation": net_status, "net_cost_paired_orders": len(net_pairs), "mean_net_cost_error_bps": float(np.mean([row["net_cost_error_bps"] for row in net_pairs])) if net_pairs else None, "fee_contract": {"currency": "USD", "basis": "per_fill_total", "cost_denominator": "arrival_price_times_each_quantity", "validation_meaning": "complete_comparable_measurements_not_cost_accuracy_or_profitability"}, "limitations": ["Identical arrival benchmark required; quantities may differ and are reported.", "Paper fills do not validate market impact or queue position.", "No promotion authority; sample thresholds establish comparison support, not profitability."]}


def latest_completed_session(now: datetime) -> str:
    day = now.astimezone(ZoneInfo("America/New_York")).date()
    while not is_trading_day(day) or session_info(day).end_utc + timedelta(minutes=15) > now:
        day -= timedelta(days=1)
    return day.isoformat()


def capture_readiness(boundaries: list[dict[str, Any]], *, account_id: str, now: datetime) -> dict[str, Any]:
    """Expose capture liveness and the next full session's required windows."""
    day = now.astimezone(ZoneInfo("America/New_York")).date()
    while not is_trading_day(day) or session_info(day).start_utc <= now:
        day += timedelta(days=1)
    session = session_info(day)
    timestamps = [pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce") for row in boundaries if row.get("account_id") == account_id and row.get("trading_mode") == "paper" and row.get("positions_complete") is True]
    latest = max((ts for ts in timestamps if not pd.isna(ts) and ts <= now), default=None)
    age = (pd.Timestamp(now) - latest).total_seconds() if latest is not None else None
    return {"status": "capture_active" if account_id and age is not None and age <= 180 else "capture_missing_or_stale", "latest_complete_snapshot": latest.isoformat() if latest is not None else None, "snapshot_age_seconds": age, "maximum_snapshot_age_seconds": 180, "next_full_session": day.isoformat(), "opening_window_utc": [(session.start_utc - timedelta(minutes=15)).isoformat(), session.start_utc.isoformat()], "closing_window_utc": [session.end_utc.isoformat(), (session.end_utc + timedelta(minutes=15)).isoformat()], "complete_session_observed": False, "orders_sent": 0}


def completion_gaps(audit: dict[str, Any], costs: dict[str, Any], source_errors: list[str]) -> list[str]:
    """A supported slippage comparison alone cannot complete net-cost validation."""
    gaps = []
    if source_errors:
        gaps.append("input_evidence_gaps")
    if audit.get("status") != "paper_session_reconciled":
        gaps.append("paper_session_not_reconciled")
    if costs.get("status") != "comparison_available":
        gaps.append("execution_comparison_not_supported")
    if costs.get("net_cost_validation") != "validated":
        gaps.append("net_cost_validation_unavailable")
    return gaps


def allocation_order_sensitivity(source: Path) -> dict[str, Any]:
    """Predeclared diagnostic: reverse exact timestamp ties, preserving chronology."""
    from ai_trading.core.bot_engine import _load_replay_bars, _replay_initial_positions_from_rows, _replay_summary_metrics
    from ai_trading.replay.event_loop import ReplayEventLoop

    bars = _load_replay_bars(path=str(source), symbols={"AAPL", "AMZN", "MSFT"}, timeframes={"5Min"}, start_date=None, end_date=None)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(bars, key=lambda row: row["ts"]):
        groups[str(row["ts"])].append(row)
    original = [row for group in groups.values() for row in group]
    reversed_ties = [row for group in groups.values() for row in reversed(group)]
    initial = _replay_initial_positions_from_rows(original)

    def strategy(bar: Any) -> dict[str, Any]:
        return {"symbol": bar["symbol"], "side": bar.get("side", "buy"), "qty": bar.get("qty", 0), "type": bar.get("order_type", "limit"), "price": bar["close"], "limit_price": bar["close"], "intent_key": bar["client_order_id"], "client_order_id": bar["client_order_id"]}

    variants = []
    for name, rows in [("recorded_tie_order", original), ("reversed_exact_timestamp_ties", reversed_ties)]:
        result = ReplayEventLoop(strategy=strategy, seed=42, max_symbol_notional=25000, max_gross_notional=150000, initial_positions=initial, clip_intents_to_caps=True).run(rows)
        summary = _replay_summary_metrics(result, market_rows=original, max_markout_hours=24)
        variants.append({"name": name, "net_edge_bps": summary["net_edge_bps"], "sample_count": summary["sample_count"], "orders": len(result["orders"]), "cap_adjustments": len(result["cap_adjustments"]), "violations": result["violations"]})
    return {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "rows": len(original), "timestamp_groups_with_ties": sum(len(group) > 1 for group in groups.values()), "pretrade_score_rows": sum(row.get("expected_net_edge_bps") is not None for row in original), "seed": 42, "max_symbol_notional": 25000, "max_gross_notional": 150000, "variants": variants, "reversed_minus_recorded_bps": variants[1]["net_edge_bps"] - variants[0]["net_edge_bps"], "decision": "diagnostic_only_no_outcome_selected_ordering", "limits_changed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--session")
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument("--accounting-report", type=Path)
    parser.add_argument("--allocation-source", type=Path, help="Optional fixed replay input for the predeclared tie-order diagnostic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    now = datetime.now(UTC)
    inputs, sources = {}, {}
    for name, filename in {"decisions": "decision_records.jsonl", "orders": "order_events.jsonl", "fills": "fill_events.jsonl", "tca": "tca_records.jsonl", "boundaries": "broker_position_boundaries.jsonl"}.items():
        path = args.runtime_dir / filename
        if path.exists():
            inputs[name], _, sources[name] = _read(path)
        else:
            inputs[name], sources[name] = [], {"path": str(path), "missing": True}
    replay = {}
    if args.replay_report.exists():
        raw = args.replay_report.read_bytes()
        replay = json.loads(raw)
        sources["replay"] = {"path": str(args.replay_report), "sha256": hashlib.sha256(raw).hexdigest()}
        if isinstance(replay.get("replay"), dict) and replay["replay"].get("path"):
            path = Path(replay["replay"]["path"])
            raw = path.read_bytes()
            replay = json.loads(raw)
            sources["replay_artifact"] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    else:
        sources["replay"] = {"path": str(args.replay_report), "missing": True}
    accounts = {row["account_id"] for row in inputs["boundaries"] if row.get("account_id") and row.get("trading_mode") == "paper"}
    session = args.session or latest_completed_session(now)
    account = next(iter(accounts)) if len(accounts) == 1 else ""
    audit = reconcile_session(session_date=session, account_id=account, **inputs) if account else {"status": "evidence_gaps", "session_gaps": ["unique_paper_account_identity_required"]}
    report: dict[str, Any] = {"generated_at": datetime.now(UTC).isoformat(), "session_date": session, "session_audit": audit, "cap_selection": cap_selection_review(replay), "execution_cost_comparison": compare_execution_costs(replay, inputs["fills"], account_id=account), "sources": sources, "promotion_authority": False, "orders_sent": 0}
    report["capture_readiness"] = capture_readiness(inputs["boundaries"], account_id=account, now=now)
    training_reports = {}
    for name, path in (("training", args.training_report), ("selection", args.selection_report), ("accounting", args.accounting_report)):
        if path is not None:
            if path.exists():
                raw = path.read_bytes()
                training_reports[name] = json.loads(raw)
                sources[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
            else:
                sources[name] = {"path": str(path), "missing": True}
    if args.accounting_report is not None:
        report["broker_accounting"] = training_reports.get("accounting", {"status": "missing"})
    if args.training_report is not None or args.selection_report is not None:
        training = training_reports.get("training", {})
        selection = training_reports.get("selection", {})
        report["candidate_abstention_reviews"] = [{"model_id": row.get("model_id"), "review": row.get("abstention_review", {"status": "missing"})} for row in training.get("candidates", [])]
        report["training_readiness"] = {"training_status": training.get("status", "missing"), "generated_at": training.get("generated_at"), "data_provenance": training.get("data_provenance", {}), "candidate_count": len(training.get("candidates", [])), "selection_status": selection.get("status", "missing"), "selection_decisions": [{key: row.get(key) for key in ("candidate_model_id", "selected_model_id", "status", "samples", "reasons")} for row in selection.get("decisions", [])], "runtime_model_replaced": False, "promotion_authority": False, "interpretation": "Training completion does not clear serving freshness or promotion gates."}
    if args.allocation_source:
        report["allocation_order_sensitivity"] = allocation_order_sensitivity(args.allocation_source)
    source_errors = [key for key, source in sources.items() if source.get("missing") or source.get("invalid_rows", 0) or source.get("stable_during_read") is False]
    report["input_evidence_gaps"] = source_errors
    report["completion_gaps"] = completion_gaps(audit, report["execution_cost_comparison"], source_errors)
    if args.accounting_report is not None and report["broker_accounting"].get("status") != "accounting_compared":
        report["completion_gaps"].append("broker_accounting_incomplete")
    report["status"] = "evidence_pending" if report["completion_gaps"] else "review_complete"
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    get_logger(__name__).info("PAPER_EVIDENCE_REVIEW_COMPLETE", extra={"status": report["status"], "output": str(args.output)})


if __name__ == "__main__":
    main()
