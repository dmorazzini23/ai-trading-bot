"""Reproducible research readiness evidence; never grants trading authority."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ai_trading.config.research_policy import reset_policy

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.execution_evidence_reconciliation import _identities, _number
from ai_trading.tools.research_foundation import audit_quotes

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config/research_reset.json"


def stamp(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, utc=False, errors="coerce")
    if not isinstance(parsed, pd.Timestamp) or pd.isna(parsed) or parsed.tzinfo is None:
        return None
    return parsed.tz_convert("UTC")


def read_window(path: Path, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hash the source, retain only seven days, and make excluded rows visible."""
    if not path.exists():
        return [], {"path": str(path), "status": "missing"}
    before = path.stat()
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    latest: pd.Timestamp | None = None
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            counts["rows_read"] += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not an object")
            except (ValueError, UnicodeDecodeError):
                counts["malformed"] += 1
                continue
            metrics = row.get("metrics") or {}
            ts = stamp(metrics.get("recorded_at") or row.get("ts") or row.get("bar_ts"))
            if ts is None:
                counts["invalid_timestamp"] += 1
            elif ts > now:
                counts["future_timestamp"] += 1
            elif ts < now - timedelta(days=7):
                counts["before_window"] += 1
            else:
                latest = max(latest, ts) if latest is not None else ts
                # Drop bulky configuration snapshots while preserving evidence fields.
                row.pop("config_snapshot", None)
                rows.append(row)
    after = path.stat()
    stable = (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    return rows, {"path": str(path), "sha256": digest.hexdigest(), "status": "read", "stable_during_read": stable, "counts": dict(counts), "rows_in_window": len(rows), "latest_in_window": latest.isoformat() if latest is not None else None}


def execution_chain(decisions: list[dict[str, Any]], orders: list[dict[str, Any]], fills: list[dict[str, Any]]) -> dict[str, Any]:
    """Require causal, unambiguous, account-scoped links and explicit total fees."""
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    order_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rows, target in ((decisions, index), (orders, order_index)):
        for row in rows:
            for identity_key in _identities(row):
                target[identity_key].append(row)
    counts: Counter[str] = Counter()
    unique: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for row in fills:
        key = str(row.get("fill_id") or "")
        if not key:
            counts["fill_id_missing"] += 1
        elif key in unique:
            if row != unique[key]:
                conflicts.add(key)
            else:
                counts["duplicate_fill"] += 1
        else:
            unique[key] = row
    complete = 0
    for key, fill in unique.items():
        gaps: set[str] = set()
        if key in conflicts:
            gaps.add("conflicting_fill")
        price, qty = _number(fill.get("fill_price")), _number(fill.get("fill_qty"))
        if price is None or qty is None or price <= 0 or qty <= 0 or fill.get("side") not in {"buy", "sell"}:
            gaps.add("invalid_fill")
        account = fill.get("account_id")
        if not account or fill.get("trading_mode") != "paper":
            gaps.add("paper_account_unverified")
        candidates = {id(row): row for identity in _identities(fill) for row in index[identity] if row.get("symbol") == fill.get("symbol")}
        if len(candidates) != 1:
            gaps.add("decision_missing_or_ambiguous")
        decision = next(iter(candidates.values())) if len(candidates) == 1 else {}
        metrics = decision.get("metrics") or {}
        decision_time = stamp(metrics.get("decision_ts"))
        quote_time = stamp(metrics.get("quote_timestamp"))
        source_time = stamp(metrics.get("source_timestamp") or decision.get("bar_ts"))
        fill_time = stamp(fill.get("ts"))
        if metrics.get("decision_ts_basis") != "explicit" or decision_time is None:
            gaps.add("explicit_decision_time_missing")
        if decision_time is None or source_time is None or source_time + pd.Timedelta(minutes=1) > decision_time:
            gaps.add("completed_source_bar_unverified")
        bid, ask = _number(metrics.get("raw_quote_bid")), _number(metrics.get("raw_quote_ask"))
        if bid is None or ask is None or bid <= 0 or ask < bid or decision_time is None or quote_time is None or not 0 <= (decision_time - quote_time).total_seconds() <= 1:
            gaps.add("causal_quote_unverified")
        order_identities = {identity for identity in _identities(fill) if identity[0] in {"order_id", "client_order_id"}}
        matching_orders = {id(row): row for identity in order_identities for row in order_index[identity] if row.get("symbol") == fill.get("symbol") and row.get("side") == fill.get("side") and row.get("account_id") == account and account and row.get("trading_mode") == "paper"}
        causal_orders = [row for row in matching_orders.values() if (ts := stamp(row.get("ts"))) is not None and decision_time is not None and fill_time is not None and decision_time <= ts <= fill_time]
        if not causal_orders:
            gaps.add("causal_account_order_unverified")
        fee = _number(fill.get("fee_amount"))
        if fill.get("fee_source") not in {"broker_payload", "broker_activity"} or fill.get("fee_basis") != "per_fill_total" or fill.get("fee_currency") != "USD" or fee is None or fee < 0:
            gaps.add("verified_total_fees_missing")
        if gaps:
            counts.update(gaps)
        else:
            complete += 1
    return {"unique_fills": len(unique), "complete_chains": complete, "complete_chain_fraction": complete / len(unique) if unique and not counts.get("fill_id_missing") and not conflicts else None, "gap_counts": dict(counts), "status": "supported" if unique and complete == len(unique) and not counts.get("fill_id_missing") else "evidence_pending", "scope": "seven_day_observed_fills_not_broker_completeness_or_round_trip_profit"}


def read_json_source(path: Path, now: datetime, *, max_age_days: int = 8) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        return {}, {"path": str(path), "status": "missing"}
    raw = path.read_bytes()
    source: dict[str, Any] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("not an object")
    except (ValueError, UnicodeDecodeError):
        return {}, {**source, "status": "invalid_json"}
    generated = stamp(payload.get("generated_at"))
    source["generated_at"] = generated.isoformat() if generated is not None else None
    source["status"] = "fresh" if generated is not None and timedelta(0) <= now - generated <= timedelta(days=max_age_days) else "stale_or_undated"
    return payload, source


def campaign_results(paths: list[Path], now: datetime) -> tuple[list[dict[str, Any]], dict[str, int]]:
    results = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for path in paths:
        state, source = read_json_source(path, now)
        contract = state.get("contract", {})
        campaign_id = str(contract.get("campaign_id") or "")
        if not campaign_id or campaign_id in seen:
            counts["missing_or_duplicate_campaign"] += 1
            continue
        seen.add(campaign_id)
        canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        if canonical != state.get("contract_hash"):
            counts["contract_hash_mismatch"] += 1
            continue
        for trial in state.get("trials", []):
            report_path = Path(trial.get("report_path") or "__missing__")
            report, report_source = read_json_source(report_path, now)
            verified = bool(trial.get("finished_at")) and report_source.get("sha256") == trial.get("report_sha256") and report.get("decision") == trial.get("status")
            completed_at = stamp(trial.get("finished_at"))
            in_week = completed_at is not None and timedelta(0) <= now - completed_at <= timedelta(days=7)
            if verified:
                counts["concluded_total"] += 1
                if in_week:
                    counts["concluded_last_seven_days"] += 1
            else:
                counts["unverified_or_claimed_trial"] += 1
            uncertainty = report.get("block_bootstrap_95_bps", report.get("session_bootstrap_95_bps"))
            uncertainty_status = "reported_not_independent_validation"
            if "decision_periods" in report and report["decision_periods"] < 8:
                uncertainty = None
                uncertainty_status = "unavailable_fewer_than_two_blocks"
            results.append({"campaign_id": campaign_id, "verified_completion": verified, "decision": trial.get("status"), "completed_at": trial.get("finished_at"), "source": source, "report_source": report_source, "aggregate": report.get("aggregate"), "criteria": report.get("criteria"), "uncertainty": uncertainty, "uncertainty_status": uncertainty_status, "decision_periods": report.get("decision_periods"), "folds": report.get("folds"), "symbols": report.get("symbols"), "holdout_evaluated": report.get("holdout_evaluated"), "metric": report.get("metric")})
    return results, dict(counts)


def build_scorecard(runtime_dir: Path, accounting_path: Path, paper_path: Path, campaigns: list[Path], now: datetime, sampling_path: Path | None = None) -> dict[str, Any]:
    inputs, sources = {}, {}
    for name, filename in (("decisions", "decision_records.jsonl"), ("orders", "order_events.jsonl"), ("fills", "fill_events.jsonl")):
        inputs[name], sources[name] = read_window(runtime_dir / filename, now)
    quotes = audit_quotes(inputs["decisions"], now)
    chain = execution_chain(**inputs)
    accounting, sources["accounting"] = read_json_source(accounting_path, now, max_age_days=2)
    paper, sources["paper_review"] = read_json_source(paper_path, now, max_age_days=2)
    trials, experiment_counts = campaign_results(campaigns, now)
    sampling: dict[str, Any] = {}
    if sampling_path is not None:
        sampling, sources["sampling_support"] = read_json_source(sampling_path, now)
    stable = all(sources[k].get("stable_during_read") is True and not any(sources[k].get("counts", {}).get(reason, 0) for reason in ("malformed", "invalid_timestamp", "future_timestamp")) for k in inputs)
    blockers = []
    if sampling_path is not None:
        if sources["sampling_support"]["status"] != "fresh":
            blockers.append("sampling_support_audit_missing_or_stale")
        elif sampling.get("status") == "blocked_sampling_support":
            blockers.append("sampling_support_insufficient")
    if chain["status"] != "supported" or not stable:
        blockers.append("complete_causal_execution_chains_unverified")
    if sources["accounting"]["status"] != "fresh" or accounting.get("net_fee_validation") != "verified":
        blockers.append("verified_total_execution_costs_unavailable")
    if sources["paper_review"]["status"] != "fresh" or paper.get("completion_gaps"):
        blockers.append("paper_session_evidence_pending")
    blockers.append("untouched_evaluation_not_performed")
    return {"artifact_type": "research_reset_scorecard", "generated_at": now.isoformat(), "status": "evidence_pending", "window": {"start": (now - timedelta(days=7)).isoformat(), "end": now.isoformat()}, "highest_value_blocker": blockers[0], "blockers": blockers, "sampling_support": {k: sampling.get(k) for k in ("status", "possible_common_periods", "candidate_periods", "maximum_possible_selections", "minimum_required", "trial_claimed")}, "execution_chain": chain, "quotes": quotes, "experiments": experiment_counts, "trials": trials, "accounting": {k: accounting.get(k) for k in ("status", "activity_count", "fee_coverage", "order_quantity_counts", "net_fee_validation", "daily_account_fee_totals", "document_evidence", "fee_record_coverage")}, "paper_session": {k: paper.get(k) for k in ("session_date", "completion_gaps", "session_audit", "execution_cost_comparison")}, "untouched_net_performance": {"status": "unavailable", "reason": "reserved_holdout_not_evaluated"}, "sources": sources, "limitations": ["Weekly snapshots are observations, not additional experiments.", "No fills means unavailable execution coverage, not 100 percent coverage.", "Quote spreads and fixed cost scenarios do not certify total costs.", "Development results are not untouched performance.", "No missing fees are imputed as zero; no account-level charges are allocated to guessed fills."], "promotion_authority": False, "orders_sent": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--accounting-report", type=Path, required=True)
    parser.add_argument("--paper-report", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-support", type=Path, default=ROOT / "artifacts/research_reset/sampling_support.json")
    args = parser.parse_args(argv)
    report = build_scorecard(args.runtime_dir, args.accounting_report, args.paper_report, args.campaign_state, datetime.now(UTC), args.sampling_support)
    atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    get_logger(__name__).info("RESEARCH_RESET_SCORECARD_COMPLETE", extra={"status": report["status"], "blocker": report["highest_value_blocker"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
