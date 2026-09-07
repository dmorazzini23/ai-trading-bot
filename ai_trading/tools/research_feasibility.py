"""Screen a frozen hypothesis using development data before another study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.utils.market_calendar import is_trading_day


def protocol_hash(protocol: dict[str, Any]) -> str:
    contract = {key: value for key, value in protocol.items() if key != "feasibility_review"}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def screen_development(protocol: dict[str, Any], ledgers: dict[int, pd.DataFrame]) -> dict[str, Any]:
    """Use session bootstrap bounds as a feasibility screen, not proof of edge."""
    development = next(row for row in protocol["retrospective_intervals"] if row["name"] == "development")
    days = [str(day.date()) for day in pd.date_range(development["start"], development["end"]) if is_trading_day(day.date())]
    acceptance = protocol["acceptance"]
    validation_sessions = sum(is_trading_day(day.date()) for row in protocol["retrospective_intervals"] if row["name"] != "development" for day in pd.date_range(row["start"], row["end"]))
    results = []
    for horizon in protocol["horizons_minutes"]:
        ledger = ledgers[int(horizon)]
        selected = ledger.loc[ledger["session"].isin(days) & ledger["selected"].eq(True)]
        gross = pd.to_numeric(selected["gross_return_bps"], errors="coerce")
        if not np.isfinite(gross).all():
            raise ValueError("development markouts must be finite")
        grouped = selected.assign(gross=gross).groupby("session")["gross"]
        counts = grouped.count().reindex(days, fill_value=0).to_numpy(dtype=float)
        sums = grouped.sum().reindex(days, fill_value=0).to_numpy(dtype=float)
        enough = len(days) >= 40 and len(selected) >= 20
        expected_lower = gross_upper = None
        if enough:
            rng = np.random.default_rng(20260907)
            block = 5
            starts = rng.integers(0, len(days) - block + 1, size=(5000, int(np.ceil(len(days) / block))))
            indices = (starts[..., None] + np.arange(block)).reshape(5000, -1)[:, :len(days)]
            totals = counts[indices].sum(axis=1)
            expected_lower = float(np.quantile(totals / len(days) * validation_sessions, 0.05))
            usable = totals > 0
            gross_upper = float(np.quantile(sums[indices].sum(axis=1)[usable] / totals[usable], 0.95))
        reasons = []
        if not enough:
            reasons.append("development_support_insufficient")
        if expected_lower is None or expected_lower < acceptance["minimum_completed_validation_trades"]:
            reasons.append("expected_trade_support_below_requirement")
        if gross_upper is None or gross_upper <= 2 * acceptance["primary_one_way_cost_bps"]:
            reasons.append("optimistic_gross_edge_does_not_cover_costs")
        if not protocol.get("hypothesis") or not protocol.get("stopping_rule"):
            reasons.append("hypothesis_or_stopping_rule_missing")
        results.append({"horizon_minutes": horizon, "development_trades": len(selected), "development_sessions": len(days), "projected_validation_sessions": validation_sessions, "expected_validation_trades_lower": expected_lower, "optimistic_gross_edge_bps": gross_upper, "required_round_trip_cost_bps": 2 * acceptance["primary_one_way_cost_bps"], "reasons": reasons, "eligible": not reasons})
    return {"status": "eligible_for_fixed_study" if results and all(row["eligible"] for row in results) else "blocked", "protocol_hash": protocol_hash(protocol), "horizons": results, "scope": "development_only_feasibility_not_confirmation", "method": "five_session_block_bootstrap_5000_replicates_seed_20260907", "limitations": ["Projection assumes development trade frequency persists.", "Optimistic gross upper bound is only a rejection screen, not evidence of positive net edge.", "Prior project reuse of these data remains a selection-bias limitation."], "promotion_authority": False}


def require_feasibility(protocol: dict[str, Any]) -> None:
    review = protocol.get("feasibility_review")
    if not isinstance(review, dict) or not review.get("path") or not review.get("sha256"):
        raise ValueError("feasibility_review_required_before_new_study")
    raw = Path(review["path"]).read_bytes()
    report = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != review["sha256"] or report.get("protocol_hash") != protocol_hash(protocol):
        raise ValueError("feasibility_review_contract_mismatch")
    if report.get("status") != "eligible_for_fixed_study":
        raise ValueError("research_feasibility_blocked")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    paths = {int(h): args.ledger_dir / f"opportunities_h{h}.csv" for h in protocol["horizons_minutes"]}
    report = screen_development(protocol, {h: pd.read_csv(path) for h, path in paths.items()})
    report["ledger_sha256"] = {str(h): hashlib.sha256(path.read_bytes()).hexdigest() for h, path in paths.items()}
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    get_logger(__name__).info("RESEARCH_FEASIBILITY_SCREENED", extra={"status": report["status"], "output": str(args.output)})


if __name__ == "__main__":
    main()
