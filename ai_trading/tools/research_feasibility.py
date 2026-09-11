"""Screen a frozen hypothesis using development data before another study."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.utils.market_calendar import is_trading_day, session_info


def sampling_support(protocol: dict[str, Any], indices: dict[str, pd.DatetimeIndex]) -> dict[str, Any]:
    """Count possible fixed-rule observations using timestamps, never prices."""
    if protocol.get("lookback_sessions") != 60 or protocol.get("holding_sessions") != 5:
        raise ValueError("unsupported sampling contract")
    start, end = protocol["development_start"], protocol["development_end"]
    if not start <= end < protocol["holdout_start"] <= protocol["holdout_end"]:
        raise ValueError("invalid development interval or holdout overlap")
    dates = [day.date() for day in pd.date_range(start, end) if is_trading_day(day.date())]
    parts = []
    for day in dates:
        session = session_info(day)
        parts.append(pd.date_range(session.start_utc, session.end_utc, freq="min", inclusive="left"))
    if not parts:
        raise ValueError("no development sessions")
    expected = parts[0].append(parts[1:])
    candidate_indices = list(range(61, len(dates) - 5, 5))
    common = set(candidate_indices)
    by_symbol = {}
    invalid = []
    for symbol in protocol["symbols"]:
        if symbol not in indices:
            invalid.append(f"symbol_missing:{symbol}")
            common.clear()
            continue
        index = indices[symbol]
        if index.tz is None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
            invalid.append(f"invalid_timestamp_grain:{symbol}")
        if index.tz is None:
            common.clear()
            continue
        missing = expected.difference(index)
        gaps = Counter(missing.tz_convert("America/New_York").date)
        complete = {i for i, day in enumerate(dates) if day not in gaps}
        usable = {i for i in candidate_indices if all(j in complete for j in range(i - 61, i + 6))}
        common &= usable
        boundaries = {stamp for part in parts for stamp in (part[0], part[-1])}
        by_symbol[symbol] = {"expected_minutes": len(expected), "missing_minutes": len(missing), "missing_ratio": len(missing) / len(expected), "expected_sessions": len(dates), "complete_sessions": len(complete), "incomplete_sessions": len(gaps), "missing_boundary_minutes": sum(stamp in boundaries for stamp in missing), "possible_periods": len(usable), "excluded_periods": len(candidate_indices) - len(usable), "gap_sessions": [{"session": day.isoformat(), "missing_minutes": count} for day, count in sorted(gaps.items())]}
    folds = Counter(str(pd.Timestamp(dates[i]).to_period("Q")) for i in common)
    maximum_selections = len(common) * len(protocol["symbols"])
    supported_quarters_upper = sum(n * len(protocol["symbols"]) >= 12 for n in folds.values())
    possible = len(common) >= 60 and maximum_selections >= 100 and supported_quarters_upper >= 6 and not invalid
    return {"status": "support_possible_not_strategy_eligible" if possible else "blocked_sampling_support", "possible_common_periods": len(common), "candidate_periods": len(candidate_indices), "maximum_possible_selections": maximum_selections, "supported_quarters_upper_bound": supported_quarters_upper, "minimum_required": {"periods": 60, "selections": 100, "quarters_with_12_selections": 6}, "common_entry_dates": [dates[i].isoformat() for i in sorted(common)], "periods_by_quarter": dict(folds), "symbols": by_symbol, "invalid_inputs": invalid, "prices_read_for_analysis": False, "strategy_outcomes_computed": False, "trial_claimed": False, "promotion_authority": False, "limitations": ["Possible observations are upper bounds; actual signal selection and valid prices can reduce support.", "Missing bars may reflect no eligible trades; this audit alone does not identify their cause.", "Passing this audit cannot reopen a consumed campaign or authorize a new trial."]}


def audit_sampling_acquisition(protocol: dict[str, Any], acquisition_path: Path) -> dict[str, Any]:
    """Verify frozen source hashes and inspect only their timestamp columns."""
    acquisition_raw = acquisition_path.read_bytes()
    acquisition = json.loads(acquisition_raw)
    manifest_path = Path(acquisition["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = acquisition_path.parent / manifest_path
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    identity = manifest["dataset_identity"]
    interval = identity["request_interval"]
    if interval["start_date"] != protocol["development_start"] or interval["end_date"] != protocol["development_end"] or identity["feed"] != protocol["feed"] or identity["adjustment"] != protocol["adjustment"] or sorted(identity["symbols"]) != sorted(protocol["symbols"]):
        raise ValueError("acquisition differs from frozen development contract")
    if not protocol["development_end"] < protocol["holdout_start"]:
        raise ValueError("holdout overlap")
    if acquisition.get("quality_passed") is not True or manifest.get("quality_passed") is not True:
        raise ValueError("acquisition quality unverified")
    sources, indices = {}, {}
    for symbol in protocol["symbols"]:
        rows = [row for row in manifest["symbols"] if row["symbol"] == symbol]
        if len(rows) != 1:
            raise ValueError("missing or duplicate symbol provenance")
        row = rows[0]
        path = Path(row["csv_path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.resolve() != (manifest_path.parent / f"{symbol}.csv").resolve():
            raise ValueError("source outside governed dataset")
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["content_sha256"]:
            raise ValueError(f"source hash mismatch:{symbol}")
        frame = pd.read_csv(path, usecols=["timestamp"], dtype=str)
        if not frame["timestamp"].str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", na=False).all():
            raise ValueError(f"timezone missing from source:{symbol}")
        index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True, errors="raise"))
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"source changed during audit:{symbol}")
        indices[symbol] = index
        sources[symbol] = {"path": str(path), "sha256": digest, "stable_during_read": True, "columns_used": ["timestamp"]}
    report = sampling_support(protocol, indices)
    report.update(generated_at=datetime.now(UTC).isoformat(), protocol_hash=protocol_hash(protocol), sources=sources, acquisition_sha256=hashlib.sha256(acquisition_raw).hexdigest(), manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())
    return report


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
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--ledger-dir", type=Path)
    inputs.add_argument("--sampling-acquisition", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.sampling_acquisition is not None:
        report = audit_sampling_acquisition(protocol, args.sampling_acquisition)
        atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
        get_logger(__name__).info("SAMPLING_SUPPORT_AUDITED", extra={"status": report["status"], "output": str(args.output)})
        return
    paths = {int(h): args.ledger_dir / f"opportunities_h{h}.csv" for h in protocol["horizons_minutes"]}
    report = screen_development(protocol, {h: pd.read_csv(path) for h, path in paths.items()})
    report["ledger_sha256"] = {str(h): hashlib.sha256(path.read_bytes()).hexdigest() for h, path in paths.items()}
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    get_logger(__name__).info("RESEARCH_FEASIBILITY_SCREENED", extra={"status": report["status"], "output": str(args.output)})


if __name__ == "__main__":
    main()
