"""Audit required observations for a draft data contract, without signals or returns."""
from __future__ import annotations

import argparse
import hashlib
import io
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


def observation_grid(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    if (proposal["lookback_sessions"], proposal["holding_sessions"], proposal["first_entry_session_index"]) != (60, 5, 61):
        raise ValueError("unsupported fixed observation grid")
    if not proposal["development_start"] <= proposal["development_end"] < proposal["protected_holdout_start"] <= proposal["protected_holdout_end"]:
        raise ValueError("development interval overlaps protected holdout")
    if not proposal["symbols"] or len(set(proposal["symbols"])) != len(proposal["symbols"]):
        raise ValueError("invalid symbol grain")
    days = [day.date() for day in pd.date_range(proposal["development_start"], proposal["development_end"]) if is_trading_day(day.date())]
    sessions = [session_info(day) for day in days]
    rows = []
    for i in range(61, len(days) - 5, 5):
        rows.append({"session": days[i].isoformat(), "quarter": str(pd.Timestamp(days[i]).to_period("Q")), "observations": {"lookback_close": (pd.Timestamp(sessions[i - 61].end_utc) - pd.Timedelta(minutes=1), "close"), "previous_close": (pd.Timestamp(sessions[i - 1].end_utc) - pd.Timedelta(minutes=1), "close"), "entry_open": (pd.Timestamp(sessions[i].start_utc), "open"), "exit_open": (pd.Timestamp(sessions[i + 5].start_utc), "open")}})
    return rows


def check_required_bars(frame: pd.DataFrame, grid: list[dict[str, Any]]) -> dict[str, Any]:
    """Test validity of required bars; deliberately never form a return or signal."""
    if frame.index.tz is None or frame.index.hasnans:
        raise ValueError("timestamps must be valid and timezone-aware")
    duplicates = set(frame.index[frame.index.duplicated(keep=False)])
    counts: Counter[str] = Counter()
    valid_sessions = []
    rejected = []
    for opportunity in grid:
        gaps = []
        for role, (timestamp, _) in opportunity["observations"].items():
            if timestamp in duplicates:
                reason = "duplicate_required_bar"
            elif timestamp not in frame.index:
                reason = "missing_required_bar"
            else:
                prices = pd.to_numeric(frame.loc[timestamp, ["open", "high", "low", "close"]], errors="coerce").to_numpy(dtype=float)
                opening, high, low, close = prices
                reason = "" if np.isfinite(prices).all() and (prices > 0).all() and low <= min(opening, close) <= max(opening, close) <= high else "invalid_required_ohlc"
            if reason:
                counts[f"{role}:{reason}"] += 1
                gaps.append(f"{role}:{reason}")
        if gaps:
            rejected.append({"session": opportunity["session"], "reasons": gaps})
        else:
            valid_sessions.append(opportunity["session"])
    return {"possible_periods": len(valid_sessions), "valid_entry_dates": valid_sessions, "rejection_counts": dict(counts), "rejected_periods": rejected}


def audit(proposal: dict[str, Any], acquisition_path: Path) -> dict[str, Any]:
    grid = observation_grid(proposal)
    acquisition_raw = acquisition_path.read_bytes()
    acquisition = json.loads(acquisition_raw)
    manifest_path = Path(acquisition["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = acquisition_path.parent / manifest_path
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    identity = manifest["dataset_identity"]
    interval = identity["request_interval"]
    if interval["start_date"] != proposal["development_start"] or interval["end_date"] != proposal["development_end"] or identity["feed"] != proposal["feed"] or identity["adjustment"] != proposal["audit_source_adjustment"] or identity["timeframe"] != proposal["timeframe"] or sorted(identity["symbols"]) != sorted(proposal["symbols"]):
        raise ValueError("acquisition does not match allowed development audit source")
    if acquisition.get("quality_passed") is not True or manifest.get("quality_passed") is not True:
        raise ValueError("governed acquisition quality unverified")
    results, sources = {}, {}
    common = {row["session"] for row in grid}
    required = {ts for row in grid for ts, _ in row["observations"].values()}
    for symbol in proposal["symbols"]:
        entries = [row for row in manifest["symbols"] if row["symbol"] == symbol]
        if len(entries) != 1:
            raise ValueError("missing or duplicate symbol provenance")
        entry = entries[0]
        path = Path(entry["csv_path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.resolve() != (manifest_path.parent / f"{symbol}.csv").resolve():
            raise ValueError("source outside governed dataset")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["content_sha256"]:
            raise ValueError(f"source hash mismatch:{symbol}")
        # Parse the exact byte snapshot that was hashed. No race between hash/read.
        frame = pd.read_csv(io.BytesIO(raw), usecols=["timestamp", "open", "high", "low", "close"], dtype=str)
        if not frame["timestamp"].str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", na=False).all():
            raise ValueError("source timezone unverified")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp"), utc=True, errors="raise"))
        if not frame.index.is_monotonic_increasing:
            raise ValueError("source timestamps are not chronological")
        results[symbol] = check_required_bars(frame.loc[frame.index.isin(required)], grid)
        common &= set(results[symbol]["valid_entry_dates"])
        sources[symbol] = {"path": str(path), "sha256": digest, "required_unique_bar_timestamps": len(required)}
    quarters = Counter(row["quarter"] for row in grid if row["session"] in common)
    thresholds = proposal["support_requirements"]
    criteria = {"possible_periods": len(common) >= thresholds["periods"], "maximum_selections": len(common) * len(proposal["symbols"]) >= thresholds["selected_opportunities"], "possible_supported_quarters": sum(n * len(proposal["symbols"]) >= 12 for n in quarters.values()) >= thresholds["quarters_with_12_selections"]}
    adjustment_matches = identity["adjustment"] == proposal["proposed_research_adjustment"]
    blockers = ["corporate_action_treatment_not_verified", "untouched_evaluation_plan_not_registered"]
    if not adjustment_matches:
        blockers.insert(0, "proposed_adjustment_contract_not_met")
    if not all(criteria.values()):
        blockers.insert(0, "required_observation_support_insufficient")
    return {"generated_at": datetime.now(UTC).isoformat(), "proposal_id": proposal["proposal_id"], "proposal_sha256": hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest(), "recommendation": "stop_before_registration", "observation_feasibility": "passes_upper_bound_screen" if all(criteria.values()) else "insufficient", "blockers": blockers, "candidate_periods": len(grid), "common_valid_periods": len(common), "maximum_possible_selections": len(common) * len(proposal["symbols"]), "actual_selections": None, "support_criteria": criteria, "periods_by_quarter": dict(sorted(quarters.items())), "common_entry_dates": sorted(common), "symbols": results, "baselines": {"common_mask_shared": True, "timing_shared": True, "allocation": proposal["allocation"], "interpretation": "Alignment by construction only; no baseline or strategy returns evaluated."}, "adjustments": {"observed": identity["adjustment"], "proposed": proposal["proposed_research_adjustment"], "matches": adjustment_matches}, "cost_status": proposal["costs"]["status"], "sources": sources, "acquisition_sha256": hashlib.sha256(acquisition_raw).hexdigest(), "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(), "prices_used_only_for_validity": True, "signals_computed": False, "returns_computed": False, "trial_claimed": False, "holdout_evaluated": False, "registration_authority": False, "promotion_authority": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(json.loads(args.proposal.read_text()), args.acquisition)
    atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    get_logger(__name__).info("ENDPOINT_FEASIBILITY_AUDITED", extra={"recommendation": report["recommendation"], "possible_periods": report["common_valid_periods"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
