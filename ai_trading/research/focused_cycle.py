"""Fixed-grid research accounting for the preregistered intraday study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.data.historical_bars import load_historical_bars
from ai_trading.data.training_provenance import validate_training_provenance
from ai_trading.logging import get_logger
from ai_trading.models.contracts import infer_day_sleeve_regimes
from ai_trading.runtime.atomic_io import atomic_write_text

logger = get_logger(__name__)
REGISTRY_PATH = Path(__file__).resolve().parents[2] / "docs" / "FOCUSED_RESEARCH_REGISTRY.json"


def selected_signals(bars: pd.DataFrame, features: pd.DataFrame, rule: str) -> pd.Series:
    """Fixed rules only; shock volatility excludes the measured shock window."""
    if rule == "positive_sma_macd":
        return ((features["sma_spread"] > 0) & (features["macd_signal_gap"] > 0)).where(
            np.isfinite(features[["sma_spread", "macd_signal_gap"]]).all(axis=1))
    if rule != "negative_15m_shock_2sigma_prior60":
        raise ValueError(f"unregistered signal rule: {rule}")
    close = bars["close"].where(bars["close"] > 0)
    returns = np.log(close).diff()
    volatility = returns.shift(15).rolling(60, min_periods=60).std(ddof=1)
    shock = np.log(close / close.shift(15))
    timestamps = pd.Series(bars.index, index=bars.index)
    dates = pd.Series(bars.index.tz_convert("America/New_York").date, index=bars.index)
    valid = ((timestamps - timestamps.shift(75)) == pd.Timedelta(minutes=75)) & (dates == dates.shift(75))
    valid &= np.isfinite(shock) & np.isfinite(volatility) & (volatility > 0)
    return (shock <= -2 * np.sqrt(15) * volatility).where(valid)


def check_study_permission(protocol: dict[str, Any], output_dir: Path) -> None:
    """Completed grids cannot be repeatedly tested under a new protocol name."""
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rule = protocol.get("signal_rule", "positive_sma_macd")
    for record in registry["completed_families"]:
        if record["signal_rule"] == rule and record["universe"] == protocol["universe"]:
            if set(record["closed_horizons_minutes"]) & set(protocol["horizons_minutes"]):
                raise ValueError("completed research variants are closed; record a different hypothesis")
    if (output_dir / "study_report.json").exists():
        raise ValueError("study output already exists; preserve completed evidence")
    from ai_trading.tools.research_feasibility import require_feasibility

    require_feasibility(protocol)


def build_opportunity_ledger(
    bars: pd.DataFrame, features: pd.DataFrame, *, horizon_minutes: int,
    entry_start: str = "10:00", flat_by: str = "15:00",
    signal_rule: str = "positive_sma_macd",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Use past-only signals, next-minute opens and exact contiguous session bars."""
    if horizon_minutes < 1:
        raise ValueError("horizon_minutes must be positive")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("bars require timezone-aware timestamps")
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted and unique")
    if not features.index.equals(bars.index):
        raise ValueError("features and bars must share the same index")
    signals = selected_signals(bars, features, signal_rule)
    local = bars.index.tz_convert("America/New_York")
    start_hour, start_minute = map(int, entry_start.split(":"))
    end_hour, end_minute = map(int, flat_by.split(":"))
    start = start_hour * 60 + start_minute
    end = end_hour * 60 + end_minute
    diagnostics = {"grid_opportunities": 0, "insufficient_warmup": 0, "missing_contiguous_bars": 0, "outside_session_cutoff": 0, "invalid_fields": 0}
    rows = []
    for index, timestamp in enumerate(bars.index):
        minute = local[index].hour * 60 + local[index].minute
        if minute < start or minute >= end or (minute - start) % horizon_minutes:
            continue
        diagnostics["grid_opportunities"] += 1
        if index < 200:
            diagnostics["insufficient_warmup"] += 1
            continue
        entry = timestamp + pd.Timedelta(minutes=1)
        exit_time = entry + pd.Timedelta(minutes=horizon_minutes)
        exit_local = exit_time.tz_convert("America/New_York")
        if exit_local.date() != local[index].date() or exit_local.hour * 60 + exit_local.minute > end:
            diagnostics["outside_session_cutoff"] += 1
            continue
        expected = pd.date_range(timestamp, exit_time, freq="min")
        actual = bars.index[index:index + len(expected)]
        if not actual.equals(expected):
            diagnostics["missing_contiguous_bars"] += 1
            continue
        entry_price = float(bars.loc[entry, "open"])
        exit_price = float(bars.loc[exit_time, "open"])
        selected = signals.iloc[index]
        if pd.isna(selected) or not np.isfinite([entry_price, exit_price]).all() or min(entry_price, exit_price) <= 0:
            diagnostics["invalid_fields"] += 1
            continue
        rows.append({
            "signal_timestamp": timestamp.isoformat(), "entry_timestamp": entry.isoformat(),
            "exit_timestamp": exit_time.isoformat(), "session": str(local[index].date()),
            "horizon_minutes": horizon_minutes, "selected": bool(selected),
            "entry_price": entry_price, "exit_price": exit_price,
            "gross_return_bps": (exit_price / entry_price - 1) * 10_000,
        })
    return pd.DataFrame(rows, columns=["signal_timestamp", "entry_timestamp", "exit_timestamp", "session", "horizon_minutes", "selected", "entry_price", "exit_price", "gross_return_bps"]), diagnostics


def summarize_period(
    ledger: pd.DataFrame, *, sessions: list[str], one_way_cost_bps: float,
) -> dict[str, Any]:
    """Account at equal initial notional; abstention has zero P&L and zero turnover."""
    if not np.isfinite(one_way_cost_bps) or one_way_cost_bps < 0:
        raise ValueError("one-way cost must be finite and nonnegative")
    period = ledger.loc[ledger["session"].isin(sessions)].copy()
    gross = period["gross_return_bps"].astype(float)
    net = gross - 2 * one_way_cost_bps
    selected = period["selected"].astype(bool)
    candidate = net.where(selected, 0.0).groupby(period["session"]).sum().reindex(sessions, fill_value=0.0)
    baseline = net.groupby(period["session"]).sum().reindex(sessions, fill_value=0.0)
    selected_gross = gross.loc[selected]
    return {
        "sessions": len(sessions), "opportunities": len(period), "completed_trades": int(selected.sum()),
        "one_way_cost_bps": one_way_cost_bps,
        "mean_net_trade_bps": float(net.loc[selected].mean()) if selected.any() else None,
        "break_even_one_way_cost_bps": float(selected_gross.mean() / 2) if len(selected_gross) else None,
        "mean_daily_net_bps": float(candidate.mean()) if len(candidate) else None,
        "mean_daily_baseline_bps": float(baseline.mean()) if len(baseline) else None,
        "mean_daily_improvement_bps": float((candidate - baseline).mean()) if len(candidate) else None,
        "daily": [{"session": day, "candidate_net_bps": float(candidate.loc[day]), "baseline_net_bps": float(baseline.loc[day])} for day in sessions],
        "accounting": "fixed_initial_notional_nonoverlapping_positions_no_compounding",
        "execution_evidence": "simulated_open_prices_plus_assumed_costs",
    }


def adjusted_daily_bounds(daily: list[dict[str, Any]], *, settings: dict[str, Any]) -> dict[str, Any]:
    """Resample contiguous session blocks, keeping strategy and control paired."""
    if len(daily) < 40:
        return {"status": "insufficient_sessions", "net_lower_bps": None, "improvement_lower_bps": None}
    candidate = np.asarray([row["candidate_net_bps"] for row in daily], dtype=float)
    baseline = np.asarray([row["baseline_net_bps"] for row in daily], dtype=float)
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("bootstrap requires finite daily P&L")
    length = int(settings["bootstrap_block_sessions"])
    replicates = int(settings["bootstrap_replicates"])
    if length < 1 or replicates < 100 or length > len(daily):
        raise ValueError("invalid bootstrap block or replicate count")
    rng = np.random.default_rng(int(settings["bootstrap_seed"]))
    starts = rng.integers(0, len(daily) - length + 1, size=(replicates, int(np.ceil(len(daily) / length))))
    indices = (starts[..., None] + np.arange(length)).reshape(replicates, -1)[:, :len(daily)]
    net_means = candidate[indices].mean(axis=1)
    improvement_means = (candidate - baseline)[indices].mean(axis=1)
    alpha = float(settings["familywise_alpha"]) / int(settings["simultaneous_bounds"])
    return {
        "status": "estimated", "method": "paired_moving_session_block_percentile_bootstrap",
        "net_lower_bps": float(np.quantile(net_means, alpha)),
        "improvement_lower_bps": float(np.quantile(improvement_means, alpha)),
        "per_bound_alpha": alpha, "block_sessions": length, "replicates": replicates,
        "scope": "this_preregistered_family_only_not_all_prior_project_searches",
    }


def run_study(protocol_path: Path, manifests: list[Path], output_dir: Path) -> dict[str, Any]:
    from ai_trading.tools.train_replay_aligned_model import _feature_frame

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["universe"] != ["AAPL"]:
        raise ValueError("this fixed study only supports the registered AAPL universe")
    check_study_permission(protocol, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    ordered = sorted(manifests, key=lambda path: json.loads(path.read_text())["dataset_identity"]["request_interval"]["end_date"])
    frames = []
    conflicting_closes = 0
    previous = pd.DataFrame()
    for path in ordered:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        row = next(item for item in manifest["symbols"] if item["symbol"] == "AAPL")
        source = path.parent / "AAPL.csv"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != row["content_sha256"] or row["quality_passed"] is not True:
            raise ValueError(f"AAPL source identity/quality failed: {path}")
        bars, load_report = load_historical_bars(source, require_timestamp=True)
        if not previous.empty:
            overlap = previous.index.intersection(bars.index)
            conflicting_closes += int((previous.loc[overlap, "close"] != bars.loc[overlap, "close"]).sum())
        frames.append(bars)
        previous = pd.concat(frames)
        previous = previous.loc[~previous.index.duplicated(keep="last")].sort_index()
        sources.append({"manifest": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "csv_sha256": digest, "parent_quality_passed": manifest["quality_passed"], "symbol_quality_passed": row["quality_passed"], "loader": load_report.as_dict()})
    if not frames:
        raise ValueError("at least one source manifest is required")
    cutoff = pd.Timestamp(protocol["untouched_final_period"]["start"], tz="America/New_York")
    bars = previous.loc[previous.index < cutoff].copy()
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    bars.rename_axis("timestamp").reset_index().to_csv(data_dir / "AAPL.csv", index=False)
    intervals = protocol["retrospective_intervals"]
    provenance = validate_training_provenance(data_dir, symbols=["AAPL"], dataset_identity={"request_interval": {"start_date": intervals[0]["start"], "end_date": intervals[-1]["end"]}}, min_sessions=200, max_missing_ratio=0.02)
    if not provenance["quality_passed"]:
        raise ValueError(f"combined coverage failed: {provenance['rejection_reasons']}")
    features = _feature_frame(bars, symbol="AAPL")
    session_dates = provenance["coverage"]["AAPL"]["observed_session_dates"]
    local_dates = bars.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
    regime_labels = pd.Series(infer_day_sleeve_regimes(bars["close"]), index=bars.index)
    market_conditions = {}
    for interval in intervals:
        mask = (local_dates >= interval["start"]) & (local_dates <= interval["end"])
        period_bars = bars.loc[mask]
        groups = period_bars.groupby(period_bars.index.tz_convert("America/New_York").date)
        daily_changes = (groups["close"].last() / groups["open"].first() - 1) * 10000
        market_conditions[interval["name"]] = {
            "sessions": len(daily_changes), "mean_open_to_close_bps": float(daily_changes.mean()),
            "daily_open_to_close_std_bps": float(daily_changes.std()),
            "positive_session_fraction": float((daily_changes > 0).mean()),
            "regime_distribution": {str(name): int(count) for name, count in regime_labels.loc[mask].value_counts().items()},
            "scope": "descriptive_only_not_used_to_change_entry_rule",
        }
    evaluations = []
    for horizon in protocol["horizons_minutes"]:
        ledger, diagnostics = build_opportunity_ledger(bars, features, horizon_minutes=int(horizon), entry_start=protocol["entry_start"], flat_by=protocol["flat_by"], signal_rule=protocol.get("signal_rule", "positive_sma_macd"))
        ledger.to_csv(output_dir / f"opportunities_h{horizon}.csv", index=False)
        period_results = []
        daily_validation = []
        for interval in intervals:
            dates = [day for day in session_dates if interval["start"] <= day <= interval["end"]]
            scenarios = [summarize_period(ledger, sessions=dates, one_way_cost_bps=float(cost)) for cost in [*protocol["one_way_cost_scenarios_bps"], 32.58272830737331]]
            primary = next(row for row in scenarios if row["one_way_cost_bps"] == protocol["acceptance"]["primary_one_way_cost_bps"])
            period_results.append({"name": interval["name"], "scenarios": scenarios})
            if interval["name"] != "development":
                daily_validation.extend(primary["daily"])
        bounds = adjusted_daily_bounds(daily_validation, settings=protocol["acceptance"])
        validations = [next(row for row in result["scenarios"] if row["one_way_cost_bps"] == protocol["acceptance"]["primary_one_way_cost_bps"]) for result in period_results if result["name"] != "development"]
        enough_sessions = all(row["sessions"] >= protocol["acceptance"]["minimum_validation_sessions_per_interval"] for row in validations)
        enough_trades = sum(row["completed_trades"] for row in validations) >= protocol["acceptance"]["minimum_completed_validation_trades"]
        positive_intervals = sum(row["mean_daily_net_bps"] is not None and row["mean_daily_net_bps"] > 0 for row in validations)
        passes = bool(enough_sessions and enough_trades and positive_intervals >= protocol["acceptance"]["positive_validation_intervals_required"] and bounds["status"] == "estimated" and bounds["net_lower_bps"] > 0 and bounds["improvement_lower_bps"] > 0)
        evaluations.append({"horizon_minutes": horizon, "opportunity_diagnostics": diagnostics, "periods": period_results, "adjusted_bounds": bounds, "coverage_sufficient": enough_sessions and enough_trades, "positive_validation_intervals": positive_intervals, "retrospective_passed": passes, "disposition": "await_untouched_confirmation" if passes else "retire_tested_variant" if enough_sessions and enough_trades else "inconclusive_do_not_promote"})
    decision = "revise" if any(row["retrospective_passed"] for row in evaluations) or not all(row["coverage_sufficient"] for row in evaluations) else "retire"
    report = {"protocol": protocol, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(), "sources": sources, "conflicting_overlap_closes": conflicting_closes, "provenance": provenance, "evaluations": evaluations, "decision": decision, "decision_reason": "retrospective_survivor_requires_untouched_and_execution_confirmation" if any(row["retrospective_passed"] for row in evaluations) else "insufficient_support" if decision == "revise" else "no_predeclared_horizon_passed", "untouched_final_period": {**protocol["untouched_final_period"], "consumed": False}, "promotion_authority": False, "execution_validation": "separate_reconciliation_required"}
    report["market_conditions"] = market_conditions
    report["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["feature_implementation_sha256"] = hashlib.sha256((Path(__file__).parents[1] / "tools" / "train_replay_aligned_model.py").read_bytes()).hexdigest()
    atomic_write_text(output_dir / "study_report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info("FOCUSED_RESEARCH_COMPLETE", extra={"decision": decision, "report": str(output_dir / "study_report.json")})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_study(args.protocol, args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
