"""One registered, nonoverlapping five-session ETF trend development trial."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.data.historical_bars import load_historical_bars
from ai_trading.data.training_provenance import validate_training_provenance
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.experiment_ledger import claim_campaign_trial, finish_campaign_trial, register_campaign
from ai_trading.tools.research_feasibility import audit_sampling_acquisition
from ai_trading.tools.train_replay_aligned_model import _resolve_training_input
from ai_trading.utils.market_calendar import is_trading_day, session_info

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config/slower_horizon_campaign.json"
STATE_PATH = ROOT / "artifacts/research_reset/campaign_state.json"


def build_opportunities(frame: pd.DataFrame, symbol: str, start: str, end: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dates = [day.date() for day in pd.date_range(start, end) if is_trading_day(day.date())]
    daily: dict[int, tuple[float, float]] = {}
    rejected: Counter[str] = Counter()
    for i, day in enumerate(dates):
        session = session_info(day)
        expected = pd.date_range(session.start_utc, session.end_utc, freq="min", inclusive="left")
        if not expected.isin(frame.index).all():
            rejected["incomplete_session"] += 1
            continue
        bars = frame.loc[expected]
        prices = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            rejected["invalid_prices"] += 1
            continue
        daily[i] = float(bars["open"].iloc[0]), float(bars["close"].iloc[-1])
    rows = []
    for i in range(61, len(dates) - 5, 5):
        # Missing sessions never shorten the lookback or move entry/exit dates.
        if any(j not in daily for j in range(i - 61, i + 6)):
            rejected["opportunity_has_incomplete_session"] += 1
            continue
        signal = daily[i - 1][1] / daily[i - 61][1] - 1
        rows.append({"symbol": symbol, "session": dates[i].isoformat(), "fold": str(pd.Timestamp(dates[i]).to_period("Q")), "selected": signal > 0, "signal_return": signal, "entry": session_info(dates[i]).start_utc.isoformat(), "exit": session_info(dates[i + 5]).start_utc.isoformat(), "gross_bps": (daily[i + 5][0] / daily[i][0] - 1) * 10000})
    return rows, dict(rejected)


def metrics(data: pd.DataFrame) -> dict[str, Any]:
    selected = data.loc[data["selected"], "gross_bps"]
    net = float((selected - 10).sum() / len(data))
    control = float((data["gross_bps"] - 10).mean())
    return {"opportunities": len(data), "selected": len(selected), "net_per_opportunity_bps": net, "net_per_selection_bps": float(selected.mean() - 10) if len(selected) else None, "gross_break_even_cost_bps": float(selected.mean()) if len(selected) else None, "cash_net_bps": 0, "always_long_net_bps": control, "paired_vs_cash_bps": net, "paired_vs_always_long_bps": net - control, "stress_net_bps": {str(cost): float((selected - cost).sum() / len(data)) for cost in (6, 10, 20)}}


def evaluate(data: pd.DataFrame, symbols: list[str]) -> dict[str, Any]:
    if data.empty:
        return {"decision": "inconclusive_budget_consumed", "reason": "no_common_opportunities"}
    if data.duplicated(["symbol", "session"]).any():
        raise ValueError("duplicate opportunity grain")
    counts = data.groupby("session")["symbol"].agg(set)
    common = counts[counts.apply(lambda value: value == set(symbols))].index
    data = data.loc[data["session"].isin(common)].copy()
    if data.empty:
        return {"decision": "inconclusive_budget_consumed", "reason": "no_common_opportunities"}
    aggregate = metrics(data)
    folds = {str(k): metrics(v) for k, v in data.groupby("fold")}
    by_symbol = {str(k): metrics(v) for k, v in data.groupby("symbol")}
    periods = data.assign(net=np.where(data["selected"], data["gross_bps"] - 10, 0), passive=data["gross_bps"] - 10).groupby("session")[["net", "passive"]].mean().sort_index()
    n = len(periods)
    rng = np.random.default_rng(20260908)
    starts = rng.integers(0, n, size=(10000, (n + 3) // 4))
    indices = ((starts[:, :, None] + np.arange(4)) % n).reshape(10000, -1)[:, :n]
    net = periods["net"].to_numpy()
    excess = (periods["net"] - periods["passive"]).to_numpy()
    if n < 8:
        return {"decision": "inconclusive_budget_consumed", "aggregate": aggregate, "folds": folds, "symbols": by_symbol, "decision_periods": n, "block_bootstrap_95_bps": None, "uncertainty_status": "unavailable_fewer_than_two_blocks", "criteria": {"adequate_support": False}}
    intervals = {"vs_cash": np.quantile(net[indices].mean(axis=1), [.025, .975]).tolist(), "vs_always_long": np.quantile(excess[indices].mean(axis=1), [.025, .975]).tolist()}
    support = n >= 60 and aggregate["selected"] >= 100 and sum(row["selected"] >= 12 for row in folds.values()) >= 6
    criteria = {"adequate_support": support, "positive_cash_lower_bound": intervals["vs_cash"][0] > 0, "positive_passive_lower_bound": intervals["vs_always_long"][0] > 0, "positive_at_20bps": aggregate["stress_net_bps"]["20"] > 0, "five_profitable_quarters": sum(row["net_per_opportunity_bps"] > 0 for row in folds.values()) >= 5, "four_profitable_symbols": sum(row["net_per_opportunity_bps"] > 0 for row in by_symbol.values()) >= 4}
    return {"decision": "eligible_for_untouched_test_planning" if all(criteria.values()) else "retired" if support else "inconclusive_budget_consumed", "aggregate": aggregate, "folds": folds, "symbols": by_symbol, "decision_periods": n, "block_bootstrap_95_bps": intervals, "criteria": criteria, "concentration": {"largest_symbol_share_of_absolute_net": max(abs(row["net_per_opportunity_bps"] * row["opportunities"]) for row in by_symbol.values()) / sum(abs(row["net_per_opportunity_bps"] * row["opportunities"]) for row in by_symbol.values()) if any(row["net_per_opportunity_bps"] for row in by_symbol.values()) else None}, "excluded_noncommon_opportunities": int(len(counts) - n)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/research_reset/study")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args(argv)
    contract = json.loads(CONTRACT_PATH.read_text())
    if contract["hypotheses"] != ["etf_60_session_trend_5_session_hold_v1"] or contract["lookback_sessions"] != 60 or contract["holding_sessions"] != 5 or contract["primary_round_trip_cost_bps"] != 10 or contract["stress_round_trip_cost_bps"] != [6, 10, 20]:
        raise ValueError("unsupported fixed study contract")
    state = register_campaign(STATE_PATH, contract)
    if args.register_only:
        get_logger(__name__).info("SLOWER_HORIZON_CAMPAIGN_REGISTERED")
        return 0
    if state["trials"]:
        raise ValueError("campaign budget exhausted; inspect existing immutable result")
    # Check acquisition identity before opening any price files, especially holdout data.
    manifest = json.loads(args.acquisition.read_text())
    provenance_path = Path(manifest.get("manifest_path") or "__missing__")
    if not provenance_path.is_absolute():
        provenance_path = args.acquisition.parent / provenance_path
    identity = json.loads(provenance_path.read_text()).get("dataset_identity", {})
    interval = identity.get("request_interval", {})
    if interval.get("start_date") != contract["development_start"] or interval.get("end_date") != contract["development_end"] or identity.get("feed") != contract["feed"] or identity.get("adjustment") != contract["adjustment"]:
        raise ValueError("acquisition differs from frozen development contract")
    support = audit_sampling_acquisition(contract, args.acquisition)
    atomic_write_text(args.output_dir / "sampling_support.json", json.dumps(support, indent=2))
    if support["status"] != "support_possible_not_strategy_eligible":
        atomic_write_text(args.output_dir / "report.json", json.dumps({"decision": "blocked_sampling_support", "budget_consumed": False, "sampling_support": support}, indent=2))
        return 2
    data_dir, provenance = _resolve_training_input(argparse.Namespace(acquisition_manifest_json=args.acquisition, symbols=",".join(contract["symbols"]), timestamp_col="timestamp"))
    quality = validate_training_provenance(data_dir, symbols=contract["symbols"], dataset_identity=identity, min_sessions=400, max_missing_ratio=.02)
    atomic_write_text(args.output_dir / "quality.json", json.dumps(quality, indent=2))
    if provenance.get("quality_passed") is not True or quality.get("quality_passed") is not True:
        atomic_write_text(args.output_dir / "report.json", json.dumps({"decision": "blocked_data_quality", "budget_consumed": False, "quality": quality}, indent=2))
        return 2
    claim_campaign_trial(STATE_PATH, hypothesis_id=contract["hypotheses"][0], evidence_signature=provenance["dataset_hash"], evaluation_start=contract["development_start"], evaluation_end=contract["development_end"], quality_passed=True)
    rows, rejected = [], {}
    for symbol in contract["symbols"]:
        frame, _ = load_historical_bars(data_dir / f"{symbol}.csv", timestamp_col="timestamp", require_timestamp=True)
        opportunities, rejected[symbol] = build_opportunities(frame, symbol, contract["development_start"], contract["development_end"])
        rows.extend(opportunities)
    data = pd.DataFrame(rows)
    report = evaluate(data, contract["symbols"])
    report.update(generated_at=datetime.now(UTC).isoformat(), contract=contract, dataset_hash=provenance["dataset_hash"], rejection_counts=rejected, models_fitted=0, ml_incremental_value="not_established_no_model_comparison", holdout_evaluated=False, promotion_authority=False, orders_sent=0, metric="equal_notional_split_adjusted_price_markouts_not_total_portfolio_returns")
    atomic_write_text(args.output_dir / "opportunities.csv", data.to_csv(index=False))
    atomic_write_text(args.output_dir / "report.json", json.dumps(report, indent=2, allow_nan=False))
    finish_campaign_trial(STATE_PATH, evidence_signature=provenance["dataset_hash"], decision=report["decision"], report_path=(args.output_dir / "report.json").resolve())
    get_logger(__name__).info("SLOWER_HORIZON_STUDY_COMPLETE", extra={"decision": report["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
