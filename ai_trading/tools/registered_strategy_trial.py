"""One registered daily-rebalance development trial of an unmodified registry strategy; never trades."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.experiment_ledger import (
    claim_campaign_trial, finish_campaign_trial, register_campaign,
)
from ai_trading.tools.research_feasibility import (
    DAILY_FOLD_KIND, HOLDOUT_YEAR_START, NEW_YORK, DailyGrid, daily_development_dates,
    daily_rebalance_support, load_governed_development_timestamps, quarter_fold,
    regular_session_minutes, strategy_bucket, usable_daily_periods,
)
from ai_trading.tools.stock_development_readiness import valid_bars

LOGGER = get_logger(__name__)
RESET_CONFIG = Path("config/research_reset.json")
COST_ROLES = ("decision", "required_positive_aggregate", "reported_only")
CODE_FILES = ("tools/registered_strategy_trial.py", "tools/research_feasibility.py",
              "tools/stock_development_readiness.py", "tools/experiment_ledger.py",
              "strategies/cross_sectional_momentum.py", "strategies/base.py", "utils/market_calendar.py")
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class CostTier:
    name: str
    round_trip_bps: float
    role: str


@dataclass(frozen=True)
class StrategyTrialSpec:
    """Contract values that fix the strategy, grid, costs, thresholds and bootstrap before evaluation."""

    strategy: str
    grid: DailyGrid
    tiers: tuple[CostTier, ...]
    block: int
    replicates: int
    seed: int
    min_periods: int
    min_selections: int
    min_selections_per_fold: int
    min_supported_folds: int
    min_profitable_folds: int

    @property
    def primary(self) -> CostTier:
        return next(t for t in self.tiers if t.role == "decision")

    @property
    def stress(self) -> CostTier:
        return next(t for t in self.tiers if t.role == "required_positive_aggregate")


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return int(value)


def build_strategy(contract: dict[str, Any]) -> Any:
    """Construct the registry class with no arguments and prove its defaults match the contract."""
    from ai_trading.strategies import REGISTRY

    key = contract.get("strategy")
    if key not in REGISTRY:
        raise ValueError("strategy is not in the runtime registry")
    expected = contract.get("strategy_parameters")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("contract must declare strategy_parameters")
    strategy = REGISTRY[key]()
    missing = object()
    if any(getattr(strategy, name, missing) != value for name, value in expected.items()):
        raise ValueError("strategy constructor defaults differ from contract")
    if not callable(getattr(strategy, "generate_signals", None)):
        raise ValueError("strategy lacks a pure generate_signals")
    return strategy


def contract_spec(contract: dict[str, Any]) -> StrategyTrialSpec:
    """Validate everything that must be fixed before data are read; fail closed on any ambiguity."""
    if not re.fullmatch(r"[a-z0-9_]+", str(contract.get("campaign_id", ""))):
        raise ValueError("contract campaign_id must name its artifact directory")
    if contract.get("signal_side") != "buy":
        raise ValueError("only long-only buy signals are supported")
    if contract.get("promotion_authority") is not False:
        raise ValueError("a development trial never carries promotion authority")
    daily_development_dates(contract)  # Holdout and holdout-year refusal precede any file read.
    grid = DailyGrid.from_protocol(contract)
    strategy = build_strategy(contract)
    if getattr(strategy, "lookback", None) != grid.lookback_sessions:
        raise ValueError("rebalance lookback must equal the strategy lookback")
    if contract.get("folds") != DAILY_FOLD_KIND:
        raise ValueError(f"folds must be {DAILY_FOLD_KIND}")
    if strategy_bucket(contract) == 0:
        raise ValueError("universe is below the strategy min_universe")
    raw_tiers = contract.get("cost_tiers")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        raise ValueError("contract must declare cost_tiers")
    tiers = []
    for tier in raw_tiers:
        cost = tier.get("round_trip_bps")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise ValueError("cost tier round_trip_bps must be finite and nonnegative")
        if tier.get("role") not in COST_ROLES or not isinstance(tier.get("name"), str):
            raise ValueError("cost tier needs a name and a supported role")
        tiers.append(CostTier(tier["name"], float(cost), tier["role"]))
    roles = Counter(t.role for t in tiers)
    if len({t.name for t in tiers}) != len(tiers) or roles["decision"] != 1 or roles["required_positive_aggregate"] != 1:
        raise ValueError("cost tiers need unique names, one decision tier and one required stress tier")
    primary = next(t for t in tiers if t.role == "decision")
    if primary.round_trip_bps != float(contract.get("primary_round_trip_cost_bps", math.nan)):
        raise ValueError("decision tier must equal primary_round_trip_cost_bps")
    acceptance, thresholds, bootstrap = contract["acceptance"], contract["support_thresholds"], contract["bootstrap"]
    if not all(acceptance.get(k) is True for k in ("positive_cash_lower_bound_at_primary", "positive_always_long_lower_bound", "positive_aggregate_at_stress_high")):
        raise ValueError("acceptance criteria cannot be disabled")
    replicates = _positive_int(bootstrap.get("replicates"), "bootstrap replicates")
    if replicates % 100:
        raise ValueError("bootstrap replicates must be a positive multiple of 100")
    seed = bootstrap.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    return StrategyTrialSpec(
        strategy=contract["strategy"], grid=grid, tiers=tuple(tiers),
        block=_positive_int(bootstrap.get("block_periods"), "bootstrap block_periods"), replicates=replicates, seed=seed,
        min_periods=_positive_int(thresholds.get("minimum_decision_periods"), "minimum_decision_periods"),
        min_selections=_positive_int(thresholds.get("minimum_selections"), "minimum_selections"),
        min_selections_per_fold=_positive_int(thresholds.get("minimum_selections_per_fold"), "minimum_selections_per_fold"),
        min_supported_folds=_positive_int(thresholds.get("minimum_supported_folds"), "minimum_supported_folds"),
        min_profitable_folds=_positive_int(acceptance.get("minimum_profitable_folds"), "minimum_profitable_folds", allow_zero=True))


def check_development_bounds(indices: dict[str, pd.DatetimeIndex], contract: dict[str, Any]) -> None:
    """Refuse any source row outside development, in the holdout or in 2026, before a price is read."""
    lower = pd.Timestamp(contract["development_start"], tz=NEW_YORK)
    upper = min(pd.Timestamp(contract["development_end"], tz=NEW_YORK) + pd.Timedelta(days=1),
                pd.Timestamp(contract["holdout_start"], tz=NEW_YORK), pd.Timestamp(HOLDOUT_YEAR_START, tz=NEW_YORK))
    for symbol, index in indices.items():
        if index.tz is None or (len(index) and (index.min() < lower or index.max() >= upper)):
            raise ValueError(f"source timestamps outside development window:{symbol}")


def daily_bars(minutes: pd.DataFrame, dates: list[date], grid: DailyGrid) -> tuple[pd.DataFrame, dict[str, int]]:
    """Regular-session OHLCV per development session; incomplete or invalid sessions stay NaN and incomplete."""
    index = minutes.index
    if index.tz is None or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("source timestamps must be timezone-aware, unique and ordered")
    rows, rejected = [], Counter[str]()
    for day, expected in zip(dates, regular_session_minutes(dates)):
        part = minutes.iloc[index.searchsorted(expected[0]):index.searchsorted(expected[-1], side="right")]
        row: dict[str, Any] = dict(session=day.isoformat(), open=np.nan, high=np.nan, low=np.nan,
                                   close=np.nan, volume=np.nan, complete=False)
        if not part.index.isin(expected).all():
            rejected["off_grid_timestamps"] += 1
        elif not grid.session_complete(expected.isin(part.index)):
            rejected["incomplete_session"] += 1
        elif not valid_bars(part).all():
            rejected["invalid_prices"] += 1
        else:
            row.update(open=float(part.loc[expected[0], "open"]), high=float(part["high"].max()),
                       low=float(part["low"].min()), close=float(part.loc[expected[-1], "close"]),
                       volume=float(part["volume"].sum()), complete=True)
        rows.append(row)
    return pd.DataFrame(rows).set_index("session"), dict(rejected)


def point_in_time_prices(closes: dict[str, pd.Series], decision: int) -> dict[str, pd.Series]:
    """Each symbol's daily closes through the decision session only; nothing later is visible."""
    return {symbol: series.iloc[:decision + 1] for symbol, series in closes.items()}


def build_periods(panel: dict[str, pd.DataFrame], dates: list[date], grid: DailyGrid, strategy: Any,
                  symbols: list[str]) -> pd.DataFrame:
    """Call generate_signals at each usable decision, keep buys, and record open-to-open markouts."""
    schedule = grid.schedule(len(dates))
    periods = usable_daily_periods(grid, schedule, {s: panel[s]["complete"].to_numpy(dtype=bool) for s in symbols}, symbols)
    closes = {s: panel[s]["close"] for s in symbols}
    opens = {s: panel[s]["open"].to_numpy(dtype=float) for s in symbols}
    rows = []
    for decision, entry, exit_at in periods:
        prices = point_in_time_prices(closes, decision)
        if any(series.index[-1] != dates[decision].isoformat() for series in prices.values()):
            raise ValueError("price history is not point-in-time")
        buys = sorted({str(sig.symbol) for sig in strategy.generate_signals({"prices": prices}) if sig.side == "buy"})
        if set(buys) - set(symbols):
            raise ValueError("strategy selected a symbol outside the universe")
        gross = {s: (opens[s][exit_at] / opens[s][entry] - 1) * 10000 for s in symbols}
        rows.append(dict(decision_session=dates[decision].isoformat(), entry_session=dates[entry].isoformat(),
                         exit_session=dates[exit_at].isoformat(), decision_index=decision, entry_index=entry,
                         exit_index=exit_at, fold=quarter_fold(dates[entry]), selected="|".join(buys),
                         selections=len(buys),
                         strategy_gross_bps=float(np.mean([gross[s] for s in buys])) if buys else 0.0,
                         control_gross_bps=float(np.mean(list(gross.values()))),
                         **{f"gross_bps_{s}": gross[s] for s in symbols}))
    return pd.DataFrame(rows)


def validate_periods(periods: pd.DataFrame, contract: dict[str, Any], grid: DailyGrid) -> None:
    if periods.empty or periods.decision_index.duplicated().any():
        raise ValueError("empty or duplicate decision periods")
    if not (periods.entry_index == periods.decision_index + 1).all():
        raise ValueError("entry must be the session after the decision")
    if not (periods.exit_index == periods.entry_index + grid.hold_sessions).all():
        raise ValueError("hold horizon mismatch")
    ordered = periods.sort_values("entry_index")
    if (ordered.entry_index.to_numpy()[1:] < ordered.exit_index.to_numpy()[:-1]).any():
        raise ValueError("overlapping holds")
    if not (periods.decision_index >= grid.lookback_sessions).all():
        raise ValueError("decision before the full lookback")
    if not (periods.decision_session.ge(contract["development_start"]) & periods.exit_session.le(contract["development_end"])).all():
        raise ValueError("period outside development")
    numeric = periods[["strategy_gross_bps", "control_gross_bps"]].to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("nonfinite markouts")


def tier_returns(periods: pd.DataFrame, cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-period net: each selected slot pays one round trip; abstention is cash; the control pays the same."""
    strategy = np.where(periods.selections.to_numpy() > 0, periods.strategy_gross_bps.to_numpy() - cost, 0.0)
    return strategy, periods.control_gross_bps.to_numpy() - cost


def bootstrap_indices(n: int, *, block: int, replicates: int, seed: int) -> np.ndarray | None:
    """Circular block resampling of decision periods; None when fewer than two blocks exist."""
    if n < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(replicates, math.ceil(n / block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(replicates, -1)[:, :n]


def evaluate(periods: pd.DataFrame, spec: StrategyTrialSpec) -> dict[str, Any]:
    tiers: dict[str, dict[str, Any]] = {}
    for tier in spec.tiers:
        strategy, control = tier_returns(periods, tier.round_trip_bps)
        selected = periods.selections > 0
        tiers[tier.name] = dict(round_trip_bps=tier.round_trip_bps, role=tier.role,
                                net_per_period_bps=float(strategy.mean()),
                                net_per_selected_period_bps=float(strategy[selected].mean()) if selected.any() else None,
                                always_long_net_per_period_bps=float(control.mean()), cash_net_per_period_bps=0.0,
                                paired_vs_always_long_bps=float((strategy - control).mean()))
    strategy, control = tier_returns(periods, spec.primary.round_trip_bps)
    frame = periods.assign(net=strategy)
    folds: dict[str, dict[str, Any]] = {str(k): dict(periods=len(v), selections=int(v.selections.sum()), net_per_period_bps=float(v.net.mean()))
             for k, v in frame.groupby("fold")}
    supported_folds = sum(f["selections"] >= spec.min_selections_per_fold for f in folds.values())
    support = (len(periods) >= spec.min_periods and int(periods.selections.sum()) >= spec.min_selections
               and supported_folds >= spec.min_supported_folds)
    selections = Counter(s for row in periods.selected for s in row.split("|") if s)
    base = dict(decision_periods=len(periods), selections=int(periods.selections.sum()),
                abstentions=int((periods.selections == 0).sum()), supported_folds=supported_folds,
                folds=folds, cost_tiers=tiers, selections_by_symbol=dict(sorted(selections.items())))
    indices = bootstrap_indices(len(periods), block=spec.block, replicates=spec.replicates, seed=spec.seed)
    if indices is None:
        return dict(base, decision="inconclusive_budget_consumed", block_bootstrap_95_bps=None,
                    uncertainty_status="unavailable_fewer_than_two_blocks", criteria={"adequate_support": False})
    excess = strategy - control
    intervals = {"vs_cash": np.quantile(strategy[indices].mean(axis=1), [.025, .975]).tolist(),
                 "vs_always_long": np.quantile(excess[indices].mean(axis=1), [.025, .975]).tolist()}
    criteria = {"adequate_support": bool(support),
                "positive_cash_lower_bound_at_primary": intervals["vs_cash"][0] > 0,
                "positive_always_long_lower_bound": intervals["vs_always_long"][0] > 0,
                "positive_aggregate_at_stress_high": tiers[spec.stress.name]["net_per_period_bps"] > 0,
                "profitable_folds": sum(f["net_per_period_bps"] > 0 for f in folds.values()) >= spec.min_profitable_folds}
    decision = ("eligible_for_untouched_test_planning" if all(criteria.values())
                else "retired" if support else "inconclusive_budget_consumed")
    return dict(base, decision=decision, block_bootstrap_95_bps=intervals, criteria=criteria,
                decision_tier=spec.primary.name, reported_only_tiers=[t.name for t in spec.tiers if t.role == "reported_only"])


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def evidence_signature(contract_hash: str, provenance: dict[str, Any], code_hashes: dict[str, str]) -> str:
    stable = {key: value for key, value in provenance.items() if key != "generated_at"}
    return hashlib.sha256(json.dumps(dict(contract=contract_hash, provenance=stable, code=code_hashes),
                                     sort_keys=True).encode()).hexdigest()


def load_daily_panel(contract: dict[str, Any], provenance: dict[str, Any], indices: dict[str, pd.DatetimeIndex],
                     dates: list[date], grid: DailyGrid) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    """Read prices only from hash-verified sources whose timestamps were already audited."""
    panel, rejected = {}, {}
    for symbol in contract["symbols"]:
        source = provenance["sources"][symbol]
        path = Path(source["path"])
        if digest(path) != source["sha256"]:
            raise ValueError(f"source changed before price read:{symbol}")
        minutes = pd.read_csv(path, usecols=["timestamp", *PRICE_COLUMNS])
        minutes.index = pd.DatetimeIndex(pd.to_datetime(minutes.pop("timestamp"), utc=True))
        if digest(path) != source["sha256"] or not minutes.index.equals(indices[symbol]):
            raise ValueError(f"source changed during price read:{symbol}")
        panel[symbol], rejected[symbol] = daily_bars(minutes, dates, grid)
    return panel, rejected


def run(root: Path, *, fit: bool, contract_path: Path) -> dict[str, Any]:
    root = root.resolve()
    contract = json.loads((root / contract_path).resolve().read_text())
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if contract.get("status") != "registered":
        raise ValueError("contract is not registered; a draft cannot be preflighted or fitted")
    spec = contract_spec(contract)
    reset = json.loads((root / RESET_CONFIG).read_text())
    if contract["campaign_id"] not in reset.get("registered_trials", []):
        raise ValueError("research reset does not list this campaign in registered_trials")
    directory = root / "artifacts" / contract["campaign_id"]
    state_path = directory / "campaign_state.json"
    # Require the original registration. A different root must not create a budget.
    if not state_path.is_file():
        raise ValueError("original campaign registration is required")
    state = register_campaign(state_path, contract)
    if state["trials"]:
        return {"status": "budget_already_claimed", "trials": state["trials"]}
    indices, provenance = load_governed_development_timestamps(contract, root / contract["acquisition"])
    check_development_bounds(indices, contract)
    support = daily_rebalance_support(contract, indices)
    support.update(provenance)
    write_json(directory / "sampling_support.json", support)
    blocked = dict(trial_claimed=False, holdout_evaluated=False, promotion_authority=False)
    if support["status"] != "support_possible_not_strategy_eligible":
        write_json(directory / "preflight.json", dict(blocked, status="blocked_sampling_support", reasons=support["blocking_reasons"]))
        return dict(blocked, status="blocked_sampling_support", reasons=support["blocking_reasons"])
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = {p: digest(code_root / p) for p in CODE_FILES}
    signature = evidence_signature(canonical, provenance, code_hashes)
    dates = daily_development_dates(contract)
    panel, rejected = load_daily_panel(contract, provenance, indices, dates, spec.grid)
    symbols = list(contract["symbols"])
    periods = usable_daily_periods(spec.grid, spec.grid.schedule(len(dates)),
                                   {s: panel[s]["complete"].to_numpy(dtype=bool) for s in symbols}, symbols)
    bucket = strategy_bucket(contract)
    per_fold = Counter(quarter_fold(dates[entry]) for _, entry, _ in periods)
    priced_support = (len(periods) >= spec.min_periods and len(periods) * bucket >= spec.min_selections
                      and sum(n * bucket >= spec.min_selections_per_fold for n in per_fold.values()) >= spec.min_supported_folds)
    preflight = dict(blocked, status="preflight_passed" if priced_support else "blocked_price_validated_support",
                     signature=signature, contract_hash=canonical, possible_periods=len(periods),
                     possible_selections=len(periods) * bucket, periods_by_fold=dict(sorted(per_fold.items())),
                     rejection_counts=rejected, provenance=provenance, code=code_hashes)
    write_json(directory / "preflight.json", preflight)
    if not fit or not priced_support:
        return preflight
    claim_campaign_trial(state_path, hypothesis_id=contract["hypotheses"][0], evidence_signature=signature,
                         evaluation_start=contract["development_start"], evaluation_end=contract["development_end"],
                         quality_passed=True)
    # A crash after the claim intentionally prevents automatic re-evaluation.
    frame = build_periods(panel, dates, spec.grid, build_strategy(contract), symbols)
    validate_periods(frame, contract, spec.grid)
    report = evaluate(frame, spec)
    periods_path = directory / "periods.csv"
    atomic_write_text(periods_path, frame.to_csv(index=False))
    report.update(signature=signature, contract_hash=canonical, periods_sha256=digest(periods_path),
                  rejection_counts=rejected, strategy=contract["strategy"], strategy_parameters=contract["strategy_parameters"],
                  holdout_evaluated=False, promotion_authority=False, runtime_authority=False, orders_sent=0,
                  models_fitted=0, evidence_type="historical_open_to_open_split_adjusted_markouts_not_executable_fills")
    report_path = directory / "report.json"
    write_json(report_path, report)
    finish_campaign_trial(state_path, evidence_signature=signature, decision=report["decision"], report_path=report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True,
                        help="campaign contract, relative to the repository root unless absolute")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="verify data and support; never claims")
    mode.add_argument("--fit", action="store_true", help="claim the single trial once and evaluate")
    args = parser.parse_args(argv)
    result = run(args.repository_root, fit=args.fit, contract_path=args.contract)
    LOGGER.info("REGISTERED_STRATEGY_TRIAL_RESULT", extra={"status": result.get("decision", result.get("status"))})
    return 0 if result.get("status") == "preflight_passed" or "decision" in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
