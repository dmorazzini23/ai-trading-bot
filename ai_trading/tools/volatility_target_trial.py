"""One registered volatility-target core development trial; acquires daily bars, never trades.

Modes, each refusing a draft contract, an unlisted campaign or a missing original
ledger before any data is touched:

* ``--acquire`` writes the daily SIP bars once, with a hash manifest, refusing any
  session outside development, in the holdout or in 2026 before a file is written;
* ``--preflight`` runs the timestamp-only support audit, then verifies prices and
  counts usable periods; it never claims the trial;
* ``--fit`` claims the single trial, computes weights and writes the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.experiment_ledger import claim_campaign_trial, finish_campaign_trial, register_campaign
from ai_trading.tools.registered_strategy_trial import bootstrap_indices, digest, evidence_signature, write_json
from ai_trading.tools.research_feasibility import HOLDOUT_YEAR_START, NEW_YORK, daily_development_dates

LOGGER = get_logger(__name__)
RESET_CONFIG = Path("config/research_reset.json")
CODE_FILES = ("tools/volatility_target_trial.py", "tools/registered_strategy_trial.py",
              "tools/research_feasibility.py", "tools/experiment_ledger.py")
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
SUPPORT_OK = "support_possible_not_strategy_eligible"
STRESS_ROLE = "required_positive_sharpe_difference"
COST_ROLES = ("decision", STRESS_ROLE, "reported_only")


@dataclass(frozen=True)
class VolTargetSpec:
    """Contract values fixed before any price is read."""

    risk: str
    cash: str
    lookback: int
    annualization: int
    target: float
    max_weight: float
    every: int
    first_decision: int
    band: float
    tiers: tuple[tuple[str, float, str], ...]
    min_periods: int
    min_years: int
    min_per_year: int
    drawdown_ratio: float
    max_turnover: float
    block: int
    replicates: int
    seed: int

    @property
    def symbols(self) -> list[str]:
        return [self.risk, self.cash]

    @property
    def periods_per_year(self) -> float:
        return self.annualization / self.every

    def tier(self, role: str) -> tuple[str, float, str]:
        return next(t for t in self.tiers if t[2] == role)


def _int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, name: str, *, low: float, high: float, low_open: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not ((low < value) if low_open else (low <= value)) or value > high:
        raise ValueError(f"{name} is out of range")
    return float(value)


def contract_spec(contract: dict[str, Any]) -> VolTargetSpec:
    """Validate everything that must be fixed before data are read; fail closed on ambiguity."""
    if not re.fullmatch(r"[a-z0-9_]+", str(contract.get("campaign_id", ""))):
        raise ValueError("contract campaign_id must name its artifact directory")
    if contract.get("promotion_authority") is not False or contract.get("runtime_authority") is not False:
        raise ValueError("a development trial never carries promotion or runtime authority")
    daily_development_dates(contract)  # Holdout and holdout-year refusal precede any file read.
    if contract.get("feed") != "sip" or contract.get("adjustment") != "all":
        raise ValueError("contract must use SIP daily bars with adjustment=all")
    symbols, vol, grid = contract.get("symbols"), contract.get("volatility"), contract.get("rebalance_grid")
    if not isinstance(symbols, dict) or not all(isinstance(symbols.get(k), str) for k in ("risk_asset", "cash_asset")):
        raise ValueError("symbols must name risk_asset and cash_asset")
    if not isinstance(vol, dict) or not isinstance(grid, dict):
        raise ValueError("contract must declare volatility and rebalance_grid")
    lookback = _int(vol.get("lookback_sessions"), "lookback_sessions", minimum=2)
    first = _int(grid.get("first_decision_index"), "first_decision_index")
    if first < lookback:
        raise ValueError("first decision precedes the full volatility lookback")
    tiers = []
    for tier in contract.get("cost_tiers") or []:
        if tier.get("role") not in COST_ROLES or not isinstance(tier.get("name"), str):
            raise ValueError("cost tier needs a name and a supported role")
        tiers.append((tier["name"], _number(tier.get("round_trip_bps"), "round_trip_bps", low=0, high=1000, low_open=False), tier["role"]))
    roles = Counter(t[2] for t in tiers)
    if len({t[0] for t in tiers}) != len(tiers) or roles["decision"] != 1 or roles[STRESS_ROLE] != 1:
        raise ValueError("cost tiers need unique names, one decision tier and one required stress tier")
    if next(t[1] for t in tiers if t[2] == "decision") != float(contract.get("primary_round_trip_cost_bps", math.nan)):
        raise ValueError("decision tier must equal primary_round_trip_cost_bps")
    acceptance, support, bootstrap = contract["acceptance"], contract["support_thresholds"], contract["bootstrap"]
    if not all(acceptance.get(k) is True for k in ("positive_sharpe_difference_lower_bound_at_primary", "positive_sharpe_difference_at_stress_high")):
        raise ValueError("acceptance criteria cannot be disabled")
    replicates = _int(bootstrap.get("replicates"), "bootstrap replicates")
    if replicates % 100:
        raise ValueError("bootstrap replicates must be a positive multiple of 100")
    seed = bootstrap.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    return VolTargetSpec(
        risk=symbols["risk_asset"], cash=symbols["cash_asset"], lookback=lookback,
        annualization=_int(vol.get("annualization"), "annualization"),
        target=_number(contract.get("target_volatility_annual"), "target_volatility_annual", low=0, high=1),
        max_weight=_number(contract.get("max_risk_weight"), "max_risk_weight", low=0, high=1),
        every=_int(grid.get("every_sessions"), "every_sessions"), first_decision=first,
        band=_number(grid.get("no_trade_band"), "no_trade_band", low=0, high=1, low_open=False),
        tiers=tuple(tiers),
        min_periods=_int(support.get("minimum_periods"), "minimum_periods"),
        min_years=_int(support.get("minimum_supported_years"), "minimum_supported_years"),
        min_per_year=_int(support.get("minimum_periods_per_year"), "minimum_periods_per_year"),
        drawdown_ratio=_number(acceptance.get("drawdown_ratio_vs_static_mix_at_most"), "drawdown ratio", low=0, high=1),
        max_turnover=_number(acceptance.get("max_annual_turnover"), "max_annual_turnover", low=0, high=1000),
        block=_int(bootstrap.get("block_periods"), "bootstrap block_periods"), replicates=replicates, seed=seed)


def check_development_sessions(sessions: dict[str, list[str]], contract: dict[str, Any]) -> None:
    """Refuse any session outside development, in the holdout or in 2026 before a price is read."""
    upper = min(contract["holdout_start"], HOLDOUT_YEAR_START)
    for symbol, days in sessions.items():
        if any(day < contract["development_start"] or day > contract["development_end"] or day >= upper for day in days):
            raise ValueError(f"source sessions outside development window:{symbol}")


def schedule(spec: VolTargetSpec, n_sessions: int) -> list[tuple[int, int, int]]:
    """(decision, entry, exit) indices: entry the next session, exit at the next entry; never overlapping."""
    return [(d, d + 1, d + 1 + spec.every) for d in range(spec.first_decision, n_sessions - 1 - spec.every, spec.every)]


def usable_periods(spec: VolTargetSpec, complete: np.ndarray, n_sessions: int) -> list[tuple[int, int, int]]:
    """Periods whose full volatility window, entry and exit sessions are complete for every symbol."""
    return [(d, e, x) for d, e, x in schedule(spec, n_sessions)
            if complete[d - spec.lookback:d + 1].all() and complete[e] and complete[x]]


def support_summary(spec: VolTargetSpec, periods: list[tuple[int, int, int]], dates: list[date]) -> dict[str, Any]:
    per_year = Counter(str(dates[e].year) for _, e, _ in periods)
    supported = sum(n >= spec.min_per_year for n in per_year.values())
    reasons = [reason for reason, failed in (("periods_below_minimum", len(periods) < spec.min_periods),
                                             ("supported_years_below_minimum", supported < spec.min_years)) if failed]
    return dict(possible_periods=len(periods), periods_by_year=dict(sorted(per_year.items())), supported_years=supported,
                minimum_required=dict(periods=spec.min_periods, supported_years=spec.min_years, periods_per_year=spec.min_per_year),
                status="blocked_sampling_support" if reasons else SUPPORT_OK, blocking_reasons=reasons)


# --- acquisition ---------------------------------------------------------------------------------

def _default_data_client() -> Any:
    from ai_trading.config.managed_secrets import hydrate_managed_secrets

    hydrate_managed_secrets(required_keys=("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
    from ai_trading.alpaca_api import _get_rest

    return _get_rest(bars=True)


def fetch_daily_rows(client: Any, symbol: str, start: str, end: str) -> list[dict[str, Any]]:
    """Daily SIP bars (adjustment=all) keyed by New York session date."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    request = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
        start=datetime.combine(date.fromisoformat(start), time(0), NEW_YORK),
        end=datetime.combine(date.fromisoformat(end), time(23, 59), NEW_YORK),
        feed=DataFeed.SIP, adjustment=Adjustment.ALL)
    rows = []
    for bar in getattr(client.get_stock_bars(request), "data", {}).get(symbol, []):
        rows.append(dict(session=bar.timestamp.astimezone(NEW_YORK).date().isoformat(),
                         **{column: float(getattr(bar, column)) for column in PRICE_COLUMNS}))
    return rows


def acquire(root: Path, contract: dict[str, Any], spec: VolTargetSpec, canonical: str, *,
            client_factory: Callable[[], Any] = _default_data_client) -> dict[str, Any]:
    """Write each symbol's daily bars once and a hash manifest; refuse an existing acquisition."""
    manifest_path = root / contract["acquisition_manifest"]
    directory = manifest_path.parent
    if manifest_path.exists():
        raise ValueError("acquisition already exists and is immutable")
    client = client_factory()
    fetched = {s: fetch_daily_rows(client, s, contract["development_start"], contract["development_end"]) for s in spec.symbols}
    check_development_sessions({s: [r["session"] for r in rows] for s, rows in fetched.items()}, contract)
    sources = {}
    for symbol, rows in fetched.items():
        if not rows:
            raise ValueError(f"no bars returned:{symbol}")
        frame = pd.DataFrame(rows, columns=["session", *PRICE_COLUMNS]).sort_values("session")
        if frame.session.duplicated().any():
            raise ValueError(f"duplicate sessions:{symbol}")
        path = directory / f"{symbol}.csv"
        atomic_write_text(path, frame.to_csv(index=False))
        sources[symbol] = dict(path=str(path.relative_to(root)), sha256=digest(path), rows=len(frame),
                               first_session=str(frame.session.iloc[0]), last_session=str(frame.session.iloc[-1]))
    manifest = dict(campaign_id=contract["campaign_id"], contract_hash=canonical, feed="sip", adjustment="all",
                    request=dict(start=contract["development_start"], end=contract["development_end"]),
                    fetched_at=datetime.now(NEW_YORK).isoformat(), sources=sources,
                    prices_read_for_analysis=False, quality_passed=True)
    write_json(manifest_path, manifest)
    return dict(status="acquisition_written", manifest=str(manifest_path), sources=sources, trial_claimed=False)


# --- reading ---------------------------------------------------------------------------------------

def _verified_read(root: Path, source: dict[str, Any], columns: list[str]) -> pd.DataFrame:
    path = root / source["path"]
    if digest(path) != source["sha256"]:
        raise ValueError(f"source changed before read:{source['path']}")
    frame = pd.read_csv(path, usecols=columns, dtype={"session": str})
    if digest(path) != source["sha256"]:
        raise ValueError(f"source changed during read:{source['path']}")
    return frame


def load_sessions(root: Path, manifest: dict[str, Any], spec: VolTargetSpec) -> dict[str, list[str]]:
    """Session dates only; no price column is read."""
    return {s: _verified_read(root, manifest["sources"][s], ["session"]).session.tolist() for s in spec.symbols}


def valid_daily(frame: pd.DataFrame) -> pd.Series:
    values = frame[PRICE_COLUMNS].astype(float)
    finite = np.isfinite(values.to_numpy()).all(axis=1)
    positive = (values[["open", "high", "low", "close"]] > 0).all(axis=1)
    geometry = (values.high >= values[["open", "close"]].max(axis=1)) & (values.low <= values[["open", "close"]].min(axis=1))
    return pd.Series(finite, index=frame.index) & positive & geometry & (values.volume >= 0)


def load_panel(root: Path, manifest: dict[str, Any], spec: VolTargetSpec, sessions: dict[str, list[str]],
               dates: list[date]) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    """Prices from hash-verified sources whose sessions equal the audited ones, aligned to development dates."""
    index = [d.isoformat() for d in dates]
    panel, rejected = {}, {}
    for symbol in spec.symbols:
        frame = _verified_read(root, manifest["sources"][symbol], ["session", *PRICE_COLUMNS])
        if frame.session.tolist() != sessions[symbol]:
            raise ValueError(f"sessions changed after audit:{symbol}")
        valid = valid_daily(frame)
        counts = Counter(dict(invalid_prices=int((~valid).sum()), off_calendar=int((~frame.session.isin(index)).sum())))
        aligned = frame[valid].set_index("session").reindex(index)
        aligned["complete"] = aligned.close.notna()
        counts["missing_sessions"] = int((~aligned.complete).sum())
        panel[symbol], rejected[symbol] = aligned, {k: v for k, v in counts.items() if v}
    return panel, rejected


# --- evaluation ------------------------------------------------------------------------------------

def build_periods(panel: dict[str, pd.DataFrame], dates: list[date], spec: VolTargetSpec,
                  periods: list[tuple[int, int, int]]) -> pd.DataFrame:
    """Point-in-time volatility, target weight with the no-trade band, and open-to-open returns."""
    risk_close = panel[spec.risk].close.to_numpy(dtype=float)
    opens = {s: panel[s].open.to_numpy(dtype=float) for s in spec.symbols}
    rows, weight = [], None
    for d, e, x in periods:
        window = risk_close[d - spec.lookback:d + 1]  # closes through the decision session only
        sigma = float(np.std(np.diff(np.log(window)), ddof=1) * math.sqrt(spec.annualization))
        target = min(spec.max_weight, spec.target / sigma) if sigma > 0 else spec.max_weight
        if weight is None:
            traded, weight = 1.0, target  # initial entry of the whole portfolio from cash
        elif abs(target - weight) >= spec.band:
            traded, weight = abs(target - weight) * 2, target  # both legs move by |dw|
        else:
            traded = 0.0
        rows.append(dict(decision_session=dates[d].isoformat(), entry_session=dates[e].isoformat(),
                         exit_session=dates[x].isoformat(), decision_index=d, entry_index=e, exit_index=x,
                         year=str(dates[e].year), realized_vol=sigma, target_weight=target, risk_weight=weight,
                         traded_weight=traded, risk_return=opens[spec.risk][x] / opens[spec.risk][e] - 1,
                         cash_return=opens[spec.cash][x] / opens[spec.cash][e] - 1))
    return pd.DataFrame(rows)


def validate_periods(frame: pd.DataFrame, contract: dict[str, Any], spec: VolTargetSpec) -> None:
    if frame.empty or frame.decision_index.duplicated().any():
        raise ValueError("empty or duplicate decision periods")
    if not ((frame.entry_index == frame.decision_index + 1).all() and (frame.exit_index == frame.entry_index + spec.every).all()):
        raise ValueError("entry or exit off the grid")
    if (frame.entry_index.to_numpy()[1:] < frame.exit_index.to_numpy()[:-1]).any():
        raise ValueError("overlapping holds")
    if not (frame.decision_index >= spec.lookback).all():
        raise ValueError("decision before the full lookback")
    if not (frame.decision_session.ge(contract["development_start"]) & frame.exit_session.le(contract["development_end"])).all():
        raise ValueError("period outside development")
    numeric = frame[["realized_vol", "risk_weight", "risk_return", "cash_return"]].to_numpy()
    if not np.isfinite(numeric).all() or not frame.risk_weight.between(0, spec.max_weight).all():
        raise ValueError("nonfinite values or weight outside [0, max_risk_weight]")


def sharpe(excess: np.ndarray, periods_per_year: float) -> np.ndarray:
    """Annualized Sharpe along the last axis; NaN where the deviation is zero."""
    deviation = excess.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(deviation > 0, excess.mean(axis=-1) / deviation * math.sqrt(periods_per_year), np.nan)


def max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1 + returns)
    return float(np.max(1 - equity / np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]))


def tier_returns(frame: pd.DataFrame, cost_bps: float) -> dict[str, np.ndarray]:
    """Net per-period returns: strategy, buy-and-hold and the same-average-exposure static mix."""
    cost = cost_bps / 10_000 / 2  # half a round trip per unit of weight traded in a leg
    risk, cash, weight = frame.risk_return.to_numpy(), frame.cash_return.to_numpy(), frame.risk_weight.to_numpy()
    entry = np.zeros(len(frame))
    entry[0] = cost  # one initial entry of the whole portfolio for each control
    average = float(weight.mean())
    return dict(strategy=weight * risk + (1 - weight) * cash - cost * frame.traded_weight.to_numpy(),
                buy_and_hold=risk - entry, static_mix=average * risk + (1 - average) * cash - entry, cash=cash)


def _finite(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _difference(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def _summary(returns: np.ndarray, cash: np.ndarray, years: float, periods_per_year: float) -> dict[str, Any]:
    growth = float(np.prod(1 + returns))
    return dict(sharpe=_finite(sharpe(returns - cash, periods_per_year)), max_drawdown=max_drawdown(returns),
                cagr=growth ** (1 / years) - 1, annual_volatility=float(returns.std(ddof=1) * math.sqrt(periods_per_year)))


def evaluate(frame: pd.DataFrame, spec: VolTargetSpec) -> dict[str, Any]:
    years = len(frame) / spec.periods_per_year
    per_year = frame.year.value_counts()
    supported_years = int((per_year >= spec.min_per_year).sum())
    support = len(frame) >= spec.min_periods and supported_years >= spec.min_years
    turnover = float(frame.traded_weight.iloc[1:].sum() / 2 / years)
    average_weight = float(frame.risk_weight.mean())
    tiers: dict[str, dict[str, Any]] = {}
    for name, bps, role in spec.tiers:
        nets = tier_returns(frame, bps)
        summaries = {k: _summary(nets[k], nets["cash"], years, spec.periods_per_year) for k in ("strategy", "buy_and_hold", "static_mix")}
        tiers[name] = dict(round_trip_bps=bps, role=role, **summaries,
                           sharpe_difference_vs_buy_and_hold=_difference(summaries["strategy"]["sharpe"], summaries["buy_and_hold"]["sharpe"]),
                           drawdown_ratio_vs_static_mix=(summaries["strategy"]["max_drawdown"] / summaries["static_mix"]["max_drawdown"]
                                                         if summaries["static_mix"]["max_drawdown"] > 0 else None))
    primary_name, primary_bps, _ = spec.tier("decision")
    stress_name = spec.tier(STRESS_ROLE)[0]
    nets = tier_returns(frame, primary_bps)
    by_year = {}
    for year, part in frame.assign(**{f"net_{k}": v for k, v in nets.items()}).groupby("year"):
        by_year[str(year)] = dict(periods=len(part), strategy_return=float(np.prod(1 + part["net_strategy"]) - 1),
                                  buy_and_hold_return=float(np.prod(1 + part["net_buy_and_hold"]) - 1),
                                  average_risk_weight=float(part.risk_weight.mean()))
    base = dict(decision_periods=len(frame), supported_years=supported_years, years=years, average_risk_weight=average_weight,
                annual_turnover=turnover, cost_tiers=tiers, by_year=by_year, decision_tier=primary_name,
                reported_only_tiers=[n for n, _, r in spec.tiers if r == "reported_only"])
    indices = bootstrap_indices(len(frame), block=spec.block, replicates=spec.replicates, seed=spec.seed)
    if indices is None:
        return dict(base, decision="inconclusive_budget_consumed", sharpe_difference_bootstrap_95=None,
                    uncertainty_status="unavailable_fewer_than_two_blocks", criteria={"adequate_support": False})
    differences = (sharpe(nets["strategy"][indices] - nets["cash"][indices], spec.periods_per_year)
                   - sharpe(nets["buy_and_hold"][indices] - nets["cash"][indices], spec.periods_per_year))
    finite = differences[np.isfinite(differences)]
    interval = np.quantile(finite, [.025, .975]).tolist() if finite.size else [math.nan, math.nan]
    ratio = tiers[primary_name]["drawdown_ratio_vs_static_mix"]
    stress_difference = tiers[stress_name]["sharpe_difference_vs_buy_and_hold"]
    criteria = {"adequate_support": bool(support),
                "positive_sharpe_difference_lower_bound_at_primary": bool(interval[0] > 0),
                "drawdown_ratio_vs_static_mix": ratio is not None and ratio <= spec.drawdown_ratio,
                "positive_sharpe_difference_at_stress_high": stress_difference is not None and stress_difference > 0,
                "annual_turnover": turnover <= spec.max_turnover}
    decision = ("eligible_for_untouched_test_planning" if all(criteria.values())
                else "retired" if support else "inconclusive_budget_consumed")
    return dict(base, decision=decision, sharpe_difference_bootstrap_95=[_finite(v) for v in interval], criteria=criteria)


# --- orchestration ---------------------------------------------------------------------------------

def run(root: Path, *, mode: str, contract_path: Path,
        client_factory: Callable[[], Any] = _default_data_client) -> dict[str, Any]:
    root = root.resolve()
    contract = json.loads((root / contract_path).resolve().read_text())
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if contract.get("status") != "registered":
        raise ValueError("contract is not registered; a draft cannot be acquired, preflighted or fitted")
    spec = contract_spec(contract)
    reset = json.loads((root / RESET_CONFIG).read_text())
    if contract["campaign_id"] not in reset.get("registered_trials", []):
        raise ValueError("research reset does not list this campaign in registered_trials")
    directory = root / "artifacts" / contract["campaign_id"]
    state_path = directory / "campaign_state.json"
    if not state_path.is_file():
        raise ValueError("original campaign registration is required")
    if (root / contract["acquisition_manifest"]).resolve().parent.parent != directory.resolve():
        raise ValueError("acquisition manifest must live under the campaign artifact directory")
    state = register_campaign(state_path, contract)
    if state["trials"]:
        return {"status": "budget_already_claimed", "trials": state["trials"]}
    if mode == "acquire":
        return acquire(root, contract, spec, canonical, client_factory=client_factory)
    manifest = json.loads((root / contract["acquisition_manifest"]).read_text())
    if manifest.get("contract_hash") != canonical or manifest.get("quality_passed") is not True:
        raise ValueError("acquisition manifest does not belong to this registered contract")
    sessions = load_sessions(root, manifest, spec)
    check_development_sessions(sessions, contract)
    dates = daily_development_dates(contract)
    index = [d.isoformat() for d in dates]
    present = [set(sessions[s]) for s in spec.symbols]
    complete = np.array([all(day in days for days in present) for day in index])
    blocked = dict(trial_claimed=False, holdout_evaluated=False, promotion_authority=False)
    audit = dict(support_summary(spec, usable_periods(spec, complete, len(dates)), dates), audit="daily_grid_timestamp_only",
                 development_sessions=len(dates), sessions_by_symbol={s: len(v) for s, v in sessions.items()},
                 contract_hash=canonical, prices_read_for_analysis=False, strategy_outcomes_computed=False, **blocked)
    write_json(directory / "support_audit.json", audit)
    if audit["status"] != SUPPORT_OK:
        write_json(directory / "preflight.json", dict(blocked, status="blocked_sampling_support", reasons=audit["blocking_reasons"]))
        return dict(blocked, status="blocked_sampling_support", reasons=audit["blocking_reasons"])
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = {p: digest(code_root / p) for p in CODE_FILES}
    provenance = {k: v for k, v in manifest.items() if k != "fetched_at"}
    signature = evidence_signature(canonical, provenance, code_hashes)
    panel, rejected = load_panel(root, manifest, spec, sessions, dates)
    priced = np.logical_and.reduce([panel[s].complete.to_numpy(dtype=bool) for s in spec.symbols])
    periods = usable_periods(spec, priced, len(dates))
    priced_support = support_summary(spec, periods, dates)
    preflight = dict(blocked, status="preflight_passed" if priced_support["status"] == SUPPORT_OK else "blocked_price_validated_support",
                     signature=signature, contract_hash=canonical, possible_periods=len(periods),
                     periods_by_year=priced_support["periods_by_year"], supported_years=priced_support["supported_years"],
                     rejection_counts=rejected, provenance=provenance, code=code_hashes)
    write_json(directory / "preflight.json", preflight)
    if mode != "fit" or preflight["status"] != "preflight_passed":
        return preflight
    claim_campaign_trial(state_path, hypothesis_id=contract["hypotheses"][0], evidence_signature=signature,
                         evaluation_start=contract["development_start"], evaluation_end=contract["development_end"],
                         quality_passed=True)
    # A crash after the claim intentionally prevents automatic re-evaluation.
    frame = build_periods(panel, dates, spec, periods)
    validate_periods(frame, contract, spec)
    report = evaluate(frame, spec)
    periods_path = directory / "periods.csv"
    atomic_write_text(periods_path, frame.to_csv(index=False))
    report.update(signature=signature, contract_hash=canonical, periods_sha256=digest(periods_path), rejection_counts=rejected,
                  holdout_evaluated=False, promotion_authority=False, runtime_authority=False, orders_sent=0, models_fitted=0,
                  evidence_type="historical_open_to_open_adjusted_markouts_not_executable_fills")
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
    mode.add_argument("--acquire", action="store_true", help="write the daily bars once; never claims")
    mode.add_argument("--preflight", action="store_true", help="support audit and price checks; never claims")
    mode.add_argument("--fit", action="store_true", help="claim the single trial once and evaluate")
    args = parser.parse_args(argv)
    selected = "acquire" if args.acquire else "fit" if args.fit else "preflight"
    result = run(args.repository_root, mode=selected, contract_path=args.contract)
    status = result.get("decision", result.get("status"))
    LOGGER.info("VOLATILITY_TARGET_TRIAL_RESULT", extra={"mode": selected, "status": status})
    return 0 if status in ("acquisition_written", "preflight_passed") or "decision" in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
