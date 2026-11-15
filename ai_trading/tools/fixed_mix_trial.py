"""One registered fixed-mix rebalancing development trial for an owner-chosen N-asset mix; never trades.

The mix, tolerance band and calendar rule come from the contract, so the same runner scores
whatever mix the owner confirms. The rebalance rule follows portfolio manager mode: a weight
strictly outside the absolute band, or a due calendar period with drift above the minimum,
rebalances every asset to target. Differences forced by daily bars are listed in the contract
under ``portfolio_manager_differences``.

Modes, each refusing a draft contract, an unconfirmed mix, an unlisted campaign or a missing
original ledger before any data is touched:

* ``--acquire`` writes the daily SIP bars once, with a hash manifest, refusing any session
  outside development, in the holdout or in 2026 before a file is written;
* ``--support-audit`` reads session dates only and writes the sampling-support audit;
* ``--preflight`` runs the audit, then verifies prices; it never claims the trial;
* ``--fit`` claims the single trial, simulates the mix and writes the report.
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
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools import volatility_target_trial as daily
from ai_trading.tools.experiment_ledger import claim_campaign_trial, finish_campaign_trial, register_campaign
from ai_trading.tools.registered_strategy_trial import bootstrap_indices, digest, evidence_signature, write_json
from ai_trading.tools.research_feasibility import daily_development_dates

LOGGER = get_logger(__name__)
RESET_CONFIG = Path("config/research_reset.json")
CODE_FILES = ("tools/fixed_mix_trial.py", "tools/volatility_target_trial.py", "tools/registered_strategy_trial.py",
              "tools/research_feasibility.py", "tools/experiment_ledger.py")
SUPPORT_OK = "support_possible_not_strategy_eligible"
STRESS_ROLE = "required_criteria_at_stress"
COST_ROLES = ("decision", STRESS_ROLE, "reported_only")
FREQUENCY_MONTHS = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
WEIGHT_TOLERANCE = 1e-12  # float guard so drift exactly on the band or the minimum does not trigger
SERIES = ("strategy", "never_rebalanced", "benchmark")


@dataclass(frozen=True)
class FixedMixSpec:
    """Contract values fixed before any price is read."""

    assets: tuple[tuple[str, float], ...]
    cash_buffer: float
    benchmark: str
    risk_free: str
    band: float
    frequency: str
    calendar_min_drift: float
    period_sessions: int
    annualization: int
    tiers: tuple[tuple[str, float, str], ...]
    min_sessions: int
    min_years: int
    min_per_year: int
    max_missing: int
    drawdown_ratio_benchmark: float
    drawdown_ratio_never: float
    sharpe_margin: float
    max_turnover: float
    block: int
    replicates: int
    seed: int

    @property
    def symbols(self) -> list[str]:
        """Every symbol whose bars are needed, in a stable order without repeats."""
        return list(dict.fromkeys([*(s for s, _ in self.assets), self.benchmark, self.risk_free]))

    @property
    def periods_per_year(self) -> float:
        return self.annualization / self.period_sessions

    def tier(self, role: str) -> tuple[str, float, str]:
        return next(t for t in self.tiers if t[2] == role)


def _weight(value: Any, name: str, *, low_open: bool) -> float:
    """Accept a number or the decimal string the portfolio plan uses."""
    try:
        number = float(value) if isinstance(value, str) else value
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    return daily._number(number, name, low=0, high=1, low_open=low_open)


def contract_spec(contract: dict[str, Any]) -> FixedMixSpec:
    """Validate everything that must be fixed before data are read; fail closed on ambiguity."""
    if not re.fullmatch(r"[a-z0-9_]+", str(contract.get("campaign_id", ""))):
        raise ValueError("contract campaign_id must name its artifact directory")
    if contract.get("promotion_authority") is not False or contract.get("runtime_authority") is not False:
        raise ValueError("a development trial never carries promotion or runtime authority")
    daily_development_dates(contract)  # Holdout and holdout-year refusal precede any file read.
    if contract.get("feed") != "sip" or contract.get("adjustment") != "all":
        raise ValueError("contract must use SIP daily bars with adjustment=all")
    mix, rule, symbols = contract.get("mix"), contract.get("rebalance"), contract.get("symbols")
    evaluation = contract.get("evaluation")
    if not all(isinstance(part, dict) for part in (mix, rule, symbols, evaluation)):
        raise ValueError("contract must declare mix, rebalance, symbols and evaluation")
    mix, rule, symbols, evaluation = (cast(dict[str, Any], part) for part in (mix, rule, symbols, evaluation))
    if mix.get("owner_confirmed") is not True:
        raise ValueError("mix is not owner-confirmed; set the owner's mix before registration")
    raw_assets = mix.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("mix must list at least one asset")
    assets = []
    for asset in raw_assets:
        symbol = asset.get("symbol") if isinstance(asset, dict) else None
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z.]{0,9}", symbol):
            raise ValueError("mix asset needs an upper-case symbol")
        assets.append((symbol, _weight(asset.get("target_weight"), f"target_weight:{symbol}", low_open=True)))
    if len({s for s, _ in assets}) != len(assets):
        raise ValueError("mix symbols must be unique")
    cash_buffer = _weight(mix.get("cash_buffer_weight"), "cash_buffer_weight", low_open=False)
    if abs(sum(w for _, w in assets) + cash_buffer - 1) > 1e-9:
        raise ValueError("target weights and cash buffer must sum to 1")
    if not all(isinstance(symbols.get(k), str) and symbols[k] for k in ("benchmark", "risk_free")):
        raise ValueError("symbols must name benchmark and risk_free")
    frequency = rule.get("calendar_frequency")
    if frequency not in FREQUENCY_MONTHS:
        raise ValueError("calendar_frequency must be monthly, quarterly, semiannual or annual")
    tiers = []
    for tier in contract.get("cost_tiers") or []:
        if tier.get("role") not in COST_ROLES or not isinstance(tier.get("name"), str):
            raise ValueError("cost tier needs a name and a supported role")
        tiers.append((tier["name"], daily._number(tier.get("round_trip_bps"), "round_trip_bps", low=0, high=1000, low_open=False),
                      tier["role"]))
    roles = Counter(t[2] for t in tiers)
    if len({t[0] for t in tiers}) != len(tiers) or roles["decision"] != 1 or roles[STRESS_ROLE] != 1:
        raise ValueError("cost tiers need unique names, one decision tier and one required stress tier")
    if next(t[1] for t in tiers if t[2] == "decision") != float(contract.get("primary_round_trip_cost_bps", math.nan)):
        raise ValueError("decision tier must equal primary_round_trip_cost_bps")
    acceptance, support, bootstrap = contract["acceptance"], contract["support_thresholds"], contract["bootstrap"]
    replicates = daily._int(bootstrap.get("replicates"), "bootstrap replicates")
    if replicates % 100:
        raise ValueError("bootstrap replicates must be a positive multiple of 100")
    seed = bootstrap.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    return FixedMixSpec(
        assets=tuple(assets), cash_buffer=cash_buffer, benchmark=symbols["benchmark"], risk_free=symbols["risk_free"],
        band=_weight(rule.get("band_abs"), "band_abs", low_open=True), frequency=frequency,
        calendar_min_drift=_weight(rule.get("calendar_min_drift"), "calendar_min_drift", low_open=False),
        period_sessions=daily._int(evaluation.get("period_sessions"), "period_sessions"),
        annualization=daily._int(evaluation.get("annualization"), "annualization"), tiers=tuple(tiers),
        min_sessions=daily._int(support.get("minimum_sessions"), "minimum_sessions"),
        min_years=daily._int(support.get("minimum_supported_years"), "minimum_supported_years"),
        min_per_year=daily._int(support.get("minimum_sessions_per_year"), "minimum_sessions_per_year"),
        max_missing=daily._int(support.get("maximum_missing_sessions"), "maximum_missing_sessions", minimum=0),
        drawdown_ratio_benchmark=daily._number(acceptance.get("drawdown_ratio_vs_benchmark_at_most"),
                                               "drawdown ratio vs benchmark", low=0, high=1),
        drawdown_ratio_never=daily._number(acceptance.get("drawdown_ratio_vs_never_rebalanced_at_most"),
                                           "drawdown ratio vs never-rebalanced", low=0, high=2),
        sharpe_margin=daily._number(acceptance.get("sharpe_difference_vs_benchmark_at_least"),
                                    "sharpe difference", low=-5, high=5, low_open=False),
        max_turnover=daily._number(acceptance.get("max_annual_turnover"), "max_annual_turnover", low=0, high=1000),
        block=daily._int(bootstrap.get("block_periods"), "bootstrap block_periods"), replicates=replicates, seed=seed)


# --- support ---------------------------------------------------------------------------------------

def support_summary(spec: FixedMixSpec, complete: np.ndarray, dates: list[date]) -> dict[str, Any]:
    """Counts of sessions on which every symbol has a bar; no price is involved."""
    per_year = Counter(str(day.year) for day, ok in zip(dates, complete) if ok)
    sessions, missing = int(complete.sum()), int((~complete).sum())
    supported = sum(n >= spec.min_per_year for n in per_year.values())
    reasons = [reason for reason, failed in (("sessions_below_minimum", sessions < spec.min_sessions),
                                             ("supported_years_below_minimum", supported < spec.min_years),
                                             ("missing_sessions_above_maximum", missing > spec.max_missing)) if failed]
    return dict(complete_sessions=sessions, missing_sessions=missing, sessions_by_year=dict(sorted(per_year.items())),
                supported_years=supported, possible_periods=sessions // spec.period_sessions,
                minimum_required=dict(sessions=spec.min_sessions, supported_years=spec.min_years,
                                      sessions_per_year=spec.min_per_year, maximum_missing_sessions=spec.max_missing),
                status="blocked_sampling_support" if reasons else SUPPORT_OK, blocking_reasons=reasons)


# --- simulation ------------------------------------------------------------------------------------

def period_index(day: date, months: int) -> int:
    return (day.year * 12 + day.month - 1) // months


def decide_trigger(spec: FixedMixSpec, drifts: np.ndarray, last_rebalance: date, trade_day: date) -> str:
    """Portfolio manager semantics: band first (strictly outside), then a due calendar period above the minimum drift."""
    largest = float(np.abs(drifts).max())
    if largest > spec.band + WEIGHT_TOLERANCE:
        return "band"
    months = FREQUENCY_MONTHS[spec.frequency]
    if period_index(trade_day, months) > period_index(last_rebalance, months) and largest > spec.calendar_min_drift + WEIGHT_TOLERANCE:
        return "calendar"
    return "none"


def simulate(panel: dict[str, pd.DataFrame], dates: list[date], spec: FixedMixSpec) -> pd.DataFrame:
    """Daily gross path of the rebalanced mix and its controls.

    Drift is measured at the close of session t; a triggered rebalance trades every asset to
    target at the open of session t+1. Costs are applied later from ``traded``, which never
    changes a trigger because a proportional cost leaves weights unchanged.
    """
    names = [s for s, _ in spec.assets]
    targets = np.array([w for _, w in spec.assets])
    opens = np.column_stack([panel[s].open.to_numpy(dtype=float) for s in names])
    closes = np.column_stack([panel[s].close.to_numpy(dtype=float) for s in names])
    extra = {key: (panel[symbol].open.to_numpy(dtype=float), panel[symbol].close.to_numpy(dtype=float))
             for key, symbol in (("benchmark", spec.benchmark), ("risk_free", spec.risk_free))}
    units = targets / opens[0]  # initial entry of one unit of equity at the first open
    never_units, cash, last, pending = units.copy(), spec.cash_buffer, dates[0], "initial"
    previous = dict(strategy=1.0, never_rebalanced=1.0)
    rows = []
    for t, day in enumerate(dates):
        traded = float(targets.sum()) if t == 0 else 0.0
        if t > 0 and pending != "none":
            values = units * opens[t]
            equity_open = float(values.sum() + cash)
            traded = float(np.abs(targets * equity_open - values).sum() / equity_open)
            units, cash, last = targets * equity_open / opens[t], spec.cash_buffer * equity_open, day
        equity = float(units @ closes[t] + cash)
        never = float(never_units @ closes[t] + spec.cash_buffer)
        drifts = np.append(units * closes[t] / equity - targets, cash / equity - spec.cash_buffer)
        row: dict[str, Any] = dict(
            session=day.isoformat(), year=str(day.year), trigger=pending, traded=traded,
            strategy_gross=equity / previous["strategy"] - 1, never_rebalanced_gross=never / previous["never_rebalanced"] - 1,
            max_abs_drift=float(np.abs(drifts).max()))
        for key, (open_, close) in extra.items():
            row[f"{key}_return"] = float(close[t] / (open_[0] if t == 0 else close[t - 1]) - 1)
        rows.append(row)
        previous = dict(strategy=equity, never_rebalanced=never)
        pending = decide_trigger(spec, drifts, last, dates[t + 1]) if t + 1 < len(dates) else "none"
    return pd.DataFrame(rows)


def validate_sessions(frame: pd.DataFrame, contract: dict[str, Any]) -> None:
    if frame.empty or not frame.session.is_monotonic_increasing or frame.session.duplicated().any():
        raise ValueError("empty, unordered or duplicate sessions")
    if not (frame.session.ge(contract["development_start"]) & frame.session.le(contract["development_end"])).all():
        raise ValueError("session outside development")
    numeric = frame[["traded", "strategy_gross", "never_rebalanced_gross", "benchmark_return", "risk_free_return", "max_abs_drift"]]
    if not np.isfinite(numeric.to_numpy()).all() or (frame.traded < 0).any() or (frame.traded > 2 + 1e-9).any():
        raise ValueError("nonfinite values or impossible traded weight")
    if frame.trigger.iloc[0] != "initial" or not frame.trigger.iloc[1:].isin(["none", "band", "calendar"]).all():
        raise ValueError("unknown trigger")


def tier_returns(frame: pd.DataFrame, cost_bps: float, initial_weight: float) -> dict[str, np.ndarray]:
    """Net daily returns. Each trade pays half a round trip on the weight traded; controls pay one entry."""
    cost = cost_bps / 10_000 / 2
    entry = np.zeros(len(frame))
    entry[0] = 1.0
    return dict(strategy=(1 + frame.strategy_gross.to_numpy()) * (1 - cost * frame.traded.to_numpy()) - 1,
                never_rebalanced=(1 + frame.never_rebalanced_gross.to_numpy()) * (1 - cost * initial_weight * entry) - 1,
                benchmark=(1 + frame.benchmark_return.to_numpy()) * (1 - cost * entry) - 1,
                risk_free=frame.risk_free_return.to_numpy())


def to_periods(returns: np.ndarray, sessions: int) -> np.ndarray:
    """Compound daily returns into non-overlapping periods; a trailing partial period is dropped."""
    count = len(returns) // sessions
    return np.asarray(np.prod(1 + returns[:count * sessions].reshape(count, sessions), axis=1) - 1)


def _summary(returns: np.ndarray, cash: np.ndarray, spec: FixedMixSpec) -> dict[str, Any]:
    years = len(returns) / spec.annualization
    periods, cash_periods = to_periods(returns, spec.period_sessions), to_periods(cash, spec.period_sessions)
    return dict(sharpe=daily._finite(daily.sharpe(periods - cash_periods, spec.periods_per_year)),
                max_drawdown=daily.max_drawdown(returns), cagr=float(np.prod(1 + returns)) ** (1 / years) - 1,
                annual_volatility=float(periods.std(ddof=1) * math.sqrt(spec.periods_per_year)))


def _tier_criteria(tier: dict[str, Any], spec: FixedMixSpec) -> dict[str, bool]:
    difference, to_benchmark, to_never = (tier[k] for k in ("sharpe_difference_vs_benchmark", "drawdown_ratio_vs_benchmark",
                                                             "drawdown_ratio_vs_never_rebalanced"))
    return {"drawdown_ratio_vs_benchmark": to_benchmark is not None and to_benchmark <= spec.drawdown_ratio_benchmark,
            "drawdown_ratio_vs_never_rebalanced": to_never is not None and to_never <= spec.drawdown_ratio_never,
            "sharpe_difference_vs_benchmark": difference is not None and difference >= spec.sharpe_margin}


def evaluate(frame: pd.DataFrame, spec: FixedMixSpec) -> dict[str, Any]:
    years = len(frame) / spec.annualization
    per_year = frame.year.value_counts()
    supported_years = int((per_year >= spec.min_per_year).sum())
    support = len(frame) >= spec.min_sessions and supported_years >= spec.min_years
    turnover = float(frame.traded.iloc[1:].sum() / 2 / years)  # one-way; the initial entry is excluded
    initial_weight = float(sum(w for _, w in spec.assets))
    tiers: dict[str, dict[str, Any]] = {}
    for name, bps, role in spec.tiers:
        nets = tier_returns(frame, bps, initial_weight)
        summaries = {k: _summary(nets[k], nets["risk_free"], spec) for k in SERIES}
        ratios = {f"drawdown_ratio_vs_{k}": (summaries["strategy"]["max_drawdown"] / summaries[k]["max_drawdown"]
                                              if summaries[k]["max_drawdown"] > 0 else None) for k in SERIES[1:]}
        tiers[name] = dict(round_trip_bps=bps, role=role, **summaries, **ratios,
                           sharpe_difference_vs_benchmark=daily._difference(summaries["strategy"]["sharpe"],
                                                                            summaries["benchmark"]["sharpe"]))
    primary_name, primary_bps, _ = spec.tier("decision")
    stress_name = spec.tier(STRESS_ROLE)[0]
    nets = tier_returns(frame, primary_bps, initial_weight)
    by_year = {str(year): dict(sessions=len(part), **{f"{k}_return": float(np.prod(1 + part[k]) - 1) for k in SERIES})
               for year, part in frame[["year"]].assign(**{k: nets[k] for k in SERIES}).groupby("year")}
    triggers = Counter(frame.trigger.iloc[1:])
    base = dict(sessions=len(frame), supported_years=supported_years, years=years, annual_turnover=turnover,
                rebalances=dict(band=triggers["band"], calendar=triggers["calendar"]),
                largest_close_drift=float(frame.max_abs_drift.max()), cost_tiers=tiers, by_year=by_year,
                decision_tier=primary_name, reported_only_tiers=[n for n, _, r in spec.tiers if r == "reported_only"])
    periods = {k: to_periods(nets[k], spec.period_sessions) for k in (*SERIES, "risk_free")}
    indices = bootstrap_indices(len(periods["strategy"]), block=spec.block, replicates=spec.replicates, seed=spec.seed)
    if indices is None:
        return dict(base, decision="inconclusive_budget_consumed", sharpe_difference_bootstrap_95=None,
                    uncertainty_status="unavailable_fewer_than_two_blocks", criteria={"adequate_support": False})
    differences = (daily.sharpe(periods["strategy"][indices] - periods["risk_free"][indices], spec.periods_per_year)
                   - daily.sharpe(periods["benchmark"][indices] - periods["risk_free"][indices], spec.periods_per_year))
    finite = differences[np.isfinite(differences)]
    interval = np.quantile(finite, [.025, .975]).tolist() if finite.size else [math.nan, math.nan]
    criteria = {"adequate_support": bool(support), **_tier_criteria(tiers[primary_name], spec),
                "all_criteria_at_stress": all(_tier_criteria(tiers[stress_name], spec).values()),
                "annual_turnover": turnover <= spec.max_turnover}
    decision = ("eligible_for_untouched_test_planning" if all(criteria.values())
                else "retired" if support else "inconclusive_budget_consumed")
    return dict(base, decision=decision, criteria=criteria,
                sharpe_difference_bootstrap_95_reported_only=[daily._finite(v) for v in interval])


# --- orchestration ---------------------------------------------------------------------------------

def run(root: Path, *, mode: str, contract_path: Path,
        client_factory: Callable[[], Any] = daily._default_data_client) -> dict[str, Any]:
    root = root.resolve()
    contract = json.loads((root / contract_path).resolve().read_text())
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if contract.get("status") != "registered":
        raise ValueError("contract is not registered; a draft cannot be acquired, audited, preflighted or fitted")
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
    reader = cast(Any, spec)  # the daily-bar helpers need only ``symbols``
    if mode == "acquire":
        return daily.acquire(root, contract, reader, canonical, client_factory=client_factory)
    manifest = json.loads((root / contract["acquisition_manifest"]).read_text())
    if manifest.get("contract_hash") != canonical or manifest.get("quality_passed") is not True:
        raise ValueError("acquisition manifest does not belong to this registered contract")
    sessions = daily.load_sessions(root, manifest, reader)
    daily.check_development_sessions(sessions, contract)
    dates = daily_development_dates(contract)
    index = [d.isoformat() for d in dates]
    present = [set(sessions[s]) for s in spec.symbols]
    complete = np.array([all(day in days for days in present) for day in index])
    blocked = dict(trial_claimed=False, holdout_evaluated=False, promotion_authority=False)
    audit = dict(support_summary(spec, complete, dates), audit="daily_grid_timestamp_only", development_sessions=len(dates),
                 sessions_by_symbol={s: len(v) for s, v in sessions.items()}, contract_hash=canonical,
                 prices_read_for_analysis=False, strategy_outcomes_computed=False, **blocked)
    write_json(directory / "support_audit.json", audit)
    if mode == "support-audit":
        return audit
    if audit["status"] != SUPPORT_OK:
        write_json(directory / "preflight.json", dict(blocked, status="blocked_sampling_support", reasons=audit["blocking_reasons"]))
        return dict(blocked, status="blocked_sampling_support", reasons=audit["blocking_reasons"])
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = {p: digest(code_root / p) for p in CODE_FILES}
    provenance = {k: v for k, v in manifest.items() if k != "fetched_at"}
    signature = evidence_signature(canonical, provenance, code_hashes)
    panel, rejected = daily.load_panel(root, manifest, reader, sessions, dates)
    priced = np.logical_and.reduce([panel[s].complete.to_numpy(dtype=bool) for s in spec.symbols])
    priced_support = support_summary(spec, priced, dates)
    preflight = dict(blocked, status="preflight_passed" if priced_support["status"] == SUPPORT_OK else "blocked_price_validated_support",
                     signature=signature, contract_hash=canonical, complete_sessions=priced_support["complete_sessions"],
                     missing_sessions=priced_support["missing_sessions"], sessions_by_year=priced_support["sessions_by_year"],
                     supported_years=priced_support["supported_years"], rejection_counts=rejected, provenance=provenance,
                     code=code_hashes)
    write_json(directory / "preflight.json", preflight)
    if mode != "fit" or preflight["status"] != "preflight_passed":
        return preflight
    claim_campaign_trial(state_path, hypothesis_id=contract["hypotheses"][0], evidence_signature=signature,
                         evaluation_start=contract["development_start"], evaluation_end=contract["development_end"],
                         quality_passed=True)
    # A crash after the claim intentionally prevents automatic re-evaluation.
    frame = simulate({s: panel[s][priced] for s in spec.symbols}, [d for d, ok in zip(dates, priced) if ok], spec)
    validate_sessions(frame, contract)
    report = evaluate(frame, spec)
    sessions_path = directory / "sessions.csv"
    atomic_write_text(sessions_path, frame.to_csv(index=False))
    report.update(signature=signature, contract_hash=canonical, sessions_sha256=digest(sessions_path), rejection_counts=rejected,
                  holdout_evaluated=False, promotion_authority=False, runtime_authority=False, orders_sent=0, models_fitted=0,
                  evidence_type="historical_daily_adjusted_markouts_not_executable_fills")
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
    mode.add_argument("--support-audit", action="store_true", help="session dates only; never reads prices or claims")
    mode.add_argument("--preflight", action="store_true", help="support audit and price checks; never claims")
    mode.add_argument("--fit", action="store_true", help="claim the single trial once and evaluate")
    args = parser.parse_args(argv)
    selected = next(name for name in ("acquire", "support-audit", "fit", "preflight") if getattr(args, name.replace("-", "_")))
    result = run(args.repository_root, mode=selected, contract_path=args.contract)
    status = result.get("decision", result.get("status"))
    LOGGER.info("FIXED_MIX_TRIAL_RESULT", extra={"mode": selected, "status": status})
    return 0 if status in ("acquisition_written", SUPPORT_OK, "preflight_passed") or "decision" in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
