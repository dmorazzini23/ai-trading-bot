"""One registered multi-day swing trial: sector-relative short-term reversal in liquid US stocks; never trades.

Every five sessions the trial ranks a point-in-time universe of the most liquid stocks by their
trailing return relative to their sector, buys the recent losers (and, in the long-short version,
shorts the recent winners), keeps names inside a hold band, and scales exposure by trailing SPY
volatility. Every parameter comes from the contract. The scoring core is pure: it takes price
tables, a frozen universe and a sector map, and returns per-period accounting rows.

Modes. All but ``--probe`` refuse a draft contract, an unlisted campaign or a missing ledger
before any data is touched:

* ``--probe`` asks the data provider which delisted symbols it still serves; dates and counts
  only, no price is kept; allowed on a draft because its answer belongs in the contract;
* ``--acquire`` freezes the asset list, the monthly raw dollar volume, the universe, the sector
  map and the daily bars, with a hash manifest, refusing any row outside development or in 2026;
* ``--support-audit`` reads session dates and the frozen universe only;
* ``--preflight`` runs the audit, then verifies prices; it never claims the trial;
* ``--fit`` claims the single trial, simulates both versions and writes the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools import volatility_target_trial as daily
from ai_trading.tools.experiment_ledger import claim_campaign_trial, finish_campaign_trial, register_campaign
from ai_trading.tools.registered_strategy_trial import bootstrap_indices, digest, evidence_signature, write_json
from ai_trading.tools.research_feasibility import HOLDOUT_YEAR_START, NEW_YORK, daily_development_dates

LOGGER = get_logger(__name__)
RESET_CONFIG = Path("config/research_reset.json")
CODE_FILES = ("tools/swing_reversal_trial.py", "tools/volatility_target_trial.py", "tools/registered_strategy_trial.py",
              "tools/research_feasibility.py", "tools/experiment_ledger.py")
SUPPORT_OK = "support_possible_not_strategy_eligible"
STRESS_ROLE = "required_positive_at_stress"
COST_ROLES = ("decision", STRESS_ROLE, "reported_only")
UNMAPPED = "UNMAPPED"
VERSIONS = ("long_only", "long_short")
PROBE_STATUSES = {"delisted_symbols_served": "none_detected_by_probe", "delisted_symbols_missing_or_partial": "upward_biased"}
PRICE_COLUMNS = daily.PRICE_COLUMNS
CHUNK = 100
SECTOR_LOOKUP_ERRORS = (OSError, ValueError, KeyError, AttributeError, TypeError)


@dataclass(frozen=True)
class SwingSpec:
    """Contract values fixed before any price is read."""

    market: str
    universe_size: int
    volume_months: int
    min_price: float
    exchanges: tuple[str, ...]
    excluded_name: str
    lookback: int
    min_group: int
    slots: int
    entry_quantile: float
    hold_quantile: float
    every: int
    first_decision: int
    vol_lookback: int
    annualization: int
    reference_vol: float
    exposure_floor: float
    exposure_cap: float
    exposure_band: float
    htb_lookback: int
    htb_fraction: float
    adverse_return: float
    tiers: tuple[tuple[str, float, float, str], ...]  # name, per-side bps, annual borrow bps, role
    min_periods: int
    min_years: int
    min_per_year: int
    min_eligible: int
    min_positive_years: int
    block: int
    replicates: int
    seed: int
    lower_quantile: float

    @property
    def periods_per_year(self) -> float:
        return self.annualization / self.every

    def tier(self, role: str) -> tuple[str, float, float, str]:
        return next(t for t in self.tiers if t[3] == role)


def _section(contract: dict[str, Any], name: str) -> dict[str, Any]:
    value = contract.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"contract must declare {name}")
    return value


def contract_spec(contract: dict[str, Any]) -> SwingSpec:
    """Validate everything that must be fixed before data are read; fail closed on ambiguity."""
    if not re.fullmatch(r"[a-z0-9_]+", str(contract.get("campaign_id", ""))):
        raise ValueError("contract campaign_id must name its artifact directory")
    if contract.get("promotion_authority") is not False or contract.get("runtime_authority") is not False:
        raise ValueError("a development trial never carries promotion or runtime authority")
    daily_development_dates(contract)  # Holdout and holdout-year refusal precede any file read.
    if contract.get("feed") != "sip" or contract.get("adjustment") != "all":
        raise ValueError("contract must use SIP daily bars with adjustment=all")
    universe, signal, portfolio = (_section(contract, k) for k in ("universe", "signal", "portfolio"))
    grid, fear, shorts = (_section(contract, k) for k in ("rebalance_grid", "fear_conditioning", "shorts"))
    survivorship, acceptance = _section(contract, "survivorship"), _section(contract, "acceptance")
    support, bootstrap = _section(contract, "support_thresholds"), _section(contract, "bootstrap")
    if survivorship.get("probe_status") not in PROBE_STATUSES:
        raise ValueError("survivorship.probe_status must record the probe result before registration")
    market = _section(contract, "symbols").get("market")
    if not isinstance(market, str) or not market:
        raise ValueError("symbols must name the market proxy")
    if list(_section(contract, "versions")) != list(VERSIONS):
        raise ValueError("versions must be long_only (primary) then long_short (secondary)")
    exchanges = universe.get("exchanges")
    if not isinstance(exchanges, list) or not exchanges or not all(isinstance(x, str) for x in exchanges):
        raise ValueError("universe must list exchanges")
    pattern = universe.get("excluded_name_pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("universe must declare excluded_name_pattern")
    re.compile(pattern)
    every = daily._int(grid.get("every_sessions"), "every_sessions")
    lookback = daily._int(signal.get("lookback_sessions"), "lookback_sessions")
    vol_lookback = daily._int(fear.get("lookback_sessions"), "fear lookback_sessions", minimum=2)
    htb = _section(shorts, "hard_to_borrow_proxy")
    htb_lookback = daily._int(htb.get("volatility_lookback_sessions"), "volatility_lookback_sessions", minimum=2)
    first = daily._int(grid.get("first_decision_index"), "first_decision_index")
    if first < max(lookback, vol_lookback, htb_lookback):
        raise ValueError("first decision precedes a full lookback")
    if daily._int(portfolio.get("hold_sessions"), "hold_sessions") != every:
        raise ValueError("hold_sessions must equal the rebalance interval; holds never overlap")
    entry = daily._number(portfolio.get("entry_quantile"), "entry_quantile", low=0, high=0.5)
    hold = daily._number(portfolio.get("hold_quantile"), "hold_quantile", low=0, high=1)
    if hold < entry or portfolio.get("weighting") != "equal":
        raise ValueError("hold band must contain the entry slice and weighting must be equal")
    tiers = []
    for tier in contract.get("cost_tiers") or []:
        if tier.get("role") not in COST_ROLES or not isinstance(tier.get("name"), str):
            raise ValueError("cost tier needs a name and a supported role")
        if tier.get("per_side_bps") is None:
            raise ValueError(f"cost tier placeholder must be filled or removed before registration:{tier['name']}")
        tiers.append((tier["name"], daily._number(tier.get("per_side_bps"), "per_side_bps", low=0, high=1000, low_open=False),
                      daily._number(tier.get("borrow_annual_bps"), "borrow_annual_bps", low=0, high=10_000, low_open=False),
                      tier["role"]))
    roles = Counter(t[3] for t in tiers)
    if len({t[0] for t in tiers}) != len(tiers) or roles["decision"] != 1 or roles[STRESS_ROLE] != 1:
        raise ValueError("cost tiers need unique names, one decision tier and one required stress tier")
    if next(t[1] for t in tiers if t[3] == "decision") != float(contract.get("primary_per_side_cost_bps", math.nan)):
        raise ValueError("decision tier must equal primary_per_side_cost_bps")
    if not all(acceptance.get(k) is True for k in ("positive_lower_bound_vs_cash_at_primary", "positive_at_stress",
                                                   "positive_under_survivorship_stress",
                                                   "long_only_positive_lower_bound_vs_exposure_matched_market_at_primary")):
        raise ValueError("acceptance criteria cannot be disabled")
    replicates = daily._int(bootstrap.get("replicates"), "bootstrap replicates")
    if replicates % 100:
        raise ValueError("bootstrap replicates must be a positive multiple of 100")
    seed = bootstrap.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    return SwingSpec(
        market=market, universe_size=daily._int(universe.get("size"), "universe size"),
        volume_months=daily._int(universe.get("dollar_volume_months"), "dollar_volume_months"),
        min_price=daily._number(universe.get("minimum_raw_price"), "minimum_raw_price", low=0, high=1e6, low_open=False),
        exchanges=tuple(exchanges), excluded_name=pattern, lookback=lookback,
        min_group=daily._int(signal.get("minimum_group_size"), "minimum_group_size", minimum=2),
        slots=daily._int(portfolio.get("slice_names"), "slice_names"), entry_quantile=entry, hold_quantile=hold,
        every=every, first_decision=first, vol_lookback=vol_lookback,
        annualization=daily._int(fear.get("annualization"), "annualization"),
        reference_vol=daily._number(fear.get("reference_volatility_annual"), "reference_volatility_annual", low=0, high=5),
        exposure_floor=daily._number(fear.get("floor"), "exposure floor", low=0, high=1, low_open=False),
        exposure_cap=daily._number(fear.get("cap"), "exposure cap", low=0, high=1),
        exposure_band=daily._number(fear.get("no_trade_band"), "exposure no_trade_band", low=0, high=1, low_open=False),
        htb_lookback=htb_lookback,
        htb_fraction=daily._number(htb.get("exclude_top_volatility_fraction"), "exclude_top_volatility_fraction", low=0, high=1, low_open=False),
        adverse_return=daily._number(survivorship.get("adverse_return_when_data_ends"), "adverse_return_when_data_ends", low=0, high=1),
        tiers=tuple(tiers), min_periods=daily._int(support.get("minimum_periods"), "minimum_periods"),
        min_years=daily._int(support.get("minimum_supported_years"), "minimum_supported_years"),
        min_per_year=daily._int(support.get("minimum_periods_per_year"), "minimum_periods_per_year"),
        min_eligible=daily._int(support.get("minimum_eligible_names"), "minimum_eligible_names"),
        min_positive_years=daily._int(acceptance.get("minimum_positive_years"), "minimum_positive_years", minimum=0),
        block=daily._int(bootstrap.get("block_periods"), "bootstrap block_periods"), replicates=replicates, seed=seed,
        lower_quantile=daily._number(bootstrap.get("lower_quantile"), "lower_quantile", low=0, high=0.5))


# --- pure core: universe ---------------------------------------------------------------------------

def previous_months(month: str, count: int) -> list[str]:
    """The ``count`` calendar months before ``month`` (YYYY-MM), oldest first."""
    index = int(month[:4]) * 12 + int(month[5:7]) - 1
    return [f"{(index - k) // 12:04d}-{(index - k) % 12 + 1:02d}" for k in range(count, 0, -1)]


def build_universe(monthly: pd.DataFrame, months: Iterable[str], spec: SwingSpec) -> dict[str, list[str]]:
    """Members for each month from completed earlier months only.

    ``monthly`` has one row per symbol and month with the raw (unadjusted) close and dollar
    volume. A symbol qualifies for month M if it has a row in each of the previous
    ``volume_months`` months and its latest raw close is at least the minimum price; the top
    ``size`` by summed dollar volume are members. Ties break by symbol.
    """
    volume = monthly.pivot(index="month", columns="symbol", values="dollar_volume")
    close = monthly.pivot(index="month", columns="symbol", values="close")
    members: dict[str, list[str]] = {}
    for month in months:
        window = previous_months(month, spec.volume_months)
        if not all(m in volume.index for m in window):
            members[month] = []
            continue
        totals = volume.loc[window].sum(axis=0, skipna=False)
        eligible = totals[totals.notna() & (totals > 0) & (close.loc[window[-1]] >= spec.min_price)]
        ranked = sorted(eligible.items(), key=lambda item: (-float(item[1]), str(item[0])))
        members[month] = sorted(str(symbol) for symbol, _ in ranked[:spec.universe_size])
    return members


# --- pure core: signal and selection ---------------------------------------------------------------

def schedule(spec: SwingSpec, n_sessions: int) -> list[tuple[int, int, int]]:
    """(decision, entry, exit) indices: entry the next session, exit at the next entry; never overlapping."""
    return [(d, d + 1, d + 1 + spec.every) for d in range(spec.first_decision, n_sessions - 1 - spec.every, spec.every)]


def relative_ranks(closes: pd.DataFrame, decision: int, members: list[str], sectors: dict[str, str], spec: SwingSpec) -> pd.Series:
    """Rank in [0, 1] of each member's trailing return relative to its sector; 0 is the largest loser.

    Uses closes at the decision session and ``lookback`` sessions earlier only. A sector with
    fewer than ``min_group`` valid members, and any unmapped symbol, is measured against the
    whole universe instead.
    """
    names = sorted(s for s in members if s in closes.columns)
    raw = (closes.iloc[decision][names] / closes.iloc[decision - spec.lookback][names] - 1).astype(float)
    raw = raw[np.isfinite(raw)]
    if raw.empty:
        return pd.Series(dtype=float)
    groups = pd.Series({s: sectors.get(s) or UNMAPPED for s in raw.index})
    in_sector = (groups != UNMAPPED) & (groups.map(groups.value_counts()) >= spec.min_group)
    relative = raw - np.where(in_sector, raw.groupby(groups).transform("mean"), raw.mean())
    order = relative.rank(method="first")  # index is sorted, so ties break by symbol
    return (order - 1) / max(len(order) - 1, 1)


def high_volatility_names(closes: pd.DataFrame, decision: int, names: list[str], spec: SwingSpec) -> set[str]:
    """Names in the top volatility fraction of the universe: the pre-declared hard-to-borrow proxy."""
    window = closes.iloc[decision - spec.htb_lookback:decision + 1][names]
    volatility = np.log(window).diff().iloc[1:].std(ddof=1, skipna=False).dropna()
    count = math.floor(len(volatility) * spec.htb_fraction)
    return set(volatility.sort_values(ascending=False, kind="stable").index[:count]) if count else set()


def select(ranks: pd.Series, held: Iterable[str], spec: SwingSpec, *, side: str, blocked: frozenset[str] = frozenset()) -> list[str]:
    """Names for one side: held names still inside the hold band, then the most extreme new names.

    Long: keep while rank <= hold_quantile, enter while rank <= entry_quantile. Short mirrors it
    from the top. ``blocked`` names are never newly shorted. At most ``slots`` names.
    """
    score = ranks if side == "long" else 1 - ranks
    kept = sorted((s for s in held if s in score.index and score[s] <= spec.hold_quantile), key=lambda s: (score[s], s))[:spec.slots]
    fresh = [s for s in score[score <= spec.entry_quantile].sort_values(kind="stable").index if s not in kept and s not in blocked]
    return kept + fresh[:spec.slots - len(kept)]


def target_exposure(market_close: np.ndarray, decision: int, spec: SwingSpec) -> tuple[float, float]:
    """(realized volatility, exposure target): linear in trailing market volatility, floored and capped."""
    window = market_close[decision - spec.vol_lookback:decision + 1]
    sigma = float(np.std(np.diff(np.log(window)), ddof=1) * math.sqrt(spec.annualization))
    return sigma, min(spec.exposure_cap, max(spec.exposure_floor, sigma / spec.reference_vol))


# --- pure core: simulation -------------------------------------------------------------------------

def simulate(opens: pd.DataFrame, closes: pd.DataFrame, dates: list[date], universe: dict[str, list[str]],
             sectors: dict[str, str], not_shortable: frozenset[str], spec: SwingSpec, *, version: str,
             conditioned: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-period accounting rows and the positions behind them for one version.

    Decisions use closes through the decision session only; trades are at the next session's
    open. Weights are fractions of equity at the period's entry. ``traded`` is the absolute
    weight bought plus sold at the entry, measured from drifted weights, so hold-band keeps,
    exposure changes, replacements and forced exits are all charged later per tier.
    """
    if version not in VERSIONS:
        raise ValueError("unknown version")
    market_open, market_close = opens[spec.market].to_numpy(dtype=float), closes[spec.market].to_numpy(dtype=float)
    last_bar = {s: int(np.flatnonzero(np.isfinite(opens[s].to_numpy(dtype=float)))[-1]) if np.isfinite(opens[s]).any() else -1
                for s in opens.columns}
    exposure: float | None = None
    drifted: dict[str, dict[str, float]] = {"long": {}, "short": {}}
    rows, positions = [], []
    periods = schedule(spec, len(dates))
    for number, (d, e, x) in enumerate(periods):
        members = universe.get(dates[d].isoformat()[:7], [])
        ranks = relative_ranks(closes, d, members, sectors, spec)
        sigma, target = target_exposure(market_close, d, spec)
        if not conditioned:
            exposure = 1.0
        elif exposure is None or abs(target - exposure) >= spec.exposure_band:
            exposure = target
        slot = exposure / spec.slots
        blocked = frozenset(high_volatility_names(closes, d, list(ranks.index), spec)) | not_shortable
        sides = {"long": select(ranks, drifted["long"], spec, side="long")}
        if version == "long_short":
            sides["short"] = select(ranks, drifted["short"], spec, side="short", blocked=blocked)
        traded, gross, kept_count = 0.0, 0.0, 0
        weights, terminal = {"long": 0.0, "short": 0.0}, {"long": 0.0, "short": 0.0}
        changes: dict[str, dict[str, float]] = {"long": {}, "short": {}}
        forced: set[tuple[str, str]] = set()
        for side, sign in (("long", 1.0), ("short", -1.0)):
            names = [s for s in sides.get(side, []) if np.isfinite(opens[s].iat[e])]  # no open, no fill: the slot stays cash
            traded += sum(abs((slot if s in names else 0.0) - drifted[side].get(s, 0.0)) for s in set(names) | set(drifted[side]))
            for symbol in names:
                exit_price = float(opens[symbol].iat[x])
                if not np.isfinite(exit_price):  # halted or data ended: marked at the last close inside the hold
                    inside = closes[symbol].iloc[e:x].dropna()
                    exit_price = float(inside.iloc[-1]) if len(inside) else float(opens[symbol].iat[e])
                    forced.add((side, symbol))
                    if last_bar[symbol] < x:
                        terminal[side] += slot
                change = exit_price / float(opens[symbol].iat[e]) - 1
                changes[side][symbol] = change
                gross += sign * slot * change
                weights[side] += slot
                kept = symbol in drifted[side]
                kept_count += kept
                positions.append(dict(decision_index=d, decision_session=dates[d].isoformat(), symbol=symbol, side=side,
                                      weight=slot, kept=bool(kept), rank=float(ranks[symbol]), position_return=change,
                                      forced_exit=(side, symbol) in forced))
        forced_weight = 0.0
        for side in ("long", "short"):
            carried = {}
            for symbol, change in changes[side].items():
                value = slot * (1 + change) / (1 + gross)
                if (side, symbol) in forced:
                    forced_weight += value  # the exit trade is charged in this period
                else:
                    carried[symbol] = value
            drifted[side] = carried
        if number == len(periods) - 1:  # final liquidation is charged too
            forced_weight += sum(sum(v.values()) for v in drifted.values())
        rows.append(dict(
            decision_session=dates[d].isoformat(), entry_session=dates[e].isoformat(), exit_session=dates[x].isoformat(),
            decision_index=d, entry_index=e, exit_index=x, year=str(dates[e].year), eligible_names=len(ranks),
            realized_vol=sigma, target_exposure=target, exposure=exposure, names_long=len(changes["long"]),
            names_short=len(changes["short"]), kept_names=kept_count, long_weight=weights["long"], short_weight=weights["short"],
            traded=traded + forced_weight, gross_return=gross, market_return=float(market_open[x] / market_open[e] - 1),
            terminal_weight=terminal["long"] + terminal["short"], hold_sessions=x - e))
    return pd.DataFrame(rows), pd.DataFrame(positions)


def validate_periods(frame: pd.DataFrame, contract: dict[str, Any], spec: SwingSpec) -> None:
    if frame.empty or frame.decision_index.duplicated().any():
        raise ValueError("empty or duplicate decision periods")
    if not ((frame.entry_index == frame.decision_index + 1).all() and (frame.exit_index == frame.entry_index + spec.every).all()):
        raise ValueError("entry or exit off the grid; entry must be the session after the decision")
    if (frame.entry_index.to_numpy()[1:] < frame.exit_index.to_numpy()[:-1]).any():
        raise ValueError("overlapping holds")
    if not (frame.decision_session.ge(contract["development_start"]) & frame.exit_session.le(contract["development_end"])).all():
        raise ValueError("period outside development")
    numeric = frame[["exposure", "long_weight", "short_weight", "traded", "gross_return", "market_return", "terminal_weight"]].to_numpy()
    if not np.isfinite(numeric).all() or (frame.traded < 0).any():
        raise ValueError("nonfinite values or negative traded weight")
    if (frame.long_weight > frame.exposure + 1e-9).any() or (frame.short_weight > frame.exposure + 1e-9).any():
        raise ValueError("side weight above the exposure cap")
    if not frame.exposure.between(0, spec.exposure_cap + 1e-12).all() and not (frame.exposure == 1.0).all():
        raise ValueError("exposure outside its bounds")


def tier_returns(frame: pd.DataFrame, per_side_bps: float, borrow_annual_bps: float, spec: SwingSpec, *,
                 survivorship_stress: bool = False) -> dict[str, np.ndarray]:
    """Net per-period returns and the exposure-matched market return.

    Cost is ``per_side_bps`` on every unit of weight traded. Borrow is charged on the short weight
    for the sessions held. Under the survivorship stress every position whose data ended while
    held loses a further ``adverse_return`` of its weight.
    """
    net = (frame.gross_return.to_numpy() - per_side_bps / 10_000 * frame.traded.to_numpy()
           - borrow_annual_bps / 10_000 * frame.short_weight.to_numpy() * frame.hold_sessions.to_numpy() / spec.annualization)
    if survivorship_stress:
        net = net - spec.adverse_return * frame.terminal_weight.to_numpy()
    matched = (frame.long_weight.to_numpy() - frame.short_weight.to_numpy()) * frame.market_return.to_numpy()
    return dict(net=net, matched_market=matched)


def _bps(value: float) -> float | None:
    return daily._finite(value * 10_000)


def _lower_bound(values: np.ndarray, indices: np.ndarray, spec: SwingSpec) -> float | None:
    return _bps(float(np.quantile(values[indices].mean(axis=1), spec.lower_quantile)))


def _version_report(frame: pd.DataFrame, spec: SwingSpec, *, version: str, decides: bool) -> dict[str, Any]:
    years = len(frame) / spec.periods_per_year
    per_year = frame.year.value_counts()
    supported_years = int((per_year >= spec.min_per_year).sum())
    support = (len(frame) >= spec.min_periods and supported_years >= spec.min_years
               and float(frame.eligible_names.median()) >= spec.min_eligible)
    benchmark = "matched_market" if version == "long_only" else "cash"
    tiers: dict[str, dict[str, Any]] = {}
    for name, per_side, borrow, role in spec.tiers:
        nets = tier_returns(frame, per_side, borrow, spec)
        tiers[name] = dict(per_side_bps=per_side, borrow_annual_bps=borrow, role=role,
                           net_per_period_bps=_bps(float(nets["net"].mean())),
                           matched_market_per_period_bps=_bps(float(nets["matched_market"].mean())),
                           paired_vs_matched_market_bps=_bps(float((nets["net"] - nets["matched_market"]).mean())),
                           sharpe=daily._finite(daily.sharpe(nets["net"], spec.periods_per_year)),
                           max_drawdown=daily.max_drawdown(nets["net"]))
    base = dict(decision_periods=len(frame), supported_years=supported_years, cost_tiers=tiers,
                annual_one_way_turnover=float(frame.traded.sum() / 2 / years), average_exposure=float(frame.exposure.mean()),
                average_names_long=float(frame.names_long.mean()), average_names_short=float(frame.names_short.mean()),
                share_of_names_kept=float(frame.kept_names.sum() / max(frame.names_long.sum() + frame.names_short.sum(), 1)),
                median_eligible_names=float(frame.eligible_names.median()),
                periods_with_data_ended_positions=int((frame.terminal_weight > 0).sum()), benchmark=benchmark)
    if not decides:
        return base
    primary, stress = spec.tier("decision"), spec.tier(STRESS_ROLE)
    nets, stressed = tier_returns(frame, primary[1], primary[2], spec), tier_returns(frame, stress[1], stress[2], spec)
    survivor = tier_returns(frame, primary[1], primary[2], spec, survivorship_stress=True)
    reference = nets["matched_market"] if version == "long_only" else np.zeros(len(frame))
    paired = nets["net"] - reference
    by_year = {str(year): dict(periods=len(part), paired_bps=_bps(float(part.paired.mean())), net_bps=_bps(float(part.net.mean())))
               for year, part in frame[["year"]].assign(paired=paired, net=nets["net"]).groupby("year")}
    positive_years = sum(1 for v in by_year.values() if v["paired_bps"] is not None and v["paired_bps"] > 0)
    indices = bootstrap_indices(len(frame), block=spec.block, replicates=spec.replicates, seed=spec.seed)
    if indices is None:
        return dict(base, decision="inconclusive_budget_consumed", by_year=by_year,
                    uncertainty_status="unavailable_fewer_than_two_blocks", criteria={"adequate_support": False})
    bounds = dict(vs_cash_bps=_lower_bound(nets["net"], indices, spec), vs_benchmark_bps=_lower_bound(paired, indices, spec))
    criteria = {"adequate_support": bool(support),
                "positive_lower_bound_vs_cash_at_primary": bounds["vs_cash_bps"] is not None and bounds["vs_cash_bps"] > 0,
                "positive_at_stress": bool((stressed["net"] - (stressed["matched_market"] if version == "long_only" else 0)).mean() > 0),
                "consistent_years": positive_years >= spec.min_positive_years,
                "positive_under_survivorship_stress": bool((survivor["net"] - reference).mean() > 0)}
    if version == "long_only":
        criteria["positive_lower_bound_vs_exposure_matched_market_at_primary"] = (
            bounds["vs_benchmark_bps"] is not None and bounds["vs_benchmark_bps"] > 0)
    decision = ("eligible_for_untouched_test_planning" if all(criteria.values())
                else "retired" if support else "inconclusive_budget_consumed")
    return dict(base, decision=decision, criteria=criteria, by_year=by_year, positive_years=positive_years,
                lower_bounds=bounds, lower_quantile=spec.lower_quantile,
                survivorship_stress_paired_bps=_bps(float((survivor["net"] - reference).mean())))


def evaluate(frames: dict[str, pd.DataFrame], spec: SwingSpec) -> dict[str, Any]:
    """Both versions; the ledger decision is the long-only (primary) decision."""
    versions = {v: _version_report(frames[v], spec, version=v, decides=True) for v in VERSIONS}
    unconditioned = {v: _version_report(frames[f"{v}_unconditioned"], spec, version=v, decides=False) for v in VERSIONS}
    return dict(decision=versions["long_only"]["decision"], secondary_decision=versions["long_short"]["decision"],
                decision_tier=spec.tier("decision")[0], versions=versions, reported_only_unconditioned=unconditioned)


# --- acquisition and probe -------------------------------------------------------------------------

@dataclass(frozen=True)
class Clients:
    trading: Any
    data: Any


def _default_clients() -> Clients:
    from ai_trading.config.managed_secrets import hydrate_managed_secrets

    hydrate_managed_secrets(required_keys=("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
    from ai_trading.alpaca_api import _get_rest

    return Clients(trading=_get_rest(bars=False), data=_get_rest(bars=True))


def _default_sector_lookup(symbol: str) -> str | None:
    """Today's sector label from the provider the runtime risk code falls back to; not point-in-time."""
    import yfinance

    value = (yfinance.Ticker(symbol).info or {}).get("sector")
    return str(value) if value else None


def list_assets(trading: Any) -> list[dict[str, Any]]:
    """Active and inactive US equities as plain rows, sorted for a stable hash."""
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    rows = []
    for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        for asset in trading.get_all_assets(GetAssetsRequest(status=status, asset_class=AssetClass.US_EQUITY)):
            rows.append(dict(symbol=str(asset.symbol), name=str(asset.name or ""), status=status.value,
                             exchange=str(getattr(asset.exchange, "value", asset.exchange)),
                             shortable=bool(asset.shortable), easy_to_borrow=bool(asset.easy_to_borrow)))
    return sorted(rows, key=lambda row: (row["symbol"], row["status"]))


def candidate_symbols(assets: list[dict[str, Any]], spec: SwingSpec) -> tuple[list[str], dict[str, int]]:
    """Symbols on the listed exchanges whose name does not match the exclusion pattern; counts of each exclusion."""
    pattern = re.compile(spec.excluded_name, re.IGNORECASE)
    kept: set[str] = set()
    counts: Counter[str] = Counter()
    for row in assets:
        if row["exchange"] not in spec.exchanges:
            counts["excluded_exchange"] += 1
        elif pattern.search(row["name"]):
            counts["excluded_name"] += 1
        elif not re.fullmatch(r"[A-Z]{1,5}", row["symbol"]):
            counts["excluded_symbol_form"] += 1
        else:
            kept.add(row["symbol"])
    return sorted(kept), dict(counts)


def fetch_bars(data: Any, symbols: list[str], start: str, end: str, *, monthly: bool) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Bars keyed by symbol. Monthly bars are raw; daily bars use adjustment=all. Failures are recorded, never hidden."""
    from alpaca.common.exceptions import APIError
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    request_errors = (APIError, OSError, ValueError)

    def request(batch: list[str]) -> dict[str, list[Any]]:
        query = StockBarsRequest(
            symbol_or_symbols=batch, timeframe=TimeFrame.Month if monthly else TimeFrame.Day,
            start=datetime.combine(date.fromisoformat(start), time(0), NEW_YORK),
            end=datetime.combine(date.fromisoformat(end), time(23, 59), NEW_YORK),
            feed=DataFeed.SIP, adjustment=Adjustment.RAW if monthly else Adjustment.ALL)
        return dict(getattr(data.get_stock_bars(query), "data", {}))

    found: dict[str, list[dict[str, Any]]] = {}
    failed: dict[str, str] = {}
    for offset in range(0, len(symbols), CHUNK):
        batch = symbols[offset:offset + CHUNK]
        try:
            groups = [request(batch)]
        except request_errors as exc:  # one unknown symbol can fail a batch: retry singly and record
            LOGGER.warning("SWING_REVERSAL_BATCH_RETRY", extra={"first_symbol": batch[0], "error": type(exc).__name__})
            groups = []
            for symbol in batch:
                try:
                    groups.append(request([symbol]))
                except request_errors as single:  # recorded in the manifest, never hidden
                    failed[symbol] = type(single).__name__
        for group in groups:
            for symbol, bars in group.items():
                found[symbol] = [dict(session=bar.timestamp.astimezone(NEW_YORK).date().isoformat(),
                                      **{column: float(getattr(bar, column)) for column in PRICE_COLUMNS},
                                      vwap=float(getattr(bar, "vwap", None) or bar.close)) for bar in bars]
    return found, failed


def _check_window(sessions: Iterable[str], contract: dict[str, Any], label: str) -> None:
    upper = min(contract["holdout_start"], HOLDOUT_YEAR_START)
    if any(day < contract["development_start"][:8] + "01" or day > contract["development_end"] or day >= upper for day in sessions):
        raise ValueError(f"source sessions outside development window:{label}")


def _write_hashed(root: Path, path: Path, text: str) -> dict[str, Any]:
    atomic_write_text(path, text)
    return dict(path=str(path.relative_to(root)), sha256=digest(path))


def acquire(root: Path, contract: dict[str, Any], spec: SwingSpec, canonical: str, *, client_factory: Callable[[], Clients],
            sector_lookup: Callable[[str], str | None]) -> dict[str, Any]:
    """Freeze assets, monthly raw volume, the universe, sectors and daily bars; refuse an existing acquisition."""
    manifest_path = root / contract["acquisition_manifest"]
    directory = manifest_path.parent
    if manifest_path.exists():
        raise ValueError("acquisition already exists and is immutable")
    clients = client_factory()
    start, end = contract["development_start"], contract["development_end"]
    assets = list_assets(clients.trading)
    candidates, exclusions = candidate_symbols(assets, spec)
    monthly_bars, monthly_failed = fetch_bars(clients.data, candidates, start, end, monthly=True)
    monthly_rows = [dict(symbol=s, month=bar["session"][:7], close=bar["close"], dollar_volume=bar["vwap"] * bar["volume"])
                    for s in sorted(monthly_bars) for bar in monthly_bars[s]]
    _check_window((f"{row['month']}-01" for row in monthly_rows), contract, "monthly")
    monthly = pd.DataFrame(monthly_rows, columns=["symbol", "month", "close", "dollar_volume"])
    if monthly.empty or monthly.duplicated(["symbol", "month"]).any():
        raise ValueError("monthly bars empty or duplicated")
    months = sorted({d.isoformat()[:7] for d in daily_development_dates(contract)})
    universe = build_universe(monthly, months, spec)
    ever = sorted({s for names in universe.values() for s in names})
    daily_bars, daily_failed = fetch_bars(clients.data, sorted({*ever, spec.market}), start, end, monthly=False)
    daily.check_development_sessions({s: [bar["session"] for bar in bars] for s, bars in daily_bars.items()}, contract)
    if not daily_bars.get(spec.market):
        raise ValueError(f"no bars returned:{spec.market}")
    sources = {}
    for symbol in sorted(daily_bars):
        frame = pd.DataFrame(daily_bars[symbol], columns=["session", *PRICE_COLUMNS]).sort_values("session")
        if frame.empty:
            continue
        if frame.session.duplicated().any():
            raise ValueError(f"duplicate sessions:{symbol}")
        sources[symbol] = dict(_write_hashed(root, directory / "daily" / f"{symbol}.csv", frame.to_csv(index=False)), rows=len(frame),
                               first_session=str(frame.session.iloc[0]), last_session=str(frame.session.iloc[-1]))
    sector_map = {}
    for symbol in ever:
        try:
            sector_map[symbol] = sector_lookup(symbol) or UNMAPPED
        except SECTOR_LOOKUP_ERRORS as exc:  # an unmapped symbol is measured against the universe; counted below
            LOGGER.warning("SWING_REVERSAL_SECTOR_UNMAPPED", extra={"symbol": symbol, "error": type(exc).__name__})
            sector_map[symbol] = UNMAPPED
    as_json = lambda payload: json.dumps(payload, indent=1, sort_keys=True) + "\n"  # noqa: E731
    frozen = dict(
        assets=_write_hashed(root, directory / "assets.json", as_json(assets)),
        monthly=_write_hashed(root, directory / "monthly_raw.csv", monthly.sort_values(["symbol", "month"]).to_csv(index=False)),
        universe=_write_hashed(root, directory / "universe.json", as_json(universe)),
        sectors=_write_hashed(root, directory / "sectors.json", as_json(sector_map)))
    counts = dict(assets=len(assets), inactive_assets=sum(row["status"] == "inactive" for row in assets),
                  candidates=len(candidates), **exclusions, symbols_with_monthly_bars=len(monthly_bars),
                  ever_in_universe=len(ever), ever_in_universe_without_daily_bars=len([s for s in ever if s not in sources]),
                  ever_in_universe_inactive=len({row["symbol"] for row in assets if row["status"] == "inactive"} & set(ever)),
                  unmapped_sectors=sum(v == UNMAPPED for v in sector_map.values()))
    manifest = dict(campaign_id=contract["campaign_id"], contract_hash=canonical, feed="sip", adjustment="all",
                    request=dict(start=start, end=end), fetched_at=datetime.now(NEW_YORK).isoformat(), sources=sources,
                    frozen=frozen, counts=counts, failed=dict(monthly=monthly_failed, daily=daily_failed),
                    prices_read_for_analysis=False, quality_passed=True)
    write_json(manifest_path, manifest)
    return dict(status="acquisition_written", manifest=str(manifest_path), counts=counts, trial_claimed=False)


def probe(contract: dict[str, Any], *, client_factory: Callable[[], Clients]) -> dict[str, Any]:
    """Which delisted symbols the provider still lists and serves. Dates and counts only; no price is kept."""
    survivorship = _section(contract, "survivorship")
    symbols = [str(s) for s in survivorship.get("probe_symbols") or []]
    if not symbols:
        raise ValueError("survivorship.probe_symbols is empty")
    clients = client_factory()
    assets = list_assets(clients.trading)
    by_symbol: dict[str, list[str]] = {}
    for row in assets:
        by_symbol.setdefault(row["symbol"], []).append(row["status"])
    bars, failed = fetch_bars(clients.data, symbols, contract["development_start"], contract["development_end"], monthly=True)
    results = {}
    for symbol in symbols:
        months = sorted(bar["session"][:7] for bar in bars.get(symbol, []))
        results[symbol] = dict(asset_statuses=sorted(by_symbol.get(symbol, [])), monthly_bars=len(months),
                               first_month=months[0] if months else None, last_month=months[-1] if months else None,
                               request_error=failed.get(symbol))
    served = sum(1 for r in results.values() if r["monthly_bars"])
    return dict(status="probe_complete", assets=len(assets), inactive_assets=sum(r["status"] == "inactive" for r in assets),
                probe_symbols=len(symbols), probe_symbols_listed=sum(1 for r in results.values() if r["asset_statuses"]),
                probe_symbols_with_bars=served, results=results, prices_read_for_analysis=False, trial_claimed=False)


# --- reading ---------------------------------------------------------------------------------------

def load_frozen(root: Path, manifest: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, str], frozenset[str]]:
    """The frozen universe, sector map and the symbols flagged not shortable, each hash-verified."""
    def read(name: str) -> Any:
        source = manifest["frozen"][name]
        if digest(root / source["path"]) != source["sha256"]:
            raise ValueError(f"source changed before read:{source['path']}")
        return json.loads((root / source["path"]).read_text())

    not_shortable = frozenset(row["symbol"] for row in read("assets") if not (row["shortable"] and row["easy_to_borrow"]))
    return read("universe"), read("sectors"), not_shortable


def support_summary(spec: SwingSpec, sessions: dict[str, list[str]], universe: dict[str, list[str]], dates: list[date]) -> dict[str, Any]:
    """Possible periods and names with bars on the signal and entry sessions; session dates only."""
    index = {d.isoformat(): i for i, d in enumerate(dates)}
    present = {s: {index[day] for day in days if day in index} for s, days in sessions.items()}
    market = present.get(spec.market, set())
    periods, eligible, incomplete_market = [], [], 0
    for d, e, x in schedule(spec, len(dates)):
        if not all(i in market for i in range(d - spec.vol_lookback, x + 1)):
            incomplete_market += 1
            continue
        names = [s for s in universe.get(dates[d].isoformat()[:7], []) if {d - spec.lookback, d, e} <= present.get(s, set())]
        periods.append(e)
        eligible.append(len(names))
    per_year = Counter(str(dates[e].year) for e in periods)
    supported = sum(n >= spec.min_per_year for n in per_year.values())
    median = float(np.median(eligible)) if eligible else 0.0
    reasons = [reason for reason, failed in (("periods_below_minimum", len(periods) < spec.min_periods),
                                             ("supported_years_below_minimum", supported < spec.min_years),
                                             ("eligible_names_below_minimum", median < spec.min_eligible),
                                             ("market_proxy_incomplete", incomplete_market > 0)) if failed]
    return dict(possible_periods=len(periods), periods_by_year=dict(sorted(per_year.items())), supported_years=supported,
                median_eligible_names=median, minimum_eligible_names=min(eligible) if eligible else 0,
                periods_with_incomplete_market_proxy=incomplete_market,
                minimum_required=dict(periods=spec.min_periods, supported_years=spec.min_years, periods_per_year=spec.min_per_year,
                                      eligible_names=spec.min_eligible),
                status="blocked_sampling_support" if reasons else SUPPORT_OK, blocking_reasons=reasons)


def load_prices(root: Path, manifest: dict[str, Any], sessions: dict[str, list[str]], dates: list[date]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Open and close tables on the development calendar from hash-verified sources; invalid bars are dropped and counted."""
    index = [d.isoformat() for d in dates]
    opens: dict[str, np.ndarray] = {}
    closes: dict[str, np.ndarray] = {}
    rejected: Counter[str] = Counter()
    for symbol in sorted(manifest["sources"]):
        frame = daily._verified_read(root, manifest["sources"][symbol], ["session", *PRICE_COLUMNS])
        if frame.session.tolist() != sessions[symbol]:
            raise ValueError(f"sessions changed after audit:{symbol}")
        valid = daily.valid_daily(frame)
        rejected["invalid_prices"] += int((~valid).sum())
        rejected["off_calendar"] += int((~frame.session.isin(index)).sum())
        aligned = frame[valid].set_index("session").reindex(index)
        opens[symbol], closes[symbol] = aligned.open.to_numpy(dtype=float), aligned.close.to_numpy(dtype=float)
    return pd.DataFrame(opens), pd.DataFrame(closes), {k: v for k, v in rejected.items() if v}


# --- orchestration ---------------------------------------------------------------------------------

def run(root: Path, *, mode: str, contract_path: Path, client_factory: Callable[[], Clients] = _default_clients,
        sector_lookup: Callable[[str], str | None] = _default_sector_lookup) -> dict[str, Any]:
    root = root.resolve()
    contract = json.loads((root / contract_path).resolve().read_text())
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    directory = root / "artifacts" / str(contract.get("campaign_id"))
    if mode == "probe":
        daily_development_dates(contract)
        result = probe(contract, client_factory=client_factory)
        write_json(directory / "survivorship_probe.json", dict(result, contract_hash=canonical))
        return result
    if contract.get("status") != "registered":
        raise ValueError("contract is not registered; a draft cannot be acquired, audited, preflighted or fitted")
    spec = contract_spec(contract)
    reset = json.loads((root / RESET_CONFIG).read_text())
    if contract["campaign_id"] not in reset.get("registered_trials", []):
        raise ValueError("research reset does not list this campaign in registered_trials")
    state_path = directory / "campaign_state.json"
    if not state_path.is_file():
        raise ValueError("original campaign registration is required")
    if (root / contract["acquisition_manifest"]).resolve().parent.parent != directory.resolve():
        raise ValueError("acquisition manifest must live under the campaign artifact directory")
    state = register_campaign(state_path, contract)
    if state["trials"]:
        return {"status": "budget_already_claimed", "trials": state["trials"]}
    if mode == "acquire":
        return acquire(root, contract, spec, canonical, client_factory=client_factory, sector_lookup=sector_lookup)
    manifest = json.loads((root / contract["acquisition_manifest"]).read_text())
    if manifest.get("contract_hash") != canonical or manifest.get("quality_passed") is not True:
        raise ValueError("acquisition manifest does not belong to this registered contract")
    sessions = {s: daily._verified_read(root, source, ["session"]).session.tolist() for s, source in sorted(manifest["sources"].items())}
    daily.check_development_sessions(sessions, contract)
    dates = daily_development_dates(contract)
    universe, sectors, not_shortable = load_frozen(root, manifest)
    if any(month < contract["development_start"][:7] or month > contract["development_end"][:7] for month in universe):
        raise ValueError("frozen universe has a month outside development")
    blocked = dict(trial_claimed=False, holdout_evaluated=False, promotion_authority=False)
    audit = dict(support_summary(spec, sessions, universe, dates), audit="daily_grid_timestamp_only", development_sessions=len(dates),
                 symbols_with_bars=len(sessions), acquisition_counts=manifest.get("counts", {}), contract_hash=canonical,
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
    opens, closes, rejected = load_prices(root, manifest, sessions, dates)
    market_complete = bool(np.isfinite(opens[spec.market]).all() and np.isfinite(closes[spec.market]).all())
    preflight = dict(blocked, status="preflight_passed" if market_complete else "blocked_price_validated_support",
                     signature=signature, contract_hash=canonical, possible_periods=audit["possible_periods"],
                     rejection_counts=rejected, symbols=len(sessions), code=code_hashes,
                     provenance_sha256=hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest())
    write_json(directory / "preflight.json", preflight)
    if mode != "fit" or preflight["status"] != "preflight_passed":
        return preflight
    claim_campaign_trial(state_path, hypothesis_id=contract["hypotheses"][0], evidence_signature=signature,
                         evaluation_start=contract["development_start"], evaluation_end=contract["development_end"],
                         quality_passed=True)
    # A crash after the claim intentionally prevents automatic re-evaluation.
    frames, hashes = {}, {}
    for version in VERSIONS:
        for conditioned in (True, False):
            key = version if conditioned else f"{version}_unconditioned"
            frame, positions = simulate(opens, closes, dates, universe, sectors, not_shortable, spec, version=version,
                                        conditioned=conditioned)
            validate_periods(frame, contract, spec)
            frames[key] = frame
            for label, table in (("periods", frame), ("positions", positions)):
                path = directory / f"{label}_{key}.csv"
                atomic_write_text(path, table.to_csv(index=False))
                hashes[path.name] = digest(path)
    report = evaluate(frames, spec)
    survivorship = contract["survivorship"]
    report.update(signature=signature, contract_hash=canonical, output_sha256=hashes, rejection_counts=rejected,
                  acquisition_counts=manifest.get("counts", {}), survivorship_probe_status=survivorship.get("probe_status"),
                  survivorship_bias_label=PROBE_STATUSES[survivorship["probe_status"]], holdout_evaluated=False, promotion_authority=False,
                  runtime_authority=False, orders_sent=0, models_fitted=0,
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
    mode.add_argument("--probe", action="store_true", help="which delisted symbols the provider serves; dates only; allowed on a draft")
    mode.add_argument("--acquire", action="store_true", help="freeze the universe and bars once; never claims")
    mode.add_argument("--support-audit", action="store_true", help="session dates only; never reads prices or claims")
    mode.add_argument("--preflight", action="store_true", help="support audit and price checks; never claims")
    mode.add_argument("--fit", action="store_true", help="claim the single trial once and evaluate")
    args = parser.parse_args(argv)
    selected = next(name for name in ("probe", "acquire", "support-audit", "fit", "preflight") if getattr(args, name.replace("-", "_")))
    result = run(args.repository_root, mode=selected, contract_path=args.contract)
    status = result.get("decision", result.get("status"))
    LOGGER.info("SWING_REVERSAL_TRIAL_RESULT", extra={"mode": selected, "status": status})
    return 0 if status in ("probe_complete", "acquisition_written", SUPPORT_OK, "preflight_passed") or "decision" in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
