"""One registered daily trial: buy SPY and QQQ after a weak close (internal bar strength below 0.2); never trades.

At 15:45 New York time on each full session the trial measures, for each ETF on its own, where the
price sits inside the session's range so far: IBS = (price - low) / (high - low), from the
fifteen-minute bars stamped 09:30 to 15:30. Below the contract's threshold it buys half of equity in
that session's closing auction and sells in the next session's closing auction, holding on without
a trade while the signal repeats. Every parameter comes from the contract. The scoring core is pure:
it takes fifteen-minute and daily bars and a session calendar and returns one accounting row per
close-to-close interval.

Modes. All refuse a draft contract, an unlisted campaign or a missing ledger before any data is
touched:

* ``--acquire`` downloads the bars once, keeps regular-session bars only and writes a hash manifest,
  refusing any bar outside development or in 2026 and any degraded download;
* ``--support-audit`` reads sessions and bar times only;
* ``--preflight`` runs the audit, then verifies prices; it never claims the trial;
* ``--fit`` claims the single trial, scores it and writes the report.
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
from functools import cache, partial
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools import intraday_momentum_trial as momentum
from ai_trading.tools import volatility_target_trial as daily
from ai_trading.tools.experiment_ledger import claim_campaign_trial, finish_campaign_trial, register_campaign
from ai_trading.tools.registered_strategy_trial import digest, evidence_signature, write_json
from ai_trading.tools.research_feasibility import NEW_YORK, daily_development_dates
from ai_trading.utils.market_calendar import session_info

LOGGER = get_logger(__name__)
RESET_CONFIG = Path("config/research_reset.json")
CODE_FILES = ("tools/weak_close_trial.py", "tools/intraday_momentum_trial.py", "tools/volatility_target_trial.py",
              "tools/registered_strategy_trial.py", "tools/research_feasibility.py", "tools/experiment_ledger.py",
              "utils/market_calendar.py")
SUPPORT_OK = "support_possible_not_strategy_eligible"
STRESS_ROLE = "required_positive_at_stress"
COST_ROLES = ("decision", STRESS_ROLE, "reported_only")
SYMBOLS = ("SPY", "QQQ")
SLOT_WEIGHT = 1 / len(SYMBOLS)  # one equal slot per ETF; both filled is 100 percent invested, never more
PRICE_COLUMNS = daily.PRICE_COLUMNS
BAR_COLUMNS = ["symbol", "session", "time", *PRICE_COLUMNS]
DAILY_COLUMNS = ["symbol", "session", *PRICE_COLUMNS]
TIMEFRAME, DAILY_TIMEFRAME = "15Min", "1Day"
BAR_MINUTES = 15
REGULAR_OPEN, REGULAR_CLOSE = "09:30", "16:00"
FULL_SESSION = tuple(f"{m // 60:02d}:{m % 60:02d}" for m in range(9 * 60 + 30, 16 * 60, BAR_MINUTES))  # 26 bars
SNAPSHOT_TIME = "15:45"  # the signal is read here; the broker's closing-auction cut-off is 15:50
SNAPSHOT_BARS = tuple(stamp for stamp in FULL_SESSION if stamp < SNAPSHOT_TIME)  # 09:30 to 15:30: 25 bars
PRICE_BAR = SNAPSHOT_BARS[-1]  # its close is the last trade before 15:45
SIGNAL_POSSIBLE = "signal_possible"
EXPECTED_SIGNAL = dict(bar_minutes=BAR_MINUTES, snapshot_time=SNAPSHOT_TIME, comparison="strictly_below",
                       zero_range="no_signal")
EXPECTED_TRADE = dict(entry="market_on_close_same_session", exit="market_on_close_next_session", price="daily_bar_close",
                      repeat_signal="hold_without_trading", early_close_sessions="no_entry")
FETCH_ATTEMPTS = 4
RETRY_PAUSE_SECONDS = 20.0
MAX_INCOMPLETE_SESSION_FRACTION = 0.01  # of full sessions per ETF; above this the download is degraded and refused
CLOSE_TOLERANCE = 0.01  # daily close against the last regular fifteen-minute close
MAX_CLOSE_MISMATCH_FRACTION = 0.01  # of sessions per ETF; above this the preflight blocks


@dataclass(frozen=True)
class WeakCloseSpec:
    """Contract values fixed before any price is read."""

    threshold: float
    tiers: tuple[tuple[str, float, str], ...]  # name, per-side bps, role
    annualization: int
    min_observations: int
    min_years: int
    min_per_year: int
    min_snapshot_sessions: int
    min_signal_sessions: int
    min_positive_years: int
    block: int
    replicates: int
    seed: int
    lower_quantile: float

    def tier(self, role: str) -> tuple[str, float, str]:
        return next(t for t in self.tiers if t[2] == role)


@dataclass(frozen=True)
class SessionCalendar:
    """Development sessions with the fifteen-minute bar stamps each one should have (New York time)."""

    dates: tuple[date, ...]
    stamps: tuple[tuple[str, ...], ...]

    @property
    def sessions(self) -> list[str]:
        return [d.isoformat() for d in self.dates]

    def full(self, index: int) -> bool:
        return self.stamps[index] == FULL_SESSION


def _section(contract: dict[str, Any], name: str) -> dict[str, Any]:
    value = contract.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"contract must declare {name}")
    return value


def contract_spec(contract: dict[str, Any]) -> WeakCloseSpec:
    """Validate everything that must be fixed before data are read; fail closed on ambiguity."""
    if not re.fullmatch(r"[a-z0-9_]+", str(contract.get("campaign_id", ""))):
        raise ValueError("contract campaign_id must name its artifact directory")
    if contract.get("promotion_authority") is not False or contract.get("runtime_authority") is not False:
        raise ValueError("a development trial never carries promotion or runtime authority")
    daily_development_dates(contract)  # Holdout and holdout-year refusal precede any file read.
    if (contract.get("feed") != "sip" or contract.get("adjustment") != "all"
            or contract.get("timeframes") != {"signal": TIMEFRAME, "prices": DAILY_TIMEFRAME}):
        raise ValueError(f"contract must use SIP {TIMEFRAME} and {DAILY_TIMEFRAME} bars with adjustment=all")
    if _section(contract, "symbols") != {"instruments": list(SYMBOLS)}:
        raise ValueError(f"the instruments are {' and '.join(SYMBOLS)}")
    if list(_section(contract, "versions")) != ["long_only"]:
        raise ValueError("the only version is long_only")
    signal, trade = _section(contract, "signal"), _section(contract, "trade")
    if any(signal.get(k) != v or isinstance(signal.get(k), bool) for k, v in EXPECTED_SIGNAL.items()):
        raise ValueError("signal must be the 15:45 IBS of fifteen-minute bars, strictly below the threshold, none on a zero range")
    threshold = daily._number(signal.get("threshold"), "signal threshold", low=0, high=1)
    if any(trade.get(k) != v for k, v in EXPECTED_TRADE.items()):
        raise ValueError("trade must buy in the closing auction and sell in the next one, holding through a repeated signal")
    if trade.get("slot_weight") != SLOT_WEIGHT or isinstance(trade.get("slot_weight"), bool):
        raise ValueError("each ETF gets one equal slot without leverage")
    support, acceptance, bootstrap = (_section(contract, k) for k in ("support_thresholds", "acceptance", "bootstrap"))
    tiers = []
    for tier in contract.get("cost_tiers") or []:
        if tier.get("role") not in COST_ROLES or not isinstance(tier.get("name"), str):
            raise ValueError("cost tier needs a name and a supported role")
        tiers.append((tier["name"], daily._number(tier.get("per_side_bps"), "per_side_bps", low=0, high=1000, low_open=False),
                      tier["role"]))
    roles = Counter(t[2] for t in tiers)
    if len({t[0] for t in tiers}) != len(tiers) or roles["decision"] != 1 or roles[STRESS_ROLE] != 1:
        raise ValueError("cost tiers need unique names, one decision tier and one required stress tier")
    primary, stress = (next(t[1] for t in tiers if t[2] == role) for role in ("decision", STRESS_ROLE))
    if primary != float(contract.get("primary_per_side_cost_bps", math.nan)) or stress <= primary:
        raise ValueError("decision tier must equal primary_per_side_cost_bps and the stress tier must cost more")
    if not all(acceptance.get(k) is True for k in (
            "positive_lower_bound_vs_cash_at_primary", "positive_lower_bound_vs_exposure_matched_at_primary",
            "positive_paired_mean_at_stress", "each_symbol_positive_paired_mean_at_primary")):
        raise ValueError("acceptance criteria cannot be disabled")
    replicates = daily._int(bootstrap.get("replicates"), "bootstrap replicates")
    if replicates % 100:
        raise ValueError("bootstrap replicates must be a positive multiple of 100")
    seed = bootstrap.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    return WeakCloseSpec(
        threshold=threshold, tiers=tuple(tiers), annualization=daily._int(contract.get("annualization"), "annualization"),
        min_observations=daily._int(support.get("minimum_observations"), "minimum_observations"),
        min_years=daily._int(support.get("minimum_supported_years"), "minimum_supported_years"),
        min_per_year=daily._int(support.get("minimum_observations_per_year"), "minimum_observations_per_year"),
        min_snapshot_sessions=daily._int(support.get("minimum_snapshot_sessions_per_symbol"),
                                         "minimum_snapshot_sessions_per_symbol"),
        min_signal_sessions=daily._int(support.get("minimum_signal_sessions_per_symbol"), "minimum_signal_sessions_per_symbol"),
        min_positive_years=daily._int(acceptance.get("minimum_positive_years"), "minimum_positive_years", minimum=0),
        block=daily._int(bootstrap.get("block_days"), "bootstrap block_days"), replicates=replicates, seed=seed,
        lower_quantile=daily._number(bootstrap.get("lower_quantile"), "lower_quantile", low=0, high=0.5))


# --- pure core: calendar and eligibility -----------------------------------------------------------

@cache
def _session_stamps(day: date) -> tuple[str, ...]:
    session = session_info(day)
    grid = pd.date_range(session.start_utc, session.end_utc, freq=f"{BAR_MINUTES}min", inclusive="left")
    return tuple(grid.tz_convert(NEW_YORK).strftime("%H:%M"))


def session_calendar(contract: dict[str, Any]) -> SessionCalendar:
    from ai_trading.utils import market_calendar

    if market_calendar._get_calendar() is None:
        # The rule-based fallback misdates sessions (Juneteenth before 2022, unscheduled closures).
        raise ValueError("exchange calendar unavailable; the rule-based fallback is not accepted for a registered trial")
    dates = tuple(daily_development_dates(contract))
    return SessionCalendar(dates=dates, stamps=tuple(_session_stamps(day) for day in dates))


def snapshot_statuses(calendar: SessionCalendar, present: set[tuple[str, str]]) -> list[str]:
    """One status per session for one ETF, from bar stamps alone.

    A signal is possible on a full session (09:30 to 16:00) whose 25 snapshot bars are all present,
    except the last development session, whose exit would fall in 2026.
    """
    statuses = []
    last = len(calendar.dates) - 1
    for index, session in enumerate(calendar.sessions):
        if index == last:
            statuses.append("last_session")
        elif not calendar.full(index):
            statuses.append("early_close")
        elif not all((session, stamp) in present for stamp in SNAPSHOT_BARS):
            statuses.append("missing_snapshot_bars")
        else:
            statuses.append(SIGNAL_POSSIBLE)
    return statuses


def support_summary(spec: WeakCloseSpec, calendar: SessionCalendar, statuses: dict[str, list[str]],
                    daily_sessions: dict[str, set[str]]) -> dict[str, Any]:
    """Observations by year and snapshot sessions per ETF against the thresholds; nothing here depends on a price."""
    sessions = calendar.sessions
    per_year = Counter(calendar.dates[i].year for i in range(1, len(sessions))
                       if all(sessions[i - 1] in days and sessions[i] in days for days in daily_sessions.values()))
    observations, supported = sum(per_year.values()), sum(n >= spec.min_per_year for n in per_year.values())
    snapshot = {symbol: sum(status == SIGNAL_POSSIBLE for status in statuses[symbol]) for symbol in SYMBOLS}
    reasons = [reason for reason, failed in (("observations_below_minimum", observations < spec.min_observations),
                                             ("supported_years_below_minimum", supported < spec.min_years)) if failed]
    reasons += [f"snapshot_sessions_below_minimum:{s}" for s in SYMBOLS if snapshot[s] < spec.min_snapshot_sessions]
    return dict(observations=observations, observations_by_year={str(y): n for y, n in sorted(per_year.items())},
                supported_years=supported, snapshot_sessions=snapshot,
                sessions_without_signal={s: dict(sorted(Counter(x for x in statuses[s] if x != SIGNAL_POSSIBLE).items()))
                                         for s in SYMBOLS},
                minimum_required=dict(observations=spec.min_observations, supported_years=spec.min_years,
                                      observations_per_year=spec.min_per_year,
                                      snapshot_sessions_per_symbol=spec.min_snapshot_sessions),
                status="blocked_sampling_support" if reasons else SUPPORT_OK, blocking_reasons=reasons)


def _statuses(calendar: SessionCalendar, bars: pd.DataFrame) -> dict[str, list[str]]:
    return {s: snapshot_statuses(calendar, set(zip(part.session, part.time)))
            for s, part in ((s, bars[bars.symbol == s]) for s in SYMBOLS)}


# --- pure core: signal and accounting rows ---------------------------------------------------------

def internal_bar_strength(price: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """(price - low) / (high - low); NaN where the range is zero or a value is missing."""
    width = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.where(width > 0, (price - low) / width, np.nan), dtype=float)


def signal_weights(ibs: np.ndarray, threshold: float) -> np.ndarray:
    """One slot where IBS is strictly below the threshold; NaN (no signal possible, zero range) is cash."""
    finite = np.isfinite(ibs)
    signal = np.zeros(len(ibs), dtype=bool)
    signal[finite] = ibs[finite] < threshold
    return np.where(signal, SLOT_WEIGHT, 0.0)


def snapshot_ibs(bars: pd.DataFrame, calendar: SessionCalendar, statuses: list[str]) -> np.ndarray:
    """One ETF's 15:45 IBS per session from the bars stamped 09:30 to 15:30 only; NaN where no signal is possible."""
    part = bars[bars.time.isin(SNAPSHOT_BARS)]
    grouped = part.groupby("session")
    sessions = calendar.sessions
    high = grouped.high.max().reindex(sessions).to_numpy(dtype=float)
    low = grouped.low.min().reindex(sessions).to_numpy(dtype=float)
    price = part[part.time == PRICE_BAR].set_index("session").close.reindex(sessions).to_numpy(dtype=float)
    ibs = internal_bar_strength(price, high, low)
    ibs[np.array([status != SIGNAL_POSSIBLE for status in statuses])] = np.nan
    return ibs


def build_days(bars: pd.DataFrame, daily_bars: pd.DataFrame, calendar: SessionCalendar,
               spec: WeakCloseSpec) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One row per close-to-close interval (prior session's close to this session's close), and every status.

    For each ETF the row holds the weight decided at the prior session's 15:45 snapshot (``held``),
    the weights decided one session earlier (``before``) and at this session's snapshot (``after``),
    which fix the entry and exit trades, and the daily closes. The reported-only textbook columns use
    each session's final daily bar instead of the snapshot.
    """
    sessions = calendar.sessions
    statuses = _statuses(calendar, bars)
    columns: dict[str, Any] = dict(session=sessions[1:], prior_session=sessions[:-1], year=[s[:4] for s in sessions[1:]])
    for symbol in SYMBOLS:
        frame = daily_bars[daily_bars.symbol == symbol].set_index("session").reindex(sessions)
        if frame[["open", "high", "low", "close"]].isna().any(axis=None):
            raise ValueError(f"daily bar missing for a development session:{symbol}")
        close, opens = frame.close.to_numpy(dtype=float), frame.open.to_numpy(dtype=float)
        ibs = snapshot_ibs(bars[bars.symbol == symbol], calendar, statuses[symbol])
        final = internal_bar_strength(close, frame.high.to_numpy(dtype=float), frame.low.to_numpy(dtype=float))
        final[-1] = np.nan  # no entry on the last development session in either version
        for prefix, values in (("", ibs), ("textbook_", final)):
            weight = signal_weights(values, spec.threshold)
            columns[f"{symbol}_{prefix}signal_ibs"] = values[:-1]
            columns[f"{symbol}_{prefix}held"] = weight[:-1]
            columns[f"{symbol}_{prefix}before"] = np.concatenate([[0.0], weight[:-2]])
            columns[f"{symbol}_{prefix}after"] = weight[1:]
        columns[f"{symbol}_prior_close"], columns[f"{symbol}_open"], columns[f"{symbol}_close"] = close[:-1], opens[1:], close[1:]
        columns[f"{symbol}_return"] = close[1:] / close[:-1] - 1
        columns[f"{symbol}_overnight"] = opens[1:] / close[:-1] - 1
    return pd.DataFrame(columns), statuses


def validate_days(frame: pd.DataFrame, contract: dict[str, Any]) -> None:
    if frame.empty or frame.session.duplicated().any() or not frame.session.is_monotonic_increasing:
        raise ValueError("empty, duplicate or unordered sessions")
    if not (frame.prior_session.ge(contract["development_start"]) & frame.session.le(contract["development_end"])
            & (frame.prior_session < frame.session)).all():
        raise ValueError("session outside development or not after its prior session")
    prices = frame[[f"{s}_{k}" for s in SYMBOLS for k in ("prior_close", "open", "close")]].to_numpy(dtype=float)
    returns = frame[[f"{s}_{k}" for s in SYMBOLS for k in ("return", "overnight")]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or not np.isfinite(returns).all() or (prices <= 0).any():
        raise ValueError("nonfinite values or nonpositive prices")
    for symbol in SYMBOLS:
        for prefix in ("", "textbook_"):
            held, before, after = (frame[f"{symbol}_{prefix}{k}"].to_numpy(dtype=float) for k in ("held", "before", "after"))
            if not np.isin(np.concatenate([held, before, after]), (0.0, SLOT_WEIGHT)).all():
                raise ValueError("weights must be zero or one slot")
            if before[0] != 0 or after[-1] != 0 or (held[1:] != after[:-1]).any() or (before[1:] != held[:-1]).any():
                raise ValueError("position chain broken: every round trip must open and close inside the window")


def symbol_net(frame: pd.DataFrame, symbol: str, per_side_bps: float, *, prefix: str = "") -> np.ndarray:
    """One ETF's net return per interval: the held slot's close-to-close return less a side on each entry and exit."""
    held, before, after = (frame[f"{symbol}_{prefix}{k}"].to_numpy(dtype=float) for k in ("held", "before", "after"))
    sides = np.maximum(held - before, 0.0) + np.maximum(held - after, 0.0)
    return np.asarray(held * frame[f"{symbol}_return"].to_numpy(dtype=float) - per_side_bps / 10_000 * sides, dtype=float)


def matched_control(frame: pd.DataFrame, symbol: str, *, prefix: str = "") -> np.ndarray:
    """The same ETF at the strategy's average weight on every interval, charged no cost."""
    average = float(frame[f"{symbol}_{prefix}held"].mean())
    return np.asarray(average * frame[f"{symbol}_return"].to_numpy(dtype=float), dtype=float)


def portfolio(frame: pd.DataFrame, per_side_bps: float, *, prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Net strategy return and exposure-matched control return per interval, summed over both ETFs."""
    net = sum(symbol_net(frame, s, per_side_bps, prefix=prefix) for s in SYMBOLS)
    control = sum(matched_control(frame, s, prefix=prefix) for s in SYMBOLS)
    return np.asarray(net, dtype=float), np.asarray(control, dtype=float)


def round_trip_lengths(held: np.ndarray) -> list[int]:
    """Lengths, in intervals, of each run of held positions."""
    lengths, run = [], 0
    for value in held:
        if value > 0:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    return lengths + ([run] if run else [])


def _bps(value: float) -> float | None:
    return daily._finite(value * 10_000)


def _lower_bound(values: np.ndarray, spec: WeakCloseSpec) -> float | None:
    means = momentum.bootstrap_means(values, block=spec.block, replicates=spec.replicates, seed=spec.seed)
    return None if means is None else _bps(float(np.quantile(means, spec.lower_quantile)))


def _tier_summary(frame: pd.DataFrame, spec: WeakCloseSpec) -> dict[str, dict[str, Any]]:
    tiers: dict[str, dict[str, Any]] = {}
    for name, per_side, role in spec.tiers:
        net, control = portfolio(frame, per_side)
        tiers[name] = dict(
            per_side_bps=per_side, role=role, net_per_observation_bps=_bps(float(net.mean())),
            net_annualized_percent=daily._finite(float(net.mean()) * spec.annualization * 100),
            exposure_matched_per_observation_bps=_bps(float(control.mean())),
            paired_vs_exposure_matched_bps=_bps(float((net - control).mean())),
            paired_by_symbol_bps={s: _bps(float((symbol_net(frame, s, per_side) - matched_control(frame, s)).mean()))
                                  for s in SYMBOLS},
            sharpe=daily._finite(daily.sharpe(net, spec.annualization)), max_drawdown=daily.max_drawdown(net))
    return tiers


def _reported_only(frame: pd.DataFrame, spec: WeakCloseSpec) -> dict[str, Any]:
    primary = spec.tier("decision")[1]
    net, control = portfolio(frame, primary, prefix="textbook_")
    held = {s: frame[f"{s}_held"].to_numpy(dtype=float) > 0 for s in SYMBOLS}
    final = {s: frame[f"{s}_textbook_held"].to_numpy(dtype=float) > 0 for s in SYMBOLS}
    lengths = {s: round_trip_lengths(frame[f"{s}_held"].to_numpy(dtype=float)) for s in SYMBOLS}
    exposure = sum(frame[f"{s}_held"].to_numpy(dtype=float) for s in SYMBOLS)
    return dict(
        textbook_final_bar_not_executable=dict(
            net_per_observation_bps=_bps(float(net.mean())), paired_vs_exposure_matched_bps=_bps(float((net - control).mean())),
            signal_sessions={s: int(final[s].sum()) for s in SYMBOLS}),
        snapshot_versus_final_signal={s: dict(both=int((held[s] & final[s]).sum()), snapshot_only=int((held[s] & ~final[s]).sum()),
                                              final_only=int((~held[s] & final[s]).sum())) for s in SYMBOLS},
        gross_split_bps={s: dict(close_to_close=_bps(float((frame[f"{s}_held"] * frame[f"{s}_return"]).mean())),
                                 overnight_part=_bps(float((frame[f"{s}_held"] * frame[f"{s}_overnight"]).mean())))
                         for s in SYMBOLS},
        buy_and_hold={s: dict(mean_per_observation_bps=_bps(float(frame[f"{s}_return"].mean())),
                              sharpe=daily._finite(daily.sharpe(frame[f"{s}_return"].to_numpy(dtype=float), spec.annualization)))
                      for s in SYMBOLS},
        average_exposure=daily._finite(float(exposure.mean())),
        sessions_with_any_position=int((exposure > 0).sum()), sessions_with_both=int((exposure > SLOT_WEIGHT).sum()),
        holding_lengths={s: {str(k): v for k, v in sorted(Counter(lengths[s]).items())} for s in SYMBOLS})


def evaluate(frame: pd.DataFrame, spec: WeakCloseSpec) -> dict[str, Any]:
    """The single long-only decision and its reported-only figures."""
    primary, stress = spec.tier("decision"), spec.tier(STRESS_ROLE)
    net, control = portfolio(frame, primary[1])
    paired = net - control
    stressed, stressed_control = portfolio(frame, stress[1])
    signal_sessions = {s: int((frame[f"{s}_held"] > 0).sum()) for s in SYMBOLS}
    round_trips = {s: len(round_trip_lengths(frame[f"{s}_held"].to_numpy(dtype=float))) for s in SYMBOLS}
    supported_years = int((frame.year.value_counts() >= spec.min_per_year).sum())
    support = (len(frame) >= spec.min_observations and supported_years >= spec.min_years
               and all(n >= spec.min_signal_sessions for n in signal_sessions.values()))
    by_year = {str(year): dict(observations=len(part), net_bps=_bps(float(part.net.mean())),
                               paired_vs_exposure_matched_bps=_bps(float(part.paired.mean())))
               for year, part in frame[["year"]].assign(net=net, paired=paired).groupby("year")}
    positive_years = sum(1 for v in by_year.values() if v["paired_vs_exposure_matched_bps"] is not None
                         and v["paired_vs_exposure_matched_bps"] > 0)
    by_symbol = {s: float((symbol_net(frame, s, primary[1]) - matched_control(frame, s)).mean()) for s in SYMBOLS}
    base = dict(decision_tier=primary[0], observations=len(frame), supported_years=supported_years,
                signal_sessions=signal_sessions, round_trips=round_trips, cost_tiers=_tier_summary(frame, spec),
                gross_per_observation_bps=_bps(float(portfolio(frame, 0.0)[0].mean())), by_year=by_year,
                positive_years=positive_years, paired_by_symbol_bps={s: _bps(v) for s, v in by_symbol.items()},
                benchmark="exposure_matched_same_etf", lower_quantile=spec.lower_quantile,
                reported_only=_reported_only(frame, spec))
    bounds = dict(vs_cash_bps=_lower_bound(net, spec), vs_exposure_matched_bps=_lower_bound(paired, spec))
    if bounds["vs_cash_bps"] is None or bounds["vs_exposure_matched_bps"] is None:
        return dict(base, decision="inconclusive_budget_consumed", uncertainty_status="unavailable_fewer_than_two_blocks",
                    criteria={"adequate_support": False})
    criteria = {"adequate_support": bool(support),
                "positive_lower_bound_vs_cash_at_primary": bounds["vs_cash_bps"] > 0,
                "positive_lower_bound_vs_exposure_matched_at_primary": bounds["vs_exposure_matched_bps"] > 0,
                "positive_paired_mean_at_stress": bool((stressed - stressed_control).mean() > 0),
                "consistent_years": positive_years >= spec.min_positive_years,
                "each_symbol_positive_paired_mean_at_primary": all(v > 0 for v in by_symbol.values())}
    decision = ("eligible_for_untouched_test_planning" if all(criteria.values())
                else "retired" if support else "inconclusive_budget_consumed")
    return dict(base, decision=decision, criteria=criteria, lower_bounds=bounds)


# --- acquisition -----------------------------------------------------------------------------------

def _with_retries(call: Callable[[], Any], errors: tuple[type[BaseException], ...], *, label: str) -> Any:
    """A short provider outage must not end the download; the last error is re-raised and nothing is written."""
    for attempt in range(FETCH_ATTEMPTS):
        try:
            return call()
        except errors as exc:
            if attempt == FETCH_ATTEMPTS - 1:
                raise
            LOGGER.warning("WEAK_CLOSE_FETCH_RETRY", extra={"request": label, "error": type(exc).__name__})
            sleep(RETRY_PAUSE_SECONDS * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_rows(data: Any, symbol: str, start: str, end: str, *, intraday: bool) -> list[dict[str, Any]]:
    """SIP bars (adjustment=all) stamped in New York time: fifteen-minute bars one request per calendar
    year, daily bars in one request."""
    from alpaca.common.exceptions import APIError
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    def request(query: Any) -> list[Any]:
        return list(getattr(data.get_stock_bars(query), "data", {}).get(symbol, []))

    spans = ([(max(start, f"{y}-01-01"), min(end, f"{y}-12-31")) for y in range(int(start[:4]), int(end[:4]) + 1)]
             if intraday else [(start, end)])
    rows: list[dict[str, Any]] = []
    for first, last in spans:
        query = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute) if intraday else TimeFrame.Day,
            start=datetime.combine(date.fromisoformat(first), time(0), NEW_YORK),
            end=datetime.combine(date.fromisoformat(last), time(23, 59), NEW_YORK),
            feed=DataFeed.SIP, adjustment=Adjustment.ALL)
        label = f"{symbol}:{TIMEFRAME if intraday else DAILY_TIMEFRAME}:{first[:4]}"
        bars = _with_retries(partial(request, query), (APIError, OSError, ValueError), label=label)
        if not bars:
            raise ValueError(f"no bars returned:{label}")
        for bar in bars:
            stamp = bar.timestamp.astimezone(NEW_YORK)
            row = dict(symbol=symbol, session=stamp.date().isoformat(), time=stamp.strftime("%H:%M"),
                       **{column: float(getattr(bar, column)) for column in PRICE_COLUMNS})
            if not intraday:
                row.pop("time")
            rows.append(row)
    return rows


def regular_session_bars(rows: list[dict[str, Any]], calendar: SessionCalendar) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep bars inside each session's regular hours; refuse a bar inside them but off the fifteen-minute grid."""
    expected = dict(zip(calendar.sessions, calendar.stamps))
    kept, counts = [], Counter(bars_returned=len(rows))
    for row in rows:
        stamps = expected.get(row["session"])
        if stamps is None:
            if REGULAR_OPEN <= row["time"] < REGULAR_CLOSE:
                raise ValueError(f"regular-hours bar on a date the exchange calendar lacks:{row['session']}T{row['time']}")
            counts["off_calendar_dropped"] += 1
        elif row["time"] in stamps:
            kept.append(row)
        elif stamps[0] < row["time"] < _end_of(stamps[-1]):  # inside regular hours but not on the grid
            raise ValueError(f"bar off the fifteen-minute grid:{row['session']}T{row['time']}")
        else:
            counts["extended_hours_dropped"] += 1
    frame = pd.DataFrame(kept, columns=BAR_COLUMNS).sort_values(["symbol", "session", "time"], ignore_index=True)
    if frame.empty or frame.duplicated(["symbol", "session", "time"]).any():
        raise ValueError("regular-session bars empty or duplicated")
    return frame, dict(counts, regular_bars=len(frame), sessions_with_bars=int(frame.session.nunique()))


def _end_of(stamp: str) -> str:
    """The end of the bar stamped ``stamp``: the session's close for its last bar."""
    minute = int(stamp[:2]) * 60 + int(stamp[3:]) + BAR_MINUTES
    return f"{minute // 60:02d}:{minute % 60:02d}"


def daily_frame(rows: list[dict[str, Any]], calendar: SessionCalendar, symbol: str) -> pd.DataFrame:
    """One daily bar per development session; refuse off-calendar, duplicated or missing sessions."""
    frame = pd.DataFrame(rows, columns=DAILY_COLUMNS).sort_values("session", ignore_index=True)
    if frame.session.duplicated().any():
        raise ValueError(f"duplicate daily bars:{symbol}")
    off = sorted(set(frame.session) - set(calendar.sessions))
    if off:
        raise ValueError(f"daily bar on a date the exchange calendar lacks:{symbol}:{off[0]}")
    missing = sorted(set(calendar.sessions) - set(frame.session))
    if missing:
        raise ValueError(f"daily bars missing:{symbol}:{len(missing)}:first={missing[0]}")
    return frame


def _write_source(root: Path, path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    atomic_write_text(path, frame.to_csv(index=False))
    return dict(path=str(path.relative_to(root)), sha256=digest(path), rows=len(frame),
                first_session=str(frame.session.min()), last_session=str(frame.session.max()))


def acquire(root: Path, contract: dict[str, Any], canonical: str, *,
            client_factory: Callable[[], momentum.Clients]) -> dict[str, Any]:
    """Download once, keep regular-session bars, write the files and their hash manifest; refuse an existing acquisition."""
    manifest_path = root / contract["acquisition_manifest"]
    if manifest_path.exists():
        raise ValueError("acquisition already exists and is immutable")
    calendar = session_calendar(contract)
    start, end = contract["development_start"], contract["development_end"]
    data = client_factory().data
    intraday = {s: fetch_rows(data, s, start, end, intraday=True) for s in SYMBOLS}
    days = {s: fetch_rows(data, s, start, end, intraday=False) for s in SYMBOLS}
    daily.check_development_sessions({f"{s}_{TIMEFRAME}": [r["session"] for r in rows] for s, rows in intraday.items()}
                                     | {f"{s}_{DAILY_TIMEFRAME}": [r["session"] for r in rows] for s, rows in days.items()},
                                     contract)
    full = [i for i in range(len(calendar.dates)) if calendar.full(i)]
    sources, counts, frames = {}, {}, {}
    for symbol in SYMBOLS:
        bars, symbol_counts = regular_session_bars(intraday[symbol], calendar)
        present = set(zip(bars.session, bars.time))
        incomplete = sum(not all((calendar.sessions[i], stamp) in present for stamp in SNAPSHOT_BARS) for i in full)
        if incomplete > MAX_INCOMPLETE_SESSION_FRACTION * len(full):
            raise ValueError(f"degraded download:{symbol}:incomplete_full_sessions={incomplete}:full_sessions={len(full)}")
        counts[symbol] = dict(symbol_counts, full_sessions=len(full), incomplete_full_sessions=incomplete,
                              early_close_sessions=len(calendar.dates) - len(full))
        frames[symbol] = (bars, daily_frame(days[symbol], calendar, symbol))
    for symbol, (bars, day_bars) in frames.items():  # nothing is written until every check has passed
        sources[f"{symbol}_{TIMEFRAME}"] = _write_source(root, manifest_path.parent / f"{symbol}_{TIMEFRAME}.csv", bars)
        sources[f"{symbol}_{DAILY_TIMEFRAME}"] = _write_source(root, manifest_path.parent / f"{symbol}_{DAILY_TIMEFRAME}.csv",
                                                               day_bars)
    manifest = dict(campaign_id=contract["campaign_id"], contract_hash=canonical, feed="sip", adjustment="all",
                    timeframes=[TIMEFRAME, DAILY_TIMEFRAME], request=dict(start=start, end=end),
                    fetched_at=datetime.now(NEW_YORK).isoformat(), sources=sources, counts=counts,
                    prices_read_for_analysis=False, quality_passed=True)
    write_json(manifest_path, manifest)
    return dict(status="acquisition_written", manifest=str(manifest_path), counts=counts, trial_claimed=False)


# --- reading ---------------------------------------------------------------------------------------

def _sources(manifest: dict[str, Any], timeframe: str) -> list[dict[str, Any]]:
    return [manifest["sources"][f"{symbol}_{timeframe}"] for symbol in SYMBOLS]


def load_stamps(root: Path, manifest: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Symbols, sessions and bar times only; no price column is read."""
    stamps = pd.concat([daily._verified_read(root, source, ["symbol", "session", "time"])
                        for source in _sources(manifest, TIMEFRAME)], ignore_index=True)
    if stamps.duplicated().any():
        raise ValueError("duplicate bars in source")
    sessions = {}
    for symbol, source in zip(SYMBOLS, _sources(manifest, DAILY_TIMEFRAME)):
        frame = daily._verified_read(root, source, ["symbol", "session"])
        if frame.session.duplicated().any() or set(frame.symbol) != {symbol}:
            raise ValueError(f"daily source is not one bar per session for {symbol}")
        sessions[symbol] = set(frame.session)
    return stamps, sessions


def load_prices(root: Path, manifest: dict[str, Any], stamps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Hash-verified bars; invalid fifteen-minute bars are dropped and counted, invalid daily bars counted."""
    bars = pd.concat([daily._verified_read(root, source, BAR_COLUMNS) for source in _sources(manifest, TIMEFRAME)],
                     ignore_index=True)
    if not bars[["symbol", "session", "time"]].equals(stamps):
        raise ValueError("bar stamps changed after audit")
    day_bars = pd.concat([daily._verified_read(root, source, DAILY_COLUMNS) for source in _sources(manifest, DAILY_TIMEFRAME)],
                         ignore_index=True)
    valid, valid_days = daily.valid_daily(bars), daily.valid_daily(day_bars)
    rejected = dict(invalid_prices=int((~valid).sum()), invalid_daily_bars=int((~valid_days).sum()))
    return bars[valid].reset_index(drop=True), day_bars, {k: v for k, v in rejected.items() if v}


def close_mismatches(bars: pd.DataFrame, day_bars: pd.DataFrame, calendar: SessionCalendar) -> dict[str, int]:
    """Sessions per ETF whose daily close differs from the last regular fifteen-minute close by more than the tolerance."""
    last = dict(zip(calendar.sessions, (stamps[-1] for stamps in calendar.stamps)))
    counts = {}
    for symbol in SYMBOLS:
        part = bars[(bars.symbol == symbol) & (bars.time == bars.session.map(last))].set_index("session").close
        closes = day_bars[day_bars.symbol == symbol].set_index("session").close.reindex(part.index)
        counts[symbol] = int(((closes / part - 1).abs() > CLOSE_TOLERANCE).sum())
    return counts


# --- orchestration ---------------------------------------------------------------------------------

def run(root: Path, *, mode: str, contract_path: Path,
        client_factory: Callable[[], momentum.Clients] = momentum._default_clients) -> dict[str, Any]:
    root = root.resolve()
    contract = json.loads((root / contract_path).resolve().read_text())
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    directory = root / "artifacts" / str(contract.get("campaign_id"))
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
        return acquire(root, contract, canonical, client_factory=client_factory)
    manifest = json.loads((root / contract["acquisition_manifest"]).read_text())
    if manifest.get("contract_hash") != canonical or manifest.get("quality_passed") is not True:
        raise ValueError("acquisition manifest does not belong to this registered contract")
    stamps, daily_sessions = load_stamps(root, manifest)
    daily.check_development_sessions({f"{s}_{TIMEFRAME}": stamps.session[stamps.symbol == s].tolist() for s in SYMBOLS}
                                     | {f"{s}_{DAILY_TIMEFRAME}": sorted(daily_sessions[s]) for s in SYMBOLS}, contract)
    calendar = session_calendar(contract)
    blocked = dict(trial_claimed=False, holdout_evaluated=False, promotion_authority=False)
    audit = dict(support_summary(spec, calendar, _statuses(calendar, stamps), daily_sessions),
                 audit="fifteen_minute_and_daily_bar_timestamp_only", development_sessions=len(calendar.dates),
                 bars=len(stamps), acquisition_counts=manifest.get("counts", {}), contract_hash=canonical,
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
    bars, day_bars, rejected = load_prices(root, manifest, stamps)
    validated = support_summary(spec, calendar, _statuses(calendar, bars), daily_sessions)
    mismatches = close_mismatches(bars, day_bars, calendar)
    reasons = list(validated["blocking_reasons"])
    reasons += ["invalid_daily_bars"] if rejected.get("invalid_daily_bars") else []
    reasons += [f"daily_close_mismatch:{s}" for s, n in mismatches.items() if n > MAX_CLOSE_MISMATCH_FRACTION * len(calendar.dates)]
    preflight = dict(blocked, status="blocked_price_validated_support" if reasons else "preflight_passed", reasons=reasons,
                     signature=signature, contract_hash=canonical, observations=validated["observations"],
                     snapshot_sessions=validated["snapshot_sessions"], sessions_without_signal=validated["sessions_without_signal"],
                     daily_close_mismatch_sessions=mismatches, rejection_counts=rejected, code=code_hashes,
                     provenance_sha256=hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest())
    write_json(directory / "preflight.json", preflight)
    if mode != "fit" or reasons:
        return preflight
    claim_campaign_trial(state_path, hypothesis_id=contract["hypotheses"][0], evidence_signature=signature,
                         evaluation_start=contract["development_start"], evaluation_end=contract["development_end"],
                         quality_passed=True)
    # A crash after the claim intentionally prevents automatic re-evaluation.
    frame, statuses = build_days(bars, day_bars, calendar, spec)
    validate_days(frame, contract)
    days_path = directory / "days.csv"
    atomic_write_text(days_path, frame.to_csv(index=False))
    report = evaluate(frame, spec)
    report.update(signature=signature, contract_hash=canonical, output_sha256={days_path.name: digest(days_path)},
                  rejection_counts=rejected, daily_close_mismatch_sessions=mismatches,
                  sessions_without_signal={s: dict(sorted(Counter(x for x in statuses[s] if x != SIGNAL_POSSIBLE).items()))
                                           for s in SYMBOLS},
                  acquisition_counts=manifest.get("counts", {}), holdout_evaluated=False, promotion_authority=False,
                  runtime_authority=False, orders_sent=0, models_fitted=0,
                  evidence_type="historical_daily_close_to_close_markouts_on_a_15_45_signal_not_executable_fills")
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
    mode.add_argument("--acquire", action="store_true", help="download the bars once; never claims")
    mode.add_argument("--support-audit", action="store_true", help="sessions and bar times only; never reads prices or claims")
    mode.add_argument("--preflight", action="store_true", help="support audit and price checks; never claims")
    mode.add_argument("--fit", action="store_true", help="claim the single trial once and evaluate")
    args = parser.parse_args(argv)
    selected = next(name for name in ("acquire", "support-audit", "fit", "preflight") if getattr(args, name.replace("-", "_")))
    result = run(args.repository_root, mode=selected, contract_path=args.contract)
    status = result.get("decision", result.get("status"))
    LOGGER.info("WEAK_CLOSE_TRIAL_RESULT", extra={"mode": selected, "status": status})
    return 0 if status in ("acquisition_written", SUPPORT_OK, "preflight_passed") or "decision" in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
