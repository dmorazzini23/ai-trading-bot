"""One registered intraday trial: SPY end-of-day momentum from the first half-hour signal; never trades.

Each full session the trial takes the sign of SPY's return from the prior session's last regular
price to 10:00 New York time. On a positive sign it holds SPY from the open of the 15:30 half-hour
bar to that bar's close (long-only, the decision); the long-short version also shorts on a negative
sign. Every parameter comes from the contract. The scoring core is pure: it takes half-hour bars
and a session calendar and returns one accounting row per eligible session.

Modes. All refuse a draft contract, an unlisted campaign or a missing ledger before any data is
touched:

* ``--acquire`` downloads the half-hour bars once, keeps regular-session bars only and writes a
  hash manifest, refusing any bar outside development or in 2026 and any degraded download;
* ``--support-audit`` reads bar timestamps only;
* ``--preflight`` runs the audit, then verifies prices; it never claims the trial;
* ``--fit`` claims the single trial, scores both versions and writes the report.
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
from ai_trading.tools import volatility_target_trial as daily
from ai_trading.tools.experiment_ledger import claim_campaign_trial, finish_campaign_trial, register_campaign
from ai_trading.tools.registered_strategy_trial import digest, evidence_signature, write_json
from ai_trading.tools.research_feasibility import NEW_YORK, daily_development_dates
from ai_trading.utils.market_calendar import session_info

LOGGER = get_logger(__name__)
RESET_CONFIG = Path("config/research_reset.json")
CODE_FILES = ("tools/intraday_momentum_trial.py", "tools/volatility_target_trial.py", "tools/registered_strategy_trial.py",
              "tools/research_feasibility.py", "tools/experiment_ledger.py", "utils/market_calendar.py")
SUPPORT_OK = "support_possible_not_strategy_eligible"
STRESS_ROLE = "required_positive_at_stress"
COST_ROLES = ("decision", STRESS_ROLE, "reported_only")
VERSIONS = ("long_only", "long_short")
VARIANTS = ("both_predictors_agree", "high_volatility")
PRICE_COLUMNS = daily.PRICE_COLUMNS
BAR_COLUMNS = ["session", "time", *PRICE_COLUMNS]
INSTRUMENT = "SPY"
TIMEFRAME = "30Min"
BAR_MINUTES = 30
FULL_SESSION = tuple(f"{m // 60:02d}:{m % 60:02d}" for m in range(9 * 60 + 30, 16 * 60, BAR_MINUTES))  # 13 bars
SIGNAL_BAR = "09:30"  # its close is the last regular price before 10:00
SECOND_BARS = ("14:30", "15:00")  # close of the first to close of the second is the 15:00-15:30 return
TRADE_BAR = "15:30"  # entered at its open, exited at its close
ELIGIBLE = "eligible"
FETCH_ATTEMPTS = 4
RETRY_PAUSE_SECONDS = 20.0
MAX_INCOMPLETE_SESSION_FRACTION = 0.01  # of full sessions; above this the download is degraded and refused


@dataclass(frozen=True)
class MomentumSpec:
    """Contract values fixed before any price is read."""

    tiers: tuple[tuple[str, float, str], ...]  # name, per-side bps, role
    annualization: int
    vol_lookback: int
    vol_reference: int
    min_days: int
    min_years: int
    min_per_year: int
    min_traded: int
    min_positive_years: int
    block: int
    replicates: int
    seed: int
    lower_quantile: float

    def tier(self, role: str) -> tuple[str, float, str]:
        return next(t for t in self.tiers if t[2] == role)


@dataclass(frozen=True)
class SessionCalendar:
    """Development sessions with the half-hour bar stamps each one should have (New York time)."""

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


def contract_spec(contract: dict[str, Any]) -> MomentumSpec:
    """Validate everything that must be fixed before data are read; fail closed on ambiguity."""
    if not re.fullmatch(r"[a-z0-9_]+", str(contract.get("campaign_id", ""))):
        raise ValueError("contract campaign_id must name its artifact directory")
    if contract.get("promotion_authority") is not False or contract.get("runtime_authority") is not False:
        raise ValueError("a development trial never carries promotion or runtime authority")
    daily_development_dates(contract)  # Holdout and holdout-year refusal precede any file read.
    if contract.get("feed") != "sip" or contract.get("adjustment") != "all" or contract.get("timeframe") != TIMEFRAME:
        raise ValueError(f"contract must use SIP {TIMEFRAME} bars with adjustment=all")
    if _section(contract, "symbols") != {"instrument": INSTRUMENT}:
        raise ValueError(f"the only instrument is {INSTRUMENT}")
    if list(_section(contract, "versions")) != list(VERSIONS):
        raise ValueError("versions must be long_only (primary) then long_short (secondary)")
    signal, trade = _section(contract, "signal"), _section(contract, "trade")
    if (signal.get("bar") != SIGNAL_BAR or signal.get("reference") != "prior_session_final_regular_bar_close"
            or signal.get("threshold") != 0 or isinstance(signal.get("threshold"), bool)):
        raise ValueError("signal must be the sign of the return from the prior session's close to 10:00, with no threshold")
    if (trade.get("entry_bar") != TRADE_BAR or trade.get("exit_bar") != TRADE_BAR or trade.get("entry_price") != "open"
            or trade.get("exit_price") != "close" or trade.get("exposure") != 1.0 or isinstance(trade.get("exposure"), bool)):
        raise ValueError("trade must enter at the 15:30 bar open, exit at its close, fully invested without leverage")
    if trade.get("early_close_sessions") != "skip" or trade.get("sessions_with_missing_bars") != "skip":
        raise ValueError("early closes and sessions with missing bars must be skipped")
    reported = _section(_section(contract, "reported_only_variants"), "high_volatility")
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
    if not all(acceptance.get(k) is True for k in ("positive_lower_bound_vs_cash_at_primary", "positive_at_stress",
                                                   "long_only_positive_mean_vs_always_in_at_primary")):
        raise ValueError("acceptance criteria cannot be disabled")
    replicates = daily._int(bootstrap.get("replicates"), "bootstrap replicates")
    if replicates % 100:
        raise ValueError("bootstrap replicates must be a positive multiple of 100")
    seed = bootstrap.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    return MomentumSpec(
        tiers=tuple(tiers), annualization=daily._int(contract.get("annualization"), "annualization"),
        vol_lookback=daily._int(reported.get("lookback_sessions"), "high_volatility lookback_sessions", minimum=2),
        vol_reference=daily._int(reported.get("reference_sessions"), "high_volatility reference_sessions", minimum=2),
        min_days=daily._int(support.get("minimum_eligible_days"), "minimum_eligible_days"),
        min_years=daily._int(support.get("minimum_supported_years"), "minimum_supported_years"),
        min_per_year=daily._int(support.get("minimum_eligible_days_per_year"), "minimum_eligible_days_per_year"),
        min_traded=daily._int(support.get("minimum_traded_days"), "minimum_traded_days"),
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
    dates = tuple(daily_development_dates(contract))
    return SessionCalendar(dates=dates, stamps=tuple(_session_stamps(day) for day in dates))


def eligibility(calendar: SessionCalendar, present: set[tuple[str, str]]) -> list[str]:
    """One status per session from bar stamps alone.

    A session is eligible when it is a full session (09:30 to 16:00), every one of its 13
    half-hour bars is present, and the previous session's final regular bar is present. Early
    closes are skipped but their final bar still serves as the next session's prior close.
    """
    sessions = calendar.sessions
    statuses = []
    for index, session in enumerate(sessions):
        if index == 0:
            statuses.append("no_prior_session")
        elif not calendar.full(index):
            statuses.append("early_close")
        elif not all((session, stamp) in present for stamp in FULL_SESSION):
            statuses.append("missing_bars")
        elif (sessions[index - 1], calendar.stamps[index - 1][-1]) not in present:
            statuses.append("prior_close_missing")
        else:
            statuses.append(ELIGIBLE)
    return statuses


def support_summary(spec: MomentumSpec, calendar: SessionCalendar, statuses: list[str]) -> dict[str, Any]:
    """Eligible sessions by year against the contract thresholds; nothing here depends on a price."""
    per_year = Counter(day.year for day, status in zip(calendar.dates, statuses) if status == ELIGIBLE)
    eligible, supported = sum(per_year.values()), sum(n >= spec.min_per_year for n in per_year.values())
    reasons = [reason for reason, failed in (("eligible_days_below_minimum", eligible < spec.min_days),
                                             ("supported_years_below_minimum", supported < spec.min_years)) if failed]
    return dict(eligible_days=eligible, eligible_days_by_year={str(y): n for y, n in sorted(per_year.items())},
                supported_years=supported, skipped_sessions=dict(sorted(Counter(s for s in statuses if s != ELIGIBLE).items())),
                minimum_required=dict(eligible_days=spec.min_days, supported_years=spec.min_years,
                                      eligible_days_per_year=spec.min_per_year),
                status="blocked_sampling_support" if reasons else SUPPORT_OK, blocking_reasons=reasons)


# --- pure core: accounting rows --------------------------------------------------------------------

def build_days(bars: pd.DataFrame, calendar: SessionCalendar, spec: MomentumSpec) -> tuple[pd.DataFrame, list[str]]:
    """One row per eligible session, and every session's status.

    The signal uses the prior session's final regular close and the close of the 09:30 bar only.
    The trade is the open and the close of the 15:30 bar. The reported-only second predictor
    uses the closes of the 14:30 and 15:00 bars; the reported-only volatility flag uses final
    closes through the prior session.
    """
    present = set(zip(bars.session, bars.time))
    statuses = eligibility(calendar, present)
    opens = bars.pivot(index="session", columns="time", values="open")
    closes = bars.pivot(index="session", columns="time", values="close")
    sessions = calendar.sessions
    final_close = pd.Series([float(closes.at[s, stamps[-1]]) if (s, stamps[-1]) in present else math.nan
                             for s, stamps in zip(sessions, calendar.stamps)])
    volatility = np.log(final_close).diff().rolling(spec.vol_lookback).std(ddof=1) * math.sqrt(spec.annualization)
    reference = volatility.rolling(spec.vol_reference, min_periods=spec.vol_reference // 2).median()
    known_volatility, high = volatility.shift(1), (volatility > reference).shift(1, fill_value=False)
    rows = []
    for index, session in enumerate(sessions):
        if statuses[index] != ELIGIBLE:
            continue
        prior, at_ten = float(final_close.iat[index - 1]), float(closes.at[session, SIGNAL_BAR])
        entry, exit_price = float(opens.at[session, TRADE_BAR]), float(closes.at[session, TRADE_BAR])
        rows.append(dict(
            session=session, year=session[:4], prior_session=sessions[index - 1], prior_close=prior, signal_price=at_ten,
            signal_return=at_ten / prior - 1,
            second_return=float(closes.at[session, SECOND_BARS[1]]) / float(closes.at[session, SECOND_BARS[0]]) - 1,
            entry_price=entry, exit_price=exit_price, trade_return=exit_price / entry - 1,
            trailing_volatility=float(known_volatility.iat[index]), high_volatility=bool(high.iat[index])))
    return pd.DataFrame(rows), statuses


def validate_days(frame: pd.DataFrame, contract: dict[str, Any]) -> None:
    if frame.empty or frame.session.duplicated().any() or not frame.session.is_monotonic_increasing:
        raise ValueError("empty, duplicate or unordered sessions")
    if not (frame.prior_session.ge(contract["development_start"]) & frame.session.le(contract["development_end"])
            & (frame.prior_session < frame.session)).all():
        raise ValueError("session outside development or not after its prior session")
    numeric = frame[["prior_close", "signal_price", "signal_return", "second_return", "entry_price", "exit_price",
                     "trade_return"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (frame[["prior_close", "signal_price", "entry_price", "exit_price"]] <= 0).any(axis=None):
        raise ValueError("nonfinite values or nonpositive prices")


def positions(frame: pd.DataFrame, version: str, *, variant: str | None = None) -> np.ndarray:
    """Weight held from 15:30 to the close: +1 long, -1 short, 0 flat. A zero signal is flat."""
    if version not in VERSIONS:
        raise ValueError("unknown version")
    sign = np.sign(frame.signal_return.to_numpy(dtype=float))
    if variant == "both_predictors_agree":
        sign = np.where(np.sign(frame.second_return.to_numpy(dtype=float)) == sign, sign, 0.0)
    elif variant == "high_volatility":
        sign = np.where(frame.high_volatility.to_numpy(dtype=bool), sign, 0.0)
    elif variant is not None:
        raise ValueError("unknown variant")
    return np.asarray(np.maximum(sign, 0.0) if version == "long_only" else sign, dtype=float)


def net_returns(position: np.ndarray, trade_return: np.ndarray, per_side_bps: float) -> np.ndarray:
    """Return per eligible session: the position's markout less two sides of cost on every traded session."""
    return np.asarray(position * trade_return - 2 * per_side_bps / 10_000 * np.abs(position), dtype=float)


def bootstrap_means(values: np.ndarray, *, block: int, replicates: int, seed: int) -> np.ndarray | None:
    """Means of circular block resamples; None when fewer than two blocks exist.

    Draws the same block starts as ``registered_strategy_trial.bootstrap_indices`` and sums each
    block from a cumulative sum, so 10,000 replicates of about 2,500 sessions need no index table.
    """
    n = len(values)
    if n < 2 * block:
        return None
    starts = np.random.default_rng(seed).integers(0, n, size=(replicates, math.ceil(n / block)))
    running = np.concatenate([[0.0], np.cumsum(np.concatenate([values, values[:block]]))])
    tail = n - block * (starts.shape[1] - 1)  # the final block is cut so each resample has n sessions
    whole = (running[starts[:, :-1] + block] - running[starts[:, :-1]]).sum(axis=1)
    return np.asarray((whole + running[starts[:, -1] + tail] - running[starts[:, -1]]) / n, dtype=float)


def _bps(value: float) -> float | None:
    return daily._finite(value * 10_000)


def _lower_bound(values: np.ndarray, spec: MomentumSpec) -> float | None:
    means = bootstrap_means(values, block=spec.block, replicates=spec.replicates, seed=spec.seed)
    return None if means is None else _bps(float(np.quantile(means, spec.lower_quantile)))


def _tier_summary(position: np.ndarray, trade_return: np.ndarray, spec: MomentumSpec) -> dict[str, dict[str, Any]]:
    traded = position != 0
    tiers: dict[str, dict[str, Any]] = {}
    for name, per_side, role in spec.tiers:
        net, always = net_returns(position, trade_return, per_side), net_returns(np.ones(len(position)), trade_return, per_side)
        tiers[name] = dict(per_side_bps=per_side, role=role, net_per_eligible_day_bps=_bps(float(net.mean())),
                           net_per_traded_day_bps=_bps(float(net[traded].mean())) if traded.any() else None,
                           net_annualized_percent=daily._finite(float(net.mean()) * spec.annualization * 100),
                           always_in_net_per_eligible_day_bps=_bps(float(always.mean())),
                           paired_vs_always_in_bps=_bps(float((net - always).mean())),
                           sharpe=daily._finite(daily.sharpe(net, spec.annualization)), max_drawdown=daily.max_drawdown(net))
    return tiers


def _version_report(frame: pd.DataFrame, spec: MomentumSpec, *, version: str, variant: str | None = None) -> dict[str, Any]:
    position, trade_return = positions(frame, version, variant=variant), frame.trade_return.to_numpy(dtype=float)
    traded_days = int((position != 0).sum())
    base = dict(eligible_days=len(frame), traded_days=traded_days, long_days=int((position > 0).sum()),
                short_days=int((position < 0).sum()), cost_tiers=_tier_summary(position, trade_return, spec),
                gross_per_eligible_day_bps=_bps(float((position * trade_return).mean())))
    if variant is not None:
        return base
    supported_years = int((frame.year.value_counts() >= spec.min_per_year).sum())
    support = len(frame) >= spec.min_days and supported_years >= spec.min_years and traded_days >= spec.min_traded
    primary, stress = spec.tier("decision"), spec.tier(STRESS_ROLE)
    net, stressed = net_returns(position, trade_return, primary[1]), net_returns(position, trade_return, stress[1])
    paired = net - net_returns(np.ones(len(frame)), trade_return, primary[1])
    by_year = {str(year): dict(eligible_days=len(part), traded_days=int((part.position != 0).sum()),
                               net_bps=_bps(float(part.net.mean())), paired_vs_always_in_bps=_bps(float(part.paired.mean())))
               for year, part in frame[["year"]].assign(position=position, net=net, paired=paired).groupby("year")}
    positive_years = sum(1 for v in by_year.values() if v["net_bps"] is not None and v["net_bps"] > 0)
    base.update(supported_years=supported_years, by_year=by_year, benchmark="cash")
    bounds = dict(vs_cash_bps=_lower_bound(net, spec), vs_always_in_bps_reported_only=_lower_bound(paired, spec))
    if bounds["vs_cash_bps"] is None:
        return dict(base, decision="inconclusive_budget_consumed", uncertainty_status="unavailable_fewer_than_two_blocks",
                    criteria={"adequate_support": False})
    criteria = {"adequate_support": bool(support),
                "positive_lower_bound_vs_cash_at_primary": bounds["vs_cash_bps"] > 0,
                "positive_at_stress": bool(stressed.mean() > 0),
                "consistent_years": positive_years >= spec.min_positive_years}
    if version == "long_only":
        criteria["positive_mean_vs_always_in_at_primary"] = bool(paired.mean() > 0)
    decision = ("eligible_for_untouched_test_planning" if all(criteria.values())
                else "retired" if support else "inconclusive_budget_consumed")
    return dict(base, decision=decision, criteria=criteria, positive_years=positive_years, lower_bounds=bounds,
                lower_quantile=spec.lower_quantile)


def evaluate(frame: pd.DataFrame, spec: MomentumSpec) -> dict[str, Any]:
    """Both versions; the ledger decision is the long-only (primary) decision."""
    versions = {v: _version_report(frame, spec, version=v) for v in VERSIONS}
    variants = {name: {v: _version_report(frame, spec, version=v, variant=name) for v in VERSIONS} for name in VARIANTS}
    trade_return, long_only = frame.trade_return.to_numpy(dtype=float), positions(frame, "long_only")
    return dict(decision=versions["long_only"]["decision"], secondary_decision=versions["long_short"]["decision"],
                decision_tier=spec.tier("decision")[0], versions=versions, reported_only_variants=variants,
                reported_only=dict(
                    gross_timing_difference_vs_always_in_bps=_bps(float(((long_only - 1) * trade_return).mean())),
                    last_half_hour_gross_per_eligible_day_bps=_bps(float(trade_return.mean())),
                    zero_signal_days=int((frame.signal_return == 0).sum()),
                    high_volatility_days=int(frame.high_volatility.sum())))


# --- acquisition -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Clients:
    trading: Any
    data: Any


def _default_clients() -> Clients:
    from ai_trading.config.managed_secrets import hydrate_managed_secrets

    hydrate_managed_secrets(required_keys=("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
    from ai_trading.alpaca_api import _get_rest

    return Clients(trading=_get_rest(bars=False), data=_get_rest(bars=True))


def _with_retries(call: Callable[[], Any], errors: tuple[type[BaseException], ...], *, label: str) -> Any:
    """A short provider outage must not end the download; the last error is re-raised and nothing is written."""
    for attempt in range(FETCH_ATTEMPTS):
        try:
            return call()
        except errors as exc:
            if attempt == FETCH_ATTEMPTS - 1:
                raise
            LOGGER.warning("INTRADAY_MOMENTUM_FETCH_RETRY", extra={"request": label, "error": type(exc).__name__})
            sleep(RETRY_PAUSE_SECONDS * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_half_hour_bars(data: Any, start: str, end: str) -> list[dict[str, Any]]:
    """Every half-hour SIP bar (adjustment=all), one request per calendar year, stamped in New York time."""
    from alpaca.common.exceptions import APIError
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    def request(query: Any) -> list[Any]:
        return list(getattr(data.get_stock_bars(query), "data", {}).get(INSTRUMENT, []))

    rows: list[dict[str, Any]] = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        first, last = max(start, f"{year}-01-01"), min(end, f"{year}-12-31")
        query = StockBarsRequest(
            symbol_or_symbols=INSTRUMENT, timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
            start=datetime.combine(date.fromisoformat(first), time(0), NEW_YORK),
            end=datetime.combine(date.fromisoformat(last), time(23, 59), NEW_YORK),
            feed=DataFeed.SIP, adjustment=Adjustment.ALL)
        bars = _with_retries(partial(request, query), (APIError, OSError, ValueError), label=str(year))
        if not bars:
            raise ValueError(f"no bars returned:{INSTRUMENT}:{year}")
        for bar in bars:
            stamp = bar.timestamp.astimezone(NEW_YORK)
            rows.append(dict(session=stamp.date().isoformat(), time=stamp.strftime("%H:%M"),
                             **{column: float(getattr(bar, column)) for column in PRICE_COLUMNS}))
    return rows


def regular_session_bars(rows: list[dict[str, Any]], calendar: SessionCalendar) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep bars inside each session's regular hours; refuse a bar that is inside them but off the half-hour grid."""
    expected = dict(zip(calendar.sessions, calendar.stamps))
    kept, counts = [], Counter(bars_returned=len(rows))
    for row in rows:
        stamps = expected.get(row["session"])
        if stamps is None:
            counts["off_calendar_dropped"] += 1
        elif row["time"] in stamps:
            kept.append(row)
        elif stamps[0] < row["time"] < f"{stamps[-1][:3]}{int(stamps[-1][3:]) + BAR_MINUTES}":  # before the session's close
            raise ValueError(f"bar off the half-hour grid:{row['session']}T{row['time']}")
        else:
            counts["extended_hours_dropped"] += 1
    frame = pd.DataFrame(kept, columns=BAR_COLUMNS).sort_values(["session", "time"], ignore_index=True)
    if frame.empty or frame.duplicated(["session", "time"]).any():
        raise ValueError("regular-session bars empty or duplicated")
    return frame, dict(counts, regular_bars=len(frame), sessions_with_bars=int(frame.session.nunique()))


def acquire(root: Path, contract: dict[str, Any], canonical: str, *, client_factory: Callable[[], Clients]) -> dict[str, Any]:
    """Download once, keep regular-session bars, write the file and its hash manifest; refuse an existing acquisition."""
    manifest_path = root / contract["acquisition_manifest"]
    if manifest_path.exists():
        raise ValueError("acquisition already exists and is immutable")
    calendar = session_calendar(contract)
    start, end = contract["development_start"], contract["development_end"]
    rows = fetch_half_hour_bars(client_factory().data, start, end)
    daily.check_development_sessions({INSTRUMENT: [row["session"] for row in rows]}, contract)
    frame, counts = regular_session_bars(rows, calendar)
    present = set(zip(frame.session, frame.time))
    full = [s for i, s in enumerate(calendar.sessions) if calendar.full(i)]
    incomplete = sum(not all((s, stamp) in present for stamp in FULL_SESSION) for s in full)
    if incomplete > MAX_INCOMPLETE_SESSION_FRACTION * len(full):
        raise ValueError(f"degraded download:incomplete_full_sessions={incomplete}:full_sessions={len(full)}")
    path = manifest_path.parent / f"{INSTRUMENT}_{TIMEFRAME}.csv"
    atomic_write_text(path, frame.to_csv(index=False))
    source = dict(path=str(path.relative_to(root)), sha256=digest(path), rows=len(frame),
                  first_session=str(frame.session.iloc[0]), last_session=str(frame.session.iloc[-1]))
    counts.update(full_sessions=len(full), incomplete_full_sessions=incomplete, early_close_sessions=len(calendar.dates) - len(full))
    manifest = dict(campaign_id=contract["campaign_id"], contract_hash=canonical, feed="sip", adjustment="all", timeframe=TIMEFRAME,
                    request=dict(start=start, end=end), fetched_at=datetime.now(NEW_YORK).isoformat(),
                    sources={INSTRUMENT: source}, counts=counts, prices_read_for_analysis=False, quality_passed=True)
    write_json(manifest_path, manifest)
    return dict(status="acquisition_written", manifest=str(manifest_path), counts=counts, trial_claimed=False)


# --- reading ---------------------------------------------------------------------------------------

def load_stamps(root: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    """Session and bar time only; no price column is read."""
    frame = daily._verified_read(root, manifest["sources"][INSTRUMENT], ["session", "time"])
    if frame.duplicated().any():
        raise ValueError("duplicate bars in source")
    return frame


def load_bars(root: Path, manifest: dict[str, Any], stamps: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Hash-verified bars with invalid prices dropped and counted; a dropped bar makes its session incomplete."""
    frame = daily._verified_read(root, manifest["sources"][INSTRUMENT], BAR_COLUMNS)
    if not frame[["session", "time"]].equals(stamps):
        raise ValueError("bar stamps changed after audit")
    valid = daily.valid_daily(frame)
    return frame[valid].reset_index(drop=True), {k: v for k, v in dict(invalid_prices=int((~valid).sum())).items() if v}


# --- orchestration ---------------------------------------------------------------------------------

def run(root: Path, *, mode: str, contract_path: Path, client_factory: Callable[[], Clients] = _default_clients) -> dict[str, Any]:
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
    stamps = load_stamps(root, manifest)
    daily.check_development_sessions({INSTRUMENT: stamps.session.tolist()}, contract)
    calendar = session_calendar(contract)
    blocked = dict(trial_claimed=False, holdout_evaluated=False, promotion_authority=False)
    audit = dict(support_summary(spec, calendar, eligibility(calendar, set(zip(stamps.session, stamps.time)))),
                 audit="half_hour_bar_timestamp_only", development_sessions=len(calendar.dates), bars=len(stamps),
                 acquisition_counts=manifest.get("counts", {}), contract_hash=canonical, prices_read_for_analysis=False,
                 strategy_outcomes_computed=False, **blocked)
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
    bars, rejected = load_bars(root, manifest, stamps)
    validated = support_summary(spec, calendar, eligibility(calendar, set(zip(bars.session, bars.time))))
    preflight = dict(blocked, status="preflight_passed" if validated["status"] == SUPPORT_OK else "blocked_price_validated_support",
                     signature=signature, contract_hash=canonical, eligible_days=validated["eligible_days"],
                     skipped_sessions=validated["skipped_sessions"], rejection_counts=rejected, code=code_hashes,
                     provenance_sha256=hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest())
    write_json(directory / "preflight.json", preflight)
    if mode != "fit" or preflight["status"] != "preflight_passed":
        return preflight
    claim_campaign_trial(state_path, hypothesis_id=contract["hypotheses"][0], evidence_signature=signature,
                         evaluation_start=contract["development_start"], evaluation_end=contract["development_end"],
                         quality_passed=True)
    # A crash after the claim intentionally prevents automatic re-evaluation.
    frame, statuses = build_days(bars, calendar, spec)
    validate_days(frame, contract)
    days_path = directory / "days.csv"
    atomic_write_text(days_path, frame.to_csv(index=False))
    report = evaluate(frame, spec)
    report.update(signature=signature, contract_hash=canonical, output_sha256={days_path.name: digest(days_path)},
                  rejection_counts=rejected, skipped_sessions=dict(sorted(Counter(s for s in statuses if s != ELIGIBLE).items())),
                  acquisition_counts=manifest.get("counts", {}), holdout_evaluated=False, promotion_authority=False,
                  runtime_authority=False, orders_sent=0, models_fitted=0,
                  evidence_type="historical_half_hour_bar_open_to_close_markouts_not_executable_fills")
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
    mode.add_argument("--acquire", action="store_true", help="download the half-hour bars once; never claims")
    mode.add_argument("--support-audit", action="store_true", help="bar timestamps only; never reads prices or claims")
    mode.add_argument("--preflight", action="store_true", help="support audit and price checks; never claims")
    mode.add_argument("--fit", action="store_true", help="claim the single trial once and evaluate")
    args = parser.parse_args(argv)
    selected = next(name for name in ("acquire", "support-audit", "fit", "preflight") if getattr(args, name.replace("-", "_")))
    result = run(args.repository_root, mode=selected, contract_path=args.contract)
    status = result.get("decision", result.get("status"))
    LOGGER.info("INTRADAY_MOMENTUM_TRIAL_RESULT", extra={"mode": selected, "status": status})
    return 0 if status in ("acquisition_written", SUPPORT_OK, "preflight_passed") or "decision" in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
