"""One registered intraday trial: 5-minute opening-range breakout on "stocks in play"; never trades.

Each session the trial takes the US stocks that pass a price, volume and range filter, ranks them
by the volume of their first five minutes against their own average for that window over the
prior 14 sessions, and keeps the top 20 with at least average volume. A name whose first
five-minute bar closed above its open is bought once a later one-minute bar trades through that
bar's high; below its open, it is sold short through its low. The deciding fill model enters at
the next one-minute bar's open. A filled order has a stop loss one tenth of the 14-session
average true range from the fill and is otherwise closed at the session's end. Three other fill
models are scored beside it and reported only. Every parameter comes from the contract. The
scoring core is pure: it takes bar tables and a session calendar and returns one row per order.

Modes. All but ``--probe`` refuse a draft contract, an unlisted campaign or a missing ledger
before any data is touched:

* ``--probe`` asks the provider for bar counts and stamps on a few symbols and sessions, among
  them delisted companies; no price is kept; allowed on a draft because its answer belongs in
  the contract;
* ``--acquire`` freezes the asset list, monthly and daily raw bars, the opening five-minute bars,
  the daily selection and the minute bars of the selected names, with a hash manifest;
* ``--support-audit`` reads sessions, symbols and bar times only;
* ``--preflight`` runs the audit, rebuilds the selection from the frozen bars and checks prices;
  it never computes a trade and never claims the trial;
* ``--fit`` claims the single trial, scores both versions and writes the report.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import partial
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools import intraday_momentum_trial as intraday
from ai_trading.tools import swing_reversal_trial as swing
from ai_trading.tools import volatility_target_trial as daily
from ai_trading.tools.experiment_ledger import claim_campaign_trial, finish_campaign_trial, register_campaign
from ai_trading.tools.registered_strategy_trial import digest, evidence_signature, write_json
from ai_trading.tools.research_feasibility import NEW_YORK, daily_development_dates

LOGGER = get_logger(__name__)
RESET_CONFIG = Path("config/research_reset.json")
CODE_FILES = ("tools/opening_range_trial.py", "tools/intraday_momentum_trial.py", "tools/swing_reversal_trial.py",
              "tools/volatility_target_trial.py", "tools/registered_strategy_trial.py", "tools/research_feasibility.py",
              "tools/experiment_ledger.py", "utils/market_calendar.py")
SUPPORT_OK = "support_possible_not_strategy_eligible"
STRESS_ROLE = "required_positive_at_stress"
COST_ROLES = ("decision", STRESS_ROLE, "reported_only")
VERSIONS = ("long_only", "long_short")
AS_PUBLISHED = "long_short_every_short_taken"  # reported only: no shortability rule, as the paper trades it
PROBE_STATUSES = {"delisted_symbols_served": "none_detected_by_probe",
                  "delisted_symbols_missing_or_partial": "incomplete_direction_unknown"}
FILL_MODELS = ("bar_executable", "conservative", "paper_exact", "lower_bound")
DECIDING_MODEL, CONSERVATIVE_MODEL = FILL_MODELS[0], FILL_MODELS[1]
FILL_RULES = {"deciding_model": DECIDING_MODEL, "reported_only_models": ["conservative", "paper_exact", "lower_bound"],
              "session_end_exit": "last_bar_close"}
EQUAL_NOTIONAL = "equal_notional_sizing"  # reported only: every fill at the leverage cap, the other reading of the paper
MIN_PRICE = 0.01
PRICE_COLUMNS = daily.PRICE_COLUMNS
DAILY_COLUMNS = ["session", "symbol", *PRICE_COLUMNS]
MINUTE_COLUMNS = ["session", "symbol", "time", *PRICE_COLUMNS]
MARKET_COLUMNS = ["session", "time", *PRICE_COLUMNS]
SELECTION_COLUMNS = ["session", "symbol", "rank", "rel_volume", "direction", "trigger", "atr", "stop_distance", "prior_close",
                     "short_ok", "or_open", "or_close"]
OPENING_MINUTES = 5
REGULAR_OPEN = "09:30"
ORDER_TIME = "09:35"  # the first minute an order can be working: the opening range has closed
ELIGIBLE = "eligible"
DAILY_CHUNK = 100
OPENING_CHUNK = 500
PAGE_ROWS = 10_000  # bars per page of the provider's answer
REQUEST_PAUSE_SECONDS = 0.6  # about 100 requests a minute: half the measured limit, which the running bot shares
MAX_FAILED_FETCH_FRACTION = 0.01
MAX_INCOMPLETE_SESSION_FRACTION = 0.01
MIN_OPENING_COVERAGE = 0.90  # of the names asked for on one session; the probe returned 599 of 600 for current liquid names
SPLIT_TOLERANCE = 0.01  # a raw/split-adjusted close ratio that moves by more than this marks a split
EXCLUDED = "excluded_provider_outage"  # a session status: listed in the contract, unscored like market_incomplete
EXCLUDED_REASONS = frozenset({"provider_outage"})
EXCLUDED_KEYS = frozenset({"session", "reason", "decision", "evidence"})


@dataclass(frozen=True)
class OrbSpec:
    """Contract values fixed before any price is read."""

    market: str
    exchanges: tuple[str, ...]
    excluded_name: str
    min_open_price: float
    min_avg_volume: float
    min_atr: float
    lookback: int
    min_rel_volume: float
    top_names: int
    stop_atr_fraction: float
    risk_fraction: float
    max_leverage: float
    commission: float
    stale_minutes: int
    stale_adverse_atr: float
    ssr_decline: float
    tiers: tuple[tuple[str, float, str], ...]  # name, per-side bps, role
    annualization: int
    min_days: int
    min_years: int
    min_per_year: int
    min_trades: int
    min_selected: int
    min_positive_years: int
    block: int
    replicates: int
    seed: int
    lower_quantile: float
    excluded: tuple[str, ...] = ()  # sessions the contract leaves unscored as documented provider outages

    @property
    def first_index(self) -> int:
        """The first session with a full lookback: 14 true ranges need 15 earlier daily bars."""
        return self.lookback + 1

    def tier(self, role: str) -> tuple[str, float, str]:
        return next(t for t in self.tiers if t[2] == role)


@dataclass(frozen=True)
class SessionCalendar:
    """Development sessions with each one's regular close (New York time); early closes end at 13:00."""

    dates: tuple[date, ...]
    closes: tuple[str, ...]

    @property
    def sessions(self) -> list[str]:
        return [d.isoformat() for d in self.dates]

    def regular_minutes(self, index: int) -> int:
        return _minute(self.closes[index]) - _minute(REGULAR_OPEN)


def _minute(stamp: str) -> int:
    return int(stamp[:2]) * 60 + int(stamp[3:5])


def _stamp(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def contract_spec(contract: dict[str, Any]) -> OrbSpec:
    """Validate everything that must be fixed before data are read; fail closed on ambiguity."""
    if not re.fullmatch(r"[a-z0-9_]+", str(contract.get("campaign_id", ""))):
        raise ValueError("contract campaign_id must name its artifact directory")
    if contract.get("promotion_authority") is not False or contract.get("runtime_authority") is not False:
        raise ValueError("a development trial never carries promotion or runtime authority")
    development = daily_development_dates(contract)  # Holdout and holdout-year refusal precede any file read.
    excluded = excluded_sessions(contract, {day.isoformat() for day in development})
    if (contract.get("feed") != "sip" or contract.get("adjustment") != "raw"
            or contract.get("timeframes") != {"opening_range": "5Min", "path": "1Min", "filters": "1Day"}):
        raise ValueError("contract must use raw SIP bars: 5Min opening range, 1Min path, 1Day filters")
    section = swing._section
    universe, signal, trade, sizing = (section(contract, k) for k in ("universe", "signal", "trade", "sizing"))
    fills, shorts, survivorship = (section(contract, k) for k in ("fills", "shorts", "survivorship"))
    support, acceptance, bootstrap = (section(contract, k) for k in ("support_thresholds", "acceptance", "bootstrap"))
    if survivorship.get("probe_status") not in PROBE_STATUSES:
        raise ValueError("survivorship.probe_status must record the probe result before registration")
    market = section(contract, "symbols").get("market")
    if not isinstance(market, str) or not market:
        raise ValueError("symbols must name the market proxy")
    if list(section(contract, "versions")) != list(VERSIONS):
        raise ValueError("versions must be long_only (primary) then long_short (secondary)")
    exchanges, pattern = universe.get("exchanges"), universe.get("excluded_name_pattern")
    if not isinstance(exchanges, list) or not exchanges or not all(isinstance(x, str) for x in exchanges):
        raise ValueError("universe must list exchanges")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("universe must declare excluded_name_pattern")
    re.compile(pattern)
    if signal.get("opening_range_minutes") != OPENING_MINUTES or signal.get("doji") != "no_order":
        raise ValueError("signal must be the first 5-minute bar, with no order on a doji")
    if trade.get("exit") != "session_close" or trade.get("early_close_sessions") != "trade_to_the_early_close":
        raise ValueError("trade must exit at the session close, early closes included")
    if any(fills.get(key) != value for key, value in FILL_RULES.items()):
        raise ValueError(f"fills must name the fill models the runner implements:{FILL_RULES}")
    if shorts.get("require_current_shortable_and_easy_to_borrow") is not True:
        raise ValueError("shorts must require the shortable and easy-to-borrow flags")
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
    if primary != float(contract.get("primary_per_side_cost_bps", math.nan)) or stress <= primary or primary <= 0:
        raise ValueError("decision tier must equal primary_per_side_cost_bps and the stress tier must cost more")
    if not all(acceptance.get(k) is True for k in ("positive_lower_bound_vs_cash_at_primary", "positive_at_stress",
                                                   "long_only_positive_lower_bound_vs_exposure_matched_market_at_primary")):
        raise ValueError("acceptance criteria cannot be disabled")
    replicates = daily._int(bootstrap.get("replicates"), "bootstrap replicates")
    if replicates % 100:
        raise ValueError("bootstrap replicates must be a positive multiple of 100")
    seed = bootstrap.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    number, integer = daily._number, daily._int
    return OrbSpec(
        market=market, exchanges=tuple(exchanges), excluded_name=pattern,
        min_open_price=number(universe.get("minimum_open_price"), "minimum_open_price", low=0, high=1e6),
        min_avg_volume=number(universe.get("minimum_average_volume"), "minimum_average_volume", low=0, high=1e12),
        min_atr=number(universe.get("minimum_atr"), "minimum_atr", low=0, high=1e6),
        lookback=integer(universe.get("lookback_sessions"), "lookback_sessions", minimum=2),
        min_rel_volume=number(signal.get("minimum_relative_volume"), "minimum_relative_volume", low=0, high=1e6),
        top_names=integer(signal.get("top_names"), "top_names"),
        stop_atr_fraction=number(trade.get("stop_atr_fraction"), "stop_atr_fraction", low=0, high=10),
        risk_fraction=number(sizing.get("risk_fraction_of_slot"), "risk_fraction_of_slot", low=0, high=1),
        max_leverage=number(sizing.get("max_leverage"), "max_leverage", low=0, high=4),
        commission=number(fills.get("commission_per_share"), "commission_per_share", low=0, high=1, low_open=False),
        stale_minutes=integer(fills.get("stale_exit_minutes"), "stale_exit_minutes"),
        stale_adverse_atr=number(fills.get("stale_exit_adverse_atr"), "stale_exit_adverse_atr", low=0, high=100, low_open=False),
        ssr_decline=number(shorts.get("short_sale_restriction_decline"), "short_sale_restriction_decline", low=0, high=1),
        tiers=tuple(tiers), annualization=integer(contract.get("annualization"), "annualization"),
        min_days=integer(support.get("minimum_eligible_days"), "minimum_eligible_days"),
        min_years=integer(support.get("minimum_supported_years"), "minimum_supported_years"),
        min_per_year=integer(support.get("minimum_eligible_days_per_year"), "minimum_eligible_days_per_year"),
        min_trades=integer(support.get("minimum_filled_trades"), "minimum_filled_trades"),
        min_selected=integer(support.get("minimum_median_selected_names"), "minimum_median_selected_names"),
        min_positive_years=integer(acceptance.get("minimum_positive_years"), "minimum_positive_years", minimum=0),
        block=integer(bootstrap.get("block_days"), "bootstrap block_days"), replicates=replicates, seed=seed,
        lower_quantile=number(bootstrap.get("lower_quantile"), "lower_quantile", low=0, high=0.5), excluded=excluded)


def excluded_sessions(contract: dict[str, Any], development: set[str]) -> tuple[str, ...]:
    """The sessions the contract leaves unscored as documented provider outages, validated; any doubt refuses.

    Each entry names one development session, the reason ``provider_outage``, the owner's decision
    and the evidence counts, and nothing else. The list is the rule: a listed session is unscored
    even if its data turn out complete.
    """
    entries = contract.get("excluded_sessions")
    if not isinstance(entries, list):
        raise ValueError("contract must list excluded_sessions, empty if there are none")
    sessions = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != EXCLUDED_KEYS:
            raise ValueError(f"excluded_sessions entry must have exactly the keys {sorted(EXCLUDED_KEYS)}")
        session = entry["session"]
        if not isinstance(session, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", session) or session not in development:
            raise ValueError(f"excluded session is not a development session:{session}")
        if entry["reason"] not in EXCLUDED_REASONS:
            raise ValueError(f"excluded session reason must be one of {sorted(EXCLUDED_REASONS)}:{session}")
        if not isinstance(entry["decision"], str) or not entry["decision"].strip():
            raise ValueError(f"excluded session needs the owner's decision:{session}")
        if not isinstance(entry["evidence"], dict) or not entry["evidence"]:
            raise ValueError(f"excluded session needs its evidence counts:{session}")
        sessions.append(session)
    if len(set(sessions)) != len(sessions):
        raise ValueError("excluded_sessions lists a session twice")
    return tuple(sorted(sessions))


# --- pure core: calendar and session eligibility ---------------------------------------------------

def session_calendar(contract: dict[str, Any]) -> SessionCalendar:
    """The exchange calendar's sessions and closes; the rule-based fallback calendar is refused."""
    base = intraday.session_calendar(contract)
    return SessionCalendar(dates=base.dates, closes=tuple(_stamp(_minute(stamps[-1]) + intraday.BAR_MINUTES) for stamps in base.stamps))


def session_statuses(calendar: SessionCalendar, market_bars: Counter[str], spec: OrbSpec) -> list[str]:
    """One status per session from the market proxy's bar stamps alone.

    A session is eligible when it has a full lookback behind it and the market proxy has a
    one-minute bar for every minute of its regular hours. The proxy trades every minute, so a
    gap in it means the provider's record of that session is incomplete and no name is traded.
    A session the contract lists under ``excluded_sessions`` (a documented provider outage that
    the market proxy does not show) is unscored the same way, whatever its market record.
    """
    if set(spec.excluded) - set(calendar.sessions):
        raise ValueError(f"excluded session is not on the exchange calendar:{sorted(set(spec.excluded) - set(calendar.sessions))}")
    statuses = []
    for index, session in enumerate(calendar.sessions):
        if index < spec.first_index:
            statuses.append("insufficient_history")
        elif session in spec.excluded:
            statuses.append(EXCLUDED)
        elif market_bars.get(session, 0) != calendar.regular_minutes(index):
            statuses.append("market_incomplete")
        else:
            statuses.append(ELIGIBLE)
    return statuses


def support_summary(spec: OrbSpec, calendar: SessionCalendar, statuses: list[str], selected: Counter[str],
                    with_minutes: Counter[str]) -> dict[str, Any]:
    """Eligible sessions by year and names selected per session; nothing here depends on a price."""
    eligible = [s for s, status in zip(calendar.sessions, statuses) if status == ELIGIBLE]
    per_year = Counter(s[:4] for s in eligible)
    supported = sum(n >= spec.min_per_year for n in per_year.values())
    median = float(np.median([selected.get(s, 0) for s in eligible])) if eligible else 0.0
    reasons = [reason for reason, failed in (("eligible_days_below_minimum", len(eligible) < spec.min_days),
                                             ("supported_years_below_minimum", supported < spec.min_years),
                                             ("selected_names_below_minimum", median < spec.min_selected)) if failed]
    return dict(eligible_days=len(eligible), eligible_days_by_year=dict(sorted(per_year.items())), supported_years=supported,
                median_selected_names=median, selected_name_days=sum(selected.get(s, 0) for s in eligible),
                selected_name_days_with_minute_bars=sum(with_minutes.get(s, 0) for s in eligible),
                skipped_sessions=dict(sorted(Counter(s for s in statuses if s != ELIGIBLE).items())),
                minimum_required=dict(eligible_days=spec.min_days, supported_years=spec.min_years,
                                      eligible_days_per_year=spec.min_per_year, median_selected_names=spec.min_selected),
                status="blocked_sampling_support" if reasons else SUPPORT_OK, blocking_reasons=reasons)


# --- pure core: daily filters ----------------------------------------------------------------------

def split_events(raw: pd.DataFrame, adjusted: pd.DataFrame) -> list[tuple[str, str]]:
    """(symbol, session) of each first post-split session: where the raw/split-adjusted close ratio moves."""
    merged = raw[["symbol", "session", "close"]].merge(adjusted[["symbol", "session", "close"]], on=["symbol", "session"],
                                                      suffixes=("", "_adjusted")).sort_values(["symbol", "session"])
    ratio = merged.close / merged.close_adjusted
    change = ratio / ratio.groupby(merged.symbol, observed=True).shift(1)
    moved = merged[(change - 1).abs() > SPLIT_TOLERANCE]
    return sorted(zip(moved.symbol.astype(str), moved.session.astype(str)))


def daily_features(bars: pd.DataFrame, sessions: list[str], splits: Iterable[tuple[str, str]], spec: OrbSpec) -> pd.DataFrame:
    """Rows (session index, symbol) that pass the price, volume and range filters, from raw daily bars.

    For session t the filters use the daily bars of sessions t-15 to t-1 and the open of t only:
    the open of t above the minimum price; the mean volume of t-14..t-1 at least the minimum;
    the mean true range of t-14..t-1 (each needs the close before it) above the minimum. A symbol
    with a missing bar anywhere in that window, or a split effective on any session from t-14 to
    t, is not eligible: raw prices and volumes on the two sides of a split do not compare.
    """
    index = {session: i for i, session in enumerate(sessions)}
    split_at: dict[str, list[int]] = {}
    for symbol, session in splits:
        if session in index:
            split_at.setdefault(symbol, []).append(index[session])
    n, window = len(sessions), spec.lookback
    frames = []
    for symbol, group in bars.groupby("symbol", observed=True, sort=True):
        rows = group.session.map(index)
        group = group[rows.notna()]
        values = np.full((5, n), np.nan)
        values[:, rows.dropna().to_numpy(dtype=int)] = group[PRICE_COLUMNS].to_numpy(dtype=float).T
        opens, highs, lows, closes, volumes = values
        before = np.concatenate([[np.nan], closes[:-1]])
        true_range = np.maximum(highs - lows, np.maximum(np.abs(highs - before), np.abs(lows - before)))
        atr = pd.Series(true_range).rolling(window).mean().shift(1).to_numpy()
        volume = pd.Series(volumes).rolling(window).mean().shift(1).to_numpy()
        split = np.zeros(n, dtype=bool)
        for at in split_at.get(str(symbol), []):
            split[at:at + window + 1] = True
        with np.errstate(invalid="ignore"):
            keep = (opens > spec.min_open_price) & (volume >= spec.min_avg_volume) & (atr > spec.min_atr) & ~split
        keep[:spec.first_index] = False
        where = np.flatnonzero(keep)
        if not len(where):
            continue
        two_before = np.concatenate([[np.nan, np.nan], closes[:-2]])
        yesterday_low = np.concatenate([[np.nan], lows[:-1]])
        frames.append(pd.DataFrame(dict(
            idx=where.astype(np.int32), symbol=np.full(len(where), sys.intern(str(symbol)), dtype=object), open=opens[where],
            atr=atr[where], avg_volume=volume[where], prior_close=before[where],
            ssr_carry=yesterday_low[where] <= (1 - spec.ssr_decline) * two_before[where])))
    columns = ["idx", "symbol", "open", "atr", "avg_volume", "prior_close", "ssr_carry"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


# --- pure core: relative volume and the daily selection --------------------------------------------

def _symbol_codes(symbols: pd.Series, codes: dict[str, int], *, grow: bool) -> np.ndarray:
    """An integer per row for the symbol, from a shared table; -1 for a symbol the table lacks when it may not grow."""
    column = symbols.astype("category")
    lookup = np.empty(len(column.cat.categories), dtype=np.int64)
    for j, name in enumerate(column.cat.categories):
        lookup[j] = codes.setdefault(str(name), len(codes)) if grow else codes.get(str(name), -1)
    return np.asarray(lookup[column.cat.codes.to_numpy()], dtype=np.int64)


def select_from_parts(features: pd.DataFrame, parts: Callable[[], Iterable[pd.DataFrame]], sessions: list[str],
                      not_shortable: frozenset[str], spec: OrbSpec) -> pd.DataFrame:
    """The names ordered each session: the top ``top_names`` by relative volume among filtered names.

    ``parts`` yields the opening five-minute bars in pieces (one month each at full scale) and is
    called twice, so memory holds one piece plus three numbers per bar, never the whole table.

    Relative volume for session t is the volume of the first five-minute bar of t divided by the
    mean volume of that bar over sessions t-14..t-1, all 14 of which must have the bar; until
    they do, the name is unranked. It uses nothing after 09:35 of t. Names below
    ``min_rel_volume`` are dropped; ties break by symbol. The direction is the sign of the bar's
    close minus its open; zero (a doji) keeps its rank and places no order. ``short_ok`` is the
    shortability rule for the long-short version.
    """
    index = {session: i for i, session in enumerate(sessions)}
    window = spec.lookback
    stride = len(sessions) + window + 1  # wider than any session difference, so equal key gaps mean the same symbol
    codes: dict[str, int] = {}

    def keyed(part: pd.DataFrame, *, grow: bool) -> tuple[pd.DataFrame, np.ndarray]:
        where = part.session.map(index).to_numpy(dtype=float)
        code = _symbol_codes(part.symbol, codes, grow=grow)
        keep = np.isfinite(where) & (code >= 0)
        return part[keep], code[keep] * stride + where[keep].astype(np.int64)

    key_parts, volume_parts = [], []
    for part in parts():
        kept, key = keyed(part, grow=True)
        key_parts.append(key)
        volume_parts.append(kept.volume.to_numpy(dtype=float))
    if features.empty or not key_parts or not sum(len(k) for k in key_parts):
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    keys, volumes = np.concatenate(key_parts), np.concatenate(volume_parts)
    del key_parts, volume_parts
    order = np.argsort(keys, kind="stable")
    keys, volumes = keys[order], volumes[order]
    del order
    if (np.diff(keys) <= 0).any():
        raise ValueError("duplicate opening bars")
    # Sorted by symbol then session: the 14 rows before a row are its 14 prior sessions only if the row 14 back is 14 sessions back.
    running = np.concatenate([[0.0], np.cumsum(volumes)])
    relative = np.full(len(keys), np.nan)
    if len(keys) > window:
        complete = keys[window:] - keys[:-window] == window  # same symbol, exactly 14 sessions earlier
        mean = (running[window:-1] - running[:-window - 1]) / window
        with np.errstate(divide="ignore", invalid="ignore"):
            relative[window:] = np.where(complete & (mean > 0), volumes[window:] / mean, np.nan)
    del running, volumes
    wanted = features.symbol.map(codes).to_numpy(dtype=float)
    feature_key = np.where(np.isfinite(wanted), wanted, -1).astype(np.int64) * stride + features.idx.to_numpy(dtype=np.int64)
    at = np.minimum(np.searchsorted(keys, feature_key), len(keys) - 1)
    found = np.isfinite(wanted) & (keys[at] == feature_key)
    value = np.where(found, relative[at], np.nan)
    del relative, keys
    passing = np.isfinite(value) & (value >= spec.min_rel_volume)
    rows = features[passing].assign(rel_volume=value[passing], key=feature_key[passing])
    rows = rows.sort_values(["idx", "rel_volume", "symbol"], ascending=[True, False, True], kind="stable")
    rows = rows.groupby("idx", sort=False).head(spec.top_names).copy()
    rows["rank"] = rows.groupby("idx").cumcount() + 1
    chosen = np.sort(rows.key.to_numpy())
    bars = []
    for part in parts():  # second pass: only the selected names' bars are kept
        kept, key = keyed(part, grow=False)
        hit = np.isin(key, chosen)
        bars.append(pd.DataFrame(dict(key=key[hit], **{f"or_{c}": kept[c].to_numpy(dtype=float)[hit] for c in ("open", "high", "low", "close")})))
    rows = rows.merge(pd.concat(bars, ignore_index=True), on="key", how="left")
    rows["session"] = [sessions[i] for i in rows.idx]
    rows["direction"] = np.sign(rows.or_close - rows.or_open).astype(int)
    rows["trigger"] = np.where(rows.direction > 0, rows.or_high, np.where(rows.direction < 0, rows.or_low, np.nan))
    rows["stop_distance"] = spec.stop_atr_fraction * rows.atr
    rows["short_ok"] = ((rows.direction < 0) & ~rows.symbol.isin(not_shortable) & ~rows.ssr_carry.astype(bool)
                        & (rows.trigger > (1 - spec.ssr_decline) * rows.prior_close))
    return rows[SELECTION_COLUMNS].reset_index(drop=True)


def build_selection(features: pd.DataFrame, opening: pd.DataFrame, sessions: list[str], not_shortable: frozenset[str],
                    spec: OrbSpec) -> pd.DataFrame:
    """``select_from_parts`` on one in-memory table of opening bars."""
    return select_from_parts(features, lambda: [opening], sessions, not_shortable, spec)


# --- pure core: one order ---------------------------------------------------------------------------

def simulate_trade(times: list[str], opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, *,
                   direction: int, trigger: float, stop_distance: float, close_stamp: str, stale_minutes: int,
                   stale_adverse: float, model: str = DECIDING_MODEL) -> dict[str, Any] | None:
    """Entry and exit of one order from one-minute bars at or after 09:35; None if there is no fill.

    Written for a buy; a sell runs the same logic on negated prices. The order triggers on the
    first bar whose high reaches the trigger. What happens next depends on the fill model:

    * ``bar_executable`` (decides). Enter at the open of the NEXT bar, whatever it is; a trigger
      on the session's last bar is no fill. The stop loss sits ``stop_distance`` below that fill.
      A later bar that opens at or below the stop exits at its open; a bar whose low reaches the
      stop exits at the stop price. Every price used is one a one-minute record can show in
      order, so on a process with no edge the result averages zero before costs.
    * ``conservative`` (reported). The same entry; a bar whose low reaches the stop
      exits at the worse of the stop and the next bar's open.
    * ``paper_exact`` (reported, upper bound). Enter at exactly the trigger; the trigger bar
      cannot stop the trade; a later low at the stop exits at exactly the stop, gaps included.
    * ``lower_bound`` (reported). A gap through the trigger fills at the bar's open; a trigger
      bar whose low is ``stop_distance`` below the trigger is a stop-out; otherwise the entry is
      the worse of the trigger and the next open; stop exits are the worse of the stop and the
      next open. It keeps the bad half of every minute's move and scores a no-edge process
      well below zero.

    In every model an unstopped trade exits at the close of the last bar. If that bar is more
    than ``stale_minutes`` before the session close (a halt, or the record ends), the exit is
    moved against the position by ``stale_adverse``, but never below one cent.
    """
    if model not in FILL_MODELS:
        raise ValueError("unknown fill model")
    n = len(times)
    side = 1.0 if direction > 0 else -1.0
    xo, xc = side * opens, side * closes
    xh, xl = np.maximum(side * highs, side * lows), np.minimum(side * highs, side * lows)
    level = side * trigger
    reached = np.flatnonzero(xh >= level)
    if not len(reached):
        return None
    k = int(reached[0])

    def after(i: int) -> float:
        return float(xo[i + 1]) if i + 1 < n else float(xc[i])

    gap = bool(xo[k] >= level)
    entry_index, at_open = k, False
    exit_index, exit_value, reason = n - 1, float(xc[-1]), "session_end"
    if model == "lower_bound":
        entry, at_open = (float(xo[k]) if gap else level), gap
        stop = entry - stop_distance
        if xl[k] <= stop:
            exit_index, exit_value, reason = k, min(stop, after(k)), "stop_in_entry_bar"
        else:
            if not gap:  # alive at the end of the trigger bar: pay the worse price, and measure the stop from it
                entry = max(level, after(k))
                stop = entry - stop_distance
            for m in range(k + 1, n):
                if xo[m] <= stop:
                    exit_index, exit_value, reason = m, float(xo[m]), "stop_gap"
                    break
                if xl[m] <= stop:
                    exit_index, exit_value, reason = m, min(stop, after(m)), "stop"
                    break
    else:
        if model == "paper_exact":
            entry = level
        elif k + 1 >= n:
            return None
        else:
            entry, entry_index, at_open = float(xo[k + 1]), k + 1, True
        stop = entry - stop_distance
        for m in range(k + 1, n):
            if model != "paper_exact" and m > entry_index and xo[m] <= stop:
                exit_index, exit_value, reason = m, float(xo[m]), "stop_gap"
                break
            if xl[m] <= stop:
                exit_index, exit_value, reason = m, (min(stop, after(m)) if model == CONSERVATIVE_MODEL else stop), "stop"
                break
    exit_price = side * exit_value
    if reason == "session_end" and _minute(close_stamp) - 1 - _minute(times[-1]) > stale_minutes:
        # A long marked down by a full range could go below zero; the floor touches no other exit.
        exit_price, reason = max(side * (exit_value - stale_adverse), MIN_PRICE), "stale_last_print"
    return dict(entry_time=times[entry_index], entry_price=side * entry, exit_time=times[exit_index], exit_price=exit_price,
                exit_reason=reason, gap_entry=gap, entry_at_open=at_open)


def shares_per_equity(trigger: float, stop_distance: float, spec: OrbSpec) -> float:
    """Shares per dollar of equity: one slot is equity / top_names; a stop-out loses ``risk_fraction``
    of the slot, and the position is capped at ``max_leverage`` times the slot at the trigger price."""
    slot = 1.0 / spec.top_names
    return min(spec.risk_fraction * slot / stop_distance, spec.max_leverage * slot / trigger)


def build_trades(selection: pd.DataFrame, minute: pd.DataFrame, market: pd.DataFrame, calendar: SessionCalendar,
                 eligible: set[str], spec: OrbSpec, *, model: str = DECIDING_MODEL) -> pd.DataFrame:
    """One row per selected name on each eligible session, filled or not, with its accounting under one fill model.

    ``weight`` is the entry value as a fraction of equity. ``gross`` is the price move on the
    shares held; ``commission`` covers both sides; ``turnover`` is entry value plus exit value, on
    which each tier charges its per-side cost. ``matched_market`` is the market proxy's move from
    the entry (its open of that minute for a fill at a bar's open, else its close) to its close
    of the exit minute, on the same signed weight. ``cap_scale`` is how many times larger the
    position would be at the leverage cap.
    """
    closes = dict(zip(calendar.sessions, calendar.closes))
    market_close = {(s, t): float(c) for s, t, c in zip(market.session, market.time, market.close)}
    market_open = {(s, t): float(o) for s, t, o in zip(market.session, market.time, market.open)}
    paths = {key: group for key, group in minute.groupby(["session", "symbol"], observed=True, sort=False)}
    rows = []
    for order in selection.itertuples(index=False):
        if order.session not in eligible:
            continue
        row = dict(session=order.session, year=order.session[:4], symbol=order.symbol, rank=int(order.rank),
                   rel_volume=float(order.rel_volume), direction=int(order.direction), short_ok=bool(order.short_ok))
        path = paths.get((order.session, order.symbol))
        if order.direction == 0:
            rows.append(dict(row, status="doji"))
            continue
        if path is not None:
            path = path[(path.time >= ORDER_TIME) & (path.time < closes[order.session])].sort_values("time")
        if path is None or path.empty:
            rows.append(dict(row, status="no_minute_bars"))
            continue
        times = path.time.tolist()
        trade = simulate_trade(times, *(path[c].to_numpy(dtype=float) for c in PRICE_COLUMNS[:4]), direction=order.direction,
                               trigger=float(order.trigger), stop_distance=float(order.stop_distance),
                               close_stamp=closes[order.session], stale_minutes=spec.stale_minutes,
                               stale_adverse=spec.stale_adverse_atr * float(order.atr), model=model)
        if trade is None:
            rows.append(dict(row, status="not_triggered"))
            continue
        shares = shares_per_equity(float(order.trigger), float(order.stop_distance), spec)
        entry, exit_price = trade["entry_price"], trade["exit_price"]
        weight = shares * entry
        reference = (market_open if trade["entry_at_open"] else market_close)[(order.session, trade["entry_time"])]
        market_move = market_close[(order.session, trade["exit_time"])] / reference - 1
        cap = spec.max_leverage / spec.top_names / float(order.trigger)
        rows.append(dict(row, status="filled", **trade, shares=shares, weight=weight, cap_scale=cap / shares,
                         gross=shares * order.direction * (exit_price - entry), commission=2 * shares * spec.commission,
                         turnover=shares * (entry + exit_price), matched_market=order.direction * weight * market_move,
                         r_multiple=order.direction * (exit_price - entry) / float(order.stop_distance)))
    columns = ["session", "year", "symbol", "rank", "rel_volume", "direction", "short_ok", "status", "entry_time", "entry_price",
               "exit_time", "exit_price", "exit_reason", "gap_entry", "entry_at_open", "shares", "weight", "cap_scale", "gross",
               "commission", "turnover", "matched_market", "r_multiple"]
    return pd.DataFrame(rows, columns=columns)


def validate_trades(trades: pd.DataFrame, calendar: SessionCalendar, spec: OrbSpec) -> None:
    filled = trades[trades.status == "filled"]
    if trades.duplicated(["session", "symbol"]).any() or (trades.groupby("session").size() > spec.top_names).any():
        raise ValueError("duplicate orders or more names than the daily limit")
    if filled.empty:
        return
    closes = dict(zip(calendar.sessions, calendar.closes))
    numeric = filled[["entry_price", "exit_price", "shares", "weight", "gross", "commission", "turnover", "matched_market"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (filled[["entry_price", "shares"]] <= 0).any(axis=None):
        raise ValueError("nonfinite trade values or nonpositive entry")
    if not ((filled.entry_time >= ORDER_TIME) & (filled.entry_time <= filled.exit_time)
            & (filled.exit_time < filled.session.map(closes))).all():
        raise ValueError("a trade outside 09:35 to the session close")
    if (filled.exit_price < MIN_PRICE).any() or (filled.groupby("session").weight.sum() > spec.max_leverage * 1.25).any():
        raise ValueError("nonpositive exit price or a session far above the leverage cap")


# --- pure core: daily returns and the decision -----------------------------------------------------

def version_trades(trades: pd.DataFrame, version: str) -> pd.DataFrame:
    """Filled trades a version takes: long-only takes the buys; long-short adds the sells that pass the short rule."""
    filled = trades[trades.status == "filled"]
    if version == "long_only":
        return filled[filled.direction > 0]
    if version == "long_short":
        return filled[(filled.direction > 0) | filled.short_ok.astype(bool)]
    if version == AS_PUBLISHED:
        return filled
    raise ValueError("unknown version")


def day_returns(part: pd.DataFrame, sessions: list[str], per_side_bps: float) -> tuple[np.ndarray, np.ndarray]:
    """(net return, matched market return) per eligible session as fractions of that morning's equity.

    Net is the price move less commission on both sides less ``per_side_bps`` of the value
    traded at entry and at exit. A session with no fill is an observation of zero.
    """
    net = part.gross - part.commission - per_side_bps / 10_000 * part.turnover
    by_session = pd.DataFrame(dict(session=part.session, net=net, matched=part.matched_market)).groupby("session").sum()
    aligned = by_session.reindex(sessions, fill_value=0.0)
    return aligned.net.to_numpy(dtype=float), aligned.matched.to_numpy(dtype=float)


def _bps(value: float) -> float | None:
    return daily._finite(value * 10_000)


def _lower_bound(values: np.ndarray, spec: OrbSpec) -> float | None:
    means = intraday.bootstrap_means(values, block=spec.block, replicates=spec.replicates, seed=spec.seed)
    return None if means is None else _bps(float(np.quantile(means, spec.lower_quantile)))


def equal_notional(trades: pd.DataFrame) -> pd.DataFrame:
    """The same fills with every position at the leverage cap: the '1 percent of equity' reading of the paper."""
    scaled = trades.copy()
    for column in ("shares", "weight", "gross", "commission", "turnover", "matched_market"):
        scaled[column] = scaled[column] * scaled.cap_scale
    return scaled


def _version_report(trades: pd.DataFrame, sessions: list[str], spec: OrbSpec, *, version: str, decides: bool) -> dict[str, Any]:
    part = version_trades(trades, version)
    years = np.array([s[:4] for s in sessions])
    tiers: dict[str, dict[str, Any]] = {}
    for name, per_side, role in spec.tiers:
        net, matched = day_returns(part, sessions, per_side)
        tiers[name] = dict(per_side_bps=per_side, role=role, net_per_day_bps=_bps(float(net.mean())),
                           net_annualized_percent=daily._finite(float(net.mean()) * spec.annualization * 100),
                           paired_vs_matched_market_bps=_bps(float((net - matched).mean())),
                           sharpe=daily._finite(daily.sharpe(net, spec.annualization)), max_drawdown=daily.max_drawdown(net))
    reasons = part.exit_reason.value_counts()
    base = dict(eligible_days=len(sessions), filled_trades=len(part), long_trades=int((part.direction > 0).sum()),
                short_trades=int((part.direction < 0).sum()), days_with_a_fill=int(part.session.nunique()), cost_tiers=tiers,
                exit_reasons={str(k): int(v) for k, v in sorted(reasons.items())}, gap_entries=int(part.gap_entry.astype(bool).sum()),
                hit_ratio_before_costs=daily._finite((part.gross > 0).mean()) if len(part) else None,
                average_r_before_costs=daily._finite(part.r_multiple.mean()) if len(part) else None,
                average_gross_leverage=daily._finite(part.weight.sum() / max(len(sessions), 1)),
                commission_per_day_bps=_bps(float(part.commission.sum() / max(len(sessions), 1))),
                turnover_per_day=daily._finite(part.turnover.sum() / max(len(sessions), 1)),
                benchmark="exposure_matched_market" if version == "long_only" else "cash")
    if not decides:
        return base
    per_year = Counter(years.tolist())
    supported_years = sum(n >= spec.min_per_year for n in per_year.values())
    support = len(sessions) >= spec.min_days and supported_years >= spec.min_years and len(part) >= spec.min_trades
    primary, stress = spec.tier("decision"), spec.tier(STRESS_ROLE)
    net, matched = day_returns(part, sessions, primary[1])
    stressed, _ = day_returns(part, sessions, stress[1])
    paired = net - matched
    by_year = {year: dict(eligible_days=int((years == year).sum()), net_bps=_bps(float(net[years == year].mean())),
                          paired_vs_matched_market_bps=_bps(float(paired[years == year].mean())),
                          filled_trades=int((part.year == year).sum())) for year in sorted(per_year)}
    positive_years = sum(1 for v in by_year.values() if v["net_bps"] is not None and v["net_bps"] > 0)
    base.update(supported_years=supported_years, by_year=by_year)
    bounds = dict(vs_cash_bps=_lower_bound(net, spec), vs_matched_market_bps=_lower_bound(paired, spec))
    if bounds["vs_cash_bps"] is None or bounds["vs_matched_market_bps"] is None:
        return dict(base, decision="inconclusive_budget_consumed", uncertainty_status="unavailable_fewer_than_two_blocks",
                    criteria={"adequate_support": False})
    criteria = {"adequate_support": bool(support),
                "positive_lower_bound_vs_cash_at_primary": bounds["vs_cash_bps"] > 0,
                "positive_at_stress": bool(stressed.mean() > 0),
                "consistent_years": positive_years >= spec.min_positive_years}
    if version == "long_only":
        criteria["positive_lower_bound_vs_exposure_matched_market_at_primary"] = bounds["vs_matched_market_bps"] > 0
    decision = ("eligible_for_untouched_test_planning" if all(criteria.values())
                else "retired" if support else "inconclusive_budget_consumed")
    return dict(base, decision=decision, criteria=criteria, positive_years=positive_years, lower_bounds=bounds,
                lower_quantile=spec.lower_quantile)


def evaluate(trades: dict[str, pd.DataFrame], sessions: list[str], spec: OrbSpec) -> dict[str, Any]:
    """Both versions under the deciding fill model; the ledger decision is the long-only decision.

    ``trades`` maps each fill model to its trade table. Only the deciding model's table enters
    a criterion; the other models, the every-short book and equal-notional sizing are reported only.
    """
    main = trades[DECIDING_MODEL]
    versions = {v: _version_report(main, sessions, spec, version=v, decides=True) for v in VERSIONS}
    others = {model: {v: _version_report(table, sessions, spec, version=v, decides=False) for v in VERSIONS}
              for model, table in trades.items() if model != DECIDING_MODEL}
    sized = equal_notional(main)
    return dict(decision=versions["long_only"]["decision"], secondary_decision=versions["long_short"]["decision"],
                secondary_decision_permits="nothing; reported as information only",
                decision_tier=spec.tier("decision")[0], deciding_fill_model=DECIDING_MODEL, versions=versions,
                reported_only={"fill_models": others,
                               AS_PUBLISHED: _version_report(main, sessions, spec, version=AS_PUBLISHED, decides=False),
                               EQUAL_NOTIONAL: {v: _version_report(sized, sessions, spec, version=v, decides=False) for v in VERSIONS},
                               "order_statuses": {str(k): int(v) for k, v in sorted(main.status.value_counts().items())},
                               "shorts_blocked_by_the_short_rule": int(((main.direction < 0) & ~main.short_ok.astype(bool)).sum())})


# --- acquisition and probe -------------------------------------------------------------------------

Clients = swing.Clients


def fetch_frame(data: Any, symbols: list[str], start: datetime, end: datetime, *, unit: str,
                adjustment: str = "raw") -> tuple[pd.DataFrame, dict[str, str]]:
    """SIP bars for ``symbols`` as one table stamped in New York time, and the symbols whose request failed.

    ``unit`` is ``day``, ``opening`` (5-minute) or ``minute``. One request for the batch; a
    transient error is retried; a batch that still fails is retried symbol by symbol and each
    failure is recorded, never hidden. Every request is followed by a pause, because the running
    bot shares the rate limit.
    """
    from alpaca.common.exceptions import APIError
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    timeframe = {"day": TimeFrame.Day, "opening": TimeFrame(OPENING_MINUTES, TimeFrameUnit.Minute), "minute": TimeFrame.Minute}[unit]
    errors = (APIError, OSError, ValueError)

    def request(batch: list[str]) -> dict[str, list[Any]]:
        query = StockBarsRequest(symbol_or_symbols=batch, timeframe=timeframe, start=start, end=end, feed=DataFeed.SIP,
                                 adjustment=Adjustment(adjustment))
        try:
            found = dict(getattr(data.get_stock_bars(query), "data", {}))
        except errors:
            sleep(REQUEST_PAUSE_SECONDS)
            raise
        # The client pages through large answers without pausing; wait once for each page it fetched.
        sleep(REQUEST_PAUSE_SECONDS * (1 + sum(len(bars) for bars in found.values()) // PAGE_ROWS))
        return found

    failed: dict[str, str] = {}
    try:
        groups = [swing._with_retries(partial(request, symbols), errors)]
    except errors as exc:
        LOGGER.warning("OPENING_RANGE_BATCH_RETRY", extra={"first_symbol": symbols[0], "error": type(exc).__name__})
        groups = []
        for symbol in symbols:
            try:
                groups.append(swing._with_retries(partial(request, [symbol]), errors, attempts=2))
            except errors as single:  # recorded in the manifest, never hidden
                failed[symbol] = type(single).__name__
    columns: dict[str, list[Any]] = {name: [] for name in MINUTE_COLUMNS}
    for group in groups:
        for symbol, bars in group.items():
            name = sys.intern(str(symbol))
            for bar in bars:
                stamp = bar.timestamp.astimezone(NEW_YORK)
                columns["session"].append(sys.intern(stamp.date().isoformat()))
                columns["symbol"].append(name)
                columns["time"].append(sys.intern(stamp.strftime("%H:%M")))
                for column in PRICE_COLUMNS:
                    columns[column].append(float(getattr(bar, column)))
    text = {name: object for name in ("session", "symbol", "time")}  # an empty answer must still compare as text
    return pd.DataFrame(columns, columns=MINUTE_COLUMNS).astype(text), failed


def _at(session: str, stamp: str, *, seconds_before: int = 0) -> datetime:
    moment = datetime.combine(date.fromisoformat(session), time(int(stamp[:2]), int(stamp[3:5])), NEW_YORK)
    return moment - timedelta(seconds=seconds_before)


def _write_table(root: Path, path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """A gzip CSV with a fixed header time, so the same rows always hash the same."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as stream:
        stream.write(frame.to_csv(index=False).encode())
    os.replace(temporary, path)
    return dict(path=str(path.relative_to(root)), sha256=digest(path), rows=len(frame))


def regular_market_bars(frame: pd.DataFrame, calendar: SessionCalendar) -> pd.DataFrame:
    """The market proxy's bars inside each session's regular hours; a regular-hours bar off the calendar is refused."""
    closes = dict(zip(calendar.sessions, calendar.closes))
    close = frame.session.map(closes)
    stray = close.isna() & (frame.time >= REGULAR_OPEN) & (frame.time < "16:00")
    if stray.any():
        raise ValueError(f"regular-hours bar on a date the exchange calendar lacks:{frame.session[stray].iloc[0]}")
    kept = frame[close.notna() & (frame.time >= REGULAR_OPEN) & (frame.time < close.fillna(""))]
    if kept.duplicated(["session", "time"]).any():
        raise ValueError("duplicate market bars")
    return kept.sort_values(["session", "time"], ignore_index=True)[MARKET_COLUMNS]


def acquire(root: Path, contract: dict[str, Any], spec: OrbSpec, canonical: str, *,
            client_factory: Callable[[], Clients]) -> dict[str, Any]:
    """Freeze every input in five steps; refuse an existing acquisition, a degraded download and any row outside development.

    1. The asset list and monthly raw bars of every candidate, to shortlist the symbols that could
       ever pass the volume and price filters.
    2. Daily raw bars of the shortlist (and split-adjusted closes, kept only as split dates).
    3. The opening five-minute bar of each name that passes the daily filters that session or on
       one of the next 14 sessions (the later sessions need it as volume history).
    4. The selection: the top names by relative volume each session.
    5. One-minute bars from 09:35 to the close for the selected names, and for the market proxy.
    """
    manifest_path = root / contract["acquisition_manifest"]
    directory = manifest_path.parent
    if manifest_path.exists():
        raise ValueError("acquisition already exists and is immutable")
    calendar = session_calendar(contract)
    if set(spec.excluded) - set(calendar.sessions):
        raise ValueError(f"excluded session is not on the exchange calendar:{sorted(set(spec.excluded) - set(calendar.sessions))}")
    sessions, clients = calendar.sessions, client_factory()
    start, end = contract["development_start"], contract["development_end"]
    sources: dict[str, dict[str, Any]] = {}
    failed: dict[str, dict[str, str]] = {}

    assets = swing.list_assets(clients.trading)
    candidates, exclusions = swing.candidate_symbols(assets, spec)  # type: ignore[arg-type]
    monthly_bars, failed["monthly"] = swing.fetch_bars(clients.data, candidates, start, end, monthly=True)
    if len(failed["monthly"]) > MAX_FAILED_FETCH_FRACTION * len(candidates):
        raise ValueError(f"too many monthly fetch failures:{len(failed['monthly'])}:candidates={len(candidates)}")
    monthly = pd.DataFrame([dict(symbol=s, month=bar["session"][:7], high=bar["high"], volume=bar["volume"])
                            for s in sorted(monthly_bars) for bar in monthly_bars[s]], columns=["symbol", "month", "high", "volume"])
    swing._check_window((f"{month}-01" for month in monthly.month.unique()), contract, "monthly")
    # A 14-session window with a mean of X shares a day lies in at most two months, so one of them has at least 7X.
    busy = set(monthly.symbol[monthly.volume >= spec.lookback * spec.min_avg_volume / 2])
    shortlist = sorted(busy & set(monthly.symbol[monthly.high > spec.min_open_price]))
    if not shortlist:
        raise ValueError("no candidate could pass the filters")

    features, splits = [], []
    failed["daily"] = {}
    for part, offset in enumerate(range(0, len(shortlist), DAILY_CHUNK)):
        batch = shortlist[offset:offset + DAILY_CHUNK]
        raw, lost = fetch_frame(clients.data, batch, _at(start, "00:00"), _at(end, "23:59"), unit="day")
        adjusted, lost_adjusted = fetch_frame(clients.data, batch, _at(start, "00:00"), _at(end, "23:59"), unit="day", adjustment="split")
        failed["daily"].update(lost | lost_adjusted)
        raw = raw[~raw.symbol.isin(failed["daily"])].sort_values(["symbol", "session"], ignore_index=True)[DAILY_COLUMNS]
        if raw.duplicated(["symbol", "session"]).any():
            raise ValueError("duplicate daily bars")
        daily.check_development_sessions({"daily": raw.session.unique().tolist()}, contract)
        adjusted = adjusted[~adjusted.symbol.isin(failed["daily"])].sort_values(["symbol", "session"], ignore_index=True)
        if not raw[["symbol", "session"]].equals(adjusted[["symbol", "session"]]):
            # A split on a session the adjusted series lacks would go unseen and the split guard would not fire.
            raise ValueError(f"split-adjusted daily bars do not cover the raw bars:part={part}")
        events = split_events(raw, adjusted)
        splits += events
        sources[f"daily/part_{part:03d}"] = _write_table(root, directory / "daily" / f"part_{part:03d}.csv.gz", raw)
        features.append(daily_features(raw[daily.valid_daily(raw)], sessions, events, spec))
    if len(failed["daily"]) > MAX_FAILED_FETCH_FRACTION * len(shortlist):
        raise ValueError(f"too many daily fetch failures:{len(failed['daily'])}:shortlist={len(shortlist)}")
    feature = pd.concat(features, ignore_index=True)
    if feature.empty:
        raise ValueError("no name passes the daily filters on any session")

    names = sorted(feature.symbol.unique())
    column = {symbol: j for j, symbol in enumerate(names)}
    wanted = np.zeros((len(sessions), len(names)), dtype=bool)
    wanted[feature.idx.to_numpy(dtype=int), feature.symbol.map(column).to_numpy(dtype=int)] = True
    passing = wanted.copy()
    for ahead in range(1, spec.lookback + 1):  # the next 14 sessions need this session's opening volume as history
        wanted[:-ahead] |= passing[ahead:]
    failed["opening"], failed["minute"] = {}, {}
    asked, opening_bars = 0, 0
    coverage: dict[str, list[int]] = {}
    thin: list[str] = []  # sessions whose opening bars cover too few names; judged once the market proxy's record is known
    month_rows: list[pd.DataFrame] = []

    def flush(kind: str, month: str, columns: list[str]) -> int:
        if not month_rows:
            return 0
        table = pd.concat(month_rows, ignore_index=True)[columns]
        sources[f"{kind}/{month}"] = _write_table(root, directory / kind / f"{month}.csv.gz", table)
        month_rows.clear()
        return len(table)

    for i, session in enumerate(sessions):
        if i and session[:7] != sessions[i - 1][:7]:
            opening_bars += flush("opening", sessions[i - 1][:7], DAILY_COLUMNS)
        needed = [names[j] for j in np.flatnonzero(wanted[i])]
        returned, lost_here = 0, 0
        for offset in range(0, len(needed), OPENING_CHUNK):
            batch = needed[offset:offset + OPENING_CHUNK]
            frame, lost = fetch_frame(clients.data, batch, _at(session, REGULAR_OPEN), _at(session, ORDER_TIME, seconds_before=1), unit="opening")
            failed["opening"].update({f"{session}:{symbol}": error for symbol, error in lost.items()})
            frame = frame[(frame.session == session) & (frame.time == REGULAR_OPEN)]
            returned, lost_here = returned + int(frame.symbol.nunique()), lost_here + len(lost)
            month_rows.append(frame)
        asked += len(needed)
        if needed:
            coverage[session] = [len(needed), returned, lost_here]
            # Every name asked for has a daily bar that session, so a missing opening bar means no trade in five minutes.
            # One outage must not pass because the other 2,513 sessions were fine: each session is checked on its own.
            if lost_here > MAX_FAILED_FETCH_FRACTION * len(needed):
                raise ValueError(f"too many opening-bar fetch failures:{session}:failed={lost_here}:asked={len(needed)}")
            # A shortfall is allowed one name, and is refused below only if the session will be scored: on a market-wide
            # halt just after the open (2020-03-16) few names trade in the first five minutes and the session is unscored.
            if len(needed) - returned > max(1.0, (1 - MIN_OPENING_COVERAGE) * len(needed)):
                thin.append(session)
    opening_bars += flush("opening", sessions[-1][:7], DAILY_COLUMNS)

    not_shortable = frozenset(row["symbol"] for row in assets if not (row["shortable"] and row["easy_to_borrow"]))
    selection = select_from_parts(feature, lambda: _opening_parts(root, sources), sessions, not_shortable, spec)
    if selection.empty:
        raise ValueError("no name was selected on any session")
    orders = selection[selection.direction != 0]
    without_bars, market_parts = 0, 0
    incomplete: set[str] = set()
    by_month: dict[str, list[int]] = {}
    for i, session in enumerate(sessions):
        by_month.setdefault(session[:7], []).append(i)
    for month, members in by_month.items():
        for i in members:
            session = sessions[i]
            batch = sorted(orders.symbol[orders.session == session])
            if not batch:
                continue
            frame, lost = fetch_frame(clients.data, batch, _at(session, ORDER_TIME), _at(session, calendar.closes[i], seconds_before=1), unit="minute")
            failed["minute"].update({f"{session}:{symbol}": error for symbol, error in lost.items()})
            frame = frame[(frame.session == session) & (frame.time >= ORDER_TIME) & (frame.time < calendar.closes[i])]
            without_bars += len(set(batch) - set(frame.symbol))
            month_rows.append(frame)
        flush("minute", month, MINUTE_COLUMNS)
        first, last = members[0], members[-1]
        frame, lost = fetch_frame(clients.data, [spec.market], _at(sessions[first], REGULAR_OPEN), _at(sessions[last], calendar.closes[last]), unit="minute")
        if lost:
            raise ValueError(f"market proxy fetch failed:{month}")
        market = regular_market_bars(frame[frame.symbol == spec.market], calendar)
        counts = Counter(market.session)
        incomplete.update(sessions[i] for i in members if counts.get(sessions[i], 0) != calendar.regular_minutes(i))
        sources[f"market/{month}"] = _write_table(root, directory / "market" / f"{month}.csv.gz", market)
        market_parts += len(market)
    for session in thin:
        # A session the contract excludes is unscored like one with an incomplete SPY record; every other one refuses.
        if session not in incomplete and session not in spec.excluded:
            asked_here, returned, _ = coverage[session]
            raise ValueError(f"opening bars cover too few names:{session}:returned={returned}:asked={asked_here}")
    excluded = {session: dict(reason="contract_excluded_session", opening_asked_returned_failed=coverage.get(session),
                              opening_shortfall=session in thin, market_record_complete=session not in incomplete,
                              note=("opening bars below the coverage floor; unscored because the contract lists it" if session in thin
                                    else "opening bars met the coverage floor; unscored anyway because the contract lists it"))
                for session in spec.excluded}
    # ``without_bars`` already counts a name whose request failed: it has no bars either.
    if without_bars > MAX_FAILED_FETCH_FRACTION * len(orders):
        raise ValueError(f"too many selected names without minute bars:{without_bars}:orders={len(orders)}")
    if len(incomplete) > MAX_INCOMPLETE_SESSION_FRACTION * len(sessions):
        raise ValueError(f"degraded download:incomplete_market_sessions={len(incomplete)}:sessions={len(sessions)}")

    as_json = lambda payload: json.dumps(payload, indent=1, sort_keys=True) + "\n"  # noqa: E731
    frozen = dict(
        assets=swing._write_hashed(root, directory / "assets.json", as_json(assets)),
        monthly=swing._write_hashed(root, directory / "monthly_raw.csv", monthly.sort_values(["symbol", "month"]).to_csv(index=False)),
        splits=swing._write_hashed(root, directory / "splits.json", as_json(sorted(splits))),
        selection=swing._write_hashed(root, directory / "selection.csv", selection.to_csv(index=False)))
    counts_out = dict(assets=len(assets), inactive_assets=sum(row["status"] == "inactive" for row in assets),
                      candidates=len(candidates), **exclusions, shortlist=len(shortlist), split_events=len(splits),
                      names_ever_passing_daily_filters=len(names), name_days_passing_daily_filters=len(feature),
                      opening_bars_requested=asked, opening_bars=opening_bars, selected_name_days=len(selection),
                      orders=len(orders), orders_without_minute_bars=without_bars, market_bars=market_parts,
                      incomplete_market_sessions=len(incomplete), opening_shortfall_sessions_unscored=len(thin),
                      excluded_sessions=len(excluded),
                      selected_inactive_symbols=len({row["symbol"] for row in assets if row["status"] == "inactive"} & set(selection.symbol)))
    manifest = dict(campaign_id=contract["campaign_id"], contract_hash=canonical, feed="sip", adjustment="raw",
                    request=dict(start=start, end=end), fetched_at=datetime.now(NEW_YORK).isoformat(), sources=sources,
                    frozen=frozen, counts=counts_out, failed=failed, opening_coverage=coverage, opening_shortfall_sessions=thin,
                    excluded_sessions=excluded, prices_read_for_analysis=False,
                    quality_passed=True)
    write_json(manifest_path, manifest)
    return dict(status="acquisition_written", manifest=str(manifest_path), counts=counts_out, trial_claimed=False)


def probe(contract: dict[str, Any], *, client_factory: Callable[[], Clients]) -> dict[str, Any]:
    """What the provider serves for a few symbols and sessions. Bar counts and stamps only; no price is kept."""
    plan = swing._section(contract, "data_probe")
    calendar = session_calendar(contract)
    closes = dict(zip(calendar.sessions, calendar.closes))
    clients = client_factory()
    statuses: dict[str, list[str]] = {}
    for row in swing.list_assets(clients.trading):
        statuses.setdefault(row["symbol"], []).append(row["status"])

    def look(symbol: str, session: str) -> dict[str, Any]:
        if session not in closes:
            raise ValueError(f"probe session is not a development session:{session}")
        close = closes[session]
        minutes, lost = fetch_frame(clients.data, [symbol], _at(session, REGULAR_OPEN), _at(session, close, seconds_before=1), unit="minute")
        opening, _ = fetch_frame(clients.data, [symbol], _at(session, REGULAR_OPEN), _at(session, ORDER_TIME, seconds_before=1), unit="opening")
        days, _ = fetch_frame(clients.data, [symbol], _at(session, "00:00"), _at(session, "23:59"), unit="day")
        minutes = minutes[(minutes.time >= REGULAR_OPEN) & (minutes.time < close)]
        first_five = minutes[minutes.time < ORDER_TIME]
        bar = opening[opening.time == REGULAR_OPEN]
        agrees = None
        if len(bar) == 1 and len(first_five):
            agrees = bool(abs(first_five.volume.sum() - bar.volume.iloc[0]) < 0.5 and first_five.high.max() == bar.high.iloc[0]
                          and first_five.low.min() == bar.low.iloc[0])
        return dict(session=session, asset_statuses=sorted(statuses.get(symbol, [])), regular_minutes=_minute(close) - _minute(REGULAR_OPEN),
                    minute_bars=len(minutes), first_minute=minutes.time.min() if len(minutes) else None,
                    last_minute=minutes.time.max() if len(minutes) else None, opening_bar=len(bar) == 1, daily_bar=len(days) == 1,
                    opening_bar_equals_its_minutes=agrees, request_error=lost.get(symbol))

    coverage = {f"{symbol}:{session}": look(symbol, session) for symbol in plan["coverage_symbols"] for session in plan["coverage_sessions"]}
    delisted = {row["symbol"]: look(row["symbol"], row["session"]) for row in plan["delisted"]}
    return dict(status="probe_complete", coverage=coverage, delisted=delisted, delisted_probed=len(delisted),
                delisted_in_asset_list=sum(bool(r["asset_statuses"]) for r in delisted.values()),
                delisted_with_minute_bars=sum(r["minute_bars"] > 0 for r in delisted.values()),
                delisted_with_opening_bar=sum(r["opening_bar"] for r in delisted.values()),
                prices_read_for_analysis=False, trial_claimed=False)


# --- reading ---------------------------------------------------------------------------------------

def _read(root: Path, source: dict[str, Any], columns: list[str], *, compact_sessions: bool = False) -> pd.DataFrame:
    """A hash-verified table. Symbols are read as a category so 18 million minute rows stay small.

    Only an empty field in a number column is missing: the tickers ``NA``, ``NAN`` and ``NULL``
    are text, not gaps.
    """
    path = root / source["path"]
    if digest(path) != source["sha256"]:
        raise ValueError(f"source changed before read:{source['path']}")
    types = {"session": "category" if compact_sessions else str, "time": str, "symbol": "category"}
    frame = pd.read_csv(path, usecols=columns, dtype={k: v for k, v in types.items() if k in columns}, keep_default_na=False,
                        na_values={c: [""] for c in columns if c not in types})
    if digest(path) != source["sha256"]:
        raise ValueError(f"source changed during read:{source['path']}")
    return frame[columns]


def _opening_parts(root: Path, sources: dict[str, dict[str, Any]], rejected: Counter[str] | None = None) -> Iterable[pd.DataFrame]:
    """The valid opening bars one month at a time; invalid bars are counted once if a counter is given."""
    for name, source in sorted(sources.items()):
        if name.startswith("opening/"):
            frame = _read(root, source, DAILY_COLUMNS, compact_sessions=True)
            valid = daily.valid_daily(frame)
            if rejected is not None:
                rejected["invalid_opening_bars"] += int((~valid).sum())
            yield frame[valid]


def _group(manifest: dict[str, Any], kind: str) -> list[tuple[str, dict[str, Any]]]:
    return sorted((name, source) for name, source in manifest["sources"].items() if name.startswith(f"{kind}/"))


def load_stamps(root: Path, manifest: dict[str, Any]) -> tuple[Counter[str], Counter[str], Counter[str], list[str]]:
    """Market bars per session, selected names per session, selected names with minute bars, and every session seen.

    Reads session, symbol and time columns only; no price column is read.
    """
    market: Counter[str] = Counter()
    seen: set[str] = set()
    for _, source in _group(manifest, "market"):
        frame = _read(root, source, ["session", "time"])
        if frame.duplicated().any():
            raise ValueError("duplicate market bars in source")
        market.update(frame.session)
    selection = _read(root, manifest["frozen"]["selection"], ["session", "symbol"])
    with_minutes: Counter[str] = Counter()
    for _, source in _group(manifest, "minute"):
        frame = _read(root, source, ["session", "symbol"]).drop_duplicates()
        with_minutes.update(frame.session)
        seen.update(frame.session.unique())
    for kind in ("daily", "opening"):
        for _, source in _group(manifest, kind):
            seen.update(_read(root, source, ["session"]).session.unique())
    return market, Counter(selection.session), with_minutes, sorted(seen | set(market) | set(selection.session))


def rebuild_selection(root: Path, manifest: dict[str, Any], sessions: list[str], spec: OrbSpec) -> tuple[pd.DataFrame, dict[str, int]]:
    """The selection recomputed from the frozen daily and opening bars; it must equal the frozen selection."""
    def frozen(name: str) -> Any:
        source = manifest["frozen"][name]
        if digest(root / source["path"]) != source["sha256"]:
            raise ValueError(f"source changed before read:{source['path']}")
        return json.loads((root / source["path"]).read_text())

    splits = [(str(symbol), str(session)) for symbol, session in frozen("splits")]
    not_shortable = frozenset(row["symbol"] for row in frozen("assets") if not (row["shortable"] and row["easy_to_borrow"]))
    rejected: Counter[str] = Counter()
    features = []
    for _, source in _group(manifest, "daily"):
        frame = _read(root, source, DAILY_COLUMNS)
        valid = daily.valid_daily(frame)
        rejected["invalid_daily_bars"] += int((~valid).sum())
        symbols = set(frame.symbol.astype(str))
        features.append(daily_features(frame[valid], sessions, [event for event in splits if event[0] in symbols], spec))
    counted: list[Counter[str] | None] = [rejected]  # the bars are read twice; invalid ones are counted on the first pass only

    def parts() -> Iterable[pd.DataFrame]:
        return _opening_parts(root, manifest["sources"], counted.pop() if counted else None)

    selection = select_from_parts(pd.concat(features, ignore_index=True), parts, sessions, not_shortable, spec)
    stored = _read(root, manifest["frozen"]["selection"], SELECTION_COLUMNS).astype({"symbol": str})
    same = (len(stored) == len(selection) and stored[["session", "symbol", "rank", "direction"]].astype(str).equals(
        selection[["session", "symbol", "rank", "direction"]].astype(str))
            and np.allclose(stored[["rel_volume", "atr"]].to_numpy(dtype=float), selection[["rel_volume", "atr"]].to_numpy(dtype=float)))
    if not same:
        raise ValueError("the frozen selection does not match the selection rebuilt from the frozen bars")
    return selection, {k: v for k, v in rejected.items() if v}


def load_month(root: Path, manifest: dict[str, Any], month: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """One month of minute bars for the selected names and the market proxy; invalid bars are dropped and counted."""
    market = _read(root, manifest["sources"][f"market/{month}"], MARKET_COLUMNS)
    source = manifest["sources"].get(f"minute/{month}")
    minute = _read(root, source, MINUTE_COLUMNS) if source else pd.DataFrame(columns=MINUTE_COLUMNS)
    if minute.empty:
        return minute, market, 0
    valid = daily.valid_daily(minute)
    return minute[valid], market, int((~valid).sum())


# --- orchestration ---------------------------------------------------------------------------------

def _section_list(contract: dict[str, Any]) -> list[str]:
    """Sessions the contract asks the support audit to show opening-bar coverage for."""
    return [str(session) for session in (contract.get("support_audit") or {}).get("report_opening_coverage_for") or []]


def run(root: Path, *, mode: str, contract_path: Path, client_factory: Callable[[], Clients] = swing._default_clients) -> dict[str, Any]:
    root = root.resolve()
    contract = json.loads((root / contract_path).resolve().read_text())
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    directory = root / "artifacts" / str(contract.get("campaign_id"))
    if mode == "probe":
        daily_development_dates(contract)
        result = probe(contract, client_factory=client_factory)
        write_json(directory / "data_probe.json", dict(result, contract_hash=canonical))
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
        return acquire(root, contract, spec, canonical, client_factory=client_factory)
    manifest = json.loads((root / contract["acquisition_manifest"]).read_text())
    if manifest.get("contract_hash") != canonical or manifest.get("quality_passed") is not True:
        raise ValueError("acquisition manifest does not belong to this registered contract")
    if sorted(manifest.get("excluded_sessions", {})) != list(spec.excluded):
        raise ValueError("acquisition manifest does not record the contract's excluded sessions")
    market_bars, selected, with_minutes, seen = load_stamps(root, manifest)
    daily.check_development_sessions({"sources": seen}, contract)
    calendar = session_calendar(contract)
    sessions = calendar.sessions
    statuses = session_statuses(calendar, market_bars, spec)
    blocked = dict(trial_claimed=False, holdout_evaluated=False, promotion_authority=False)
    audit = dict(support_summary(spec, calendar, statuses, selected, with_minutes), audit="session_symbol_and_bar_time_only",
                 development_sessions=len(sessions), acquisition_counts=manifest.get("counts", {}), contract_hash=canonical,
                 prices_read_for_analysis=False, strategy_outcomes_computed=False, **blocked)
    watched = _section_list(contract)
    audit["pre_claim_review"] = dict(  # counts to be looked at before the claim; none is a price or an outcome
        split_events=manifest.get("counts", {}).get("split_events"),
        opening_shortfall_sessions=manifest.get("opening_shortfall_sessions", []),
        excluded_sessions={session: dict(manifest["excluded_sessions"][session], session_status=statuses[sessions.index(session)])
                           for session in spec.excluded},
        opening_coverage_asked_returned_failed={session: manifest.get("opening_coverage", {}).get(session) for session in watched},
        session_status={session: statuses[sessions.index(session)] for session in watched if session in sessions})
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
    selection, rejected = rebuild_selection(root, manifest, sessions, spec)
    months = sorted({session[:7] for session in sessions})
    invalid_minutes = 0
    for month in months:  # every source is hash-checked and price-checked before the claim; no trade is computed here
        minute, market, invalid = load_month(root, manifest, month)
        invalid_minutes += invalid
        if not daily.valid_daily(market).all():
            raise ValueError(f"invalid market proxy bar:{month}")
    if invalid_minutes:
        rejected["invalid_minute_bars"] = invalid_minutes
    preflight = dict(blocked, status="preflight_passed", signature=signature, contract_hash=canonical,
                     eligible_days=audit["eligible_days"], selected_name_days=len(selection), rejection_counts=rejected,
                     code=code_hashes, provenance_sha256=hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest())
    write_json(directory / "preflight.json", preflight)
    if mode != "fit":
        return preflight
    claim_campaign_trial(state_path, hypothesis_id=contract["hypotheses"][0], evidence_signature=signature,
                         evaluation_start=contract["development_start"], evaluation_end=contract["development_end"],
                         quality_passed=True)
    # A crash after the claim intentionally prevents automatic re-evaluation.
    eligible = [s for s, status in zip(sessions, statuses) if status == ELIGIBLE]
    tables: dict[str, list[pd.DataFrame]] = {model: [] for model in FILL_MODELS}
    for month in months:
        minute, market, _ = load_month(root, manifest, month)
        chosen = selection[selection.session.str[:7] == month]
        for model in FILL_MODELS:
            tables[model].append(build_trades(chosen, minute, market, calendar, set(eligible), spec, model=model))
    trades, hashes = {}, {}
    for model in FILL_MODELS:
        trades[model] = pd.concat(tables[model], ignore_index=True)
        validate_trades(trades[model], calendar, spec)
        path = directory / f"trades_{model}.csv"
        atomic_write_text(path, trades[model].to_csv(index=False))
        hashes[path.name] = digest(path)
    report = evaluate(trades, eligible, spec)
    survivorship = contract["survivorship"]
    report.update(signature=signature, contract_hash=canonical, output_sha256=hashes,
                  rejection_counts=rejected, skipped_sessions=dict(sorted(Counter(s for s in statuses if s != ELIGIBLE).items())),
                  excluded_sessions=list(spec.excluded),
                  acquisition_counts=manifest.get("counts", {}), survivorship_probe_status=survivorship.get("probe_status"),
                  survivorship_label=PROBE_STATUSES[survivorship["probe_status"]], holdout_evaluated=False,
                  promotion_authority=False, runtime_authority=False, orders_sent=0, models_fitted=0,
                  evidence_type="historical_one_minute_bar_order_simulation_not_executable_fills")
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
    mode.add_argument("--probe", action="store_true", help="bar counts for a few symbols and sessions; allowed on a draft")
    mode.add_argument("--acquire", action="store_true", help="freeze the bars and the selection once; never claims")
    mode.add_argument("--support-audit", action="store_true", help="sessions, symbols and bar times only; never claims")
    mode.add_argument("--preflight", action="store_true", help="support audit, selection rebuild and price checks; never claims")
    mode.add_argument("--fit", action="store_true", help="claim the single trial once and evaluate")
    args = parser.parse_args(argv)
    selected = next(name for name in ("probe", "acquire", "support-audit", "fit", "preflight") if getattr(args, name.replace("-", "_")))
    result = run(args.repository_root, mode=selected, contract_path=args.contract)
    status = result.get("decision", result.get("status"))
    LOGGER.info("OPENING_RANGE_TRIAL_RESULT", extra={"mode": selected, "status": status})
    return 0 if status in ("probe_complete", "acquisition_written", SUPPORT_OK, "preflight_passed") or "decision" in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
