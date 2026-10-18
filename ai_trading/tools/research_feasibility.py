"""Screen a frozen hypothesis using development data before another study."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from functools import cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.utils.market_calendar import is_trading_day, session_info


NEW_YORK = ZoneInfo("America/New_York")
DECISION_DELAY = pd.Timedelta(minutes=5, seconds=2)
FEATURE_LOOKBACK = pd.Timedelta(days=10)
WARMUP_BARS = 200
MINIMUM_FOLD_OPPORTUNITIES = 250


@dataclass(frozen=True)
class IntradayGrid:
    """Fixed decision, entry, exit and flatten-cutoff grid shared with the trial."""

    exit_minutes: int
    cutoff_et: time | None = None
    lead_minutes: int = 0

    @classmethod
    def from_protocol(cls, protocol: dict[str, Any]) -> IntradayGrid | None:
        """Return the declared intraday grid, or None for a session-horizon protocol."""
        if "exit_horizon_minutes" not in protocol:
            return None
        minutes, lead = protocol["exit_horizon_minutes"], protocol.get("flatten_lead_minutes", 0)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (minutes, lead)) or minutes <= 0 or lead < 0:
            raise ValueError("exit_horizon_minutes must be a positive integer and flatten_lead_minutes nonnegative")
        cutoff = protocol.get("flatten_cutoff_et")
        if cutoff is not None and not (isinstance(cutoff, str) and len(cutoff) == 5):
            raise ValueError("flatten_cutoff_et must be HH:MM America/New_York")
        return cls(minutes, None if cutoff is None else time.fromisoformat(cutoff), lead)

    @cache
    def cutoff(self, day: date) -> pd.Timestamp:
        """Exits at or after this instant are excluded for the session."""
        limit = pd.Timestamp(session_info(day).end_utc) - pd.Timedelta(minutes=self.lead_minutes)
        if self.cutoff_et is not None:
            limit = min(limit, pd.Timestamp(datetime.combine(day, self.cutoff_et, NEW_YORK)))
        return limit.tz_convert("UTC")


def intraday_schedule(bar_starts: pd.DatetimeIndex, grid: IntradayGrid) -> list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Same-session, non-overlapping opportunities with full feature history, from timestamps only."""
    rows = []
    last_exit = None
    for i, stamp in enumerate(bar_starts):
        decision = stamp + DECISION_DELAY
        entry = decision.ceil("min")
        exit_at = entry + pd.Timedelta(minutes=grid.exit_minutes)
        if exit_at >= grid.cutoff(stamp.tz_convert(NEW_YORK).date()) or (last_exit is not None and entry < last_exit):
            continue
        if i + 1 - bar_starts.searchsorted(decision - FEATURE_LOOKBACK) < WARMUP_BARS:
            continue
        rows.append((i, stamp, decision, entry, exit_at))
        last_exit = exit_at
    return rows


def chronological_folds(days: list[str]) -> list[tuple[str, list[str]]]:
    """Six ordered session chunks: the first trains; each later chunk tests after a one-session embargo."""
    chunks = np.array_split(np.asarray(days), 6)
    return [(days[days.index(chunk[0]) - 1], [str(day) for day in chunk]) for chunk in chunks[1:]]


def intraday_support(protocol: dict[str, Any], indices: dict[str, pd.DatetimeIndex]) -> dict[str, Any]:
    """Count possible intraday-grid opportunities per symbol and fold using timestamps, never prices."""
    grid = IntradayGrid.from_protocol(protocol)
    if grid is None:
        raise ValueError("protocol has no intraday exit horizon")
    start, end = protocol["development_start"], protocol["development_end"]
    if not start <= end < protocol["holdout_start"] <= protocol["holdout_end"]:
        raise ValueError("invalid development interval or holdout overlap")
    dates = [day.date() for day in pd.date_range(start, end) if is_trading_day(day.date())]
    if not dates:
        raise ValueError("no development sessions")
    sessions = {day: session_info(day) for day in dates}
    minute_parts = [pd.date_range(s.start_utc, s.end_utc, freq="min", inclusive="left") for s in sessions.values()]
    expected = minute_parts[0].append(minute_parts[1:])
    invalid: list[str] = []
    by_symbol: dict[str, dict[str, Any]] = {}
    opportunities: list[tuple[str, str, pd.Timestamp]] = []
    for symbol in protocol["symbols"]:
        if symbol not in indices:
            invalid.append(f"symbol_missing:{symbol}")
            continue
        index = indices[symbol]
        if index.tz is None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
            invalid.append(f"invalid_timestamp_grain:{symbol}")
        if index.tz is None:
            continue
        missing = expected.difference(index)
        gaps = Counter(missing.tz_convert(NEW_YORK).date)
        if gaps:
            # The trial rejects any incomplete regular session before labeling.
            invalid.append(f"incomplete_sessions:{symbol}")
        bar_parts = [pd.date_range(s.start_utc, s.end_utc, freq="5min", inclusive="left") for day, s in sessions.items() if day not in gaps]
        bars = bar_parts[0].append(bar_parts[1:]) if bar_parts else pd.DatetimeIndex([], tz="UTC")
        schedule = intraday_schedule(bars, grid)
        opportunities.extend((symbol, str(stamp.tz_convert(NEW_YORK).date()), exit_at) for _, stamp, _, _, exit_at in schedule)
        by_symbol[symbol] = {"expected_sessions": len(dates), "complete_sessions": len(dates) - len(gaps), "incomplete_sessions": len(gaps), "missing_minutes": len(missing), "possible_opportunities": len(schedule), "gap_sessions": [{"session": day.isoformat(), "missing_minutes": count} for day, count in sorted(gaps.items())]}
    frame = pd.DataFrame(opportunities, columns=["symbol", "session", "exit_at"])
    days = sorted(frame.session.unique())
    folds: list[dict[str, Any]] = []
    if len(days) >= 30:
        for number, (embargo_day, chunk) in enumerate(chronological_folds(days), 1):
            boundary = pd.Timestamp(session_info(pd.Timestamp(embargo_day).date()).start_utc)
            test = frame.loc[frame.session.isin(chunk)]
            train = frame.loc[(frame.session < embargo_day) & (frame.exit_at < boundary)]
            folds.append({"fold": number, "test_start": chunk[0], "test_end": chunk[-1], "test_sessions": len(chunk), "possible_test_opportunities": len(test), "possible_train_opportunities": len(train), "test_opportunities_by_symbol": {s: int((test.symbol == s).sum()) for s in protocol["symbols"]}})
    possible = not invalid and len(folds) == 5 and all(f["possible_test_opportunities"] >= MINIMUM_FOLD_OPPORTUNITIES and f["possible_train_opportunities"] > 0 for f in folds)
    cutoff = grid.cutoff_et.strftime("%H:%M") if grid.cutoff_et is not None else None
    return {"status": "support_possible_not_strategy_eligible" if possible else "blocked_sampling_support", "audit": "intraday_grid_timestamp_only", "grid": {"decision": "five_minute_bar_start_plus_5m2s", "entry": "first_minute_after_decision", "exit_horizon_minutes": grid.exit_minutes, "flatten_cutoff_et": cutoff, "flatten_lead_minutes": grid.lead_minutes, "same_session_only": True, "overlapping_positions": False, "warmup_bars": WARMUP_BARS, "feature_lookback_days": FEATURE_LOOKBACK.days}, "possible_opportunities": len(frame), "sessions_with_opportunities": len(days), "folds": folds, "minimum_required": {"sessions": 30, "test_opportunities_per_fold": MINIMUM_FOLD_OPPORTUNITIES, "folds": 5}, "symbols": by_symbol, "invalid_inputs": invalid, "prices_read_for_analysis": False, "strategy_outcomes_computed": False, "trial_claimed": False, "promotion_authority": False, "limitations": ["Possible opportunities are upper bounds; price validity checks, label class balance and model selection can reduce support.", "Fold boundaries assume every counted session yields trial rows; incomplete sessions block the trial outright.", "Passing this audit cannot reopen a consumed campaign or authorize a new trial."]}


def sampling_support(protocol: dict[str, Any], indices: dict[str, pd.DatetimeIndex]) -> dict[str, Any]:
    """Count possible fixed-rule observations using timestamps, never prices."""
    if protocol.get("lookback_sessions") != 60 or protocol.get("holding_sessions") != 5:
        raise ValueError("unsupported sampling contract")
    start, end = protocol["development_start"], protocol["development_end"]
    if not start <= end < protocol["holdout_start"] <= protocol["holdout_end"]:
        raise ValueError("invalid development interval or holdout overlap")
    dates = [day.date() for day in pd.date_range(start, end) if is_trading_day(day.date())]
    parts = []
    for day in dates:
        session = session_info(day)
        parts.append(pd.date_range(session.start_utc, session.end_utc, freq="min", inclusive="left"))
    if not parts:
        raise ValueError("no development sessions")
    expected = parts[0].append(parts[1:])
    candidate_indices = list(range(61, len(dates) - 5, 5))
    common = set(candidate_indices)
    by_symbol = {}
    invalid = []
    for symbol in protocol["symbols"]:
        if symbol not in indices:
            invalid.append(f"symbol_missing:{symbol}")
            common.clear()
            continue
        index = indices[symbol]
        if index.tz is None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
            invalid.append(f"invalid_timestamp_grain:{symbol}")
        if index.tz is None:
            common.clear()
            continue
        missing = expected.difference(index)
        gaps = Counter(missing.tz_convert("America/New_York").date)
        complete = {i for i, day in enumerate(dates) if day not in gaps}
        usable = {i for i in candidate_indices if all(j in complete for j in range(i - 61, i + 6))}
        common &= usable
        boundaries = {stamp for part in parts for stamp in (part[0], part[-1])}
        by_symbol[symbol] = {"expected_minutes": len(expected), "missing_minutes": len(missing), "missing_ratio": len(missing) / len(expected), "expected_sessions": len(dates), "complete_sessions": len(complete), "incomplete_sessions": len(gaps), "missing_boundary_minutes": sum(stamp in boundaries for stamp in missing), "possible_periods": len(usable), "excluded_periods": len(candidate_indices) - len(usable), "gap_sessions": [{"session": day.isoformat(), "missing_minutes": count} for day, count in sorted(gaps.items())]}
    folds = Counter(str(pd.Timestamp(dates[i]).to_period("Q")) for i in common)
    maximum_selections = len(common) * len(protocol["symbols"])
    supported_quarters_upper = sum(n * len(protocol["symbols"]) >= 12 for n in folds.values())
    possible = len(common) >= 60 and maximum_selections >= 100 and supported_quarters_upper >= 6 and not invalid
    return {"status": "support_possible_not_strategy_eligible" if possible else "blocked_sampling_support", "possible_common_periods": len(common), "candidate_periods": len(candidate_indices), "maximum_possible_selections": maximum_selections, "supported_quarters_upper_bound": supported_quarters_upper, "minimum_required": {"periods": 60, "selections": 100, "quarters_with_12_selections": 6}, "common_entry_dates": [dates[i].isoformat() for i in sorted(common)], "periods_by_quarter": dict(folds), "symbols": by_symbol, "invalid_inputs": invalid, "prices_read_for_analysis": False, "strategy_outcomes_computed": False, "trial_claimed": False, "promotion_authority": False, "limitations": ["Possible observations are upper bounds; actual signal selection and valid prices can reduce support.", "Missing bars may reflect no eligible trades; this audit alone does not identify their cause.", "Passing this audit cannot reopen a consumed campaign or authorize a new trial."]}


DAILY_COMPLETENESS_RULES = ("every_minute", "open_and_close_minutes")
DAILY_FOLD_KIND = "calendar_quarter_of_entry_session"
# Holdout-year data is never read by a development study, whatever the contract says.
HOLDOUT_YEAR_START = "2026-01-01"


@dataclass(frozen=True)
class DailyGrid:
    """Daily rebalance grid and session-completeness rule shared with the registered strategy trial."""

    lookback_sessions: int
    hold_sessions: int
    completeness: str = "every_minute"
    # open_and_close_minutes: only the first and last regular minutes are required unless a floor is declared.
    minimum_coverage: float = 1.0

    @classmethod
    def from_protocol(cls, protocol: dict[str, Any]) -> DailyGrid:
        grid, rule = protocol.get("rebalance_grid"), protocol.get("session_completeness")
        if not isinstance(grid, dict) or not isinstance(rule, dict):
            raise ValueError("protocol must declare rebalance_grid and session_completeness")
        lookback, hold = grid.get("lookback_sessions"), grid.get("hold_sessions")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (lookback, hold)):
            raise ValueError("lookback_sessions and hold_sessions must be positive integers")
        name = rule.get("rule")
        if name not in DAILY_COMPLETENESS_RULES:
            raise ValueError("unsupported session completeness rule")
        coverage = rule.get("minimum_minute_coverage", 1.0 if name == "every_minute" else 0.0)
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0 <= coverage <= 1:
            raise ValueError("minimum_minute_coverage must lie in [0, 1]")
        if name == "every_minute" and coverage != 1:
            raise ValueError("every_minute completeness implies full coverage")
        return cls(lookback, hold, name, float(coverage))

    def schedule(self, sessions: int) -> list[tuple[int, int, int]]:
        """Decide after the close of t, enter at the open of t+1, exit at the open of t+1+hold.

        The next decision is t+hold, so its entry coincides with the previous exit and holds never overlap.
        """
        last = sessions - 1 - self.hold_sessions
        return [(t, t + 1, t + 1 + self.hold_sessions) for t in range(self.lookback_sessions, last, self.hold_sessions)]

    def required_sessions(self, decision: int) -> range:
        """Every session from the lookback start through the exit must be complete; gaps never shorten a window."""
        return range(decision - self.lookback_sessions, decision + self.hold_sessions + 2)

    def session_complete(self, present: np.ndarray) -> bool:
        if self.completeness == "every_minute":
            return bool(present.all())
        return bool(present[0] and present[-1] and present.mean() >= self.minimum_coverage)


def daily_development_dates(protocol: dict[str, Any]) -> list[date]:
    """Development sessions, refusing any window that touches the holdout or the holdout year."""
    start, end = protocol["development_start"], protocol["development_end"]
    if not start <= end < protocol["holdout_start"] <= protocol["holdout_end"]:
        raise ValueError("invalid development interval or holdout overlap")
    if end >= HOLDOUT_YEAR_START:
        raise ValueError("development window must end before 2026; holdout-year data is never read")
    dates = list(_trading_days(start, end))
    if not dates:
        raise ValueError("no development sessions")
    return dates


@cache
def _trading_days(start: str, end: str) -> tuple[date, ...]:
    return tuple(day.date() for day in pd.date_range(start, end) if is_trading_day(day.date()))


@cache
def _session_minutes(day: date) -> pd.DatetimeIndex:
    session = session_info(day)
    return pd.date_range(session.start_utc, session.end_utc, freq="min", inclusive="left")


def regular_session_minutes(dates: list[date]) -> list[pd.DatetimeIndex]:
    """Expected regular-session minute stamps per date; calendar lookups are cached because they are slow."""
    return [_session_minutes(day) for day in dates]


def daily_session_flags(index: pd.DatetimeIndex, parts: list[pd.DatetimeIndex], grid: DailyGrid) -> np.ndarray:
    """Per-session completeness from timestamps alone (early closes use their own session length)."""
    present = parts[0].append(parts[1:]).isin(index)
    bounds = np.cumsum([0] + [len(part) for part in parts])
    return np.array([grid.session_complete(present[a:b]) for a, b in zip(bounds[:-1], bounds[1:])], dtype=bool)


def strategy_bucket(protocol: dict[str, Any]) -> int:
    """Buy signals per decision for a full universe, mirroring CrossSectionalMomentumStrategy's bucket rule."""
    params, universe = protocol["strategy_parameters"], len(protocol["symbols"])
    if universe < int(params["min_universe"]):
        return 0
    return max(1, int(round(universe * float(params["top_quantile"]))))


def quarter_fold(day: date | str) -> str:
    return str(pd.Timestamp(day).to_period("Q"))


def usable_daily_periods(grid: DailyGrid, schedule: list[tuple[int, int, int]], flags: dict[str, np.ndarray], symbols: list[str]) -> list[tuple[int, int, int]]:
    if any(symbol not in flags for symbol in symbols):
        return []
    return [row for row in schedule if all(flags[s][j] for s in symbols for j in grid.required_sessions(row[0]))]


def daily_rebalance_support(protocol: dict[str, Any], indices: dict[str, pd.DatetimeIndex]) -> dict[str, Any]:
    """Count possible daily-rebalance decisions and selections per fold using timestamps, never prices."""
    grid = DailyGrid.from_protocol(protocol)
    if protocol.get("folds") != DAILY_FOLD_KIND:
        raise ValueError(f"folds must be {DAILY_FOLD_KIND}")
    thresholds = protocol["support_thresholds"]
    dates = daily_development_dates(protocol)
    symbols = list(protocol["symbols"])
    parts = regular_session_minutes(dates)
    schedule = grid.schedule(len(dates))
    # The alternative rule is reported for transparency only; the contract's rule decides support.
    rules = {rule: grid if rule == grid.completeness else DailyGrid(grid.lookback_sessions, grid.hold_sessions, rule, 1.0 if rule == "every_minute" else 0.0) for rule in DAILY_COMPLETENESS_RULES}
    lower, upper = pd.Timestamp(protocol["development_start"], tz=NEW_YORK), pd.Timestamp(protocol["development_end"], tz=NEW_YORK) + pd.Timedelta(days=1)
    invalid: list[str] = []
    flags: dict[str, dict[str, np.ndarray]] = {rule: {} for rule in rules}
    by_symbol: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        if symbol not in indices:
            invalid.append(f"symbol_missing:{symbol}")
            continue
        index = indices[symbol]
        if index.tz is None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
            invalid.append(f"invalid_timestamp_grain:{symbol}")
        if index.tz is None:
            continue
        if len(index) and (index.min() < lower or index.max() >= upper or index.max() >= pd.Timestamp(HOLDOUT_YEAR_START, tz="UTC")):
            invalid.append(f"timestamps_outside_development:{symbol}")
        for rule, rule_grid in rules.items():
            flags[rule][symbol] = daily_session_flags(index, parts, rule_grid)
        contract_flags = flags[grid.completeness][symbol]
        by_symbol[symbol] = {"expected_sessions": len(dates), "complete_sessions_by_rule": {rule: int(f[symbol].sum()) for rule, f in flags.items()}, "incomplete_sessions": int((~contract_flags).sum()), "first_incomplete_sessions": [dates[i].isoformat() for i in np.flatnonzero(~contract_flags)[:10]]}
    bucket = strategy_bucket(protocol)
    by_rule = {}
    for rule, rule_grid in rules.items():
        periods = usable_daily_periods(rule_grid, schedule, flags[rule], symbols)
        by_rule[rule] = {"possible_periods": len(periods), "possible_selections": len(periods) * bucket}
    periods = usable_daily_periods(grid, schedule, flags[grid.completeness], symbols)
    counts = Counter(quarter_fold(dates[entry]) for _, entry, _ in periods)
    folds = [{"fold": fold, "possible_periods": n, "possible_selections": n * bucket} for fold, n in sorted(counts.items())]
    supported_folds = sum(row["possible_selections"] >= thresholds["minimum_selections_per_fold"] for row in folds)
    reasons = []
    if invalid:
        reasons.append("invalid_inputs")
    if bucket == 0:
        reasons.append("universe_below_strategy_min_universe")
    if len(periods) < thresholds["minimum_decision_periods"]:
        reasons.append("decision_periods_below_minimum")
    if len(periods) * bucket < thresholds["minimum_selections"]:
        reasons.append("selections_below_minimum")
    if supported_folds < thresholds["minimum_supported_folds"]:
        reasons.append("supported_folds_below_minimum")
    return {
        "status": "blocked_sampling_support" if reasons else "support_possible_not_strategy_eligible",
        "audit": "daily_rebalance_grid_timestamp_only", "blocking_reasons": reasons,
        "grid": {"decision": "after_close_of_session_t_using_closes_through_t", "entry": "open_of_session_t_plus_1", "exit": f"open_of_session_t_plus_{1 + grid.hold_sessions}", "lookback_sessions": grid.lookback_sessions, "hold_sessions": grid.hold_sessions, "overlapping_positions": False, "session_completeness": grid.completeness, "minimum_minute_coverage": grid.minimum_coverage, "required_complete_sessions_per_period": grid.lookback_sessions + grid.hold_sessions + 2},
        "universe": {"symbols": symbols, "size": len(symbols), "strategy_min_universe": int(protocol["strategy_parameters"]["min_universe"]), "top_quantile": float(protocol["strategy_parameters"]["top_quantile"]), "buy_selections_per_period": bucket},
        "development_sessions": len(dates), "candidate_periods": len(schedule),
        "possible_periods": len(periods), "possible_selections": len(periods) * bucket,
        "supported_folds": supported_folds, "folds": folds, "fold_kind": DAILY_FOLD_KIND,
        "by_completeness_rule": by_rule, "minimum_required": dict(thresholds),
        "common_entry_dates": [dates[entry].isoformat() for _, entry, _ in periods],
        "symbols": by_symbol, "invalid_inputs": invalid,
        "prices_read_for_analysis": False, "strategy_outcomes_computed": False, "trial_claimed": False, "promotion_authority": False,
        "limitations": ["Possible selections are upper bounds; price validity checks and the strategy's own abstentions can reduce support.", "Completeness is judged from timestamps; the trial additionally rejects sessions with invalid prices.", "Passing this audit cannot reopen a consumed campaign or authorize a new trial."],
    }


def load_governed_development_timestamps(protocol: dict[str, Any], acquisition_path: Path) -> tuple[dict[str, pd.DatetimeIndex], dict[str, Any]]:
    """Verify frozen source hashes and inspect only their timestamp columns."""
    acquisition_raw = acquisition_path.read_bytes()
    acquisition = json.loads(acquisition_raw)
    manifest_path = Path(acquisition["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = acquisition_path.parent / manifest_path
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    identity = manifest["dataset_identity"]
    interval = identity["request_interval"]
    if interval["start_date"] != protocol["development_start"] or interval["end_date"] != protocol["development_end"] or identity["feed"] != protocol["feed"] or identity["adjustment"] != protocol["adjustment"] or sorted(identity["symbols"]) != sorted(protocol["symbols"]):
        raise ValueError("acquisition differs from frozen development contract")
    if not protocol["development_end"] < protocol["holdout_start"]:
        raise ValueError("holdout overlap")
    if acquisition.get("quality_passed") is not True or manifest.get("quality_passed") is not True:
        raise ValueError("acquisition quality unverified")
    sources, indices = {}, {}
    for symbol in protocol["symbols"]:
        rows = [row for row in manifest["symbols"] if row["symbol"] == symbol]
        if len(rows) != 1:
            raise ValueError("missing or duplicate symbol provenance")
        row = rows[0]
        path = Path(row["csv_path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.resolve() != (manifest_path.parent / f"{symbol}.csv").resolve():
            raise ValueError("source outside governed dataset")
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["content_sha256"]:
            raise ValueError(f"source hash mismatch:{symbol}")
        frame = pd.read_csv(path, usecols=["timestamp"], dtype=str)
        if not frame["timestamp"].str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", na=False).all():
            raise ValueError(f"timezone missing from source:{symbol}")
        index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True, errors="raise"))
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"source changed during audit:{symbol}")
        indices[symbol] = index
        sources[symbol] = {"path": str(path), "sha256": digest, "stable_during_read": True, "columns_used": ["timestamp"]}
    report = dict(generated_at=datetime.now(UTC).isoformat(), protocol_hash=protocol_hash(protocol), sources=sources, acquisition_sha256=hashlib.sha256(acquisition_raw).hexdigest(), manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())
    return indices, report


def audit_sampling_acquisition(protocol: dict[str, Any], acquisition_path: Path) -> dict[str, Any]:
    """Audit fixed-grid support after canonical source verification."""
    if "rebalance_grid" in protocol:
        DailyGrid.from_protocol(protocol)
        daily_development_dates(protocol)  # Refuse holdout or holdout-year windows before any file read.
        indices, provenance = load_governed_development_timestamps(protocol, acquisition_path)
        report = daily_rebalance_support(protocol, indices)
        report.update(provenance)
        return report
    intraday = IntradayGrid.from_protocol(protocol) is not None
    indices, provenance = load_governed_development_timestamps(protocol, acquisition_path)
    report = intraday_support(protocol, indices) if intraday else sampling_support(protocol, indices)
    report.update(provenance)
    return report


def protocol_hash(protocol: dict[str, Any]) -> str:
    contract = {key: value for key, value in protocol.items() if key != "feasibility_review"}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def screen_development(protocol: dict[str, Any], ledgers: dict[int, pd.DataFrame]) -> dict[str, Any]:
    """Use session bootstrap bounds as a feasibility screen, not proof of edge."""
    development = next(row for row in protocol["retrospective_intervals"] if row["name"] == "development")
    days = [str(day.date()) for day in pd.date_range(development["start"], development["end"]) if is_trading_day(day.date())]
    acceptance = protocol["acceptance"]
    validation_sessions = sum(is_trading_day(day.date()) for row in protocol["retrospective_intervals"] if row["name"] != "development" for day in pd.date_range(row["start"], row["end"]))
    results = []
    for horizon in protocol["horizons_minutes"]:
        ledger = ledgers[int(horizon)]
        selected = ledger.loc[ledger["session"].isin(days) & ledger["selected"].eq(True)]
        gross = pd.to_numeric(selected["gross_return_bps"], errors="coerce")
        if not np.isfinite(gross).all():
            raise ValueError("development markouts must be finite")
        grouped = selected.assign(gross=gross).groupby("session")["gross"]
        counts = grouped.count().reindex(days, fill_value=0).to_numpy(dtype=float)
        sums = grouped.sum().reindex(days, fill_value=0).to_numpy(dtype=float)
        enough = len(days) >= 40 and len(selected) >= 20
        expected_lower = gross_upper = None
        if enough:
            rng = np.random.default_rng(20260907)
            block = 5
            starts = rng.integers(0, len(days) - block + 1, size=(5000, int(np.ceil(len(days) / block))))
            indices = (starts[..., None] + np.arange(block)).reshape(5000, -1)[:, :len(days)]
            totals = counts[indices].sum(axis=1)
            expected_lower = float(np.quantile(totals / len(days) * validation_sessions, 0.05))
            usable = totals > 0
            gross_upper = float(np.quantile(sums[indices].sum(axis=1)[usable] / totals[usable], 0.95))
        reasons = []
        if not enough:
            reasons.append("development_support_insufficient")
        if expected_lower is None or expected_lower < acceptance["minimum_completed_validation_trades"]:
            reasons.append("expected_trade_support_below_requirement")
        if gross_upper is None or gross_upper <= 2 * acceptance["primary_one_way_cost_bps"]:
            reasons.append("optimistic_gross_edge_does_not_cover_costs")
        if not protocol.get("hypothesis") or not protocol.get("stopping_rule"):
            reasons.append("hypothesis_or_stopping_rule_missing")
        results.append({"horizon_minutes": horizon, "development_trades": len(selected), "development_sessions": len(days), "projected_validation_sessions": validation_sessions, "expected_validation_trades_lower": expected_lower, "optimistic_gross_edge_bps": gross_upper, "required_round_trip_cost_bps": 2 * acceptance["primary_one_way_cost_bps"], "reasons": reasons, "eligible": not reasons})
    return {"status": "eligible_for_fixed_study" if results and all(row["eligible"] for row in results) else "blocked", "protocol_hash": protocol_hash(protocol), "horizons": results, "scope": "development_only_feasibility_not_confirmation", "method": "five_session_block_bootstrap_5000_replicates_seed_20260907", "limitations": ["Projection assumes development trade frequency persists.", "Optimistic gross upper bound is only a rejection screen, not evidence of positive net edge.", "Prior project reuse of these data remains a selection-bias limitation."], "promotion_authority": False}


def require_feasibility(protocol: dict[str, Any]) -> None:
    review = protocol.get("feasibility_review")
    if not isinstance(review, dict) or not review.get("path") or not review.get("sha256"):
        raise ValueError("feasibility_review_required_before_new_study")
    raw = Path(review["path"]).read_bytes()
    report = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != review["sha256"] or report.get("protocol_hash") != protocol_hash(protocol):
        raise ValueError("feasibility_review_contract_mismatch")
    if report.get("status") != "eligible_for_fixed_study":
        raise ValueError("research_feasibility_blocked")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--ledger-dir", type=Path)
    inputs.add_argument("--sampling-acquisition", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.sampling_acquisition is not None:
        report = audit_sampling_acquisition(protocol, args.sampling_acquisition)
        atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
        get_logger(__name__).info("SAMPLING_SUPPORT_AUDITED", extra={"status": report["status"], "output": str(args.output)})
        return
    paths = {int(h): args.ledger_dir / f"opportunities_h{h}.csv" for h in protocol["horizons_minutes"]}
    report = screen_development(protocol, {h: pd.read_csv(path) for h, path in paths.items()})
    report["ledger_sha256"] = {str(h): hashlib.sha256(path.read_bytes()).hexdigest() for h, path in paths.items()}
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    get_logger(__name__).info("RESEARCH_FEASIBILITY_SCREENED", extra={"status": report["status"], "output": str(args.output)})


if __name__ == "__main__":
    main()
