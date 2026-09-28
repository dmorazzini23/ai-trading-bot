"""Audit weekly ETF input support without computing signals or returns."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.endpoint_feasibility import check_required_bars
from ai_trading.tools.research_feasibility import load_governed_development_timestamps
from ai_trading.utils.market_calendar import is_trading_day, session_info

DEVELOPMENT_START = "2024-01-01"
DEVELOPMENT_END = "2025-12-31"
SYMBOLS = ["DIA", "IWM", "QQQ", "SPY", "XLE", "XLF"]
PROTOCOL = {
    "development_start": DEVELOPMENT_START,
    "development_end": DEVELOPMENT_END,
    "holdout_start": "2026-09-09",
    "holdout_end": "2026-12-08",
    "symbols": SYMBOLS,
    "feed": "sip",
    "adjustment": "all",
}


def weekly_observation_grid(days: list[date]) -> list[dict[str, Any]]:
    """Require 21 completed closes, the next open, and the five-session exit open."""
    sessions = [session_info(day) for day in days]
    grid = []
    for index in range(20, len(days) - 6):
        if days[index].weekday() != 4:
            continue
        observations = {
            f"close_{lag}": (pd.Timestamp(sessions[index - lag].end_utc) - pd.Timedelta(minutes=1), "close")
            for lag in range(20, -1, -1)
        }
        observations["entry_open"] = (pd.Timestamp(sessions[index + 1].start_utc), "open")
        observations["exit_open"] = (pd.Timestamp(sessions[index + 6].start_utc), "open")
        grid.append({"session": days[index].isoformat(), "quarter": str(pd.Timestamp(days[index]).to_period("Q")), "observations": observations})
    return grid


def development_actions(capture: dict[str, Any], days: list[date]) -> dict[str, list[dict[str, Any]]]:
    """Validate saved event identities and ex-date joins; never infer absent events."""
    request = capture.get("request") or {}
    start = date.fromisoformat(str(request.get("start")))
    end = date.fromisoformat(str(request.get("end")))
    if (capture.get("status") != "captured" or capture.get("pagination_complete") is not True
            or request.get("data_quality") != "complete" or set(str(request.get("symbols", "")).split(",")) != set(SYMBOLS)
            or start > date.fromisoformat(DEVELOPMENT_START) or end < date.fromisoformat(DEVELOPMENT_END)):
        raise ValueError("corporate_action_capture_incomplete_or_wrong_scope")
    pages = capture.get("pages")
    if not isinstance(pages, list) or not pages or pages[-1].get("next_page_token"):
        raise ValueError("corporate_action_pagination_unverified")
    available = set(days)
    seen: set[str] = set()
    events: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in SYMBOLS}
    for page in pages:
        groups = page.get("corporate_actions")
        if not isinstance(groups, dict) or set(groups) - {"cash_dividends", "forward_splits"}:
            raise ValueError("unsupported_corporate_action_group")
        for kind, rows in groups.items():
            if not isinstance(rows, list):
                raise ValueError("invalid_corporate_action_rows")
            for row in rows:
                identifier = row.get("id")
                symbol = row.get("symbol")
                if not identifier or identifier in seen or symbol not in events:
                    raise ValueError("invalid_corporate_action_identity")
                seen.add(identifier)
                process_date = date.fromisoformat(str(row.get("process_date")))
                if not start <= process_date <= end:
                    raise ValueError("corporate_action_process_date_outside_capture")
                ex_date = date.fromisoformat(str(row.get("ex_date")))
                if not DEVELOPMENT_START <= ex_date.isoformat() <= DEVELOPMENT_END:
                    continue
                if ex_date not in available:
                    raise ValueError("corporate_action_ex_date_not_session")
                if kind == "cash_dividends":
                    rate = float(row.get("rate", "nan"))
                    if not math.isfinite(rate) or rate <= 0:
                        raise ValueError("invalid_dividend_rate")
                else:
                    old_rate, new_rate = float(row.get("old_rate", "nan")), float(row.get("new_rate", "nan"))
                    if not all(math.isfinite(rate) and rate > 0 for rate in (old_rate, new_rate)):
                        raise ValueError("invalid_split_rate")
                events[symbol].append({"id": identifier, "kind": kind, "ex_date": ex_date})
    return events


def audit(acquisition_path: Path, actions_path: Path) -> dict[str, Any]:
    """Verify governed sources and endpoint availability without evaluating the rule."""
    _, provenance = load_governed_development_timestamps(PROTOCOL, acquisition_path)
    days = [day.date() for day in pd.date_range(DEVELOPMENT_START, DEVELOPMENT_END) if is_trading_day(day.date())]
    grid = weekly_observation_grid(days)
    capture_raw = actions_path.read_bytes()
    events = development_actions(json.loads(capture_raw), days)
    day_index = {day: index for index, day in enumerate(days)}
    results: dict[str, Any] = {}
    common = {row["session"] for row in grid}
    for symbol in SYMBOLS:
        path = Path(provenance["sources"][symbol]["path"])
        before = path.stat()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != provenance["sources"][symbol]["sha256"]:
            raise ValueError(f"source_changed_after_provenance_check:{symbol}")
        frame = pd.read_csv(io.BytesIO(raw), usecols=["timestamp", "open", "high", "low", "close"])
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp"), utc=True, errors="raise"))
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"source_timestamps_not_chronological:{symbol}")
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"source_changed_during_bar_read:{symbol}")
        support = check_required_bars(frame, grid)
        common &= set(support["valid_entry_dates"])
        action_grid = []
        for event in events[symbol]:
            index = day_index[event["ex_date"]]
            if index == 0:
                raise ValueError("action_has_no_development_previous_session")
            previous = session_info(days[index - 1])
            current = session_info(event["ex_date"])
            action_grid.append({"session": str(event["id"]), "observations": {
                "previous_close": (pd.Timestamp(previous.end_utc) - pd.Timedelta(minutes=1), "close"),
                "ex_open": (pd.Timestamp(current.start_utc), "open"),
                "ex_close": (pd.Timestamp(current.end_utc) - pd.Timedelta(minutes=1), "close"),
            }})
        action_support = check_required_bars(frame, action_grid)
        kinds = Counter(event["kind"] for event in events[symbol])
        results[symbol] = {
            "valid_weeks": support["possible_periods"],
            "weekly_rejection_counts": support["rejection_counts"],
            "weekly_rejected_sessions": support["rejected_periods"],
            "captured_action_counts": dict(kinds),
            "captured_actions_with_valid_boundary_bars": action_support["possible_periods"],
            "captured_action_boundary_rejections": action_support["rejected_periods"],
        }
    quarters = Counter(row["quarter"] for row in grid if row["session"] in common)
    return {
        "artifact_type": "weekly_etf_data_preflight",
        "generated_at": datetime.now(UTC).isoformat(),
        "development_interval": [DEVELOPMENT_START, DEVELOPMENT_END],
        "protected_holdout_interval": [PROTOCOL["holdout_start"], PROTOCOL["holdout_end"]],
        "candidate_friday_sessions": len(grid),
        "common_valid_weeks": len(common),
        "common_valid_fridays": sorted(common),
        "valid_weeks_by_quarter": dict(sorted(quarters.items())),
        "symbols": results,
        "sources": provenance["sources"],
        "acquisition_sha256": provenance["acquisition_sha256"],
        "manifest_sha256": provenance["manifest_sha256"],
        "corporate_action_capture_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "status": "observation_support_measured_action_completeness_unverified",
        "limitations": [
            "The Friday-only schedule and 20-session lookback are a draft observation grid, not an approved trial contract.",
            "Captured action identity, ex-date and bar joins do not independently certify all distributions or provider revision history.",
            "Historical all-adjusted bars do not establish point-in-time action availability or executable opening fills.",
        ],
        "signals_computed": False,
        "returns_computed": False,
        "trial_claimed": False,
        "holdout_evaluated": False,
        "orders_sent": 0,
        "promotion_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--corporate-actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.acquisition, args.corporate_actions)
    atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    get_logger(__name__).info("WEEKLY_ETF_DATA_PREFLIGHT_COMPLETE", extra={"common_valid_weeks": report["common_valid_weeks"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
