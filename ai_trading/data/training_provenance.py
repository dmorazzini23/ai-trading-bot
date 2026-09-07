"""Recompute representative training coverage from timestamped source bars."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ai_trading.data.historical_backfill import build_fetch_windows
from ai_trading.data.historical_bars import load_historical_bars
from ai_trading.models.contracts import infer_day_sleeve_regimes
from ai_trading.utils.market_calendar import session_info


def validate_training_provenance(
    data_dir: Path, *, symbols: Sequence[str], dataset_identity: Mapping[str, Any],
    min_sessions: int = 20, max_missing_ratio: float = 0.02,
    timestamp_col: str = "timestamp",
) -> dict[str, Any]:
    """Validate actual calendar coverage; manifest quality flags are insufficient."""
    if min_sessions < 1 or not 0 <= max_missing_ratio < 1:
        raise ValueError("invalid training coverage thresholds")
    interval = dataset_identity.get("request_interval")
    if not isinstance(interval, Mapping):
        raise ValueError("training acquisition date interval is missing")
    start = date.fromisoformat(str(interval.get("start_date")))
    end = date.fromisoformat(str(interval.get("end_date")))
    windows = build_fetch_windows(start, end, window_sessions=20)
    dates = [day for window in windows for day in window.session_dates]
    expected_parts = []
    for day in dates:
        session = session_info(day)
        expected_parts.append(pd.date_range(session.start_utc, session.end_utc, freq="min", inclusive="left"))
    if not expected_parts:
        raise ValueError("training acquisition interval contains no sessions")
    expected = expected_parts[0].append(expected_parts[1:])
    reasons: list[str] = []
    coverage: dict[str, Any] = {}
    for symbol in sorted(set(symbols)):
        path = data_dir / f"{symbol}.csv"
        if not path.is_file():
            reasons.append(f"symbol_missing:{symbol}")
            continue
        frame, diagnostics = load_historical_bars(path, timestamp_col=timestamp_col, require_timestamp=True)
        index = pd.DatetimeIndex(frame.index)
        observed = expected.intersection(index)
        session_dates = sorted(set(observed.tz_convert("America/New_York").date))
        missing = expected.difference(index)
        ratio = len(missing) / len(expected)
        in_session = frame.loc[index.isin(expected)]
        distribution = pd.Series(infer_day_sleeve_regimes(in_session["close"])).value_counts()
        if len(session_dates) < min_sessions:
            reasons.append(f"insufficient_sessions:{symbol}")
        if ratio > max_missing_ratio:
            reasons.append(f"excess_missing_bars:{symbol}")
        if not len(distribution):
            reasons.append(f"regime_evidence_missing:{symbol}")
        coverage[symbol] = {
            "start": index.min().isoformat(), "end": index.max().isoformat(),
            "observed_session_count": len(session_dates),
            "observed_session_dates": [day.isoformat() for day in session_dates],
            "expected_session_count": len(dates), "expected_rows": len(expected),
            "observed_rows": len(observed), "missing_rows": len(missing),
            "missing_ratio": ratio, "out_of_session_rows": len(index.difference(expected)),
            "sessions_with_gaps": len(set(missing.tz_convert("America/New_York").date)),
            "regime_distribution": {str(key): int(value) for key, value in distribution.items()},
            "load_diagnostics": diagnostics.as_dict(),
        }
    return {
        "quality_passed": not reasons, "rejection_reasons": reasons,
        "coverage": coverage, "requested_symbols": sorted(set(symbols)),
        "date_interval": {"start": start.isoformat(), "end": end.isoformat()},
        "criteria": {"min_sessions": min_sessions, "max_missing_ratio": max_missing_ratio},
        "regime_classifier": "day_sleeve_past_only_v1", "research_only": True,
    }
