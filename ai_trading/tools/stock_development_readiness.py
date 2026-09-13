"""Fixed development-only OHLCV and feature-input audit, without model evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.research_feasibility import load_governed_development_timestamps
from ai_trading.utils.market_calendar import is_trading_day, session_info

ROOT = Path(__file__).resolve().parents[2]


def valid_bars(frame: pd.DataFrame) -> pd.Series:
    """Strict price geometry and positive volume; no repair or imputation."""
    data = frame[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
    finite = pd.Series(np.isfinite(data.to_numpy()).all(axis=1), index=data.index)
    return (finite & data.gt(0).all(axis=1)
            & data['high'].ge(data[['open', 'close', 'low']].max(axis=1))
            & data['low'].le(data[['open', 'close', 'high']].min(axis=1)))


def assess(frame: pd.DataFrame, sessions: list[tuple[pd.Timestamp, pd.Timestamp]]) -> dict:
    unique = ~frame.index.duplicated(keep=False)
    valid = valid_bars(frame) & unique
    valid_index = frame.index[valid]
    observed_index = frame.index[unique]
    complete_blocks: list[bool] = []
    candidates = []
    missing_minutes = invalid_minutes = 0
    for opening, closing in sessions:
        expected = pd.date_range(opening, closing, freq='min', inclusive='left')
        observed = expected.isin(observed_index)
        good = expected.isin(valid_index)
        missing_minutes += int((~observed).sum())
        invalid_minutes += int((observed & ~good).sum())
        offset = len(complete_blocks)
        blocks = good.reshape(-1, 5).all(axis=1)
        complete_blocks.extend(bool(x) for x in blocks)
        # Decision is after a completed 5Min bar; exit is D+6, strictly before close.
        for minute in range(5, len(expected), 5):
            boundary = minute + 6 >= len(expected)
            window = slice(minute - 5, minute + 7)
            candidates.append((offset + minute // 5 - 1, boundary,
                               bool(observed[window].all()) if not boundary else False,
                               bool(good[window].all()) if not boundary else False,
                               bool(good[:minute].all())))
    history = pd.Series(complete_blocks, dtype=float).rolling(200, min_periods=200).sum().eq(200)
    counts = dict(candidate_slots=len(candidates), boundary_excluded=0,
                  missing_or_duplicate_window=0, invalid_ohlcv_window=0,
                  feature_history_unavailable=0, feature_input_supported=0,
                  complete_timestamp_windows=0, valid_ohlcv_windows=0)
    for block, boundary, observed, good, session_good in candidates:
        if boundary:
            counts['boundary_excluded'] += 1
        elif not observed:
            counts['missing_or_duplicate_window'] += 1
        else:
            counts['complete_timestamp_windows'] += 1
            if not good:
                counts['invalid_ohlcv_window'] += 1
            else:
                counts['valid_ohlcv_windows'] += 1
                if not history.iloc[block] or not session_good:
                    counts['feature_history_unavailable'] += 1
                else:
                    counts['feature_input_supported'] += 1
    assert counts['candidate_slots'] == sum(counts[k] for k in [
        'boundary_excluded', 'missing_or_duplicate_window', 'invalid_ohlcv_window',
        'feature_history_unavailable', 'feature_input_supported'])
    return dict(counts=counts, missing_or_duplicate_session_minutes=missing_minutes,
                invalid_session_minutes=invalid_minutes,
                feature_status='necessary_input_support_not_full_feature_or_execution_validation')


def main() -> None:
    protocol = dict(development_start='2024-01-01', development_end='2025-12-31',
                    holdout_start='2026-09-09', feed='sip', adjustment='split',
                    symbols=['AAPL', 'AMZN', 'MSFT'])
    acquisition = ROOT / 'artifacts/stock_development/acquisition.json'
    _, provenance = load_governed_development_timestamps(protocol, acquisition)
    sessions = []
    for day in pd.date_range(protocol['development_start'], protocol['development_end']):
        if is_trading_day(day.date()):
            session = session_info(day.date())
            sessions.append((pd.Timestamp(session.start_utc), pd.Timestamp(session.end_utc)))
    results = {}
    for symbol, source in provenance['sources'].items():
        path = Path(source['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('source hash mismatch before audit')
        frame = pd.read_csv(path, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop('timestamp'), utc=True))
        if frame.index.min() < pd.Timestamp('2024-01-01', tz='UTC') or frame.index.max() >= pd.Timestamp('2026-01-01', tz='UTC'):
            raise ValueError('actual data outside frozen development interval')
        results[symbol] = assess(frame, sessions)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('source changed during audit')
    report = dict(protocol=protocol, results=results, sources=provenance['sources'],
                  acquisition_sha256=provenance['acquisition_sha256'],
                  manifest_sha256=provenance['manifest_sha256'],
                  feature_requirement='200_complete_regular_session_5Min_bars_and_complete_current_session_to_decision',
                  feature_parity_verified=False, execution_verified=False,
                  returns_computed=False, models_fitted=0, trial_claimed=False,
                  holdout_evaluated=False, promotion_authority=False)
    atomic_write_text(ROOT / 'artifacts/stock_development/readiness.json',
                      json.dumps(report, indent=2, sort_keys=True) + '\n')
    get_logger(__name__).info('STOCK_DEVELOPMENT_READINESS_AUDITED', extra={'results': results})


if __name__ == '__main__':
    main()
