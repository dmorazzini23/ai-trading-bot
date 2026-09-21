"""Bounded development feature comparisons without models, signals or returns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trading.features.day_sleeve import build_day_sleeve_features
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.research_feasibility import load_governed_development_timestamps
from ai_trading.tools.stock_development_readiness import valid_bars
from ai_trading.tools.train_replay_aligned_model import _feature_frame
from ai_trading.utils.market_calendar import is_trading_day, session_info

ROOT = Path(__file__).resolve().parents[2]


def compare_history(bars: pd.DataFrame, *, symbol: str, end: int) -> dict:
    prefix = bars.iloc[:end].copy()
    training = _feature_frame(prefix, symbol=symbol).iloc[-1:]
    runtime = build_day_sleeve_features(prefix)
    future = _feature_frame(bars.iloc[:end + 10].copy(), symbol=symbol).iloc[end - 1:end]
    truncated = build_day_sleeve_features(prefix.iloc[-200:].copy())
    # Match the default runtime fetch window at the first finalized decision.
    # This is distinct from the minimum 200-bar feature support check.
    decision_at = prefix.index[-1] + pd.Timedelta(minutes=5, seconds=2)
    runtime_window = prefix.loc[prefix.index >= decision_at - pd.Timedelta(days=10)]
    rolling = build_day_sleeve_features(runtime_window)
    columns = list(runtime.columns)
    same = np.isclose(training[columns].to_numpy(), runtime.to_numpy(), rtol=1e-10, atol=1e-10)
    causal = np.isclose(training[columns].to_numpy(), future[columns].to_numpy(), rtol=1e-10, atol=1e-10)
    history = np.isclose(runtime.to_numpy(), truncated[columns].to_numpy(), rtol=1e-10, atol=1e-10)
    serving = np.isclose(training[columns].to_numpy(), rolling[columns].to_numpy(), rtol=1e-10, atol=1e-10)
    return dict(symbol=symbol, prefix_bars=end, bar_start=prefix.index[-1].isoformat(),
                same_history_equal=bool(same.all()), prefix_causal=bool(causal.all()),
                truncated_history_equal=bool(history.all()),
                runtime_lookback_days=10,
                runtime_window_bars=len(runtime_window),
                runtime_window_equal=bool(serving.all()),
                runtime_window_mismatch_columns=[c for c, ok in zip(columns, serving[0]) if not ok],
                same_history_mismatch_columns=[c for c, ok in zip(columns, same[0]) if not ok],
                truncated_history_mismatch_columns=[c for c, ok in zip(columns, history[0]) if not ok])


def main() -> None:
    protocol = dict(development_start='2024-01-01', development_end='2025-12-31',
                    holdout_start='2026-09-09', feed='sip', adjustment='split',
                    symbols=['AAPL', 'AMZN', 'MSFT'])
    _, provenance = load_governed_development_timestamps(
        protocol, ROOT / 'artifacts/stock_development/acquisition.json')
    results = []
    for symbol, source in provenance['sources'].items():
        path = Path(source['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('source changed before feature audit')
        frame = pd.read_csv(path, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop('timestamp'), utc=True))
        if frame.index.min() < pd.Timestamp('2024-01-01', tz='UTC') or frame.index.max() >= pd.Timestamp('2026-01-01', tz='UTC'):
            raise ValueError('actual timestamps outside development')
        parts = []
        for day in pd.date_range('2024-01-01', '2025-12-31'):
            if not is_trading_day(day.date()):
                continue
            session = session_info(day.date())
            expected = pd.date_range(session.start_utc, session.end_utc, freq='min', inclusive='left')
            subset = frame.loc[(frame.index >= expected[0]) & (frame.index <= expected[-1])]
            if subset.index.has_duplicates or not subset.index.sort_values().equals(expected) or not valid_bars(subset).all():
                raise ValueError('incomplete or invalid session; do not compress feature history')
            parts.append(subset.resample('5min', origin=expected[0]).agg(
                dict(open='first', high='max', low='min', close='last', volume='sum')))
        bars = pd.concat(parts).sort_index()
        # Fixed first eligible development prefixes; no return-dependent selection.
        for end in [250, 500, 1000]:
            results.append(compare_history(bars, symbol=symbol, end=end))
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('source changed during feature audit')
    report = dict(results=results, sources=provenance['sources'],
                  scope='nine_fixed_prefix_checks_not_exhaustive_serving_parity',
                  tolerance=dict(rtol=1e-10, atol=1e-10), models_loaded=False,
                  returns_computed=False, trial_claimed=False, holdout_evaluated=False,
                  promotion_authority=False, live_history_equivalence_verified=False)
    atomic_write_text(ROOT / 'artifacts/stock_development/feature_parity.json',
                      json.dumps(report, indent=2, sort_keys=True) + '\n')
    get_logger(__name__).info('STOCK_FEATURE_PARITY_AUDITED', extra={'results': results})


if __name__ == '__main__':
    main()
