"""Fixed development-only 30-minute momentum study; no trading authority."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.data.historical_bars import load_historical_bars
from ai_trading.data.training_provenance import validate_training_provenance
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.train_replay_aligned_model import _feature_frame, _resolve_training_input
from ai_trading.utils.market_calendar import is_trading_day, session_info


def opportunities(frame: pd.DataFrame, features: pd.DataFrame, folds: list[dict[str, Any]], symbol: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Use bar-start timestamps: signal close precedes next bar's entry open."""
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    previous_end = None
    for fold in folds:
        start, end = pd.Timestamp(fold['test_start']), pd.Timestamp(fold['test_end'])
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError('invalid or overlapping folds')
        previous_end = end
        for day in pd.date_range(start.normalize(), end.normalize(), freq='D'):
            if not is_trading_day(day.date()):
                continue
            session = session_info(day.date())
            opening, closing = pd.Timestamp(session.start_utc), pd.Timestamp(session.end_utc)
            for entry in pd.date_range(opening + pd.Timedelta(minutes=30), closing, freq='30min', inclusive='left'):
                signal, exit_at = entry - pd.Timedelta(minutes=1), entry + pd.Timedelta(minutes=30)
                if signal < start or signal > end:
                    rejected['signal_outside_fold'] += 1
                    continue
                if exit_at >= closing:
                    rejected['exit_outside_session'] += 1
                    continue
                if exit_at > end:
                    rejected['exit_outside_fold'] += 1
                    continue
                required = pd.date_range(signal, exit_at, freq='min')
                if not required.isin(frame.index).all():
                    rejected['missing_contiguous_bars'] += 1
                    continue
                values = frame.loc[required, ['open', 'high', 'low', 'close']].to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values <= 0).any():
                    rejected['invalid_price'] += 1
                    continue
                spread = float(features.loc[signal, 'sma_spread'])
                if not np.isfinite(spread):
                    rejected['invalid_feature'] += 1
                    continue
                gross = (float(frame.loc[exit_at, 'open']) / float(frame.loc[entry, 'open']) - 1) * 10000
                rows.append({'symbol': symbol, 'session': day.date().isoformat(), 'fold': fold['fold_index'], 'signal': signal.isoformat(), 'entry': entry.isoformat(), 'exit': exit_at.isoformat(), 'momentum': spread > 0, 'gross_bps': gross})
    return rows, dict(rejected)


def summarize(data: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ('cash', 'always_long', 'momentum'):
        selected = np.zeros(len(data), dtype=bool) if name == 'cash' else np.ones(len(data), dtype=bool) if name == 'always_long' else data['momentum'].to_numpy(dtype=bool)
        gross = data['gross_bps'].to_numpy()[selected]
        result[name] = {'opportunities': len(data), 'selected': int(selected.sum()), 'gross_per_selection_bps': float(gross.mean()) if len(gross) else None, 'net_per_selection_bps': float(gross.mean() - 6) if len(gross) else None, 'gross_per_opportunity_bps': float(gross.sum() / len(data)), 'net_per_opportunity_bps': float((gross - 6).sum() / len(data)), 'break_even_cost_bps': float(gross.mean()) if len(gross) else None, 'stress_net_per_opportunity_bps': {str(cost): float((gross - cost).sum() / len(data)) for cost in (0, 3, 6, 10)}}
    result['paired_momentum_minus_cash_bps'] = result['momentum']['net_per_opportunity_bps']
    result['paired_momentum_minus_always_long_bps'] = result['momentum']['net_per_opportunity_bps'] - result['always_long']['net_per_opportunity_bps']
    return result


def evaluate(data: pd.DataFrame) -> dict[str, Any]:
    if data.empty:
        return {'decision': 'inconclusive', 'reason': 'no_valid_opportunities'}
    aggregate = summarize(data)
    folds = {str(k): summarize(v) for k, v in data.groupby('fold')}
    symbols = {str(k): summarize(v) for k, v in data.groupby('symbol')}
    daily = data.assign(net=np.where(data['momentum'], data['gross_bps'] - 6, 0)).groupby('session').agg(net=('net', 'sum'), count=('net', 'size'))
    rng = np.random.default_rng(20260908)
    indices = rng.integers(0, len(daily), size=(10000, len(daily)))
    samples = daily['net'].to_numpy()[indices].sum(axis=1) / daily['count'].to_numpy()[indices].sum(axis=1)
    interval = np.quantile(samples, [.025, .975]).tolist()
    supported = len(folds) == 5 and all(v['momentum']['selected'] >= 30 for v in folds.values())
    criteria = {'five_supported_folds': supported, 'positive_net_edge': aggregate['momentum']['net_per_opportunity_bps'] > 0, 'positive_bootstrap_lower_bound': interval[0] > 0, 'three_profitable_folds': sum(v['momentum']['net_per_opportunity_bps'] > 0 for v in folds.values()) >= 3, 'beats_always_long': aggregate['paired_momentum_minus_always_long_bps'] > 0, 'two_profitable_symbols': sum(v['momentum']['net_per_opportunity_bps'] > 0 for v in symbols.values()) >= 2}
    return {'decision': 'eligible_for_untouched_test_planning' if all(criteria.values()) else 'retired' if supported else 'inconclusive', 'criteria': criteria, 'aggregate': aggregate, 'folds': folds, 'symbols': symbols, 'session_block_bootstrap_95_interval_bps': interval, 'bootstrap_sessions': len(daily), 'bootstrap_replicates': 10000, 'bootstrap_seed': 20260908}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--acquisition-manifest', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('preserve prior study artifacts: output directory already exists')
    baseline = json.loads(args.benchmark.read_text())
    metadata = {'study_id': 'fixed_momentum_30m_development_v1', 'started_at': datetime.now(UTC).isoformat(), 'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(), 'benchmark_sha256': hashlib.sha256(args.benchmark.read_bytes()).hexdigest(), 'dataset_hash': baseline['dataset_hash'], 'folds': baseline['protocol']['folds'], 'models_fitted': 0, 'holdout_evaluated': False, 'promotion_authority': False, 'metric': 'equal_notional_opportunity_markouts_not_portfolio_returns'}
    atomic_write_text(args.output_dir / 'protocol.md', args.protocol.read_text())
    atomic_write_text(args.output_dir / 'metadata.json', json.dumps(metadata, indent=2))
    data_dir, provenance = _resolve_training_input(argparse.Namespace(acquisition_manifest_json=args.acquisition_manifest, symbols='AAPL,AMZN,MSFT', timestamp_col='timestamp'))
    if provenance.get('quality_passed') is not True or provenance.get('dataset_hash') != baseline['dataset_hash']:
        raise ValueError('governed dataset mismatch')
    quality = validate_training_provenance(data_dir, symbols=['AAPL', 'AMZN', 'MSFT'], dataset_identity=provenance['dataset_identity'], min_sessions=20, max_missing_ratio=.02)
    atomic_write_text(args.output_dir / 'quality.json', json.dumps(quality, indent=2))
    if not quality['quality_passed']:
        raise ValueError('governed dataset quality failed')
    rows, rejections = [], {}
    for symbol in ('AAPL', 'AMZN', 'MSFT'):
        frame, diagnostics = load_historical_bars(data_dir / f'{symbol}.csv', timestamp_col='timestamp', require_timestamp=True)
        frame = frame.loc[frame.index <= pd.Timestamp(metadata['folds'][-1]['test_end'])]
        features = _feature_frame(frame, symbol=symbol)
        selected, rejected = opportunities(frame, features, metadata['folds'], symbol)
        rows.extend(selected)
        rejections[symbol] = {'opportunities': rejected, 'loader': diagnostics.as_dict()}
    data = pd.DataFrame(rows)
    report = evaluate(data)
    report.update(metadata=metadata, rejection_counts=rejections)
    atomic_write_text(args.output_dir / 'opportunities.csv', data.to_csv(index=False))
    atomic_write_text(args.output_dir / 'report.json', json.dumps(report, indent=2) + '\n')
    get_logger(__name__).info('MOMENTUM_HORIZON_STUDY_COMPLETE', extra={'decision': report['decision']})


if __name__ == '__main__':
    main()
