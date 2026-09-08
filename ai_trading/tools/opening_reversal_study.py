"""Run the single registered ETF opening-shock reversal development trial."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.data.historical_bars import load_historical_bars
from ai_trading.data.training_provenance import validate_training_provenance
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.experiment_ledger import claim_campaign_trial, finish_campaign_trial
from ai_trading.tools.train_replay_aligned_model import _resolve_training_input
from ai_trading.utils.market_calendar import is_trading_day, session_info


def build_opportunities(frame: pd.DataFrame, symbol: str, start: str, end: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    prior_ranges: list[float] = []
    for date in pd.date_range(start, end, freq='D'):
        if not is_trading_day(date.date()):
            continue
        session = session_info(date.date())
        expected = pd.date_range(session.start_utc, session.end_utc, freq='min', inclusive='left')
        # Only evaluate sessions with complete minute observations. Never interpolate.
        if not expected.isin(frame.index).all():
            rejected['incomplete_session'] += 1
            continue
        bars = frame.loc[expected]
        prices = bars[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            rejected['invalid_prices'] += 1
            continue
        today_range = (float(bars['high'].max()) - float(bars['low'].min())) / float(bars['close'].iloc[-1])
        if len(prior_ranges) < 20:
            rejected['warmup_complete_sessions'] += 1
        else:
            threshold = -.5 * float(np.median(prior_ranges[-20:]))
            opening_return = float(bars['close'].iloc[29]) / float(bars['open'].iloc[0]) - 1
            gross = (float(bars['open'].iloc[151]) / float(bars['open'].iloc[31]) - 1) * 10000
            rows.append({'symbol': symbol, 'session': date.date().isoformat(), 'fold': str(date.to_period('Q')), 'selected': opening_return <= threshold, 'signal_return': opening_return, 'threshold': threshold, 'entry': expected[31].isoformat(), 'exit': expected[151].isoformat(), 'gross_bps': gross})
        prior_ranges.append(today_range)
    return rows, dict(rejected)


def metrics(data: pd.DataFrame) -> dict[str, Any]:
    selected = data.loc[data['selected'], 'gross_bps']
    net = float((selected - 10).sum() / len(data))
    control = float((data['gross_bps'] - 10).mean())
    return {'opportunities': len(data), 'selected': len(selected), 'net_per_opportunity_bps': net, 'net_per_selection_bps': float(selected.mean() - 10) if len(selected) else None, 'gross_break_even_cost_bps': float(selected.mean()) if len(selected) else None, 'cash_net_bps': 0, 'always_long_net_bps': control, 'paired_vs_cash_bps': net, 'paired_vs_always_long_bps': net - control, 'stress_net_bps': {str(cost): float((selected - cost).sum() / len(data)) for cost in (6, 10, 20)}}


def evaluate(data: pd.DataFrame) -> dict[str, Any]:
    if data.empty:
        return {'decision': 'inconclusive', 'reason': 'no_valid_opportunities'}
    aggregate = metrics(data)
    folds = {str(k): metrics(v) for k, v in data.groupby('fold')}
    symbols = {str(k): metrics(v) for k, v in data.groupby('symbol')}
    daily = data.assign(net=np.where(data['selected'], data['gross_bps'] - 10, 0)).groupby('session').agg(net=('net', 'sum'), count=('net', 'size'))
    rng = np.random.default_rng(20260908)
    indices = rng.integers(0, len(daily), size=(10000, len(daily)))
    estimates = daily['net'].to_numpy()[indices].sum(axis=1) / daily['count'].to_numpy()[indices].sum(axis=1)
    interval = np.quantile(estimates, [.025, .975]).tolist()
    support = len(folds) == 8 and all(v['selected'] >= 30 for v in folds.values())
    criteria = {'eight_supported_folds': support, 'positive_net': aggregate['net_per_opportunity_bps'] > 0, 'positive_lower_bound': interval[0] > 0, 'six_profitable_folds': sum(v['net_per_opportunity_bps'] > 0 for v in folds.values()) >= 6, 'four_profitable_symbols': sum(v['net_per_opportunity_bps'] > 0 for v in symbols.values()) >= 4, 'beats_always_long': aggregate['paired_vs_always_long_bps'] > 0}
    return {'decision': 'eligible_for_untouched_test_planning' if all(criteria.values()) else 'retired' if support else 'inconclusive_budget_consumed', 'aggregate': aggregate, 'folds': folds, 'symbols': symbols, 'criteria': criteria, 'session_bootstrap_95_bps': interval}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-state', type=Path, required=True)
    parser.add_argument('--acquisition', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.campaign_state.read_text())
    contract = state['contract']
    if contract['hypotheses'] != ['etf_opening_shock_reversal_v1'] or contract['primary_round_trip_cost_bps'] != 10:
        raise ValueError('unsupported study contract')
    data_dir, provenance = _resolve_training_input(argparse.Namespace(acquisition_manifest_json=args.acquisition, symbols=','.join(contract['symbols']), timestamp_col='timestamp'))
    identity = provenance['dataset_identity']
    interval = identity['request_interval']
    if interval['start_date'] != contract['development_start'] or interval['end_date'] != contract['development_end'] or identity['feed'] != contract['feed'] or identity['adjustment'] != contract['adjustment']:
        raise ValueError('acquisition differs from frozen development contract')
    quality = validate_training_provenance(data_dir, symbols=contract['symbols'], dataset_identity=identity, min_sessions=400, max_missing_ratio=.02)
    atomic_write_text(args.output_dir / 'quality.json', json.dumps(quality, indent=2))
    if provenance.get('quality_passed') is not True or not quality['quality_passed']:
        atomic_write_text(args.output_dir / 'report.json', json.dumps({'decision': 'blocked_data_quality', 'budget_consumed': False, 'quality': quality}, indent=2))
        return
    claim_campaign_trial(args.campaign_state, hypothesis_id=contract['hypotheses'][0], evidence_signature=provenance['dataset_hash'], evaluation_start=contract['development_start'], evaluation_end=contract['development_end'], quality_passed=True)
    rows, rejections = [], {}
    for symbol in contract['symbols']:
        frame, _ = load_historical_bars(data_dir / f'{symbol}.csv', timestamp_col='timestamp', require_timestamp=True)
        selected, rejected = build_opportunities(frame, symbol, contract['development_start'], contract['development_end'])
        rows.extend(selected)
        rejections[symbol] = rejected
    data = pd.DataFrame(rows)
    report = evaluate(data)
    report.update(dataset_hash=provenance['dataset_hash'], contract=contract, rejection_counts=rejections, models_fitted=0, holdout_evaluated=False, promotion_authority=False, metric='equal_notional_markouts_not_portfolio_returns')
    atomic_write_text(args.output_dir / 'opportunities.csv', data.to_csv(index=False))
    atomic_write_text(args.output_dir / 'report.json', json.dumps(report, indent=2))
    finish_campaign_trial(args.campaign_state, evidence_signature=provenance['dataset_hash'], decision=report['decision'], report_path=args.output_dir / 'report.json')
    get_logger(__name__).info('OPENING_REVERSAL_STUDY_COMPLETE', extra={'decision': report['decision']})


if __name__ == '__main__':
    main()
