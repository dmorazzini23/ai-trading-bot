"""Audit research assumptions and register a bounded, development-only campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.experiment_ledger import register_campaign


def audit_quotes(rows: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    """Estimate observed spread distributions, never certify fees or impact."""
    rejected: Counter[str] = Counter()
    alignment: Counter[str] = Counter()
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    seen = set()
    for row in rows:
        m = row.get('metrics') or {}
        try:
            ts = pd.Timestamp(m['quote_timestamp'])
            if ts.tzinfo is None or pd.isna(ts):
                raise ValueError('timezone missing')
            bid, ask = float(m['raw_quote_bid']), float(m['raw_quote_ask'])
            age = float(m['quote_age_ms'])
            if m.get('recorded_at'):
                recorded = pd.Timestamp(m['recorded_at'])
                elapsed_ms = (recorded - ts).total_seconds() * 1000
                if not np.isfinite(elapsed_ms) or elapsed_ms < 0:
                    raise ValueError('quote later than record capture')
                age = max(age, elapsed_ms)
            if not all(np.isfinite(v) for v in (bid, ask, age)) or bid <= 0 or ask < bid or age < 0:
                raise ValueError('invalid quote')
        except (KeyError, ValueError, TypeError):
            rejected['missing_or_invalid_quote'] += 1
            continue
        if ts < now - timedelta(days=7) or ts > now:
            rejected['outside_seven_day_window'] += 1
            continue
        if age > 1000:
            rejected['quote_age_over_1000ms'] += 1
            continue
        symbol = str(row.get('symbol') or '').upper()
        if not symbol:
            rejected['symbol_missing'] += 1
            continue
        identity = (symbol, ts.isoformat(), bid, ask)
        if identity in seen:
            rejected['duplicate_quote'] += 1
            continue
        seen.add(identity)
        try:
            decision = pd.Timestamp(m['decision_ts'])
            causal = decision.tzinfo is not None and 0 <= (decision - ts).total_seconds() <= 1
        except (KeyError, ValueError, TypeError):
            causal = False
        basis = m.get('decision_ts_basis')
        alignment['aligned_to_explicit_decision' if causal and basis == 'explicit' else 'aligned_to_record_capture_only' if causal and basis == 'record_capture' else 'decision_time_alignment_unverified'] += 1
        try:
            notional = abs(float(m['opportunity_quantity']) * float(m['opportunity_price']))
            size = 'unknown' if not np.isfinite(notional) or notional <= 0 else 'up_to_1000' if notional <= 1000 else '1000_to_10000' if notional <= 10000 else '10000_to_25000' if notional <= 25000 else 'above_25000'
        except (KeyError, ValueError, TypeError):
            size = 'unknown'
        spread = (ask - bid) / ((ask + bid) / 2) * 10000
        liquidity = str(m.get('liquidity_regime') or 'unknown').lower()
        buckets[(symbol, liquidity, size)].append(spread)
    results = []
    for (symbol, liquidity, size), spreads in sorted(buckets.items()):
        p50, p90 = np.quantile(spreads, [.5, .9]).tolist()
        results.append({'symbol': symbol, 'liquidity': liquidity, 'intended_notional_bucket_usd': size, 'samples': len(spreads), 'spread_p50_bps': p50, 'spread_p90_bps': p90, 'research_round_trip_scenarios_bps': {'median_spread_plus_buffers': p50 + 6, 'p90_spread_plus_buffers': p90 + 6}, 'sufficient_quote_support': len(spreads) >= 30})
    return {'rows_read': len(rows), 'rows_used': sum(len(v) for v in buckets.values()), 'rejection_counts': dict(rejected), 'decision_alignment_counts': dict(alignment), 'buckets': results, 'cost_basis': 'one_full_spread_per_round_trip_plus_4bps_slippage_and_2bps_fee_assumptions', 'fee_buffer_observed': False, 'impact_calibrated': False, 'size_measures': 'intended_notional_not_displayed_depth_or_realized_impact', 'evidence_type': 'quote_observation_research_scenarios_not_execution_validation', 'promotion_authority': False}


def audit_bars(frame: pd.DataFrame) -> dict[str, Any]:
    """Inspect raw rows without silently cleaning quality defects."""
    ts = pd.to_datetime(frame['timestamp'], utc=True, errors='coerce')
    prices = frame[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
    valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    valid &= (prices['low'] <= prices[['open', 'close']].min(axis=1)) & (prices['high'] >= prices[['open', 'close']].max(axis=1))
    dates = ts.dt.tz_convert('America/New_York').dt.date
    overnight = (dates != dates.shift()) & (prices['open'] / prices['close'].shift() - 1).abs().gt(.2)
    return {'rows': len(frame), 'invalid_timestamps': int(ts.isna().sum()), 'duplicate_timestamps': int(ts.duplicated().sum()), 'nonmonotonic_timestamps': not ts.is_monotonic_increasing, 'invalid_ohlc_rows': int((~valid).sum()), 'overnight_moves_over_20pct': int(overnight.sum()), 'corporate_action_interpretation': 'discontinuity_screen_only_not_verified_event_history', 'adjustments': sorted(frame['adjustment'].dropna().unique().tolist()) if 'adjustment' in frame else ['unknown'], 'feeds': sorted(frame['feed'].dropna().unique().tolist()) if 'feed' in frame else ['unknown']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quotes', type=Path, required=True)
    parser.add_argument('--cost-model', type=Path, required=True)
    parser.add_argument('--paper-review', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    malformed = 0
    with args.quotes.open() as stream:
        for line in stream:
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('not object')
                rows.append(value)
            except (ValueError, TypeError):
                malformed += 1
    now = datetime.now(UTC)
    quote_audit = audit_quotes(rows, now)
    quote_audit['malformed_json_rows'] = malformed
    manifest = json.loads(args.manifest.read_text())
    bars = {}
    for source in manifest['symbols']:
        path = Path(source['csv_path'])
        bars[source['symbol']] = audit_bars(pd.read_csv(path))
        bars[source['symbol']]['content_hash_matches_manifest'] = hashlib.sha256(path.read_bytes()).hexdigest() == source['content_sha256']
    cost = json.loads(args.cost_model.read_text())
    paper = json.loads(args.paper_review.read_text())
    campaign = json.loads(args.campaign.read_text())
    register_campaign(args.output_dir / 'campaign_state.json', campaign)
    report = {'generated_at': now.isoformat(), 'quote_costs': quote_audit, 'execution_sources': cost['sources'], 'execution_validation': paper['execution_cost_comparison'], 'accounting_gaps': paper['completion_gaps'], 'bars': bars, 'label_conventions': {'existing_training': 'same_close_to_future_close_research_label_not_executable_entry', 'momentum_30m': 'completed_bar_signal_next_open_entry_30min_open_exit', 'bar_timestamp': 'start_of_interval', 'corporate_actions': 'existing_raw_prices_need_event_history_for_cross_day_features; expanded_data_requests_split_adjustment', 'fee_contract': 'assumed_research_buffers_separate_from_verified_per_fill_fees'}, 'campaign': campaign, 'promotion_authority': False, 'orders_sent': 0}
    atomic_write_text(args.output_dir / 'foundation_audit.json', json.dumps(report, indent=2) + '\n')
    get_logger(__name__).info('RESEARCH_FOUNDATION_AUDIT_COMPLETE', extra={'quote_rows': quote_audit['rows_used']})


if __name__ == '__main__':
    main()
