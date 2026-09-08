import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from ai_trading.tools.experiment_ledger import amend_campaign_feed, claim_campaign_trial, finish_campaign_trial, register_campaign
from ai_trading.tools.research_foundation import audit_bars, audit_quotes


def test_quotes_are_deduplicated_and_not_certified_execution_costs():
    now = datetime(2026, 9, 8, tzinfo=UTC)
    row = {'symbol': 'SPY', 'metrics': {'quote_timestamp': '2026-09-07T20:00:00Z', 'decision_ts': '2026-09-07T19:59:00Z', 'raw_quote_bid': 99.99, 'raw_quote_ask': 100.01, 'quote_age_ms': 100, 'opportunity_quantity': 10, 'opportunity_price': 100, 'liquidity_regime': 'thick'}}
    report = audit_quotes([row, row, {}], now)
    assert report['rows_used'] == 1
    assert report['rejection_counts'] == {'duplicate_quote': 1, 'missing_or_invalid_quote': 1}
    assert report['decision_alignment_counts']['decision_time_alignment_unverified'] == 1
    assert report['buckets'][0]['spread_p50_bps'] == pytest.approx(2)
    assert report['buckets'][0]['research_round_trip_scenarios_bps']['median_spread_plus_buffers'] == pytest.approx(8)
    assert report['fee_buffer_observed'] is False
    row['metrics']['decision_ts'] = '2026-09-07T20:00:00.500Z'
    row['metrics']['decision_ts_basis'] = 'record_capture'
    assert audit_quotes([row], now)['decision_alignment_counts'] == {'aligned_to_record_capture_only': 1}
    row['metrics']['decision_ts_basis'] = 'explicit'
    assert audit_quotes([row], now)['decision_alignment_counts'] == {'aligned_to_explicit_decision': 1}
    row['metrics']['recorded_at'] = '2026-09-07T20:00:06Z'
    assert audit_quotes([row], now)['rejection_counts'] == {'quote_age_over_1000ms': 1}


def test_bar_audit_finds_invalid_ohlc_and_duplicates():
    frame = pd.DataFrame({'timestamp': ['2025-01-02T15:00Z'] * 2, 'open': [10, 10], 'close': [10, 15], 'low': [9, 9], 'high': [11, 11]})
    result = audit_bars(frame)
    assert result['duplicate_timestamps'] == 1
    assert result['invalid_ohlc_rows'] == 1
    assert result['adjustments'] == ['unknown']


def test_campaign_is_immutable_and_budget_precedes_evaluation(tmp_path):
    path = tmp_path / 'campaign.json'
    contract = {'max_trials': 1, 'hypotheses': ['reversal'], 'development_start': '2024-01-01', 'development_end': '2025-12-31', 'holdout_start': '2026-09-09', 'holdout_end': '2026-12-08', 'feed': 'iex'}
    register_campaign(path, contract)
    with pytest.raises(ValueError, match='immutable'):
        register_campaign(path, dict(contract, hypotheses=['different']))
    args = {'hypothesis_id': 'reversal', 'evidence_signature': 'abc', 'evaluation_start': '2024-01-01', 'evaluation_end': '2025-12-31', 'quality_passed': True}
    with pytest.raises(ValueError, match='outside'):
        claim_campaign_trial(path, **dict(args, evaluation_end='2026-09-09'))
    with pytest.raises(ValueError, match='quality'):
        claim_campaign_trial(path, **dict(args, quality_passed=False))
    assert json.loads(path.read_text())['trials'] == []
    amend_campaign_feed(path, feed='sip', reason='IEX failed completeness; no outcomes viewed')
    assert json.loads(path.read_text())['amendments'][0]['previous_contract']['feed'] == 'iex'
    claim_campaign_trial(path, **args)
    with pytest.raises(ValueError, match='unevaluated'):
        amend_campaign_feed(path, feed='sip', reason='retry')
    report = tmp_path / 'report.json'
    report.write_text('{"decision":"retired"}')
    finish_campaign_trial(path, evidence_signature='abc', decision='retired', report_path=report)
    assert json.loads(path.read_text())['trials'][0]['status'] == 'retired'
    with pytest.raises(ValueError, match='exhausted'):
        claim_campaign_trial(path, **dict(args, evidence_signature='different'))
