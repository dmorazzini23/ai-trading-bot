import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ai_trading.research import focused_cycle as study
from ai_trading.tools import train_replay_aligned_model as trainer


def test_synthetic_study_preserves_sources_excludes_holdout_and_stays_unqualified(tmp_path, monkeypatch):
    index = pd.date_range('2024-01-02T14:30Z', periods=390, freq='min').append(
        pd.date_range('2024-01-03T14:30Z', periods=390, freq='min')).append(
        pd.date_range('2024-01-04T14:30Z', periods=390, freq='min'))
    prices = 100 + np.arange(len(index)) * .001
    bars = pd.DataFrame(dict(open=prices, high=prices+1, low=prices-1, close=prices, volume=1000), index=index)
    source = tmp_path / 'AAPL.csv'
    bars.rename_axis('timestamp').to_csv(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'dataset_identity': {'request_interval': {'end_date': '2024-01-04'}},
        'quality_passed': True, 'symbols': [{'symbol': 'AAPL', 'content_sha256': digest, 'quality_passed': True}]}))
    protocol = {'universe': ['AAPL'], 'horizons_minutes': [15], 'entry_start': '10:00', 'flat_by': '15:45',
        'untouched_final_period': {'start': '2024-01-04'}, 'one_way_cost_scenarios_bps': [6.],
        'retrospective_intervals': [{'name': 'development', 'start': '2024-01-02', 'end': '2024-01-02'},
                                  {'name': 'validation', 'start': '2024-01-03', 'end': '2024-01-03'}],
        'acceptance': {'primary_one_way_cost_bps': 6., 'minimum_validation_sessions_per_interval': 20,
            'minimum_completed_validation_trades': 100, 'positive_validation_intervals_required': 1,
            'bootstrap_block_sessions': 5, 'bootstrap_replicates': 10, 'bootstrap_seed': 1,
            'familywise_alpha': .05, 'simultaneous_bounds': 1}}
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    # Synthetic fixture only: production permission and quality gates have separate tests.
    monkeypatch.setattr(study, 'check_study_permission', lambda *a: None)
    monkeypatch.setattr(study, 'validate_training_provenance', lambda *a, **k:
        {'quality_passed': True, 'coverage': {'AAPL': {'observed_session_dates': ['2024-01-02', '2024-01-03']}}})
    monkeypatch.setattr(trainer, '_feature_frame', lambda frame, **k:
        pd.DataFrame({'sma_spread': 1., 'macd_signal_gap': 1.}, index=frame.index))
    output = tmp_path / 'output'
    report = study.run_study(protocol_path, [manifest], output)
    assert report['promotion_authority'] is False
    assert report['untouched_final_period']['consumed'] is False
    assert report['evaluations'][0]['disposition'] == 'inconclusive_do_not_promote'
    assert report['decision_reason'] == 'insufficient_support'
    saved = pd.read_csv(output / 'data/AAPL.csv')
    assert pd.to_datetime(saved.timestamp, utc=True).max() < pd.Timestamp('2024-01-04', tz='UTC')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert json.loads((output / 'study_report.json').read_text())['protocol_sha256'] == report['protocol_sha256']
    source.write_text('changed')
    with pytest.raises(ValueError, match='identity'):
        study.run_study(protocol_path, [manifest], tmp_path / 'other-output')
