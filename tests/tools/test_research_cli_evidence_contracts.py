import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from ai_trading.tools import baseline_benchmark as baseline, research_foundation as foundation
from ai_trading.tools.train_replay_aligned_model import _build_parser


def test_training_parser_requires_one_input_and_preserves_explicit_holdout_controls(tmp_path):
    parser = _build_parser()
    options = ['--output-dir', str(tmp_path), '--no-evaluate-holdout', '--no-research-experiments']
    args = parser.parse_args(['--acquisition-manifest-json', 'manifest.json', *options])
    assert args.acquisition_manifest_json == Path('manifest.json')
    assert args.evaluate_holdout is False and args.research_experiments is False
    assert args.allow_research_synthetic_timestamps is False
    assert args.walk_forward_min_trades == 250
    assert args.walk_forward_folds == 5
    with pytest.raises(SystemExit):
        parser.parse_args(options)
    with pytest.raises(SystemExit):
        parser.parse_args(['--data-dir', 'data', '--acquisition-manifest-json', 'manifest.json', *options])


def test_baseline_cli_writes_derived_report_without_mutating_saved_candidate(tmp_path, monkeypatch):
    costs = dict(version='static_fee_spread_slippage_v1', fee_bps=1., slippage_bps=2.)
    candidate = dict(model_name='test', horizon_bars=1, label_objective='net_markout',
        acquisition={'dataset_hash': 'data'}, walk_forward={'folds': [dict(fold_index=0,
        test_start='2024-01-02T15:00Z', test_end='2024-01-02T15:01Z', test_rows=2,
        selected_candidates=1, total_post_cost_net_edge_bps=-2., chronological_non_overlap=True,
        label_purge_ok=True, cost_model=costs)]})
    pipeline = tmp_path / 'pipeline.json'
    pipeline.write_text(json.dumps({'candidates': [candidate], 'config': {'symbols': 'AAPL'}}))
    original = pipeline.read_bytes()
    dataset = pd.DataFrame({'timestamp': pd.to_datetime(['2024-01-02T15:00Z', '2024-01-02T15:01Z']),
        'gross_long_bps': [4., -1.], 'net_long_bps': [-2., -7.], 'round_trip_cost_bps': [6., 6.], 'sma_spread': [1., -1.]})
    monkeypatch.setattr(baseline, '_resolve_training_input', lambda *a: (tmp_path,
        {'quality_passed': True, 'dataset_hash': 'data', 'dataset_identity': {}}))
    monkeypatch.setattr(baseline, 'validate_training_provenance', lambda *a, **k: {'quality_passed': True})
    monkeypatch.setattr(baseline, 'build_training_dataset', lambda **k: dataset)
    output = tmp_path / 'output'
    monkeypatch.setattr('sys.argv', ['benchmark', '--pipeline-report', str(pipeline), '--model-id', 'test',
        '--acquisition-manifest', str(tmp_path / 'manifest'), '--output-dir', str(output)])
    baseline.main()
    result = json.loads((output / 'baseline_benchmark.json').read_text())
    assert result['promotion_authority'] is False and result['models_fitted'] == 0
    assert result['holdout_evaluated'] is False
    assert pipeline.read_bytes() == original
    derived = json.loads((output / 'pipeline_with_baselines.json').read_text())
    assert derived['candidates'][0]['walk_forward']['controlled_comparisons'] == result['controlled_comparisons']
    assert result['protocol']['source_sha256'] == hashlib.sha256(original).hexdigest()


def test_foundation_cli_records_malformed_quotes_and_does_not_claim_trial(tmp_path, monkeypatch):
    source = tmp_path / 'bars.csv'
    source.write_text('timestamp,open,high,low,close\n2024-01-02T15:00Z,100,101,99,100\n')
    inputs = {
        'quotes': '{}\nmalformed\n[]\n',
        'cost-model': json.dumps({'sources': []}),
        'paper-review': json.dumps({'execution_cost_comparison': {}, 'completion_gaps': ['missing_evidence']}),
        'manifest': json.dumps({'symbols': [{'symbol': 'AAPL', 'csv_path': str(source),
            'content_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]}),
        'campaign': (Path(__file__).resolve().parents[2] / 'config/model_replacement_campaign.json').read_text(),
    }
    argv = ['foundation']
    for name, value in inputs.items():
        path = tmp_path / (name + '.json')
        path.write_text(value)
        argv += ['--' + name, str(path)]
    output = tmp_path / 'output'
    monkeypatch.setattr('sys.argv', argv + ['--output-dir', str(output)])
    foundation.main()
    report = json.loads((output / 'foundation_audit.json').read_text())
    assert report['quote_costs']['malformed_json_rows'] == 2
    assert report['quote_costs']['rows_used'] == 0
    assert report['accounting_gaps'] == ['missing_evidence']
    assert report['bars']['AAPL']['content_hash_matches_manifest'] is True
    assert report['promotion_authority'] is False and report['orders_sent'] == 0
    assert json.loads((output / 'campaign_state.json').read_text())['trials'] == []
