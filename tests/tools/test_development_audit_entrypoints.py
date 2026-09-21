import hashlib
import json
from datetime import date

import pandas as pd
import pytest

from ai_trading.tools import five_minute_coverage, stock_development_readiness, stock_feature_parity


@pytest.fixture
def source(tmp_path):
    index = pd.date_range('2024-01-02T14:30Z', periods=390, freq='min')
    path = tmp_path / 'bars.csv'
    pd.DataFrame(dict(timestamp=index, open=100., high=101., low=99., close=100., volume=1000)).to_csv(path, index=False)
    return path, dict(sources={'AAPL': dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())},
                      acquisition_sha256='acquisition', manifest_sha256='manifest')


@pytest.mark.parametrize('name', ['readiness', 'parity', 'coverage'])
def test_audit_entrypoints_keep_timestamp_features_and_execution_claims_separate(tmp_path, monkeypatch, source, name):
    path, provenance = source
    module = {'readiness': stock_development_readiness, 'parity': stock_feature_parity,
              'coverage': five_minute_coverage}[name]
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'is_trading_day', lambda day: day == date(2024, 1, 2))
    if name == 'coverage':
        (tmp_path / 'config').mkdir()
        (tmp_path / 'config/slower_horizon_campaign.json').write_text(json.dumps({'symbols': ['AAPL']}))
        for folder in ['research_reset', 'research_foundation']:
            directory = tmp_path / 'artifacts' / folder
            directory.mkdir(parents=True)
            (directory / 'campaign_state.json').write_text('{"trials":[]}')
        monkeypatch.setattr(module, 'audit_sampling_acquisition', lambda *a: provenance)
        output = tmp_path / 'artifacts/research_reset/five_minute_coverage.json'
    else:
        monkeypatch.setattr(module, 'load_governed_development_timestamps', lambda *a: (None, provenance))
        output = tmp_path / ('artifacts/stock_development/' + ('readiness.json' if name == 'readiness' else 'feature_parity.json'))
        if name == 'parity':
            def compare(bars, *, symbol, end):
                assert len(bars) == 78 and symbol == 'AAPL'
                assert bars.volume.iloc[0] == 5000
                return {'symbol': symbol, 'end': end, 'status': 'synthetic_entrypoint_test'}
            monkeypatch.setattr(module, 'compare_history', compare)
    module.main()
    report = json.loads(output.read_text())
    assert report['promotion_authority'] is False
    assert report['holdout_evaluated'] is False
    assert report['trial_claimed'] is False
    assert report['returns_computed'] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance['sources']['AAPL']['sha256']
    if name == 'readiness':
        assert report['execution_verified'] is False
        assert report['feature_parity_verified'] is False
    elif name == 'coverage':
        assert report['prices_used'] is False
        assert report['totals']['AAPL']['complete_slots'] > 0
        assert (tmp_path / 'artifacts/research_reset/campaign_state.json').read_text() == '{"trials":[]}'
    else:
        assert report['live_history_equivalence_verified'] is False
        assert len(report['results']) == 3


@pytest.mark.parametrize('module', [stock_development_readiness, stock_feature_parity])
def test_modified_source_is_rejected_before_audit(tmp_path, monkeypatch, source, module):
    path, provenance = source
    path.write_text(path.read_text() + '\n')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'load_governed_development_timestamps', lambda *a: (None, provenance))
    with pytest.raises(ValueError, match='source'):
        module.main()
    assert not (tmp_path / 'artifacts').exists()
