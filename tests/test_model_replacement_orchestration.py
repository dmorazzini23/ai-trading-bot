"""Exercise trial orchestration with synthetic rows and no estimator fitting."""
import json
from pathlib import Path

import pandas as pd
import pytest

from ai_trading.tools import model_replacement_trial as trial


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[1] / 'config/model_replacement_campaign.json'
    contract = json.loads(original.read_text())
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/model_replacement_campaign.json').write_text(original.read_text())
    directory = tmp_path / 'artifacts/model_replacement_20260921'
    directory.mkdir(parents=True)
    state_path = directory / 'campaign_state.json'
    trial.register_campaign(state_path, contract)
    source = tmp_path / 'source.csv'
    source.write_text('timestamp,open,high,low,close,volume\n2024-01-02T15:00:00Z,100,101,99,100,1000\n')
    provenance = dict(sources={s: dict(path=str(source), sha256=trial.digest(source)) for s in contract['symbols']})
    monkeypatch.setattr(trial, 'load_governed_development_timestamps', lambda *a: (None, provenance))
    monkeypatch.setattr(trial, 'regular_bars', lambda minutes, days: minutes)
    builds = []
    def build(minutes, bars, *, symbol):
        builds.append(symbol)
        return pd.DataFrame([dict(symbol=symbol, decision_at=pd.Timestamp('2024-01-02T15:05:02Z'))])
    monkeypatch.setattr(trial, 'labeled_rows', build)
    monkeypatch.setattr(trial, 'validate_rows', lambda *a: None)
    monkeypatch.setattr(trial, 'splits', lambda *a: [])
    def evaluate(rows, folds):
        return dict(status='hypothesis_rejected', promotion_authority=False,
                    _predictions=rows.copy())
    monkeypatch.setattr(trial, 'evaluate', evaluate)
    return tmp_path, directory, state_path, builds, source


def test_preflight_cache_reuse_and_corruption_rebuild(campaign):
    root, directory, state, builds, _ = campaign
    result = trial.run(root, fit=False)
    assert result['status'] == 'preflight_passed'
    assert result['trial_claimed'] is False
    assert json.loads(state.read_text())['trials'] == []
    assert len(builds) == 3
    trial.run(root, fit=False)
    assert len(builds) == 3
    (directory / 'AAPL.parquet').write_bytes(b'corrupt cache')
    trial.run(root, fit=False)
    assert builds.count('AAPL') == 2
    assert len(builds) == 4


def test_trial_claim_report_and_reentry_never_refit(campaign, monkeypatch):
    root, directory, state, _, _ = campaign
    result = trial.run(root, fit=True)
    assert result['status'] == 'hypothesis_rejected'
    assert result['predictions_sha256'] == trial.digest(directory / 'oof_predictions.parquet')
    assert json.loads((directory / 'report.json').read_text()) == result
    assert len(json.loads(state.read_text())['trials']) == 1
    monkeypatch.setattr(trial, 'evaluate', lambda *a: pytest.fail('budget must prevent refit'))
    assert trial.run(root, fit=True)['status'] == 'budget_already_claimed'


def test_evaluation_crash_consumes_claim_and_prevents_automatic_retry(campaign, monkeypatch):
    root, directory, state, _, _ = campaign
    def fail(*a):
        raise RuntimeError('evaluation interrupted')
    monkeypatch.setattr(trial, 'evaluate', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        trial.run(root, fit=True)
    assert len(json.loads(state.read_text())['trials']) == 1
    assert not (directory / 'report.json').exists()
    assert trial.run(root, fit=True)['status'] == 'budget_already_claimed'


@pytest.mark.parametrize('change', ['contract', 'registration', 'source'])
def test_provenance_failures_prevent_claim(campaign, change):
    root, _, state, _, source = campaign
    if change == 'contract':
        (root / 'config/model_replacement_campaign.json').write_text('{}')
    elif change == 'registration':
        state.unlink()
    else:
        source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError):
        trial.run(root, fit=True)
    if state.exists():
        assert json.loads(state.read_text())['trials'] == []
