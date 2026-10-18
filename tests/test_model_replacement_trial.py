from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_trading.models.contracts import DAY_SLEEVE_ML_FEATURE_COLUMNS
from ai_trading.tools import model_replacement_trial as trial


def minute_data(days):
    parts = []
    for day in days:
        info = trial.session_info(pd.Timestamp(day).date())
        index = pd.date_range(info.start_utc, info.end_utc, freq='min', inclusive='left')
        close = 100 + np.arange(len(index)) / 1000
        parts.append(pd.DataFrame(dict(open=close, high=close + 1, low=close - 1,
                                       close=close, volume=1000), index=index))
    return pd.concat(parts)


def test_complete_sessions_include_early_close_and_reject_gaps():
    days = ['2024-07-03', '2024-07-05']
    minutes = minute_data(days)
    bars = trial.regular_bars(minutes, days)
    assert len(bars) == 42 + 78
    with pytest.raises(ValueError, match='incomplete'):
        trial.regular_bars(minutes.drop(minutes.index[2]), days)
    with pytest.raises(ValueError, match='unique'):
        trial.regular_bars(pd.concat([minutes, minutes.iloc[:1]]), days)


def test_labels_use_future_opens_and_runtime_features_only(monkeypatch):
    days = ['2024-01-02', '2024-01-03', '2024-01-04']
    minutes = minute_data(days)
    bars = trial.regular_bars(minutes, days)
    windows = []

    def features(window):
        windows.append(window.copy())
        return pd.DataFrame([{c: float(window.close.iloc[-1]) for c in DAY_SLEEVE_ML_FEATURE_COLUMNS}])

    monkeypatch.setattr(trial, 'build_day_sleeve_features', features)
    rows = trial.labeled_rows(minutes, bars, symbol='AAPL')
    contract = dict(symbols=['AAPL'], development_start='2024-01-01', development_end='2025-12-31')
    trial.validate_rows(rows, contract)
    assert rows.entry_at.iloc[0] == rows.bar_start.iloc[0] + pd.Timedelta(minutes=6)
    assert rows.exit_at.iloc[0] == rows.bar_start.iloc[0] + pd.Timedelta(minutes=11)
    assert windows[0].index[-1] == rows.bar_start.iloc[0]
    assert rows.iloc[0].entry_price == minutes.loc[rows.entry_at.iloc[0], 'open']
    changed = rows.copy()
    changed.loc[changed.index[0], 'entry_at'] = changed.decision_at.iloc[0]
    with pytest.raises(ValueError, match='entry'):
        trial.validate_rows(changed, contract)


def fold_rows():
    days = [str(d.date()) for d in pd.date_range('2024-01-02', '2024-04-30')
            if trial.is_trading_day(d.date())][:60]
    return pd.DataFrame([dict(session=day, exit_at=pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=20),
                              label=i % 2) for day in days for i in range(30)])


def test_expanding_folds_embargo_whole_session_and_no_test_reuse():
    rows = fold_rows()
    folds = trial.splits(rows)
    assert len(folds) == 5
    seen = set()
    days = sorted(rows.session.unique())
    for train, test in folds:
        first = days.index(rows.iloc[test].session.min())
        assert rows.iloc[train].session.max() == days[first - 2]
        assert not set(test) & seen
        assert not set(train) & set(test)
        seen.update(test)
    rows['label'] = 0
    with pytest.raises(ValueError, match='unsupported fold'):
        trial.splits(rows)


def test_bootstrap_keeps_session_clusters_and_cash_zero():
    rows = pd.DataFrame({'session': ['a', 'a', 'b', 'b']})
    assert trial.bootstrap_lower(rows, np.zeros(4)) == 0
    assert trial.bootstrap_lower(rows, np.full(4, -2.)) == -2
    assert trial.bootstrap_lower(rows, np.ones(4)) == 1


def test_evidence_signature_ignores_audit_time_but_tracks_source_and_code():
    source = {'generated_at': 'first', 'sources': {'AAPL': {'sha256': 'abc'}}}
    signature = trial.evidence_signature('contract', source, {'builder': 'one'})
    assert signature == trial.evidence_signature('contract', {**source, 'generated_at': 'later'}, {'builder': 'one'})
    assert signature != trial.evidence_signature('contract', {**source, 'sources': {}}, {'builder': 'one'})
    assert signature != trial.evidence_signature('contract', source, {'builder': 'two'})


def test_contract_mutation_blocks_before_data_read(tmp_path, monkeypatch):
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/model_replacement_campaign.json').write_text('{}')
    monkeypatch.setattr(trial, 'load_governed_development_timestamps', lambda *a: pytest.fail('data read'))
    with pytest.raises(ValueError, match='registered specification'):
        trial.run(tmp_path, fit=True)


def test_claimed_trial_blocks_fitting_and_data_reads(tmp_path, monkeypatch):
    contract = json.loads((Path(__file__).resolve().parents[1] / 'config/model_replacement_campaign.json').read_text())
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/model_replacement_campaign.json').write_text(json.dumps(contract))
    state_path = tmp_path / 'artifacts/model_replacement_20260921/campaign_state.json'
    trial.register_campaign(state_path, contract)
    trial.claim_campaign_trial(state_path, hypothesis_id=contract['hypotheses'][0], evidence_signature='test',
                               evaluation_start='2024-01-01', evaluation_end='2025-12-31', quality_passed=True)
    monkeypatch.setattr(trial, 'load_governed_development_timestamps', lambda *a: pytest.fail('data read'))
    monkeypatch.setattr(trial, 'evaluate', lambda *a: pytest.fail('refit'))
    assert trial.run(tmp_path, fit=True)['status'] == 'budget_already_claimed'


def test_evaluation_fits_only_train_and_retains_abstention_opportunities(monkeypatch):
    import sklearn.pipeline

    rows = fold_rows()
    rows['net_bps'] = np.where(rows.label == 1, 5., -10.)
    for column in DAY_SLEEVE_ML_FEATURE_COLUMNS:
        rows[column] = np.arange(len(rows), dtype=float)
    folds = trial.splits(rows)
    fits = []

    class AbstainingModel:
        def fit(self, features, labels):
            fits.append(features.index.to_numpy())
            assert features.index.equals(labels.index)

        def predict_proba(self, features):
            assert not set(features.index) & set(fits[-1])
            return np.tile([.9, .1], (len(features), 1))

    monkeypatch.setattr(sklearn.pipeline, 'make_pipeline', lambda *a: AbstainingModel())
    report = trial.evaluate(rows, folds)
    for fitted, (train, _) in zip(fits, folds):
        np.testing.assert_array_equal(fitted, train)
    assert report['selected_trades'] == 0
    assert report['mean_net_bps_per_opportunity'] == 0
    assert report['status'] == 'inconclusive_budget_consumed'
    assert report['oof_rows'] == sum(len(test) for _, test in folds)
    assert not report['promotion_authority']


def legacy_labeled_rows(minutes, bars, *, symbol):
    """Verbatim pre-contract labeling loop, kept to prove the Sept 21 rows are unchanged."""
    rows = []
    for i, stamp in enumerate(bars.index):
        decision = stamp + pd.Timedelta(minutes=5, seconds=2)
        entry = decision.ceil('min')
        exit_at = entry + pd.Timedelta(minutes=5)
        session = trial.session_info(stamp.tz_convert('America/New_York').date())
        if exit_at >= pd.Timestamp(session.end_utc):
            continue
        start = bars.index.searchsorted(decision - pd.Timedelta(days=10))
        history = bars.iloc[start:i + 1]
        if len(history) < 200:
            continue
        features = trial.build_day_sleeve_features(history).iloc[0]
        entry_price, exit_price = float(minutes.loc[entry, 'open']), float(minutes.loc[exit_at, 'open'])
        gross = (exit_price / entry_price - 1) * 10000
        rows.append(dict(symbol=symbol, session=str(stamp.tz_convert('America/New_York').date()),
                         feature_start=history.index[0], bar_start=stamp, decision_at=decision,
                         entry_at=entry, exit_at=exit_at, entry_price=entry_price, exit_price=exit_price,
                         gross_bps=gross, net_bps=gross - 10.0, label=int(gross > 10.0),
                         **{c: float(features[c]) for c in DAY_SLEEVE_ML_FEATURE_COLUMNS}))
    return pd.DataFrame(rows)


REPO_CONTRACT = Path(__file__).resolve().parents[1] / 'config/model_replacement_campaign.json'


def horizon_contract(**changes):
    contract = json.loads(REPO_CONTRACT.read_text())
    contract.update(campaign_id='horizon_trial_test', exit_horizon_minutes=60, flatten_cutoff_et='15:55',
                    classifier={'family': 'hist_gradient_boosting',
                                'params': {'max_iter': 200, 'learning_rate': 0.05, 'max_leaf_nodes': 15,
                                           'min_samples_leaf': 200, 'l2_regularization': 1.0,
                                           'random_state': 42}},
                    bootstrap={'unit': 'whole_session_all_symbols_together', 'replicates': 10000,
                               'seed': 20261018})
    contract.update(changes)
    return contract


def test_registered_contract_rows_are_byte_identical_to_pre_contract_labels():
    import hashlib

    contract = json.loads(REPO_CONTRACT.read_text())
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert canonical == trial.CONTRACT_HASH
    spec = trial.contract_spec(contract, legacy=True)
    assert spec == trial.TrialSpec()
    days = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    minutes = minute_data(days)
    minutes['open'] = minutes['open'] * (1 + 0.002 * np.sin(np.arange(len(minutes)) / 7))
    bars = trial.regular_bars(minutes, days)
    expected = legacy_labeled_rows(minutes, bars, symbol='AAPL')
    for actual in (trial.labeled_rows(minutes, bars, symbol='AAPL'),
                   trial.labeled_rows(minutes, bars, symbol='AAPL', spec=spec)):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert actual.to_parquet(index=False) == expected.to_parquet(index=False)
    assert expected.label.nunique() == 2


def flat_features(monkeypatch):
    monkeypatch.setattr(trial, 'build_day_sleeve_features', lambda window: pd.DataFrame(
        [{c: float(window.close.iloc[-1]) for c in DAY_SLEEVE_ML_FEATURE_COLUMNS}]))


def test_sixty_minute_exits_respect_cutoff_and_never_overlap(monkeypatch):
    flat_features(monkeypatch)
    days = ['2024-06-26', '2024-06-27', '2024-06-28', '2024-07-01', '2024-07-02', '2024-07-03', '2024-07-05']
    minutes = minute_data(days)
    bars = trial.regular_bars(minutes, days)
    contract = horizon_contract(symbols=['AAPL'])
    spec = trial.contract_spec(contract)
    rows = trial.labeled_rows(minutes, bars, symbol='AAPL', spec=spec)
    trial.validate_rows(rows, contract, spec)
    assert (rows.exit_at - rows.entry_at == pd.Timedelta(minutes=60)).all()
    assert (rows.entry_at - rows.bar_start == pd.Timedelta(minutes=6)).all()
    assert (rows.entry_at.iloc[1:].to_numpy() >= rows.exit_at.iloc[:-1].to_numpy()).all()
    local = rows.set_index('session').bar_start.dt.tz_convert('America/New_York').dt.strftime('%H:%M')
    assert list(local.loc['2024-07-02']) == ['09:30', '10:30', '11:30', '12:30', '13:30', '14:30']
    # Early close: exits must precede the 13:00 close.
    assert list(local.loc['2024-07-03']) == ['09:30', '10:30', '11:30']
    with pytest.raises(ValueError, match='exit horizon'):
        trial.validate_rows(rows, contract)
    overlapping = rows.copy()
    overlapping.loc[overlapping.index[1], ['entry_at', 'exit_at']] = (
        overlapping.entry_at.iloc[0] + pd.Timedelta(minutes=30), overlapping.exit_at.iloc[0] + pd.Timedelta(minutes=30))
    with pytest.raises(ValueError, match='overlapping|decision|entry'):
        trial.validate_rows(overlapping, contract, spec)


def test_exits_at_or_after_flatten_cutoff_are_excluded(monkeypatch):
    flat_features(monkeypatch)
    days = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    minutes = minute_data(days)
    bars = trial.regular_bars(minutes, days)

    def last_bars(**grid):
        contract = horizon_contract(symbols=['AAPL'], exit_horizon_minutes=19, **grid)
        contract = {k: v for k, v in contract.items() if v is not None}
        spec = trial.contract_spec(contract)
        rows = trial.labeled_rows(minutes, bars, symbol='AAPL', spec=spec)
        trial.validate_rows(rows, contract, spec)
        day = rows.loc[rows.session == '2024-01-05']
        return list(day.bar_start.dt.tz_convert('America/New_York').dt.strftime('%H:%M')), day

    # Twenty-minute cadence from 09:30 puts the 15:30 bar's exit exactly at 15:55.
    unconstrained, _ = last_bars(flatten_cutoff_et=None)
    assert unconstrained[-1] == '15:30'
    for grid in ({'flatten_cutoff_et': '15:55'}, {'flatten_cutoff_et': None, 'flatten_lead_minutes': 5}):
        kept, day = last_bars(**grid)
        assert kept == unconstrained[:-1]
        assert (day.exit_at.dt.tz_convert('America/New_York').dt.strftime('%H:%M') < '15:55').all()
    rows = trial.labeled_rows(minutes, bars, symbol='AAPL', spec=trial.contract_spec(
        {k: v for k, v in horizon_contract(exit_horizon_minutes=19, flatten_cutoff_et=None).items() if v is not None}))
    with pytest.raises(ValueError, match='flatten cutoff'):
        trial.validate_rows(rows, {'symbols': ['AAPL'], 'development_start': '2024-01-01',
                                   'development_end': '2025-12-31'},
                            trial.contract_spec(horizon_contract(exit_horizon_minutes=19)))


def test_early_close_cutoff_semantics():
    from datetime import date, time

    from ai_trading.tools.research_feasibility import IntradayGrid

    early = date(2024, 7, 3)
    as_local = lambda stamp: stamp.tz_convert('America/New_York').strftime('%H:%M')  # noqa: E731
    assert as_local(IntradayGrid(60, time(15, 55)).cutoff(early)) == '13:00'
    assert as_local(IntradayGrid(60, lead_minutes=5).cutoff(early)) == '12:55'
    assert as_local(IntradayGrid(60, time(15, 55)).cutoff(date(2024, 7, 2))) == '15:55'


def test_contract_spec_requires_explicit_values_for_new_contracts():
    spec = trial.contract_spec(horizon_contract())
    assert spec.grid.exit_minutes == 60 and spec.seed == 20261018 and spec.replicates == 10000
    model = trial.build_classifier(spec)
    assert type(model.steps[-1][1]).__name__ == 'HistGradientBoostingClassifier'
    assert model.steps[-1][1].get_params()['min_samples_leaf'] == 200
    legacy = json.loads(REPO_CONTRACT.read_text())
    with pytest.raises(ValueError, match='must declare'):
        trial.contract_spec(legacy)
    with pytest.raises(ValueError, match='must declare'):
        trial.contract_spec({k: v for k, v in horizon_contract().items() if k != 'classifier'})
    with pytest.raises(ValueError, match='classifier'):
        trial.contract_spec(horizon_contract(classifier={'family': 'random_forest', 'params': {}}))
    with pytest.raises(ValueError, match='replicates'):
        trial.contract_spec(horizon_contract(bootstrap={'replicates': 150, 'seed': 1}))
    with pytest.raises(ValueError, match='exit_horizon'):
        trial.contract_spec(horizon_contract(exit_horizon_minutes=0))


def test_bootstrap_seed_comes_from_spec():
    rows = pd.DataFrame({'session': [f's{i}' for i in range(40)]})
    values = np.linspace(-5, 5, 40)
    assert trial.bootstrap_lower(rows, values) == trial.bootstrap_lower(rows, values, seed=42, replicates=10000)
    assert trial.bootstrap_lower(rows, values) != trial.bootstrap_lower(rows, values, seed=20261018)


@pytest.mark.parametrize('contract', [horizon_contract(),
                                      {k: v for k, v in horizon_contract().items() if k != 'exit_horizon_minutes'}])
def test_alternate_contract_blocks_without_registration_or_writes(tmp_path, monkeypatch, contract):
    path = tmp_path / 'config/horizon.json'
    path.parent.mkdir()
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(trial, 'load_governed_development_timestamps', lambda *a: pytest.fail('data read'))
    monkeypatch.setattr(trial, 'claim_campaign_trial', lambda *a, **k: pytest.fail('claim'))
    with pytest.raises(ValueError, match='registration is required|must declare'):
        trial.run(tmp_path, fit=False, contract_path=Path('config/horizon.json'))
    assert not (tmp_path / 'artifacts').exists()


def test_alternate_contract_uses_its_own_campaign_directory(tmp_path, monkeypatch):
    contract = horizon_contract()
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/horizon.json').write_text(json.dumps(contract))
    state_path = tmp_path / 'artifacts/horizon_trial_test/campaign_state.json'
    trial.register_campaign(state_path, contract)
    trial.claim_campaign_trial(state_path, hypothesis_id=contract['hypotheses'][0], evidence_signature='test',
                               evaluation_start='2024-01-01', evaluation_end='2025-12-31', quality_passed=True)
    monkeypatch.setattr(trial, 'load_governed_development_timestamps', lambda *a: pytest.fail('data read'))
    assert trial.run(tmp_path, fit=True, contract_path=tmp_path / 'config/horizon.json')['status'] == 'budget_already_claimed'
    assert not (tmp_path / 'artifacts/model_replacement_20260921').exists()
