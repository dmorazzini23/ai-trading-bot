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
