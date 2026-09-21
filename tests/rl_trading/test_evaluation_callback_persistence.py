import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ai_trading.rl_trading import train


class CallbackBase:
    def __init__(self, verbose):
        self.verbose = verbose
        self.n_calls = 0


@pytest.fixture
def callback(monkeypatch, tmp_path):
    monkeypatch.setattr(train, 'np', np)
    env = SimpleNamespace(reset=lambda: ('start', {}), step=lambda action:
        ('end', 3., True, False, {'turnover_penalty': .1, 'drawdown_penalty': .2,
                                'variance_penalty': .3, 'constraint_violations': ['limit']}))
    cls = train._build_detailed_eval_callback(CallbackBase)
    cb = cls(env, eval_freq=2, save_path=str(tmp_path))
    cb.model = SimpleNamespace(predict=Mock(return_value=(1, None)), save=Mock())
    monkeypatch.setattr(train, 'evaluate_policy', Mock(return_value=([1., 3.], [2, 4])))
    return cb


def test_callback_scheduling_metrics_and_best_model_metadata(callback, tmp_path):
    cb = callback
    cb.n_calls = 1
    assert cb._on_step() is True and cb.eval_results == []
    cb.n_calls = 2
    assert cb._on_step() is True
    row = cb.eval_results[0]
    assert row['mean_reward'] == 2. and row['std_reward'] == 1.
    assert row['mean_episode_length'] == 3.
    assert row['avg_turnover_penalty'] == .1
    assert row['constraint_violation_rate'] == 1.
    assert json.loads((tmp_path / 'best_model_meta.json').read_text()) == row
    cb.model.save.assert_called_once_with(str(tmp_path / 'best_model.zip'))
    cb.n_calls = 4
    cb._on_step()
    cb.model.save.assert_called_once()
    cb.save_results(str(tmp_path / 'evaluations.json'))
    assert json.loads((tmp_path / 'evaluations.json').read_text()) == cb.eval_results


@pytest.mark.parametrize('history,expected', [([], 0.), ([{'mean_reward': 2.}] * 3, 0.),
    ([{'mean_reward': 1.}, {'mean_reward': 3.}], 2.), ([{}, {}], 0.)])
def test_sharpe_handles_sparse_flat_and_invalid_history(callback, history, expected):
    callback.eval_results = history
    assert callback._calculate_sharpe_ratio() == expected


def test_evaluation_and_io_failures_do_not_crash_training_callback(callback, monkeypatch, tmp_path):
    monkeypatch.setattr(train, 'evaluate_policy', Mock(side_effect=ValueError('bad reward')))
    callback._evaluate_model()
    assert callback.eval_results == []
    callback.model.save.assert_not_called()
    callback.eval_env.reset = Mock(side_effect=ValueError('bad observation'))
    assert callback._collect_detailed_metrics() == {}
    callback.save_results(str(tmp_path / 'missing' / 'results.json'))
    assert not (tmp_path / 'missing/results.json').exists()
