import io
import types

import pandas as pd
import pytest

import meta_learning
import sklearn.linear_model


@pytest.mark.slow
def test_retrain_meta_learner_success(monkeypatch):
    df = pd.DataFrame({
        "entry_price": [1, 2, 3, 4],
        "exit_price": [2, 3, 4, 5],
        "signal_tags": ["a", "a+b", "b", "a"],
        "side": ["buy", "sell", "buy", "sell"],
    })
    monkeypatch.setattr(meta_learning.Path, "exists", lambda self: True)
    monkeypatch.setattr(pd, "read_csv", lambda p: df)
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda m, p: None)
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda p: [])
    monkeypatch.setattr(meta_learning, "open", lambda *a, **k: io.BytesIO())

    class DummyModel:
        def fit(self, X, y, sample_weight=None):
            self.fitted = True

        def predict(self, X):
            return [0] * len(X)

    monkeypatch.setattr(sklearn.linear_model, "Ridge", lambda *a, **k: DummyModel())
    ok = meta_learning.retrain_meta_learner("trades.csv", "m.pkl", "hist.pkl", min_samples=1)
    assert ok


@pytest.mark.slow
def test_retrain_meta_insufficient(monkeypatch):
    df = pd.DataFrame({"entry_price": [1], "exit_price": [2], "signal_tags": ["a"], "side": ["buy"]})
    monkeypatch.setattr(meta_learning.Path, "exists", lambda self: True)
    monkeypatch.setattr(pd, "read_csv", lambda p: df)
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(meta_learning, "open", lambda *a, **k: io.BytesIO())
    assert not meta_learning.retrain_meta_learner("trades.csv", "m.pkl", "hist.pkl", min_samples=5)


@pytest.mark.slow
def test_retrain_meta_training_fail(monkeypatch):
    df = pd.DataFrame({
        "entry_price": [1, 2],
        "exit_price": [2, 3],
        "signal_tags": ["a", "b"],
        "side": ["buy", "sell"],
    })
    monkeypatch.setattr(meta_learning.Path, "exists", lambda self: True)
    monkeypatch.setattr(pd, "read_csv", lambda p: df)
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(meta_learning, "open", lambda *a, **k: io.BytesIO())

    class Bad:
        def fit(self, X, y, sample_weight=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(sklearn.linear_model, "Ridge", lambda *a, **k: Bad())
    assert not meta_learning.retrain_meta_learner("trades.csv", "m.pkl", "hist.pkl", min_samples=1)


@pytest.mark.slow
def test_retrain_meta_load_history(monkeypatch):
    df = pd.DataFrame({
        "entry_price": [1, 2],
        "exit_price": [2, 3],
        "signal_tags": ["a", "b"],
        "side": ["buy", "buy"],
    })
    monkeypatch.setattr(meta_learning.Path, "exists", lambda self: True)
    monkeypatch.setattr(pd, "read_csv", lambda p: df)
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(meta_learning, "open", lambda *a, **k: io.BytesIO())

    monkeypatch.setattr(
        sklearn.linear_model,
        "Ridge",
        lambda *a, **k: types.SimpleNamespace(
            fit=lambda X, y, sample_weight=None: None,
            predict=lambda X: [0] * len(X),
        ),
    )
    ok = meta_learning.retrain_meta_learner("trades.csv", "m.pkl", "hist.pkl", min_samples=1)
    assert ok
