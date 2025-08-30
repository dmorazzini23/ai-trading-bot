import io
import types

import pytest

pd = pytest.importorskip("pandas")
sklearn = pytest.importorskip("sklearn")
import sklearn.linear_model
from ai_trading.utils.device import TORCH_AVAILABLE
if not TORCH_AVAILABLE:
    pytest.skip("torch not installed", allow_module_level=True)
import torch
try:
    import pydantic_settings  # noqa: F401
    from ai_trading import meta_learning
except ImportError:
    pytest.skip("pydantic v2 required", allow_module_level=True)

pytestmark = pytest.mark.slow

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
    monkeypatch.setattr(
        meta_learning, "load_model_checkpoint", lambda p: {"mock": "model"}
    )
    monkeypatch.setattr(meta_learning, "open", lambda *a, **k: io.BytesIO())

    class DummyModel:
        def fit(self, X, y, sample_weight=None):
            self.fitted = True

        def predict(self, X):
            return [0] * len(X)

    monkeypatch.setattr(sklearn.linear_model, "Ridge", lambda *a, **k: DummyModel())
    ok = meta_learning.retrain_meta_learner("trades.csv", "m.pkl", "hist.pkl", min_samples=1)
    assert ok


def test_retrain_meta_insufficient(monkeypatch):
    df = pd.DataFrame({"entry_price": [1], "exit_price": [2], "signal_tags": ["a"], "side": ["buy"]})
    monkeypatch.setattr(meta_learning.Path, "exists", lambda self: True)
    monkeypatch.setattr(pd, "read_csv", lambda p: df)
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(meta_learning, "open", lambda *a, **k: io.BytesIO())
    assert not meta_learning.retrain_meta_learner("trades.csv", "m.pkl", "hist.pkl", min_samples=5)


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
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda p: {"mock": "model"})

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
