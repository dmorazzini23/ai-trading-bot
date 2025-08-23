import types
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("sklearn", reason="Optional heavy dependency; guard at import time")  # AI-AGENT-REF: guard sklearn
import sklearn.linear_model
from ai_trading import meta_learning


def test_load_weights_save_fail(monkeypatch, tmp_path, caplog):
    """Failure to write default weights is logged and default returned."""
    p = tmp_path / "w.csv"
    monkeypatch.setattr(meta_learning.Path, "exists", lambda self: False)
    def fail(*a, **k):
        raise OSError("fail")
    monkeypatch.setattr(meta_learning.np, "savetxt", fail)
    caplog.set_level("ERROR")
    arr = meta_learning.load_weights(str(p), default=np.array([1.0]))
    assert arr.tolist() == [1.0]
    assert "Failed initializing" in caplog.text


def test_update_signal_weights_edge_cases():
    """Handles empty weights and zero performance."""
    assert meta_learning.update_signal_weights({}, {}) is None
    w = {"a": 1.0}
    perf = {"a": 0.0}
    assert meta_learning.update_signal_weights(w, perf) == w


def test_save_and_load_checkpoint(tmp_path):
    """Model checkpoints can be saved and loaded."""
    path = tmp_path / "m.pkl"
    meta_learning.save_model_checkpoint({"x": 1}, str(path))
    obj = meta_learning.load_model_checkpoint(str(path))
    assert obj == {"x": 1}


def test_retrain_meta_learner(monkeypatch, tmp_path):
    """Meta learner retrains with small dataset."""
    data = Path(tmp_path / "trades.csv")
    df = meta_learning.pd.DataFrame({
        "entry_price": [1, 2],
        "exit_price": [2, 3],
        "signal_tags": ["a", "b"],
        "side": ["buy", "sell"],
    })
    df.to_csv(data, index=False)
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda *a, **k: [])
    monkeypatch.setattr(sklearn.linear_model, "Ridge", lambda *a, **k: types.SimpleNamespace(fit=lambda X,y, sample_weight=None: None, predict=lambda X:[0]*len(X)))
    ok = meta_learning.retrain_meta_learner(str(data), str(tmp_path/"m.pkl"), str(tmp_path/"hist.pkl"), min_samples=1)
    assert ok


def test_optimize_signals(monkeypatch):
    """optimize_signals uses model predictions when available."""
    m = types.SimpleNamespace(predict=lambda X: [1,2,3])
    data = [1,2,3]
    assert meta_learning.optimize_signals(data, types.SimpleNamespace(MODEL_PATH=""), model=m) == [1,2,3]
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda path: None)
    assert meta_learning.optimize_signals(data, types.SimpleNamespace(MODEL_PATH=""), model=None) == data
