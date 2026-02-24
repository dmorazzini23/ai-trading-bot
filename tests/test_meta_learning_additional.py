import types
from pathlib import Path

import pytest
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
sklearn = pytest.importorskip("sklearn")
import sklearn.linear_model
from ai_trading import meta_learning


def _stub_ridge(*args, **kwargs):
    """Return a simple object with ``fit`` and ``predict`` methods."""

    def fit(X, y, sample_weight=None):
        return None

    def predict(X):
        return [0] * len(X)

    return types.SimpleNamespace(fit=fit, predict=predict)


def test_load_weights_save_fail(monkeypatch, tmp_path, caplog):
    """Failure to write default weights is logged and default returned."""
    p = tmp_path / "w.csv"
    monkeypatch.setattr(meta_learning.Path, "exists", lambda self: False)
    def fail(*a, **k):
        raise OSError("fail")
    monkeypatch.setattr(np, "savetxt", fail)
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


def test_load_checkpoint_roundtrip(tmp_path):
    """`load_checkpoint` returns the stored mapping."""
    path = tmp_path / "chk.pkl"
    data = {"foo": 1}
    meta_learning.save_model_checkpoint(data, str(path))
    loaded = meta_learning.load_checkpoint(str(path))
    assert loaded == data


def test_retrain_meta_learner(monkeypatch, tmp_path):
    """Meta learner retrains with small dataset."""
    data = Path(tmp_path / "trades.csv")
    df = pd.DataFrame({
        "entry_price": [1, 2],
        "exit_price": [2, 3],
        "signal_tags": ["a", "b"],
        "side": ["buy", "sell"],
    })
    df.to_csv(data, index=False)
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda *a, **k: [])
    monkeypatch.setattr(sklearn.linear_model, "Ridge", _stub_ridge)
    ok = meta_learning.retrain_meta_learner(str(data), str(tmp_path/"m.pkl"), str(tmp_path/"hist.pkl"), min_samples=1)
    assert ok


def test_retrain_meta_learner_filters_non_decimal_prices(monkeypatch, tmp_path):
    """Rows with non-decimal price strings are excluded before training."""

    data = Path(tmp_path / "strict.csv")
    df = pd.DataFrame(
        {
            "entry_price": ["100.50", "1e2", "invalid", "50.25"],
            "exit_price": ["105.75", "99.00", "110.00", "55.00"],
            "signal_tags": ["momentum", "mean_revert", "momentum", "trend"],
            "side": ["buy", "sell", "buy", "sell"],
        }
    )
    df.to_csv(data, index=False)

    captured: dict[str, object] = {}

    def _capturing_ridge(*args, **kwargs):
        def fit(X, y, sample_weight=None):
            captured["x_shape"] = getattr(X, "shape", None)
            captured["sample_count"] = len(y)
            captured["weight_len"] = len(sample_weight) if sample_weight is not None else None
            return None

        def predict(X):
            return np.zeros(len(X))

        return types.SimpleNamespace(fit=fit, predict=predict)

    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda *a, **k: [])
    monkeypatch.setattr(sklearn.linear_model, "Ridge", _capturing_ridge)

    ok = meta_learning.retrain_meta_learner(
        str(data), str(tmp_path / "m.pkl"), str(tmp_path / "hist.pkl"), min_samples=2
    )

    assert ok
    assert captured["sample_count"] == 2
    assert captured["weight_len"] == 2
    assert captured["x_shape"] and captured["x_shape"][0] == 2


def test_retrain_meta_learner_excludes_synthetic_rows_by_default(monkeypatch, tmp_path):
    data = Path(tmp_path / "synthetic_default.csv")
    df = pd.DataFrame(
        {
            "entry_price": [100.0, 101.0, 102.0],
            "exit_price": [101.0, 102.0, 103.0],
            "signal_tags": ["momentum", "synthetic_bootstrap_data", "trend"],
            "side": ["buy", "sell", "buy"],
            "strategy": ["live_trading", "bootstrap_generated", "live_trading"],
        }
    )
    df.to_csv(data, index=False)

    captured: dict[str, object] = {}

    def _capturing_ridge(*args, **kwargs):
        def fit(X, y, sample_weight=None):
            captured["sample_count"] = len(y)
            return None

        def predict(X):
            return np.zeros(len(X))

        return types.SimpleNamespace(fit=fit, predict=predict)

    monkeypatch.setenv("AI_TRADING_META_LEARNING_ALLOW_SYNTHETIC_BOOTSTRAP", "0")
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda *a, **k: [])
    monkeypatch.setattr(sklearn.linear_model, "Ridge", _capturing_ridge)

    ok = meta_learning.retrain_meta_learner(
        str(data), str(tmp_path / "m.pkl"), str(tmp_path / "hist.pkl"), min_samples=2
    )

    assert ok
    assert captured["sample_count"] == 2


def test_retrain_meta_learner_keeps_synthetic_rows_with_override(monkeypatch, tmp_path):
    data = Path(tmp_path / "synthetic_override.csv")
    df = pd.DataFrame(
        {
            "entry_price": [100.0, 101.0, 102.0],
            "exit_price": [101.0, 102.0, 103.0],
            "signal_tags": ["momentum", "synthetic_bootstrap_data", "trend"],
            "side": ["buy", "sell", "buy"],
            "strategy": ["live_trading", "bootstrap_generated", "live_trading"],
        }
    )
    df.to_csv(data, index=False)

    captured: dict[str, object] = {}

    def _capturing_ridge(*args, **kwargs):
        def fit(X, y, sample_weight=None):
            captured["sample_count"] = len(y)
            return None

        def predict(X):
            return np.zeros(len(X))

        return types.SimpleNamespace(fit=fit, predict=predict)

    monkeypatch.setenv("AI_TRADING_META_LEARNING_ALLOW_SYNTHETIC_BOOTSTRAP", "1")
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda *a, **k: [])
    monkeypatch.setattr(sklearn.linear_model, "Ridge", _capturing_ridge)

    ok = meta_learning.retrain_meta_learner(
        str(data), str(tmp_path / "m.pkl"), str(tmp_path / "hist.pkl"), min_samples=3
    )

    assert ok
    assert captured["sample_count"] == 3


def test_retrain_meta_learner_handles_non_iterable_columns(monkeypatch, tmp_path):
    """Handles DataFrames where ``columns`` is not iterable."""
    path = tmp_path / "trades.csv"
    pd.DataFrame({
        "entry_price": [1],
        "exit_price": [1],
        "signal_tags": ["a"],
        "side": ["buy"],
    }).to_csv(path, index=False)

    def fake_getattr(obj, name, default=None):
        if name == "columns":
            return 123  # non-iterable placeholder
        return getattr(obj, name, default)

    monkeypatch.setattr(meta_learning, "getattr", fake_getattr)
    monkeypatch.setattr(meta_learning, "save_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda *a, **k: [])
    monkeypatch.setattr(sklearn.linear_model, "Ridge", _stub_ridge)
    result = meta_learning.retrain_meta_learner(
        str(path), str(tmp_path / "m.pkl"), str(tmp_path / "hist.pkl"), min_samples=2
    )
    assert not result


def test_optimize_signals(monkeypatch):
    """optimize_signals uses model predictions when available."""
    m = types.SimpleNamespace(predict=lambda X: [1,2,3])
    data = [1,2,3]
    assert meta_learning.optimize_signals(data, types.SimpleNamespace(MODEL_PATH=""), model=m) == [1,2,3]
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda path: None)
    assert meta_learning.optimize_signals(data, types.SimpleNamespace(MODEL_PATH=""), model=None) == data
