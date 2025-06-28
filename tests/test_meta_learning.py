import pickle
import types

import numpy as np
import pandas as pd
import pytest

np.random.seed(0)

import meta_learning
import sklearn.linear_model


def test_load_weights_creates_default(tmp_path):
    p = tmp_path / "w.csv"
    arr = meta_learning.load_weights(str(p), default=np.array([1.0]))
    assert p.exists()
    assert np.allclose(arr, [1.0])


def test_update_weights_no_change(tmp_path):
    p = tmp_path / "w.csv"
    np.savetxt(p, np.array([0.1, 0.2]), delimiter=",")
    ok = meta_learning.update_weights(str(p), np.array([0.1, 0.2]), {"m": 1})
    assert ok is False


def test_update_signal_weights_normal():
    w = {"a": 1.0, "b": 1.0}
    perf = {"a": 1.0, "b": 3.0}
    res = meta_learning.update_signal_weights(w, perf)
    assert round(res["b"], 2) == 0.75


def test_save_and_load_checkpoint(monkeypatch):
    data = {}
    buf = {}

    def fake_dump(obj, fh):
        buf["v"] = obj

    def fake_load(fh):
        return buf["v"]

    monkeypatch.setattr(meta_learning.pickle, "dump", fake_dump)
    monkeypatch.setattr(meta_learning.pickle, "load", fake_load)
    monkeypatch.setattr(meta_learning, "open", lambda *a, **k: open("/dev/null", "wb"))
    meta_learning.save_model_checkpoint(data, "x.pkl")
    obj = meta_learning.load_model_checkpoint("x.pkl")
    assert obj is data


def test_optimize_signals(monkeypatch):
    m = types.SimpleNamespace(predict=lambda X: [0] * len(X))
    data = [1, 2]
    cfg = types.SimpleNamespace(MODEL_PATH="")
    assert meta_learning.optimize_signals(data, cfg, model=m) == [0, 0]
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda p: None)
    assert meta_learning.optimize_signals(data, cfg, model=None) == data


def test_retrain_meta_missing(tmp_path):
    assert not meta_learning.retrain_meta_learner(str(tmp_path / "no.csv"))


def test_update_weights_history_error(tmp_path):
    w = tmp_path / "w.csv"
    h = tmp_path / "hist.json"
    np.savetxt(w, np.array([0.1]), delimiter=",")
    h.write_text("{bad")
    assert meta_learning.update_weights(str(w), np.array([0.2]), {"m": 1}, str(h))


def test_load_weights_corrupted(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("bad,data")
    arr = meta_learning.load_weights(str(p), default=np.array([0.5]))
    assert arr.tolist() == [0.5]


def test_update_weights_success(tmp_path):
    p = tmp_path / "w.csv"
    np.savetxt(p, np.array([0.1]), delimiter=",")
    h = tmp_path / "hist.json"
    res = meta_learning.update_weights(str(p), np.array([0.2]), {"m": 1}, str(h))
    assert res


def test_load_model_checkpoint_missing(tmp_path):
    path = tmp_path / "x.pkl"
    assert meta_learning.load_model_checkpoint(str(path)) is None


def test_update_signal_weights_zero(caplog):
    caplog.set_level("WARNING")
    w = {"a": 1.0}
    perf = {"a": 0.0}
    res = meta_learning.update_signal_weights(w, perf)
    assert res == w
    assert "Total performance sum is zero" in caplog.text


def test_load_weights_existing(tmp_path):
    p = tmp_path / "w.csv"
    np.savetxt(p, np.array([0.5, 0.6]), delimiter=",")
    arr = meta_learning.load_weights(str(p), default=np.array([1.0]))
    assert np.allclose(arr, [0.5, 0.6])


def test_optimize_signals_failure():
    class Bad:
        def predict(self, d):
            raise ValueError("x")

    cfg = types.SimpleNamespace(MODEL_PATH="")
    res = meta_learning.optimize_signals([1], cfg, model=Bad())
    assert res == [1]


def test_load_weights_default_zero(tmp_path):
    arr = meta_learning.load_weights(str(tmp_path / "none.csv"))
    assert arr.size == 0


def test_update_weights_failure(monkeypatch, tmp_path):
    p = tmp_path / "w.csv"
    monkeypatch.setattr(
        meta_learning.np,
        "savetxt",
        lambda *a, **k: (_ for _ in ()).throw(OSError("bad")),
    )
    ok = meta_learning.update_weights(
        str(p), np.array([1.0]), {}, str(tmp_path / "h.json")
    )
    assert not ok


def test_update_signal_weights_empty(caplog):
    caplog.set_level("ERROR")
    assert meta_learning.update_signal_weights({}, {}) is None
    assert "Empty weights" in caplog.text


def test_update_signal_weights_norm_zero(caplog):
    caplog.set_level("WARNING")
    w = {"a": 0.0}
    perf = {"a": 1.0}
    res = meta_learning.update_signal_weights(w, perf)
    assert res == w
    assert "Normalization factor zero" in caplog.text


def test_portfolio_rl_trigger(monkeypatch):
    try:
        learner = meta_learning.PortfolioReinforcementLearner()
    except AttributeError as exc:
        if "SymBool" in str(exc) or "Linear" in str(exc):
            monkeypatch.setattr(
                meta_learning.torch.nn,
                "Linear",
                lambda *a, **k: types.SimpleNamespace(forward=lambda x: x),
            )
            learner = meta_learning.PortfolioReinforcementLearner()
        else:
            raise
    state = np.random.rand(10)
    weights = learner.rebalance_portfolio(state)
    assert np.isclose(weights.sum(), 1, atol=0.1)
