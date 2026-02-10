import types

import pytest
np = pytest.importorskip("numpy")

np.random.seed(0)

from ai_trading import meta_learning
from ai_trading.meta_learning import MetaLearning
from ai_trading.meta import checkpoint


def _stub_portfolio_rl_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a lightweight torch substitute for portfolio RL tests."""
    import ai_trading.portfolio_rl as portfolio_rl

    class _Tensor:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def numpy(self):
            return np.asarray(self._values, dtype=np.float32)

    class _Linear:
        def __init__(self, _in_dim: int, out_dim: int):
            self._out_dim = out_dim

        def __call__(self, tensor):
            values = np.asarray(getattr(tensor, "_values", tensor), dtype=np.float32)
            base = float(values.mean()) if values.size else 0.0
            return _Tensor(np.full(self._out_dim, base, dtype=np.float32))

        def parameters(self):
            return []

    class _ReLU:
        def __call__(self, tensor):
            return tensor

        def parameters(self):
            return []

    class _Softmax:
        def __init__(self, dim: int = -1):
            self.dim = dim

        def __call__(self, tensor):
            values = np.asarray(getattr(tensor, "_values", tensor), dtype=np.float32)
            values = values - float(np.max(values)) if values.size else values
            exps = np.exp(values)
            denom = float(np.sum(exps)) or 1.0
            return _Tensor(exps / denom)

        def parameters(self):
            return []

    class _Sequential:
        def __init__(self, *layers):
            self._layers = list(layers)

        def __getitem__(self, index: int):
            return self._layers[index]

        def __call__(self, tensor):
            out = tensor
            for layer in self._layers:
                out = layer(out)
            return out

        def parameters(self):
            params = []
            for layer in self._layers:
                if hasattr(layer, "parameters"):
                    params.extend(layer.parameters())
            return params

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = types.SimpleNamespace(
        float32="float32",
        tensor=lambda values, dtype=None: _Tensor(values),
        no_grad=lambda: _NoGrad(),
    )
    fake_nn = types.SimpleNamespace(
        Sequential=_Sequential,
        Linear=_Linear,
        ReLU=_ReLU,
        Softmax=_Softmax,
    )
    fake_optim = types.SimpleNamespace(
        Adam=lambda params, lr: types.SimpleNamespace(params=list(params), lr=lr),
    )
    portfolio_rl._lazy_import_torch.cache_clear()
    monkeypatch.setattr(
        portfolio_rl,
        "_lazy_import_torch",
        lambda: (fake_torch, fake_nn, fake_optim),
    )


def test_meta_learning_instantiation():
    ml = MetaLearning()
    assert isinstance(ml, MetaLearning)


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


def test_save_and_load_checkpoint(tmp_path):
    path = tmp_path / "x.pkl"
    data = {"x": 1}
    saved = checkpoint.save_checkpoint(data, str(path))
    assert saved == data
    loaded = checkpoint.load_checkpoint(str(path))
    assert loaded == data


def test_optimize_signals(monkeypatch):
    m = types.SimpleNamespace(predict=lambda X: [0] * len(X))
    data = [1, 2]
    cfg = types.SimpleNamespace(MODEL_PATH="")
    assert meta_learning.optimize_signals(data, cfg, model=m) == [0, 0]
    monkeypatch.setattr(meta_learning, "load_model_checkpoint", lambda p: None)
    assert meta_learning.optimize_signals(data, cfg, model=None) == data


def test_optimize_signals_no_cfg():
    m = types.SimpleNamespace(predict=lambda X: [1] * len(X))
    data = [3, 4]
    assert meta_learning.optimize_signals(data, cfg=None, model=m) == [1, 1]
    assert meta_learning.optimize_signals(data, cfg=None, model=None) == data


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
        np,
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
    _stub_portfolio_rl_torch(monkeypatch)
    learner = meta_learning.PortfolioReinforcementLearner()
    state = np.random.rand(10)
    weights = learner.rebalance_portfolio(state)
    assert np.isclose(weights.sum(), 1.0, atol=0.1)
