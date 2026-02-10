import numpy as np
import pytest

import types


def _stub_portfolio_rl_torch(monkeypatch: pytest.MonkeyPatch):
    class _Tensor:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def numpy(self):
            return np.asarray(self._values, dtype=np.float32)

    class _Linear:
        def __init__(self, _in_dim: int, out_dim: int):
            self.out_dim = out_dim

        def __call__(self, tensor):
            values = np.asarray(getattr(tensor, "_values", tensor), dtype=np.float32)
            base = float(values.mean()) if values.size else 0.0
            return _Tensor(np.full(self.out_dim, base, dtype=np.float32))

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
            self.layers = list(layers)

        def __getitem__(self, index: int):
            return self.layers[index]

        def __call__(self, tensor):
            out = tensor
            for layer in self.layers:
                out = layer(out)
            return out

        def parameters(self):
            params = []
            for layer in self.layers:
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
        nn=types.SimpleNamespace(Linear=_Linear),
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
    monkeypatch.setattr(
        "ai_trading.portfolio_rl._lazy_import_torch",
        lambda: (fake_torch, fake_nn, fake_optim),
    )
    return fake_torch


def test_rebalance_portfolio_normalizes_weights(monkeypatch):
    from ai_trading.portfolio_rl import PortfolioReinforcementLearner

    fake_torch = _stub_portfolio_rl_torch(monkeypatch)
    learner = PortfolioReinforcementLearner()
    state = np.random.rand(learner.state_dim)
    weights = learner.rebalance_portfolio(state)
    assert weights.shape[0] == learner.action_dim
    assert isinstance(learner.actor.net[0], fake_torch.nn.Linear)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)


def test_import_error_when_torch_missing(monkeypatch):
    import builtins
    import sys
    import ai_trading.portfolio_rl as prl

    prl._lazy_import_torch.cache_clear()
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("torch"):
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="PyTorch is required for ai_trading.portfolio_rl"):
        prl.PortfolioReinforcementLearner()
