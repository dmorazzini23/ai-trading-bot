from tests.optdeps import require
require("numpy")
import ai_trading.rl_trading.inference as inf
import ai_trading.rl_trading.train as train_mod
import numpy as np
import ai_trading.rl.module as rl_mod


def test_rl_train_and_infer(monkeypatch, tmp_path):
    data = np.random.rand(20, 4)
    class DummyPPO:
        def __init__(self, *_a, **_k): pass
        def learn(self, *a, **k): return None
        def save(self, path): open(path, 'wb').write(b'0')
        def predict(self, state, deterministic=True): return (1, None)
        @classmethod
        def load(cls, path):
            return cls()
    monkeypatch.setattr(train_mod, "PPO", DummyPPO)
    import ai_trading.rl_trading as rl
    monkeypatch.setattr(rl, "PPO", DummyPPO)
    monkeypatch.setattr(rl, "is_rl_available", lambda: True)
    path = tmp_path / "model.zip"
    train_mod.train(data, path, timesteps=10)
    agent = inf.load_policy(path)
    sig = inf.predict_signal(agent, data[0])
    assert sig and sig.side == "buy"


def test_rl_train_module_reload_preserves_train_attr():
    import importlib

    reloaded = importlib.reload(train_mod)
    assert hasattr(reloaded, "train")
    assert callable(reloaded.train)

def test_rl_wrapper_without_c(monkeypatch, tmp_path):
    data = np.random.rand(20, 4)
    class DummyPPO:
        def __init__(self, *_a, **_k):
            pass
        def learn(self, *a, **k):
            return None
        def save(self, path):
            open(path, "wb").write(b"0")
        def predict(self, state, deterministic=True):
            return (1, None)
        @classmethod
        def load(cls, path):
            return cls()
    monkeypatch.setattr(train_mod, "PPO", DummyPPO)
    monkeypatch.setattr(rl_mod._rl, "PPO", DummyPPO)
    monkeypatch.setattr(rl_mod._rl, "is_rl_available", lambda: True)
    monkeypatch.delattr(rl_mod, "_C", raising=False)
    cfg = rl_mod.RLConfig(timesteps=5)
    path = tmp_path / "model.zip"
    rl_mod.train(data, path, cfg)
    agent = rl_mod.load(path)
    sig = rl_mod.predict(agent, data[0])
    assert sig and sig.side == "buy"
