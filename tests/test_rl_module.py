from tests.optdeps import require

require("numpy")
import ai_trading.rl_trading.inference as inf
import numpy as np
import ai_trading.rl.module as rl_mod
import ai_trading.rl_trading.train as train_mod


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
    monkeypatch.setattr(inf, "RLAgent", rl.RLAgent)
    monkeypatch.setattr(rl, "PPO", DummyPPO)
    monkeypatch.setattr(rl, "is_rl_available", lambda: True)
    path = tmp_path / "model.zip"
    train_mod.train(data, path, timesteps=10)
    agent = inf.load_policy(path)
    sig = inf.predict_signal(agent, data[0])
    assert sig and sig.side == "buy"


def test_rl_train_module_reload_preserves_train_attr(monkeypatch):
    import importlib
    import sys

    import ai_trading.rl_trading as rl_pkg

    module_name = "ai_trading.rl_trading.train"
    original_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == module_name:
            raise ModuleNotFoundError("simulated missing RL training module")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    sys.modules.pop(module_name, None)
    rl_pkg.__dict__.pop("train", None)

    stub_module = rl_pkg._load_train_module()
    assert stub_module.__spec__ is not None
    assert sys.modules[module_name] is stub_module

    reloaded = importlib.reload(stub_module)
    assert hasattr(reloaded, "train")
    assert callable(reloaded.train)
    assert getattr(reloaded, "USING_RL_TRAIN_STUB", False)

    # Restore the real module so subsequent tests see the canonical implementation.
    monkeypatch.setattr(importlib, "import_module", original_import_module)
    sys.modules.pop(module_name, None)
    restored = rl_pkg._load_train_module()
    globals()["train_mod"] = restored

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


def test_rl_wrapper_without_c_defaults(monkeypatch, tmp_path):
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
    path = tmp_path / "model.zip"
    rl_mod.train(data, path)
    agent = rl_mod.load(path)
    sig = rl_mod.predict(agent, data[0])
    assert sig and sig.side == "buy"


def test_rl_agent_stub_mode_is_non_trading(monkeypatch, tmp_path):
    import ai_trading.rl_trading as rl

    monkeypatch.setattr(rl, "is_rl_available", lambda: False)

    agent = rl.RLAgent(tmp_path / "missing.zip")
    agent.load()
    signal = agent.predict(np.zeros(4, dtype=float))

    assert agent._using_stub_model is True
    assert signal is None
