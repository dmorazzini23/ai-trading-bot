from tests.optdeps import require

require("numpy")
import json
import ai_trading.rl_trading.inference as inf
import numpy as np
import pytest
import ai_trading.rl.module as rl_mod
import ai_trading.rl_trading.train as train_mod
import zipfile


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

    with pytest.raises(ModuleNotFoundError):
        rl_pkg._load_train_module()

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
    with pytest.raises(ImportError):
        agent.load()


def test_rl_agent_loads_algorithm_from_sidecar_metadata(monkeypatch, tmp_path):
    import ai_trading.rl_trading as rl

    loaded: list[str] = []

    class DummyPPO:
        @classmethod
        def load(cls, path):
            loaded.append("PPO")
            return cls()

    class DummyA2C:
        @classmethod
        def load(cls, path):
            loaded.append("A2C")
            return cls()

    model_path = tmp_path / "model_a2c.zip"
    model_path.write_bytes(b"not-a-real-sb3-zip")
    (tmp_path / "meta.json").write_text(json.dumps({"algorithm": "A2C"}), encoding="utf-8")

    monkeypatch.setattr(rl, "is_rl_available", lambda: True)
    monkeypatch.setattr(rl, "PPO", DummyPPO)
    monkeypatch.setattr(rl, "A2C", DummyA2C)

    agent = rl.RLAgent(model_path)
    agent.load()

    assert loaded == ["A2C"]
    assert isinstance(agent.model, DummyA2C)


def test_rl_agent_loads_algorithm_from_model_zip_hint(monkeypatch, tmp_path):
    import ai_trading.rl_trading as rl

    loaded: list[str] = []

    class DummyTD3:
        @classmethod
        def load(cls, path):
            loaded.append("TD3")
            return cls()

    model_path = tmp_path / "model.zip"
    with zipfile.ZipFile(model_path, mode="w") as model_zip:
        model_zip.writestr("rl_model_metadata.json", json.dumps({"sb3_algorithm": "TD3"}))

    monkeypatch.setattr(rl, "is_rl_available", lambda: True)
    monkeypatch.setattr(rl, "TD3", DummyTD3)

    agent = rl.RLAgent(model_path)
    agent.load()

    assert loaded == ["TD3"]
    assert isinstance(agent.model, DummyTD3)
