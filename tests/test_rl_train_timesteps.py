from tests.optdeps import require
require("numpy")

import numpy as np
import ai_trading.rl_trading.train as train_mod


def test_train_passes_timesteps(monkeypatch, tmp_path):
    captured = {}

    class DummyPPO:
        def __init__(self, *args, **kwargs):
            pass

        def learn(self, *args, **kwargs):
            captured["timesteps"] = kwargs.get("total_timesteps")

        def save(self, path):
            open(path, "wb").write(b"0")

    monkeypatch.setattr(train_mod, "PPO", DummyPPO)
    train_mod.train(np.zeros((2, 2)), tmp_path / "model.zip", timesteps=7)
    assert captured["timesteps"] == 7
