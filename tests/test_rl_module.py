from tests.optdeps import require
require("numpy")

import pytest
import numpy as np

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")
pytest.importorskip("torch")

import ai_trading.rl_trading.inference as inf
import ai_trading.rl_trading.train as train_mod


def test_rl_train_and_infer(tmp_path):
    data = np.random.rand(20, 4)
    trainer = train_mod.RLTrainer(total_timesteps=10)
    trainer.train(data)
    path = tmp_path / "model.zip"
    trainer.save(path)
    agent = inf.load_policy(path)
    sig = inf.predict_signal(agent, data[0])
    assert sig is None or sig.side in {"buy", "sell", "hold"}
