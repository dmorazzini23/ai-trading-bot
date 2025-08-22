import ai_trading.rl_trading.inference as inf
import ai_trading.rl_trading.train as train_mod
import numpy as np


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
    path = tmp_path / "model.zip"
    train_mod.train(data, path, timesteps=10)
    agent = inf.load_policy(path)
    sig = inf.predict_signal(agent, data[0])
    assert sig and sig.side == "buy"
