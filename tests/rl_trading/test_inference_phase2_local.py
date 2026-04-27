from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ai_trading.rl_trading import inference as inf
from ai_trading.rl_trading.env import ActionSpaceConfig, RewardConfig
from ai_trading.rl_trading.inference import InferenceConfig, UnifiedRLInference


class _Model:
    def __init__(self, actions, action_space) -> None:
        self.actions = list(actions)
        self.action_space = action_space
        self.calls: list[tuple[object, bool]] = []

    def predict(self, obs, deterministic: bool = True):
        self.calls.append((obs, deterministic))
        action = self.actions.pop(0) if self.actions else 0
        return action, None


class _Agent:
    model = None
    loaded: list[str] = []

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = _Agent.model

    def load(self) -> None:
        _Agent.loaded.append(self.model_path)

    def predict(self, state):
        return state


def _make_inference(monkeypatch: pytest.MonkeyPatch, *, action_type: str = "discrete", actions=None) -> UnifiedRLInference:
    _Agent.model = _Model(actions or [1], SimpleNamespace(n=4) if action_type == "discrete" else SimpleNamespace(shape=(2,)))
    _Agent.loaded = []
    monkeypatch.setattr(inf, "RLAgent", _Agent)
    cfg = InferenceConfig(
        model_path="model.zip",
        action_config=ActionSpaceConfig(action_type=action_type, discrete_actions=3),
        reward_config=RewardConfig(),
        observation_window=3,
        confidence_threshold=0.25,
    )
    return UnifiedRLInference(cfg)


def test_preprocess_and_discrete_postprocess_with_action_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    inference = _make_inference(monkeypatch, actions=[2])

    obs1 = inference.preprocess_observation(np.array([1.0, 2.0]))
    obs2 = inference.preprocess_observation(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))
    masked_to_hold = inference.postprocess_action(2, obs1, action_mask=[1.0, 1.0, 0.0])
    masked_to_first_valid = inference.postprocess_action(1, obs1, action_mask=[0.0, 0.0, 1.0])
    invalid_mask = inference.postprocess_action(1, obs1, action_mask=object())

    assert obs1.shape == (3, 2)
    assert obs2.shape == (3, 2)
    assert masked_to_hold["action"] == "hold"
    assert masked_to_hold["action_masked"] is True
    assert masked_to_first_valid["action"] == "sell"
    assert invalid_mask["action"] == "buy"
    assert _Agent.loaded == ["model.zip"]


def test_predict_batch_stats_reset_and_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    inference = _make_inference(monkeypatch, actions=[1, 0, 2])

    buy = inference.predict(np.array([1.0, 2.0]), symbol="AAPL")
    batch = inference.predict_batch([np.array([1.0, 2.0]), np.array([2.0, 3.0])], ["MSFT", "TSLA"])
    stats = inference.get_stats()

    assert buy is not None and buy.symbol == "AAPL" and buy.side == "buy"
    assert [signal.side if signal else None for signal in batch] == ["hold", "sell"]
    assert stats["total_predictions"] == 3
    assert stats["buy_predictions"] == 1
    assert stats["hold_predictions"] == 1
    assert stats["sell_predictions"] == 1
    assert "config" in stats

    inference.reset_stats()
    assert inference.get_stats()["total_predictions"] == 0

    with pytest.raises(ValueError, match="same length"):
        inference.predict_batch([np.array([1.0])], ["AAPL", "MSFT"])

    created = inf.create_unified_inference("created.zip", action_type="discrete", observation_window=2)
    assert created.config.model_path == "created.zip"


def test_continuous_actions_and_prediction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    inference = _make_inference(monkeypatch, action_type="continuous", actions=[np.array([0.5]), -0.5, 0.1])
    obs = np.array([1.0, 2.0])

    assert inference.postprocess_action(np.array([0.5]), obs)["action"] == "buy"
    assert inference.postprocess_action(-0.5, obs)["action"] == "sell"
    assert inference.postprocess_action("bad", obs)["action"] == "hold"

    buy = inference.predict(obs)
    sell = inference.predict(obs)
    hold = inference.predict(obs)

    assert [buy.side if buy else None, sell.side if sell else None, hold.side if hold else None] == ["buy", "sell", "hold"]

    inference.agent.model = None
    assert inference.predict(obs) is None

    class BrokenModel:
        action_space = None

        def predict(self, *_args, **_kwargs):
            raise ValueError("bad predict")

    inference.agent.model = BrokenModel()
    assert inference.predict(obs) is None


def test_backward_compatible_load_and_predict_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _Agent("loaded.zip")
    monkeypatch.setattr(inf, "RLAgent", lambda model_path: loaded)
    assert inf.load_policy("loaded.zip") is loaded

    first = SimpleNamespace(side="buy")
    assert inf.predict_signal(SimpleNamespace(predict=lambda _state: [first]), np.array([1.0])) is first
    assert inf.predict_signal(SimpleNamespace(predict=lambda _state: []), np.array([1.0])) is None
    sell = SimpleNamespace(side="sell")
    assert inf.predict_signal(SimpleNamespace(predict=lambda _state: sell), np.array([1.0])) is sell
