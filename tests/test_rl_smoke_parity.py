"""Smoke tests for RL training/inference parity."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_action_space_parity() -> None:
    """Training env and inference wrapper should share action-space config."""

    pytest.importorskip("gymnasium")
    from ai_trading.rl_trading import is_rl_available
    from ai_trading.rl_trading.env import ActionSpaceConfig, RewardConfig, TradingEnv
    from ai_trading.rl_trading.inference import InferenceConfig, UnifiedRLInference

    if not is_rl_available():
        pytest.skip("RL stack unavailable")

    rng = np.random.default_rng(42)
    test_data = rng.normal(size=(100, 5))

    action_config = ActionSpaceConfig(action_type="discrete", discrete_actions=3)
    reward_config = RewardConfig(normalize_rewards=True)
    env = TradingEnv(
        data=test_data,
        window=10,
        action_config=action_config,
        reward_config=reward_config,
    )

    obs, _ = env.reset()
    assert obs.shape == (10, 5)
    for action in [0, 1, 2, 0, 1]:
        _obs, _reward, terminated, _truncated, info = env.step(action)
        assert "position" in info
        if terminated:
            break

    continuous_env = TradingEnv(
        data=test_data,
        window=10,
        action_config=ActionSpaceConfig(
            action_type="continuous",
            continuous_bounds=(-1.0, 1.0),
        ),
        reward_config=reward_config,
    )
    continuous_obs, _ = continuous_env.reset()
    assert continuous_obs.shape == (10, 5)

    with tempfile.TemporaryDirectory() as temp_dir:
        inference_config = InferenceConfig(
            model_path=str(Path(temp_dir) / "mock_model"),
            action_config=action_config,
            reward_config=reward_config,
            observation_window=10,
        )
        try:
            inference = UnifiedRLInference(inference_config)
        except (RuntimeError, FileNotFoundError) as exc:
            assert "model" in str(exc).lower()
        else:
            processed_obs = inference.preprocess_observation(test_data[50])
            assert processed_obs.shape == (10, 5)
            action_details = inference.postprocess_action(1, processed_obs)
            assert action_details["action"] in {"buy", "hold", "sell"}


def test_reward_normalization() -> None:
    """Reward normalization should produce finite rewards in the environment."""

    pytest.importorskip("gymnasium")
    from ai_trading.rl_trading import is_rl_available
    from ai_trading.rl_trading.env import RewardConfig, RunningStats, TradingEnv

    if not is_rl_available():
        pytest.skip("RL stack unavailable")

    stats = RunningStats(window=10)
    for value in [1.0, 2.0, -1.0, 3.0, 0.5, -0.5, 2.5, 1.5, -0.2, 0.8]:
        stats.update(value)
    assert np.isfinite(stats.normalize(1.5))

    rng = np.random.default_rng(42)
    env = TradingEnv(
        data=rng.normal(size=(50, 3)),
        window=5,
        reward_config=RewardConfig(normalize_rewards=True, reward_window=20),
    )
    env.reset()
    for _ in range(10):
        _obs, reward, terminated, _truncated, info = env.step(int(rng.choice([0, 1, 2])))
        assert np.isfinite(reward)
        assert "raw_reward" in info
        if terminated:
            break
