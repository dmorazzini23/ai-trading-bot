"""Smoke tests for RL training/inference parity."""

from __future__ import annotations


import numpy as np
import pytest




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
