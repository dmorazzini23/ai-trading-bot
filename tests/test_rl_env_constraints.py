from __future__ import annotations

import types

import numpy as np

import ai_trading.rl_trading.env as env_mod


def _stub_gym_stack(monkeypatch) -> None:
    class _EnvBase:
        def __init__(self, *_a, **_k):
            pass

        def reset(self, *, seed=None):
            return None

    class _Discrete:
        def __init__(self, n):
            self.n = int(n)

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    dummy_gym = types.SimpleNamespace(
        Env=_EnvBase,
        spaces=types.SimpleNamespace(Discrete=_Discrete, Box=_Box),
    )
    monkeypatch.setattr(env_mod, "_load_rl_stack", lambda: {"gym": dummy_gym})


def _ohlcv_rows(n: int = 80) -> np.ndarray:
    close = np.linspace(100.0, 102.0, n, dtype=np.float32)
    open_px = close - 0.1
    high = close + 0.2
    low = close - 0.3
    volume = np.linspace(1_000.0, 2_000.0, n, dtype=np.float32)
    return np.column_stack([open_px, high, low, close, volume]).astype(np.float32)


def test_env_executes_fractional_trade_with_realistic_cash(monkeypatch):
    _stub_gym_stack(monkeypatch)
    data = _ohlcv_rows(120)
    env = env_mod.TradingEnv(
        data,
        window=10,
        action_config=env_mod.ActionSpaceConfig(action_type="continuous"),
        constraint_config=env_mod.ConstraintConfig(initial_cash=10_000.0),
    )
    env.reset()
    _, _, _, _, info = env.step(1.0)
    assert info["position"] > 0.0
    assert info["trade_units"] > 0.0
    assert info["constraint_violations"] == ()


def test_env_clips_turnover_when_constraint_is_tight(monkeypatch):
    _stub_gym_stack(monkeypatch)
    data = _ohlcv_rows(120)
    env = env_mod.TradingEnv(
        data,
        window=10,
        action_config=env_mod.ActionSpaceConfig(action_type="continuous"),
        constraint_config=env_mod.ConstraintConfig(
            initial_cash=20_000.0,
            max_turnover_per_step=0.10,
        ),
    )
    env.reset()
    _, _, _, _, info = env.step(1.0)
    assert info["trade_size"] <= 0.100001
    assert info["constraint_adjusted"] is True


def test_env_terminates_on_drawdown_violation(monkeypatch):
    _stub_gym_stack(monkeypatch)
    n = 80
    close = np.linspace(100.0, 40.0, n, dtype=np.float32)
    open_px = close + 0.2
    high = close + 0.3
    low = close - 0.4
    volume = np.linspace(1_000.0, 1_500.0, n, dtype=np.float32)
    data = np.column_stack([open_px, high, low, close, volume]).astype(np.float32)

    env = env_mod.TradingEnv(
        data,
        window=10,
        action_config=env_mod.ActionSpaceConfig(
            action_type="discrete",
            discrete_step=1.0,
        ),
        constraint_config=env_mod.ConstraintConfig(
            initial_cash=100_000.0,
            max_drawdown=0.02,
            terminate_on_violation=True,
        ),
    )
    env.reset()
    env.step(1)  # enter long exposure

    terminated = False
    violation_seen = False
    for _ in range(30):
        _, _, terminated, _, info = env.step(0)  # hold
        if info["constraint_violations"]:
            violation_seen = True
        if terminated:
            break

    assert violation_seen is True
    assert terminated is True
    assert info["constraint_terminated"] is True
