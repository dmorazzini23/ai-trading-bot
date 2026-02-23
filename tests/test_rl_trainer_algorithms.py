from tests.optdeps import require

require("numpy")

import numpy as np
import pytest

import ai_trading.rl_trading.env as env_mod
import ai_trading.rl_trading.train as train_mod


def test_resolve_algo_config_supports_continuous_algorithms() -> None:
    assert train_mod._resolve_algo_config("SAC").requires_continuous_actions is True
    assert train_mod._resolve_algo_config("td3").requires_continuous_actions is True


def test_rl_trainer_uses_continuous_action_space_for_sac(monkeypatch) -> None:
    created_action_types: list[str | None] = []

    class DummyEnv:
        def __init__(self, _data, **kwargs):
            action_cfg = kwargs.get("action_config")
            created_action_types.append(
                getattr(action_cfg, "action_type", None) if action_cfg is not None else None
            )

    monkeypatch.setattr(env_mod, "TradingEnv", DummyEnv)
    trainer = train_mod.RLTrainer(algorithm="SAC")
    trainer._create_environments(np.zeros((40, 4), dtype=float), env_params={})

    assert created_action_types
    assert all(action_type == "continuous" for action_type in created_action_types)


def test_rl_trainer_creates_td3_model(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class DummyTD3:
        def __init__(self, policy, env, **kwargs):
            captured["policy"] = policy
            captured["env"] = env
            captured["kwargs"] = kwargs

    monkeypatch.setattr(train_mod, "TD3", DummyTD3)
    trainer = train_mod.RLTrainer(algorithm="TD3", seed=123)
    trainer.train_env = object()
    trainer._create_model(
        model_params={
            "learning_rate": 0.002,
            "tensorboard_log": str(tmp_path / "tb"),
        }
    )

    assert isinstance(trainer.model, DummyTD3)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["learning_rate"] == 0.002
    assert kwargs["seed"] == 123


def test_rl_trainer_unknown_algorithm_raises() -> None:
    with pytest.raises(ValueError, match="Unknown algorithm"):
        train_mod._resolve_algo_config("NOT_REAL")


def test_rl_trainer_state_builder_wires_price_series(monkeypatch) -> None:
    captured: list[tuple[np.ndarray, np.ndarray | None]] = []

    class DummyVec(list):
        def __init__(self, env_fns):
            super().__init__([fn() for fn in env_fns])

    class DummyEnv:
        def __init__(self, data, **kwargs):
            captured.append((np.asarray(data), kwargs.get("price_series")))

    monkeypatch.setattr(train_mod, "DummyVecEnv", DummyVec)
    monkeypatch.setattr(env_mod, "TradingEnv", DummyEnv)

    n = 120
    close = np.linspace(100.0, 102.0, n, dtype=np.float32)
    raw = np.column_stack(
        [
            close - 0.1,  # open
            close + 0.2,  # high
            close - 0.3,  # low
            close,  # close
            np.linspace(1_000.0, 2_000.0, n, dtype=np.float32),  # volume
        ]
    )

    trainer = train_mod.RLTrainer(algorithm="PPO")
    trainer._create_environments(raw, env_params={"use_state_builder": True})

    assert len(captured) == 2
    train_states, train_prices = captured[0]
    eval_states, eval_prices = captured[1]
    assert train_states.shape[1] == 6
    assert eval_states.shape[1] == 6
    assert train_prices is not None and len(train_prices) == train_states.shape[0]
    assert eval_prices is not None and len(eval_prices) == eval_states.shape[0]
