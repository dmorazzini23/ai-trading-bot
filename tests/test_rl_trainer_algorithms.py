from tests.optdeps import require

require("numpy")

import json
from pathlib import Path

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


def test_rl_trainer_rejects_mismatched_price_series_length() -> None:
    trainer = train_mod.RLTrainer(algorithm="PPO")
    data = np.zeros((64, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="price_series length must match RL training rows"):
        trainer._create_environments(
            data,
            env_params={"price_series": np.linspace(1.0, 2.0, 32, dtype=np.float32)},
        )


def test_train_multi_seed_aggregates_metrics(monkeypatch, tmp_path: Path) -> None:
    init_calls: list[dict[str, object]] = []
    train_calls: list[dict[str, object]] = []

    class DummyTrainer:
        def __init__(
            self,
            *,
            algorithm: str,
            total_timesteps: int,
            eval_freq: int,
            early_stopping_patience: int,
            seed: int,
        ) -> None:
            self._seed = int(seed)
            init_calls.append(
                {
                    "algorithm": algorithm,
                    "total_timesteps": total_timesteps,
                    "eval_freq": eval_freq,
                    "early_stopping_patience": early_stopping_patience,
                    "seed": seed,
                }
            )

        def train(
            self,
            *,
            data: np.ndarray,
            env_params: dict[str, object],
            model_params: dict[str, object],
            save_path: str | None,
        ) -> dict[str, object]:
            train_calls.append(
                {
                    "seed": self._seed,
                    "data_shape": tuple(np.asarray(data).shape),
                    "env_params": dict(env_params),
                    "model_params": dict(model_params),
                    "save_path": save_path,
                }
            )
            return {
                "seed": self._seed,
                "final_evaluation": {
                    "mean_reward": float(self._seed),
                    "avg_episode_net_return": float(self._seed) / 10.0,
                    "constraint_violation_rate": float(self._seed) / 100.0,
                },
            }

    monkeypatch.setattr(train_mod, "RLTrainer", DummyTrainer)

    summary = train_mod.train_multi_seed(
        data=np.ones((50, 6), dtype=np.float32),
        seeds=[1, 3, 2],
        algorithm="PPO",
        total_timesteps=123,
        eval_freq=25,
        early_stopping_patience=4,
        env_params={"transaction_cost": 0.001},
        model_params={"learning_rate": 0.0005},
        save_root=str(tmp_path / "seed_runs"),
    )

    assert [call["seed"] for call in init_calls] == [1, 3, 2]
    assert [call["seed"] for call in train_calls] == [1, 3, 2]
    assert train_calls[0]["save_path"] == str(tmp_path / "seed_runs" / "seed_1")
    assert train_calls[1]["save_path"] == str(tmp_path / "seed_runs" / "seed_3")
    assert train_calls[2]["save_path"] == str(tmp_path / "seed_runs" / "seed_2")
    assert summary["run_count"] == 3
    assert summary["seeds"] == [1, 3, 2]
    assert summary["mean_reward_mean"] == pytest.approx(2.0)
    assert summary["mean_reward_median"] == pytest.approx(2.0)
    assert summary["mean_reward_min"] == pytest.approx(1.0)
    assert summary["mean_reward_max"] == pytest.approx(3.0)
    assert summary["avg_episode_net_return_mean"] == pytest.approx(0.2)
    assert summary["avg_episode_net_return_median"] == pytest.approx(0.2)
    assert summary["constraint_violation_rate_mean"] == pytest.approx(0.02)
    assert len(summary["runs"]) == 3


def test_rl_trainer_save_persists_registry_governance(monkeypatch, tmp_path: Path) -> None:
    import ai_trading.model_registry as registry_mod

    class DummyModel:
        def save(self, path: str) -> None:
            Path(path).write_bytes(b"rl-model")

    class DummyEvalCallback:
        eval_results = [{"mean_reward": 1.0}]

        def save_results(self, path: str) -> None:
            Path(path).write_text(json.dumps(self.eval_results), encoding="utf-8")

    captured: dict[str, object] = {}

    class DummyRegistry:
        def register_model(
            self,
            model: object,
            strategy: str,
            model_type: str,
            *,
            metadata: dict[str, object] | None = None,
            dataset_fingerprint: str | None = None,
            tags: list[str] | None = None,
            activate: bool = True,
        ) -> str:
            captured["register"] = {
                "model": model,
                "strategy": strategy,
                "model_type": model_type,
                "metadata": metadata,
                "dataset_fingerprint": dataset_fingerprint,
                "tags": list(tags or []),
                "activate": activate,
            }
            return "rl-registry-001"

        def update_governance_status(
            self,
            model_id: str,
            status: str,
            *,
            extra: dict[str, object] | None = None,
        ) -> None:
            captured["governance"] = {
                "model_id": model_id,
                "status": status,
                "extra": extra or {},
            }

    monkeypatch.setattr(registry_mod, "ModelRegistry", DummyRegistry)
    monkeypatch.setattr(
        train_mod.RLTrainer,
        "_resolve_governance_decision",
        lambda self: ("production", {"gates": {"reward": True}}),
    )

    trainer = train_mod.RLTrainer(algorithm="PPO", total_timesteps=777, seed=99)
    trainer.model = DummyModel()
    trainer.eval_callback = DummyEvalCallback()
    trainer.training_results = {
        "algorithm": "PPO",
        "final_evaluation": {
            "mean_reward": 1.5,
            "avg_episode_net_return": 0.2,
            "avg_episode_max_drawdown": 0.1,
            "constraint_violation_rate": 0.0,
            "constraint_termination_rate": 0.0,
        },
        "state_builder": {"enabled": True, "schema": "v1"},
    }
    trainer._artifact_context = {
        "register_model": True,
        "strategy": "rl_overlay",
        "model_type": "ppo",
        "dataset_fingerprint": "dataset-abc",
        "feature_spec_hash": "feature-def",
        "tags": ["after_hours"],
        "requested_governance_status": "production",
    }

    trainer._save_model_and_results(str(tmp_path))

    assert (tmp_path / "model_ppo.zip").exists()
    assert (tmp_path / "training_results.json").exists()
    assert (tmp_path / "evaluation_results.json").exists()
    assert (tmp_path / "meta.json").exists()

    saved_payload = json.loads((tmp_path / "training_results.json").read_text(encoding="utf-8"))
    assert saved_payload["model_id"] == "rl-registry-001"
    assert saved_payload["governance_status"] == "production"
    assert trainer.training_results["model_id"] == "rl-registry-001"
    assert trainer.training_results["governance_status"] == "production"

    register_call = captured["register"]
    assert isinstance(register_call, dict)
    assert register_call["strategy"] == "rl_overlay"
    assert register_call["model_type"] == "ppo"
    assert register_call["dataset_fingerprint"] == "dataset-abc"
    assert register_call["activate"] is True
    tags = register_call["tags"]
    assert isinstance(tags, list)
    assert "after_hours" in tags
    assert "rl" in tags
    assert "ppo" in tags

    governance_call = captured["governance"]
    assert isinstance(governance_call, dict)
    assert governance_call["model_id"] == "rl-registry-001"
    assert governance_call["status"] == "production"
