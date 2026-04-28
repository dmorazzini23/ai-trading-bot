from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ai_trading.rl_trading import train as train_mod


def test_env_signature_helpers_and_config_validation() -> None:
    assert train_mod._reset_obs(SimpleNamespace(reset=lambda: ("obs", {"ignored": True}))) == "obs"
    assert train_mod._reset_obs(SimpleNamespace(reset=lambda: "raw")) == "raw"

    five_step = SimpleNamespace(step=lambda _action: ("obs2", "1.25", True, False, {"x": 1}))
    assert train_mod._step_env(five_step, 0) == ("obs2", 1.25, True, False, {"x": 1})

    four_step = SimpleNamespace(step=lambda _action: ("obs3", 2, False, "not-dict"))
    assert train_mod._step_env(four_step, 0) == ("obs3", 2.0, False, False, {})

    with pytest.raises(ValueError, match="non-tuple"):
        train_mod._step_env(SimpleNamespace(step=lambda _action: "bad"), 0)
    with pytest.raises(ValueError, match="unexpected environment step payload size"):
        train_mod._step_env(SimpleNamespace(step=lambda _action: ("too", "short")), 0)

    assert train_mod._normalise_tags([" rl ", "", "rl", "shadow"]) == ["rl", "shadow"]
    assert train_mod._resolve_algo_config("ppo").name == "PPO"
    with pytest.raises(ValueError, match="Unknown algorithm"):
        train_mod._resolve_algo_config("made-up")


def test_dataset_fingerprint_and_artifact_context_are_stable() -> None:
    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    first = train_mod._dataset_fingerprint_from_matrix(data, algorithm="ppo", seed=7)
    second = train_mod._dataset_fingerprint_from_matrix(data.copy(), algorithm="PPO", seed=7)

    assert first == second
    assert first != train_mod._dataset_fingerprint_from_matrix(data, algorithm="A2C", seed=7)

    trainer = train_mod.RLTrainer(algorithm="DQN", seed=7)
    context = trainer._build_artifact_context(
        data=data,
        prices=None,
        env_params={
            "register_model": False,
            "registry_tags": [" one ", "one", "two"],
            "registry_requested_status": "production",
        },
    )

    assert context["dataset_fingerprint"] == train_mod._dataset_fingerprint_from_matrix(
        data,
        algorithm="DQN",
        seed=7,
    )
    assert context["feature_spec_hash"]
    assert context["register_model"] is False
    assert context["tags"] == ["one", "two"]
    assert context["requested_governance_status"] == "production"


def test_early_stopping_callback_reads_model_rewards() -> None:
    class FakeBaseCallback:
        def __init__(self, verbose: int = 0) -> None:
            self.verbose = verbose
            self.training_env = None

    callback_cls = train_mod._build_early_stopping_callback(FakeBaseCallback)
    callback = callback_cls(patience=2, min_improvement=0.5)
    callback.model = SimpleNamespace(ep_info_buffer=[{"r": 10.0}, {"r": 12.0}])

    assert callback._on_rollout_end() is True
    assert callback.best_mean_reward == 11.0

    callback.model = SimpleNamespace(ep_info_buffer=[{"r": 11.0}])
    assert callback._on_rollout_end() is True
    assert callback.patience_counter == 1
    assert callback._on_rollout_end() is False


def test_final_evaluation_handles_missing_and_bad_env(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = train_mod.RLTrainer()
    assert trainer._final_evaluation() == {}

    trainer.model = SimpleNamespace(predict=lambda *_args, **_kwargs: (0, None))
    trainer.eval_env = SimpleNamespace(reset=lambda: "obs", step=lambda _action: "bad")
    monkeypatch.setattr(train_mod, "evaluate_policy", lambda *_args, **_kwargs: (1.0, 0.5))

    assert trainer._final_evaluation() == {}


def test_final_evaluation_keeps_episode_level_governance_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = train_mod.RLTrainer()
    trainer.model = SimpleNamespace(predict=lambda *_args, **_kwargs: (0, None))

    class EvalEnv:
        def __init__(self) -> None:
            self.steps = 0

        def reset(self):
            self.steps = 0
            return "obs"

        def step(self, _action):
            self.steps += 1
            info = {
                "turnover_penalty": 0.2,
                "drawdown_penalty": 0.1,
                "variance_penalty": 0.04,
                "drawdown": 0.08 if self.steps == 1 else 0.20,
            }
            if self.steps >= 2:
                info["episode_stats"] = {"total_return": 0.12, "max_drawdown": 0.20}
                return "obs", 1.0, True, False, info
            return "obs", 1.0, False, False, info

    trainer.eval_env = EvalEnv()
    monkeypatch.setattr(train_mod, "evaluate_policy", lambda *_args, **_kwargs: (1.0, 0.5))

    metrics = trainer._final_evaluation()

    assert metrics["avg_turnover_penalty"] == pytest.approx(0.2)
    assert metrics["avg_episode_net_return"] == pytest.approx(0.12)
    assert metrics["avg_episode_max_drawdown"] == pytest.approx(0.20)
