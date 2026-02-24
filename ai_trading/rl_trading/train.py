"""Enhanced RL training with reward shaping and evaluation callbacks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

try:  # optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - exercised via guard paths
    np = None


def _require_numpy(context: str) -> None:
    """Ensure NumPy is available before using NumPy-dependent helpers."""

    if np is None:
        raise ImportError(
            "NumPy is required for "
            f"{context}. Install the 'numpy' package or the 'ai-trading-bot[rl]' extras."
        )
from ai_trading.config.management import get_env
from ai_trading.logging import logger
from . import _load_rl_stack, is_rl_available


class _SB3Stub:
    """Minimal stub used when RL stack is unavailable."""

    def __init__(self, *a, **k):
        pass

    def learn(self, *a, **k):
        return self

    def save(self, *a, **k):
        pass

    @classmethod
    def load(cls, *a, **k):
        return cls()


PPO = A2C = DQN = SAC = TD3 = _SB3Stub


class BaseCallback:
    def __init__(self, *a, **k):
        pass


class EvalCallback(BaseCallback):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)


def make_vec_env(*a, **k):
    import types

    return types.SimpleNamespace()


class DummyVecEnv(list):
    pass


def evaluate_policy(*a, **k):
    return (0.0, 0.0)


@dataclass
class TrainingConfig:
    """Configuration for dummy RL training."""

    data: Any | None = None
    model_path: str | os.PathLike[str] | None = None
    timesteps: int = 0


@dataclass(frozen=True, slots=True)
class RLAlgoConfig:
    """Declarative algorithm config used by the trainer."""

    name: str
    policy: str = "MlpPolicy"
    requires_continuous_actions: bool = False
    default_params: tuple[tuple[str, Any], ...] = ()

    def merged_params(
        self,
        *,
        common: Mapping[str, Any],
        overrides: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(common)
        merged.update(dict(self.default_params))
        merged.update(dict(overrides or {}))
        return merged


RL_ALGO_CONFIGS: dict[str, RLAlgoConfig] = {
    "PPO": RLAlgoConfig(
        name="PPO",
        default_params=(
            ("learning_rate", 0.0003),
            ("n_steps", 2048),
            ("batch_size", 64),
            ("n_epochs", 10),
            ("gamma", 0.99),
            ("gae_lambda", 0.95),
            ("clip_range", 0.2),
            ("ent_coef", 0.0),
        ),
    ),
    "A2C": RLAlgoConfig(
        name="A2C",
        default_params=(
            ("learning_rate", 0.0007),
            ("n_steps", 20),
            ("gamma", 0.99),
            ("gae_lambda", 1.0),
            ("ent_coef", 0.0),
            ("vf_coef", 0.25),
        ),
    ),
    "DQN": RLAlgoConfig(
        name="DQN",
        default_params=(
            ("learning_rate", 0.0001),
            ("buffer_size", 50000),
            ("learning_starts", 1000),
            ("batch_size", 32),
            ("tau", 1.0),
            ("gamma", 0.99),
            ("train_freq", 4),
            ("gradient_steps", 1),
            ("target_update_interval", 1000),
            ("exploration_fraction", 0.1),
            ("exploration_initial_eps", 1.0),
            ("exploration_final_eps", 0.05),
        ),
    ),
    "SAC": RLAlgoConfig(
        name="SAC",
        requires_continuous_actions=True,
        default_params=(
            ("learning_rate", 0.0003),
            ("buffer_size", 100000),
            ("learning_starts", 1000),
            ("batch_size", 256),
            ("tau", 0.005),
            ("gamma", 0.99),
            ("train_freq", 1),
            ("gradient_steps", 1),
            ("ent_coef", "auto"),
        ),
    ),
    "TD3": RLAlgoConfig(
        name="TD3",
        requires_continuous_actions=True,
        default_params=(
            ("learning_rate", 0.001),
            ("buffer_size", 100000),
            ("learning_starts", 1000),
            ("batch_size", 100),
            ("tau", 0.005),
            ("gamma", 0.99),
            ("train_freq", (1, "step")),
            ("gradient_steps", 1),
            ("policy_delay", 2),
            ("target_policy_noise", 0.2),
            ("target_noise_clip", 0.5),
        ),
    ),
}

_ARTIFACT_ENV_PARAM_KEYS: tuple[str, ...] = (
    "register_model",
    "registry_strategy",
    "registry_model_type",
    "registry_tags",
    "registry_requested_status",
    "dataset_fingerprint",
    "feature_spec_hash",
)


def _resolve_algo_config(algorithm: str) -> RLAlgoConfig:
    normalized = str(algorithm or "").strip().upper()
    config = RL_ALGO_CONFIGS.get(normalized)
    if config is None:
        supported = ", ".join(sorted(RL_ALGO_CONFIGS))
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Supported algorithms: {supported}")
    return config


class Model:
    """Minimal stand-in for an RL model used in tests."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def predict(self, state: Any, deterministic: bool = True) -> tuple[int, None]:
        # Default to hold when running in stub mode to avoid accidental buys.
        return (0, None)

    def save(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_bytes(b"0")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Model":
        return cls(TrainingConfig(model_path=str(path)))


def train(
    data: Any,
    model_path: str | os.PathLike[str],
    timesteps: int = 0,
) -> Model:
    """Train a minimal RL model and save it.

    Parameters
    ----------
    data:
        Training data (unused but kept for API compatibility).
    model_path:
        Where to save the trained model.
    timesteps:
        Number of timesteps passed to the underlying algorithm's ``learn`` method.
    """

    config = TrainingConfig(data=data, model_path=str(model_path), timesteps=timesteps)
    model = Model(config)
    algo = PPO("MlpPolicy", data)
    algo.learn(total_timesteps=timesteps)
    # Prefer the algorithm's save method when available; fall back to the
    # minimal stub model to keep the interface consistent in tests.
    if hasattr(algo, "save"):
        algo.save(str(model_path))
    elif config.model_path:
        model.save(config.model_path)
    return model


def _ensure_rl() -> bool:
    """Import the RL stack on demand, replacing global stubs."""
    global PPO, A2C, DQN, SAC, TD3, BaseCallback, EvalCallback, make_vec_env, evaluate_policy, DummyVecEnv
    if PPO is not _SB3Stub:
        _refresh_callback_classes()
        return True
    if not is_rl_available():
        return False
    stack = _load_rl_stack()
    sb3 = stack["sb3"]
    PPO = sb3.PPO
    A2C = sb3.A2C
    DQN = sb3.DQN
    SAC = sb3.SAC
    TD3 = sb3.TD3
    BaseCallback = sb3.common.callbacks.BaseCallback
    EvalCallback = sb3.common.callbacks.EvalCallback
    make_vec_env = sb3.common.env_util.make_vec_env
    evaluate_policy = sb3.common.evaluation.evaluate_policy
    DummyVecEnv = sb3.common.vec_env.DummyVecEnv
    _refresh_callback_classes()
    return True


def _reset_obs(env: Any) -> Any:
    """Reset an environment and return the observation only."""

    payload = env.reset()
    if isinstance(payload, tuple):
        return payload[0]
    return payload


def _step_env(env: Any, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Execute one environment step supporting gym and gymnasium signatures."""

    payload = env.step(action)
    if not isinstance(payload, tuple):
        raise ValueError("environment step returned non-tuple payload")
    if len(payload) == 5:
        obs, reward, terminated, truncated, info = payload
    elif len(payload) == 4:
        obs, reward, done, info = payload
        terminated = bool(done)
        truncated = False
    else:
        raise ValueError(f"unexpected environment step payload size: {len(payload)}")
    if not isinstance(info, dict):
        info = {}
    return obs, float(reward), bool(terminated), bool(truncated), info


def _normalise_tags(raw_tags: Sequence[str] | None) -> list[str]:
    if not raw_tags:
        return []
    tags: list[str] = []
    for item in raw_tags:
        token = str(item).strip()
        if token and token not in tags:
            tags.append(token)
    return tags


def _dataset_fingerprint_from_matrix(
    matrix: np.ndarray,
    *,
    algorithm: str,
    seed: int,
) -> str:
    _require_numpy("dataset fingerprint")
    payload = {
        "shape": list(matrix.shape),
        "algorithm": str(algorithm).upper(),
        "seed": int(seed),
        "sha_data": hashlib.sha256(matrix.tobytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _build_early_stopping_callback(base_cls: type[Any]) -> type[Any]:
    class EarlyStoppingCallbackImpl(base_cls):
        """
        Early stopping callback for RL training.

        Stops training when performance doesn't improve for a specified
        number of evaluations.
        """

        def __init__(self, patience: int = 10, min_improvement: float = 0.01, verbose: int = 0):
            _require_numpy("EarlyStoppingCallback")
            super().__init__(verbose)
            self.patience = patience
            self.min_improvement = min_improvement
            self.best_mean_reward = -np.inf
            self.patience_counter = 0

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> bool:
            """Called at the end of each rollout."""
            _require_numpy("EarlyStoppingCallback rollout handling")
            if hasattr(self.training_env, "get_attr"):
                try:
                    env_rewards = self.training_env.get_attr("episode_returns")
                    if env_rewards and len(env_rewards[0]) > 0:
                        current_mean_reward = np.mean(env_rewards[0][-10:])
                        if current_mean_reward > self.best_mean_reward + self.min_improvement:
                            self.best_mean_reward = current_mean_reward
                            self.patience_counter = 0
                            if self.verbose > 0:
                                logger.info(f"New best reward: {self.best_mean_reward:.4f}")
                        else:
                            self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            if self.verbose > 0:
                                logger.info(
                                    f"Early stopping after {self.patience} evaluations without improvement"
                                )
                            return False
                except (AttributeError, TypeError, ValueError) as e:  # env may lack returns or contain bad data
                    if self.verbose > 0:
                        logger.warning(f"Error in early stopping callback: {e}")
            return True

    EarlyStoppingCallbackImpl.__name__ = "EarlyStoppingCallback"
    EarlyStoppingCallbackImpl.__qualname__ = "EarlyStoppingCallback"
    return EarlyStoppingCallbackImpl


def _build_detailed_eval_callback(base_cls: type[Any]) -> type[Any]:
    class DetailedEvalCallbackImpl(base_cls):
        """
        Enhanced evaluation callback with detailed metrics tracking.
        """

        def __init__(
            self,
            eval_env,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            deterministic: bool = True,
            save_path: str | None = None,
            verbose: int = 1,
        ):
            _require_numpy("DetailedEvalCallback")
            super().__init__(verbose)
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.deterministic = deterministic
            self.save_path = save_path
            self.eval_results = []
            self.best_mean_reward = -np.inf

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                self._evaluate_model()
            return True

        def _evaluate_model(self) -> None:
            """Run detailed evaluation."""
            _require_numpy("Detailed evaluation metrics")
            try:
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                )
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_length = np.mean(episode_lengths)
                detailed_metrics = self._collect_detailed_metrics()
                eval_result = {
                    "timestep": self.n_calls,
                    "mean_reward": float(mean_reward),
                    "std_reward": float(std_reward),
                    "mean_episode_length": float(mean_length),
                    "timestamp": datetime.now(UTC).isoformat(),
                    **detailed_metrics,
                }
                self.eval_results.append(eval_result)
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.save_path:
                        best_model_path = os.path.join(self.save_path, "best_model.zip")
                        self.model.save(best_model_path)
                        meta_path = os.path.join(self.save_path, "best_model_meta.json")
                        with open(meta_path, "w") as f:
                            json.dump(eval_result, f, indent=2)
                if self.verbose > 0:
                    logger.info(
                        f"Eval at step {self.n_calls}: mean_reward={mean_reward:.4f} ± {std_reward:.4f}"
                    )
            except (OSError, AttributeError, TypeError, ValueError) as e:  # file I/O or numeric issues during evaluation
                logger.error(f"Error in evaluation: {e}")

        def _collect_detailed_metrics(self) -> dict[str, float]:
            """Collect detailed performance metrics."""
            try:
                obs = _reset_obs(self.eval_env)
                total_reward = 0
                total_turnover = 0
                total_drawdown = 0
                total_variance = 0
                total_constraint_violations = 0
                episode_length = 0
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, terminated, truncated, info = _step_env(self.eval_env, action)
                    total_reward += reward
                    episode_length += 1
                    if isinstance(info, dict):
                        total_turnover += info.get("turnover_penalty", 0)
                        total_drawdown += info.get("drawdown_penalty", 0)
                        total_variance += info.get("variance_penalty", 0)
                        total_constraint_violations += len(info.get("constraint_violations", ()) or ())
                    done = terminated or truncated
                avg_turnover = total_turnover / episode_length if episode_length > 0 else 0
                avg_drawdown = total_drawdown / episode_length if episode_length > 0 else 0
                avg_variance = total_variance / episode_length if episode_length > 0 else 0
                violation_rate = float(total_constraint_violations) / float(max(1, episode_length))
                return {
                    "avg_turnover_penalty": float(avg_turnover),
                    "avg_drawdown_penalty": float(avg_drawdown),
                    "avg_variance_penalty": float(avg_variance),
                    "constraint_violation_rate": float(violation_rate),
                    "sharpe_ratio": self._calculate_sharpe_ratio(),
                    "max_drawdown": float(total_drawdown) if total_drawdown > 0 else 0.0,
                }
            except (AttributeError, TypeError, ValueError) as e:  # model or env returned unexpected values
                logger.error(f"Error collecting detailed metrics: {e}")
                return {}

        def _calculate_sharpe_ratio(self) -> float:
            """Calculate approximate Sharpe ratio from recent evaluations."""
            _require_numpy("Sharpe ratio calculation")
            try:
                if len(self.eval_results) < 2:
                    return 0.0
                recent_rewards = [r["mean_reward"] for r in self.eval_results[-10:]]
                if len(recent_rewards) > 1:
                    mean_return = np.mean(recent_rewards)
                    std_return = np.std(recent_rewards)
                    return float(mean_return / std_return) if std_return > 0 else 0.0
                return 0.0
            except (KeyError, TypeError, ValueError, ZeroDivisionError):  # reward history missing or invalid
                return 0.0

        def save_results(self, path: str) -> None:
            """Save evaluation results."""
            try:
                with open(path, "w") as f:
                    json.dump(self.eval_results, f, indent=2)
                logger.info(f"Evaluation results saved to {path}")
            except (OSError, TypeError, ValueError) as e:  # disk or serialization issues
                logger.error(f"Error saving evaluation results: {e}")

    DetailedEvalCallbackImpl.__name__ = "DetailedEvalCallback"
    DetailedEvalCallbackImpl.__qualname__ = "DetailedEvalCallback"
    return DetailedEvalCallbackImpl


def _refresh_callback_classes() -> None:
    """Rebuild callbacks when ``BaseCallback`` changes at runtime."""

    global EarlyStoppingCallback, DetailedEvalCallback
    if not isinstance(BaseCallback, type):
        return
    try:
        early_compatible = isinstance(EarlyStoppingCallback, type) and issubclass(
            EarlyStoppingCallback, BaseCallback
        )
    except TypeError:
        early_compatible = False
    try:
        detailed_compatible = isinstance(DetailedEvalCallback, type) and issubclass(
            DetailedEvalCallback, BaseCallback
        )
    except TypeError:
        detailed_compatible = False
    if not early_compatible:
        EarlyStoppingCallback = _build_early_stopping_callback(BaseCallback)
    if not detailed_compatible:
        DetailedEvalCallback = _build_detailed_eval_callback(BaseCallback)


EarlyStoppingCallback = _build_early_stopping_callback(BaseCallback)
DetailedEvalCallback = _build_detailed_eval_callback(BaseCallback)

class RLTrainer:
    """
    Enhanced RL trainer with reward shaping and evaluation.
    """

    def __init__(self, algorithm: str='PPO', total_timesteps: int=100000, eval_freq: int=10000, early_stopping_patience: int=10, seed: int=42):
        """
        Initialize RL trainer.

        Args:
            algorithm: RL algorithm ('PPO', 'A2C', 'DQN', 'SAC', 'TD3')
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            early_stopping_patience: Early stopping patience
            seed: Random seed
        """
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.model = None
        self.train_env = None
        self.eval_env = None
        self.state_builder = None
        self._raw_data: np.ndarray | None = None
        self._raw_prices: np.ndarray | None = None
        self._artifact_env_overrides: dict[str, Any] = {}
        self._artifact_context: dict[str, Any] = {}
        self.training_results = {}
        self.eval_callback = None
        logger.info(f'RLTrainer initialized with {algorithm} algorithm')

    @staticmethod
    def _pop_artifact_env_params(params: dict[str, Any]) -> dict[str, Any]:
        """Extract non-environment artifact keys from env params."""

        extracted: dict[str, Any] = {}
        for key in _ARTIFACT_ENV_PARAM_KEYS:
            if key in params:
                extracted[key] = params.pop(key)
        return extracted

    def _build_artifact_context(
        self,
        *,
        data: np.ndarray,
        prices: np.ndarray | None,
        env_params: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        params = dict(env_params or {})
        tags = _normalise_tags(params.pop("registry_tags", None))
        dataset_fingerprint = str(params.pop("dataset_fingerprint", "") or "").strip()
        strategy = str(params.pop("registry_strategy", "rl_overlay") or "rl_overlay").strip()
        model_type = str(params.pop("registry_model_type", self.algorithm.lower()) or self.algorithm.lower()).strip()
        requested_governance_status = str(
            params.pop("registry_requested_status", "shadow") or "shadow"
        ).strip().lower()
        register_model = bool(params.pop("register_model", True))
        feature_spec_hash = str(params.pop("feature_spec_hash", "") or "").strip()
        if not dataset_fingerprint:
            dataset_fingerprint = _dataset_fingerprint_from_matrix(
                data,
                algorithm=self.algorithm,
                seed=self.seed,
            )
        if not feature_spec_hash:
            feature_payload = {
                "columns": int(data.shape[1]),
                "state_builder": (
                    self.state_builder.describe()
                    if self.state_builder is not None and hasattr(self.state_builder, "describe")
                    else {"schema": "raw"}
                ),
            }
            feature_spec_hash = hashlib.sha256(
                json.dumps(feature_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
        return {
            "dataset_fingerprint": dataset_fingerprint,
            "feature_spec_hash": feature_spec_hash,
            "strategy": strategy,
            "model_type": model_type,
            "register_model": register_model,
            "tags": tags,
            "requested_governance_status": requested_governance_status or "shadow",
        }

    def _resolve_governance_decision(self) -> tuple[str, dict[str, Any]]:
        final_eval = self.training_results.get("final_evaluation", {})
        if not isinstance(final_eval, Mapping):
            final_eval = {}
        mean_reward = float(final_eval.get("mean_reward", 0.0) or 0.0)
        avg_net_return = float(final_eval.get("avg_episode_net_return", 0.0) or 0.0)
        avg_max_drawdown = float(final_eval.get("avg_episode_max_drawdown", 0.0) or 0.0)
        violation_rate = float(final_eval.get("constraint_violation_rate", 0.0) or 0.0)
        termination_rate = float(final_eval.get("constraint_termination_rate", 0.0) or 0.0)

        min_reward = float(get_env("AI_TRADING_RL_MIN_MEAN_REWARD", 0.0, cast=float))
        min_net_return = float(get_env("AI_TRADING_RL_MIN_AVG_NET_RETURN", 0.0, cast=float))
        max_drawdown = float(get_env("AI_TRADING_RL_MAX_EVAL_DRAWDOWN", 0.25, cast=float))
        max_violation_rate = float(
            get_env("AI_TRADING_RL_MAX_CONSTRAINT_VIOLATION_RATE", 0.0, cast=float)
        )
        max_termination_rate = float(
            get_env("AI_TRADING_RL_MAX_CONSTRAINT_TERMINATION_RATE", 0.0, cast=float)
        )

        gates = {
            "reward": mean_reward >= min_reward,
            "net_return": avg_net_return >= min_net_return,
            "drawdown": avg_max_drawdown <= max_drawdown,
            "constraint_violations": violation_rate <= max_violation_rate,
            "constraint_terminations": termination_rate <= max_termination_rate,
        }
        all_pass = all(gates.values())
        requested = str(self._artifact_context.get("requested_governance_status", "shadow")).lower()
        if requested == "production" and all_pass:
            status = "production"
        elif all_pass and bool(get_env("AI_TRADING_RL_AUTO_PROMOTE", False, cast=bool)):
            status = "production"
        else:
            status = "shadow"
        details = {
            "gates": gates,
            "thresholds": {
                "min_reward": min_reward,
                "min_net_return": min_net_return,
                "max_drawdown": max_drawdown,
                "max_violation_rate": max_violation_rate,
                "max_termination_rate": max_termination_rate,
            },
            "metrics": {
                "mean_reward": mean_reward,
                "avg_net_return": avg_net_return,
                "avg_max_drawdown": avg_max_drawdown,
                "constraint_violation_rate": violation_rate,
                "constraint_termination_rate": termination_rate,
            },
        }
        return status, details

    def train(self, data: np.ndarray, env_params: dict[str, Any] | None=None, model_params: dict[str, Any] | None=None, save_path: str | None=None) -> dict[str, Any]:
        """
        Train RL model with enhanced reward shaping.

        Args:
            data: Training data
            env_params: Environment parameters
            model_params: Model parameters
            save_path: Path to save model and results

        Returns:
            Training results
        """
        _require_numpy("RLTrainer training")
        if not _ensure_rl():
            logger.warning('Stable-baselines3 not available - returning dummy results')
            return {'training_time': 0.0, 'final_evaluation': {'mean_reward': 0.0, 'std_reward': 0.0}, 'total_timesteps': 0, 'algorithm': self.algorithm}
        try:
            env_params_local = dict(env_params or {})
            model_params_local = dict(model_params or {})
            self._artifact_env_overrides = self._pop_artifact_env_params(env_params_local)
            logger.info(f'Starting RL training with {len(data)} data points')
            self._create_environments(data, env_params_local)
            self._create_model(model_params_local)
            callbacks = self._setup_callbacks(save_path)
            start_time = datetime.now(UTC)
            progress_bar = bool(get_env("AI_TRADING_RL_PROGRESS_BAR", False, cast=bool))
            if progress_bar:
                if importlib.util.find_spec("tqdm") is None or importlib.util.find_spec("rich") is None:
                    progress_bar = False
                    logger.warning("RL_PROGRESS_BAR_UNAVAILABLE")
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                progress_bar=progress_bar,
            )
            end_time = datetime.now(UTC)
            artifact_context = self._build_artifact_context(
                data=self._raw_data if self._raw_data is not None else np.asarray(data, dtype=np.float32),
                prices=self._raw_prices,
                env_params=self._artifact_env_overrides,
            )
            self._artifact_context = artifact_context
            state_builder_summary = (
                self.state_builder.describe()
                if self.state_builder is not None and hasattr(self.state_builder, "describe")
                else {"enabled": False}
            )
            env_summary: dict[str, Any] = {}
            for key, value in env_params_local.items():
                if key == "price_series":
                    env_summary["price_series"] = f"array[{len(np.asarray(value).reshape(-1))}]"
                else:
                    env_summary[str(key)] = value
            self.training_results = {
                'algorithm': self.algorithm,
                'total_timesteps': self.total_timesteps,
                'training_time_seconds': (end_time - start_time).total_seconds(),
                'seed': self.seed,
                'final_evaluation': self._final_evaluation(),
                'env_params': env_summary,
                'model_params': model_params_local,
                'state_builder': state_builder_summary,
                'dataset_fingerprint': artifact_context.get("dataset_fingerprint"),
                'feature_spec_hash': artifact_context.get("feature_spec_hash"),
            }
            if save_path:
                self._save_model_and_results(save_path)
            logger.info('RL training completed successfully')
            return self.training_results
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:  # failures during env/model setup or learning
            logger.error(f'Error in RL training: {e}')
            raise

    def _create_environments(self, data: np.ndarray, env_params: dict[str, Any] | None) -> None:
        """Create training and evaluation environments."""
        try:
            from .env import ActionSpaceConfig, TradingEnv  # noqa: E402 - local import
            from .state_builder import MarketStateBuilder, StateBuilderConfig  # noqa: E402 - local import
            env_class: type = TradingEnv
            stack = _load_rl_stack()
            if stack is not None:
                gym = stack["gym"]
                if not issubclass(env_class, gym.Env):
                    env_class = type(
                        "SB3TradingEnv",
                        (env_class, gym.Env),
                        {},
                    )

            matrix = np.asarray(data, dtype=np.float32)
            if matrix.ndim != 2:
                raise ValueError("RL training data must be a 2D matrix")
            env_params = dict(env_params or {})
            self._pop_artifact_env_params(env_params)
            price_series_raw = env_params.pop("price_series", None)
            if price_series_raw is None:
                prices = np.asarray(matrix[:, 0], dtype=np.float32).reshape(-1)
            else:
                prices = np.asarray(price_series_raw, dtype=np.float32).reshape(-1)
            if prices.shape[0] != matrix.shape[0]:
                raise ValueError(
                    "price_series length must match RL training rows"
                )
            prices = np.clip(np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0), 1e-06, None)

            self._raw_data = matrix.copy()
            self._raw_prices = prices.copy()
            window = max(int(env_params.get("window", 10)), 1)
            min_rows = max(2 * (window + 1), 20)
            if len(matrix) < min_rows:
                raise ValueError(
                    f"RL training data must include at least {min_rows} rows for window={window}"
                )

            split_idx = int(len(matrix) * 0.8)
            split_idx = max(window + 1, min(split_idx, len(matrix) - (window + 1)))
            raw_train = matrix[:split_idx]
            raw_eval = matrix[split_idx:]
            raw_train_prices = prices[:split_idx]
            raw_eval_prices = prices[split_idx:]
            if len(raw_train) <= window or len(raw_eval) <= window:
                raise ValueError("RL train/eval split is too small for configured window")

            builder_enabled = bool(env_params.pop("use_state_builder", True))
            builder_config_raw = env_params.pop("state_builder_config", None)
            if isinstance(builder_config_raw, StateBuilderConfig):
                builder_config = builder_config_raw
            elif isinstance(builder_config_raw, Mapping):
                builder_config = StateBuilderConfig(**dict(builder_config_raw))
            else:
                builder_config = StateBuilderConfig()

            self.state_builder = None
            if builder_enabled:
                builder = MarketStateBuilder(builder_config)
                train_data = builder.fit_transform(raw_train)
                eval_data = builder.transform(raw_eval)
                self.state_builder = builder
            else:
                train_data = raw_train
                eval_data = raw_eval

            enhanced_env_params = {'transaction_cost': 0.001, 'slippage': 0.0005, 'half_spread': 0.0002, **env_params}
            algo_config = _resolve_algo_config(self.algorithm)
            if (
                algo_config.requires_continuous_actions
                and "action_config" not in enhanced_env_params
            ):
                enhanced_env_params["action_config"] = ActionSpaceConfig(action_type="continuous")

            def make_train_env():
                return env_class(train_data, price_series=raw_train_prices, **enhanced_env_params)

            def make_eval_env():
                return env_class(eval_data, price_series=raw_eval_prices, **enhanced_env_params)
            self.train_env = DummyVecEnv([make_train_env])
            self.eval_env = make_eval_env()
            logger.debug(
                "RL_ENVIRONMENTS_CREATED",
                extra={
                    "train_rows": len(train_data),
                    "eval_rows": len(eval_data),
                    "state_builder_enabled": bool(self.state_builder is not None),
                    "state_builder_schema": (
                        self.state_builder.describe().get("schema")
                        if self.state_builder is not None
                        else "raw"
                    ),
                },
            )
        except (ImportError, AttributeError, TypeError, ValueError) as e:  # TradingEnv missing or params invalid
            logger.error(f'Error creating environments: {e}')
            raise

    def _create_model(self, model_params: dict[str, Any] | None) -> None:
        """Create RL model."""
        try:
            model_params = dict(model_params or {})
            algo_config = _resolve_algo_config(self.algorithm)
            default_params: dict[str, Any] = {"verbose": 1, "seed": self.seed}
            tensorboard_available = importlib.util.find_spec("tensorboard") is not None
            requested_tensorboard = "tensorboard_log" in model_params
            if requested_tensorboard and not tensorboard_available:
                model_params.pop("tensorboard_log", None)
                logger.warning("RL_TENSORBOARD_UNAVAILABLE")
            if tensorboard_available:
                if "tensorboard_log" in model_params:
                    raw_log_dir = str(model_params.get("tensorboard_log"))
                    tensorboard_path = Path(raw_log_dir).expanduser()
                    if not tensorboard_path.is_absolute():
                        from ai_trading.paths import OUTPUT_DIR

                        tensorboard_path = (OUTPUT_DIR / tensorboard_path).resolve()
                else:
                    from ai_trading.paths import OUTPUT_DIR

                    tensorboard_path = (OUTPUT_DIR / "tensorboard").resolve()
                tensorboard_path.mkdir(parents=True, exist_ok=True)
                default_params["tensorboard_log"] = str(tensorboard_path)
            final_params = algo_config.merged_params(common=default_params, overrides=model_params)
            algo_klass_map = {
                "PPO": PPO,
                "A2C": A2C,
                "DQN": DQN,
                "SAC": SAC,
                "TD3": TD3,
            }
            algo_klass = algo_klass_map.get(algo_config.name)
            if algo_klass is None:
                raise ValueError(f'Unknown algorithm: {self.algorithm}')
            self.model = algo_klass(algo_config.policy, self.train_env, **final_params)
            logger.debug(f'Model created: {self.algorithm} with {len(final_params)} parameters')
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:  # invalid algorithm or params
            logger.error(f'Error creating model: {e}')
            raise

    def _setup_callbacks(self, save_path: str | None) -> list[BaseCallback]:
        """Setup training callbacks."""
        try:
            _refresh_callback_classes()
            if (
                not hasattr(BaseCallback, "init_callback")
                or not issubclass(EarlyStoppingCallback, BaseCallback)
                or not issubclass(DetailedEvalCallback, BaseCallback)
            ):
                logger.warning("RL_CALLBACK_COMPAT_DISABLED")
                self.eval_callback = None
                return []
            callbacks = []
            early_stopping = EarlyStoppingCallback(patience=self.early_stopping_patience, verbose=1)
            callbacks.append(early_stopping)
            self.eval_callback = DetailedEvalCallback(eval_env=self.eval_env, eval_freq=self.eval_freq, n_eval_episodes=5, deterministic=True, save_path=save_path, verbose=1)
            callbacks.append(self.eval_callback)
            return callbacks
        except (AttributeError, TypeError, ValueError) as e:  # callback or env misconfiguration
            logger.error(f'Error setting up callbacks: {e}')
            return []

    def _final_evaluation(self) -> dict[str, float]:
        """Perform final comprehensive evaluation."""
        _require_numpy("final evaluation metrics")
        try:
            if self.model is None or self.eval_env is None:
                return {}
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=10, deterministic=True)
            detailed_results = []
            for _ in range(10):
                obs = _reset_obs(self.eval_env)
                episode_reward = 0.0
                episode_metrics = {
                    'turnover': 0.0,
                    'drawdown': 0.0,
                    'variance': 0.0,
                    'constraint_violations': 0.0,
                    'constraint_terminations': 0.0,
                    'net_return': 0.0,
                    'max_drawdown': 0.0,
                }
                steps = 0
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = _step_env(self.eval_env, action)
                    episode_reward += reward
                    steps += 1
                    if isinstance(info, dict):
                        episode_metrics['turnover'] += info.get('turnover_penalty', 0)
                        episode_metrics['drawdown'] += info.get('drawdown_penalty', 0)
                        episode_metrics['variance'] += info.get('variance_penalty', 0)
                        episode_metrics['constraint_violations'] += float(
                            len(info.get('constraint_violations', ()) or ())
                        )
                        episode_metrics['constraint_terminations'] += 1.0 if info.get('constraint_terminated') else 0.0
                        try:
                            episode_metrics['max_drawdown'] = max(
                                float(episode_metrics['max_drawdown']),
                                float(info.get('drawdown', 0.0) or 0.0),
                            )
                        except (TypeError, ValueError):
                            pass
                        stats = info.get("episode_stats")
                        if isinstance(stats, Mapping):
                            try:
                                episode_metrics['net_return'] = float(
                                    stats.get('total_return', episode_metrics['net_return']) or episode_metrics['net_return']
                                )
                            except (TypeError, ValueError):
                                pass
                    done = terminated or truncated
                for key in episode_metrics:
                    episode_metrics[key] = episode_metrics[key] / steps if steps > 0 else 0
                detailed_results.append({'total_reward': episode_reward, **episode_metrics})
            final_metrics = {
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'avg_turnover_penalty': float(np.mean([r['turnover'] for r in detailed_results])),
                'avg_drawdown_penalty': float(np.mean([r['drawdown'] for r in detailed_results])),
                'avg_variance_penalty': float(np.mean([r['variance'] for r in detailed_results])),
                'avg_episode_net_return': float(np.mean([r['net_return'] for r in detailed_results])),
                'avg_episode_max_drawdown': float(np.mean([r['max_drawdown'] for r in detailed_results])),
                'constraint_violation_rate': float(np.mean([r['constraint_violations'] for r in detailed_results])),
                'constraint_termination_rate': float(np.mean([r['constraint_terminations'] for r in detailed_results])),
                'reward_stability': float(1.0 / (1.0 + std_reward)),
            }
            return final_metrics
        except (AttributeError, TypeError, ValueError) as e:  # evaluation failed due to bad env or metrics
            logger.error(f'Error in final evaluation: {e}')
            return {}

    def _save_model_and_results(self, save_path: str) -> None:
        """Save trained model and results."""
        try:
            os.makedirs(save_path, exist_ok=True)
            model_path = os.path.join(save_path, f'model_{self.algorithm.lower()}.zip')
            self.model.save(model_path)
            results_path = os.path.join(save_path, 'training_results.json')
            with open(results_path, 'w') as f:
                json.dump(self.training_results, f, indent=2, default=str)
            eval_path: str | None = None
            if self.eval_callback and hasattr(self.eval_callback, 'eval_results'):
                eval_path = os.path.join(save_path, 'evaluation_results.json')
                self.eval_callback.save_results(eval_path)
            metadata = {'algorithm': self.algorithm, 'training_timestamp': datetime.now(UTC).isoformat(), 'total_timesteps': self.total_timesteps, 'seed': self.seed, 'model_file': os.path.basename(model_path)}
            meta_path = os.path.join(save_path, 'meta.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            if bool(self._artifact_context.get("register_model", True)):
                try:
                    from ai_trading.model_registry import ModelRegistry

                    registry = ModelRegistry()
                    descriptor = {
                        "artifact_kind": "sb3_zip",
                        "artifact_path": str(model_path),
                        "results_path": str(results_path),
                        "evaluation_path": str(eval_path) if eval_path else "",
                        "metadata_path": str(meta_path),
                    }
                    model_id = registry.register_model(
                        descriptor,
                        strategy=str(self._artifact_context.get("strategy", "rl_overlay")),
                        model_type=str(self._artifact_context.get("model_type", self.algorithm.lower())),
                        metadata={
                            "algorithm": self.algorithm,
                            "seed": self.seed,
                            "total_timesteps": self.total_timesteps,
                            "training_results": self.training_results,
                            "dataset_fingerprint": self._artifact_context.get("dataset_fingerprint"),
                            "feature_spec_hash": self._artifact_context.get("feature_spec_hash"),
                            "state_builder": self.training_results.get("state_builder"),
                            "paths": descriptor,
                        },
                        dataset_fingerprint=str(
                            self._artifact_context.get("dataset_fingerprint", "")
                        )
                        or None,
                        tags=_normalise_tags(
                            list(self._artifact_context.get("tags", []))
                            + ["rl", self.algorithm.lower()]
                        ),
                        activate=True,
                    )
                    governance_status, governance_extra = self._resolve_governance_decision()
                    registry.update_governance_status(
                        model_id,
                        governance_status,
                        extra=governance_extra,
                    )
                    self.training_results["model_id"] = model_id
                    self.training_results["governance_status"] = governance_status
                    with open(results_path, 'w') as f:
                        json.dump(self.training_results, f, indent=2, default=str)
                except (
                    ImportError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    OSError,
                    RuntimeError,
                ) as exc:  # pragma: no cover - governance errors should not delete artifacts
                    logger.warning("RL_MODEL_REGISTRY_UPDATE_FAILED: %s", exc)

            logger.info(f'Model and results saved to {save_path}')
        except (OSError, AttributeError, TypeError, ValueError) as e:  # file or serialization problems
            logger.error(f'Error saving model and results: {e}')


def train_multi_seed(
    data: np.ndarray,
    *,
    seeds: Sequence[int],
    algorithm: str = "PPO",
    total_timesteps: int = 100000,
    eval_freq: int = 10000,
    early_stopping_patience: int = 10,
    env_params: Mapping[str, Any] | None = None,
    model_params: Mapping[str, Any] | None = None,
    save_root: str | None = None,
) -> dict[str, Any]:
    """Run RL training over multiple seeds and return aggregate robustness metrics."""

    _require_numpy("train_multi_seed")
    seeds_list = [int(seed) for seed in seeds]
    if not seeds_list:
        raise ValueError("train_multi_seed requires at least one seed")

    runs: list[dict[str, Any]] = []
    for seed in seeds_list:
        trainer = RLTrainer(
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            early_stopping_patience=early_stopping_patience,
            seed=seed,
        )
        run_save_path = None
        if save_root:
            run_save_path = str(Path(save_root) / f"seed_{seed}")
        result = trainer.train(
            data=np.asarray(data, dtype=np.float32),
            env_params=dict(env_params or {}),
            model_params=dict(model_params or {}),
            save_path=run_save_path,
        )
        final_eval = result.get("final_evaluation", {}) if isinstance(result, Mapping) else {}
        runs.append(
            {
                "seed": seed,
                "mean_reward": float(final_eval.get("mean_reward", 0.0) or 0.0),
                "avg_episode_net_return": float(
                    final_eval.get("avg_episode_net_return", 0.0) or 0.0
                ),
                "constraint_violation_rate": float(
                    final_eval.get("constraint_violation_rate", 0.0) or 0.0
                ),
                "result": result,
            }
        )

    reward_values = np.asarray([run["mean_reward"] for run in runs], dtype=np.float64)
    net_return_values = np.asarray(
        [run["avg_episode_net_return"] for run in runs], dtype=np.float64
    )
    violation_values = np.asarray(
        [run["constraint_violation_rate"] for run in runs], dtype=np.float64
    )
    summary = {
        "algorithm": algorithm,
        "seeds": seeds_list,
        "run_count": len(runs),
        "mean_reward_mean": float(np.mean(reward_values)),
        "mean_reward_median": float(np.median(reward_values)),
        "mean_reward_min": float(np.min(reward_values)),
        "mean_reward_max": float(np.max(reward_values)),
        "avg_episode_net_return_mean": float(np.mean(net_return_values)),
        "avg_episode_net_return_median": float(np.median(net_return_values)),
        "constraint_violation_rate_mean": float(np.mean(violation_values)),
        "runs": runs,
    }
    return summary

__all__ = [
    "RLAlgoConfig",
    "TrainingConfig",
    "Model",
    "train",
    "RLTrainer",
    "train_multi_seed",
    "train_rl_model_cli",
]

def train_rl_model_cli() -> None:
    """CLI interface for RL training."""
    _require_numpy("train_rl_model_cli")
    if not _ensure_rl():
        logger.warning('Stable-baselines3 not available - RL training CLI disabled')
        return
    try:
        logger.info('Starting RL model training (CLI)')
        np.random.seed(42)
        n_samples = 1000
        n_features = 4
        data = np.random.randn(n_samples, n_features)
        data[:, 0] = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)
        data[:, 1] = np.random.randn(n_samples) * 0.1
        data[:, 2] = np.abs(np.random.randn(n_samples) * 0.05)
        data[:, 3] = np.random.exponential(1000, n_samples)
        trainer = RLTrainer(algorithm='PPO', total_timesteps=50000, eval_freq=5000, early_stopping_patience=5)
        results = trainer.train(data=data, env_params={'transaction_cost': 0.001, 'slippage': 0.0005}, save_path='models/rl_demo')
        logger.info(f"Training completed with final reward: {results['final_evaluation'].get('mean_reward', 'N/A')}")
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:  # training or file errors
        logger.error(f'Error in RL CLI training: {e}')
        raise
if __name__ == '__main__':
    train_rl_model_cli()
