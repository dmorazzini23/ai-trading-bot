"""Enhanced RL training with reward shaping and evaluation callbacks."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import UTC, datetime
from typing import Any

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


PPO = A2C = DQN = _SB3Stub


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


class Model:
    """Minimal stand-in for an RL model used in tests."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def predict(self, state: Any, deterministic: bool = True) -> tuple[int, None]:
        return (1, None)

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
    global PPO, A2C, DQN, BaseCallback, EvalCallback, make_vec_env, evaluate_policy, DummyVecEnv
    if PPO is not _SB3Stub:
        return True
    if not is_rl_available():
        return False
    stack = _load_rl_stack()
    sb3 = stack["sb3"]
    PPO = sb3.PPO
    A2C = sb3.A2C
    DQN = sb3.DQN
    BaseCallback = sb3.common.callbacks.BaseCallback
    EvalCallback = sb3.common.callbacks.EvalCallback
    make_vec_env = sb3.common.env_util.make_vec_env
    evaluate_policy = sb3.common.evaluation.evaluate_policy
    DummyVecEnv = sb3.common.vec_env.DummyVecEnv
    return True


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback for RL training.

    Stops training when performance doesn't improve for a specified
    number of evaluations.
    """

    def __init__(self, patience: int=10, min_improvement: float=0.01, verbose: int=0):
        _require_numpy("EarlyStoppingCallback")
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.patience_counter = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        _require_numpy("EarlyStoppingCallback rollout handling")
        if hasattr(self.training_env, 'get_attr'):
            try:
                env_rewards = self.training_env.get_attr('episode_returns')
                if env_rewards and len(env_rewards[0]) > 0:
                    current_mean_reward = np.mean(env_rewards[0][-10:])
                    if current_mean_reward > self.best_mean_reward + self.min_improvement:
                        self.best_mean_reward = current_mean_reward
                        self.patience_counter = 0
                        if self.verbose > 0:
                            logger.info(f'New best reward: {self.best_mean_reward:.4f}')
                    else:
                        self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        if self.verbose > 0:
                            logger.info(f'Early stopping after {self.patience} evaluations without improvement')
                        return False
            except (AttributeError, TypeError, ValueError) as e:  # env may lack returns or contain bad data
                if self.verbose > 0:
                    logger.warning(f'Error in early stopping callback: {e}')
        return True

class DetailedEvalCallback(BaseCallback):
    """
    Enhanced evaluation callback with detailed metrics tracking.
    """

    def __init__(self, eval_env, eval_freq: int=10000, n_eval_episodes: int=5, deterministic: bool=True, save_path: str | None=None, verbose: int=1):
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
            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=self.deterministic, return_episode_rewards=True)
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            detailed_metrics = self._collect_detailed_metrics()
            eval_result = {'timestep': self.n_calls, 'mean_reward': float(mean_reward), 'std_reward': float(std_reward), 'mean_episode_length': float(mean_length), 'timestamp': datetime.now(UTC).isoformat(), **detailed_metrics}
            self.eval_results.append(eval_result)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.save_path:
                    best_model_path = os.path.join(self.save_path, 'best_model.zip')
                    self.model.save(best_model_path)
                    meta_path = os.path.join(self.save_path, 'best_model_meta.json')
                    with open(meta_path, 'w') as f:
                        json.dump(eval_result, f, indent=2)
            if self.verbose > 0:
                logger.info(f'Eval at step {self.n_calls}: mean_reward={mean_reward:.4f} ± {std_reward:.4f}')
        except (OSError, AttributeError, TypeError, ValueError) as e:  # file I/O or numeric issues during evaluation
            logger.error(f'Error in evaluation: {e}')

    def _collect_detailed_metrics(self) -> dict[str, float]:
        """Collect detailed performance metrics."""
        try:
            obs = self.eval_env.reset()
            total_reward = 0
            total_turnover = 0
            total_drawdown = 0
            total_variance = 0
            episode_length = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                episode_length += 1
                if isinstance(info, dict):
                    total_turnover += info.get('turnover_penalty', 0)
                    total_drawdown += info.get('drawdown_penalty', 0)
                    total_variance += info.get('variance_penalty', 0)
            avg_turnover = total_turnover / episode_length if episode_length > 0 else 0
            avg_drawdown = total_drawdown / episode_length if episode_length > 0 else 0
            avg_variance = total_variance / episode_length if episode_length > 0 else 0
            return {'avg_turnover_penalty': float(avg_turnover), 'avg_drawdown_penalty': float(avg_drawdown), 'avg_variance_penalty': float(avg_variance), 'sharpe_ratio': self._calculate_sharpe_ratio(), 'max_drawdown': float(total_drawdown) if total_drawdown > 0 else 0.0}
        except (AttributeError, TypeError, ValueError) as e:  # model or env returned unexpected values
            logger.error(f'Error collecting detailed metrics: {e}')
            return {}

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate approximate Sharpe ratio from recent evaluations."""
        _require_numpy("Sharpe ratio calculation")
        try:
            if len(self.eval_results) < 2:
                return 0.0
            recent_rewards = [r['mean_reward'] for r in self.eval_results[-10:]]
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
            with open(path, 'w') as f:
                json.dump(self.eval_results, f, indent=2)
            logger.info(f'Evaluation results saved to {path}')
        except (OSError, TypeError, ValueError) as e:  # disk or serialization issues
            logger.error(f'Error saving evaluation results: {e}')

class RLTrainer:
    """
    Enhanced RL trainer with reward shaping and evaluation.
    """

    def __init__(self, algorithm: str='PPO', total_timesteps: int=100000, eval_freq: int=10000, early_stopping_patience: int=10, seed: int=42):
        """
        Initialize RL trainer.

        Args:
            algorithm: RL algorithm ('PPO', 'A2C', 'DQN')
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
        self.training_results = {}
        self.eval_callback = None
        logger.info(f'RLTrainer initialized with {algorithm} algorithm')

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
            logger.info(f'Starting RL training with {len(data)} data points')
            self._create_environments(data, env_params)
            self._create_model(model_params)
            callbacks = self._setup_callbacks(save_path)
            start_time = datetime.now(UTC)
            self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks, progress_bar=True)
            end_time = datetime.now(UTC)
            self.training_results = {'algorithm': self.algorithm, 'total_timesteps': self.total_timesteps, 'training_time_seconds': (end_time - start_time).total_seconds(), 'seed': self.seed, 'final_evaluation': self._final_evaluation(), 'env_params': env_params or {}, 'model_params': model_params or {}}
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
            from .env import TradingEnv  # noqa: E402 - local import

            env_params = env_params or {}
            split_idx = int(len(data) * 0.8)
            train_data = data[:split_idx]
            eval_data = data[split_idx:]
            enhanced_env_params = {'transaction_cost': 0.001, 'slippage': 0.0005, 'half_spread': 0.0002, **env_params}

            def make_train_env():
                return TradingEnv(train_data, **enhanced_env_params)

            def make_eval_env():
                return TradingEnv(eval_data, **enhanced_env_params)
            self.train_env = DummyVecEnv([make_train_env])
            self.eval_env = make_eval_env()
            logger.debug(f'Environments created: train_data={len(train_data)}, eval_data={len(eval_data)}')
        except (ImportError, AttributeError, TypeError, ValueError) as e:  # TradingEnv missing or params invalid
            logger.error(f'Error creating environments: {e}')
            raise

    def _create_model(self, model_params: dict[str, Any] | None) -> None:
        """Create RL model."""
        try:
            model_params = model_params or {}
            default_params = {'verbose': 1, 'seed': self.seed, 'tensorboard_log': './tensorboard_logs/'}
            if self.algorithm == 'PPO':
                default_params.update({'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.0})
                final_params = {**default_params, **model_params}
                self.model = PPO('MlpPolicy', self.train_env, **final_params)
            elif self.algorithm == 'A2C':
                default_params.update({'learning_rate': 0.0007, 'n_steps': 20, 'gamma': 0.99, 'gae_lambda': 1.0, 'ent_coef': 0.0, 'vf_coef': 0.25})
                final_params = {**default_params, **model_params}
                self.model = A2C('MlpPolicy', self.train_env, **final_params)
            elif self.algorithm == 'DQN':
                default_params.update({'learning_rate': 0.0001, 'buffer_size': 50000, 'learning_starts': 1000, 'batch_size': 32, 'tau': 1.0, 'gamma': 0.99, 'train_freq': 4, 'gradient_steps': 1, 'target_update_interval': 1000, 'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05})
                final_params = {**default_params, **model_params}
                self.model = DQN('MlpPolicy', self.train_env, **final_params)
            else:
                raise ValueError(f'Unknown algorithm: {self.algorithm}')
            logger.debug(f'Model created: {self.algorithm} with {len(final_params)} parameters')
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:  # invalid algorithm or params
            logger.error(f'Error creating model: {e}')
            raise

    def _setup_callbacks(self, save_path: str | None) -> list[BaseCallback]:
        """Setup training callbacks."""
        try:
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
                obs = self.eval_env.reset()
                episode_reward = 0
                episode_metrics = {'turnover': 0, 'drawdown': 0, 'variance': 0}
                steps = 0
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward
                    steps += 1
                    if isinstance(info, dict):
                        episode_metrics['turnover'] += info.get('turnover_penalty', 0)
                        episode_metrics['drawdown'] += info.get('drawdown_penalty', 0)
                        episode_metrics['variance'] += info.get('variance_penalty', 0)
                for key in episode_metrics:
                    episode_metrics[key] = episode_metrics[key] / steps if steps > 0 else 0
                detailed_results.append({'total_reward': episode_reward, **episode_metrics})
            final_metrics = {'mean_reward': float(mean_reward), 'std_reward': float(std_reward), 'avg_turnover_penalty': float(np.mean([r['turnover'] for r in detailed_results])), 'avg_drawdown_penalty': float(np.mean([r['drawdown'] for r in detailed_results])), 'avg_variance_penalty': float(np.mean([r['variance'] for r in detailed_results])), 'reward_stability': float(1.0 / (1.0 + std_reward))}
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
            if self.eval_callback and hasattr(self.eval_callback, 'eval_results'):
                eval_path = os.path.join(save_path, 'evaluation_results.json')
                self.eval_callback.save_results(eval_path)
            metadata = {'algorithm': self.algorithm, 'training_timestamp': datetime.now(UTC).isoformat(), 'total_timesteps': self.total_timesteps, 'seed': self.seed, 'model_file': os.path.basename(model_path)}
            meta_path = os.path.join(save_path, 'meta.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f'Model and results saved to {save_path}')
        except (OSError, AttributeError, TypeError, ValueError) as e:  # file or serialization problems
            logger.error(f'Error saving model and results: {e}')

__all__ = [
    "TrainingConfig",
    "Model",
    "train",
    "RLTrainer",
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
