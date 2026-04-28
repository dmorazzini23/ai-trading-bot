"""Load a trained RL policy and produce trade signals with unified action space."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.logging import get_logger
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast
import json
import numpy as np
import zipfile
from ai_trading.strategies.base import StrategySignal
from . import RLAgent
from .env import ActionSpaceConfig, RewardConfig
from .state_builder import MarketStateBuilder
TradeSignal: TypeAlias = StrategySignal
logger = get_logger(__name__)
_STATE_BUILDER_ZIP_MEMBER = "rl_state_builder.json"

@dataclass
class InferenceConfig:
    """Configuration for RL inference to ensure training-inference parity."""
    model_path: str
    action_config: ActionSpaceConfig
    reward_config: RewardConfig | None = None
    deterministic: bool = True
    observation_window: int = 10
    confidence_threshold: float = 0.1
    state_builder_metadata: dict[str, Any] | None = None

class UnifiedRLInference:
    """
    Unified RL inference wrapper ensuring same action space and pre/post pipeline as training.

    Provides consistent interface between training environment and live inference
    with proper action space handling and confidence estimation.
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize unified RL inference.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self.agent = RLAgent(config.model_path)
        self.agent.load()
        self.state_builder = self._load_state_builder()
        self._validate_action_space()
        self._obs_buffer: list[np.ndarray] = []
        self._raw_obs_buffer: list[np.ndarray] = []
        self._last_prediction: TradeSignal | None = None
        self._prediction_confidence = 0.0
        self._inference_stats: dict[str, Any] = {'total_predictions': 0, 'hold_predictions': 0, 'buy_predictions': 0, 'sell_predictions': 0, 'avg_confidence': 0.0}

    @staticmethod
    def _extract_state_builder_metadata(payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, Mapping):
            return None
        candidate = payload.get("state_builder")
        if isinstance(candidate, Mapping):
            return dict(candidate)
        nested = payload.get("training_results")
        if isinstance(nested, Mapping):
            candidate = nested.get("state_builder")
            if isinstance(candidate, Mapping):
                return dict(candidate)
        if "schema" in payload and "mean" in payload and "std" in payload:
            return dict(payload)
        return None

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _read_zipped_state_builder(path: Path) -> dict[str, Any] | None:
        try:
            with zipfile.ZipFile(path) as model_zip:
                with model_zip.open(_STATE_BUILDER_ZIP_MEMBER) as member:
                    payload = json.loads(member.read().decode("utf-8"))
        except (OSError, KeyError, ValueError, UnicodeDecodeError, zipfile.BadZipFile):
            return None
        return payload if isinstance(payload, dict) else None

    def _load_state_builder_metadata(self) -> dict[str, Any] | None:
        if self.config.state_builder_metadata is not None:
            return dict(self.config.state_builder_metadata)

        model_path = Path(self.config.model_path)
        payload = self._read_zipped_state_builder(model_path)
        metadata = self._extract_state_builder_metadata(payload)
        if metadata is not None:
            return metadata

        candidates = [
            Path(f"{model_path}.state_builder.json"),
            Path(f"{model_path}.meta.json"),
            model_path.with_suffix(f"{model_path.suffix}.state_builder.json")
            if model_path.suffix
            else Path(f"{model_path}.state_builder.json"),
            model_path.parent / "meta.json",
            model_path.parent / "training_results.json",
        ]
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if not candidate.is_file():
                continue
            metadata = self._extract_state_builder_metadata(self._read_json(candidate))
            if metadata is not None:
                return metadata
        return None

    def _load_state_builder(self) -> MarketStateBuilder | None:
        metadata = self._load_state_builder_metadata()
        if metadata is None or not bool(metadata.get("enabled", True)):
            return None
        try:
            return MarketStateBuilder.from_metadata(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid RL state builder metadata: {exc}") from exc

    def _validate_action_space(self) -> None:
        """Validate that model action space matches configuration."""
        if self.agent.model is None:
            self.logger.warning('Cannot validate action space - model not loaded')
            return
        model_action_space = getattr(self.agent.model, "action_space", None)
        if model_action_space is None:
            self.logger.warning("Cannot validate action space - model has no action_space")
            return
        if self.config.action_config.action_type == 'discrete':
            if hasattr(model_action_space, 'n'):
                if model_action_space.n != self.config.action_config.discrete_actions:
                    self.logger.error(f'Action space mismatch: model has {model_action_space.n} actions, config expects {self.config.action_config.discrete_actions}')
        elif hasattr(model_action_space, 'shape'):
            if model_action_space.shape[0] != 1:
                self.logger.error(f'Continuous action space mismatch: model shape {model_action_space.shape}, expected (1,)')
        self.logger.info(f'Action space validation complete: {self.config.action_config.action_type}')

    def preprocess_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Preprocess observation for model input (same as training).

        Args:
            observation: Raw observation data

        Returns:
            Preprocessed observation ready for model
        """
        obs = np.array(observation, dtype=np.float32)
        if len(obs.shape) == 1:
            buffer = self._raw_obs_buffer if self.state_builder is not None else self._obs_buffer
            buffer.append(obs)
            if len(buffer) > self.config.observation_window:
                buffer.pop(0)
            while len(buffer) < self.config.observation_window:
                buffer.insert(0, buffer[0] if buffer else np.zeros_like(obs))
            obs = np.array(buffer)
        if obs.shape[0] > self.config.observation_window:
            obs = obs[-self.config.observation_window:]
        elif obs.shape[0] < self.config.observation_window:
            padding = np.tile(obs[0:1], (self.config.observation_window - obs.shape[0], 1))
            obs = np.vstack([padding, obs])
        if self.state_builder is not None:
            obs = cast(np.ndarray, self.state_builder.transform(obs))
        return cast(np.ndarray, obs.astype(np.float32, copy=False))

    def _validate_observation_shape(self, observation: np.ndarray) -> None:
        model = self.agent.model
        observation_space = getattr(model, "observation_space", None)
        expected_shape = getattr(observation_space, "shape", None)
        if expected_shape is None:
            return
        expected = tuple(int(value) for value in expected_shape)
        observed = tuple(int(value) for value in observation.shape)
        if expected != observed:
            raise ValueError(
                "RL observation shape mismatch: "
                f"model expects {expected}, inference produced {observed}"
            )

    @staticmethod
    def _finite_unit_interval(value: Any, *, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(numeric):
            return default
        return float(max(0.0, min(1.0, numeric)))

    def postprocess_action(
        self,
        raw_action: int | float | np.ndarray,
        observation: np.ndarray,
        *,
        action_mask: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        """
        Postprocess model action to trading signal (same as training).

        Args:
            raw_action: Raw action from model
            observation: Current observation for confidence estimation

        Returns:
            Dictionary with action details and confidence
        """
        if self.config.action_config.action_type == 'discrete':
            action_int = int(raw_action)
            mask_array: np.ndarray | None = None
            if action_mask is not None:
                try:
                    mask_array = np.asarray(action_mask, dtype=np.float32).reshape(-1)
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    mask_array = None
            action_masked = False
            if (
                mask_array is not None
                and mask_array.size > action_int
                and float(mask_array[action_int]) <= 0.0
            ):
                action_masked = True
                if mask_array.size > 0 and float(mask_array[0]) > 0.0:
                    action_int = 0
                else:
                    valid_actions = [idx for idx, enabled in enumerate(mask_array.tolist()) if float(enabled) > 0.0]
                    action_int = int(valid_actions[0]) if valid_actions else 0
            action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
            action_name = action_map.get(action_int, 'hold')
            confidence = 0.8 if action_int != 0 else 0.2
        else:
            if np.isscalar(raw_action):
                if isinstance(raw_action, (bool, int, float, np.integer, np.floating)):
                    action_float = float(raw_action)
                else:
                    action_float = 0.0
            else:
                action_array = np.asarray(raw_action, dtype=np.float32).reshape(-1)
                action_float = float(action_array[0]) if action_array.size else 0.0
            if not np.isfinite(action_float):
                action_float = 0.0
            if action_float > self.config.confidence_threshold:
                action_name = 'buy'
                confidence = min(abs(action_float), 1.0)
            elif action_float < -self.config.confidence_threshold:
                action_name = 'sell'
                confidence = min(abs(action_float), 1.0)
            else:
                action_name = 'hold'
                confidence = 1.0 - abs(action_float)
            action_masked = False
        confidence = self._finite_unit_interval(confidence)
        strength = 0.0 if action_name == "hold" else confidence
        return {'action': action_name, 'confidence': confidence, 'strength': strength, 'raw_action': raw_action, 'action_type': self.config.action_config.action_type, 'action_masked': bool(action_masked)}

    def predict(
        self,
        observation: np.ndarray,
        symbol: str='RL',
        *,
        action_mask: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> TradeSignal | None:
        """
        Predict trading signal from observation.

        Args:
            observation: Market observation data
            symbol: Symbol for the signal

        Returns:
            TradeSignal or None if prediction fails
        """
        if self.agent.model is None:
            self.logger.error('Model not loaded')
            return None
        try:
            processed_obs = self.preprocess_observation(observation)
            self._validate_observation_shape(processed_obs)
            raw_action, _ = self.agent.model.predict(processed_obs, deterministic=self.config.deterministic)
            action_details = self.postprocess_action(
                raw_action,
                processed_obs,
                action_mask=action_mask,
            )
            self._update_stats(action_details)
            signal = TradeSignal(symbol=symbol, side=action_details['action'], strength=action_details['strength'], confidence=action_details['confidence'], strategy='rl_unified', metadata={'action_type': action_details['action_type'], 'raw_action': str(action_details['raw_action']), 'model_path': self.config.model_path, 'action_masked': bool(action_details.get('action_masked', False))})
            self._last_prediction = signal
            self._prediction_confidence = action_details['confidence']
            return signal
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            self.logger.error(f'RL prediction failed: {e}')
            return None

    def predict_batch(self, observations: list[np.ndarray], symbols: list[str]) -> list[TradeSignal | None]:
        """
        Predict trading signals for batch of observations.

        Args:
            observations: List of market observations
            symbols: List of symbols

        Returns:
            List of TradeSignal objects (one per symbol)
        """
        if len(observations) != len(symbols):
            raise ValueError('Observations and symbols must have same length')
        signals: list[TradeSignal | None] = []
        for obs, symbol in zip(observations, symbols, strict=False):
            signal = self.predict(obs, symbol)
            signals.append(signal)
        return signals

    def _update_stats(self, action_details: dict[str, Any]) -> None:
        """Update inference statistics."""
        self._inference_stats['total_predictions'] += 1
        action = action_details['action']
        if action == 'hold':
            self._inference_stats['hold_predictions'] += 1
        elif action == 'buy':
            self._inference_stats['buy_predictions'] += 1
        elif action == 'sell':
            self._inference_stats['sell_predictions'] += 1
        total = self._inference_stats['total_predictions']
        current_avg = self._inference_stats['avg_confidence']
        new_confidence = action_details['confidence']
        self._inference_stats['avg_confidence'] = (current_avg * (total - 1) + new_confidence) / total

    def get_stats(self) -> dict[str, Any]:
        """Get inference statistics."""
        stats = self._inference_stats.copy()
        stats['config'] = asdict(self.config)
        stats['last_prediction'] = str(self._last_prediction) if self._last_prediction else None
        stats['last_confidence'] = self._prediction_confidence
        return stats

    def reset_stats(self) -> None:
        """Reset inference statistics."""
        self._inference_stats = {'total_predictions': 0, 'hold_predictions': 0, 'buy_predictions': 0, 'sell_predictions': 0, 'avg_confidence': 0.0}

def load_policy(model_path: str | Path) -> RLAgent:
    """Load RL policy (backward compatibility)."""
    agent = RLAgent(model_path)
    agent.load()
    return agent

def predict_signal(agent: RLAgent, state: np.ndarray) -> TradeSignal | None:
    """Predict signal using agent (backward compatibility)."""
    prediction = agent.predict(state)
    if isinstance(prediction, list):
        return cast(TradeSignal, prediction[0]) if prediction else None
    return cast(TradeSignal, prediction)

def create_unified_inference(model_path: str, action_type: str='discrete', discrete_actions: int=3, observation_window: int=10) -> UnifiedRLInference:
    """
    Create unified RL inference with specified configuration.

    Args:
        model_path: Path to trained model
        action_type: "discrete" or "continuous"
        discrete_actions: Number of discrete actions (if discrete)
        observation_window: Observation window size

    Returns:
        UnifiedRLInference instance
    """
    action_config = ActionSpaceConfig(action_type=action_type, discrete_actions=discrete_actions)
    inference_config = InferenceConfig(model_path=model_path, action_config=action_config, observation_window=observation_window)
    return UnifiedRLInference(inference_config)
