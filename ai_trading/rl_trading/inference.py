"""Load a trained RL policy and produce trade signals with unified action space."""

from __future__ import annotations

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict

from . import RLAgent
from .env import ActionSpaceConfig, RewardConfig
from ai_trading.strategies.base import StrategySignal

# Type alias for backward compatibility  
TradeSignal = StrategySignal

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for RL inference to ensure training-inference parity."""
    model_path: str
    action_config: ActionSpaceConfig
    reward_config: Optional[RewardConfig] = None
    deterministic: bool = True
    observation_window: int = 10
    confidence_threshold: float = 0.1  # Minimum confidence for non-hold actions


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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load the trained model
        self.agent = RLAgent(config.model_path)
        self.agent.load()
        
        # Validate action space consistency
        self._validate_action_space()
        
        # Preprocessing pipeline state
        self._obs_buffer = []
        self._last_prediction = None
        self._prediction_confidence = 0.0
        
        # Statistics tracking
        self._inference_stats = {
            'total_predictions': 0,
            'hold_predictions': 0,
            'buy_predictions': 0,
            'sell_predictions': 0,
            'avg_confidence': 0.0
        }
    
    def _validate_action_space(self) -> None:
        """Validate that model action space matches configuration."""
        if self.agent.model is None:
            self.logger.warning("Cannot validate action space - model not loaded")
            return
        
        # Check action space compatibility
        model_action_space = self.agent.model.action_space
        
        if self.config.action_config.action_type == "discrete":
            if hasattr(model_action_space, 'n'):
                if model_action_space.n != self.config.action_config.discrete_actions:
                    self.logger.error(
                        f"Action space mismatch: model has {model_action_space.n} actions, "
                        f"config expects {self.config.action_config.discrete_actions}"
                    )
        else:
            if hasattr(model_action_space, 'shape'):
                if model_action_space.shape[0] != 1:
                    self.logger.error(
                        f"Continuous action space mismatch: model shape {model_action_space.shape}, "
                        f"expected (1,)"
                    )
        
        self.logger.info(f"Action space validation complete: {self.config.action_config.action_type}")
    
    def preprocess_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Preprocess observation for model input (same as training).
        
        Args:
            observation: Raw observation data
            
        Returns:
            Preprocessed observation ready for model
        """
        # Ensure proper shape and dtype
        obs = np.array(observation, dtype=np.float32)
        
        # Handle windowing if needed
        if len(obs.shape) == 1:
            # Single time step - need to create window
            self._obs_buffer.append(obs)
            if len(self._obs_buffer) > self.config.observation_window:
                self._obs_buffer.pop(0)
            
            # Pad if we don't have enough history
            while len(self._obs_buffer) < self.config.observation_window:
                self._obs_buffer.insert(0, self._obs_buffer[0] if self._obs_buffer else np.zeros_like(obs))
            
            obs = np.array(self._obs_buffer)
        
        # Ensure proper window size
        if obs.shape[0] > self.config.observation_window:
            obs = obs[-self.config.observation_window:]
        elif obs.shape[0] < self.config.observation_window:
            # Pad with first observation
            padding = np.tile(obs[0:1], (self.config.observation_window - obs.shape[0], 1))
            obs = np.vstack([padding, obs])
        
        return obs
    
    def postprocess_action(self, raw_action: Union[int, float, np.ndarray], observation: np.ndarray) -> Dict[str, Any]:
        """
        Postprocess model action to trading signal (same as training).
        
        Args:
            raw_action: Raw action from model
            observation: Current observation for confidence estimation
            
        Returns:
            Dictionary with action details and confidence
        """
        if self.config.action_config.action_type == "discrete":
            action_int = int(raw_action)
            action_map = {0: "hold", 1: "buy", 2: "sell"}
            action_name = action_map.get(action_int, "hold")
            
            # For discrete actions, confidence is based on action value distribution
            # This is simplified - in practice, would use model's action probabilities
            confidence = 0.8 if action_int != 0 else 0.2  # Higher confidence for non-hold actions
            
        else:
            action_float = float(raw_action) if np.isscalar(raw_action) else float(raw_action[0])
            
            # Map continuous action to discrete signal
            if action_float > self.config.confidence_threshold:
                action_name = "buy"
                confidence = min(abs(action_float), 1.0)
            elif action_float < -self.config.confidence_threshold:
                action_name = "sell" 
                confidence = min(abs(action_float), 1.0)
            else:
                action_name = "hold"
                confidence = 1.0 - abs(action_float)  # High confidence for small actions (hold)
        
        return {
            'action': action_name,
            'confidence': confidence,
            'raw_action': raw_action,
            'action_type': self.config.action_config.action_type
        }
    
    def predict(self, observation: np.ndarray, symbol: str = "RL") -> Optional[TradeSignal]:
        """
        Predict trading signal from observation.
        
        Args:
            observation: Market observation data
            symbol: Symbol for the signal
            
        Returns:
            TradeSignal or None if prediction fails
        """
        if self.agent.model is None:
            self.logger.error("Model not loaded")
            return None
        
        try:
            # Preprocess observation
            processed_obs = self.preprocess_observation(observation)
            
            # Get prediction from model
            raw_action, _ = self.agent.model.predict(
                processed_obs, 
                deterministic=self.config.deterministic
            )
            
            # Postprocess action
            action_details = self.postprocess_action(raw_action, processed_obs)
            
            # Update statistics
            self._update_stats(action_details)
            
            # Create trade signal
            signal = TradeSignal(
                symbol=symbol,
                side=action_details['action'],
                confidence=action_details['confidence'],
                strategy="rl_unified",
                metadata={
                    'action_type': action_details['action_type'],
                    'raw_action': str(action_details['raw_action']),
                    'model_path': self.config.model_path
                }
            )
            
            self._last_prediction = signal
            self._prediction_confidence = action_details['confidence']
            
            return signal
            
        except Exception as e:
            self.logger.error(f"RL prediction failed: {e}")
            return None
    
    def predict_batch(self, observations: List[np.ndarray], symbols: List[str]) -> List[Optional[TradeSignal]]:
        """
        Predict trading signals for batch of observations.
        
        Args:
            observations: List of market observations
            symbols: List of symbols
            
        Returns:
            List of TradeSignal objects (one per symbol)
        """
        if len(observations) != len(symbols):
            raise ValueError("Observations and symbols must have same length")
        
        signals = []
        for obs, symbol in zip(observations, symbols):
            signal = self.predict(obs, symbol)
            signals.append(signal)
        
        return signals
    
    def _update_stats(self, action_details: Dict[str, Any]) -> None:
        """Update inference statistics."""
        self._inference_stats['total_predictions'] += 1
        
        action = action_details['action']
        if action == 'hold':
            self._inference_stats['hold_predictions'] += 1
        elif action == 'buy':
            self._inference_stats['buy_predictions'] += 1
        elif action == 'sell':
            self._inference_stats['sell_predictions'] += 1
        
        # Update running average confidence
        total = self._inference_stats['total_predictions']
        current_avg = self._inference_stats['avg_confidence']
        new_confidence = action_details['confidence']
        self._inference_stats['avg_confidence'] = (current_avg * (total - 1) + new_confidence) / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = self._inference_stats.copy()
        stats['config'] = asdict(self.config)
        stats['last_prediction'] = str(self._last_prediction) if self._last_prediction else None
        stats['last_confidence'] = self._prediction_confidence
        return stats
    
    def reset_stats(self) -> None:
        """Reset inference statistics."""
        self._inference_stats = {
            'total_predictions': 0,
            'hold_predictions': 0,
            'buy_predictions': 0,
            'sell_predictions': 0,
            'avg_confidence': 0.0
        }


# Convenience functions for backward compatibility
def load_policy(model_path: str | Path) -> RLAgent:
    """Load RL policy (backward compatibility)."""
    agent = RLAgent(model_path)
    agent.load()
    return agent


def predict_signal(agent: RLAgent, state: np.ndarray) -> TradeSignal | None:
    """Predict signal using agent (backward compatibility)."""
    return agent.predict(state)


# AI-AGENT-REF: New unified inference function
def create_unified_inference(
    model_path: str,
    action_type: str = "discrete",
    discrete_actions: int = 3,
    observation_window: int = 10
) -> UnifiedRLInference:
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
    action_config = ActionSpaceConfig(
        action_type=action_type,
        discrete_actions=discrete_actions
    )
    
    inference_config = InferenceConfig(
        model_path=model_path,
        action_config=action_config,
        observation_window=observation_window
    )
    
    return UnifiedRLInference(inference_config)
