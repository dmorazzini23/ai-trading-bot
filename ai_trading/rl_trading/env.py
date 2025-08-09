"""Simple trading environment for RL agent."""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Tuple
from dataclasses import dataclass
from collections import deque

try:
    import gymnasium as gym
    # Use the base Env class when gymnasium is available
    EnvBase = gym.Env
except Exception:  # pragma: no cover - optional dependency
    gym = None
    # Define a lightweight fallback base class when gymnasium is missing
    class _EnvFallback:
        """Fallback base class when gymnasium is not installed."""

        pass

    EnvBase = _EnvFallback


# AI-AGENT-REF: Unified action space configuration
@dataclass
class ActionSpaceConfig:
    """Configuration for action space - ensures consistency between training and inference."""
    action_type: str = "discrete"  # "discrete" or "continuous"
    discrete_actions: int = 3  # 0=hold, 1=buy, 2=sell
    continuous_bounds: Tuple[float, float] = (-1.0, 1.0)  # For continuous actions
    position_limits: Tuple[float, float] = (-1.0, 1.0)  # Position bounds


@dataclass
class RewardConfig:
    """Configuration for reward calculation and normalization."""
    normalize_rewards: bool = True
    reward_window: int = 100  # Window for running statistics
    base_reward_weight: float = 1.0
    turnover_penalty: float = 0.1
    drawdown_penalty: float = 2.0
    variance_penalty: float = 0.5
    sharpe_bonus: float = 0.1


class RunningStats:
    """Running statistics for reward normalization."""
    
    def __init__(self, window: int = 100):
        self.window = window
        self.values = deque(maxlen=window)
        self._mean = 0.0
        self._std = 1.0
    
    def update(self, value: float) -> None:
        """Update running statistics with new value."""
        self.values.append(value)
        if len(self.values) > 1:
            self._mean = np.mean(self.values)
            self._std = max(np.std(self.values), 1e-8)  # Avoid division by zero
    
    def normalize(self, value: float) -> float:
        """Normalize value using running statistics."""
        return (value - self._mean) / self._std
    
    @property
    def mean(self) -> float:
        return self._mean
    
    @property
    def std(self) -> float:
        return self._std


class TradingEnv(EnvBase):  # type: ignore[misc]
    """
    Enhanced trading environment for RL with unified action space and reward normalization.

    This version includes:
    - Configurable action space (discrete/continuous)  
    - Reward normalization and enhanced shaping
    - Entropy scheduling support
    - Consistent interface for training and inference
    """

    def __init__(
        self, 
        data: np.ndarray, 
        window: int = 10, 
        *, 
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        half_spread: float = 0.0002,
        action_config: Optional[ActionSpaceConfig] = None,
        reward_config: Optional[RewardConfig] = None
    ) -> None:
        # When gymnasium is unavailable, raise a clearer error at runtime
        if gym is None:
            raise ImportError("gymnasium required; install gymnasium to use TradingEnv")
        
        self.data = data.astype(np.float32)
        self.window = window
        self.current = window
        
        # AI-AGENT-REF: Configure action space
        self.action_config = action_config or ActionSpaceConfig()
        self.reward_config = reward_config or RewardConfig()
        
        # Setup action space based on configuration
        if self.action_config.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(self.action_config.discrete_actions)
        else:
            self.action_space = gym.spaces.Box(
                low=self.action_config.continuous_bounds[0],
                high=self.action_config.continuous_bounds[1],
                shape=(1,),
                dtype=np.float32
            )
        
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(window, data.shape[1]), dtype=np.float32
        )
        
        # Position and cash state
        self.position = 0.0  # Allow fractional positions for continuous actions
        self.cash = 1.0
        
        # Cost parameters
        self.transaction_cost = max(transaction_cost, 0.0)
        self.slippage = max(slippage, 0.0)
        self.half_spread = max(half_spread, 0.0)
        
        # Enhanced tracking
        self._last_net_worth = self.cash
        self._position_history = deque(maxlen=50)
        self._turnover_history = deque(maxlen=50)
        self._drawdown_peak = self.cash
        self._returns_history = deque(maxlen=50)
        
        # AI-AGENT-REF: Reward normalization
        self._reward_stats = RunningStats(self.reward_config.reward_window) if self.reward_config.normalize_rewards else None
        
        # Entropy tracking for scheduling
        self._action_entropy_history = deque(maxlen=100)
        
        # Episode statistics
        self._episode_stats = {
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'turnover': 0.0,
            'actions_taken': 0
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current = self.window
        self.position = 0.0
        self.cash = 1.0
        
        # Reset enhanced tracking
        self._last_net_worth = self.cash
        self._position_history.clear()
        self._turnover_history.clear()
        self._drawdown_peak = self.cash
        self._returns_history.clear()
        self._action_entropy_history.clear()
        
        # Reset episode statistics
        self._episode_stats = {
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'turnover': 0.0,
            'actions_taken': 0
        }
        
        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        return self.data[self.current - self.window : self.current]

    def _execute_discrete_action(self, action: int, price: float) -> Tuple[float, float]:
        """Execute discrete action (0=hold, 1=buy, 2=sell)."""
        old_position = self.position
        
        # Calculate execution price including half-spread and slippage
        if action == 1:  # Buy
            exec_price = price * (1 + self.half_spread + self.slippage)
            if self.cash >= exec_price * (1.0 + self.transaction_cost):
                self.position += 1
                self.cash -= exec_price * (1.0 + self.transaction_cost)
        elif action == 2 and self.position > 0:  # Sell
            exec_price = price * (1 - self.half_spread - self.slippage)
            self.position -= 1
            self.cash += exec_price * (1.0 - self.transaction_cost)
        
        return old_position, abs(self.position - old_position)

    def _execute_continuous_action(self, action: float, price: float) -> Tuple[float, float]:
        """Execute continuous action (target position change)."""
        old_position = self.position
        
        # Clip action to valid range
        action = np.clip(action, self.action_config.continuous_bounds[0], self.action_config.continuous_bounds[1])
        
        # Calculate target position change
        position_change = action * 0.5  # Scale action to reasonable position change
        target_position = np.clip(
            self.position + position_change,
            self.action_config.position_limits[0],
            self.action_config.position_limits[1]
        )
        
        # Execute the position change
        trade_size = target_position - self.position
        
        if abs(trade_size) > 1e-6:  # Only trade if meaningful change
            if trade_size > 0:  # Buying
                exec_price = price * (1 + self.half_spread + self.slippage)
                cost = abs(trade_size) * exec_price * (1.0 + self.transaction_cost)
                if self.cash >= cost:
                    self.position = target_position
                    self.cash -= cost
            else:  # Selling
                exec_price = price * (1 - self.half_spread - self.slippage)
                proceeds = abs(trade_size) * exec_price * (1.0 - self.transaction_cost)
                self.position = target_position
                self.cash += proceeds
        
        return old_position, abs(trade_size)

    def step(self, action: Union[int, float, np.ndarray]):
        """
        Execute one step in the environment with enhanced reward calculation.

        Parameters
        ----------
        action : int | float | np.ndarray
            Action to execute (discrete or continuous based on action space).

        Returns
        -------
        tuple
            A 5‑tuple consisting of (next_state, reward, terminated,
            truncated, info) as per the gymnasium API.
        """
        price = float(self.data[self.current, 0])
        
        # Handle different action types
        if self.action_config.action_type == "discrete":
            action_int = int(action)
            old_position, trade_size = self._execute_discrete_action(action_int, price)
            # Track action entropy for discrete actions
            action_probs = np.zeros(self.action_config.discrete_actions)
            action_probs[action_int] = 1.0
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        else:
            action_float = float(action) if np.isscalar(action) else float(action[0])
            old_position, trade_size = self._execute_continuous_action(action_float, price)
            # For continuous actions, entropy is related to action variance
            entropy = 0.5 * np.log(2 * np.pi * np.e)  # Maximum entropy for normalized action
        
        self._action_entropy_history.append(entropy)
        
        # Calculate current net worth
        net_worth = self.cash + self.position * price
        
        # Base reward: change in net worth
        base_reward = (net_worth - self._last_net_worth) * self.reward_config.base_reward_weight
        
        # Penalty components
        turnover_penalty = self.reward_config.turnover_penalty * trade_size
        
        # Drawdown tracking and penalty
        if net_worth > self._drawdown_peak:
            self._drawdown_peak = net_worth
        
        current_drawdown = (self._drawdown_peak - net_worth) / self._drawdown_peak if self._drawdown_peak > 0 else 0
        drawdown_penalty = self.reward_config.drawdown_penalty * current_drawdown
        
        # Variance penalty (rolling volatility)
        returns = (net_worth / self._last_net_worth - 1) if self._last_net_worth > 0 else 0
        self._returns_history.append(returns)
        
        if len(self._returns_history) > 5:
            rolling_variance = np.var(self._returns_history)
            variance_penalty = self.reward_config.variance_penalty * rolling_variance
        else:
            variance_penalty = 0.0
        
        # Sharpe ratio bonus
        if len(self._returns_history) > 10:
            returns_array = np.array(self._returns_history)
            sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
            sharpe_bonus = self.reward_config.sharpe_bonus * max(0, sharpe_ratio)
        else:
            sharpe_bonus = 0.0
        
        # Combined reward
        raw_reward = base_reward - turnover_penalty - drawdown_penalty - variance_penalty + sharpe_bonus
        
        # Apply reward normalization if enabled
        if self._reward_stats is not None:
            self._reward_stats.update(raw_reward)
            reward = self._reward_stats.normalize(raw_reward)
        else:
            reward = raw_reward
        
        # Update tracking
        self._last_net_worth = net_worth
        self._position_history.append(self.position)
        self._turnover_history.append(trade_size)
        
        # Update episode statistics
        self._episode_stats['total_return'] = (net_worth / 1.0) - 1  # Assuming initial cash = 1.0
        self._episode_stats['max_drawdown'] = max(self._episode_stats['max_drawdown'], current_drawdown)
        self._episode_stats['turnover'] += trade_size
        self._episode_stats['actions_taken'] += 1 if trade_size > 1e-6 else 0
        
        # Calculate Sharpe ratio for episode
        if len(self._returns_history) > 1:
            returns_array = np.array(self._returns_history)
            self._episode_stats['sharpe_ratio'] = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
        
        # Advance to next timestep
        self.current += 1
        terminated = self.current >= len(self.data)
        
        # Enhanced info with action space consistency
        info = {
            'net_worth': net_worth,
            'raw_reward': raw_reward,
            'normalized_reward': reward,
            'base_reward': base_reward,
            'turnover_penalty': turnover_penalty,
            'drawdown_penalty': drawdown_penalty,
            'variance_penalty': variance_penalty,
            'sharpe_bonus': sharpe_bonus,
            'position': self.position,
            'cash': self.cash,
            'drawdown': current_drawdown,
            'trade_size': trade_size,
            'action_entropy': entropy,
            'avg_entropy': np.mean(self._action_entropy_history) if self._action_entropy_history else 0,
            'episode_stats': self._episode_stats.copy(),
            'action_config': self.action_config,
            'reward_stats': {
                'mean': self._reward_stats.mean if self._reward_stats else 0,
                'std': self._reward_stats.std if self._reward_stats else 1
            } if self._reward_stats else None
        }
        
        return self._get_state(), reward, terminated, terminated, info

    def get_action_space_config(self) -> ActionSpaceConfig:
        """Get action space configuration for inference consistency."""
        return self.action_config
    
    def get_reward_config(self) -> RewardConfig:
        """Get reward configuration."""
        return self.reward_config
