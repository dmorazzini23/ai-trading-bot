"""Simple trading environment for RL agent."""

from __future__ import annotations

import numpy as np

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


class TradingEnv(EnvBase):  # type: ignore[misc]
    """
    Minimal gym environment for offline training.

    This version includes support for transaction costs and slippage and
    computes rewards based on the change in net worth rather than raw
    cash changes.  Future extensions could support multiple assets and
    dynamic state windows.
    """

    def __init__(
        self, 
        data: np.ndarray, 
        window: int = 10, 
        *, 
        transaction_cost: float = 0.001,  # Increased default
        slippage: float = 0.0005,        # Added slippage
        half_spread: float = 0.0002      # Added half-spread
    ) -> None:
        # When gymnasium is unavailable, raise a clearer error at runtime
        if gym is None:
            raise ImportError("gymnasium required; install gymnasium to use TradingEnv")
        self.data = data.astype(np.float32)
        self.window = window
        self.current = window
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(window, data.shape[1]), dtype=np.float32
        )
        # position expressed in units of the asset
        self.position = 0
        # starting cash balance
        self.cash = 1.0
        # per‑trade transaction cost (fraction)
        self.transaction_cost = max(transaction_cost, 0.0)
        # slippage fraction applied to execution price
        self.slippage = max(slippage, 0.0)
        # half-spread for realistic bid-ask modeling
        self.half_spread = max(half_spread, 0.0)
        
        # Enhanced tracking for reward shaping
        self._last_net_worth = self.cash
        self._position_history = []
        self._turnover_history = []
        self._drawdown_peak = self.cash
        self._returns_history = []
        
        # Reward shaping parameters
        self.turnover_penalty_lambda = 0.1
        self.drawdown_penalty_lambda = 2.0
        self.variance_penalty_lambda = 0.5
        self.rolling_window = 20  # For variance calculation

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current = self.window
        self.position = 0
        self.cash = 1.0
        # Reset enhanced tracking
        self._last_net_worth = self.cash
        self._position_history = []
        self._turnover_history = []
        self._drawdown_peak = self.cash
        self._returns_history = []
        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        return self.data[self.current - self.window : self.current]

    def step(self, action: int):
        """
        Execute one step in the environment with enhanced reward shaping.

        Parameters
        ----------
        action : int
            Discrete action (0=hold, 1=buy, 2=sell).

        Returns
        -------
        tuple
            A 5‑tuple consisting of (next_state, reward, terminated,
            truncated, info) as per the gymnasium API.
        """
        price = float(self.data[self.current, 0])
        
        # Track position before action
        old_position = self.position
        
        # Calculate execution price including half-spread and slippage
        if action == 1:  # Buy
            exec_price = price * (1 + self.half_spread + self.slippage)
        elif action == 2:  # Sell
            exec_price = price * (1 - self.half_spread - self.slippage)
        else:  # Hold
            exec_price = price
        
        # Execute trade with transaction costs
        if action == 1:  # Buy
            total_cost = exec_price * (1.0 + self.transaction_cost)
            if self.cash >= total_cost:
                self.position += 1
                self.cash -= total_cost
        elif action == 2 and self.position > 0:  # Sell
            sale_proceeds = exec_price * (1.0 - self.transaction_cost)
            self.position -= 1
            self.cash += sale_proceeds
        
        # Calculate current net worth
        net_worth = self.cash + self.position * price
        
        # Base reward: change in net worth
        base_reward = net_worth - self._last_net_worth
        
        # Calculate position change for turnover penalty
        position_change = abs(self.position - old_position)
        self._turnover_history.append(position_change)
        if len(self._turnover_history) > self.rolling_window:
            self._turnover_history.pop(0)
        
        # Turnover penalty
        turnover_penalty = self.turnover_penalty_lambda * position_change
        
        # Drawdown penalty
        if net_worth > self._drawdown_peak:
            self._drawdown_peak = net_worth
        
        current_drawdown = (self._drawdown_peak - net_worth) / self._drawdown_peak if self._drawdown_peak > 0 else 0
        drawdown_penalty = self.drawdown_penalty_lambda * current_drawdown
        
        # Variance penalty (rolling volatility)
        returns = (net_worth / self._last_net_worth - 1) if self._last_net_worth > 0 else 0
        self._returns_history.append(returns)
        if len(self._returns_history) > self.rolling_window:
            self._returns_history.pop(0)
        
        if len(self._returns_history) > 5:
            rolling_variance = np.var(self._returns_history)
            variance_penalty = self.variance_penalty_lambda * rolling_variance
        else:
            variance_penalty = 0.0
        
        # Combined reward with penalties
        reward = base_reward - turnover_penalty - drawdown_penalty - variance_penalty
        
        # Update tracking
        self._last_net_worth = net_worth
        self._position_history.append(self.position)
        if len(self._position_history) > self.rolling_window:
            self._position_history.pop(0)
        
        # Advance to next timestep
        self.current += 1
        terminated = self.current >= len(self.data)
        
        # Enhanced info
        info = {
            'net_worth': net_worth,
            'base_reward': base_reward,
            'turnover_penalty': turnover_penalty,
            'drawdown_penalty': drawdown_penalty,
            'variance_penalty': variance_penalty,
            'position': self.position,
            'cash': self.cash,
            'drawdown': current_drawdown,
            'recent_turnover': np.mean(self._turnover_history) if self._turnover_history else 0,
            'recent_volatility': np.std(self._returns_history) if len(self._returns_history) > 1 else 0
        }
        
        return self._get_state(), reward, terminated, terminated, info
