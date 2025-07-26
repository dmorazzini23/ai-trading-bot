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

    def __init__(self, data: np.ndarray, window: int = 10, *, transaction_cost: float = 0.0, slippage: float = 0.0) -> None:
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
        # last recorded net worth used to compute reward deltas
        self._last_net_worth = self.cash

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current = self.window
        self.position = 0
        self.cash = 1.0
        # reset net worth baseline
        self._last_net_worth = self.cash
        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        return self.data[self.current - self.window : self.current]

    def step(self, action: int):
        """
        Execute one step in the environment.

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
        # compute execution price including slippage
        exec_price = price * (1 + self.slippage) if action == 1 else price * (1 - self.slippage)
        # adjust cash and position for buy/sell with transaction costs
        if action == 1:
            # ensure sufficient cash
            total_cost = exec_price * (1.0 + self.transaction_cost)
            if self.cash >= total_cost:
                self.position += 1
                self.cash -= total_cost
        elif action == 2 and self.position > 0:
            sale_proceeds = exec_price * (1.0 - self.transaction_cost)
            self.position -= 1
            self.cash += sale_proceeds
        # compute net worth and reward as change from last net worth
        net_worth = self.cash + self.position * price
        reward = net_worth - self._last_net_worth
        self._last_net_worth = net_worth
        # advance to next timestep
        self.current += 1
        terminated = self.current >= len(self.data)
        return self._get_state(), reward, terminated, terminated, {}
