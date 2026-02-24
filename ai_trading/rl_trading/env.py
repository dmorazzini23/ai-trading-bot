"""Trading environment with configurable action space and hard risk constraints."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:  # optional dependency
    import numpy as np
except Exception:  # noqa: BLE001 - numpy is optional until env used
    np = None

from ai_trading.logging import get_logger

from . import _load_rl_stack

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - hints only
    import gymnasium as gym


@dataclass
class ActionSpaceConfig:
    """Configuration for action space - ensures consistency between training and inference."""

    action_type: str = "discrete"
    discrete_actions: int = 3
    discrete_step: float = 0.25
    continuous_bounds: tuple[float, float] = (-1.0, 1.0)
    position_limits: tuple[float, float] = (-1.0, 1.0)


@dataclass
class RewardConfig:
    """Configuration for reward calculation and normalization."""

    normalize_rewards: bool = True
    reward_window: int = 100
    base_reward_weight: float = 1.0
    turnover_penalty: float = 0.1
    drawdown_penalty: float = 2.0
    variance_penalty: float = 0.5
    sharpe_bonus: float = 0.1


@dataclass
class ConstraintConfig:
    """Hard constraints used to keep RL episodes economically realistic."""

    initial_cash: float = 100_000.0
    max_leverage: float = 1.0
    max_drawdown: float = 0.25
    max_turnover_per_step: float = 1.0
    terminate_on_violation: bool = True
    violation_penalty: float = 1.0


class RunningStats:
    """Running statistics for reward normalization."""

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self.values = deque(maxlen=window)
        self._mean = 0.0
        self._std = 1.0

    def update(self, value: float) -> None:
        """Update running statistics with new value."""
        self.values.append(value)
        if len(self.values) > 1:
            self._mean = np.mean(self.values)
            self._std = max(np.std(self.values), 1e-08)

    def normalize(self, value: float) -> float:
        """Normalize value using running statistics."""
        return (value - self._mean) / self._std

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std


class TradingEnv:
    """Enhanced trading environment with deferred :mod:`gymnasium` import."""

    def __init__(
        self,
        data: np.ndarray,
        window: int = 10,
        *,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        half_spread: float = 0.0002,
        action_config: ActionSpaceConfig | None = None,
        reward_config: RewardConfig | None = None,
        constraint_config: ConstraintConfig | None = None,
        price_series: np.ndarray | None = None,
    ) -> None:
        stack = _load_rl_stack()
        if stack is None:
            raise ImportError(
                "gymnasium required; install gymnasium to use TradingEnv"
            )
        gym = stack["gym"]
        self._gym = gym
        if np is None:
            raise ImportError("numpy required; install numpy to use TradingEnv")
        env_init = getattr(gym.Env, "__init__", None)
        if callable(env_init):
            try:
                env_init(self)
            except TypeError:
                pass

        matrix = np.asarray(data, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("TradingEnv data must be a 2D array")
        if matrix.shape[0] <= window:
            raise ValueError("TradingEnv data length must exceed window size")
        self.data = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        self.prices = self._normalize_price_series(
            price_series if price_series is not None else self.data[:, 0]
        )
        self.window = window
        self.current = window
        self.action_config = action_config or ActionSpaceConfig()
        self.reward_config = reward_config or RewardConfig()
        self.constraint_config = constraint_config or ConstraintConfig()
        if self.action_config.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(
                self.action_config.discrete_actions
            )
        else:
            self.action_space = gym.spaces.Box(
                low=self.action_config.continuous_bounds[0],
                high=self.action_config.continuous_bounds[1],
                shape=(1,),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(window, self.data.shape[1]), dtype=np.float32
        )
        self.position = 0.0
        self.cash = max(float(self.constraint_config.initial_cash), 1.0)
        self.transaction_cost = max(transaction_cost, 0.0)
        self.slippage = max(slippage, 0.0)
        self.half_spread = max(half_spread, 0.0)
        self._last_net_worth = self.cash
        self._position_history = deque(maxlen=50)
        self._turnover_history = deque(maxlen=50)
        self._drawdown_peak = self.cash
        self._returns_history = deque(maxlen=50)
        self._reward_stats = (
            RunningStats(self.reward_config.reward_window)
            if self.reward_config.normalize_rewards
            else None
        )
        self._action_entropy_history = deque(maxlen=100)
        self._episode_stats = {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "turnover": 0.0,
            "actions_taken": 0,
            "constraint_violations": 0,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        env_reset = getattr(self._gym.Env, "reset", None)
        if callable(env_reset):
            try:
                env_reset(self, seed=seed)
            except TypeError:
                env_reset(self)
        self.current = self.window
        self.position = 0.0
        self.cash = max(float(self.constraint_config.initial_cash), 1.0)
        self._last_net_worth = self.cash
        self._position_history.clear()
        self._turnover_history.clear()
        self._drawdown_peak = self.cash
        self._returns_history.clear()
        self._action_entropy_history.clear()
        self._episode_stats = {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "turnover": 0.0,
            "actions_taken": 0,
            "constraint_violations": 0,
        }
        return (self._get_state(), {})

    def _get_state(self) -> np.ndarray:
        return self.data[self.current - self.window : self.current]

    @staticmethod
    def _normalize_price_series(raw_prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(raw_prices, dtype=np.float32).reshape(-1)
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        prices = np.abs(prices)
        return np.clip(prices, 1e-06, None)

    def _net_worth(self, price: float) -> float:
        return float(self.cash + self.position * price)

    def _exposure_bounds(self) -> tuple[float, float]:
        lower, upper = self.action_config.position_limits
        if lower > upper:
            lower, upper = upper, lower
        max_lev = max(float(self.constraint_config.max_leverage), 0.0)
        lower = max(lower, -max_lev)
        upper = min(upper, max_lev)
        return lower, upper

    def _current_exposure(self, price: float) -> float:
        net_worth = max(self._net_worth(price), 1e-06)
        return float((self.position * price) / net_worth)

    def _target_exposure_from_discrete(self, action: int, price: float) -> float:
        current_exposure = self._current_exposure(price)
        step = max(float(self.action_config.discrete_step), 0.0)
        if action == 1:
            return current_exposure + step
        if action == 2:
            return current_exposure - step
        return current_exposure

    def _target_exposure_from_continuous(self, action: float) -> float:
        return float(
            np.clip(
                action,
                self.action_config.continuous_bounds[0],
                self.action_config.continuous_bounds[1],
            )
        )

    def _execute_target_exposure(
        self,
        target_exposure: float,
        price: float,
    ) -> tuple[float, float, bool]:
        old_net_worth = max(self._net_worth(price), 1e-06)
        lower, upper = self._exposure_bounds()
        clipped_target = float(np.clip(target_exposure, lower, upper))
        target_units = (clipped_target * old_net_worth) / max(price, 1e-06)
        delta_units = target_units - self.position
        adjusted = clipped_target != target_exposure

        turnover_ratio = abs(delta_units) * price / old_net_worth
        max_turnover = max(float(self.constraint_config.max_turnover_per_step), 0.0)
        if max_turnover and turnover_ratio > max_turnover:
            scale = max_turnover / max(turnover_ratio, 1e-06)
            target_units = self.position + delta_units * scale
            delta_units = target_units - self.position
            turnover_ratio = max_turnover
            adjusted = True

        if abs(delta_units) <= 1e-12:
            return 0.0, 0.0, adjusted

        if delta_units > 0:
            exec_price = price * (1.0 + self.half_spread + self.slippage)
            max_affordable = self.cash / (
                max(exec_price, 1e-06) * (1.0 + self.transaction_cost)
            )
            buy_units = min(delta_units, max(max_affordable, 0.0))
            if buy_units <= 1e-12:
                return 0.0, 0.0, True
            trade_notional = buy_units * exec_price
            trade_fees = trade_notional * self.transaction_cost
            self.position += buy_units
            self.cash -= trade_notional + trade_fees
            executed_units = buy_units
        else:
            sell_units = abs(delta_units)
            exec_price = max(price * (1.0 - self.half_spread - self.slippage), 1e-06)
            trade_notional = sell_units * exec_price
            trade_fees = trade_notional * self.transaction_cost
            self.position -= sell_units
            self.cash += trade_notional - trade_fees
            executed_units = sell_units

        turnover_ratio = abs(executed_units) * price / old_net_worth
        return float(turnover_ratio), float(executed_units), adjusted

    def step(self, action: int | float | np.ndarray):
        """Execute one step in the environment with enhanced reward calculation."""
        price = float(self.prices[self.current])
        if self.action_config.action_type == "discrete":
            action_int = int(action)
            target_exposure = self._target_exposure_from_discrete(action_int, price)
            trade_size, trade_units, constraint_adjusted = (
                self._execute_target_exposure(target_exposure, price)
            )
            action_probs = np.zeros(self.action_config.discrete_actions)
            action_probs[action_int] = 1.0
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-08))
        else:
            action_float = float(action) if np.isscalar(action) else float(action[0])
            target_exposure = self._target_exposure_from_continuous(action_float)
            trade_size, trade_units, constraint_adjusted = (
                self._execute_target_exposure(target_exposure, price)
            )
            entropy = 0.5 * np.log(2 * np.pi * np.e)
        self._action_entropy_history.append(entropy)
        net_worth = self.cash + self.position * price
        base_reward = (
            net_worth - self._last_net_worth
        ) * self.reward_config.base_reward_weight
        turnover_penalty = self.reward_config.turnover_penalty * trade_size
        self._drawdown_peak = max(self._drawdown_peak, net_worth)
        current_drawdown = (
            (self._drawdown_peak - net_worth) / self._drawdown_peak
            if self._drawdown_peak > 0
            else 0
        )
        drawdown_penalty = self.reward_config.drawdown_penalty * current_drawdown
        returns = (
            net_worth / self._last_net_worth - 1 if self._last_net_worth > 0 else 0
        )
        self._returns_history.append(returns)
        if len(self._returns_history) > 5:
            rolling_variance = np.var(self._returns_history)
            variance_penalty = (
                self.reward_config.variance_penalty * rolling_variance
            )
        else:
            variance_penalty = 0.0
        if len(self._returns_history) > 10:
            returns_array = np.array(self._returns_history)
            sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-08)
            sharpe_bonus = self.reward_config.sharpe_bonus * max(0, sharpe_ratio)
        else:
            sharpe_bonus = 0.0
        raw_reward = (
            base_reward
            - turnover_penalty
            - drawdown_penalty
            - variance_penalty
            + sharpe_bonus
        )
        leverage = abs(self.position * price) / max(net_worth, 1e-06)
        violations: list[str] = []
        if net_worth <= 0:
            violations.append("net_worth_non_positive")
        if current_drawdown > float(self.constraint_config.max_drawdown):
            violations.append("max_drawdown")
        if leverage > float(self.constraint_config.max_leverage) + 1e-06:
            violations.append("max_leverage")
        if (
            float(self.constraint_config.max_turnover_per_step) > 0
            and trade_size > float(self.constraint_config.max_turnover_per_step) + 1e-06
        ):
            violations.append("max_turnover_per_step")
        if violations:
            raw_reward -= float(self.constraint_config.violation_penalty) * len(violations)

        if self._reward_stats is not None:
            self._reward_stats.update(raw_reward)
            reward = self._reward_stats.normalize(raw_reward)
        else:
            reward = raw_reward
        self._last_net_worth = net_worth
        self._position_history.append(self.position)
        self._turnover_history.append(trade_size)
        self._episode_stats["total_return"] = (
            net_worth / max(float(self.constraint_config.initial_cash), 1e-06) - 1.0
        )
        self._episode_stats["max_drawdown"] = max(
            self._episode_stats["max_drawdown"], current_drawdown
        )
        self._episode_stats["turnover"] += trade_size
        self._episode_stats["actions_taken"] += 1 if trade_size > 1e-06 else 0
        self._episode_stats["constraint_violations"] += len(violations)
        if len(self._returns_history) > 1:
            returns_array = np.array(self._returns_history)
            self._episode_stats["sharpe_ratio"] = np.mean(returns_array) / (
                np.std(returns_array) + 1e-08
            )
        self.current += 1
        data_exhausted = self.current >= len(self.data)
        constraint_terminated = bool(
            violations and bool(self.constraint_config.terminate_on_violation)
        )
        terminated = bool(data_exhausted or constraint_terminated)
        truncated = False
        info = {
            "net_worth": net_worth,
            "raw_reward": raw_reward,
            "normalized_reward": reward,
            "base_reward": base_reward,
            "turnover_penalty": turnover_penalty,
            "drawdown_penalty": drawdown_penalty,
            "variance_penalty": variance_penalty,
            "sharpe_bonus": sharpe_bonus,
            "position": self.position,
            "cash": self.cash,
            "drawdown": current_drawdown,
            "trade_size": trade_size,
            "trade_units": trade_units,
            "leverage": leverage,
            "constraint_adjusted": constraint_adjusted,
            "constraint_violations": tuple(violations),
            "constraint_terminated": constraint_terminated,
            "action_entropy": entropy,
            "avg_entropy": np.mean(self._action_entropy_history)
            if self._action_entropy_history
            else 0,
            "episode_stats": self._episode_stats.copy(),
            "action_config": self.action_config,
            "reward_stats": {
                "mean": self._reward_stats.mean if self._reward_stats else 0,
                "std": self._reward_stats.std if self._reward_stats else 1,
            }
            if self._reward_stats
            else None,
        }
        return (self._get_state(), reward, terminated, truncated, info)

    def get_action_space_config(self) -> ActionSpaceConfig:
        """Get action space configuration for inference consistency."""
        return self.action_config

    def get_reward_config(self) -> RewardConfig:
        """Get reward configuration."""
        return self.reward_config

    def get_constraint_config(self) -> ConstraintConfig:
        """Get hard-risk constraint configuration."""
        return self.constraint_config
