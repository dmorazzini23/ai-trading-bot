"""capital_scaling.py

Utilities for adaptive capital allocation and risk-based position sizing.
"""

import numpy as np
import math


class _CapScaler:
    def scale_position(self, size):
        return size


class CapitalScalingEngine:
    def __init__(self, params=None):
        # AI-AGENT-REF: accept params but scaler no longer uses them
        self.params = params or {}
        self.scaler = _CapScaler()
        # AI-AGENT-REF: base level for compression factor calculations
        self._base = float(self.params.get("COMPRESSION_BASE", 100000))

    def compression_factor(self, balance: float) -> float:
        """Return risk compression factor based on ``balance``."""
        try:
            if balance <= 0 or self._base <= 0:
                return 1.0
            ratio = balance / self._base
            factor = 1.0 / (1.0 + math.log1p(max(ratio - 1.0, 0.0)))
            return max(0.1, min(factor, 1.0))
        except Exception:
            return 1.0

    def scale_position(self, portfolio_value, volatility, drawdown):
        # AI-AGENT-REF: dynamic position sizing with volatility and drawdown
        base_fraction = 0.05  # starting Kelly fraction
        adjusted_fraction = base_fraction * (1 - min(drawdown/0.2, 1))
        adjusted_fraction /= max(volatility, 0.01)
        position_size = portfolio_value * adjusted_fraction
        return max(position_size, 0)

    def update(self, ctx, equity_init):
        # Placeholder for future scaling logic
        pass


def volatility_parity_position(base_risk: float, atr_value: float) -> float:
    """Position size based on volatility parity."""
    if atr_value == 0:
        return 0
    return base_risk / atr_value


def dynamic_fractional_kelly(base_fraction: float, drawdown: float, volatility_spike: bool) -> float:
    """Adjust Kelly fraction based on drawdown and volatility."""
    adjustment = 1.0
    if drawdown > 0.10:
        adjustment *= 0.5
    if volatility_spike:
        adjustment *= 0.7
    return base_fraction * adjustment


# AI-AGENT-REF: adjust Kelly fraction by current drawdown
def drawdown_adjusted_kelly(account_value: float, equity_peak: float, raw_kelly: float) -> float:
    """Scale down kelly fraction during drawdowns."""
    if equity_peak == 0:
        return raw_kelly
    drawdown_ratio = account_value / equity_peak
    adjusted_kelly = raw_kelly * drawdown_ratio
    return max(0, adjusted_kelly)


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Return raw Kelly fraction based on win statistics."""
    edge = win_rate - (1 - win_rate) / win_loss_ratio
    return max(edge / win_loss_ratio, 0)


def pyramiding_add(position: float, profit_atr: float, base_size: float) -> float:
    """Increase ``position`` when profit exceeds 1 ATR up to 2x base size."""
    if position > 0 and profit_atr > 1.0:
        target = 2 * base_size
        return min(position + 0.25 * base_size, target)
    return position


def decay_position(position: float, atr: float, atr_mean: float) -> float:
    """Reduce position when ATR spikes 50%% above its mean."""
    if atr_mean and atr > 1.5 * atr_mean:
        return position * 0.9
    return position

# AI-AGENT-REF: new capital scaling helpers for risk regimes

def fractional_kelly(kelly: float, regime: str = "neutral") -> float:
    """Return Kelly fraction adjusted for market ``regime``."""
    if regime == "risk_on":
        return min(1.0, kelly * 1.2)
    if regime == "risk_off":
        return max(0.0, kelly * 0.5)
    return kelly


def volatility_parity(weights: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    """Scale ``weights`` using inverse volatility."""
    inv_vol = 1 / (volatilities + 1e-9)
    scaled = weights * inv_vol
    return scaled / np.sum(scaled) * np.sum(weights)


def cvar_scaling(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Return scaling factor based on CVaR at ``alpha`` level."""
    sorted_returns = np.sort(returns)
    var = sorted_returns[int(len(sorted_returns) * alpha)]
    cvar = np.mean(sorted_returns[sorted_returns <= var])
    return abs(cvar) if cvar < 0 else 1.0


__all__ = [
    "CapitalScalingEngine",
    "volatility_parity_position",
    "dynamic_fractional_kelly",
    "drawdown_adjusted_kelly",
    "pyramiding_add",
    "decay_position",
    "fractional_kelly",
    "kelly_fraction",
    "volatility_parity",
    "cvar_scaling",
]

# AI-AGENT-REF: alt API functions with explicit parameters
def drawdown_adjusted_kelly_alt(account_value: float, equity_peak: float, raw_kelly: float) -> float:
    """Alternate interface for drawdown_adjusted_kelly."""
    return drawdown_adjusted_kelly(account_value, equity_peak, raw_kelly)


def volatility_parity_position_alt(base_risk: float, atr_value: float) -> float:
    """Alternate interface for volatility_parity_position."""
    return volatility_parity_position(base_risk, atr_value)

# Simple aliases for backward compatibility
drawdown_adjusted_kelly_alias = drawdown_adjusted_kelly
volatility_parity_position_alias = volatility_parity_position

