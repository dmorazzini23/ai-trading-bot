"""capital_scaling.py

Utilities for adaptive capital allocation and risk-based position sizing.
"""


class _CapScaler:
    def scale_position(self, size):
        return size


class CapitalScalingEngine:
    def __init__(self, params=None):
        # AI-AGENT-REF: accept params but scaler no longer uses them
        self.params = params or {}
        self.scaler = _CapScaler()

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
    """Return position size using volatility parity."""
    return base_risk / atr_value if atr_value else 0.0


def dynamic_fractional_kelly(base_fraction: float, drawdown: float, volatility_spike: bool) -> float:
    """Adjust Kelly fraction based on drawdown and volatility."""
    adjustment = 1.0
    if drawdown > 0.10:
        adjustment *= 0.5
    if volatility_spike:
        adjustment *= 0.7
    return base_fraction * adjustment


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


__all__ = [
    "CapitalScalingEngine",
    "volatility_parity_position",
    "dynamic_fractional_kelly",
    "pyramiding_add",
    "decay_position",
]
