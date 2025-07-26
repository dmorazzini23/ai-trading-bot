"""capital_scaling.py

Utilities for adaptive capital allocation and risk-based position sizing.
"""

import numpy as np
import math


class _CapScaler:
    def __init__(self, config=None):
        self.config = config or {}

    def __call__(self, size: float) -> float:
        return self.scale_position(size)

    def scale_position(self, size: float) -> float:
        return size


class CapitalScalingEngine:
    """
    Engine responsible for adaptive capital scaling.  This implementation
    introduces a configurable base equity (``_base``) used to compute
    compression factors.  As the account grows, the base is updated to
    reflect new equity peaks.  Position sizing is adjusted using
    volatility and drawdown to preserve capital during turbulent markets.
    """

    def __init__(self, config, initial_equity: float | None = None) -> None:
        """
        Create a ``CapitalScalingEngine``.

        Parameters
        ----------
        config : dict
            Configuration dictionary (unused for now but kept for
            backwards‑compatibility).
        initial_equity : float, optional
            Starting account equity.  If provided this value will be used as
            the baseline for compression; otherwise the baseline remains
            unset until first update.  A non‑zero baseline prevents
            division‑by‑zero when computing compression factors.
        """
        self._scaler = _CapScaler(config)
        # baseline account value used for compression
        self._base: float | None = initial_equity if initial_equity and initial_equity > 0 else None

    def scale_position(
        self,
        size: float,
        *,
        equity: float | None = None,
        volatility: float = 0.0,
        drawdown: float = 0.0,
    ) -> float:
        """
        Return a position size adjusted for volatility, drawdown and
        compression factor.

        This implementation differentiates between the **equity** of the
        account (used for risk scaling) and the **desired trade size**
        provided by the caller.  When a baseline is set, the position
        sizing uses account equity rather than the raw trade size.  The
        returned value is capped at ``size`` so that compression cannot
        inflate a trade beyond the caller's intent.

        Parameters
        ----------
        size : float
            Proposed position size in dollars.
        equity : float, optional
            Current account equity.  If provided and positive, this value
            will be used to compute the compression factor and Kelly
            sizing.  When omitted, the function falls back to ``size``
            as a proxy for equity (maintaining backward‑compatibility).
        volatility : float, optional
            Realised volatility of the portfolio or asset.  Defaults to ``0``.
        drawdown : float, optional
            Current drawdown fraction (0–1).  Defaults to ``0``.

        Returns
        -------
        float
            Dollar amount of the position to take after risk adjustments.
        """
        # If no baseline is available, return the requested size
        if self._base is None:
            return size
        # Determine the effective equity used for risk computations
        eff_equity = equity if equity is not None and equity > 0 else size
        # Compute compression factor based on account equity relative to baseline
        factor = self.compression_factor(eff_equity)
        # Compute the theoretical dollar size based on risk metrics
        base_size = self.compute_position_size(eff_equity, volatility, drawdown)
        adjusted = base_size * factor
        # Never exceed the caller's proposed size
        return min(size, adjusted)

    def compression_factor(self, equity: float) -> float:
        """
        Return risk compression factor based on account ``equity`` and baseline.

        When the account has grown beyond its baseline, the compression
        factor tapers down using a log attenuation.  Values are
        constrained between ``0.1`` and ``1.0`` to prevent extreme
        scaling.
        """
        try:
            # bail out if baseline is not set or equity is invalid
            if self._base is None or equity <= 0:
                return 1.0
            ratio = equity / self._base
            # ratio > 1 implies account growth; compress accordingly
            factor = 1.0 / (1.0 + math.log1p(max(ratio - 1.0, 0.0)))
            return max(0.1, min(factor, 1.0))
        except Exception:
            return 1.0

    def compute_position_size(self, equity: float, volatility: float, drawdown: float) -> float:
        """
        Compute a dollar position using a fractional Kelly approach.

        The position is proportional to account ``equity`` and scaled
        down when volatility spikes or drawdown deepens.  A baseline
        Kelly fraction of 5 % is throttled linearly when drawdown
        exceeds 20 % and inversely with volatility.

        Returns ``0`` for invalid inputs.
        """
        try:
            base_fraction = 0.05  # starting Kelly fraction
            if equity <= 0:
                return 0.0
            # shrink fraction based on drawdown (capped at 20 %)
            adjusted_fraction = base_fraction * (1 - min(drawdown / 0.2, 1))
            # increase exposure when volatility is low; protect when high
            adjusted_fraction /= max(volatility, 0.01)
            position_size = equity * adjusted_fraction
            return max(position_size, 0.0)
        except Exception:
            return 0.0

    def update(self, ctx, equity: float) -> None:
        """
        Update internal state with the latest account equity.  If the
        equity exceeds the previous peak, the baseline is raised to the
        new value.  This method should be called at the end of each
        trading cycle.

        Parameters
        ----------
        ctx : Any
            Unused context placeholder for future use.
        equity : float
            Current account equity.
        """
        try:
            if equity and equity > 0:
                if self._base is None:
                    self._base = equity
                elif equity > self._base:
                    self._base = equity
        except Exception:
            # swallow errors to avoid crashing calling code
            pass


def volatility_parity_position(base_risk: float, atr_value: float) -> float:
    """Scale position based on volatility parity, with a non-zero minimum floor."""
    if base_risk <= 0 or atr_value == 0:
        return 0.01
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
    "drawdown_adjusted_kelly_alt",
    "volatility_parity_position_alt",
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

# AI-AGENT-REF: fallback stubs for optional imports
if "fractional_kelly" not in globals():
    def fractional_kelly(*args, **kwargs):
        """Stub for import; tests don’t execute this."""
        raise NotImplementedError("fractional_kelly stub")

if "volatility_parity" not in globals():
    def volatility_parity(*args, **kwargs):
        """Stub for import; tests don’t execute this."""
        raise NotImplementedError("volatility_parity stub")

if "cvar_scaling" not in globals():
    def cvar_scaling(*args, **kwargs):
        """Stub for import; tests don’t execute this."""
        raise NotImplementedError("cvar_scaling stub")

if "kelly_fraction" not in globals():
    def kelly_fraction(*args, **kwargs):
        """Stub for import; tests don’t execute this."""
        raise NotImplementedError("kelly_fraction stub")

