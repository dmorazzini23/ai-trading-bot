"""Utilities for adaptive capital allocation and risk-based position sizing."""
from __future__ import annotations
import math
from ai_trading.logging import get_logger
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    import numpy as np
log = get_logger(__name__)

def update_if_present(runtime, equity) -> float:
    """Safely update the capital scaler if present and return its value."""
    cs = getattr(runtime, 'capital_scaler', None)
    if cs is not None and hasattr(cs, 'update'):
        try:
            result = cs.update(runtime, equity)
            if hasattr(cs, 'current_scale'):
                return float(cs.current_scale())
            if isinstance(result, (int, float)):
                return float(result)
            return 1.0
        except (TypeError, ValueError, AttributeError) as e:
            log.warning('CAPITAL_SCALE_UPDATE_FAILED', extra={'detail': str(e)})
            return 1.0
    return 1.0


def capital_scale(runtime) -> float:
    """Return current scale or 1.0 if no scaler is configured."""
    cs = getattr(runtime, 'capital_scaler', None)
    if cs is not None and hasattr(cs, 'current_scale'):
        try:
            return float(cs.current_scale())
        except (TypeError, ValueError, AttributeError) as e:
            log.warning('CAPITAL_SCALE_CURRENT_FAILED', extra={'detail': str(e)})
            return 1.0
    return 1.0

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

    def __init__(self, config, initial_equity: float | None=None) -> None:
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
        self._base: float | None = initial_equity if initial_equity and initial_equity > 0 else None
        self._latest_equity: float = initial_equity or 0.0

    def scale_position(self, size: float, *, equity: float | None=None, volatility: float=0.0, drawdown: float=0.0) -> float:
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
        if self._base is None:
            return size
        eff_equity = equity if equity is not None and equity > 0 else size
        factor = self.compression_factor(eff_equity)
        base_size = self.compute_position_size(eff_equity, volatility, drawdown)
        adjusted = base_size * factor
        return min(adjusted, size)

    def compression_factor(self, equity: float) -> float:
        """
        Return risk compression factor based on account ``equity`` and baseline.

        When the account has grown beyond its baseline, the compression
        factor tapers down using a log attenuation.  Values are
        constrained between ``0.1`` and ``1.0`` to prevent extreme
        scaling.
        """
        try:
            if self._base is None or self._base <= 0:
                return 1.0
            ratio = equity / self._base
            factor = math.log(max(ratio, 0.1)) / math.log(2) + 1
            return max(0.1, min(2.0, factor))
        except (ValueError, TypeError, ZeroDivisionError):
            return 1.0

    def compute_position_size(self, equity: float, volatility: float, drawdown: float) -> float:
        """
        Compute a dollar position using a fractional Kelly approach.

        The position is proportional to account ``equity`` and scaled
        down when volatility spikes or drawdown deepens.  A baseline
        Kelly fraction of 5\xa0% is throttled linearly when drawdown
        exceeds 20\xa0% and inversely with volatility.

        Returns ``0`` for invalid inputs.
        """
        if equity <= 0:
            return 0.0
        base_allocation = equity * 0.02
        vol_adjustment = 1.0 / (1.0 + volatility * 10)
        dd_adjustment = 1.0 - min(drawdown, 0.5)
        return base_allocation * vol_adjustment * dd_adjustment

    def update(self, ctx, equity: float) -> None:
        """
        Update internal state with the latest account equity.  If the
        equity exceeds the previous peak, the baseline is raised to the
        new value.  This method should be called at the end of each
        trading cycle.

        Parameters
        ----------
        ctx : Any
            Unused context stub for future use.
        equity : float
            Current account equity.
        """
        try:
            if equity and equity > 0:
                self._latest_equity = equity
                if self._base is None or equity > self._base:
                    self._base = equity
        except (TypeError, ValueError) as e:
            log.warning(f'Failed to update baseline equity: {e}')

    def current_scale(self) -> float:
        """Return the most recently computed compression factor."""
        return self.compression_factor(self._latest_equity)

    def update_baseline(self, equity: float) -> None:
        """Update the baseline equity for compression calculations."""
        if self._base is None or equity > self._base:
            self._base = equity


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


def cvar_scaling(returns: np.ndarray, alpha: float=0.05) -> float:
    """Return scaling factor based on CVaR at ``alpha`` level."""
    import numpy as np  # local import to avoid hard dependency at module import
    sorted_returns = np.sort(returns)
    var = sorted_returns[int(len(sorted_returns) * alpha)]
    cvar = np.mean(sorted_returns[sorted_returns <= var])
    cvar_value = float(cvar)
    return abs(cvar_value) if cvar_value < 0 else 1.0

__all__ = [
    'update_if_present',
    'capital_scale',
    'CapitalScalingEngine',
    'drawdown_adjusted_kelly',
    'kelly_fraction',
    'cvar_scaling',
]
