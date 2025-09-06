"""Risk parameter optimization utilities.

Provides lightweight helpers for deriving risk parameters from
optimization routines. These functions intentionally avoid heavy
imports to keep startup lean.
"""
from __future__ import annotations


def optimize_stop_loss_multiplier(base: float = 2.0, reduction: float = 0.1) -> float:
    """Return an optimized stop loss multiplier.

    The previous implementation subtracted the reduction directly from
    the base value, yielding ``1.9`` for ``base=2`` and ``reduction=0.1``.
    The optimization should apply a proportional reduction instead
    (``base * (1 - reduction)``) which gives the expected ``1.8`` value.

    Args:
        base: Original stop loss multiplier before optimization.
        reduction: Fractional reduction to apply (e.g. ``0.1`` for 10%%).

    Returns:
        Optimized stop loss multiplier rounded to two decimals.
    """
    optimized = base * (1 - reduction)
    return round(optimized, 2)
