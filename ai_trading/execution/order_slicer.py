"""Utility helpers for slicing orders with fill adjustments."""
from __future__ import annotations

from typing import Any, Mapping

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def apply_fill_ratio(
    slice_data: Mapping[str, Any] | None,
    fill_ratio: float | int | None,
) -> int:
    """Return filled quantity for a slice safely applying a fill ratio.

    Args:
        slice_data: Mapping that may contain ``slice_qty``.
        fill_ratio: Ratio of quantity filled (0-1).

    Returns:
        Estimated filled quantity. ``0`` when inputs are invalid.
    """
    raw_qty = 0
    if slice_data is not None:
        raw_qty = slice_data.get("slice_qty", 0)
    try:
        slice_qty = int(raw_qty)
    except (TypeError, ValueError):
        logger.warning("Invalid slice_qty %s; defaulting to 0", raw_qty)
        slice_qty = 0

    try:
        ratio = float(fill_ratio)
    except (TypeError, ValueError):
        logger.warning("Invalid fill_ratio %s; defaulting to 0", fill_ratio)
        ratio = 0.0

    if slice_qty <= 0 or ratio <= 0:
        return 0
    return int(slice_qty * ratio)
