"""Portfolio correlation utilities."""
from __future__ import annotations

from typing import Dict, List
from statistics import StatisticsError, correlation

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def calculate_correlation_matrix(returns: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Calculate pairwise correlation coefficients for all symbols.

    Args:
        returns: Mapping of symbols to return series.

    Returns:
        Nested mapping where ``matrix[s1][s2]`` is the correlation between
        ``s1`` and ``s2``. Missing or insufficient data yields ``0.0``.
    """
    symbols = list(returns.keys())
    matrix: Dict[str, Dict[str, float]] = {}
    for i, sym1 in enumerate(symbols):
        series1 = returns.get(sym1, [])
        matrix[sym1] = {}
        for sym2 in symbols:
            if sym1 == sym2:
                continue
            series2 = returns.get(sym2, [])
            if not series1 or not series2:
                matrix[sym1][sym2] = 0.0
                continue
            length = min(len(series1), len(series2))
            try:
                coeff = correlation(series1[:length], series2[:length])
            except (StatisticsError, ZeroDivisionError):
                coeff = 0.0
            matrix[sym1][sym2] = coeff
    logger.debug("Calculated correlation matrix", extra={"matrix": matrix})
    return matrix

__all__ = ["calculate_correlation_matrix"]
