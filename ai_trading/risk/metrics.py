"""
Risk metrics calculation and drawdown analysis.

Provides comprehensive risk metrics calculation including VaR,
drawdown analysis, and institutional risk measurement tools.
"""

import logging
import math
import statistics

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger


class RiskMetricsCalculator:
    """
    Comprehensive risk metrics calculator.

    Calculates various risk metrics including VaR, Expected Shortfall,
    Sharpe ratio, and other institutional risk measures.
    """

    def __init__(self):
        """Initialize risk metrics calculator."""
        # AI-AGENT-REF: Risk metrics calculation
        self.confidence_levels = [0.90, 0.95, 0.99]
        logger.info("RiskMetricsCalculator initialized")

    def calculate_var(
        self, returns: list[float], confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: List of returns
            confidence_level: Confidence level for VaR

        Returns:
            VaR value
        """
        try:
            if len(returns) < 30:
                logger.warning(f"Insufficient data for VaR: {len(returns)} returns")
                return 0.0

            sorted_returns = sorted(returns)
            index = int((1 - confidence_level) * len(sorted_returns))
            var = abs(sorted_returns[index])

            return var

        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0

    def calculate_expected_shortfall(
        self, returns: list[float], confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Args:
            returns: List of returns
            confidence_level: Confidence level

        Returns:
            Expected Shortfall value
        """
        try:
            if len(returns) < 30:
                return 0.0

            sorted_returns = sorted(returns)
            index = int((1 - confidence_level) * len(sorted_returns))

            worst_returns = sorted_returns[:index]
            if worst_returns:
                es = abs(statistics.mean(worst_returns))
            else:
                es = self.calculate_var(returns, confidence_level)

            return es

        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.0

    def calculate_sharpe_ratio(
        self, returns: list[float], risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) < 2:
                return 0.0

            excess_returns = [r - risk_free_rate / 252 for r in returns]
            mean_excess = statistics.mean(excess_returns)

            if len(excess_returns) < 2:
                return 0.0

            std_excess = statistics.stdev(excess_returns)
            if std_excess == 0:
                return 0.0

            sharpe = mean_excess / std_excess * (252**0.5)
            return sharpe

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_sortino_ratio(
        self, returns: list[float], risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio."""
        try:
            if len(returns) < 2:
                return 0.0

            excess_returns = [r - risk_free_rate / 252 for r in returns]
            mean_excess = statistics.mean(excess_returns)

            # Only consider negative returns for downside deviation
            negative_returns = [r for r in excess_returns if r < 0]

            if not negative_returns:
                return float("inf") if mean_excess > 0 else 0.0

            downside_deviation = math.sqrt(
                sum(r**2 for r in negative_returns) / len(negative_returns)
            )

            if downside_deviation == 0:
                return 0.0

            sortino = mean_excess / downside_deviation * (252**0.5)
            return sortino

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0


class DrawdownAnalyzer:
    """
    Drawdown analysis and monitoring.

    Provides comprehensive drawdown analysis including maximum drawdown,
    drawdown duration, and recovery analysis.
    """

    def __init__(self):
        """Initialize drawdown analyzer."""
        # AI-AGENT-REF: Drawdown analysis
        logger.info("DrawdownAnalyzer initialized")

    def calculate_drawdowns(self, values: list[float]) -> dict:
        """
        Calculate comprehensive drawdown statistics.

        Args:
            values: List of portfolio values

        Returns:
            Dictionary with drawdown statistics
        """
        try:
            if not values:
                return {}

            peak = values[0]
            max_dd = 0.0
            current_dd = 0.0
            dd_start_idx = 0
            dd_end_idx = 0
            current_dd_start = 0

            drawdown_periods = []

            for i, value in enumerate(values):
                if value > peak:
                    # New peak, end current drawdown if any
                    if current_dd > 0:
                        drawdown_periods.append(
                            {
                                "start_idx": current_dd_start,
                                "end_idx": i - 1,
                                "duration": i - current_dd_start,
                                "magnitude": current_dd,
                            }
                        )

                    peak = value
                    current_dd_start = i
                    current_dd = 0
                else:
                    # Calculate current drawdown
                    current_dd = (peak - value) / peak if peak > 0 else 0

                    if current_dd > max_dd:
                        max_dd = current_dd
                        dd_start_idx = current_dd_start
                        dd_end_idx = i

            # Add final drawdown if still in one
            if current_dd > 0:
                drawdown_periods.append(
                    {
                        "start_idx": current_dd_start,
                        "end_idx": len(values) - 1,
                        "duration": len(values) - current_dd_start,
                        "magnitude": current_dd,
                    }
                )

            stats = {
                "max_drawdown": max_dd,
                "max_drawdown_start": dd_start_idx,
                "max_drawdown_end": dd_end_idx,
                "max_drawdown_duration": dd_end_idx - dd_start_idx,
                "drawdown_periods": drawdown_periods,
                "num_drawdown_periods": len(drawdown_periods),
                "average_drawdown": (
                    statistics.mean([dd["magnitude"] for dd in drawdown_periods])
                    if drawdown_periods
                    else 0
                ),
                "average_recovery_time": (
                    statistics.mean([dd["duration"] for dd in drawdown_periods])
                    if drawdown_periods
                    else 0
                ),
            }

            return stats

        except Exception as e:
            logger.error(f"Error calculating drawdowns: {e}")
            return {}

    def is_in_drawdown(
        self, current_value: float, peak_value: float
    ) -> tuple[bool, float]:
        """
        Check if currently in drawdown.

        Args:
            current_value: Current portfolio value
            peak_value: Peak portfolio value

        Returns:
            Tuple of (is_in_drawdown, drawdown_magnitude)
        """
        try:
            if peak_value <= 0:
                return False, 0.0

            if current_value >= peak_value:
                return False, 0.0

            drawdown = (peak_value - current_value) / peak_value
            return True, drawdown

        except Exception as e:
            logger.error(f"Error checking drawdown status: {e}")
            return False, 0.0

    def calculate_recovery_time(
        self, values: list[float], drawdown_start: int, drawdown_end: int
    ) -> int | None:
        """
        Calculate recovery time from a drawdown.

        Args:
            values: Portfolio values
            drawdown_start: Start index of drawdown
            drawdown_end: End index of drawdown

        Returns:
            Recovery time in periods, or None if not recovered
        """
        try:
            if drawdown_end >= len(values) or drawdown_start >= len(values):
                return None

            peak_value = values[drawdown_start]

            # Look for recovery after drawdown end
            for i in range(drawdown_end + 1, len(values)):
                if values[i] >= peak_value:
                    return i - drawdown_end

            return None  # Not recovered yet

        except Exception as e:
            logger.error(f"Error calculating recovery time: {e}")
            return None
