"""
Risk management and portfolio assessment for institutional trading.

Provides comprehensive risk monitoring, portfolio risk assessment,
and real-time risk controls for institutional trading operations.
"""

from ai_trading.exc import COMMON_EXC  # AI-AGENT-REF: narrow handler
import math
import statistics
from datetime import UTC, datetime, timedelta

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger

from ..core.constants import RISK_PARAMETERS
from ..core.enums import RiskLevel
from .kelly import KellyCalculator


class RiskManager:
    """
    Comprehensive risk management system for institutional trading.

    Provides real-time risk monitoring, position sizing controls,
    and automated risk management actions.
    """

    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        """Initialize risk manager with specified risk level."""
        # AI-AGENT-REF: Institutional risk management system
        self.risk_level = risk_level
        self.kelly_calculator = KellyCalculator()

        # Risk parameters from configuration
        self.max_portfolio_risk = RISK_PARAMETERS["MAX_PORTFOLIO_RISK"]
        self.max_position_size = risk_level.max_position_size
        self.max_drawdown_threshold = risk_level.max_drawdown_threshold
        self.max_correlation_exposure = RISK_PARAMETERS["MAX_CORRELATION_EXPOSURE"]

        # Risk state tracking
        self.current_portfolio_risk = 0.0
        self.current_drawdown = 0.0
        self.risk_alerts = []

        logger.info(
            f"RiskManager initialized with risk_level={risk_level}, "
            f"max_position_size={self.max_position_size}"
        )

    def assess_trade_risk(
        self,
        symbol: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        position_history: list[dict],
    ) -> dict:
        """
        Assess risk for a proposed trade.

        Args:
            symbol: Trading symbol
            quantity: Proposed trade quantity
            price: Proposed trade price
            portfolio_value: Current portfolio value
            position_history: Historical position data

        Returns:
            Risk assessment dictionary with recommendations
        """
        try:
            assessment = {
                "symbol": symbol,
                "approved": False,
                "risk_score": 0.0,
                "recommended_size": 0,
                "warnings": [],
                "metrics": {},
            }

            # Calculate position size as percentage of portfolio
            notional_value = abs(quantity * price)
            position_size_pct = (
                notional_value / portfolio_value if portfolio_value > 0 else 0
            )

            assessment["metrics"]["position_size_pct"] = position_size_pct
            assessment["metrics"]["notional_value"] = notional_value

            # Position size check
            if position_size_pct > self.max_position_size:
                assessment["warnings"].append(
                    f"Position size {position_size_pct:.2%} exceeds maximum {self.max_position_size:.2%}"
                )
                # Recommend reduced size
                max_notional = portfolio_value * self.max_position_size
                assessment["recommended_size"] = int(max_notional / price)
            else:
                assessment["recommended_size"] = quantity

            # Portfolio risk check
            estimated_portfolio_risk = self.current_portfolio_risk + (
                position_size_pct * 0.5
            )  # Rough estimate
            if estimated_portfolio_risk > self.max_portfolio_risk:
                assessment["warnings"].append(
                    f"Estimated portfolio risk {estimated_portfolio_risk:.2%} exceeds maximum {self.max_portfolio_risk:.2%}"
                )

            # Calculate Kelly-based recommendation if history available
            if position_history:
                kelly_fraction, kelly_stats = (
                    self.kelly_calculator.kelly_criterion.calculate_from_returns(
                        [
                            p.get("return", 0.0) for p in position_history[-30:]
                        ]  # Last 30 trades
                    )
                )

                kelly_size = int((portfolio_value * kelly_fraction) / price)
                assessment["metrics"]["kelly_recommended_size"] = kelly_size
                assessment["metrics"]["kelly_fraction"] = kelly_fraction

                if kelly_size < assessment["recommended_size"]:
                    assessment["recommended_size"] = kelly_size
                    assessment["warnings"].append(
                        f"Kelly criterion recommends smaller position: {kelly_size}"
                    )

            # Risk score calculation (0-100)
            risk_score = 0
            risk_score += min(position_size_pct * 500, 50)  # Position size component
            risk_score += len(assessment["warnings"]) * 10  # Warning penalty

            assessment["risk_score"] = risk_score
            assessment["approved"] = (
                risk_score < 70 and len(assessment["warnings"]) == 0
            )

            logger.debug(
                f"Trade risk assessment for {symbol}: risk_score={risk_score}, "
                f"approved={assessment['approved']}"
            )

            return assessment

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error assessing trade risk: {e}")
            return {
                "symbol": symbol,
                "approved": False,
                "risk_score": 100,
                "recommended_size": 0,
                "warnings": [f"Risk assessment error: {e}"],
                "metrics": {},
            }

    def check_portfolio_risk(self, positions: list[dict], market_data: dict) -> dict:
        """
        Comprehensive portfolio risk assessment.

        Args:
            positions: List of current portfolio positions
            market_data: Current market data

        Returns:
            Portfolio risk assessment
        """
        try:
            assessment = {
                "overall_risk_level": "Unknown",
                "risk_score": 0.0,
                "alerts": [],
                "metrics": {},
                "recommendations": [],
            }

            if not positions:
                assessment["overall_risk_level"] = "Low"
                return assessment

            total_value = sum(pos.get("market_value", 0) for pos in positions)

            # Concentration risk
            max_position_pct = (
                max(pos.get("market_value", 0) / total_value for pos in positions)
                if total_value > 0
                else 0
            )
            assessment["metrics"]["max_position_concentration"] = max_position_pct

            if max_position_pct > self.max_position_size:
                assessment["alerts"].append(
                    f"High concentration risk: {max_position_pct:.2%}"
                )

            # Sector concentration
            sector_exposure = {}
            for pos in positions:
                sector = pos.get("sector", "Unknown")
                sector_exposure[sector] = sector_exposure.get(sector, 0) + pos.get(
                    "market_value", 0
                )

            max_sector_pct = (
                max(exposure / total_value for exposure in sector_exposure.values())
                if total_value > 0
                else 0
            )
            assessment["metrics"]["max_sector_concentration"] = max_sector_pct

            if max_sector_pct > RISK_PARAMETERS["MAX_SECTOR_CONCENTRATION"]:
                assessment["alerts"].append(
                    f"High sector concentration: {max_sector_pct:.2%}"
                )

            # Drawdown check
            if self.current_drawdown > self.max_drawdown_threshold:
                assessment["alerts"].append(
                    f"Drawdown {self.current_drawdown:.2%} exceeds threshold {self.max_drawdown_threshold:.2%}"
                )
                assessment["recommendations"].append("Reduce position sizes")

            # Overall risk score
            risk_score = 0
            risk_score += max_position_pct * 100
            risk_score += max_sector_pct * 100
            risk_score += self.current_drawdown * 200
            risk_score += len(assessment["alerts"]) * 10

            assessment["risk_score"] = min(risk_score, 100)

            # Risk level classification
            if assessment["risk_score"] < 25:
                assessment["overall_risk_level"] = "Low"
            elif assessment["risk_score"] < 50:
                assessment["overall_risk_level"] = "Medium"
            elif assessment["risk_score"] < 75:
                assessment["overall_risk_level"] = "High"
            else:
                assessment["overall_risk_level"] = "Critical"

            self.current_portfolio_risk = assessment["risk_score"] / 100

            return assessment

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error checking portfolio risk: {e}")
            return {
                "overall_risk_level": "Critical",
                "risk_score": 100.0,
                "alerts": [f"Risk assessment error: {e}"],
                "metrics": {},
                "recommendations": ["Manual review required"],
            }

    def update_drawdown(self, current_value: float, peak_value: float):
        """Update current drawdown metrics."""
        try:
            if peak_value > 0:
                self.current_drawdown = max(
                    0, (peak_value - current_value) / peak_value
                )
                logger.debug(f"Drawdown updated: {self.current_drawdown:.3f}")
        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error updating drawdown: {e}")

    def get_risk_alerts(self) -> list[dict]:
        """Get current risk alerts."""
        return self.risk_alerts.copy()

    def add_risk_alert(self, alert_type: str, message: str, severity: str = "medium"):
        """Add a risk alert."""
        try:
            alert = {
                "timestamp": datetime.now(UTC),
                "type": alert_type,
                "message": message,
                "severity": severity,
            }
            self.risk_alerts.append(alert)

            # Keep only recent alerts
            cutoff_time = datetime.now(UTC) - timedelta(hours=24)
            self.risk_alerts = [
                a for a in self.risk_alerts if a["timestamp"] >= cutoff_time
            ]

            logger.warning(f"Risk alert added: {alert_type} - {message}")

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error adding risk alert: {e}")


class PortfolioRiskAssessor:
    """
    Portfolio-level risk assessment and monitoring.

    Provides comprehensive portfolio risk analysis including
    correlation analysis, VaR calculation, and stress testing.
    """

    def __init__(self):
        """Initialize portfolio risk assessor."""
        # AI-AGENT-REF: Portfolio risk assessment and monitoring
        self.confidence_levels = [0.95, 0.99]
        self.lookback_periods = 252  # Trading days

        logger.info("PortfolioRiskAssessor initialized")

    def calculate_var(
        self, returns: list[float], confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR) for given returns.

        Args:
            returns: List of portfolio returns
            confidence_level: Confidence level for VaR calculation

        Returns:
            VaR value (positive number representing potential loss)
        """
        try:
            if len(returns) < 30:
                logger.warning(
                    f"Insufficient data for VaR calculation: {len(returns)} returns"
                )
                return 0.0

            # Sort returns in ascending order
            sorted_returns = sorted(returns)

            # Calculate percentile index
            index = int((1 - confidence_level) * len(sorted_returns))
            var = abs(sorted_returns[index])

            logger.debug(f"VaR ({confidence_level:.0%}): {var:.4f}")
            return var

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error calculating VaR: {e}")
            return 0.0

    def calculate_expected_shortfall(
        self, returns: list[float], confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Args:
            returns: List of portfolio returns
            confidence_level: Confidence level for ES calculation

        Returns:
            Expected Shortfall value
        """
        try:
            if len(returns) < 30:
                return 0.0

            var = self.calculate_var(returns, confidence_level)

            # Calculate average of returns worse than VaR
            sorted_returns = sorted(returns)
            index = int((1 - confidence_level) * len(sorted_returns))

            worst_returns = sorted_returns[:index]
            if worst_returns:
                expected_shortfall = abs(statistics.mean(worst_returns))
            else:
                expected_shortfall = var

            logger.debug(
                f"Expected Shortfall ({confidence_level:.0%}): {expected_shortfall:.4f}"
            )
            return expected_shortfall

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.0

    def calculate_correlation_matrix(
        self, asset_returns: dict[str, list[float]]
    ) -> dict[str, float]:
        """
        Calculate correlation matrix for portfolio assets.

        Args:
            asset_returns: Dictionary mapping symbols to return lists

        Returns:
            Dictionary mapping asset pairs to correlation coefficients
        """
        try:
            correlations = {}
            symbols = list(asset_returns.keys())

            for i, symbol1 in enumerate(symbols):
                for _j, symbol2 in enumerate(symbols[i + 1 :], i + 1):
                    returns1 = asset_returns[symbol1]
                    returns2 = asset_returns[symbol2]

                    # Ensure same length
                    min_length = min(len(returns1), len(returns2))
                    if min_length < 30:
                        continue

                    returns1 = returns1[-min_length:]
                    returns2 = returns2[-min_length:]

                    # Calculate correlation
                    correlation = self._calculate_correlation(returns1, returns2)
                    correlations[f"{symbol1}_{symbol2}"] = correlation
                    correlations[f"{symbol2}_{symbol1}"] = correlation

            return correlations

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error calculating correlation matrix: {e}")
            return {}

    def _calculate_correlation(
        self, returns1: list[float], returns2: list[float]
    ) -> float:
        """Calculate correlation coefficient between two return series."""
        try:
            if len(returns1) != len(returns2) or len(returns1) < 2:
                return 0.0

            mean1 = statistics.mean(returns1)
            mean2 = statistics.mean(returns2)

            numerator = sum(
                (r1 - mean1) * (r2 - mean2)
                for r1, r2 in zip(returns1, returns2, strict=False)
            )

            sum_sq1 = sum((r1 - mean1) ** 2 for r1 in returns1)
            sum_sq2 = sum((r2 - mean2) ** 2 for r2 in returns2)

            denominator = math.sqrt(sum_sq1 * sum_sq2)

            if denominator == 0:
                return 0.0

            correlation = numerator / denominator
            return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error calculating correlation: {e}")
            return 0.0

    def stress_test_portfolio(
        self, positions: list[dict], stress_scenarios: dict
    ) -> dict:
        """
        Perform stress testing on portfolio under various scenarios.

        Args:
            positions: Current portfolio positions
            stress_scenarios: Dictionary of stress test scenarios

        Returns:
            Stress test results
        """
        try:
            results = {
                "scenarios": {},
                "worst_case_loss": 0.0,
                "best_case_gain": 0.0,
                "scenario_count": 0,
            }

            total_value = sum(pos.get("market_value", 0) for pos in positions)
            if total_value == 0:
                return results

            for scenario_name, scenario_data in stress_scenarios.items():
                scenario_result = self._apply_stress_scenario(positions, scenario_data)
                results["scenarios"][scenario_name] = scenario_result

                loss_pct = scenario_result.get("portfolio_change_pct", 0)
                results["worst_case_loss"] = min(results["worst_case_loss"], loss_pct)
                results["best_case_gain"] = max(results["best_case_gain"], loss_pct)

            results["scenario_count"] = len(stress_scenarios)

            logger.info(
                f"Stress test completed: {results['scenario_count']} scenarios, "
                f"worst case: {results['worst_case_loss']:.2%}"
            )

            return results

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error in stress testing: {e}")
            return {"error": str(e)}

    def _apply_stress_scenario(self, positions: list[dict], scenario: dict) -> dict:
        """Apply a single stress scenario to the portfolio."""
        try:
            total_before = sum(pos.get("market_value", 0) for pos in positions)
            total_after = 0.0

            for position in positions:
                symbol = position.get("symbol", "")
                current_value = position.get("market_value", 0)

                # Apply scenario shock
                shock = scenario.get(symbol, scenario.get("market_shock", 0))
                new_value = current_value * (1 + shock)
                total_after += new_value

            change_pct = (
                (total_after - total_before) / total_before if total_before > 0 else 0
            )

            return {
                "portfolio_value_before": total_before,
                "portfolio_value_after": total_after,
                "portfolio_change": total_after - total_before,
                "portfolio_change_pct": change_pct,
            }

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error applying stress scenario: {e}")
            return {"error": str(e)}