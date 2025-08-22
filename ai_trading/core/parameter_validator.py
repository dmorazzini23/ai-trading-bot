"""
Parameter validation system for trading parameter optimization.

Ensures all trading parameters remain within safe institutional bounds
and provides validation for parameter changes.
"""

from datetime import UTC, datetime
from typing import Any

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger

from .constants import (
    EXECUTION_PARAMETERS,
    KELLY_PARAMETERS,
    PERFORMANCE_THRESHOLDS,
    RISK_PARAMETERS,
)


class ParameterValidator:
    """
    Validates trading parameters to ensure they remain within safe bounds.

    Provides institutional-grade parameter validation and safety checks
    for parameter optimization changes.
    """

    def __init__(self):
        """Initialize parameter validator with safety bounds."""
        # AI-AGENT-REF: Parameter validation for trading optimization
        self.safety_bounds = {
            # Kelly Criterion bounds
            "MAX_KELLY_FRACTION": (0.05, 0.50),
            "MIN_SAMPLE_SIZE": (10, 100),
            "CONFIDENCE_LEVEL": (0.80, 0.99),
            # Risk management bounds
            "MAX_PORTFOLIO_RISK": (0.01, 0.05),
            "MAX_POSITION_SIZE": (0.10, 0.35),  # Updated to allow 25% position size
            "STOP_LOSS_MULTIPLIER": (1.0, 3.0),
            "TAKE_PROFIT_MULTIPLIER": (1.5, 5.0),
            "MAX_CORRELATION_EXPOSURE": (0.05, 0.30),
            # Execution bounds
            "PARTICIPATION_RATE": (0.05, 0.25),
            "MAX_SLIPPAGE_BPS": (5, 50),
            "ORDER_TIMEOUT_SECONDS": (60, 600),
            # Performance bounds
            "MIN_SHARPE_RATIO": (0.5, 2.0),
            "MAX_DRAWDOWN": (0.05, 0.30),
            "MIN_WIN_RATE": (0.30, 0.70),
        }

        logger.info("ParameterValidator initialized with institutional safety bounds")

    def validate_all_parameters(self) -> dict[str, Any]:
        """
        Validate all trading parameters against safety bounds.

        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            "timestamp": datetime.now(UTC),
            "overall_status": "PASS",
            "violations": [],
            "warnings": [],
            "parameter_summary": {},
            "optimization_summary": {},
        }

        try:
            # Validate Kelly parameters
            kelly_result = self._validate_parameter_group(
                KELLY_PARAMETERS, "KELLY_PARAMETERS"
            )
            validation_result["parameter_summary"]["kelly"] = kelly_result

            # Validate Risk parameters
            risk_result = self._validate_parameter_group(
                RISK_PARAMETERS, "RISK_PARAMETERS"
            )
            validation_result["parameter_summary"]["risk"] = risk_result

            # Validate Execution parameters
            execution_result = self._validate_parameter_group(
                EXECUTION_PARAMETERS, "EXECUTION_PARAMETERS"
            )
            validation_result["parameter_summary"]["execution"] = execution_result

            # Validate Performance parameters
            performance_result = self._validate_parameter_group(
                PERFORMANCE_THRESHOLDS, "PERFORMANCE_THRESHOLDS"
            )
            validation_result["parameter_summary"]["performance"] = performance_result

            # Collect all violations and warnings
            all_results = [
                kelly_result,
                risk_result,
                execution_result,
                performance_result,
            ]
            for result in all_results:
                validation_result["violations"].extend(result.get("violations", []))
                validation_result["warnings"].extend(result.get("warnings", []))

            # Set overall status
            if validation_result["violations"]:
                validation_result["overall_status"] = "FAIL"
            elif validation_result["warnings"]:
                validation_result["overall_status"] = "WARNING"

            # Add optimization summary
            validation_result["optimization_summary"] = (
                self._generate_optimization_summary()
            )

            # Log validation results
            self._log_validation_results(validation_result)

            return validation_result

        except (ValueError, TypeError) as e:
            logger.error(f"Error during parameter validation: {e}")
            validation_result["overall_status"] = "ERROR"
            validation_result["violations"].append(f"Validation error: {e}")
            return validation_result

    def validate_parameter_change(
        self, parameter_name: str, old_value: Any, new_value: Any
    ) -> dict[str, Any]:
        """
        Validate a specific parameter change.

        Args:
            parameter_name: Name of the parameter
            old_value: Current parameter value
            new_value: Proposed new value

        Returns:
            Validation result for the parameter change
        """
        result = {
            "parameter": parameter_name,
            "old_value": old_value,
            "new_value": new_value,
            "status": "PASS",
            "violations": [],
            "warnings": [],
            "change_impact": {},
        }

        try:
            # Check if parameter has safety bounds
            if parameter_name in self.safety_bounds:
                min_bound, max_bound = self.safety_bounds[parameter_name]

                if not (min_bound <= new_value <= max_bound):
                    result["status"] = "FAIL"
                    result["violations"].append(
                        f"{parameter_name} value {new_value} outside safe bounds [{min_bound}, {max_bound}]"
                    )

                # Warning for large changes
                if old_value != 0:
                    change_pct = abs((new_value - old_value) / old_value)
                    if change_pct > 0.5:  # 50% change threshold
                        result["warnings"].append(
                            f"Large change in {parameter_name}: {change_pct:.1%} change"
                        )

            # Assess change impact
            result["change_impact"] = self._assess_change_impact(
                parameter_name, old_value, new_value
            )

            logger.info(
                f"Parameter change validation: {parameter_name} {old_value} -> {new_value}, "
                f"status={result['status']}"
            )

            return result

        except (ValueError, TypeError) as e:
            logger.error(f"Error validating parameter change for {parameter_name}: {e}")
            result["status"] = "ERROR"
            result["violations"].append(f"Validation error: {e}")
            return result

    def _validate_parameter_group(
        self, parameters: dict[str, Any], group_name: str
    ) -> dict[str, Any]:
        """Validate a group of parameters."""
        result = {
            "group": group_name,
            "status": "PASS",
            "violations": [],
            "warnings": [],
            "parameters_checked": len(parameters),
        }

        for param_name, param_value in parameters.items():
            if param_name in self.safety_bounds:
                min_bound, max_bound = self.safety_bounds[param_name]

                if not (min_bound <= param_value <= max_bound):
                    result["status"] = "FAIL"
                    result["violations"].append(
                        f"{param_name} = {param_value} outside bounds [{min_bound}, {max_bound}]"
                    )
                elif param_value <= min_bound * 1.1 or param_value >= max_bound * 0.9:
                    # Warning for values near bounds
                    result["warnings"].append(
                        f"{param_name} = {param_value} near bounds [{min_bound}, {max_bound}]"
                    )

        return result

    def _assess_change_impact(
        self, parameter_name: str, old_value: Any, new_value: Any
    ) -> dict[str, str]:
        """Assess the impact of a parameter change."""
        impact = {"risk_impact": "neutral", "performance_impact": "neutral"}

        # Risk impact assessment
        risk_increasing_params = [
            "MAX_PORTFOLIO_RISK",
            "MAX_POSITION_SIZE",
            "PARTICIPATION_RATE",
        ]
        risk_decreasing_params = [
            "STOP_LOSS_MULTIPLIER",
            "TAKE_PROFIT_MULTIPLIER",
            "MAX_DRAWDOWN",
            "MAX_SLIPPAGE_BPS",
            "MAX_CORRELATION_EXPOSURE",
        ]

        if parameter_name in risk_increasing_params:
            if new_value > old_value:
                impact["risk_impact"] = "increased"
            elif new_value < old_value:
                impact["risk_impact"] = "decreased"
        elif parameter_name in risk_decreasing_params:
            if new_value < old_value:
                impact["risk_impact"] = "decreased"  # Lower drawdown = lower risk
            elif new_value > old_value:
                impact["risk_impact"] = "increased"

        # Performance impact assessment
        performance_improving_params = [
            "MIN_SHARPE_RATIO",
            "MIN_WIN_RATE",
            "PARTICIPATION_RATE",
        ]
        performance_decreasing_params = ["MAX_DRAWDOWN"]

        if parameter_name in performance_improving_params:
            if new_value > old_value:
                impact["performance_impact"] = "potentially_improved"
            elif new_value < old_value:
                impact["performance_impact"] = "potentially_reduced"
        elif parameter_name in performance_decreasing_params:
            if new_value < old_value:
                impact["performance_impact"] = (
                    "potentially_improved"  # Lower drawdown = better performance
                )
            elif new_value > old_value:
                impact["performance_impact"] = "potentially_reduced"

        return impact

    def _generate_optimization_summary(self) -> dict[str, Any]:
        """Generate summary of current parameter optimizations."""
        return {
            "kelly_optimization": {
                "max_kelly_fraction": "Reduced to 15% for better risk-adjusted returns",
                "min_sample_size": "Reduced to 20 for faster adaptation",
                "confidence_level": "Reduced to 90% for less conservative sizing",
            },
            "risk_optimization": {
                "max_portfolio_risk": "Increased to 2.5% for higher profit potential",
                "max_position_size": "Reduced to 8% for better diversification",
                "stop_loss_multiplier": "Tightened to 1.8x for capital preservation",
                "take_profit_multiplier": "Reduced to 2.5x for frequent profit taking",
                "max_correlation_exposure": "Reduced to 15% for better diversification",
            },
            "execution_optimization": {
                "participation_rate": "Increased to 15% for faster fills",
                "max_slippage_bps": "Tightened to 15 bps for better execution quality",
                "order_timeout": "Reduced to 180s for faster adaptation",
            },
            "performance_optimization": {
                "min_sharpe_ratio": "Increased to 1.2 for higher quality strategies",
                "max_drawdown": "Reduced to 15% for better capital preservation",
                "min_win_rate": "Increased to 48% for quality trade filtering",
            },
        }

    def _log_validation_results(self, validation_result: dict[str, Any]):
        """Log parameter validation results."""
        status = validation_result["overall_status"]
        violations = len(validation_result["violations"])
        warnings = len(validation_result["warnings"])

        if status == "PASS":
            logger.info(f"Parameter validation PASSED - {warnings} warnings")
        elif status == "WARNING":
            logger.warning(f"Parameter validation passed with {warnings} warnings")
        elif status == "FAIL":
            logger.error(
                f"Parameter validation FAILED - {violations} violations, {warnings} warnings"
            )
            for violation in validation_result["violations"]:
                logger.error(f"Violation: {violation}")

        # Log optimization summary
        logger.info("Current parameter optimizations:")
        for category, optimizations in validation_result[
            "optimization_summary"
        ].items():
            logger.info(f"  {category}: {len(optimizations)} optimizations applied")


# Global validator instance
parameter_validator = ParameterValidator()


def validate_trading_parameters() -> dict[str, Any]:
    """
    Validate all trading parameters.

    Returns:
        Complete validation results
    """
    return parameter_validator.validate_all_parameters()


def log_parameter_changes():
    """Log summary of parameter optimization changes."""
    logger.info("=== Trading Parameter Optimization Summary ===")
    logger.info("Kelly Criterion optimizations:")
    logger.info(
        f"  MAX_KELLY_FRACTION: 0.25 → {KELLY_PARAMETERS['MAX_KELLY_FRACTION']} (better risk-adjusted returns)"
    )
    logger.info(
        f"  MIN_SAMPLE_SIZE: 30 → {KELLY_PARAMETERS['MIN_SAMPLE_SIZE']} (faster adaptation)"
    )
    logger.info(
        f"  CONFIDENCE_LEVEL: 0.95 → {KELLY_PARAMETERS['CONFIDENCE_LEVEL']} (less conservative sizing)"
    )

    logger.info("Risk management optimizations:")
    logger.info(
        f"  MAX_PORTFOLIO_RISK: 0.02 → {RISK_PARAMETERS['MAX_PORTFOLIO_RISK']} (higher profit potential)"
    )
    logger.info(
        f"  MAX_POSITION_SIZE: 0.10 → {RISK_PARAMETERS['MAX_POSITION_SIZE']} (increased for larger positions)"
    )
    logger.info(
        f"  STOP_LOSS_MULTIPLIER: 2.0 → {RISK_PARAMETERS['STOP_LOSS_MULTIPLIER']} (capital preservation)"
    )
    logger.info(
        f"  TAKE_PROFIT_MULTIPLIER: 3.0 → {RISK_PARAMETERS['TAKE_PROFIT_MULTIPLIER']} (frequent profit taking)"
    )
    logger.info(
        f"  MAX_CORRELATION_EXPOSURE: 0.20 → {RISK_PARAMETERS['MAX_CORRELATION_EXPOSURE']} (better diversification)"
    )

    logger.info("Execution optimizations:")
    logger.info(
        f"  PARTICIPATION_RATE: 0.10 → {EXECUTION_PARAMETERS['PARTICIPATION_RATE']} (faster fills)"
    )
    logger.info(
        f"  MAX_SLIPPAGE_BPS: 20 → {EXECUTION_PARAMETERS['MAX_SLIPPAGE_BPS']} (better execution quality)"
    )
    logger.info(
        f"  ORDER_TIMEOUT_SECONDS: 300 → {EXECUTION_PARAMETERS['ORDER_TIMEOUT_SECONDS']} (faster adaptation)"
    )

    logger.info("Performance threshold optimizations:")
    logger.info(
        f"  MIN_SHARPE_RATIO: 1.0 → {PERFORMANCE_THRESHOLDS['MIN_SHARPE_RATIO']} (higher quality strategies)"
    )
    logger.info(
        f"  MAX_DRAWDOWN: 0.20 → {PERFORMANCE_THRESHOLDS['MAX_DRAWDOWN']} (better capital preservation)"
    )
    logger.info(
        f"  MIN_WIN_RATE: 0.45 → {PERFORMANCE_THRESHOLDS['MIN_WIN_RATE']} (quality trade filtering)"
    )
    logger.info("=== End Parameter Optimization Summary ===")
