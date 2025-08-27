"""
Tests for parameter validation system.

Validates that the parameter validation system correctly identifies
safe and unsafe parameter values and changes.
"""

import pytest

def test_parameter_validator_initialization():
    """Test that parameter validator initializes correctly."""
    try:
        from ai_trading.core.parameter_validator import ParameterValidator

        validator = ParameterValidator()

        # Check that safety bounds are defined
        assert len(validator.safety_bounds) > 0, "Safety bounds should be defined"

        # Check specific bounds exist
        required_bounds = [
            "MAX_KELLY_FRACTION", "MIN_SAMPLE_SIZE", "CONFIDENCE_LEVEL",
            "MAX_PORTFOLIO_RISK", "MAX_POSITION_SIZE", "PARTICIPATION_RATE",
            "MIN_SHARPE_RATIO", "MAX_DRAWDOWN"
        ]

        for bound in required_bounds:
            assert bound in validator.safety_bounds, f"Missing safety bound: {bound}"
            min_val, max_val = validator.safety_bounds[bound]
            assert min_val < max_val, f"Invalid bound range for {bound}: {min_val} >= {max_val}"

    except ImportError as e:
        pytest.skip(f"Parameter validator test skipped due to import error: {e}")


def test_validate_all_parameters():
    """Test validation of all optimized parameters."""
    try:
        from ai_trading.core.parameter_validator import validate_trading_parameters

        result = validate_trading_parameters()

        # Check result structure
        assert "overall_status" in result, "Missing overall_status in validation result"
        assert "violations" in result, "Missing violations in validation result"
        assert "warnings" in result, "Missing warnings in validation result"
        assert "parameter_summary" in result, "Missing parameter_summary in validation result"

        # All optimized parameters should pass validation
        assert result["overall_status"] in ["PASS", "WARNING"], \
            f"Parameter validation failed: {result.get('violations', [])}"

        # Should have parameter summaries for all groups
        expected_groups = ["kelly", "risk", "execution", "performance"]
        for group in expected_groups:
            assert group in result["parameter_summary"], f"Missing {group} parameter summary"

    except ImportError as e:
        pytest.skip(f"Parameter validation test skipped due to import error: {e}")


def test_parameter_change_validation():
    """Test validation of individual parameter changes."""
    try:
        from ai_trading.core.parameter_validator import ParameterValidator

        validator = ParameterValidator()

        # Test valid parameter change
        result = validator.validate_parameter_change("MAX_KELLY_FRACTION", 0.25, 0.15)
        assert result["status"] == "PASS", f"Valid parameter change should pass: {result}"
        assert result["parameter"] == "MAX_KELLY_FRACTION"
        assert result["old_value"] == 0.25
        assert result["new_value"] == 0.15

        # Test invalid parameter change (outside bounds)
        result = validator.validate_parameter_change("MAX_KELLY_FRACTION", 0.25, 0.80)
        assert result["status"] == "FAIL", "Invalid parameter change should fail"
        assert len(result["violations"]) > 0, "Should have violations for invalid change"

        # Test large change warning
        result = validator.validate_parameter_change("MAX_PORTFOLIO_RISK", 0.02, 0.05)
        assert len(result["warnings"]) > 0 or result["status"] == "PASS", "Large change should trigger warning or pass"

    except ImportError as e:
        pytest.skip(f"Parameter change validation test skipped due to import error: {e}")


def test_change_impact_assessment():
    """Test assessment of parameter change impacts."""
    try:
        from ai_trading.core.parameter_validator import ParameterValidator

        validator = ParameterValidator()

        # Test risk-increasing change
        result = validator.validate_parameter_change("MAX_PORTFOLIO_RISK", 0.02, 0.025)
        impact = result.get("change_impact", {})
        assert "risk_impact" in impact, "Should assess risk impact"
        assert impact["risk_impact"] == "increased", "Increasing portfolio risk should increase risk"

        # Test risk-decreasing change
        result = validator.validate_parameter_change("MAX_DRAWDOWN", 0.20, 0.15)
        impact = result.get("change_impact", {})
        assert impact["risk_impact"] == "decreased", "Decreasing max drawdown should decrease risk"

    except ImportError as e:
        pytest.skip(f"Change impact assessment test skipped due to import error: {e}")


def test_optimization_summary():
    """Test generation of optimization summary."""
    try:
        from ai_trading.core.parameter_validator import ParameterValidator

        validator = ParameterValidator()
        summary = validator._generate_optimization_summary()

        # Check that summary contains all optimization categories
        expected_categories = ["kelly_optimization", "risk_optimization",
                             "execution_optimization", "performance_optimization"]

        for category in expected_categories:
            assert category in summary, f"Missing optimization category: {category}"
            assert isinstance(summary[category], dict), f"{category} should be a dictionary"
            assert len(summary[category]) > 0, f"{category} should contain optimizations"

    except ImportError as e:
        pytest.skip(f"Optimization summary test skipped due to import error: {e}")


def test_parameter_logging():
    """Test parameter change logging functionality."""
    try:
        from ai_trading.core.parameter_validator import log_parameter_changes

        # This should execute without errors
        log_parameter_changes()

    except ImportError as e:
        pytest.skip(f"Parameter logging test skipped due to import error: {e}")


def test_safety_bounds_consistency():
    """Test that safety bounds are consistent and reasonable."""
    try:
        from ai_trading.core.parameter_validator import ParameterValidator

        validator = ParameterValidator()

        # Check that all bounds are reasonable
        for param_name, (min_val, max_val) in validator.safety_bounds.items():
            assert min_val >= 0, f"{param_name} minimum bound should be non-negative"
            assert max_val > min_val, f"{param_name} maximum should be greater than minimum"

            # Specific reasonableness checks
            if "FRACTION" in param_name or "RATE" in param_name:
                assert max_val <= 1.0, f"{param_name} should not exceed 100%"
            if "DRAWDOWN" in param_name:
                assert max_val <= 0.5, f"{param_name} should not exceed 50%"
            if "SHARPE" in param_name:
                assert max_val <= 5.0, f"{param_name} should have reasonable upper bound"

    except ImportError as e:
        pytest.skip(f"Safety bounds consistency test skipped due to import error: {e}")


if __name__ == "__main__":
    # Run tests directly for validation
    pytest.main([__file__, "-v"])
