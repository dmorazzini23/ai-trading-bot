"""
Tests for trading parameter optimization changes.

Validates that optimized parameters maintain safety standards while
improving profit potential.
"""

import pytest
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_kelly_parameters_optimization():
    """Test that Kelly parameters are optimized correctly."""
    from ai_trading.core.constants import KELLY_PARAMETERS
    
    # Verify optimized Kelly parameters
    assert KELLY_PARAMETERS["MAX_KELLY_FRACTION"] == 0.15, f"Expected 0.15, got {KELLY_PARAMETERS['MAX_KELLY_FRACTION']}"
    assert KELLY_PARAMETERS["MIN_SAMPLE_SIZE"] == 20, f"Expected 20, got {KELLY_PARAMETERS['MIN_SAMPLE_SIZE']}"
    assert KELLY_PARAMETERS["CONFIDENCE_LEVEL"] == 0.90, f"Expected 0.90, got {KELLY_PARAMETERS['CONFIDENCE_LEVEL']}"
    
    # Ensure parameters remain within safe bounds
    assert 0.05 <= KELLY_PARAMETERS["MAX_KELLY_FRACTION"] <= 0.50, "Kelly fraction outside safe bounds"
    assert 10 <= KELLY_PARAMETERS["MIN_SAMPLE_SIZE"] <= 100, "Sample size outside safe bounds"
    assert 0.80 <= KELLY_PARAMETERS["CONFIDENCE_LEVEL"] <= 0.99, "Confidence level outside safe bounds"


def test_risk_parameters_optimization():
    """Test that risk parameters are optimized correctly."""
    from ai_trading.core.constants import RISK_PARAMETERS
    
    # Verify optimized risk parameters
    assert RISK_PARAMETERS["MAX_PORTFOLIO_RISK"] == 0.025, f"Expected 0.025, got {RISK_PARAMETERS['MAX_PORTFOLIO_RISK']}"
    assert RISK_PARAMETERS["MAX_POSITION_SIZE"] == 0.25, f"Expected 0.25, got {RISK_PARAMETERS['MAX_POSITION_SIZE']}"
    assert RISK_PARAMETERS["STOP_LOSS_MULTIPLIER"] == 1.8, f"Expected 1.8, got {RISK_PARAMETERS['STOP_LOSS_MULTIPLIER']}"
    assert RISK_PARAMETERS["TAKE_PROFIT_MULTIPLIER"] == 2.5, f"Expected 2.5, got {RISK_PARAMETERS['TAKE_PROFIT_MULTIPLIER']}"
    assert RISK_PARAMETERS["MAX_CORRELATION_EXPOSURE"] == 0.15, f"Expected 0.15, got {RISK_PARAMETERS['MAX_CORRELATION_EXPOSURE']}"
    
    # Ensure parameters remain within safe bounds
    assert 0.01 <= RISK_PARAMETERS["MAX_PORTFOLIO_RISK"] <= 0.05, "Portfolio risk outside safe bounds"
    assert 0.05 <= RISK_PARAMETERS["MAX_POSITION_SIZE"] <= 0.30, "Position size outside safe bounds"
    assert 1.0 <= RISK_PARAMETERS["STOP_LOSS_MULTIPLIER"] <= 3.0, "Stop loss multiplier outside safe bounds"
    assert 1.5 <= RISK_PARAMETERS["TAKE_PROFIT_MULTIPLIER"] <= 5.0, "Take profit multiplier outside safe bounds"
    assert 0.05 <= RISK_PARAMETERS["MAX_CORRELATION_EXPOSURE"] <= 0.30, "Correlation exposure outside safe bounds"


def test_execution_parameters_optimization():
    """Test that execution parameters are optimized correctly."""
    from ai_trading.core.constants import EXECUTION_PARAMETERS
    
    # Verify optimized execution parameters
    assert EXECUTION_PARAMETERS["PARTICIPATION_RATE"] == 0.15, f"Expected 0.15, got {EXECUTION_PARAMETERS['PARTICIPATION_RATE']}"
    assert EXECUTION_PARAMETERS["MAX_SLIPPAGE_BPS"] == 15, f"Expected 15, got {EXECUTION_PARAMETERS['MAX_SLIPPAGE_BPS']}"
    assert EXECUTION_PARAMETERS["ORDER_TIMEOUT_SECONDS"] == 180, f"Expected 180, got {EXECUTION_PARAMETERS['ORDER_TIMEOUT_SECONDS']}"
    
    # Ensure parameters remain within safe bounds
    assert 0.05 <= EXECUTION_PARAMETERS["PARTICIPATION_RATE"] <= 0.25, "Participation rate outside safe bounds"
    assert 5 <= EXECUTION_PARAMETERS["MAX_SLIPPAGE_BPS"] <= 50, "Slippage outside safe bounds"
    assert 60 <= EXECUTION_PARAMETERS["ORDER_TIMEOUT_SECONDS"] <= 600, "Order timeout outside safe bounds"


def test_performance_thresholds_optimization():
    """Test that performance thresholds are optimized correctly."""
    from ai_trading.core.constants import PERFORMANCE_THRESHOLDS
    
    # Verify optimized performance thresholds
    assert PERFORMANCE_THRESHOLDS["MIN_SHARPE_RATIO"] == 1.2, f"Expected 1.2, got {PERFORMANCE_THRESHOLDS['MIN_SHARPE_RATIO']}"
    assert PERFORMANCE_THRESHOLDS["MAX_DRAWDOWN"] == 0.15, f"Expected 0.15, got {PERFORMANCE_THRESHOLDS['MAX_DRAWDOWN']}"
    assert PERFORMANCE_THRESHOLDS["MIN_WIN_RATE"] == 0.48, f"Expected 0.48, got {PERFORMANCE_THRESHOLDS['MIN_WIN_RATE']}"
    
    # Ensure parameters remain within safe bounds
    assert 0.5 <= PERFORMANCE_THRESHOLDS["MIN_SHARPE_RATIO"] <= 2.0, "Sharpe ratio outside safe bounds"
    assert 0.05 <= PERFORMANCE_THRESHOLDS["MAX_DRAWDOWN"] <= 0.30, "Drawdown outside safe bounds"
    assert 0.30 <= PERFORMANCE_THRESHOLDS["MIN_WIN_RATE"] <= 0.70, "Win rate outside safe bounds"


def test_parameter_consistency():
    """Test that optimized parameters maintain internal consistency."""
    from ai_trading.core.constants import RISK_PARAMETERS, PERFORMANCE_THRESHOLDS
    
    # Stop loss should be lower than take profit
    assert RISK_PARAMETERS["STOP_LOSS_MULTIPLIER"] < RISK_PARAMETERS["TAKE_PROFIT_MULTIPLIER"], \
        "Stop loss should be lower than take profit"
    
    # Drawdown should be reasonable compared to position size
    max_single_position_loss = RISK_PARAMETERS["MAX_POSITION_SIZE"] * 0.20  # Assume 20% worst case
    assert PERFORMANCE_THRESHOLDS["MAX_DRAWDOWN"] > max_single_position_loss, \
        "Max drawdown should account for potential single position losses"


def test_adaptive_sizing_optimization():
    """Test that adaptive sizing uses optimized parameters."""
    try:
        from ai_trading.risk.adaptive_sizing import AdaptivePositionSizer
        from ai_trading.core.enums import RiskLevel
        
        # Test that sizer can be instantiated with optimized parameters
        sizer = AdaptivePositionSizer(RiskLevel.MODERATE)
        
        # Verify regime multipliers are within reasonable bounds
        for regime, multiplier in sizer.regime_multipliers.items():
            assert 0.1 <= multiplier <= 2.0, f"Regime multiplier {multiplier} for {regime} outside safe bounds"
        
        # Verify volatility adjustments are within reasonable bounds  
        for vol_regime, adjustment in sizer.volatility_adjustments.items():
            assert 0.2 <= adjustment <= 2.0, f"Volatility adjustment {adjustment} for {vol_regime} outside safe bounds"
            
    except ImportError as e:
        # Skip test if dependencies not available
        pytest.skip(f"Adaptive sizing test skipped due to import error: {e}")


def test_execution_algorithm_optimization():
    """Test that execution algorithms use optimized parameters."""
    try:
        # Test VWAP participation rate
        from ai_trading.execution.algorithms import VWAPExecutor
        
        # Mock order manager for testing
        vwap = VWAPExecutor(MockOrderManager())
        
        # Verify optimized participation rate
        assert vwap.participation_rate == 0.15, f"Expected 0.15, got {vwap.participation_rate}"
        assert 0.05 <= vwap.participation_rate <= 0.30, "VWAP participation rate outside safe bounds"
        
    except ImportError as e:
        # Skip test if dependencies not available
        pytest.skip(f"Execution algorithm test skipped due to import error: {e}")


def test_constants_backward_compatibility():
    """Test that TRADING_CONSTANTS dictionary maintains backward compatibility."""
    from ai_trading.core.constants import TRADING_CONSTANTS
    
    # Verify all expected sections exist
    required_sections = [
        "MARKET_HOURS", "RISK_PARAMETERS", "KELLY_PARAMETERS", 
        "EXECUTION_PARAMETERS", "DATA_PARAMETERS", "DATABASE_PARAMETERS", 
        "PERFORMANCE_THRESHOLDS", "SYSTEM_LIMITS"
    ]
    
    for section in required_sections:
        assert section in TRADING_CONSTANTS, f"Missing required section: {section}"
        assert isinstance(TRADING_CONSTANTS[section], dict), f"Section {section} should be a dictionary"


if __name__ == "__main__":
    # Run tests directly for validation
    pytest.main([__file__, "-v"])