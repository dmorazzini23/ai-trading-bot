"""
Tests for the institutional core module.

Tests core trading enums, constants, and basic functionality
of the institutional trading platform infrastructure.
"""

from datetime import time

from ai_trading.core.enums import (
    OrderSide, OrderType, OrderStatus, RiskLevel, 
    TimeFrame, AssetClass
)
from ai_trading.core.constants import (
    TRADING_CONSTANTS, RISK_PARAMETERS,
    KELLY_PARAMETERS, EXECUTION_PARAMETERS
)


class TestOrderEnums:
    """Test order-related enumerations."""
    
    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
        assert str(OrderSide.BUY) == "buy"
        assert str(OrderSide.SELL) == "sell"
    
    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
    
    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELED.value == "canceled"
        assert OrderStatus.REJECTED.value == "rejected"
    
    def test_order_status_terminal(self):
        """Test terminal status check."""
        assert OrderStatus.FILLED.is_terminal
        assert OrderStatus.CANCELED.is_terminal
        assert OrderStatus.REJECTED.is_terminal
        assert OrderStatus.EXPIRED.is_terminal
        assert not OrderStatus.PENDING.is_terminal
        assert not OrderStatus.PARTIALLY_FILLED.is_terminal


class TestRiskLevel:
    """Test risk level enumeration."""
    
    def test_risk_level_values(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.CONSERVATIVE.value == "conservative"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.AGGRESSIVE.value == "aggressive"
    
    def test_max_position_size(self):
        """Test maximum position size by risk level."""
        assert RiskLevel.CONSERVATIVE.max_position_size == 0.02
        assert RiskLevel.MODERATE.max_position_size == 0.05
        assert RiskLevel.AGGRESSIVE.max_position_size == 0.10
    
    def test_max_drawdown_threshold(self):
        """Test maximum drawdown threshold by risk level."""
        assert RiskLevel.CONSERVATIVE.max_drawdown_threshold == 0.05
        assert RiskLevel.MODERATE.max_drawdown_threshold == 0.10
        assert RiskLevel.AGGRESSIVE.max_drawdown_threshold == 0.15


class TestTimeFrame:
    """Test timeframe enumeration."""
    
    def test_timeframe_values(self):
        """Test TimeFrame enum values."""
        assert TimeFrame.MINUTE_1.value == "1m"
        assert TimeFrame.HOUR_1.value == "1h"
        assert TimeFrame.DAY_1.value == "1d"
    
    def test_timeframe_seconds(self):
        """Test timeframe conversion to seconds."""
        assert TimeFrame.MINUTE_1.seconds == 60
        assert TimeFrame.MINUTE_5.seconds == 300
        assert TimeFrame.HOUR_1.seconds == 3600
        assert TimeFrame.DAY_1.seconds == 86400


class TestAssetClass:
    """Test asset class enumeration."""
    
    def test_asset_class_values(self):
        """Test AssetClass enum values."""
        assert AssetClass.EQUITY.value == "equity"
        assert AssetClass.BOND.value == "bond"
        assert AssetClass.COMMODITY.value == "commodity"
        assert AssetClass.CURRENCY.value == "currency"
        assert AssetClass.CRYPTO.value == "crypto"


class TestTradingConstants:
    """Test trading constants configuration."""
    
    def test_market_hours_exist(self):
        """Test market hours are defined."""
        assert "MARKET_HOURS" in TRADING_CONSTANTS
        market_hours = TRADING_CONSTANTS["MARKET_HOURS"]
        
        assert "MARKET_OPEN" in market_hours
        assert "MARKET_CLOSE" in market_hours
        assert isinstance(market_hours["MARKET_OPEN"], time)
        assert isinstance(market_hours["MARKET_CLOSE"], time)
    
    def test_risk_parameters_exist(self):
        """Test risk parameters are defined."""
        assert "RISK_PARAMETERS" in TRADING_CONSTANTS
        risk_params = TRADING_CONSTANTS["RISK_PARAMETERS"]
        
        required_params = [
            "MAX_PORTFOLIO_RISK",
            "MAX_POSITION_SIZE", 
            "STOP_LOSS_MULTIPLIER",
            "TAKE_PROFIT_MULTIPLIER"
        ]
        
        for param in required_params:
            assert param in risk_params
            assert isinstance(risk_params[param], (int, float))
    
    def test_kelly_parameters_exist(self):
        """Test Kelly Criterion parameters are defined."""
        assert "KELLY_PARAMETERS" in TRADING_CONSTANTS
        kelly_params = TRADING_CONSTANTS["KELLY_PARAMETERS"]
        
        required_params = [
            "MIN_SAMPLE_SIZE",
            "MAX_KELLY_FRACTION",
            "CONFIDENCE_LEVEL",
            "LOOKBACK_PERIODS"
        ]
        
        for param in required_params:
            assert param in kelly_params
            assert isinstance(kelly_params[param], (int, float))
    
    def test_execution_parameters_exist(self):
        """Test execution parameters are defined."""
        assert "EXECUTION_PARAMETERS" in TRADING_CONSTANTS
        exec_params = TRADING_CONSTANTS["EXECUTION_PARAMETERS"]
        
        required_params = [
            "MAX_SLIPPAGE_BPS",
            "PARTICIPATION_RATE",
            "ORDER_TIMEOUT_SECONDS",
            "RETRY_ATTEMPTS"
        ]
        
        for param in required_params:
            assert param in exec_params
            assert isinstance(exec_params[param], (int, float))
    
    def test_parameter_value_ranges(self):
        """Test parameter values are within reasonable ranges."""
        risk_params = RISK_PARAMETERS
        
        # Risk parameters should be percentages
        assert 0 < risk_params["MAX_PORTFOLIO_RISK"] <= 1
        assert 0 < risk_params["MAX_POSITION_SIZE"] <= 1
        
        # Multipliers should be positive
        assert risk_params["STOP_LOSS_MULTIPLIER"] > 0
        assert risk_params["TAKE_PROFIT_MULTIPLIER"] > 0
        
        # Kelly parameters
        kelly_params = KELLY_PARAMETERS
        assert kelly_params["MIN_SAMPLE_SIZE"] >= 10
        assert 0 < kelly_params["MAX_KELLY_FRACTION"] <= 1
        assert 0 < kelly_params["CONFIDENCE_LEVEL"] < 1
        
        # Execution parameters
        exec_params = EXECUTION_PARAMETERS
        assert exec_params["MAX_SLIPPAGE_BPS"] > 0
        assert 0 < exec_params["PARTICIPATION_RATE"] <= 1
        assert exec_params["ORDER_TIMEOUT_SECONDS"] > 0


class TestConstantsIntegration:
    """Test integration between different constant groups."""
    
    def test_all_constant_groups_present(self):
        """Test all major constant groups are present."""
        required_groups = [
            "MARKET_HOURS",
            "RISK_PARAMETERS", 
            "KELLY_PARAMETERS",
            "EXECUTION_PARAMETERS",
            "DATA_PARAMETERS",
            "PERFORMANCE_THRESHOLDS",
            "SYSTEM_LIMITS"
        ]
        
        for group in required_groups:
            assert group in TRADING_CONSTANTS
            assert isinstance(TRADING_CONSTANTS[group], dict)
    
    def test_constants_are_immutable_types(self):
        """Test constants use immutable types where appropriate."""
        # Market hours should use time objects
        market_hours = TRADING_CONSTANTS["MARKET_HOURS"]
        for hour_key, hour_value in market_hours.items():
            assert isinstance(hour_value, time)
        
        # Numeric parameters should be numbers
        risk_params = TRADING_CONSTANTS["RISK_PARAMETERS"]
        for param_key, param_value in risk_params.items():
            assert isinstance(param_value, (int, float))
    
    def test_constants_consistency(self):
        """Test consistency between related constants."""
        risk_params = RISK_PARAMETERS
        kelly_params = KELLY_PARAMETERS
        
        # Kelly max fraction should be reasonable relative to max position size
        assert kelly_params["MAX_KELLY_FRACTION"] >= risk_params["MAX_POSITION_SIZE"]
        
        # Performance thresholds should be reasonable
        perf_thresholds = TRADING_CONSTANTS["PERFORMANCE_THRESHOLDS"]
        assert perf_thresholds["MIN_SHARPE_RATIO"] > 0
        assert 0 < perf_thresholds["MAX_DRAWDOWN"] < 1
        assert 0 < perf_thresholds["MIN_WIN_RATE"] < 1