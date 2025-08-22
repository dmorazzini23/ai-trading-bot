"""
Test for the specific fix: ai_trading/core/__init__.py

This test validates that the core module ImportError mentioned in the
problem statement has been resolved by adding the missing __init__.py file.
"""
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestCoreModuleInit:
    """Test that the core module __init__.py fix works correctly."""

    def test_core_module_import(self):
        """Test that ai_trading.core module can be imported."""
        # This should not raise ImportError now that __init__.py exists
        import ai_trading.core
        assert ai_trading.core is not None

    def test_core_exports_available(self):
        """Test that all required exports are available from core module."""
        from ai_trading import core

        # Test that all the exports mentioned in the problem statement exist
        required_exports = [
            'OrderSide', 'OrderType', 'OrderStatus', 'RiskLevel',
            'TimeFrame', 'AssetClass', 'TRADING_CONSTANTS'
        ]

        for export in required_exports:
            assert hasattr(core, export), f"Missing export: {export}"

    def test_specific_imports_from_core(self):
        """Test the exact imports that were failing in the problem statement."""
        # This is the exact import pattern that was failing before the fix
        from ai_trading.core import (
            TRADING_CONSTANTS,
            OrderSide,
            OrderStatus,
            OrderType,
            RiskLevel,
        )

        # Verify the imports work correctly
        assert OrderSide.BUY.value == "buy"
        assert OrderType.MARKET.value == "market"
        assert OrderStatus.PENDING.value == "pending"
        assert RiskLevel.CONSERVATIVE.value == "conservative"
        assert isinstance(TRADING_CONSTANTS, dict)
        assert "MARKET_HOURS" in TRADING_CONSTANTS

    def test_enum_functionality(self):
        """Test that imported enums work correctly."""
        from ai_trading.core import OrderSide, RiskLevel

        # Test enum string representation
        assert str(OrderSide.BUY) == "buy"
        assert str(OrderSide.SELL) == "sell"

        # Test enum properties
        assert RiskLevel.CONSERVATIVE.max_position_size == 0.02
        assert RiskLevel.MODERATE.max_position_size == 0.05
        assert RiskLevel.AGGRESSIVE.max_position_size == 0.10

    def test_trading_constants_structure(self):
        """Test that TRADING_CONSTANTS has expected structure."""
        from ai_trading.core import TRADING_CONSTANTS

        # Test required constant groups exist
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

    def test_all_exports_list(self):
        """Test that __all__ list contains expected exports."""
        from ai_trading import core

        expected_all = [
            "OrderSide", "OrderType", "OrderStatus", "RiskLevel",
            "TimeFrame", "AssetClass", "TRADING_CONSTANTS"
        ]

        assert hasattr(core, '__all__')
        for item in expected_all:
            assert item in core.__all__, f"Missing from __all__: {item}"
