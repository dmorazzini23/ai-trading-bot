"""
Tests for institutional-grade enhancements to the AI trading bot.

Tests adaptive position sizing, pre-trade validation, market microstructure
analysis, and tax-aware rebalancing functionality.
"""

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock


# Test adaptive position sizing
class TestAdaptivePositionSizing(unittest.TestCase):
    """Test adaptive position sizing with market condition awareness."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from ai_trading.core.enums import RiskLevel
            from ai_trading.risk.adaptive_sizing import (
                AdaptivePositionSizer,
                MarketConditionAnalyzer,
                MarketRegime,
            )

            self.MarketConditionAnalyzer = MarketConditionAnalyzer
            self.AdaptivePositionSizer = AdaptivePositionSizer
            self.MarketRegime = MarketRegime
            self.RiskLevel = RiskLevel
            self.imports_available = True
        except ImportError:
            self.imports_available = False
            self.skipTest("Adaptive sizing modules not available")

    def test_market_condition_analyzer_initialization(self):
        """Test market condition analyzer initialization."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        analyzer = self.MarketConditionAnalyzer(lookback_days=30)
        self.assertEqual(analyzer.lookback_days, 30)
        self.assertEqual(analyzer.volatility_window, 20)

    def test_market_regime_classification(self):
        """Test market regime classification."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        analyzer = self.MarketConditionAnalyzer()

        # Test with mock price data
        price_data: dict[str, list[float]] = {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0] * 10
        }

        regime = analyzer.analyze_market_regime(price_data)
        self.assertIsInstance(regime, self.MarketRegime)

    def test_adaptive_position_sizer_initialization(self):
        """Test adaptive position sizer initialization."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        sizer = self.AdaptivePositionSizer(self.RiskLevel.MODERATE)
        self.assertEqual(sizer.risk_level, self.RiskLevel.MODERATE)
        self.assertIsNotNone(sizer.market_analyzer)

    def test_adaptive_position_calculation(self):
        """Test adaptive position size calculation."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        sizer = self.AdaptivePositionSizer()

        # Mock data
        symbol = "AAPL"
        account_equity = 100000
        entry_price = 150.0
        market_data = {
            "atr": 3.0,
            "volume": 50000000,
            "bid_price": 149.5,
            "ask_price": 150.5
        }
        portfolio_data = {
            "price_data": {"SPY": [100] * 60},
            "returns_data": {"AAPL": [0.01, -0.005, 0.02] * 20},
            "current_positions": {}
        }

        result = sizer.calculate_adaptive_position(
            symbol, account_equity, entry_price, market_data, portfolio_data
        )

        self.assertIn("symbol", result)
        self.assertIn("recommended_size", result)
        self.assertIn("market_adjustments", result)
        self.assertEqual(result["symbol"], symbol)
        self.assertGreaterEqual(result["recommended_size"], 0)


class TestTaxAwareRebalancing(unittest.TestCase):
    """Test tax-aware rebalancing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from ai_trading.rebalancer import TaxAwareRebalancer
            self.TaxAwareRebalancer = TaxAwareRebalancer
            self.imports_available = True
        except ImportError:
            self.imports_available = False
            self.skipTest("Tax-aware rebalancer not available")

    def test_tax_rebalancer_initialization(self):
        """Test tax-aware rebalancer initialization."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        rebalancer = self.TaxAwareRebalancer(tax_rate_short=0.35, tax_rate_long=0.15)
        self.assertEqual(rebalancer.tax_rate_short, 0.35)
        self.assertEqual(rebalancer.tax_rate_long, 0.15)

    def test_tax_impact_calculation(self):
        """Test tax impact calculation."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        rebalancer = self.TaxAwareRebalancer()

        position = {
            "entry_price": 100.0,
            "quantity": 100,
            "entry_date": datetime.now(UTC) - timedelta(days=200)
        }

        current_price = 120.0

        tax_impact = rebalancer.calculate_tax_impact(position, current_price)

        self.assertIn("total_gain_loss", tax_impact)
        self.assertIn("tax_liability", tax_impact)
        self.assertIn("is_long_term", tax_impact)
        self.assertEqual(tax_impact["total_gain_loss"], 2000.0)  # $20 * 100 shares

    def test_loss_harvesting_identification(self):
        """Test loss harvesting opportunity identification."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        rebalancer = self.TaxAwareRebalancer()

        portfolio_positions = {
            "AAPL": {
                "entry_price": 150.0,
                "quantity": 100,
                "entry_date": datetime.now(UTC) - timedelta(days=100)
            },
            "MSFT": {
                "entry_price": 300.0,
                "quantity": 50,
                "entry_date": datetime.now(UTC) - timedelta(days=200)
            }
        }

        current_prices = {
            "AAPL": 140.0,  # Loss
            "MSFT": 320.0   # Gain
        }

        opportunities = rebalancer.identify_loss_harvesting_opportunities(
            portfolio_positions, current_prices
        )

        # Should identify AAPL as loss harvesting opportunity
        self.assertGreater(len(opportunities), 0)
        if opportunities:
            self.assertEqual(opportunities[0]["symbol"], "AAPL")
            self.assertLess(opportunities[0]["total_loss"], 0)


# Integration test for enhanced rebalancer


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
