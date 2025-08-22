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
        price_data = {
            "SPY": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 10
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


class TestPreTradeValidation(unittest.TestCase):
    """Test enhanced pre-trade validation system."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from ai_trading.core.enums import RiskLevel
            from ai_trading.risk.pre_trade_validation import (
                LiquidityValidator,
                MarketHoursValidator,
                PreTradeValidator,
                RiskValidator,
                ValidationStatus,
            )

            self.PreTradeValidator = PreTradeValidator
            self.ValidationStatus = ValidationStatus
            self.MarketHoursValidator = MarketHoursValidator
            self.LiquidityValidator = LiquidityValidator
            self.RiskValidator = RiskValidator
            self.RiskLevel = RiskLevel
            self.imports_available = True
        except ImportError:
            self.imports_available = False
            self.skipTest("Pre-trade validation modules not available")

    def test_market_hours_validator(self):
        """Test market hours validation."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        validator = self.MarketHoursValidator()

        # Test during market hours (2:30 PM UTC = 9:30 AM EST)
        market_time = datetime.now(UTC).replace(hour=15, minute=0)
        result = validator.validate_market_hours(market_time)

        self.assertIn(result.status, [self.ValidationStatus.APPROVED, self.ValidationStatus.WARNING])

    def test_liquidity_validator(self):
        """Test liquidity validation."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        validator = self.LiquidityValidator()

        # Test with adequate liquidity
        market_data = {
            "avg_volume": 10000000,
            "current_volume": 8000000,
            "bid_ask_spread": 0.01,
            "bid_size": 1000,
            "ask_size": 1200,
            "last_price": 100.0
        }

        result = validator.validate_liquidity("AAPL", 1000, market_data)
        self.assertIn(result.status, [self.ValidationStatus.APPROVED, self.ValidationStatus.WARNING])

    def test_risk_validator(self):
        """Test risk validation."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        validator = self.RiskValidator(self.RiskLevel.MODERATE)

        # Test position risk validation - use smaller position within limits
        current_positions = {}
        result = validator.validate_position_risk(
            "AAPL", 50, 150.0, 100000, current_positions  # 7.5% position size
        )

        # Position should be approved or have warnings, but not rejected for normal position
        self.assertIn(result.status, [self.ValidationStatus.APPROVED, self.ValidationStatus.WARNING])

    def test_comprehensive_validation(self):
        """Test comprehensive pre-trade validation."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        validator = self.PreTradeValidator()

        trade_request = {
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.0,
            "order_type": "limit"
        }

        market_data = {
            "avg_volume": 50000000,
            "current_volume": 40000000,
            "bid_ask_spread": 0.02,
            "bid_size": 500,
            "ask_size": 600,
            "last_price": 150.0
        }

        portfolio_data = {
            "account_equity": 100000,
            "current_positions": {},
            "correlations": {}
        }

        result = validator.validate_trade(trade_request, market_data, portfolio_data)

        self.assertEqual(result.symbol, "AAPL")
        self.assertIn(result.overall_status, [
            self.ValidationStatus.APPROVED,
            self.ValidationStatus.WARNING,
            self.ValidationStatus.REJECTED
        ])
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)


class TestMarketMicrostructure(unittest.TestCase):
    """Test market microstructure feature engineering."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from ai_trading.execution.microstructure import (
                BidAskSpreadAnalyzer,
                MarketMicrostructureEngine,
                MarketRegimeFeature,
                OrderFlowAnalyzer,
            )

            self.MarketMicrostructureEngine = MarketMicrostructureEngine
            self.BidAskSpreadAnalyzer = BidAskSpreadAnalyzer
            self.OrderFlowAnalyzer = OrderFlowAnalyzer
            self.MarketRegimeFeature = MarketRegimeFeature
            self.imports_available = True
        except ImportError:
            self.imports_available = False
            self.skipTest("Market microstructure modules not available")

    def test_spread_analyzer_initialization(self):
        """Test bid-ask spread analyzer initialization."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        analyzer = self.BidAskSpreadAnalyzer()
        self.assertEqual(analyzer.lookback_periods, 100)
        self.assertEqual(analyzer.min_spread_bps, 1.0)

    def test_spread_feature_analysis(self):
        """Test spread feature analysis."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        analyzer = self.BidAskSpreadAnalyzer()

        market_data = {
            "bid_price": 99.95,
            "ask_price": 100.05,
            "last_price": 100.00,
            "bid_size": 1000,
            "ask_size": 1200
        }

        trade_history = [
            {"price": 100.0, "size": 100, "side": "buy", "timestamp": datetime.now(UTC)},  # AI-AGENT-REF: Use timezone-aware datetime
            {"price": 100.02, "size": 200, "side": "sell", "timestamp": datetime.now(UTC)},  # AI-AGENT-REF: Use timezone-aware datetime
        ]

        features = analyzer.analyze_spread_features(market_data, trade_history)

        self.assertIn("bid_ask_spread", features)
        self.assertIn("spread_bps", features)
        self.assertGreater(features["bid_ask_spread"], 0)

    def test_order_flow_analyzer(self):
        """Test order flow analysis."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        analyzer = self.OrderFlowAnalyzer()

        trade_data = [
            {"price": 100.0, "size": 100, "side": "buy", "timestamp": datetime.now(UTC)},  # AI-AGENT-REF: Use timezone-aware datetime
            {"price": 100.02, "size": 200, "side": "sell", "timestamp": datetime.now(UTC)},  # AI-AGENT-REF: Use timezone-aware datetime
        ]

        quote_data = [
            {"bid_price": 99.98, "ask_price": 100.02, "bid_size": 500, "ask_size": 600},
            {"bid_price": 99.99, "ask_price": 100.03, "bid_size": 400, "ask_size": 700},
        ]

        features = analyzer.analyze_order_flow(trade_data, quote_data)

        self.assertIn("trade_intensity", features)
        self.assertIn("order_imbalance", features)

    def test_microstructure_engine(self):
        """Test comprehensive microstructure analysis."""
        if not self.imports_available:
            self.skipTest("Dependencies not available")

        engine = self.MarketMicrostructureEngine()

        market_data = {
            "bid_price": 99.95,
            "ask_price": 100.05,
            "last_price": 100.00,
            "bid_size": 1000,
            "ask_size": 1200,
            "volume": 50000
        }

        trade_history = []
        quote_history = []

        result = engine.analyze_market_microstructure(
            "AAPL", market_data, trade_history, quote_history
        )

        self.assertEqual(result.symbol, "AAPL")
        self.assertGreaterEqual(result.bid_ask_spread, 0)


class TestTaxAwareRebalancing(unittest.TestCase):
    """Test tax-aware rebalancing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            import os
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
class TestEnhancedRebalancer(unittest.TestCase):
    """Test enhanced rebalancer integration."""

    def test_enhanced_rebalancer_fallback(self):
        """Test that enhanced rebalancer falls back gracefully."""
        import os
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))

        from ai_trading.rebalancer import enhanced_maybe_rebalance, rebalance_portfolio

        # Mock context
        ctx = Mock()
        ctx.portfolio_weights = {}

        # Should not raise exception
        try:
            rebalance_portfolio(ctx)
            enhanced_maybe_rebalance(ctx)
        except (ValueError, TypeError) as e:
            self.fail(f"Enhanced rebalancer raised exception: {e}")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
