"""
Test portfolio optimization modules for churn reduction strategy.
Validates core portfolio-level decision making and transaction cost analysis.
"""

import os

import pytest

# Set testing environment
os.environ['TESTING'] = '1'

from ai_trading.execution.transaction_costs import (  # AI-AGENT-REF: normalized import
    TradeType,
    create_transaction_cost_calculator,
)
from ai_trading.portfolio.optimizer import (  # AI-AGENT-REF: normalized import
    PortfolioDecision,
    PortfolioOptimizer,
    create_portfolio_optimizer,
)
from ai_trading.strategies.regime_detector import MarketRegime, create_regime_detector


class TestPortfolioOptimizer:
    """Test portfolio-level optimization and decision making."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = create_portfolio_optimizer()
        self.sample_positions = {
            'AAPL': 100.0,
            'MSFT': 80.0,
            'GOOGL': 60.0
        }
        self.sample_market_data = {
            'prices': {
                'AAPL': 150.0,
                'MSFT': 300.0,
                'GOOGL': 2500.0
            },
            'returns': {
                'AAPL': [0.01, -0.02, 0.015, 0.005, -0.01] * 10,  # 50 returns
                'MSFT': [0.005, 0.01, -0.01, 0.02, 0.005] * 10,
                'GOOGL': [0.02, -0.015, 0.01, -0.005, 0.01] * 10
            },
            'correlations': {
                'AAPL': {'MSFT': 0.6, 'GOOGL': 0.4},
                'MSFT': {'AAPL': 0.6, 'GOOGL': 0.5},
                'GOOGL': {'AAPL': 0.4, 'MSFT': 0.5}
            }
        }

    def test_portfolio_kelly_efficiency_calculation(self):
        """Test portfolio Kelly efficiency calculation."""
        efficiency = self.optimizer.calculate_portfolio_kelly_efficiency(
            self.sample_positions,
            self.sample_market_data['returns'],
            self.sample_market_data['prices']
        )

        assert 0.0 <= efficiency <= 1.0
        assert efficiency > 0  # Should have some efficiency with valid data

    def test_correlation_impact_calculation(self):
        """Test correlation impact calculation."""
        impact = self.optimizer.calculate_correlation_impact(
            'AAPL',
            self.sample_positions,
            self.sample_market_data['correlations']
        )

        assert 0.0 <= impact <= 1.0

        # Test with highly correlated new symbol
        high_corr_data = {
            'NVDA': {'AAPL': 0.9, 'MSFT': 0.85, 'GOOGL': 0.8}
        }
        self.sample_market_data['correlations'].update(high_corr_data)

        high_impact = self.optimizer.calculate_correlation_impact(
            'NVDA',
            self.sample_positions,
            self.sample_market_data['correlations']
        )

        assert high_impact > impact  # Higher correlation should mean higher impact

    def test_trade_impact_evaluation(self):
        """Test comprehensive trade impact evaluation."""
        impact_analysis = self.optimizer.evaluate_trade_impact(
            'AAPL',
            120.0,  # Increase position from 100 to 120
            self.sample_positions,
            self.sample_market_data
        )

        # Validate analysis structure
        assert hasattr(impact_analysis, 'expected_return_change')
        assert hasattr(impact_analysis, 'kelly_efficiency_change')
        assert hasattr(impact_analysis, 'transaction_cost')
        assert hasattr(impact_analysis, 'net_benefit')
        assert hasattr(impact_analysis, 'confidence')

        # Validate ranges
        assert 0.0 <= impact_analysis.confidence <= 1.0
        assert impact_analysis.transaction_cost >= 0.0

    def test_portfolio_decision_making(self):
        """Test portfolio-level trade decision making."""
        decision, reasoning = self.optimizer.make_portfolio_decision(
            'AAPL',
            120.0,  # Small position increase
            self.sample_positions,
            self.sample_market_data
        )

        assert isinstance(decision, PortfolioDecision)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

        # Test with large position change (should likely be rejected)
        decision_large, reasoning_large = self.optimizer.make_portfolio_decision(
            'AAPL',
            1000.0,  # Very large position increase
            self.sample_positions,
            self.sample_market_data
        )

        # Large changes are more likely to be rejected or deferred
        assert decision_large in [PortfolioDecision.REJECT, PortfolioDecision.DEFER]

    def test_rebalance_trigger_logic(self):
        """Test rebalancing trigger logic."""
        target_weights = {
            'AAPL': 0.4,
            'MSFT': 0.35,
            'GOOGL': 0.25
        }

        should_rebalance, reasoning = self.optimizer.should_trigger_rebalance(
            self.sample_positions,
            target_weights,
            self.sample_market_data['prices']
        )

        assert isinstance(should_rebalance, bool)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0


class TestTransactionCostCalculator:
    """Test transaction cost calculation and profitability validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = create_transaction_cost_calculator()
        self.sample_market_data = {
            'prices': {'AAPL': 150.0},
            'quotes': {
                'AAPL': {'bid': 149.5, 'ask': 150.5}
            },
            'volumes': {'AAPL': 50000000},  # 50M average volume
            'volatility': {'AAPL': 0.025}   # 2.5% daily volatility
        }

    def test_spread_cost_calculation(self):
        """Test bid-ask spread cost calculation."""
        spread_cost = self.calculator.calculate_spread_cost(
            'AAPL',
            100,  # 100 shares
            self.sample_market_data
        )

        assert spread_cost > 0
        assert spread_cost < 100  # Should be reasonable relative to trade size

        # Test with missing quote data (should fallback gracefully)
        no_quote_data = {'prices': {'AAPL': 150.0}}
        fallback_cost = self.calculator.calculate_spread_cost(
            'AAPL',
            100,
            no_quote_data
        )

        assert fallback_cost > 0

    def test_commission_calculation(self):
        """Test commission calculation."""
        trade_value = 15000.0  # 100 shares * $150
        commission = self.calculator.calculate_commission(
            'AAPL',
            100,
            trade_value
        )

        assert commission >= 0
        assert commission <= self.calculator.max_commission

    def test_market_impact_calculation(self):
        """Test market impact modeling."""
        temp_impact, perm_impact = self.calculator.calculate_market_impact(
            'AAPL',
            10000,  # Large trade
            self.sample_market_data
        )

        assert temp_impact >= 0
        assert perm_impact >= 0
        assert temp_impact >= perm_impact  # Temporary impact should be larger

        # Test smaller trade should have lower impact
        temp_small, perm_small = self.calculator.calculate_market_impact(
            'AAPL',
            100,  # Small trade
            self.sample_market_data
        )

        assert temp_small < temp_impact
        assert perm_small < perm_impact

    def test_total_transaction_cost(self):
        """Test comprehensive transaction cost calculation."""
        cost_breakdown = self.calculator.calculate_total_transaction_cost(
            'AAPL',
            100,
            TradeType.LIMIT_ORDER,
            self.sample_market_data
        )

        # Validate structure
        assert hasattr(cost_breakdown, 'spread_cost')
        assert hasattr(cost_breakdown, 'commission')
        assert hasattr(cost_breakdown, 'market_impact')
        assert hasattr(cost_breakdown, 'total_cost')
        assert hasattr(cost_breakdown, 'cost_percentage')

        # Validate values
        assert cost_breakdown.total_cost > 0
        assert cost_breakdown.total_cost == (
            cost_breakdown.spread_cost +
            cost_breakdown.commission +
            cost_breakdown.market_impact +
            cost_breakdown.opportunity_cost +
            cost_breakdown.borrowing_cost
        )
        assert 0.0 <= cost_breakdown.cost_percentage <= 1.0

    def test_profitability_validation(self):
        """Test trade profitability validation."""
        # Test profitable trade
        profitable_analysis = self.calculator.validate_trade_profitability(
            'AAPL',
            100,
            500.0,  # $500 expected profit
            self.sample_market_data
        )

        assert hasattr(profitable_analysis, 'is_profitable')
        assert hasattr(profitable_analysis, 'net_expected_profit')
        assert hasattr(profitable_analysis, 'safety_margin')

        # Test unprofitable trade
        unprofitable_analysis = self.calculator.validate_trade_profitability(
            'AAPL',
            100,
            10.0,  # Only $10 expected profit (likely insufficient)
            self.sample_market_data
        )

        assert not unprofitable_analysis.is_profitable


class TestRegimeDetector:
    """Test market regime detection and dynamic threshold calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = create_regime_detector()
        self.sample_market_data = {
            'prices': {'SPY': 400.0},
            'returns': {
                'SPY': [0.01, -0.005, 0.015, -0.01, 0.005] * 20  # 100 returns
            },
            'volumes': {'SPY': 100000000},
            'correlations': {
                'SPY': {'SPY': 1.0}
            }
        }

    def test_regime_detection(self):
        """Test market regime detection."""
        regime, metrics = self.detector.detect_current_regime(
            self.sample_market_data,
            'SPY'
        )

        assert isinstance(regime, MarketRegime)
        assert hasattr(metrics, 'trend_strength')
        assert hasattr(metrics, 'volatility_level')
        assert hasattr(metrics, 'regime_confidence')

        # Validate metric ranges
        assert 0.0 <= metrics.trend_strength <= 1.0
        assert 0.0 <= metrics.volatility_level <= 1.0
        assert 0.0 <= metrics.regime_confidence <= 1.0

    def test_dynamic_threshold_calculation(self):
        """Test dynamic threshold calculation based on regime."""
        regime, metrics = self.detector.detect_current_regime(self.sample_market_data)

        thresholds = self.detector.calculate_dynamic_thresholds(regime, metrics)

        assert hasattr(thresholds, 'rebalance_drift_threshold')
        assert hasattr(thresholds, 'trade_frequency_multiplier')
        assert hasattr(thresholds, 'minimum_improvement_threshold')

        # Validate positive values
        assert thresholds.rebalance_drift_threshold > 0
        assert thresholds.trade_frequency_multiplier > 0
        assert thresholds.minimum_improvement_threshold > 0

    def test_regime_specific_adjustments(self):
        """Test that different regimes produce different threshold adjustments."""
        # Test normal regime
        normal_thresholds = self.detector.calculate_dynamic_thresholds(
            MarketRegime.NORMAL,
            self.detector._fallback_regime_detection()[1]
        )

        # Test crisis regime
        crisis_thresholds = self.detector.calculate_dynamic_thresholds(
            MarketRegime.CRISIS,
            self.detector._fallback_regime_detection()[1]
        )

        # Crisis should have more conservative thresholds
        assert crisis_thresholds.trade_frequency_multiplier < normal_thresholds.trade_frequency_multiplier
        assert crisis_thresholds.minimum_improvement_threshold > normal_thresholds.minimum_improvement_threshold
        assert crisis_thresholds.safety_margin_multiplier > normal_thresholds.safety_margin_multiplier


class TestIntegration:
    """Test integration between portfolio optimization components."""

    def setup_method(self):
        """Set up integrated test fixtures."""
        self.optimizer = create_portfolio_optimizer()
        self.cost_calculator = create_transaction_cost_calculator()
        self.regime_detector = create_regime_detector()

        self.market_data = {
            'prices': {
                'AAPL': 150.0,
                'MSFT': 300.0,
                'SPY': 400.0
            },
            'returns': {
                'AAPL': [0.01, -0.02, 0.015] * 30,
                'MSFT': [0.005, 0.01, -0.01] * 30,
                'SPY': [0.01, -0.005, 0.015] * 30
            },
            'correlations': {
                'AAPL': {'MSFT': 0.6},
                'MSFT': {'AAPL': 0.6}
            },
            'volumes': {
                'AAPL': 50000000,
                'MSFT': 30000000,
                'SPY': 100000000
            }
        }

        self.current_positions = {
            'AAPL': 100.0,
            'MSFT': 80.0
        }

    def test_integrated_portfolio_decision_workflow(self):
        """Test complete portfolio decision workflow with all components."""
        # 1. Detect market regime
        regime, metrics = self.regime_detector.detect_current_regime(
            self.market_data,
            'SPY'
        )

        # 2. Calculate dynamic thresholds
        thresholds = self.regime_detector.calculate_dynamic_thresholds(regime, metrics)

        # 3. Update optimizer with dynamic thresholds
        optimizer = PortfolioOptimizer(
            improvement_threshold=thresholds.minimum_improvement_threshold,
            rebalance_drift_threshold=thresholds.rebalance_drift_threshold
        )

        # 4. Make portfolio decision
        decision, reasoning = optimizer.make_portfolio_decision(
            'AAPL',
            120.0,  # Increase position
            self.current_positions,
            self.market_data
        )

        # 5. Validate transaction costs if trade is approved
        if decision == PortfolioDecision.APPROVE:
            profitability = self.cost_calculator.validate_trade_profitability(
                'AAPL',
                20.0,  # Position change
                100.0,  # Expected profit
                self.market_data
            )

            assert isinstance(profitability.is_profitable, bool)

        assert isinstance(decision, PortfolioDecision)
        assert len(reasoning) > 0

    def test_churn_reduction_validation(self):
        """Test that the system actually reduces trading frequency."""
        # Create multiple trade proposals
        trade_proposals = [
            ('AAPL', 105.0),  # Small change
            ('AAPL', 110.0),  # Medium change
            ('MSFT', 85.0),   # Small change
            ('MSFT', 90.0),   # Medium change
        ]

        approved_trades = 0
        rejected_trades = 0
        deferred_trades = 0

        for symbol, new_position in trade_proposals:
            decision, _ = self.optimizer.make_portfolio_decision(
                symbol,
                new_position,
                self.current_positions,
                self.market_data
            )

            if decision == PortfolioDecision.APPROVE:
                approved_trades += 1
            elif decision == PortfolioDecision.REJECT:
                rejected_trades += 1
            elif decision == PortfolioDecision.DEFER:
                deferred_trades += 1

        # System should reject/defer more trades than approve (churn reduction)
        assert (rejected_trades + deferred_trades) >= approved_trades


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
