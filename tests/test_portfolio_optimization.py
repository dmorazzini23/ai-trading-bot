"""
Test portfolio optimization modules for churn reduction strategy.
Validates core portfolio-level decision making and transaction cost analysis.
"""

import os
from typing import Any, cast

import pytest

# Set testing environment
os.environ['TESTING'] = '1'

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
            cast(dict[str, list[float]], self.sample_market_data['returns']),
            cast(dict[str, float], self.sample_market_data['prices']),
        )

        assert 0.0 <= efficiency <= 1.0
        assert efficiency > 0  # Should have some efficiency with valid data

    def test_portfolio_returns_preserve_short_direction(self):
        """Short positions should contribute inverse returns, not long returns."""
        returns = self.optimizer._calculate_portfolio_returns(
            {'LONG': 1.0, 'SHORT': -1.0},
            {'LONG': [0.01, -0.02], 'SHORT': [0.01, -0.02]},
            {'LONG': 100.0, 'SHORT': 100.0},
        )

        assert returns == pytest.approx([0.0, 0.0])

    def test_correlation_impact_calculation(self):
        """Test correlation impact calculation."""
        impact = self.optimizer.calculate_correlation_impact(
            'AAPL',
            self.sample_positions,
            cast(dict[str, dict[str, float]], self.sample_market_data['correlations']),
        )

        assert 0.0 <= impact <= 1.0

        # Test with highly correlated new symbol
        high_corr_data = {
            'NVDA': {'AAPL': 0.9, 'MSFT': 0.85, 'GOOGL': 0.8}
        }
        cast(dict[str, dict[str, float]], self.sample_market_data['correlations']).update(
            cast(dict[str, dict[str, float]], high_corr_data)
        )

        high_impact = self.optimizer.calculate_correlation_impact(
            'NVDA',
            self.sample_positions,
            cast(dict[str, dict[str, float]], self.sample_market_data['correlations']),
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

    def test_expected_return_change_uses_return_units(self, monkeypatch):
        """Expected return impact should be comparable with cost ratios."""
        monkeypatch.setattr(
            "ai_trading.execution.slippage_log.get_ewma_cost_bps",
            lambda _symbol, default=2.0: default,
        )
        expected = self.optimizer._estimate_return_change(
            'AAPL',
            20.0,
            {
                'prices': {'AAPL': 150.0},
                'returns': {'AAPL': [0.01] * 10},
            },
        )

        assert expected == pytest.approx(0.01)
        assert self.optimizer._estimate_transaction_cost(
            'AAPL',
            20.0,
            {'AAPL': 150.0},
        ) == pytest.approx(0.0006)

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

    def test_portfolio_decision_defers_without_price_evidence(self):
        """Missing price evidence should defer before weak cost/return defaults can approve."""
        decision, reasoning = self.optimizer.make_portfolio_decision(
            'AAPL',
            120.0,
            self.sample_positions,
            {
                'prices': {},
                'returns': self.sample_market_data['returns'],
                'correlations': self.sample_market_data['correlations'],
            },
        )

        assert decision is PortfolioDecision.DEFER
        assert 'Low confidence' in reasoning
        assert self.optimizer._estimate_transaction_cost('AAPL', 20.0, {}) == 1.0

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
            cast(dict[str, float], self.sample_market_data['prices']),
        )

        assert isinstance(should_rebalance, bool)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_rebalance_trigger_uses_signed_current_weights(self):
        """A matching short target should not drift as if it were long."""
        should_rebalance, reasoning = self.optimizer.should_trigger_rebalance(
            {'LONG': 1.0, 'SHORT': -1.0},
            {'LONG': 0.5, 'SHORT': -0.5},
            {'LONG': 100.0, 'SHORT': 100.0},
        )

        assert should_rebalance is False
        assert 'Drift within tolerance' in reasoning




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




if __name__ == '__main__':
    pytest.main([__file__, '-v'])
