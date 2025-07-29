"""
Tests for the institutional Kelly Criterion implementation.

Tests Kelly Criterion calculation, position sizing optimization,
and risk-adjusted capital allocation functionality.
"""

import pytest
from unittest.mock import Mock, patch

from ai_trading.risk.kelly import KellyCriterion, KellyCalculator
from ai_trading.core.enums import RiskLevel


class TestKellyCriterion:
    """Test Kelly Criterion core functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.kelly = KellyCriterion(min_sample_size=10, max_fraction=0.25)
    
    def test_kelly_initialization(self):
        """Test Kelly Criterion initialization."""
        assert self.kelly.min_sample_size == 10
        assert self.kelly.max_fraction == 0.25
        assert self.kelly.confidence_level == 0.95
    
    def test_basic_kelly_calculation(self):
        """Test basic Kelly fraction calculation."""
        # Example: 60% win rate, average win 2%, average loss 1%
        kelly_fraction = self.kelly.calculate_kelly_fraction(
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=0.01
        )
        
        # Kelly formula: f = (bp - q) / b
        # b = 0.02/0.01 = 2, p = 0.6, q = 0.4
        # f = (2*0.6 - 0.4) / 2 = (1.2 - 0.4) / 2 = 0.4
        # But this gets capped at max_fraction = 0.25
        expected = 0.25  # Capped at max_fraction
        assert kelly_fraction == pytest.approx(expected, rel=1e-3)
    
    def test_kelly_with_max_fraction_cap(self):
        """Test Kelly fraction is capped at maximum."""
        # Very favorable scenario that would exceed max fraction
        kelly_fraction = self.kelly.calculate_kelly_fraction(
            win_rate=0.9,
            avg_win=0.05,
            avg_loss=0.01
        )
        
        # Should be capped at max_fraction
        assert kelly_fraction <= self.kelly.max_fraction
        assert kelly_fraction == self.kelly.max_fraction
    
    def test_kelly_negative_expectancy(self):
        """Test Kelly returns zero for negative expectancy."""
        # Losing strategy: 30% win rate, small wins, large losses
        kelly_fraction = self.kelly.calculate_kelly_fraction(
            win_rate=0.3,
            avg_win=0.01,
            avg_loss=0.03
        )
        
        # Should return 0 for negative expectancy
        assert kelly_fraction == 0.0
    
    def test_kelly_invalid_inputs(self):
        """Test Kelly handles invalid inputs gracefully."""
        # Invalid win rate
        assert self.kelly.calculate_kelly_fraction(1.5, 0.02, 0.01) == 0.0
        assert self.kelly.calculate_kelly_fraction(-0.1, 0.02, 0.01) == 0.0
        
        # Invalid win/loss values
        assert self.kelly.calculate_kelly_fraction(0.6, -0.02, 0.01) == 0.0
        assert self.kelly.calculate_kelly_fraction(0.6, 0.02, -0.01) == 0.0
        assert self.kelly.calculate_kelly_fraction(0.6, 0, 0.01) == 0.0
    
    def test_calculate_from_returns(self):
        """Test Kelly calculation from return series."""
        # Create realistic return series
        returns = [
            0.02, -0.01, 0.015, -0.008, 0.025,  # Mix of wins/losses
            -0.012, 0.018, 0.01, -0.005, 0.022,
            0.008, -0.015, 0.03, -0.01, 0.012,
            -0.008, 0.02, 0.005, -0.012, 0.028,
            0.015, -0.009, 0.018, 0.007, -0.011,
            0.025, -0.006, 0.013, 0.009, -0.014,
        ]
        
        kelly_fraction, stats = self.kelly.calculate_from_returns(returns)
        
        # Should return valid Kelly fraction and statistics
        assert 0 <= kelly_fraction <= self.kelly.max_fraction
        assert isinstance(stats, dict)
        
        # Check statistics
        assert "total_trades" in stats
        assert "winning_trades" in stats
        assert "losing_trades" in stats
        assert "win_rate" in stats
        assert "avg_win" in stats
        assert "avg_loss" in stats
        assert "kelly_fraction" in stats
        
        assert stats["total_trades"] == len(returns)
        assert stats["winning_trades"] + stats["losing_trades"] == len(returns)
        assert 0 <= stats["win_rate"] <= 1
    
    def test_insufficient_sample_size(self):
        """Test handling of insufficient sample size."""
        small_returns = [0.01, -0.005, 0.02]  # Only 3 returns, need 10
        
        kelly_fraction, stats = self.kelly.calculate_from_returns(small_returns)
        
        assert kelly_fraction == 0.0
        assert "error" in stats
        assert "Insufficient sample size" in stats["error"]
    
    def test_no_wins_or_losses(self):
        """Test handling of edge cases with no wins or losses."""
        # All wins
        all_wins = [0.01, 0.02, 0.015, 0.008, 0.025, 0.012, 0.018, 0.01, 0.005, 0.022]
        kelly_fraction, stats = self.kelly.calculate_from_returns(all_wins)
        assert kelly_fraction == 0.0
        assert "error" in stats
        
        # All losses
        all_losses = [-0.01, -0.02, -0.015, -0.008, -0.025, -0.012, -0.018, -0.01, -0.005, -0.022]
        kelly_fraction, stats = self.kelly.calculate_from_returns(all_losses)
        assert kelly_fraction == 0.0
        assert "error" in stats
    
    def test_fractional_kelly(self):
        """Test fractional Kelly calculation."""
        full_kelly = 0.20
        
        # 25% of full Kelly
        fractional = self.kelly.fractional_kelly(full_kelly, 0.25)
        assert fractional == 0.05
        
        # 50% of full Kelly
        fractional = self.kelly.fractional_kelly(full_kelly, 0.5)
        assert fractional == 0.10
    
    def test_kelly_with_confidence(self):
        """Test Kelly calculation with confidence intervals."""
        returns = [
            0.02, -0.01, 0.015, -0.008, 0.025, -0.012, 0.018, 0.01, -0.005, 0.022,
            0.008, -0.015, 0.03, -0.01, 0.012, -0.008, 0.02, 0.005, -0.012, 0.028,
            0.015, -0.009, 0.018, 0.007, -0.011, 0.025, -0.006, 0.013, 0.009, -0.014,
        ]
        
        adjusted_kelly, confidence_interval = self.kelly.kelly_with_confidence(returns)
        
        # Should return valid values
        assert 0 <= adjusted_kelly <= self.kelly.max_fraction
        assert confidence_interval >= 0
        
        # Adjusted Kelly should be lower than or equal to base Kelly due to uncertainty
        base_kelly, _ = self.kelly.calculate_from_returns(returns)
        assert adjusted_kelly <= base_kelly


class TestKellyCalculator:
    """Test advanced Kelly Calculator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = KellyCalculator()
    
    def test_kelly_calculator_initialization(self):
        """Test Kelly Calculator initialization."""
        assert self.calculator.kelly_criterion is not None
        assert self.calculator.lookback_periods == 252
        assert self.calculator.rebalance_frequency == 21
        assert isinstance(self.calculator.calculation_history, list)
    
    def test_portfolio_kelly_calculation(self):
        """Test multi-asset Kelly calculation."""
        # Create return series for multiple assets
        asset_returns = {
            "AAPL": [0.02, -0.01, 0.015, -0.008, 0.025, -0.012, 0.018, 0.01, -0.005, 0.022,
                     0.008, -0.015, 0.03, -0.01, 0.012, -0.008, 0.02, 0.005, -0.012, 0.028,
                     0.015, -0.009, 0.018, 0.007, -0.011, 0.025, -0.006, 0.013, 0.009, -0.014],
            "MSFT": [0.018, -0.012, 0.022, -0.005, 0.028, -0.009, 0.015, 0.012, -0.008, 0.02,
                     0.01, -0.018, 0.025, -0.012, 0.015, -0.006, 0.018, 0.008, -0.01, 0.025,
                     0.012, -0.011, 0.02, 0.005, -0.013, 0.022, -0.008, 0.016, 0.007, -0.012],
        }
        
        portfolio_kelly = self.calculator.calculate_portfolio_kelly(asset_returns)
        
        # Should return Kelly fractions for each asset
        assert isinstance(portfolio_kelly, dict)
        assert "AAPL" in portfolio_kelly
        assert "MSFT" in portfolio_kelly
        
        # All Kelly fractions should be non-negative and within limits
        for symbol, kelly_fraction in portfolio_kelly.items():
            assert 0 <= kelly_fraction <= self.calculator.kelly_criterion.max_fraction
    
    def test_dynamic_kelly_adjustment(self):
        """Test dynamic Kelly adjustment based on market conditions."""
        base_kelly = 0.15
        
        # High volatility scenario
        high_vol_conditions = {"volatility": 0.35, "drawdown": 0.05, "regime": "normal"}
        adjusted_kelly = self.calculator.dynamic_kelly_adjustment(base_kelly, high_vol_conditions)
        assert adjusted_kelly < base_kelly  # Should reduce position size
        
        # Low volatility scenario
        low_vol_conditions = {"volatility": 0.08, "drawdown": 0.02, "regime": "trending"}
        adjusted_kelly = self.calculator.dynamic_kelly_adjustment(base_kelly, low_vol_conditions)
        assert adjusted_kelly >= base_kelly  # Should maintain or increase position size
        
        # Crisis scenario
        crisis_conditions = {"volatility": 0.25, "drawdown": 0.12, "regime": "crisis"}
        adjusted_kelly = self.calculator.dynamic_kelly_adjustment(base_kelly, crisis_conditions)
        assert adjusted_kelly < base_kelly * 0.5  # Should drastically reduce position size
    
    def test_kelly_with_correlation(self):
        """Test Kelly calculation with correlation adjustments."""
        asset_returns = {
            "AAPL": [0.02, -0.01, 0.015, -0.008, 0.025, -0.012, 0.018, 0.01, -0.005, 0.022,
                     0.008, -0.015, 0.03, -0.01, 0.012, -0.008, 0.02, 0.005, -0.012, 0.028,
                     0.015, -0.009, 0.018, 0.007, -0.011, 0.025, -0.006, 0.013, 0.009, -0.014],
            "MSFT": [0.018, -0.012, 0.022, -0.005, 0.028, -0.009, 0.015, 0.012, -0.008, 0.02,
                     0.01, -0.018, 0.025, -0.012, 0.015, -0.006, 0.018, 0.008, -0.01, 0.025,
                     0.012, -0.011, 0.02, 0.005, -0.013, 0.022, -0.008, 0.016, 0.007, -0.012],
        }
        
        # High correlation between AAPL and MSFT
        correlation_matrix = {
            "AAPL_MSFT": 0.8,
            "MSFT_AAPL": 0.8,
        }
        
        # Calculate Kelly with and without correlation
        individual_kelly = self.calculator.calculate_portfolio_kelly(asset_returns)
        correlated_kelly = self.calculator.kelly_with_correlation(asset_returns, correlation_matrix)
        
        # Correlated Kelly should be lower due to correlation penalty
        assert correlated_kelly["AAPL"] <= individual_kelly["AAPL"]
        assert correlated_kelly["MSFT"] <= individual_kelly["MSFT"]
    
    def test_calculation_history_recording(self):
        """Test recording and retrieval of calculation history."""
        symbol = "AAPL"
        kelly_fraction = 0.15
        metadata = {"win_rate": 0.6, "avg_win": 0.02, "avg_loss": 0.01}
        
        # Record calculation
        self.calculator.record_calculation(symbol, kelly_fraction, metadata)
        
        # Check history
        history = self.calculator.get_calculation_history()
        assert len(history) == 1
        
        record = history[0]
        assert record["symbol"] == symbol
        assert record["kelly_fraction"] == kelly_fraction
        assert record["metadata"] == metadata
        assert "timestamp" in record
        
        # Test symbol-specific history
        symbol_history = self.calculator.get_calculation_history(symbol)
        assert len(symbol_history) == 1
        assert symbol_history[0]["symbol"] == symbol


class TestKellyIntegration:
    """Test Kelly Criterion integration with other components."""
    
    def test_kelly_with_risk_levels(self):
        """Test Kelly integration with different risk levels."""
        # Conservative risk level
        conservative_kelly = KellyCriterion(max_fraction=RiskLevel.CONSERVATIVE.max_position_size)
        
        # High expectancy scenario
        kelly_fraction = conservative_kelly.calculate_kelly_fraction(
            win_rate=0.7,
            avg_win=0.03,
            avg_loss=0.01
        )
        
        # Should be capped at conservative level
        assert kelly_fraction <= RiskLevel.CONSERVATIVE.max_position_size
        
        # Aggressive risk level
        aggressive_kelly = KellyCriterion(max_fraction=RiskLevel.AGGRESSIVE.max_position_size)
        
        kelly_fraction_aggressive = aggressive_kelly.calculate_kelly_fraction(
            win_rate=0.7,
            avg_win=0.03,
            avg_loss=0.01
        )
        
        # Should allow higher fraction
        assert kelly_fraction_aggressive >= kelly_fraction
    
    @patch('ai_trading.risk.kelly.logger')
    def test_kelly_logging(self, mock_logger):
        """Test Kelly Criterion logging functionality."""
        kelly = KellyCriterion()
        
        # Test successful calculation logging
        kelly.calculate_kelly_fraction(0.6, 0.02, 0.01)
        
        # Check that debug logging occurred
        mock_logger.debug.assert_called()
        
        # Test error logging with invalid inputs
        kelly.calculate_kelly_fraction(-1, 0.02, 0.01)
        
        # Check that warning logging occurred
        mock_logger.warning.assert_called()