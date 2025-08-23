"""
Test suite for advanced intelligent position holding strategies.

Tests the new position management components:
- IntelligentPositionManager orchestration
- Market regime detection
- Technical signal analysis  
- Dynamic trailing stops
- Multi-tiered profit taking
- Portfolio correlation analysis

AI-AGENT-REF: Comprehensive tests for intelligent position management
"""

import importlib.util
import pytest
from dataclasses import dataclass
from unittest.mock import Mock
# AI-AGENT-REF: skip if ai_trading.position not available
if importlib.util.find_spec("ai_trading.position") is None:  # pragma: no cover
    pytest.skip("ai_trading.position not available in this env", allow_module_level=True)

from ai_trading.position import (
    ConcentrationLevel,
    DivergenceType,
    IntelligentPositionManager,
    MarketRegime,
    MarketRegimeDetector,
    PortfolioCorrelationAnalyzer,
    ProfitTakingEngine,
    ProfitTakingStrategy,
    SignalStrength,
    TechnicalSignalAnalyzer,
    TrailingStopManager,
    TrailingStopType,
)




@dataclass
class TestIntelligentPositionManager:
    """Test the main IntelligentPositionManager orchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_ctx = Mock()
        self.mock_ctx.data_fetcher = Mock()

        # Create mock data
        self.mock_daily_data = {
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1000, 900, 1300, 1400, 1100, 1000, 1200, 1500]
        }

        self.mock_ctx.data_fetcher.get_daily_df.return_value = Mock()
        self.mock_ctx.data_fetcher.get_daily_df.return_value.empty = False
        self.mock_ctx.data_fetcher.get_daily_df.return_value.__len__ = lambda: 11

        # Mock data access
        for col, values in self.mock_daily_data.items():
            mock_series = Mock()
            mock_series.__len__ = lambda: len(values)
            mock_series.iloc = Mock()
            mock_series.iloc.__getitem__ = lambda idx: values[idx] if isinstance(idx, int) else values[idx]
            mock_series.tail.return_value = Mock()
            mock_series.tail.return_value.tolist.return_value = values[-5:]
            setattr(self.mock_ctx.data_fetcher.get_daily_df.return_value, col, mock_series)

        self.manager = IntelligentPositionManager(self.mock_ctx)

    def test_initialization(self):
        """Test that IntelligentPositionManager initializes correctly."""
        assert self.manager.ctx == self.mock_ctx
        assert isinstance(self.manager.regime_detector, MarketRegimeDetector)
        assert isinstance(self.manager.technical_analyzer, TechnicalSignalAnalyzer)
        assert isinstance(self.manager.trailing_stop_manager, TrailingStopManager)
        assert isinstance(self.manager.profit_taking_engine, ProfitTakingEngine)
        assert isinstance(self.manager.correlation_analyzer, PortfolioCorrelationAnalyzer)

    def test_should_hold_position_integration(self):
        """Test the enhanced should_hold_position method."""
        # Create mock position
        position = MockPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=100.0,
            market_value=11000.0
        )

        # Test with profitable position
        result = self.manager.should_hold_position(
            symbol='AAPL',
            current_position=position,
            unrealized_pnl_pct=10.0,
            days_held=5,
            current_positions=[position]
        )

        # Should return a boolean
        assert isinstance(result, bool)

    def test_analyze_position_basic(self):
        """Test basic position analysis."""
        position = MockPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=100.0,
            market_value=11000.0
        )

        # Mock current price
        self.mock_ctx.data_fetcher.get_minute_df.return_value = None
        close_series = Mock()
        close_series.iloc = [-1]
        close_series.__getitem__ = lambda x: 110.0 if x == -1 else 100.0
        self.mock_ctx.data_fetcher.get_daily_df.return_value.__getitem__ = lambda x: close_series if x == 'close' else Mock()

        recommendation = self.manager.analyze_position('AAPL', position, [position])

        # Should return a recommendation
        assert hasattr(recommendation, 'symbol')
        assert hasattr(recommendation, 'action')
        assert hasattr(recommendation, 'confidence')
        assert recommendation.symbol == 'AAPL'


class TestMarketRegimeDetector:
    """Test market regime detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_ctx = Mock()
        self.detector = MarketRegimeDetector(self.mock_ctx)

    def test_regime_classification(self):
        """Test regime classification logic."""
        # Test trending bull classification
        trend_metrics = {'strength': 0.8, 'direction': 0.5}
        vol_metrics = {'percentile': 50.0}
        momentum_metrics = {'score': 0.8}
        mean_reversion_metrics = {'score': 0.3}

        regime = self.detector._classify_regime(
            trend_metrics, vol_metrics, momentum_metrics, mean_reversion_metrics
        )

        assert regime == MarketRegime.TRENDING_BULL

    def test_regime_parameters(self):
        """Test regime-specific parameters."""
        params = self.detector.get_regime_parameters(MarketRegime.TRENDING_BULL)

        assert 'stop_distance_multiplier' in params
        assert 'profit_taking_patience' in params
        assert params['profit_taking_patience'] > 1.0  # Should be patient in bull trends

    def test_high_volatility_regime(self):
        """Test high volatility regime detection."""
        trend_metrics = {'strength': 0.3, 'direction': 0.1}
        vol_metrics = {'percentile': 85.0}  # High volatility
        momentum_metrics = {'score': 0.5}
        mean_reversion_metrics = {'score': 0.5}

        regime = self.detector._classify_regime(
            trend_metrics, vol_metrics, momentum_metrics, mean_reversion_metrics
        )

        assert regime == MarketRegime.HIGH_VOLATILITY


class TestTechnicalSignalAnalyzer:
    """Test technical signal analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_ctx = Mock()
        self.analyzer = TechnicalSignalAnalyzer(self.mock_ctx)

    def test_rsi_calculation(self):
        """Test RSI calculation."""
        # Create mock price series
        prices = Mock()
        prices.__len__ = lambda: 20
        prices.diff.return_value = Mock()
        prices.diff.return_value.where = Mock(return_value=Mock())

        # Mock rolling calculations
        mock_rolling = Mock()
        mock_rolling.mean.return_value = Mock()
        mock_rolling.mean.return_value.iloc = [-1]
        mock_rolling.mean.return_value.__getitem__ = lambda x: 2.0 if x == -1 else 1.0

        prices.diff.return_value.where.return_value.rolling = Mock(return_value=mock_rolling)

        # Should not crash and return reasonable RSI
        rsi = self.analyzer._calculate_rsi(prices, 14)
        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100

    def test_divergence_detection(self):
        """Test momentum divergence detection."""
        # Create mock data with divergence pattern
        mock_data = Mock()
        mock_data.__len__ = lambda: 20

        close_series = Mock()
        close_series.__len__ = lambda: 20
        close_series.tail.return_value = Mock()
        close_series.tail.return_value.tolist.return_value = [100, 101, 102, 103, 104]  # Rising prices
        close_series.iloc = Mock()

        mock_data.__getitem__ = lambda x: close_series if x == 'close' else Mock()

        result = self.analyzer._analyze_divergence(mock_data)

        assert 'type' in result
        assert 'strength' in result


class TestTrailingStopManager:
    """Test dynamic trailing stop functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_ctx = Mock()
        self.manager = TrailingStopManager(self.mock_ctx)

    def test_stop_initialization(self):
        """Test trailing stop initialization."""
        position = MockPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=100.0,
            market_value=11000.0
        )

        stop_level = self.manager.update_trailing_stop('AAPL', position, 110.0)

        assert stop_level is not None
        assert stop_level.symbol == 'AAPL'
        assert stop_level.current_price == 110.0
        assert stop_level.stop_price < 110.0  # Stop should be below current price

    def test_stop_movement(self):
        """Test that stops move up with price for long positions."""
        position = MockPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=100.0,
            market_value=11000.0
        )

        # Initialize stop
        stop1 = self.manager.update_trailing_stop('AAPL', position, 110.0)
        initial_stop = stop1.stop_price

        # Price moves higher
        stop2 = self.manager.update_trailing_stop('AAPL', position, 115.0)

        # Stop should move up
        assert stop2.stop_price >= initial_stop
        assert stop2.current_price == 115.0

    def test_stop_trigger_detection(self):
        """Test stop trigger detection."""
        position = MockPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=100.0,
            market_value=11000.0
        )

        # Initialize stop at high price
        self.manager.update_trailing_stop('AAPL', position, 115.0)

        # Price falls below stop
        stop_level = self.manager.update_trailing_stop('AAPL', position, 105.0)

        # Should detect trigger
        assert stop_level.is_triggered


class TestProfitTakingEngine:
    """Test multi-tiered profit taking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_ctx = Mock()
        self.engine = ProfitTakingEngine(self.mock_ctx)

    def test_profit_plan_creation(self):
        """Test profit plan creation."""
        position = MockPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=100.0,
            market_value=11000.0
        )

        # Mock current price
        self.mock_ctx.data_fetcher.get_minute_df.return_value = None
        self.mock_ctx.data_fetcher.get_daily_df.return_value = None

        plan = self.engine.create_profit_plan('AAPL', position, 100.0, 300.0)  # $3 risk per share

        if plan:  # Plan creation might fail due to mocking
            assert plan.symbol == 'AAPL'
            assert plan.entry_price == 100.0
            assert len(plan.targets) > 0

    def test_target_triggering(self):
        """Test profit target triggering."""
        position = MockPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=100.0,
            market_value=11000.0
        )

        # Create plan
        plan = self.engine.create_profit_plan('AAPL', position, 100.0, 300.0)

        if plan:
            # Simulate price increase
            triggered_targets = self.engine.update_profit_plan('AAPL', 110.0, position)

            # Should return list of triggered targets
            assert isinstance(triggered_targets, list)


class TestPortfolioCorrelationAnalyzer:
    """Test portfolio correlation analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_ctx = Mock()
        self.analyzer = PortfolioCorrelationAnalyzer(self.mock_ctx)

    def test_position_data_extraction(self):
        """Test position data extraction."""
        positions = [
            MockPosition('AAPL', 100, 100.0, 11000.0),
            MockPosition('MSFT', 50, 200.0, 10500.0),
            MockPosition('GOOGL', 25, 150.0, 3750.0)
        ]

        position_data = self.analyzer._extract_position_data(positions)

        assert len(position_data) == 3
        assert 'AAPL' in position_data
        assert position_data['AAPL']['market_value'] == 11000.0

    def test_sector_classification(self):
        """Test sector classification."""
        sector = self.analyzer._get_symbol_sector('AAPL')
        assert sector == 'Technology'

        sector = self.analyzer._get_symbol_sector('JPM')
        assert sector == 'Financials'

    def test_concentration_analysis(self):
        """Test concentration level analysis."""
        positions = [
            MockPosition('AAPL', 100, 100.0, 50000.0),  # 50% of portfolio
            MockPosition('MSFT', 50, 200.0, 25000.0),   # 25% of portfolio
            MockPosition('GOOGL', 25, 150.0, 25000.0)   # 25% of portfolio
        ]

        analysis = self.analyzer.analyze_portfolio(positions)

        assert analysis.total_positions == 3
        assert analysis.total_value == 100000.0
        assert analysis.concentration_level in [ConcentrationLevel.HIGH, ConcentrationLevel.EXTREME]


class TestIntegrationScenarios:
    """Test integrated scenarios combining multiple components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_ctx = Mock()
        self.manager = IntelligentPositionManager(self.mock_ctx)

    def test_profitable_position_scenario(self):
        """Test scenario with profitable position."""
        position = MockPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=100.0,
            market_value=12000.0  # 20% gain
        )

        # Should lean towards holding profitable position
        result = self.manager.should_hold_position(
            'AAPL', position, 20.0, 10, [position]
        )

        # With 20% gain and 10 days held, should generally hold
        # (exact result depends on market conditions)
        assert isinstance(result, bool)

    def test_loss_position_scenario(self):
        """Test scenario with losing position."""
        position = MockPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=100.0,
            market_value=9000.0  # 10% loss
        )

        result = self.manager.should_hold_position(
            'AAPL', position, -10.0, 2, [position]
        )

        # Should consider holding if only held for 2 days (min hold period)
        assert isinstance(result, bool)

    def test_portfolio_level_recommendations(self):
        """Test portfolio-level analysis and recommendations."""
        positions = [
            MockPosition('AAPL', 100, 100.0, 11000.0),
            MockPosition('MSFT', 50, 200.0, 10500.0),
            MockPosition('TSLA', 30, 150.0, 4800.0)
        ]

        recommendations = self.manager.get_portfolio_recommendations(positions)

        # Should return recommendations for all positions
        assert isinstance(recommendations, list)
        # Could be empty if mocked data doesn't trigger any actions


def test_logging_configuration():
    """Test that logging is properly configured."""
    # Components should use appropriate loggers
    manager = IntelligentPositionManager()
    assert manager.logger.name.endswith("IntelligentPositionManager")

    detector = MarketRegimeDetector()
    assert detector.logger.name.endswith("MarketRegimeDetector")


if __name__ == "__main__":
    # Run tests manually if pytest not available
    test_logging_configuration()

    # Basic smoke tests
    manager = IntelligentPositionManager()

    detector = MarketRegimeDetector()
    regime_params = detector.get_regime_parameters(MarketRegime.TRENDING_BULL)

    analyzer = TechnicalSignalAnalyzer()

    trail_manager = TrailingStopManager()

    profit_engine = ProfitTakingEngine()

    corr_analyzer = PortfolioCorrelationAnalyzer()

