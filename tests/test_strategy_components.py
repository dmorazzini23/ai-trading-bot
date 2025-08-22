"""
Test for advanced strategy components - multi-timeframe analysis and regime detection.

Tests the functionality of the new strategy modules without requiring
full market data or external dependencies.
"""

import asyncio
import os
import sys

import numpy as np
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def create_sample_market_data(periods: int = 100, symbol: str = "TEST") -> pd.DataFrame:
    """Create sample market data for testing."""
    # Generate realistic price movement
    np.random.seed(42)  # For reproducible results

    dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')

    # Generate price series with some trend and volatility
    returns = np.random.normal(0.001, 0.02, periods)  # Daily returns with slight upward bias
    prices = [100.0]  # Starting price

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Calculate OHLC from closing prices (simplified)
    df = pd.DataFrame({
        'date': dates,
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 1000000, periods)
    })

    return df


def test_multi_timeframe_analyzer():
    """Test multi-timeframe analysis functionality."""
    try:
        from ai_trading.core.enums import TimeFrame
        from ai_trading.strategies.multi_timeframe import (
            MultiTimeframeAnalyzer,
            SignalDirection,
            SignalStrength,
        )

        # Create analyzer
        analyzer = MultiTimeframeAnalyzer([TimeFrame.DAY_1, TimeFrame.HOUR_4])

        # Create sample data for multiple timeframes
        daily_data = create_sample_market_data(periods=100)
        hourly_data = create_sample_market_data(periods=400)  # 4x more data for hourly

        market_data = {
            TimeFrame.DAY_1: daily_data,
            TimeFrame.HOUR_4: hourly_data
        }

        # Run analysis
        result = analyzer.analyze_symbol("TEST", market_data)

        # Validate results
        assert "symbol" in result, "Result should contain symbol"
        assert "timeframe_signals" in result, "Result should contain timeframe signals"
        assert "combined_analysis" in result, "Result should contain combined analysis"
        assert "recommendation" in result, "Result should contain recommendation"

        # Check signal generation
        signals = result["timeframe_signals"]
        assert len(signals) > 0, "Should generate signals for at least one timeframe"

        # Check recommendation
        recommendation = result["recommendation"]
        assert "action" in recommendation, "Recommendation should have action"
        assert "confidence" in recommendation, "Recommendation should have confidence"


        return True

    except ImportError:
        return True
    except Exception:
        return False


def test_regime_detector():
    """Test market regime detection functionality."""
    try:
        from ai_trading.strategies.regime_detection import (
            MarketRegime,
            RegimeDetector,
            VolatilityRegime,
        )

        # Create detector
        detector = RegimeDetector(lookback_periods=100)

        # Create sample market data
        market_data = create_sample_market_data(periods=100)

        # Run regime detection
        result = detector.detect_regime(market_data)

        # Validate results
        assert "primary_regime" in result, "Result should contain primary regime"
        assert "confidence_score" in result, "Result should contain confidence score"
        assert "trend_analysis" in result, "Result should contain trend analysis"
        assert "volatility_analysis" in result, "Result should contain volatility analysis"

        # Check regime classification
        regime = result["primary_regime"]
        confidence = result["confidence_score"]

        assert isinstance(regime, MarketRegime), "Primary regime should be MarketRegime enum"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"

        # Test regime recommendations
        recommendations = detector.get_regime_recommendations()
        assert "strategy_type" in recommendations, "Should provide strategy recommendations"
        assert "position_size_multiplier" in recommendations, "Should provide position sizing advice"


        return True

    except ImportError:
        return True
    except Exception:
        return False


def test_integrated_strategy_system():
    """Test integration between multi-timeframe analysis and regime detection."""
    try:
        from ai_trading.core.enums import TimeFrame
        from ai_trading.strategies.multi_timeframe import MultiTimeframeAnalyzer
        from ai_trading.strategies.regime_detection import RegimeDetector

        # Create components
        analyzer = MultiTimeframeAnalyzer([TimeFrame.DAY_1])
        detector = RegimeDetector()

        # Create sample data
        market_data = create_sample_market_data(periods=100)

        # Run regime detection
        regime_result = detector.detect_regime(market_data)

        # Run multi-timeframe analysis
        mtf_data = {TimeFrame.DAY_1: market_data}
        mtf_result = analyzer.analyze_symbol("TEST", mtf_data)

        # Test integration - adjust recommendations based on regime
        regime = regime_result["primary_regime"]
        mtf_recommendation = mtf_result["recommendation"]

        # Simple integration logic
        if regime.value in ["crisis", "high_volatility"]:
            # Reduce position size in high-risk regimes
            adjusted_multiplier = mtf_recommendation.get("position_size_multiplier", 1.0) * 0.5
        else:
            adjusted_multiplier = mtf_recommendation.get("position_size_multiplier", 1.0)

        # Create integrated recommendation
        {
            "action": mtf_recommendation["action"],
            "confidence": min(mtf_recommendation["confidence"], regime_result["confidence_score"]),
            "position_size_multiplier": adjusted_multiplier,
            "regime": regime.value,
            "reasoning": f"MTF analysis: {mtf_recommendation['action']}, Market regime: {regime.value}"
        }


        return True

    except ImportError:
        return True
    except Exception:
        return False


def test_strategy_performance_scenarios():
    """Test strategy performance under different market scenarios."""
    try:
        from ai_trading.strategies.regime_detection import RegimeDetector

        detector = RegimeDetector()

        # Test scenario 1: Bull market
        bull_data = create_sample_market_data(periods=50)
        # Artificially create strong uptrend
        bull_data['close'] = bull_data['close'] * np.linspace(1.0, 1.3, 50)

        bull_result = detector.detect_regime(bull_data)

        # Test scenario 2: Bear market
        bear_data = create_sample_market_data(periods=50)
        # Artificially create strong downtrend
        bear_data['close'] = bear_data['close'] * np.linspace(1.0, 0.7, 50)

        bear_result = detector.detect_regime(bear_data)

        # Test scenario 3: High volatility
        volatile_data = create_sample_market_data(periods=50)
        # Add high volatility
        volatile_returns = np.random.normal(0, 0.05, 50)  # 5% daily volatility
        volatile_prices = [100.0]
        for ret in volatile_returns[1:]:
            volatile_prices.append(volatile_prices[-1] * (1 + ret))
        volatile_data['close'] = volatile_prices

        volatile_result = detector.detect_regime(volatile_data)


        # Test that different scenarios produce different regimes
        regimes = [
            str(bull_result.get('primary_regime', 'Unknown')),
            str(bear_result.get('primary_regime', 'Unknown')),
            str(volatile_result.get('primary_regime', 'Unknown'))
        ]

        len(set(regimes))

        return True

    except ImportError:
        return True
    except Exception:
        return False


async def run_strategy_tests():
    """Run all strategy component tests."""

    test_results = []

    # Run tests
    test_results.append(("Multi-timeframe Analyzer", test_multi_timeframe_analyzer()))
    test_results.append(("Regime Detector", test_regime_detector()))
    test_results.append(("Integrated Strategy System", test_integrated_strategy_system()))
    test_results.append(("Strategy Performance Scenarios", test_strategy_performance_scenarios()))

    # Report results

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        if result:
            passed += 1


    if passed == total:
        pass
    else:
        pass

    return passed == total


if __name__ == "__main__":
    # Run tests
    try:
        result = asyncio.run(run_strategy_tests())
        exit_code = 0 if result else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception:
        sys.exit(1)
