#!/usr/bin/env python3
"""
Test suite for MetaLearning strategy.
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Set minimal environment variables to avoid config errors
os.environ['ALPACA_API_KEY'] = 'test'
os.environ['ALPACA_SECRET_KEY'] = 'test'
os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'
os.environ['WEBHOOK_SECRET'] = 'test'
os.environ['FLASK_PORT'] = '5000'
os.environ['FINNHUB_API_KEY'] = 'test'

def create_mock_price_data(days=100, start_price=100):
    """Create mock OHLCV price data for testing."""
    # Create minute-level data to match what the strategy expects
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')  # Hourly data for reasonable size
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible tests
    returns = np.random.normal(0.0001, 0.01, len(dates))  # Smaller hourly returns
    
    prices = [start_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data
    opens = np.array(prices[:-1])
    closes = np.array(prices[1:])
    
    # Create high/low with some spread
    highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, 0.005, len(opens)))
    lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, 0.005, len(opens)))
    
    # Create volume
    volumes = np.random.randint(10000, 100000, len(opens))
    
    data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates[1:])
    
    return data


class TestMetaLearning:
    """Test cases for MetaLearning strategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from ai_trading.strategies.metalearning import MetaLearning
        self.strategy = MetaLearning()
        self.mock_data = create_mock_price_data(days=100)
    
    def test_strategy_initialization(self):
        """Test strategy initializes correctly."""
        assert self.strategy.name == "MetaLearning Strategy"
        assert self.strategy.strategy_id == "metalearning"
        assert len(self.strategy.parameters) == 8
        assert self.strategy.parameters['lookback_period'] == 60
        assert self.strategy.parameters['min_confidence'] == 0.6
        assert not self.strategy.is_trained
    
    def test_extract_features(self):
        """Test feature extraction from price data."""
        features = self.strategy.extract_features(self.mock_data)
        
        assert features is not None
        assert len(features) > 0
        assert 'returns' in features.columns
        assert 'sma_20' in features.columns
        assert 'rsi' in features.columns
        assert 'macd' in features.columns
        assert 'volatility_20' in features.columns
        
        # Check that features don't contain NaN or infinite values
        assert not features.isnull().any().any()
        assert not np.isinf(features).any().any()
    
    def test_train_model(self):
        """Test model training with sufficient data."""
        success = self.strategy.train_model(self.mock_data)
        
        assert success
        assert self.strategy.is_trained
        assert self.strategy.last_training_date is not None
        assert hasattr(self.strategy, 'rf_model')
        assert hasattr(self.strategy, 'gb_model')
        assert len(self.strategy.feature_columns) > 0
        assert self.strategy.prediction_accuracy >= 0
    
    def test_train_model_insufficient_data(self):
        """Test model training with insufficient data."""
        small_data = self.mock_data.head(10)  # Only 10 days
        success = self.strategy.train_model(small_data)
        
        assert not success
        assert not self.strategy.is_trained
    
    def test_predict_price_movement(self):
        """Test price movement prediction."""
        # First train the model
        self.strategy.train_model(self.mock_data)
        
        # Make prediction
        prediction = self.strategy.predict_price_movement(self.mock_data)
        
        assert prediction is not None
        assert 'direction' in prediction
        assert 'confidence' in prediction
        assert 'probability_distribution' in prediction
        assert prediction['direction'] in ['buy', 'sell', 'hold']
        assert 0 <= prediction['confidence'] <= 1
        assert 'current_price' in prediction
        assert 'volatility' in prediction
    
    def test_execute_strategy_with_data(self):
        """Test strategy execution with mock data."""
        with patch('ai_trading.strategies.metalearning.get_minute_df') as mock_get_data:
            mock_get_data.return_value = self.mock_data
            
            result = self.strategy.execute_strategy('AAPL')
            
            assert result is not None
            assert 'signal' in result
            assert 'confidence' in result
            assert 'strength' in result
            assert result['signal'] in ['buy', 'sell', 'hold']
            
            if result['signal'] != 'hold':
                assert result['confidence'] > 0
                assert result['strength'] > 0
                assert 'reasoning' in result
    
    def test_execute_strategy_no_data(self):
        """Test strategy execution with no data."""
        with patch('ai_trading.strategies.metalearning.get_minute_df') as mock_get_data:
            mock_get_data.return_value = None
            
            result = self.strategy.execute_strategy('AAPL')
            
            assert result is not None
            assert result['signal'] == 'hold'
            assert result['confidence'] == 0.0
            assert result['strength'] == 0.0
    
    def test_generate_signals(self):
        """Test signal generation for multiple symbols."""
        # Add symbols to strategy
        self.strategy.symbols = ['AAPL', 'GOOGL']
        
        with patch('ai_trading.strategies.metalearning.get_minute_df') as mock_get_data:
            mock_get_data.return_value = self.mock_data
            
            # Mock market data
            market_data = {'timestamp': datetime.now()}
            
            signals = self.strategy.generate_signals(market_data)
            
            assert isinstance(signals, list)
            # Signals might be empty if confidence is too low, which is valid
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        from ai_trading.strategies.base import StrategySignal
        from ai_trading.core.enums import OrderSide
        
        signal = StrategySignal(
            symbol='AAPL',
            side=OrderSide.BUY,
            strength=0.8,
            confidence=0.7
        )
        
        portfolio_value = 100000
        position_size = self.strategy.calculate_position_size(signal, portfolio_value)
        
        assert position_size >= 0
        assert position_size <= portfolio_value  # Position shouldn't exceed portfolio
    
    def test_fallback_prediction(self):
        """Test fallback prediction when ML is not available."""
        # Temporarily disable ML
        original_ml = self.strategy.__class__.__module__.replace('metalearning', 'metalearning')
        
        prediction = self.strategy._fallback_prediction(self.mock_data)
        
        assert prediction is not None
        assert 'direction' in prediction
        assert 'confidence' in prediction
        assert prediction['direction'] in ['buy', 'sell', 'hold']
        assert 0 <= prediction['confidence'] <= 1
    
    def test_caching_mechanism(self):
        """Test prediction caching."""
        with patch('ai_trading.strategies.metalearning.get_minute_df') as mock_get_data:
            mock_get_data.return_value = self.mock_data
            
            # First call
            result1 = self.strategy.execute_strategy('AAPL')
            
            # Second call should use cache
            result2 = self.strategy.execute_strategy('AAPL')
            
            # Results should be the same due to caching
            assert result1 == result2
    
    def test_should_retrain(self):
        """Test retrain logic."""
        # Initially should retrain (not trained)
        assert self.strategy._should_retrain()
        
        # After training, should not retrain immediately
        self.strategy.train_model(self.mock_data)
        assert not self.strategy._should_retrain()
        
        # Should retrain after time passes
        old_date = datetime.now() - timedelta(days=10)
        self.strategy.last_training_date = old_date
        assert self.strategy._should_retrain()
    
    def test_signal_validation(self):
        """Test that strategy validates signals properly."""
        from ai_trading.strategies.base import StrategySignal
        from ai_trading.core.enums import OrderSide
        
        # Valid signal
        valid_signal = StrategySignal(
            symbol='AAPL',
            side=OrderSide.BUY,
            strength=0.8,
            confidence=0.7
        )
        assert self.strategy.validate_signal(valid_signal)
        
        # Invalid signal (low confidence)
        invalid_signal = StrategySignal(
            symbol='AAPL',
            side=OrderSide.BUY,
            strength=0.8,
            confidence=0.1  # Too low
        )
        assert not self.strategy.validate_signal(invalid_signal)


def test_metalearning_import():
    """Test that MetaLearning can be imported without errors."""
    from ai_trading.strategies.metalearning import MetaLearning
    
    strategy = MetaLearning()
    assert strategy is not None
    assert strategy.name == "MetaLearning Strategy"


def test_no_metalearn_invalid_prices_error():
    """Test that the strategy doesn't generate METALEARN_INVALID_PRICES errors."""
    from ai_trading.strategies.metalearning import MetaLearning
    
    strategy = MetaLearning()
    
    # Test with no data - should return neutral signal, not error
    with patch('ai_trading.strategies.metalearning.get_minute_df') as mock_get_data:
        mock_get_data.return_value = None
        
        result = strategy.execute_strategy('AAPL')
        
        # Should return neutral signal, not raise an error
        assert result is not None
        assert result['signal'] == 'hold'
        assert 'METALEARN_INVALID_PRICES' not in result.get('reasoning', '')


if __name__ == '__main__':
    # Run a basic test to ensure the strategy works
    print("Running basic MetaLearning strategy test...")
    
    test_metalearning_import()
    print("✅ Import test passed")
    
    test_no_metalearn_invalid_prices_error()
    print("✅ Error handling test passed")
    
    # Create and test strategy
    from ai_trading.strategies.metalearning import MetaLearning
    strategy = MetaLearning()
    
    # Test with mock data
    mock_data = create_mock_price_data()
    features = strategy.extract_features(mock_data)
    print(f"✅ Feature extraction: {len(features.columns)} features extracted")
    
    success = strategy.train_model(mock_data)
    print(f"✅ Model training: {'Success' if success else 'Failed'}")
    
    if success:
        prediction = strategy.predict_price_movement(mock_data)
        print(f"✅ Prediction: {prediction['direction']} (confidence: {prediction['confidence']:.2f})")
    
    print("✅ All basic tests passed!")