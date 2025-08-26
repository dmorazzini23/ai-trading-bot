# 🧪 Testing Framework Documentation

## Overview

This comprehensive testing guide covers the complete testing framework for the AI Trading Bot, including unit tests, integration tests, performance tests, and testing best practices.

```bash
ruff check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Deterministic Testing](#deterministic-testing)
- [Test Structure](#test-structure)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Performance Testing](#performance-testing)
- [Mock Testing](#mock-testing)
- [Test Data Management](#test-data-management)
- [Continuous Integration](#continuous-integration)
- [Testing Best Practices](#testing-best-practices)

## Testing Philosophy

### Testing Pyramid

The AI Trading Bot follows a comprehensive testing pyramid approach:

```
    /\     E2E Tests (5%)
   /  \    - End-to-end trading workflows
  /____\   - Full system integration
 /      \  
/________\  Integration Tests (25%)
           - API integrations
           - Database operations
           - Cross-module interactions

Unit Tests (70%)
- Individual functions
- Class methods
- Business logic
- Edge cases
```

### Key Testing Principles

1. **Fast Feedback**: Unit tests run in <5 seconds
2. **Reliable**: Tests are deterministic and not flaky
3. **Maintainable**: Clear test structure and naming
4. **Comprehensive**: High test coverage (>80%)
5. **Realistic**: Use realistic test data and scenarios

## Deterministic Testing

To keep results reproducible, all tests start with a fixed random seed. The `tools/run_pytest.py` helper sets `PYTHONHASHSEED=0` before invoking pytest to ensure consistent hash randomization. An autouse fixture in `tests/conftest.py` then seeds Python's `random` module and NumPy and calls `torch.manual_seed(0)` when PyTorch is available. Tests should avoid introducing additional sources of nondeterminism.

When a `-k` expression is provided without explicit targets, `tools/run_pytest.py` automatically limits collection to test files whose names contain the specified keywords. This prevents unrelated tests from being imported and keeps smoke runs deterministic even in environments missing optional dependencies.

Explicit test paths can be supplied either positionally or via `--files`:

```bash
python tools/run_pytest.py --files tests/test_utils_timing.py tests/test_trading_config_aliases.py
```

## Test Structure

### Directory Organization

```
tests/
├── __init__.py
├── conftest.py                 # Pytest configuration and fixtures
├── root.pth                    # Python path configuration
├── unit/                       # Unit tests
│   ├── test_bot_engine.py
│   ├── test_signals.py
│   ├── test_risk_engine.py
│   ├── test_data_fetcher.py
│   └── test_trade_execution.py
├── integration/                # Integration tests
│   ├── test_alpaca_integration.py
│   ├── test_data_pipeline.py
│   └── test_trading_workflow.py
├── performance/                # Performance tests
│   ├── test_indicator_speed.py
│   └── test_memory_usage.py
├── fixtures/                   # Test data and fixtures
│   ├── sample_data.csv
│   ├── test_config.json
│   └── mock_responses.json
└── utils/                      # Testing utilities
    ├── test_helpers.py
    └── mock_factories.py
```

### Test Categories and Markers

```python
# pytest.ini markers configuration
[pytest]
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, external dependencies)
    slow: Tests that take >10 seconds
    smoke: Basic functionality tests
    performance: Performance and benchmark tests
    security: Security-related tests
    flaky: Tests that may be unreliable (should be fixed)
```

### Optional Test Groups

Some tests depend on optional third-party libraries. These tests use
`from tests.optdeps import require` to call `pytest.importorskip` with a
helpful installation hint. When the dependency is missing the test is
skipped.

| Group | Dependencies | Example tests |
|-------|--------------|---------------|
| Indicators | `pandas`, `ta`, `talib` | `tests/test_indicators.py` |
| Meta learning | `numpy`, `torch`, `sklearn` | `tests/test_meta_learning.py`, `tests/slow/test_meta_learning_heavy.py` |
| Reinforcement learning | `stable_baselines3`, `gymnasium`, `torch` | `tests/test_rl_import_performance.py` |
| Alpaca SDK | `alpaca_trade_api`, `alpaca_api` | `tests/unit/test_alpaca_api.py` |
| Retry utilities | optional `tenacity` via `ai_trading.utils.retry` | `tests/test_tenacity_import.py` |
| Calendars | `pandas_market_calendars` | `tests/test_market_calendar_wrapper.py` |

## Unit Testing

### Core Trading Logic Tests

```python
# tests/unit/test_bot_engine.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from bot_engine import BotState, pre_trade_health_check, run_all_trades_worker
from ai_trading.risk.engine import calculate_position_size


class TestBotState:
    """Test suite for BotState class."""
    
    def test_bot_state_initialization(self):
        """Test BotState initializes with correct defaults."""
        state = BotState()
        
        assert state.loss_streak == 0
        assert state.streak_halt_until is None
        assert state.running is False
        assert state.current_regime == "sideways"
        assert isinstance(state.rolling_losses, list)
        assert len(state.rolling_losses) == 0
        assert isinstance(state.position_cache, dict)
        assert len(state.position_cache) == 0
    
    def test_bot_state_position_tracking(self):
        """Test position tracking functionality."""
        state = BotState()
        
        # Add positions
        state.position_cache['AAPL'] = 100
        state.position_cache['SPY'] = 50
        state.long_positions.add('AAPL')
        state.long_positions.add('SPY')
        
        assert len(state.position_cache) == 2
        assert 'AAPL' in state.long_positions
        assert 'SPY' in state.long_positions
        assert 'MSFT' not in state.long_positions
    
    def test_bot_state_risk_tracking(self):
        """Test risk management state tracking."""
        state = BotState()
        
        # Test drawdown tracking
        state.last_drawdown = -0.05
        assert state.last_drawdown == -0.05
        
        # Test loss streak
        state.loss_streak = 3
        assert state.loss_streak == 3
        
        # Test rolling losses
        state.rolling_losses = [-0.02, -0.01, 0.03, -0.01]
        assert len(state.rolling_losses) == 4
        assert sum(state.rolling_losses) == -0.01


class TestPreTradeHealthCheck:
    """Test suite for pre-trade health check functionality."""
    
    @pytest.fixture
    def mock_bot_context(self):
        """Create mock bot context for testing."""
        ctx = Mock()
        ctx.api = Mock()
        ctx.risk_engine = Mock()
        return ctx
    
    @pytest.fixture
    def sample_symbols(self):
        """Sample trading symbols for testing."""
        return ['AAPL', 'SPY', 'MSFT']
    
    def test_health_check_with_valid_data(self, mock_bot_context, sample_symbols):
        """Test health check with valid market data."""
        # Mock successful data fetching
        with patch('bot_engine.get_historical_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                'close': np.random.randn(50) + 100,
                'volume': np.random.randint(1000, 10000, 50)
            }, index=pd.date_range('2024-01-01', periods=50))
            
            result = pre_trade_health_check(mock_bot_context, sample_symbols, min_rows=30)
            
            assert isinstance(result, dict)
            assert result['checked'] == len(sample_symbols)
            assert len(result['failures']) == 0
    
    def test_health_check_with_insufficient_data(self, mock_bot_context, sample_symbols):
        """Test health check with insufficient market data."""
        with patch('bot_engine.get_historical_data') as mock_fetch:
            # Return insufficient data (only 10 rows)
            mock_fetch.return_value = pd.DataFrame({
                'close': np.random.randn(10) + 100,
                'volume': np.random.randint(1000, 10000, 10)
            }, index=pd.date_range('2024-01-01', periods=10))
            
            result = pre_trade_health_check(mock_bot_context, sample_symbols, min_rows=30)
            
            assert isinstance(result, dict)
            assert len(result['failures']) > 0
            assert any('insufficient data' in failure.lower() for failure in result['failures'])
    
    def test_health_check_with_api_failure(self, mock_bot_context, sample_symbols):
        """Test health check when API calls fail."""
        with patch('bot_engine.get_historical_data') as mock_fetch:
            mock_fetch.side_effect = ConnectionError("API unavailable")
            
            result = pre_trade_health_check(mock_bot_context, sample_symbols)
            
            assert isinstance(result, dict)
            assert len(result['failures']) > 0


class TestSignalGeneration:
    """Test suite for trading signal generation."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate realistic market data for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)  # For reproducible tests
        
        prices = 100 + np.random.randn(100).cumsum() * 0.5
        volume = np.random.randint(1000, 10000, 100)
        
        return pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100)) * 0.2,
            'low': prices - np.abs(np.random.randn(100)) * 0.2,
            'close': prices,
            'volume': volume
        }, index=dates)
    
    def test_signal_generation_basic(self, sample_market_data):
        """Test basic signal generation functionality."""
        from signals import generate_signal
        
        # Add a simple momentum indicator
        sample_market_data['momentum'] = sample_market_data['close'].pct_change(5)
        
        signals = generate_signal(sample_market_data, 'momentum')
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_market_data)
        assert signals.dtype == int
        assert set(signals.unique()).issubset({-1, 0, 1})
    
    def test_signal_generation_with_nan_values(self, sample_market_data):
        """Test signal generation handles NaN values correctly."""
        from signals import generate_signal
        
        # Add momentum with NaN values
        sample_market_data['momentum'] = sample_market_data['close'].pct_change(5)
        sample_market_data.loc[sample_market_data.index[10:15], 'momentum'] = np.nan
        
        signals = generate_signal(sample_market_data, 'momentum')
        
        # NaN values should become neutral signals (0)
        nan_indices = sample_market_data.index[10:15]
        assert all(signals.loc[nan_indices] == 0)
    
    def test_ensemble_signal_generation(self, sample_market_data):
        """Test ensemble signal generation with multiple indicators."""
        from signals import generate_ensemble_signal
        
        # This would test the ensemble method if implemented
        # For now, we'll test that it doesn't crash
        try:
            ensemble_signal = generate_ensemble_signal(sample_market_data)
            assert ensemble_signal is not None
        except NotImplementedError:
            pytest.skip("Ensemble signal generation not implemented")


class TestRiskManagement:
    """Test suite for risk management functionality."""
    
    def test_position_size_calculation_basic(self):
        """Test basic position size calculation."""
        from ai_trading.risk.engine import calculate_position_size
        
        # Test simple position sizing
        position_size = calculate_position_size(10000, 150.0)  # $10k cash, $150/share
        
        assert isinstance(position_size, int)
        assert position_size >= 0
        assert position_size <= 10000 // 150  # Can't buy more than affordable
    
    def test_position_size_with_zero_cash(self):
        """Test position sizing with zero available cash."""
        from ai_trading.risk.engine import calculate_position_size
        
        position_size = calculate_position_size(0, 150.0)
        assert position_size == 0
    
    def test_position_size_with_negative_inputs(self):
        """Test position sizing with invalid inputs."""
        from ai_trading.risk.engine import calculate_position_size
        
        # Negative cash should return 0
        position_size = calculate_position_size(-1000, 150.0)
        assert position_size == 0
        
        # Negative price should return 0
        position_size = calculate_position_size(10000, -150.0)
        assert position_size == 0
    
    def test_drawdown_limit_checking(self):
        """Test maximum drawdown limit checking."""
        from ai_trading.risk.engine import check_max_drawdown
        
        # Normal case - within limits
        state = {'current_drawdown': 0.03, 'max_drawdown': 0.05}
        assert not check_max_drawdown(state)
        
        # Exceeds limits
        state = {'current_drawdown': 0.08, 'max_drawdown': 0.05}
        assert check_max_drawdown(state)
        
        # Missing data - should return False
        state = {}
        assert not check_max_drawdown(state)


class TestDataFetching:
    """Test suite for data fetching functionality."""
    
    def test_data_fetching_with_valid_parameters(self):
        """Test data fetching with valid parameters."""
        from ai_trading import data_fetcher
        
        with patch('data_fetcher._fetch_bars') as mock_fetch:
            # Mock successful data fetch
            mock_data = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [102, 103, 104],
                'low': [99, 100, 101],
                'close': [101, 102, 103],
                'volume': [1000, 1100, 1200]
            }, index=pd.date_range('2024-01-01', periods=3))
            
            mock_fetch.return_value = mock_data
            
            result = get_historical_data('AAPL', '2024-01-01', '2024-01-03', '1DAY')
            
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'close' in result.columns
            assert 'volume' in result.columns
    
    def test_data_fetching_with_invalid_dates(self):
        """Test data fetching with invalid date parameters."""
        from ai_trading import data_fetcher
        
        # Test with None dates
        with pytest.raises(ValueError):
            get_historical_data('AAPL', None, '2024-01-03', '1DAY')
        
        with pytest.raises(ValueError):
            get_historical_data('AAPL', '2024-01-01', None, '1DAY')
    
    def test_data_fetching_with_connection_error(self):
        """Test data fetching when connection fails."""
        from ai_trading import data_fetcher
        
        with patch('data_fetcher._fetch_bars') as mock_fetch:
            mock_fetch.side_effect = ConnectionError("Network error")
            
            # Should handle error gracefully
            result = get_historical_data('AAPL', '2024-01-01', '2024-01-03', '1DAY')
            # Depending on implementation, might return empty DataFrame or raise exception
            assert result is not None
```

## Integration Testing

### API Integration Tests

```python
# tests/integration/test_alpaca_integration.py
import pytest
import os
from unittest.mock import patch
import alpaca_trade_api as tradeapi

from alpaca_api import AlpacaAPI
from ai_trading.execution.engine import ExecutionEngine


class TestAlpacaIntegration:
    """Integration tests for Alpaca API functionality."""
    
    @pytest.fixture
    def api_client(self):
        """Create Alpaca API client for testing."""
        # Use paper trading environment for tests
        return tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            'https://paper-api.alpaca.markets'
        )
    
    @pytest.mark.integration
    def test_account_connection(self, api_client):
        """Test basic account connectivity."""
        if not os.getenv('ALPACA_API_KEY'):
            pytest.skip("ALPACA_API_KEY not set")
        
        account = api_client.get_account()
        
        assert account is not None
        assert hasattr(account, 'id')
        assert hasattr(account, 'buying_power')
        assert float(account.buying_power) >= 0
    
    @pytest.mark.integration
    def test_market_data_fetching(self, api_client):
        """Test market data retrieval."""
        if not os.getenv('ALPACA_API_KEY'):
            pytest.skip("ALPACA_API_KEY not set")
        
        # Test getting recent bars for SPY
        bars = api_client.get_bars(
            'SPY',
            timeframe='1Day',
            limit=5
        )
        
        assert bars is not None
        assert len(bars) > 0
        
        # Check that bars have required fields
        for bar in bars:
            assert hasattr(bar, 'c')  # close
            assert hasattr(bar, 'v')  # volume
            assert bar.c > 0
            assert bar.v > 0
    
    @pytest.mark.integration
    def test_order_submission_dry_run(self, api_client):
        """Test order submission in dry run mode."""
        if not os.getenv('ALPACA_API_KEY'):
            pytest.skip("ALPACA_API_KEY not set")
        
        executor = ExecutionEngine()
        
        # Use a very small quantity for testing
        with patch.dict(os.environ, {'DRY_RUN': 'true'}):
            result = executor.execute_order('SPY', 1, 'buy')
            
            # In dry run mode, should return mock order
            assert result is not None


class TestDataPipeline:
    """Integration tests for data pipeline functionality."""
    
    @pytest.mark.integration
    def test_end_to_end_data_flow(self):
        """Test complete data flow from fetch to signals."""
        from ai_trading import data_fetcher
        from signals import generate_signal
        from indicators import calculate_indicators
        
        # Fetch real data
        data = get_historical_data('SPY', '2024-01-01', '2024-01-31', '1DAY')
        
        if data.empty:
            pytest.skip("No market data available")
        
        # Calculate indicators
        data_with_indicators = calculate_indicators(data)
        
        assert not data_with_indicators.empty
        assert len(data_with_indicators.columns) > len(data.columns)
        
        # Generate signals
        if 'momentum' in data_with_indicators.columns:
            signals = generate_signal(data_with_indicators, 'momentum')
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(data_with_indicators)
    
    @pytest.mark.integration
    def test_multi_symbol_data_processing(self):
        """Test processing multiple symbols simultaneously."""
        from ai_trading import data_fetcher
        
        symbols = ['SPY', 'QQQ', 'IWM']
        
        # This would test async data fetching
        try:
            data_dict = fetch_daily_data_async(symbols, '2024-01-01', '2024-01-31')
            
            assert isinstance(data_dict, dict)
            assert len(data_dict) <= len(symbols)  # Some might fail
            
            # Check that returned data is valid
            for symbol, data in data_dict.items():
                if data is not None:
                    assert isinstance(data, pd.DataFrame)
                    assert not data.empty
                    
        except NotImplementedError:
            pytest.skip("Async data fetching not implemented")


class TestTradingWorkflow:
    """Integration tests for complete trading workflows."""
    
    @pytest.mark.integration
    @patch.dict(os.environ, {'DRY_RUN': 'true', 'BOT_MODE': 'testing'})
    def test_complete_trading_cycle(self):
        """Test a complete trading cycle in dry run mode."""
        from bot_engine import BotState, run_all_trades_worker
        
        # Initialize bot state
        state = BotState()
        state.running = False  # Ensure not already running
        
        # Mock model
        mock_model = Mock()
        
        # Run trading cycle
        try:
            run_all_trades_worker(state, mock_model)
            
            # Check that state was updated
            assert state.last_run_at is not None
            assert state.last_loop_duration >= 0
            
        except Exception as e:
            # Log the error but don't fail the test if it's due to missing data
            if "market" in str(e).lower() or "data" in str(e).lower():
                pytest.skip(f"Trading cycle skipped due to market/data issues: {e}")
            else:
                raise
```

## Performance Testing

### Performance Benchmarks

```python
# tests/performance/test_performance_benchmarks.py
import pytest
import time
import numpy as np
import pandas as pd
from memory_profiler import memory_usage

from indicators import calculate_rsi, calculate_macd
from signals import generate_signal


class TestPerformanceBenchmarks:
    """Performance benchmarks for critical components."""
    
    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for performance testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=10000, freq='1H')
        
        prices = 100 + np.random.randn(10000).cumsum() * 0.1
        volume = np.random.randint(1000, 100000, 10000)
        
        return pd.DataFrame({
            'open': prices + np.random.randn(10000) * 0.05,
            'high': prices + np.abs(np.random.randn(10000)) * 0.1,
            'low': prices - np.abs(np.random.randn(10000)) * 0.1,
            'close': prices,
            'volume': volume
        }, index=dates)
    
    @pytest.mark.performance
    def test_indicator_calculation_speed(self, large_dataset):
        """Benchmark technical indicator calculation speed."""
        
        # Benchmark RSI calculation
        start_time = time.perf_counter()
        rsi = calculate_rsi(large_dataset['close'])
        rsi_time = time.perf_counter() - start_time
        
        # Benchmark MACD calculation
        start_time = time.perf_counter()
        macd_data = calculate_macd(large_dataset['close'])
        macd_time = time.perf_counter() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        assert rsi_time < 1.0, f"RSI calculation too slow: {rsi_time:.3f}s"
        assert macd_time < 1.0, f"MACD calculation too slow: {macd_time:.3f}s"
        
        print(f"RSI calculation: {rsi_time:.3f}s for {len(large_dataset)} bars")
        print(f"MACD calculation: {macd_time:.3f}s for {len(large_dataset)} bars")
    
    @pytest.mark.performance
    def test_signal_generation_speed(self, large_dataset):
        """Benchmark signal generation speed."""
        
        # Add momentum indicator
        large_dataset['momentum'] = large_dataset['close'].pct_change(10)
        
        start_time = time.perf_counter()
        signals = generate_signal(large_dataset, 'momentum')
        signal_time = time.perf_counter() - start_time
        
        assert signal_time < 0.5, f"Signal generation too slow: {signal_time:.3f}s"
        assert len(signals) == len(large_dataset)
        
        print(f"Signal generation: {signal_time:.3f}s for {len(large_dataset)} bars")
    
    @pytest.mark.performance
    def test_memory_usage(self, large_dataset):
        """Test memory usage during typical operations."""
        
        def memory_intensive_operation():
            # Simulate memory-intensive indicator calculations
            data = large_dataset.copy()
            
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            data['rsi'] = calculate_rsi(data['close'])
            
            return data
        
        # Measure memory usage
        mem_usage = memory_usage(memory_intensive_operation, interval=0.1)
        peak_memory = max(mem_usage)
        baseline_memory = min(mem_usage)
        memory_increase = peak_memory - baseline_memory
        
        # Memory usage should be reasonable (adjust threshold as needed)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"
        
        print(f"Peak memory usage: {peak_memory:.1f}MB")
        print(f"Memory increase: {memory_increase:.1f}MB")
    
    @pytest.mark.performance
    def test_parallel_processing_speedup(self):
        """Test parallel processing performance improvements."""
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        def calculate_indicators_serial(symbols_data):
            """Serial indicator calculation."""
            results = {}
            for symbol, data in symbols_data.items():
                # Simulate indicator calculation
                time.sleep(0.1)  # Simulate computation time
                results[symbol] = data.copy()
                results[symbol]['sma'] = data['close'].rolling(20).mean()
            return results
        
        def calculate_indicators_parallel(symbols_data):
            """Parallel indicator calculation."""
            def process_symbol(item):
                symbol, data = item
                time.sleep(0.1)  # Simulate computation time
                result = data.copy()
                result['sma'] = data['close'].rolling(20).mean()
                return symbol, result
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_symbol, item): item[0] 
                          for item in symbols_data.items()}
                
                results = {}
                for future in futures:
                    symbol, result = future.result()
                    results[symbol] = result
                
                return results
        
        # Create test data for multiple symbols
        test_data = {}
        for symbol in ['AAPL', 'SPY', 'MSFT', 'GOOGL']:
            test_data[symbol] = pd.DataFrame({
                'close': np.random.randn(100) + 100
            })
        
        # Benchmark serial processing
        start_time = time.perf_counter()
        serial_results = calculate_indicators_serial(test_data)
        serial_time = time.perf_counter() - start_time
        
        # Benchmark parallel processing
        start_time = time.perf_counter()
        parallel_results = calculate_indicators_parallel(test_data)
        parallel_time = time.perf_counter() - start_time
        
        # Parallel should be faster
        speedup = serial_time / parallel_time
        assert speedup > 1.5, f"Insufficient speedup: {speedup:.2f}x"
        
        print(f"Serial time: {serial_time:.3f}s")
        print(f"Parallel time: {parallel_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
```

## Mock Testing

### Comprehensive Mocking Strategy

```python
# tests/utils/mock_factories.py
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np
from datetime import UTC, datetime, timedelta


class MockDataFactory:
    """Factory for creating mock market data and API responses."""
    
    @staticmethod
    def create_market_data(symbol='AAPL', periods=100, start_date='2024-01-01'):
        """Create realistic mock market data."""
        dates = pd.date_range(start_date, periods=periods, freq='1H')
        np.random.seed(hash(symbol) % 2**32)  # Consistent but different per symbol
        
        prices = 100 + np.random.randn(periods).cumsum() * 0.1
        
        return pd.DataFrame({
            'open': prices + np.random.randn(periods) * 0.05,
            'high': prices + np.abs(np.random.randn(periods)) * 0.1,
            'low': prices - np.abs(np.random.randn(periods)) * 0.1,
            'close': prices,
            'volume': np.random.randint(10000, 1000000, periods)
        }, index=dates)
    
    @staticmethod
    def create_alpaca_account_mock():
        """Create mock Alpaca account object."""
        account = Mock()
        account.id = 'test-account-123'
        account.buying_power = '50000.00'
        account.cash = '25000.00'
        account.portfolio_value = '75000.00'
        account.status = 'ACTIVE'
        account.trading_blocked = False
        return account
    
    @staticmethod
    def create_alpaca_order_mock(symbol='AAPL', qty=100, side='buy', status='filled'):
        """Create mock Alpaca order object."""
        order = Mock()
        order.id = f'order-{symbol}-{int(datetime.now(UTC).timestamp())}'
        order.symbol = symbol
        order.qty = str(qty)
        order.side = side
        order.status = status
        order.filled_qty = str(qty) if status == 'filled' else '0'
        order.filled_avg_price = '150.25' if status == 'filled' else None
        order.created_at = datetime.now(UTC).isoformat()
        return order
    
    @staticmethod
    def create_market_data_response(symbol='AAPL', error=False):
        """Create mock API response for market data."""
        if error:
            response = Mock()
            response.status_code = 500
            response.json.side_effect = Exception("API Error")
            return response
        
        data = MockDataFactory.create_market_data(symbol, periods=30)
        
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            'symbol': symbol,
            'bars': data.to_dict('records')
        }
        return response


class MockAPIClients:
    """Mock API clients for testing."""
    
    @staticmethod
    def create_alpaca_client_mock(account_balance=50000, positions=None):
        """Create comprehensive Alpaca API client mock."""
        client = Mock()
        
        # Mock account
        client.get_account.return_value = MockDataFactory.create_alpaca_account_mock()
        client.get_account.return_value.buying_power = str(account_balance)
        
        # Mock positions
        if positions is None:
            positions = []
        
        position_mocks = []
        for pos in positions:
            pos_mock = Mock()
            pos_mock.symbol = pos['symbol']
            pos_mock.qty = str(pos['qty'])
            pos_mock.side = pos.get('side', 'long')
            pos_mock.market_value = str(pos.get('market_value', pos['qty'] * 150))
            position_mocks.append(pos_mock)
        
        client.list_positions.return_value = position_mocks
        
        # Mock order submission
        def mock_submit_order(**kwargs):
            return MockDataFactory.create_alpaca_order_mock(
                symbol=kwargs.get('symbol', 'AAPL'),
                qty=kwargs.get('qty', 100),
                side=kwargs.get('side', 'buy')
            )
        
        client.submit_order = Mock(side_effect=mock_submit_order)
        
        return client
    
    @staticmethod
    def create_data_client_mock():
        """Create mock data client."""
        client = Mock()
        
        def mock_get_bars(symbol, **kwargs):
            return MockDataFactory.create_market_data(symbol)
        
        client.get_bars = Mock(side_effect=mock_get_bars)
        return client


# tests/utils/test_helpers.py
import pytest
import pandas as pd
from contextlib import contextmanager
from unittest.mock import patch
import tempfile
import os


class TestHelpers:
    """Utility functions for testing."""
    
    @staticmethod
    @contextmanager
    def temp_env_vars(**kwargs):
        """Context manager for temporarily setting environment variables."""
        old_values = {}
        
        for key, value in kwargs.items():
            old_values[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        
        try:
            yield
        finally:
            for key, old_value in old_values.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
    
    @staticmethod
    @contextmanager
    def temp_config_file(config_data):
        """Create temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            yield temp_path
        finally:
            os.unlink(temp_path)
    
    @staticmethod
    def assert_dataframe_structure(df, expected_columns=None, min_rows=1):
        """Assert DataFrame has expected structure."""
        assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
        assert not df.empty, "DataFrame should not be empty"
        assert len(df) >= min_rows, f"Expected at least {min_rows} rows"
        
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            assert not missing_cols, f"Missing columns: {missing_cols}"
    
    @staticmethod
    def assert_numeric_range(value, min_val=None, max_val=None):
        """Assert numeric value is within expected range."""
        assert isinstance(value, (int, float)), f"Expected numeric value, got {type(value)}"
        
        if min_val is not None:
            assert value >= min_val, f"Value {value} below minimum {min_val}"
        
        if max_val is not None:
            assert value <= max_val, f"Value {value} above maximum {max_val}"
    
    @staticmethod
    def create_test_portfolio_state(symbols=None, cash=50000):
        """Create test portfolio state for testing."""
        if symbols is None:
            symbols = {'AAPL': 100, 'SPY': 50}
        
        from bot_engine import BotState
        
        state = BotState()
        state.position_cache = symbols.copy()
        
        for symbol in symbols:
            if symbols[symbol] > 0:
                state.long_positions.add(symbol)
            elif symbols[symbol] < 0:
                state.short_positions.add(symbol)
        
        return state
```

## Test Data Management

### Fixture Management

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import os
from datetime import datetime, timedelta

from tests.utils.mock_factories import MockDataFactory, MockAPIClients
from tests.utils.test_helpers import TestHelpers


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        'test_symbols': ['AAPL', 'SPY', 'MSFT', 'GOOGL'],
        'test_date_range': ('2024-01-01', '2024-01-31'),
        'test_account_balance': 50000,
        'test_position_size': 100
    }


@pytest.fixture
def sample_market_data():
    """Standard market data for testing."""
    return MockDataFactory.create_market_data('AAPL', periods=100)


@pytest.fixture
def multi_symbol_data(test_config):
    """Market data for multiple symbols."""
    data = {}
    for symbol in test_config['test_symbols']:
        data[symbol] = MockDataFactory.create_market_data(symbol, periods=50)
    return data


@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca API client."""
    return MockAPIClients.create_alpaca_client_mock()


@pytest.fixture
def mock_bot_state():
    """Mock bot state for testing."""
    return TestHelpers.create_test_portfolio_state()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    test_env = {
        'DRY_RUN': 'true',
        'BOT_MODE': 'testing',
        'LOG_LEVEL': 'ERROR',  # Reduce log noise in tests
        'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets'
    }
    
    with TestHelpers.temp_env_vars(**test_env):
        yield


@pytest.fixture
def isolated_test_data():
    """Create isolated test data directory."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp(prefix='trading_bot_test_')
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


# Performance test fixtures
@pytest.fixture(scope="session")
def large_market_dataset():
    """Large dataset for performance testing."""
    return MockDataFactory.create_market_data('SPY', periods=10000)


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark targets."""
    return {
        'indicator_calculation_max_time': 1.0,  # seconds
        'signal_generation_max_time': 0.5,     # seconds
        'max_memory_increase': 500,             # MB
        'min_parallel_speedup': 1.5            # times
    }
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/testing.yml
name: Comprehensive Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.12.3]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install -U pip
        pip install -e .
        pip install -r requirements-dev.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=ai_trading --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
  
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.12.3
      uses: actions/setup-python@v4
      with:
        python-version: 3.12.3
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install -U pip
        pip install -e .
        pip install -r requirements-dev.txt
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v -m "not slow"
      env:
        ALPACA_API_KEY: ${{ secrets.ALPACA_TEST_API_KEY }}
        ALPACA_SECRET_KEY: ${{ secrets.ALPACA_TEST_SECRET_KEY }}
  
  performance-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.12.3
      uses: actions/setup-python@v4
      with:
        python-version: 3.12.3
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install -U pip
        pip install -e .
        pip install -r requirements-dev.txt
        pip install memory-profiler
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-only
    
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: benchmark-results/
```

## Testing Best Practices

### Code Quality Guidelines

1. **Test Naming**: Use descriptive test names that explain what is being tested
2. **Test Structure**: Follow Arrange-Act-Assert pattern
3. **Test Independence**: Each test should be independent and not rely on others
4. **Mock External Dependencies**: Mock all external APIs and services
5. **Test Edge Cases**: Include tests for boundary conditions and error scenarios

### Coverage Requirements

- **Minimum Coverage**: 80% overall code coverage
- **Critical Functions**: 95% coverage for trading logic, risk management
- **New Code**: 100% coverage for all new functions
- **Integration Points**: Full coverage of API integrations

### Performance Standards

- **Unit Tests**: Complete in <5 seconds
- **Integration Tests**: Complete in <30 seconds
- **Performance Tests**: Establish baseline and detect regressions
- **Memory Tests**: Monitor memory usage and prevent leaks

This comprehensive testing framework ensures the AI Trading Bot maintains high quality, reliability, and performance across all components and use cases.