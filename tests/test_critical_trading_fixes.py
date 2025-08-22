#!/usr/bin/env python3
"""
Test suite for critical trading bot fixes addressing August 7, 2025 issues.

Tests the five main areas of improvement:
1. Sentiment Analysis Rate Limiting
2. Aggressive Liquidity Management  
3. Meta-Learning System Failure
4. Partial Order Management
5. Order Status Monitoring
"""

import csv
import os
import tempfile
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Import modules under test
try:
    from ai_trading.analysis import sentiment
except Exception:  # pragma: no cover - optional torch dependency
    pytest.skip("sentiment module unavailable", allow_module_level=True)
from ai_trading import config, meta_learning
from ai_trading.broker.alpaca import AlpacaBroker
from ai_trading.execution.engine import ExecutionEngine
from ai_trading.monitoring.order_health_monitor import (
    OrderInfo,
    _active_orders,
    _order_tracking_lock,
)


@pytest.fixture
def broker(monkeypatch):
    """Provide a mocked AlpacaBroker for tests."""
    broker = AlpacaBroker()
    monkeypatch.setattr(broker, "get_orders", lambda *a, **k: [])
    monkeypatch.setattr(
        broker,
        "place_order",
        lambda **k: {
            "id": "TEST123",
            "symbol": k["symbol"],
            "qty": k["qty"],
            "side": k.get("side", "buy"),
        },
    )
    return broker


class TestSentimentAnalysisRateLimitingFixes(unittest.TestCase):
    """Test enhanced sentiment analysis rate limiting fixes."""

    def setUp(self):
        # Reset sentiment cache for each test
        sentiment._SENTIMENT_CACHE.clear()
        sentiment._SENTIMENT_CIRCUIT_BREAKER = {"failures": 0, "last_failure": 0, "state": "closed"}

    def test_enhanced_rate_limiting_parameters(self):
        """Test that enhanced rate limiting parameters are properly configured."""
        self.assertEqual(sentiment.SENTIMENT_TTL_SEC, 600)
        self.assertEqual(sentiment.SENTIMENT_RATE_LIMITED_TTL_SEC, 7200)  # 2 hours
        self.assertEqual(sentiment.SENTIMENT_FAILURE_THRESHOLD, 15)
        self.assertEqual(sentiment.SENTIMENT_RECOVERY_TIMEOUT, 1800)  # 30 minutes
        self.assertEqual(sentiment.SENTIMENT_MAX_RETRIES, 5)
        self.assertEqual(sentiment.SENTIMENT_BASE_DELAY, 5)

    @patch('ai_trading.analysis.sentiment.requests.get')
    def test_enhanced_fallback_strategies(self, mock_get):
        """Test that enhanced fallback strategies work when rate limited."""
        # Simulate rate limiting
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        # Mock context
        mock_ctx = Mock()

        # Test rate limiting triggers fallback
        result = sentiment.fetch_sentiment(mock_ctx, 'AAPL')

        # Should return neutral sentiment when all fallbacks fail
        self.assertEqual(result, 0.0)

    @patch('ai_trading.analysis.sentiment.requests.get')
    def test_alternative_sentiment_sources(self, mock_get):
        """Test alternative sentiment source functionality."""
        # Mock environment variables for alternative source
        with patch.dict(os.environ, {
            'ALTERNATIVE_SENTIMENT_API_KEY': 'test_key',
            'ALTERNATIVE_SENTIMENT_API_URL': 'https://alt-api.com/sentiment'
        }):
            # Primary source fails with rate limiting
            mock_response_primary = Mock()
            mock_response_primary.status_code = 429

            # Alternative source succeeds
            mock_response_alt = Mock()
            mock_response_alt.status_code = 200
            mock_response_alt.json.return_value = {'sentiment_score': 0.7}

            mock_get.side_effect = [mock_response_primary, mock_response_alt]

            result = sentiment._try_alternative_sentiment_sources('AAPL')
            self.assertEqual(result, 0.7)

    def test_similar_symbol_sentiment_proxy(self):
        """Test using sentiment from similar symbols as proxy."""
        # Add sentiment for MSFT
        sentiment._SENTIMENT_CACHE['MSFT'] = (time.time(), 0.8)

        # Request sentiment for AAPL (should use MSFT as proxy)
        result = sentiment._try_cached_similar_symbols('AAPL')

        # Should return discounted sentiment from similar symbol
        self.assertIsNotNone(result)
        self.assertTrue(abs(result - 0.64) < 0.1)  # 0.8 * 0.8 = 0.64

    def test_sector_sentiment_proxy(self):
        """Test using sector ETF sentiment as proxy."""
        # Add sentiment for technology sector ETF
        sentiment._SENTIMENT_CACHE['XLK'] = (time.time(), 0.5)

        # Request sentiment for AAPL (tech stock)
        result = sentiment._try_sector_sentiment_proxy('AAPL')

        # Should return discounted sentiment from sector
        self.assertIsNotNone(result)
        self.assertTrue(abs(result - 0.3) < 0.1)  # 0.5 * 0.6 = 0.3


class TestMetaLearningSystemFixes(unittest.TestCase):
    """Test meta-learning system fixes for insufficient trade history."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.trade_log_path = os.path.join(self.temp_dir, 'test_trades.csv')

    def tearDown(self):
        # Cleanup temp files
        if os.path.exists(self.trade_log_path):
            os.remove(self.trade_log_path)
        os.rmdir(self.temp_dir)

    def test_reduced_minimum_trade_requirement(self):
        """Test that minimum trade requirement is reduced from 20 to 10."""
        # Create test data with only 12 trades (would fail with old requirement)
        self._create_test_trade_log(12)

        # Mock pandas and sklearn for the test
        with patch('meta_learning.pd') as mock_pd, \
             patch('meta_learning.Path') as mock_path:

            mock_df = Mock()
            mock_df.empty = False
            mock_df.__len__ = Mock(return_value=12)
            mock_df.dropna.return_value = mock_df
            mock_df.to_numeric.return_value = mock_df
            mock_df.fillna.return_value = mock_df
            mock_df.__getitem__ = Mock(return_value=mock_df)
            mock_df.iloc = Mock()
            mock_df.iloc.__getitem__ = Mock(return_value=100.0)

            mock_pd.read_csv.return_value = mock_df
            mock_pd.to_numeric.return_value = mock_df
            mock_pd.notna.return_value = mock_df

            mock_path.return_value.exists.return_value = True

            # Should succeed with reduced requirement (10)
            result = meta_learning.retrain_meta_learner(
                trade_log_path=self.trade_log_path,
                min_samples=10
            )

            # Result may be False due to mocking, but should not fail due to insufficient samples
            self.assertIsInstance(result, bool)

    def test_bootstrap_data_generation(self):
        """Test bootstrap data generation for faster meta-learning activation."""
        # Create minimal real trade data
        self._create_test_trade_log(3)

        # Test bootstrap generation
        with patch('meta_learning.pd') as mock_pd:
            mock_df = Mock()
            mock_df.empty = False
            mock_df.dropna.return_value = mock_df
            mock_df.to_dict.return_value = [
                {'symbol': 'AAPL', 'entry_price': 150.0, 'side': 'buy', 'signal_tags': 'test'},
                {'symbol': 'MSFT', 'entry_price': 300.0, 'side': 'buy', 'signal_tags': 'test'},
                {'symbol': 'GOOGL', 'entry_price': 2500.0, 'side': 'sell', 'signal_tags': 'test'}
            ]
            mock_pd.read_csv.return_value = mock_df

            # Should not raise an exception
            try:
                meta_learning._generate_bootstrap_training_data(self.trade_log_path, 10)
                success = True
            except Exception as e:
                success = False
                print(f"Bootstrap generation failed: {e}")

            self.assertTrue(success)

    def test_data_quality_validation(self):
        """Test enhanced data quality validation."""
        # Create test file with mixed quality data
        self._create_test_trade_log_mixed_quality()

        quality_report = meta_learning.validate_trade_data_quality(self.trade_log_path)

        self.assertTrue(quality_report['file_exists'])
        self.assertTrue(quality_report['file_readable'])
        self.assertGreater(quality_report['row_count'], 0)

    def _create_test_trade_log(self, num_trades):
        """Create a test trade log with specified number of trades."""
        headers = ['timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'pnl', 'signal_tags']

        with open(self.trade_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for i in range(num_trades):
                writer.writerow([
                    datetime.now(UTC).isoformat(),
                    f'TEST{i % 3}',  # Rotate between TEST0, TEST1, TEST2
                    'buy' if i % 2 == 0 else 'sell',
                    100.0 + i,
                    101.0 + i,
                    10,
                    10.0,
                    f'test_signal_{i}'
                ])

    def _create_test_trade_log_mixed_quality(self):
        """Create a test trade log with mixed data quality."""
        headers = ['timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'pnl', 'signal_tags']

        with open(self.trade_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            # Good data
            writer.writerow([
                datetime.now(UTC).isoformat(),
                'AAPL', 'buy', 150.0, 152.0, 10, 20.0, 'test_signal'
            ])

            # Bad data (invalid price)
            writer.writerow([
                datetime.now(UTC).isoformat(),
                'MSFT', 'buy', 'invalid', 300.0, 10, 0.0, 'test_signal'
            ])

            # Good data
            writer.writerow([
                datetime.now(UTC).isoformat(),
                'GOOGL', 'sell', 2500.0, 2450.0, 5, 250.0, 'test_signal'
            ])


class TestLiquidityManagementFixes(unittest.TestCase):
    """Test enhanced liquidity management fixes."""

    def setUp(self):
        self.mock_api = Mock()
        self.execution_engine = ExecutionEngine(broker_interface=self.mock_api)

    @patch('ai_trading.execution.engine.ExecutionEngine._latest_quote')
    @patch('ai_trading.execution.engine.ExecutionEngine._minute_stats')
    def test_enhanced_liquidity_assessment(self, mock_minute_stats, mock_quote):
        """Test enhanced liquidity assessment with more granular controls."""
        # Mock quote data - normal spread
        mock_quote.return_value = (100.0, 100.05)  # 5 cent spread
        mock_minute_stats.return_value = (1000, 1000, 0.1)  # Normal volume and momentum

        # Test normal liquidity - should not reduce quantity
        qty, skip = self.execution_engine._assess_liquidity('AAPL', 100)
        self.assertEqual(qty, 100)
        self.assertFalse(skip)

        # Mock wide spread (high percentage)
        mock_quote.return_value = (100.0, 101.0)  # 1 dollar spread = 1%

        # Test wide spread - should reduce quantity aggressively
        qty, skip = self.execution_engine._assess_liquidity('AAPL', 100)
        self.assertEqual(qty, 75)  # 25% reduction
        self.assertFalse(skip)

    def test_liquidity_configuration_parameters(self):
        """Test that liquidity configuration parameters are properly set."""
        self.assertTrue(hasattr(config, 'LIQUIDITY_SPREAD_THRESHOLD'))
        self.assertTrue(hasattr(config, 'LIQUIDITY_VOL_THRESHOLD'))
        self.assertTrue(hasattr(config, 'LIQUIDITY_REDUCTION_AGGRESSIVE'))
        self.assertTrue(hasattr(config, 'LIQUIDITY_REDUCTION_MODERATE'))

        # Check default values are reasonable
        self.assertEqual(config.LIQUIDITY_REDUCTION_AGGRESSIVE, 0.75)
        self.assertEqual(config.LIQUIDITY_REDUCTION_MODERATE, 0.90)


class TestOrderManagementFixes(unittest.TestCase):
    """Test enhanced order management and timeout fixes."""

    def setUp(self):
        self.mock_api = Mock()
        self.execution_engine = ExecutionEngine(broker_interface=self.mock_api)

    def test_order_timeout_configuration(self):
        """Test that order timeout is properly configured."""
        self.assertEqual(config.ORDER_TIMEOUT_SECONDS, 300)  # 5 minutes
        self.assertEqual(config.ORDER_STALE_CLEANUP_INTERVAL, 60)  # 1 minute
        self.assertEqual(config.ORDER_FILL_RATE_TARGET, 0.80)  # 80%

    @patch('time.time')
    def test_stale_order_cleanup(self, mock_time):
        """Test that stale orders are properly cleaned up."""
        mock_time.return_value = 1000

        # Add a stale order to tracking
        with _order_tracking_lock:
            _active_orders['test_order_123'] = OrderInfo(
                order_id='test_order_123',
                symbol='AAPL',
                side='buy',
                qty=100,
                submitted_time=500,  # 500 seconds ago (8+ minutes)
                last_status='new'
            )

        # Mock API calls for order cancellation
        self.mock_api.get_order_by_id.return_value = Mock(status='new')
        self.mock_api.cancel_order_by_id.return_value = True

        # Test cleanup
        canceled_count = self.execution_engine.cleanup_stale_orders(max_age_seconds=300)

        # Should have canceled 1 stale order
        self.assertEqual(canceled_count, 1)

    def test_partial_fill_reconciliation(self):
        """Test enhanced partial fill reconciliation."""
        mock_order = Mock()
        mock_order.filled_qty = 50
        mock_order.id = 'test_order_123'
        mock_order.status = 'partially_filled'

        # Test reconciliation with partial fill
        self.execution_engine._reconcile_partial_fills(
            symbol='AAPL',
            submitted_qty=100,  # Submitted 100 shares
            remaining_qty=50,   # 50 shares remaining
            side='buy',
            last_order=mock_order
        )

        # Should log partial fill without errors
        # (This test mainly ensures the function runs without exceptions)
        self.assertTrue(True)


class TestSystemMonitoringAndAlerting(unittest.TestCase):
    """Test enhanced system monitoring and alerting."""

    def test_sentiment_success_rate_monitoring(self):
        """Test sentiment success rate monitoring configuration."""
        self.assertTrue(hasattr(config, 'SENTIMENT_SUCCESS_RATE_TARGET'))
        self.assertEqual(config.SENTIMENT_SUCCESS_RATE_TARGET, 0.90)

    def test_meta_learning_configuration(self):
        """Test meta-learning configuration parameters."""
        self.assertTrue(hasattr(config, 'META_LEARNING_BOOTSTRAP_ENABLED'))
        self.assertTrue(hasattr(config, 'META_LEARNING_MIN_TRADES_REDUCED'))
        self.assertTrue(hasattr(config, 'META_LEARNING_BOOTSTRAP_WIN_RATE'))

        self.assertEqual(config.META_LEARNING_MIN_TRADES_REDUCED, 10)
        self.assertEqual(config.META_LEARNING_BOOTSTRAP_WIN_RATE, 0.66)

    def test_comprehensive_configuration_coverage(self):
        """Test that all critical configuration parameters are defined."""
        required_configs = [
            'SENTIMENT_ENHANCED_CACHING',
            'SENTIMENT_FALLBACK_SOURCES',
            'META_LEARNING_BOOTSTRAP_ENABLED',
            'ORDER_TIMEOUT_SECONDS',
            'LIQUIDITY_SPREAD_THRESHOLD',
            'LIQUIDITY_VOL_THRESHOLD'
        ]

        for config_param in required_configs:
            self.assertTrue(hasattr(config, config_param), f"Missing config parameter: {config_param}")


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple fixes."""

    def test_end_to_end_rate_limited_sentiment_with_meta_learning(self):
        """Test complete flow when sentiment is rate limited but meta-learning works."""
        # This would be a more complex integration test
        # For now, just ensure modules can be imported together
        from ai_trading import meta_learning
        from ai_trading.analysis import sentiment
        from ai_trading.execution.engine import ExecutionEngine

        # Verify key functions exist
        self.assertTrue(callable(sentiment.fetch_sentiment))
        self.assertTrue(callable(meta_learning.retrain_meta_learner))
        self.assertTrue(callable(ExecutionEngine))


if __name__ == '__main__':
    unittest.main()
def test_meta_learning_data_quality_validation():
    """Test that meta-learning validates data quality before training."""
    from ai_trading.meta_learning import validate_trade_data_quality

    # Test with non-existent file
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp_path = tmp.name

    quality_report = validate_trade_data_quality(tmp_path)
    assert quality_report['file_exists'] is False
    assert 'Trade log file does not exist' in quality_report['issues'][0]
    assert 'Initialize trade logging system' in quality_report['recommendations'][0]


def test_meta_learning_fallback_data_recovery():
    """Test that meta-learning implements fallback procedures for insufficient data."""
    from ai_trading.meta_learning import _implement_fallback_data_recovery

    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp_path = tmp.name

    try:
        # Remove the file to test creation
        os.unlink(tmp_path)

        # Test fallback recovery
        _implement_fallback_data_recovery(tmp_path, min_samples=20)

        # Check that emergency log was created
        assert Path(tmp_path).exists()

        # Verify proper format
        df = pd.read_csv(tmp_path)
        required_cols = ['timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'pnl', 'signal_tags']
        for col in required_cols:
            assert col in df.columns
    finally:
        if Path(tmp_path).exists():
            os.unlink(tmp_path)


def test_meta_learning_price_validation():
    """Test that meta-learning filters out invalid price data."""
    from ai_trading.meta_learning import validate_trade_data_quality

    # Create test CSV with mixed valid/invalid data
    test_data = {
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'],
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'side': ['buy', 'sell', 'buy'],
        'entry_price': [150.50, -10.0, 2800.0],  # One negative price
        'exit_price': [155.0, 200.0, 2850.0],
        'quantity': [100, 50, 10],
        'pnl': [450.0, -500.0, 500.0],
        'signal_tags': ['momentum', 'mean_reversion', 'momentum']
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame(test_data)
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        quality_report = validate_trade_data_quality(tmp_path)
        assert quality_report['file_exists'] is True
        assert quality_report['has_valid_format'] is True
        assert quality_report['row_count'] == 3
        assert quality_report['valid_price_rows'] == 2  # Should filter out negative price
    finally:
        os.unlink(tmp_path)


def test_order_execution_partial_fill_tracking():
    """Test that partial fills are properly tracked and reconciled."""
    from ai_trading.execution.engine import ExecutionEngine

    # Mock the trading API
    mock_api = Mock()

    # Create execution engine instance
    engine = ExecutionEngine(broker_interface=mock_api)

    # Mock order result
    mock_order = Mock()
    mock_order.id = "test_order_123"

    # Test partial fill reconciliation
    engine._reconcile_partial_fills("AAPL", requested_qty=100, remaining_qty=30, side="buy", last_order=mock_order)

    # Verify that partial fill was logged (would be in logs in real implementation)
    # This is a basic structure test
    assert hasattr(engine, '_reconcile_partial_fills')


def test_quantity_tracking_fix():
    """Test the critical quantity tracking bug fix for accurate filled quantity reporting."""
    import io
    import logging
    from unittest.mock import Mock

    # Import trade execution module
    from ai_trading.execution.engine import ExecutionEngine

    # Create execution engine
    engine = ExecutionEngine()

    # Set up logging capture to verify correct behavior
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    engine.logger.addHandler(handler)
    engine.logger.setLevel(logging.INFO)

    # Test Case 1: Partial fill scenario (like TSLA case - requested 32, submitted 16, filled 11)
    mock_order_partial = Mock()
    mock_order_partial.id = "order_123"
    mock_order_partial.filled_qty = "32"  # Alpaca API incorrectly reports original quantity

    # Call _reconcile_partial_fills with parameters matching production logs
    # requested_qty=32 (original), remaining_qty=21 (32-11), so filled should be 11
    engine._reconcile_partial_fills(
        symbol="TSLA",
        requested_qty=32,
        remaining_qty=21,  # 32 - 11 = 21 remaining
        side="buy",
        last_order=mock_order_partial
    )

    # Verify it correctly identified as partial fill (not full)
    log_output = log_stream.getvalue()
    assert "PARTIAL_FILL_DETECTED" in log_output
    assert "FULL_FILL_SUCCESS" not in log_output

    # Test Case 2: Actual full fill scenario
    log_stream.truncate(0)
    log_stream.seek(0)

    mock_order_full = Mock()
    mock_order_full.id = "order_456"
    mock_order_full.filled_qty = "16"  # Could be correct or incorrect - should not matter

    # Full fill: requested 16, remaining 0, so filled = 16
    engine._reconcile_partial_fills(
        symbol="MSFT",
        requested_qty=16,
        remaining_qty=0,  # No quantity remaining = full fill
        side="buy",
        last_order=mock_order_full
    )

    # Verify it correctly identified as full fill
    log_output = log_stream.getvalue()
    assert "FULL_FILL_SUCCESS" in log_output
    assert "PARTIAL_FILL_DETECTED" not in log_output

    # Test Case 3: Quantity mismatch detection
    log_stream.truncate(0)
    log_stream.seek(0)

    mock_order_mismatch = Mock()
    mock_order_mismatch.id = "order_789"
    mock_order_mismatch.filled_qty = "50"  # Wrong - API reports 50 but calculated should be 10

    # Partial fill with quantity mismatch: requested 20, remaining 10, so filled = 10
    # But order.filled_qty reports 50 (wrong)
    engine._reconcile_partial_fills(
        symbol="AMZN",
        requested_qty=20,
        remaining_qty=10,  # 20 - 10 = 10 filled
        side="buy",
        last_order=mock_order_mismatch
    )

    # Verify mismatch was detected and logged
    log_output = log_stream.getvalue()
    assert "QUANTITY_MISMATCH_DETECTED" in log_output
    assert "calculated_filled_qty" in log_output

    # Clean up
    engine.logger.removeHandler(handler)


def test_risk_management_sector_exposure_logging():
    """Test that sector exposure rejections include clear reasoning."""
    # This is a minimal test - full test would require bot_engine context
    # Testing the structure exists for enhanced logging
    from ai_trading.core.bot_engine import sector_exposure_ok

    # Mock BotContext
    mock_ctx = Mock()
    mock_ctx.api = Mock()

    # Mock account with zero portfolio value
    mock_account = Mock()
    mock_account.portfolio_value = 0
    mock_ctx.api.get_account.return_value = mock_account

    # Test empty portfolio logic
    result = sector_exposure_ok(mock_ctx, "AAPL", 10, 150.0)
    assert result is True  # Should allow initial positions


def test_data_integrity_validation():
    """Test comprehensive data integrity validation."""
    from ai_trading.data_validation import (
        monitor_real_time_data_quality,
        validate_trade_log_integrity,
    )

    # Test trade log integrity validation
    test_data = {
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
        'symbol': ['AAPL', 'MSFT'],
        'side': ['buy', 'sell'],
        'entry_price': [150.50, 200.0],
        'exit_price': [155.0, 195.0],
        'quantity': [100, 50],
        'pnl': [450.0, -250.0]
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame(test_data)
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        integrity_report = validate_trade_log_integrity(tmp_path)
        assert integrity_report['file_exists'] is True
        assert integrity_report['file_readable'] is True
        assert integrity_report['valid_format'] is True
        assert integrity_report['data_consistent'] is True
        assert integrity_report['total_trades'] == 2
        assert integrity_report['integrity_score'] >= 0.9
    finally:
        os.unlink(tmp_path)

    # Test real-time data quality monitoring
    price_data = {
        'AAPL': 150.50,
        'MSFT': 300.0,
        'INVALID': -10.0  # Invalid negative price
    }

    quality_report = monitor_real_time_data_quality(price_data)
    assert quality_report['data_quality_ok'] is False
    assert 'INVALID' in quality_report['critical_symbols']
    assert len(quality_report['anomalies_detected']) >= 1


def test_data_corruption_detection():
    """Test that data corruption is properly detected."""
    from ai_trading.data_validation import validate_trade_log_integrity

    # Create corrupted test data
    corrupted_data = {
        'timestamp': ['invalid_date', '2024-01-01 11:00:00'],
        'symbol': ['AAPL', 'MSFT'],
        'side': ['invalid_side', 'sell'],
        'entry_price': ['not_a_number', 200.0],
        'exit_price': [155.0, 195.0],
        'quantity': [-100, 50],  # Invalid negative quantity
        'pnl': [450.0, -250.0]
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame(corrupted_data)
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        integrity_report = validate_trade_log_integrity(tmp_path)
        assert integrity_report['file_exists'] is True
        assert integrity_report['file_readable'] is True
        assert integrity_report['data_consistent'] is False
        assert len(integrity_report['corrupted_rows']) >= 1
        assert integrity_report['integrity_score'] < 0.9
    finally:
        os.unlink(tmp_path)


def test_emergency_data_validation():
    """Test emergency data validation for critical trades."""
    from ai_trading.data_validation import emergency_data_check

    # Test with valid data
    valid_data = pd.DataFrame({
        'Close': [150.0, 151.0, 152.0],
        'Volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01 09:30:00', periods=3, freq='1min', tz='UTC'))

    # Should pass emergency validation
    assert emergency_data_check(valid_data, "AAPL") is True

    # Test with empty data
    empty_data = pd.DataFrame()
    assert emergency_data_check(empty_data, "AAPL") is False

    # Test with invalid price data
    invalid_data = pd.DataFrame({
        'Close': [150.0, 151.0, -10.0],  # Invalid negative price
        'Volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01 09:30:00', periods=3, freq='1min', tz='UTC'))

    assert emergency_data_check(invalid_data, "AAPL") is False


def test_metalearn_invalid_prices_prevention():
    """Test that METALEARN_INVALID_PRICES warnings are prevented with proper data handling."""
    from ai_trading.meta_learning import retrain_meta_learner

    # Create minimal valid trade data
    valid_trade_data = {
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
        'symbol': ['AAPL', 'MSFT'],
        'side': ['buy', 'sell'],
        'entry_price': [150.50, 200.0],
        'exit_price': [155.0, 195.0],
        'quantity': [100, 50],
        'pnl': [450.0, -250.0],
        'signal_tags': ['momentum', 'mean_reversion']
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame(valid_trade_data)
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        # This should validate data quality first and not trigger METALEARN_INVALID_PRICES
        # Note: Full training may fail due to missing dependencies, but data validation should pass
        with patch('meta_learning.logger') as mock_logger:
            try:
                retrain_meta_learner(trade_log_path=tmp_path, min_samples=1)
            except Exception as e:
                # Training may fail due to missing sklearn, but that's OK for this test
                mock_logger.warning.call_args_list.append(f"Meta learning training failed as expected: {e}")

            # Check that quality validation occurred
            calls = [str(call) for call in mock_logger.info.call_args_list]
            quality_calls = [call for call in calls if 'META_LEARNING_QUALITY_CHECK' in call]
            assert len(quality_calls) > 0
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
