#!/usr/bin/env python3
"""
Test suite for Phase 2 enhancements: Order Management and System Health Monitoring.

Tests the enhanced order management and comprehensive system health monitoring.
"""

import unittest
import os
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import pytest

order_health_monitor = pytest.importorskip("order_health_monitor")
system_health_checker = pytest.importorskip("system_health_checker")
from order_health_monitor import OrderHealthMonitor, OrderInfo
from system_health_checker import SystemHealthChecker, ComponentHealth

# Mock environment variables for testing
test_env = {
    'ALPACA_API_KEY': 'test_key',
    'ALPACA_SECRET_KEY': 'test_secret',
    'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets',
    'WEBHOOK_SECRET': 'test_secret',
    'FLASK_PORT': '5000',
    'NEWS_API_KEY': 'test_news_key',
    'TRADE_LOG_FILE': 'test_trades.csv'
}

with patch.dict(os.environ, test_env):
    from ai_trading import config


class TestOrderHealthMonitor(unittest.TestCase):
    """Test order health monitoring system."""
    
    def setUp(self):
        self.monitor = OrderHealthMonitor()
        self.monitor.order_timeout_seconds = 10  # Short timeout for testing
        self.monitor.cleanup_interval = 1  # Short interval for testing
    
    def test_order_health_monitor_initialization(self):
        """Test that order health monitor initializes properly."""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.order_timeout_seconds, 10)
        self.assertEqual(self.monitor.cleanup_interval, 1)
        self.assertFalse(self.monitor._monitoring_active)
    
    def test_partial_fill_recording(self):
        """Test recording and tracking of partial fills."""
        # Record a partial fill
        self.monitor.record_order_fill(
            order_id="test_order_123",
            fill_time=1.5,
            is_partial=True,
            fill_qty=50,
            total_qty=100
        )
        
        # Check that partial fill was recorded
        with self.monitor._lock:
            self.assertIn("test_order_123", self.monitor._partial_fills)
            partial_info = self.monitor._partial_fills["test_order_123"]
            self.assertEqual(partial_info.filled_qty, 50)
            self.assertEqual(partial_info.total_qty, 100)
            self.assertEqual(partial_info.fill_rate, 0.5)
    
    def test_health_metrics_calculation(self):
        """Test health metrics calculation."""
        # Mock some order data
        with patch('order_health_monitor._active_orders') as mock_orders, \
             patch('order_health_monitor._order_tracking_lock'):
            
            # Add some mock orders
            mock_orders.__len__ = Mock(return_value=5)
            mock_orders.values.return_value = [
                OrderInfo("order1", "AAPL", "buy", 100, time.time() - 30, "filled"),
                OrderInfo("order2", "MSFT", "buy", 50, time.time() - 60, "filled"),
                OrderInfo("order3", "GOOGL", "sell", 25, time.time() - 600, "new"),  # Stale
                OrderInfo("order4", "TSLA", "buy", 75, time.time() - 15, "partially_filled"),
                OrderInfo("order5", "NVDA", "buy", 100, time.time() - 5, "new")
            ]
            
            metrics = self.monitor._calculate_health_metrics()
            
            self.assertEqual(metrics.total_orders, 5)
            self.assertGreater(metrics.success_rate, 0.0)
            self.assertLessEqual(metrics.success_rate, 1.0)
    
    def test_health_summary_generation(self):
        """Test health summary generation.""" 
        summary = self.monitor.get_health_summary()
        
        # Check required fields
        self.assertIn("current_metrics", summary)
        self.assertIn("alerts_enabled", summary)
        self.assertIn("monitoring_active", summary)
        self.assertIn("timestamp", summary)
        
        # Check metric fields
        current_metrics = summary["current_metrics"]
        required_metrics = [
            "total_orders", "success_rate", "avg_fill_time", 
            "stuck_orders", "partial_fills", "avg_fill_rate"
        ]
        for metric in required_metrics:
            self.assertIn(metric, current_metrics)


class TestSystemHealthChecker(unittest.TestCase):
    """Test system health checking functionality."""
    
    def setUp(self):
        self.health_checker = SystemHealthChecker()
    
    def test_system_health_checker_initialization(self):
        """Test that system health checker initializes properly."""
        self.assertIsNotNone(self.health_checker)
        self.assertFalse(self.health_checker._monitoring_active)
        self.assertIn('sentiment', self.health_checker.health_thresholds)
        self.assertIn('meta_learning', self.health_checker.health_thresholds)
        self.assertIn('order_execution', self.health_checker.health_thresholds)
    
    @patch('system_health_checker.sentiment')
    def test_sentiment_health_check(self, mock_sentiment):
        """Test sentiment analysis health checking."""
        # Mock sentiment module state
        mock_sentiment._SENTIMENT_CACHE = {'AAPL': (time.time(), 0.8)}
        mock_sentiment._SENTIMENT_CIRCUIT_BREAKER = {
            'state': 'closed',
            'failures': 2,
            'last_failure': time.time() - 300
        }
        mock_sentiment.SENTIMENT_FAILURE_THRESHOLD = 15
        
        health = self.health_checker._check_sentiment_health()
        
        self.assertEqual(health.name, "sentiment")
        self.assertIn(health.status, ["healthy", "warning", "critical"])
        self.assertGreaterEqual(health.success_rate, 0.0)
        self.assertLessEqual(health.success_rate, 1.0)
    
    @patch('system_health_checker.meta_learning')
    @patch('system_health_checker.config')
    def test_meta_learning_health_check(self, mock_config, mock_meta_learning):
        """Test meta-learning health checking."""
        # Mock configuration
        mock_config.TRADE_LOG_FILE = 'test_trades.csv'
        mock_config.META_LEARNING_MIN_TRADES_REDUCED = 10
        mock_config.META_LEARNING_BOOTSTRAP_ENABLED = True
        
        # Mock validation result
        mock_meta_learning.validate_trade_data_quality.return_value = {
            'valid_price_rows': 15,
            'data_quality_score': 0.9
        }
        
        health = self.health_checker._check_meta_learning_health()
        
        self.assertEqual(health.name, "meta_learning")
        self.assertIn(health.status, ["healthy", "warning", "critical"])
        self.assertEqual(health.details['trade_count'], 15)
        self.assertEqual(health.details['min_required'], 10)
    
    def test_overall_status_determination(self):
        """Test overall status determination from components."""
        # Test all healthy
        components = {
            'comp1': ComponentHealth("comp1", "healthy", 0.9, datetime.now(timezone.utc)),
            'comp2': ComponentHealth("comp2", "healthy", 0.8, datetime.now(timezone.utc))
        }
        status = self.health_checker._determine_overall_status(components)
        self.assertEqual(status, "healthy")
        
        # Test with warning
        components['comp1'].status = "warning"
        status = self.health_checker._determine_overall_status(components)
        self.assertEqual(status, "warning")
        
        # Test with critical
        components['comp2'].status = "critical"
        status = self.health_checker._determine_overall_status(components)
        self.assertEqual(status, "critical")
    
    def test_current_health_report(self):
        """Test current health report generation."""
        with patch.object(self.health_checker, '_check_all_components') as mock_check:
            # Mock health status
            from system_health_checker import SystemHealthStatus
            mock_health = SystemHealthStatus(
                overall_status="healthy",
                components={
                    'sentiment': ComponentHealth("sentiment", "healthy", 0.9, datetime.now(timezone.utc)),
                    'meta_learning': ComponentHealth("meta_learning", "warning", 0.7, datetime.now(timezone.utc))
                },
                alerts=["Test alert"],
                metrics={'test_metric': 0.85}
            )
            mock_check.return_value = mock_health
            
            health_report = self.health_checker.get_current_health()
            
            # Check structure
            self.assertIn("overall_status", health_report)
            self.assertIn("components", health_report)
            self.assertIn("alerts", health_report)
            self.assertIn("metrics", health_report)
            self.assertIn("timestamp", health_report)
            
            # Check content
            self.assertEqual(health_report["overall_status"], "healthy")
            self.assertIn("sentiment", health_report["components"])
            self.assertIn("meta_learning", health_report["components"])
            self.assertEqual(len(health_report["alerts"]), 1)


class TestConfigurationEnhancements(unittest.TestCase):
    """Test enhanced configuration parameters."""
    
    def test_order_management_config(self):
        """Test order management configuration parameters."""
        required_params = [
            'ORDER_TIMEOUT_SECONDS',
            'ORDER_STALE_CLEANUP_INTERVAL', 
            'ORDER_FILL_RATE_TARGET',
            'ORDER_MAX_RETRY_ATTEMPTS'
        ]
        
        for param in required_params:
            self.assertTrue(hasattr(config, param), f"Missing parameter: {param}")
        
        # Test reasonable defaults
        self.assertEqual(config.ORDER_TIMEOUT_SECONDS, 300)  # 5 minutes
        self.assertEqual(config.ORDER_STALE_CLEANUP_INTERVAL, 60)  # 1 minute
        self.assertEqual(config.ORDER_FILL_RATE_TARGET, 0.80)  # 80%
        self.assertEqual(config.ORDER_MAX_RETRY_ATTEMPTS, 3)
    
    def test_system_health_config(self):
        """Test system health monitoring configuration parameters."""
        required_params = [
            'SYSTEM_HEALTH_CHECK_INTERVAL',
            'SYSTEM_HEALTH_ALERT_THRESHOLD',
            'SYSTEM_HEALTH_EXPORT_ENABLED',
            'SYSTEM_HEALTH_REPORT_PATH'
        ]
        
        for param in required_params:
            self.assertTrue(hasattr(config, param), f"Missing parameter: {param}")
        
        # Test reasonable defaults
        self.assertEqual(config.SYSTEM_HEALTH_CHECK_INTERVAL, 60)  # 1 minute
        self.assertEqual(config.SYSTEM_HEALTH_ALERT_THRESHOLD, 0.70)  # 70%
        self.assertTrue(config.SYSTEM_HEALTH_EXPORT_ENABLED)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios for enhanced monitoring."""
    
    def test_monitoring_integration(self):
        """Test that monitoring systems can work together."""
        # This test ensures the monitoring systems can be integrated
        order_monitor = OrderHealthMonitor()
        health_checker = SystemHealthChecker()
        
        # Both should be able to coexist
        self.assertIsNotNone(order_monitor)
        self.assertIsNotNone(health_checker)
        
        # Test that they can generate reports without interfering
        order_summary = order_monitor.get_health_summary()
        health_report = health_checker.get_current_health()
        
        self.assertIsInstance(order_summary, dict)
        self.assertIsInstance(health_report, dict)
    
    def test_phase1_and_phase2_compatibility(self):
        """Test that Phase 1 and Phase 2 fixes work together."""
        # Import Phase 1 modules to ensure compatibility
        with patch.dict(os.environ, test_env):
            
            # Test that enhanced sentiment analysis works with monitoring
            health_checker = SystemHealthChecker()
            sentiment_health = health_checker._check_sentiment_health()
            
            self.assertIsNotNone(sentiment_health)
            self.assertEqual(sentiment_health.name, "sentiment")
            
            # Test that meta-learning monitoring works
            meta_health = health_checker._check_meta_learning_health()
            
            self.assertIsNotNone(meta_health)
            self.assertEqual(meta_health.name, "meta_learning")


if __name__ == '__main__':
    # Run tests with minimal output
    unittest.main(verbosity=2)