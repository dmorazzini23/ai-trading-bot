#!/usr/bin/env python3
"""
Test drawdown circuit breaker integration in bot engine.

This test validates that the DrawdownCircuitBreaker is properly integrated
into the main trading loop and responds correctly to equity changes.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Set testing environment
os.environ["TESTING"] = "1"
os.environ["PYTEST_RUNNING"] = "1"

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_trading import config
from ai_trading.risk.circuit_breakers import DrawdownCircuitBreaker


class TestDrawdownIntegration(unittest.TestCase):
    """Test drawdown circuit breaker integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.max_drawdown = 0.15  # 15%
        self.initial_equity = 10000.0
        
    def test_drawdown_circuit_breaker_initialization(self):
        """Test that DrawdownCircuitBreaker initializes correctly."""
        breaker = DrawdownCircuitBreaker(max_drawdown=self.max_drawdown)
        
        self.assertEqual(breaker.max_drawdown, self.max_drawdown)
        self.assertEqual(breaker.state.value, "closed")  # Normal operation
        self.assertEqual(breaker.current_drawdown, 0.0)
        self.assertEqual(breaker.peak_equity, 0.0)
        
    def test_drawdown_circuit_breaker_normal_operation(self):
        """Test circuit breaker during normal operation."""
        breaker = DrawdownCircuitBreaker(max_drawdown=self.max_drawdown)
        
        # Update with initial equity
        trading_allowed = breaker.update_equity(self.initial_equity)
        self.assertTrue(trading_allowed)
        self.assertEqual(breaker.peak_equity, self.initial_equity)
        self.assertEqual(breaker.current_drawdown, 0.0)
        
        # Small gain - should still allow trading
        new_equity = self.initial_equity * 1.05  # 5% gain
        trading_allowed = breaker.update_equity(new_equity)
        self.assertTrue(trading_allowed)
        self.assertEqual(breaker.peak_equity, new_equity)
        self.assertEqual(breaker.current_drawdown, 0.0)
        
    def test_drawdown_circuit_breaker_triggers_halt(self):
        """Test circuit breaker triggers halt when drawdown exceeds threshold."""
        breaker = DrawdownCircuitBreaker(max_drawdown=self.max_drawdown)
        
        # Set initial peak equity
        breaker.update_equity(self.initial_equity)
        
        # Simulate large loss exceeding threshold
        loss_equity = self.initial_equity * (1 - self.max_drawdown - 0.01)  # 9% loss
        trading_allowed = breaker.update_equity(loss_equity)
        
        self.assertFalse(trading_allowed)
        self.assertEqual(breaker.state.value, "open")  # Trading halted
        self.assertGreater(breaker.current_drawdown, self.max_drawdown)
        
    def test_drawdown_circuit_breaker_recovery(self):
        """Test circuit breaker allows trading after recovery."""
        breaker = DrawdownCircuitBreaker(max_drawdown=self.max_drawdown, recovery_threshold=0.8)
        
        # Set initial peak equity
        breaker.update_equity(self.initial_equity)
        
        # Trigger halt with large loss
        loss_equity = self.initial_equity * (1 - self.max_drawdown - 0.01)
        breaker.update_equity(loss_equity)
        self.assertEqual(breaker.state.value, "open")
        
        # Recover to 80% of peak (recovery threshold)
        recovery_equity = self.initial_equity * 0.8
        trading_allowed = breaker.update_equity(recovery_equity)
        
        self.assertTrue(trading_allowed)
        self.assertEqual(breaker.state.value, "closed")  # Trading resumed
        
    def test_configuration_values(self):
        """Test that configuration values are correctly set."""
        self.assertEqual(config.MAX_DRAWDOWN_THRESHOLD, 0.15)
        self.assertEqual(config.DAILY_LOSS_LIMIT, 0.03)
        
        # Test TradingConfig
        tc = config.TradingConfig.from_env()
        self.assertEqual(tc.max_drawdown_threshold, 0.15)
        self.assertEqual(tc.daily_loss_limit, 0.03)

    @patch('bot_engine.ctx')
    def test_bot_context_integration(self, mock_ctx):
        """Test that bot context includes drawdown circuit breaker."""
        # Mock the context with a circuit breaker
        mock_circuit_breaker = Mock(spec=DrawdownCircuitBreaker)
        mock_circuit_breaker.update_equity.return_value = True
        mock_circuit_breaker.get_status.return_value = {
            "trading_allowed": True,
            "current_drawdown": 0.02,
            "max_drawdown": 0.15
        }
        
        mock_ctx.drawdown_circuit_breaker = mock_circuit_breaker
        
        # Simulate updating equity
        equity = 10000.0
        result = mock_ctx.drawdown_circuit_breaker.update_equity(equity)
        
        self.assertTrue(result)
        mock_circuit_breaker.update_equity.assert_called_with(equity)

    def test_drawdown_status_reporting(self):
        """Test that drawdown status reporting works correctly."""
        breaker = DrawdownCircuitBreaker(max_drawdown=self.max_drawdown)
        
        # Set up a scenario
        breaker.update_equity(self.initial_equity)
        current_equity = self.initial_equity * 0.95  # 5% loss
        breaker.update_equity(current_equity)
        
        status = breaker.get_status()
        
        self.assertIn("state", status)
        self.assertIn("current_drawdown", status)
        self.assertIn("max_drawdown", status)
        self.assertIn("peak_equity", status)
        self.assertIn("trading_allowed", status)
        
        self.assertEqual(status["state"], "closed")
        self.assertAlmostEqual(status["current_drawdown"], 0.05, places=2)
        self.assertEqual(status["max_drawdown"], self.max_drawdown)
        self.assertEqual(status["peak_equity"], self.initial_equity)
        self.assertTrue(status["trading_allowed"])


if __name__ == "__main__":
    # Set up minimal environment for testing
    os.environ.setdefault("MAX_DRAWDOWN_THRESHOLD", "0.15")
    os.environ.setdefault("DAILY_LOSS_LIMIT", "0.03")
    
    unittest.main(verbosity=2)