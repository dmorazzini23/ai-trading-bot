"""Test fixes for message-shortening ellipsis and risk exposure task."""
import json
import logging
import unittest
from unittest.mock import Mock, patch

import ai_trading.logging as logger_module
from ai_trading.core.bot_engine import _update_risk_engine_exposure, _get_runtime_context_or_none


def _make_record(**extra):
    """Create a test log record."""
    rec = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="MARKET WATCH — Real Alpaca Trading SDK imported successfully",
        args=None,
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


class TestEllipsisFix(unittest.TestCase):
    """Test cases for the ellipsis and risk exposure fixes."""

    def test_json_formatter_unicode_ensure_ascii_false(self):
        """Test that JSON formatter preserves Unicode characters without escaping."""
        fmt = logger_module.JSONFormatter("%(asctime)sZ")
        
        # Create a record with Unicode ellipsis character
        rec = _make_record()
        rec.getMessage = lambda: "MARKET WATCH — Real Alpaca Trading SDK imported successfully"
        
        out = fmt.format(rec)
        data = json.loads(out)
        
        # Verify that the Unicode em dash (—) is preserved as-is, not escaped as \u2014
        self.assertIn("—", data["msg"])
        self.assertNotIn("\\u2014", out)  # Should not contain escaped Unicode
        self.assertEqual(data["msg"], "MARKET WATCH — Real Alpaca Trading SDK imported successfully")

    def test_json_formatter_log_trading_event_unicode(self):
        """Test that log_trading_event preserves Unicode without escaping."""
        # Test the log_trading_event function which also uses json.dumps
        with patch('ai_trading.logging.logging.getLogger') as mock_logger_get:
            mock_logger = Mock()
            mock_logger_get.return_value = mock_logger
            
            # Call log_trading_event with Unicode characters
            logger_module.log_trading_event(
                'TRADE_EXECUTED',
                'AAPL',
                {'side': 'buy', 'notes': 'Market analysis shows — positive trend'}
            )
            
            # Verify that info was called
            self.assertTrue(mock_logger.info.called)
            
            # Get the logged message and verify Unicode is preserved
            call_args = mock_logger.info.call_args
            logged_message = call_args[0][1]  # Second argument to info()
            
            # Verify that Unicode is preserved in the JSON string
            self.assertIn("—", logged_message)
            self.assertNotIn("\\u2014", logged_message)

    def test_get_runtime_context_or_none(self):
        """Test the runtime context accessor function."""
        with patch('ai_trading.core.bot_engine.get_ctx') as mock_get_ctx:
            # Test successful context retrieval
            mock_context = Mock()
            mock_lazy_ctx = Mock()
            mock_lazy_ctx._ensure_initialized.return_value = None
            mock_lazy_ctx._context = mock_context
            mock_get_ctx.return_value = mock_lazy_ctx
            
            result = _get_runtime_context_or_none()
            
            self.assertIs(result, mock_context)
            mock_lazy_ctx._ensure_initialized.assert_called_once()

    def test_get_runtime_context_or_none_error(self):
        """Test runtime context accessor handles errors gracefully."""
        with patch('ai_trading.core.bot_engine.get_ctx') as mock_get_ctx:
            with patch('ai_trading.core.bot_engine._log') as mock_log:
                # Test error handling
                mock_get_ctx.side_effect = Exception("Context unavailable")
                
                result = _get_runtime_context_or_none()
                
                self.assertIsNone(result)
                mock_log.warning.assert_called_once()
                self.assertIn("Context unavailable", str(mock_log.warning.call_args))

    def test_update_risk_engine_exposure_no_context(self):
        """Test risk exposure update handles missing context gracefully."""
        with patch('ai_trading.core.bot_engine._get_runtime_context_or_none') as mock_get_ctx:
            mock_get_ctx.return_value = None
            
            # Should not raise, just return quietly
            _update_risk_engine_exposure()

    def test_update_risk_engine_exposure_with_context(self):
        """Test risk exposure update works with valid context."""
        with patch('ai_trading.core.bot_engine._get_runtime_context_or_none') as mock_get_ctx:
            with patch('ai_trading.core.bot_engine._log') as mock_log:
                # Setup mock context with risk engine
                mock_context = Mock()
                mock_risk_engine = Mock()
                mock_context.risk_engine = mock_risk_engine
                mock_get_ctx.return_value = mock_context
                
                _update_risk_engine_exposure()
                
                # Verify risk engine update_exposure was called
                mock_risk_engine.update_exposure.assert_called_once_with(mock_context)

    def test_update_risk_engine_exposure_no_risk_engine(self):
        """Test risk exposure update handles missing risk engine gracefully."""
        with patch('ai_trading.core.bot_engine._get_runtime_context_or_none') as mock_get_ctx:
            with patch('ai_trading.core.bot_engine._log') as mock_log:
                # Setup mock context without risk engine
                mock_context = Mock()
                mock_context.risk_engine = None
                mock_get_ctx.return_value = mock_context
                
                _update_risk_engine_exposure()
                
                # Should log debug message about missing risk engine
                mock_log.debug.assert_called_once()
                self.assertIn("No risk_engine", str(mock_log.debug.call_args))

    def test_update_risk_engine_exposure_error(self):
        """Test risk exposure update handles errors gracefully."""
        with patch('ai_trading.core.bot_engine._get_runtime_context_or_none') as mock_get_ctx:
            with patch('ai_trading.core.bot_engine._log') as mock_log:
                # Setup mock context with failing risk engine
                mock_context = Mock()
                mock_risk_engine = Mock()
                mock_risk_engine.update_exposure.side_effect = Exception("Update failed")
                mock_context.risk_engine = mock_risk_engine
                mock_get_ctx.return_value = mock_context
                
                _update_risk_engine_exposure()
                
                # Should log warning about failure
                mock_log.warning.assert_called_once()
                self.assertIn("Risk engine exposure update failed", str(mock_log.warning.call_args))


if __name__ == '__main__':
    unittest.main()