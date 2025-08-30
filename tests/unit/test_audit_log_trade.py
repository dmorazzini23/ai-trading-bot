from unittest.mock import patch

from ai_trading import audit


def test_log_trade_permission_error_triggers_fix():
    """log_trade should attempt to fix permissions when a PermissionError occurs."""
    with patch("ai_trading.audit.open", side_effect=PermissionError), \
         patch("ai_trading.audit.fix_file_permissions", return_value=True) as mock_fix:
        audit.log_trade("AAPL", 1, "buy", 100.0, timestamp="2024-01-01")
        mock_fix.assert_called_once()
