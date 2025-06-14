import pytest
from unittest.mock import patch, MagicMock


def test_bot_main_normal(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "shadow")
    with patch("data_fetcher.get_minute_df", return_value=MagicMock()), \
         patch("alpaca_api.submit_order", return_value={"status": "mocked"}), \
         patch("signals.generate", return_value=1), \
         patch("risk_engine.calculate_position_size", return_value=10):
        import bot
        assert bot.main() is None or bot.main() is True


def test_bot_main_data_fetch_error(monkeypatch):
    with patch("data_fetcher.get_minute_df", side_effect=Exception("API error")):
        import bot
        with pytest.raises(Exception):
            bot.main()


def test_bot_main_signal_nan(monkeypatch):
    with patch("signals.generate", return_value=float('nan')), \
         patch("data_fetcher.get_minute_df", return_value=MagicMock()):
        import bot
        # Expect bot to handle or skip NaN signal, not crash
        try:
            bot.main()
        except Exception:
            pytest.fail("Bot should handle NaN signal gracefully")


def test_trade_execution_api_timeout(monkeypatch):
    with patch("alpaca_api.submit_order", side_effect=TimeoutError("Timeout")), \
         patch("trade_execution.log_order") as mock_log:
        import trade_execution
        with pytest.raises(TimeoutError):
            trade_execution.place_order("AAPL", 5, "buy")
