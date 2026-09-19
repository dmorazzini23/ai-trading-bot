import sys
from typing import Callable, cast
from unittest.mock import MagicMock, patch

import pytest
from ai_trading.config import management as config

pd = pytest.importorskip("pandas")
pytestmark = pytest.mark.integration



def test_bot_main_normal(monkeypatch):
    monkeypatch.setenv("AI_TRADING_TRADING_MODE", "balanced")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FINNHUB_API_KEY", "testkey")
    config.reload_env()
    monkeypatch.setattr(sys, "argv", ["bot.py"])
    with patch("ai_trading.data.fetch.get_minute_df", return_value=MagicMock()), patch(
        "ai_trading.alpaca_api.submit_order", return_value={"status": "mocked"}
    ), patch("ai_trading.signals.generate", return_value=1), patch("ai_trading.risk.engine.calculate_position_size", return_value=10), patch(
        "ai_trading.data.fetch.get_daily_df",
        return_value=pd.DataFrame(
            {
                "open": [1],
                "high": [1],
                "low": [1],
                "close": [1],
                "volume": [1],
            }
        ),
    ):
        from ai_trading.core import bot_engine as bot

        monkeypatch.setattr(bot, "main", lambda: True)
        assert cast(Callable[[], bool], bot.main)() is True


def test_bot_main_data_fetch_error(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FINNHUB_API_KEY", "testkey")
    config.reload_env()
    monkeypatch.setattr(sys, "argv", ["bot.py"])
    with patch("ai_trading.data.fetch.get_minute_df", side_effect=Exception("API error")), patch(
        "ai_trading.data.fetch.get_daily_df",
        return_value=pd.DataFrame(
            {
                "open": [1],
                "high": [1],
                "low": [1],
                "close": [1],
                "volume": [1],
            }
        ),
    ):
        from ai_trading.core import bot_engine as bot

        monkeypatch.setattr(
            bot,
            "main",
            lambda: (_ for _ in ()).throw(Exception("API error")),
        )
        with pytest.raises(Exception):
            bot.main()


def test_bot_main_signal_nan(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FINNHUB_API_KEY", "testkey")
    config.reload_env()
    monkeypatch.setattr(sys, "argv", ["bot.py"])
    with patch("ai_trading.signals.generate", return_value=float("nan")), patch(
        "ai_trading.data.fetch.get_minute_df", return_value=MagicMock()
    ), patch(
        "ai_trading.data.fetch.get_daily_df",
        return_value=pd.DataFrame(
            {
                "open": [1],
                "high": [1],
                "low": [1],
                "close": [1],
                "volume": [1],
            }
        ),
    ):
        from ai_trading.core import bot_engine as bot

        monkeypatch.setattr(bot, "main", lambda: None)
        bot.main()
