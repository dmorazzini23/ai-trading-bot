from __future__ import annotations

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from ai_trading import main
from ai_trading.alpaca_api import AlpacaAuthenticationError, is_alpaca_service_available


def test_run_cycle_skips_when_market_closed(monkeypatch):
    """run_cycle should exit early when the market is closed."""
    sys.modules.pop("ai_trading.core.bot_engine", None)
    monkeypatch.setattr(main, "_is_market_open_base", lambda: False)
    monkeypatch.setenv("ALLOW_AFTER_HOURS", "0")

    main.run_cycle()

    assert "ai_trading.core.bot_engine" not in sys.modules


def test_run_cycle_calls_market_close_helper_when_closed(monkeypatch):
    """run_cycle should call market-close helper before returning on closed sessions."""

    monkeypatch.setattr(main, "_is_market_open_base", lambda: False)
    monkeypatch.setenv("ALLOW_AFTER_HOURS", "0")
    monkeypatch.setenv("EXECUTION_MODE", "sim")
    calls = {"count": 0}
    monkeypatch.setattr(
        main,
        "_maybe_trigger_market_close_training",
        lambda *_args, **_kwargs: calls.__setitem__("count", calls["count"] + 1),
    )

    main.run_cycle()

    assert calls["count"] == 1


def test_market_close_training_triggers_once_per_day(monkeypatch):
    """Market-close training helper should run only once per New York date."""

    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_LEGACY_DAILY_RETRAIN_ENABLED", "0")
    monkeypatch.setattr(
        main,
        "_LAST_MARKET_CLOSE_TRAINING_DATE",
        None,
        raising=False,
    )
    calls = {"count": 0}
    monkeypatch.setattr(
        main,
        "_invoke_market_close_training",
        lambda: calls.__setitem__("count", calls["count"] + 1),
    )
    now_est = datetime(2026, 1, 6, 16, 5, tzinfo=ZoneInfo("America/New_York"))

    main._maybe_trigger_market_close_training(now_est)
    main._maybe_trigger_market_close_training(now_est)

    assert calls["count"] == 1


def test_run_cycle_aborts_on_alpaca_auth_failure(monkeypatch, caplog):
    sys.modules.pop("ai_trading.core.bot_engine", None)
    monkeypatch.setenv("ALLOW_AFTER_HOURS", "1")
    monkeypatch.setattr(main, "_is_market_open_base", lambda: True)

    import ai_trading.alpaca_api as alpaca_api

    monkeypatch.setattr(alpaca_api, "_ALPACA_SERVICE_AVAILABLE", True)

    def raise_auth(*_a, **_k):
        monkeypatch.setattr(alpaca_api, "_ALPACA_SERVICE_AVAILABLE", False)
        raise AlpacaAuthenticationError("Unauthorized")

    monkeypatch.setattr(alpaca_api, "alpaca_get", raise_auth)

    with caplog.at_level("CRITICAL"):
        main.run_cycle()

    assert "ALPACA_AUTH_PREFLIGHT_FAILED" in caplog.text
    assert "ai_trading.core.bot_engine" not in sys.modules
    assert not is_alpaca_service_available()
