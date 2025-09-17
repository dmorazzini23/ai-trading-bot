from __future__ import annotations

import sys

from ai_trading import main
from ai_trading.alpaca_api import AlpacaAuthenticationError, is_alpaca_service_available


def test_run_cycle_skips_when_market_closed(monkeypatch):
    """run_cycle should exit early when the market is closed."""
    sys.modules.pop("ai_trading.core.bot_engine", None)
    monkeypatch.setattr(main, "_is_market_open_base", lambda: False)
    monkeypatch.setenv("ALLOW_AFTER_HOURS", "0")

    main.run_cycle()

    assert "ai_trading.core.bot_engine" not in sys.modules


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
