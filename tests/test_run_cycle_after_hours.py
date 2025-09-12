from __future__ import annotations

import sys

from ai_trading import main


def test_run_cycle_skips_when_market_closed(monkeypatch):
    """run_cycle should exit early when the market is closed."""
    sys.modules.pop("ai_trading.core.bot_engine", None)
    monkeypatch.setattr(main, "_is_market_open_base", lambda: False)
    monkeypatch.setenv("ALLOW_AFTER_HOURS", "0")

    main.run_cycle()

    assert "ai_trading.core.bot_engine" not in sys.modules
